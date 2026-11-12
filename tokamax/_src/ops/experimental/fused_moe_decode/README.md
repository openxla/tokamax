# Fused MoE Decode Kernel

One Pallas call per rank that runs a MoE block's routed experts, its shared
expert and the cross-rank all-reduce, for small token counts.

## API

```python
from tokamax._src.ops.experimental.fused_moe_decode.api import (
    fused_moe_decode,
)
from tokamax._src.ops.experimental.fused_moe_decode.fused_moe_decode_kernel import (
    plan_weight_ring,
    vmem_bytes,
)

out = fused_moe_decode(
    rank,                 # int32[1], this rank's mesh index
    hidden,               # [T, H] bf16, the same on every rank
    w1, w2,               # [E, H, 2i] / [E, i, H], this rank's experts
    w1_scale, w2_scale,   # [E, 1, 1, 2i] / [E, 1, 1, H] f32 per channel
    shared_w1, shared_w1_scale,  # [H, 2i_s] / [1, 2i_s]
    shared_w2, shared_w2_scale,  # [i_s, H] / [1, H]
    shared_gate_weight,   # [H, 1], output scaled by sigmoid(hidden @ it);
                          # None for an ungated shared expert
    topk_weights, topk_ids,      # [T, k] from the caller's router
    n_ranks=8,
    expert_start=None,    # global id of the first local expert (EP); None for TP
    scatter_rows=False,   # True for DP attention, see below
    routed_scale=1.0,     # scales the routed experts only
    i_block=None,         # force blocked mode with this width (tests)
)  # -> [T, H] bf16, all-reduced, identical on every rank
   #    scatter_rows=True: [T / n_ranks, H], this rank's rows only
```

- Call it inside `shard_map` over the mesh, one call per rank (or through
  `tokamax._src.ops.experimental.fused_moe_decode.api.fused_moe_decode`).
- **TP experts:** every rank holds all experts with the intermediate dimension
  sliced (`i = I / n_ranks`).
- **EP experts:** each rank holds whole experts
  `[expert_start, expert_start + E)`.
  `topk_ids` stay global; routes owned by other ranks are skipped and summed in
  by the all-reduce.
- `scatter_rows=True` (DP attention): `hidden`, `topk_weights` and `topk_ids`
  are the global batch, all-gathered rank-major by the caller. The all-reduce
  becomes an all-to-all + local sum over token rows (reduce-scatter
  semantics). Rank `r` returns rows `[r·T/n, (r+1)·T/n)` and no all-gather runs.
  Needs `T % n_ranks == 0`.
- `routed_scale` multiplies the routed sum, never the shared expert.
- `vmem_bytes(tokens, h, i, i_s, experts, n_ranks, weight_dtype)` returns the
  VMEM one call declares. `plan_weight_ring` returns the block width (`None` for
  whole experts) and ring depth one call uses.

## When to use it

Use it when all of these hold:

- Decode or other small batches: up to a few hundred tokens per step. Above
  that, prefill-style kernels win.
- Attention is TP (every rank sees every token, output all-reduced), or DP with
  `scatter_rows=True` (see below).
- The MoE is TP-sharded, or EP with `ep_size == mesh size`.
- Weights are FP8 with per-channel scales, the activation is SiLU, and there
  are no expert biases.
- The shared expert has a sigmoid gate or is ungated.
- `i`, `i_s` and `H / n_ranks` are multiples of 128.
- Experts of any size. Those whose 3 whole-expert slots do not fit VMEM stream
  in blocks; see Design.

Do not use it for prefill-sized steps.

### DP attention

With DP attention each rank owns `T/n` tokens. The caller all-gathers every
rank's tokens and routing, rank-major, and calls the kernel with
`scatter_rows=True`. The kernel computes the experts over the global batch and
ends with an all-to-all + local sum over token rows, so each rank gets back
exactly its own `T/n` rows. The kernel runs no all-gather.

## Design

- **No dispatch.** Each active expert runs on all `T` tokens in their original
  order. The route weight zeroes the rows the expert does not own. At decode
  sizes this is cheaper than sorting, permuting and combining. Nothing
  intermediate goes to HBM.
- **Active-expert compaction.** XLA builds a dense `[T, E]` route-weight table
  and an active mask. The kernel compacts the active expert ids into SMEM and
  visits only those.
- **Weight ring.** Each active expert's `w1/w2` slices stream from HBM through
  a ring of VMEM slots. Ring depth is between 3 and 6, picked by VMEM size. The
  ring is primed while the shared expert computes. Expert `j+1`'s first matmul
  is issued during expert `j`'s elementwise chain, so the MXU stays busy while
  the VPU quantizes.
- **Blocked mode.** When 3 whole-expert slots do not fit VMEM, each active
  expert streams as `i / ib` units: a gate block with its matching up block of
  `w13`, and the matching rows of `w2`. `ib` is 256 (else 128) and the ring is
  up to 6 units deep. Each block's down-projection partial adds into the same
  accumulator. The choice depends on expert size only, so small TP expert slices
  keep whole experts and the unchanged program. The kernel name ends in
  `-ib_<ib>`.
- **Numerics.** Hidden states and the activated intermediate are quantized to
  FP8 with one scale per row per 512 columns, the block `gmm_v2` uses with the
  kernel off (in blocked mode the intermediate per row per block). Each block's
  partial product is scaled and summed in FP32, into one VMEM accumulator.
  The hidden size must be a multiple of 512.
- **Shared expert in-kernel.** Its slice is loaded once at entry and computed
  first, gated by `sigmoid(x · w_g)` or ungated.
- **In-kernel all-reduce.** A direct all-to-all with a local sum over column
  chunks, then an all-gather, with bf16 on the wire: each rank sends every
  peer the chunk that peer keeps in one remote copy, and sums the n chunks it
  receives. That gives reduce-scatter semantics without a ring or per-hop
  reduction, one hop of n concurrent copies instead of n - 1 dependent steps,
  which suits latency-bound decode sizes. With `scatter_rows` it is the
  all-to-all + local sum over token rows only. The cross-rank barrier is
  signalled at entry and waited only before the all-to-all, so the expert
  loop hides it.
- **Limits.** All tokens, the accumulator and the all-to-all buffers stay
  resident in VMEM. That caps the token count, and `vmem_bytes` computes the
  cap.
