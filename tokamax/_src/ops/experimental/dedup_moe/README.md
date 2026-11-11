# De-Dup MoE

## High level summary

-   A topology-aware, expert-parallel (EP) Mixture of Experts layer for TPU
    with fused Pallas kernels, for fp8 W8A8 and bf16 W16A16 expert weights.
-   **Wave dispatch.** Instead of an activation all-gather, each device sends
    every (token, destination device) pair once over a split-phase ragged
    all-to-all, in *expert waves*: every device visits its local experts
    fewest rows first and cuts that order into waves of growing modeled
    compute (1 : 2 : 4 : ...). The fused expert kernel starts each expert as
    soon as its wave has landed, so later waves arrive under the first
    experts' matmuls.
-   **De-Dup.** On meshes whose devices are grouped into boards connected by
    a limited number of inter-board links, a token that several devices of a
    remote board need is sent across once, to a gateway device on that board,
    which forwards it to the other devices of its board inside the fused
    kernel. Forwarded rows count in their target's own first wave, so a
    receiver waits for exactly the rows the direct send would have given it.
-   Every configuration computes the same arithmetic: the direct and De-Dup
    dispatches, and every wave count, give bit-identical outputs.

## Usage

```python
from tokamax._src.ops.experimental import dedup_moe

# Optional: devices of the mesh axis grouped into boards (contiguous blocks).
topology = dedup_moe.BoardTopology.uniform(num_devices=ep, board_size=8)
cfg = dedup_moe.DedupMoeConfig(
    ep=ep, hidden=D, num_experts=E, top_k=K, ffn=F, topology=topology
)
prepared = dedup_moe.prepare_weights(mesh, w1, w2, w1_scale, w2_scale, cfg=cfg)
y = dedup_moe.dedup_moe(x, router_logits, prepared, cfg)  # inside jax.jit
```

-   `dedup_moe(...)` (`layer/modular.py`) is the model-facing entry point. It
    rounds the token count up to a static bucket and picks one configuration
    per bucket (`DedupMoeConfig.policy`): `rules` derives the waves and
    whether to De-Dup from the topology and the shape; `auto` (default) uses
    a measured mode table where one is available and the rules elsewhere;
    a mode name (`direct_w<W>`, `dedup_w<W>`) pins one configuration.
-   `dedup_ep_moe(...)` (`layer/monolithic.py`) is the layer with every knob
    explicit. Without `dedup_topology` it runs the direct wave dispatch.
-   `BoardTopology(device_groups, gateways=None)` describes the boards as
    positions on the mesh axis (contiguous, equal-size groups) and,
    optionally, the gateway of every (source device, remote board) pair. The
    default gateway is the device with the source's local index on the
    remote board.

## Status

-   Experimental. The kernels need TPU v7x or later and are tested on TPU
    v8i. De-Dup is tested with board sizes 4 and 8; other sizes are refused.
-   The tile height (`capacity`) must be a multiple of 128 on TPU.
-   Only the TensorCore dispatch paths are included; see the roadmap.

## Relationship to `fused_moe`

This op extends the fused expert-parallel MoE kernel in
`experimental/fused_moe` and shares its code wherever the two are the same.

| Shared from `fused_moe` | Owned by `dedup_moe` (why) |
| :--- | :--- |
| `host`: layout constants, weight-format records, routing-table builders (`build_routing_tables_sharded`, `shard_*`), VMEM chip helpers and weight-slot types | `dispatch/`: wave plan, scatter-free send lists, split-phase start and drain kernels, receiver token-gather assembly (new) |
| `kernel`: expert FFN compute (`expert_ffn_fp8`, `expert_ffn_bf16`), wire quantization, row quantization, kernel and combine constants, small DMA helpers | `dedup_plan/`: board topology and the on-device De-Dup plan (new) |
| `layer`: router scoring and the packed routing exchange | `fused_kernel/`: the fused expert-parallel kernel (expert GMMs, per-wave waits, gateway forwarding, result push): body, scratch layout and config (substantive changes to the kernel) |
| `router_ops.pallas_select`, `token_parallel.TokenReplicaLayout` | `io_relayout/output_reduce.py`: combine kernel variant that also covers small (decode) token counts |
| `gmm_v2` clamped activations | `layer/`: config, plan and mode registry (new) |

No `fused_moe` code is changed.

## Roadmap

-   SparseCore gateway forwarding: forward the staged rows on the SparseCore
    instead of the TensorCore. Blocked on compiler support for zero-output
    SparseCore kernels and on the JAX writes-to-array-input check; re-enable
    once available and re-measure against TensorCore forwarding.
-   SparseCore sender path (SparseCore gather and contiguous senders):
    upstream in a follow-up once it no longer needs private JAX patches.
-   Hierarchical reduce-scatter for the combine phase: first try a SparseCore
    collective-offload hierarchical reduce-scatter; if that does not pan out,
    implement a TensorCore hierarchical reduce-scatter.
-   De-Dup on the return path (De-Dup collect): reduce partial expert outputs
    at the gateway before sending them back across boards.
-   Offload the routing exchange (all-gather) to the SparseCore collectives
    engine with multicast, once the compiler support lands.
-   Re-validate and retune the auto policy table (for example, whether
    SparseCore forwarding helps), and add decode-specific (very small
    `t_local`) and larger-slice configurations.
-   Potential: expert placement (EPLB) integration, and more weight formats
    (W4A8, sub-channel, fp4).

## Folder structure

```text
dedup_moe/
├── common/
│   └── arg.py: Supported weight formats (records shared with fused_moe).
├── dedup_plan/
│   ├── topology.py: BoardTopology and its static gateway tables.
│   └── dedup_plan.py: Scatter-free on-device De-Dup plan (sender, gateway and receiver views).
├── dispatch/
│   ├── source_plan.py: Sender-local wave plan and token-gather lists.
│   ├── tc_sender.py: Split-phase wave-major ragged all-to-all (start and drain kernels).
│   ├── tg_assemble.py: Receiver token-gather table assembly from the sender lists.
│   └── wave_dispatch.py: Ramp wave plan and dispatch orchestration.
├── fused_kernel/
│   ├── config.py: Compile-time static configuration of the fused kernel.
│   ├── kernel.py: Fused kernel entry point (operand validation, pallas_call).
│   ├── memory_ref.py: Buffer declarations, DMA helpers and gateway forwarding.
│   ├── pipeline.py: Kernel body: per-wave waits, expert tiles and result push.
│   └── plan.py: VMEM estimate and weight-slot schedule.
├── io_relayout/
│   ├── input_pack.py: Row quantization, token replicas and the shared expert.
│   └── output_reduce.py: Weighted top-k combine kernel.
├── layer/
│   ├── config.py: Layer configuration and prepared weights.
│   ├── mode_registry.py: Named configurations (direct_w<W>, dedup_w<W>).
│   ├── modular.py: Model-facing entry point (dedup_moe).
│   ├── monolithic.py: Shard-mapped layer with every knob explicit (dedup_ep_moe).
│   └── plan.py: Static t buckets and policy resolution.
└── test/
    ├── common/: Interpret-mode harness, inputs, f32 reference.
    └── {dedup_plan, dispatch, layer}/: Unit, interpret-mode and TPU tests.
```

## Testing

The CPU tests run the Pallas kernels in TPU interpret mode with the
split-phase exchange emulated; the TPU tests run the same test files on TPU
v8i devices. All test targets are tagged `manual`.
