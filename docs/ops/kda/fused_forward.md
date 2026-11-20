# Non-CP KDA forward fusion

The default aligned Mosaic path expects grouped, consecutively numbered segments
with trailing zero padding. Set `Config(canonicalize_packed_inputs=True)` to
prepare interior padding, sparse labels or interleaved labels for local aligned
execution and restore outputs and state slots to caller order. The adapter can
add metadata work. Forward fusion and packed I/O are independently selectable.
The native compact path has its own segment handling and eligibility checks.

For local aligned calls that need grouping, construct the configured Mosaic op:

```python
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic

op = mosaic.PallasMosaicTpuKimiDeltaAttention(
    config=mosaic.Config(canonicalize_packed_inputs=True)
)
output, state = op(query, key, value, gate, beta, segment_ids=ids,
                   max_num_segments=capacity)
```



The existing Mosaic forward writes `w`, `u`, `kg`, and `g_cumsum` between
the intra-chunk solve and the state/output kernel. The fused path executes
both stages in one Pallas call, passing these values through VMEM scratch
refs. It uses the existing compute bodies, including the dtype conversion
at the former HBM boundary, to isolate fusion from changes to the algorithm
or its precision policy.

This follows the persistent forward execution in
[`primatrix/pallas-kernel` at c82625a2bdadd5881b12f1ab37ec3fa2004c9780](https://github.com/primatrix/pallas-kernel/blob/c82625a2bdadd5881b12f1ab37ec3fa2004c9780/tops/ops/kda/chunk_fwd_mega.py).
The Tokamax adaptation deliberately reuses the already-landed intra and
state/output arithmetic. It does not import the source repository's packed
DMA, output workspace API, hardware tables, backward fusion, or CP code.

## Execution and support

The grid is `(heads // mini_batch, batch, chunks)` with an ordered final
axis. Each head group keeps its FP32 recurrent state in VMEM across chunks.
At each segment boundary it loads the corresponding initial state or zeros.
It writes the final state when requested; empty segments retain their initial
state, or zero when no initial state was provided.

```text
existing aligned inputs and segment metadata
    -> gate activation / prefix sum / intra-chunk solve
    -> VMEM: w, u, kg, gate prefix, Aqk, Akk
    -> state recurrence and output
    -> existing output unalignment and public layout
```

`Config.fuse_forward` selects the fused path for non-CP calls with K/V
multiples of 128. CP and other widths continue to use the existing staged
implementation. `Config(fuse_forward=False)` retains the staged path for
comparisons. The public API and implementation selector are unchanged.
Chunk size remains 64. Both BF16 and FP32 use their existing solve and
matrix-multiplication precision policies.

Mini-batch sizing includes the persistent state, bridge buffers, input/output
buffers, and solve temporaries, using Tokamax's hardware-derived VMEM budget.
Compiler allocation and performance must still be checked on TPU; CPU
interpretation does not validate Mosaic lowering or memory capacity.

## Backward contract

There is no change to `KdaResiduals`, preprocessing, or backward code.

| Forward mode | HBM residual outputs from the fused call |
| --- | --- |
| Forward only | None |
| Saved-state VJP | Aqk, Akk, gate prefix, chunk-start h |
| Rematerialized VJP, activated gate | Aqk, Akk |
| Rematerialized VJP, pre-activated gate | Aqk, Akk, gate prefix |

Prepared q/k/v/beta and the existing optional gate and normalization inputs
remain on the residual tape. Aqk/Akk retain their logical last dimension of
64. Fusion removes the stage-to-stage materialization of `w/u/kg` and avoids
materializing unused `qg`; it does not eliminate the residuals needed by the
existing backward. The saved `h` has the same dtype as the staged path.

## Validation and measurement

The focused tests compare fused and staged forward-only execution, VJP
execution, gradients, and the residual tree. Cases include BF16/FP32,
fixed/packed inputs, initial/final state, empty state slots, normalization,
gate activation, and both rematerialization settings. The existing KDA
kernel suite continues to compare Mosaic with XLA, including CP fallbacks.

Packed backward masks unwritten padding lanes before gate-parameter
reductions and restores the final-state cotangent for empty initial-state
slots. Focused tests compare the complete gradient tree, explicitly require
zero padding gradients and the empty-slot identity derivative, and retain
the original tolerances without xfails for these cases.

Non-TPU interpretation uses highest matrix-multiply precision to avoid a
backend-selected reduced-precision FP32 reference. Interpretation also keeps
optimization barriers at the rounded inter-phase tensors: the staged path
materializes these tensors between calls, whereas the fused interpreter
otherwise compiles both phases together. Native Mosaic keeps the VMEM
boundaries and its existing precision configuration. CUDA validation of the
interpreter comparison is tracked separately from native TPU correctness.

```bash
python -m pytest tokamax/_src/ops/kda/pallas_mosaic_tpu_fwd_fused_test.py -v
python -m pytest tokamax/_src/ops/kda -v
python -m tokamax.benchmarks.kda --skip_implementations=xla
```

The benchmark compares staged and fused Mosaic on the same primary fixed
and packed N=25 BF16 workloads (B=1, T=8192, H=32, K=V=128). It measures
forward and forward+VJP using `tokamax.benchmark`'s default device timer,
including preprocessing and output unalignment. Random gate parameters
are retained. Report hardware, JAX/JAXLIB/libtpu versions and measurements
together; performance is not inferred from the eliminated HBM traffic.

### Initial validation (2026-09-07)

Windows CPU, Python 3.12.14, JAX/JAXLIB 0.11.1:

- Focused interpret cases: 9 passed, 4 xfailed for the staged packed
  raw-gate parameter-gradient limitation described above.
- Additional fixed raw-gate gradient cases: 2 passed.
- KDA API/adapter tests: 36 passed, 9 subtests passed.
- CI shard consistency and its 53 unit tests passed.

TPU lowering, device correctness (including the CPU-xfailed cases), VMEM
allocation, and staged/fused device-time measurements remain pending.
No speedup is claimed from the CPU tests. Keep the PR in draft until these
checks have completed; adjust the fused dispatch heuristic if measured
performance or compiler resource limits require a narrower support set.

## Training preparation and head grouping (2026-11-16)

The training adapter now shares its packing order and one branch-local
pullback between forward and backward. This avoids rebuilding two copies of
the canonical/interleaved decision without removing the general fallback or
changing the public state and segment-label contract. The earlier description
of unchanged preparation refers to the original fusion increment.

Saved-state training already requires aligned q/k/v for the existing backward.
It therefore feeds those retained arrays to fused forward instead of preparing
both packed windows and aligned residuals. Forward-only calls retain packed
window reads. When that path normalizes packed q/k, the normalized values and
reciprocal norms are computed once and reused during alignment.

Unalignment masks padding at the token gather, including scalar beta gradients
and nontrivial output permutations. Unwritten aligned padding cannot leak NaNs
into public outputs. Early masking required by convolution and the independent
gate-parameter gradient reductions remains in place. Tests cover canonical,
interleaved, interior-hole and empty rows, H=2/32, and normalization on/off,
using the existing numerical bounds and explicit zero-padding assertions.

The aligned fused-forward schedule permits head groups up to 32, subject to
its existing hardware VMEM budget and divisibility checks. Packed windows
retain their extra staging allowance and cap of 16. This changes neither
matrix precision nor the state recurrence; native correctness, allocation and
same-input forward/full-VJP timings must validate the schedule. Performance
results are published separately from the implementation change.

### CP gated-input lowering compatibility

The CP summary needs gated Q/K in HBM before its collective. With the existing
packing conditional, libtpu 0.0.50.1 can combine their full-array broadcasts
and BF16 conversions into an invalid rank-flattened multi-output window.
Original PR 649506c6 and the performance candidate both abort on the existing
CP2 single-segment BF16 backward test; unmodified upstream d68ceb9e passes.

A chunk-local Pallas map computes the same q*exp2(g) and
k*exp2(g_last-g), with the original activation-dtype stores. WY matrix
products and their precision remain unchanged; this map is used only when
the CP summary must reconstruct missing gated inputs. The non-CP pipeline,
packing fallback and padding/gate-gradient protections are retained. The
existing failing CP test is the regression test; its tolerances are unchanged.

### Guarded padding restoration

The adapter marks only buffers allocated by `_align_seqs` with alignment greater
than one as having a zeroable guard row. Unalignment can then clear one guard
row and redirect invalid indices to it, instead of selecting zeros across an
entire floating-point output. XLA may update that dead input buffer in place
when its liveness permits. This also preserves zero outputs when unwritten
padding contains NaNs; gate-parameter reduction and convolution protections
remain unchanged.

For ordinary capacities the allocation bound is
`align_up(T + N * (chunk_size - 1), chunk_size)`. The aligned end is outside all
valid rows. If it equals the allocation bound, equality requires every sequence
to exhaust the maximum padding allowance, so the final allocated row is padding.
Large-capacity allocation reserves an additional guard chunk. Arbitrary callers
of `_unalign_output` do not carry this guarantee and retain explicit masking.
The guarantee is static residual metadata, not a device-side conditional.

Packed-output compaction forwards the same guard guarantee to its fallback,
including the actual KDA chunk size. Unsupported head/width/VMEM configurations
retain the generic gather algorithm with guarded restoration when proven;
the dedicated packed-output kernel and its existing eligibility rules remain.
