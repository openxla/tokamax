# Packed input windows for fused non-CP KDA backward

This increment adapts packed input loading from
[`primatrix/pallas-kernel` c82625a2bdadd5881b12f1ab37ec3fa2004c9780](https://github.com/primatrix/pallas-kernel/blob/c82625a2bdadd5881b12f1ab37ec3fa2004c9780/tops/ops/kda/chunk_bwd_mega.py)
to Tokamax's existing fused reverse traversal. It changes selected feature
input boundaries while preserving aligned gates, matrix/state residuals,
normalization statistics and gradient outputs.

## Dispatch and data flow

`Config(packed_backward=True)` enables packed input windows when local
backward fusion is active, for non-CP packed calls with 128-aligned K/V.
Saved-state backward is supported directly. The rematerialized-state mode
also requires `fuse_rematerialization=True`; its reconstruction pass stays
aligned. Fixed-length, CP, disabled backward fusion and staged state
rematerialization retain their existing paths. The new option defaults false.

When normalization is enabled, backward reads the aligned unit Q/K already
retained in the residuals instead of replaying normalization and shifting
original Q/K windows. V still uses its original packed window. Without
normalization, original Q/K/V windows are supported. Scalar beta uses aligned
token-minor storage in either case.

Per-chunk start/end metadata is derived from aligned labels and both boundary
arrays. Invalid chunks receive an empty window. Selected feature inputs read
72 rows at an eight-row-aligned origin, select 64 tokens, and mask sequence
tails in VMEM. Guard rows protect allocation-end loads. Tests visit every
window offset in both normalized and unnormalized modes, multiple head groups
and an empty batch.

Head-group estimates include the extra window and staging buffers. The
implementation uses Pallas BlockSpec pipelining and element indices on all
input dimensions, consistent with TPU lowering requirements. It does not
import manual DMA scheduling from the source.

## Memory and numerical contract

The aligned residual tree is unchanged. Gate prefix recomputation, state
rematerialization, Q/K normalization backward and gradient unalignment still
use aligned data. Thus this is native packed input loading for the reverse
pass, not a complete native packed backward or removal of all aligned
training buffers. Original-input guards and window shifts have costs; only device measurement
can establish the net benefit. Packed forward can normalize raw Q/K in its
VMEM staging tiles and populate the existing unit-Q/K and RSTD residual slots.
The public residual schema is unchanged.

Output gradients, including padding and empty-state semantics, continue
through the existing postprocessing. FP32 v_new from rematerialized states
is still read from its aligned buffer, preserving the preceding PR's
precision contract.

## Validation and measurement

Tests compare full gradient trees in BF16/FP32 with packed forward enabled,
for saved and rematerialized states. Further cases cover every window offset,
non-eight-aligned source length, multiple head groups and batches, empty
batches, no initial/final state, preactivated gates and K/V=256. Dispatch tests
retain fixed-length, disabled-fusion and staged-rematerialization fallbacks.

```bash
python -m pytest tokamax/_src/ops/kda/pallas_mosaic_tpu_packed_bwd_test.py -v
python -m tokamax.benchmarks.kda --skip_implementations=xla,mosaic_staged,mosaic_fwd_fused,mosaic_fused,mosaic_packed_inputs,mosaic_remat,mosaic_remat_fused
```

Compare `mosaic_packed_outputs` and `mosaic_packed_backward`: forward input
loading, output compaction and saved-state policy are identical. Measure
forward+VJP with the default device timer and record hardware/JAX/JAXLIB/libtpu
versions. TPU lowering, allocation, device correctness and performance remain
pending. No speedup or memory reduction is claimed from CPU interpretation;
keep draft and keep the new option disabled until device validation completes.

Direct packed gradient stores and CP backward fusion remain separate work.

[Current full-call and VJP measurements](https://github.com/antgroup/tokamax/blob/921c3554c129577b22c5ed00482cf8dc558ca4ec/docs/ops/kda/backward_performance_followup_20261115.md)
include same-head default and forward-only controls; earlier unfavorable rows
remain archived.
