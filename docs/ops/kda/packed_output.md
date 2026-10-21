# Pallas compaction of packed KDA outputs

This increment adapts the sequential compaction strategy from
[`primatrix/pallas-kernel` c82625a2bdadd5881b12f1ab37ec3fa2004c9780](https://github.com/primatrix/pallas-kernel/blob/c82625a2bdadd5881b12f1ab37ec3fa2004c9780/tops/ops/kda/chunk_fwd_mega.py).
It replaces output unalignment with an optional Pallas call while keeping
Tokamax's existing head-first aligned output and backward residual layouts.

## Ownership and padding

One program owns one hardware-sized head group and one batch for the whole
output. It copies each aligned sequence chunk into token-major VMEM scratch
in sequence order. Arbitrary packed starts are major-axis slices. A later
sequence overwrites the copied tail of the preceding sequence, and the
final write explicitly masks all output padding to zero. Empty sequences
copy nothing; a completely empty batch produces zeros. Distinct programs
write disjoint head/batch regions.

Scratch includes one guard chunk for partial final copies. The output
allocation is rounded to the hardware sublane count, then sliced to the
original token count. Dtype and visible shapes are preserved. Tests poison
unused source rows with NaNs, including a case with no trailing source guard,
to check that neither padding nor adjacent sequences read those values.

## Dispatch and scope

`Config(packed_output=True)` enables this step for non-CP packed calls.
It is independent of packed input loading and forward fusion. The default
is false until TPU compilation and measurements are available.

Compaction requires a complete hardware-sized head group (currently eight
heads), V divisible by 128, and BT=64. A conservative estimate includes
source, token-major scratch and output allocations; oversized groups retain
XLA gather. The estimate uses `get_tpu_limits()` and its reserved compiler
headroom. Widths and head counts outside this scope also retain gather.
The fallback explicitly masks output padding to zero.

The aligned forward output still exists in HBM and is read by this extra
Pallas call. This is output compaction, not direct packed stores from the
megakernel or removal of the intermediate output allocation. Backward's
input alignment and residual layouts are unchanged. Native packed backward
and CP backward fusion remain separate changes.

## Validation and measurement

Tests cover BF16/FP32 compaction, multiple head groups and batches, adjacent
short sequences, middle empty sequences, a fully empty batch, non-aligned
output length, and allocation/shape fallbacks. Forward-only and full VJP
comparisons combine packed input loading with saved-state and rematerialized
backward. The memory-fallback tests force retracing when hardware limits
change and verify which branch actually executes.

```bash
python -m pytest tokamax/_src/ops/kda/pallas_mosaic_tpu_output_test.py -v
python -m tokamax.benchmarks.kda --skip_implementations=xla,mosaic_staged,mosaic_fwd_fused,mosaic_fused,mosaic_remat,mosaic_remat_fused
```

Compare `mosaic_packed_inputs` with `mosaic_packed_outputs`: only output
compaction differs. Use the default device timer and report hardware,
JAX/JAXLIB/libtpu versions, and whether the input fits the compaction dispatch.
No performance improvement is inferred from CPU interpretation. TPU lowering,
VMEM allocation, device correctness and timings remain pending; keep draft
and keep this option disabled by default until those checks are complete.
