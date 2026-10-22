# Packed KDA gradient compaction

This increment follows the gradient-output boundary in
[`primatrix/pallas-kernel` c82625a2bdadd5881b12f1ab37ec3fa2004c9780](https://github.com/primatrix/pallas-kernel/blob/c82625a2bdadd5881b12f1ab37ec3fa2004c9780/tops/ops/kda/chunk_bwd_mega.py):
four feature gradients are compacted separately, while beta retains gather.
It reuses the Pallas compactor introduced for forward output.

`Config(packed_gradients=True)` enables compaction of dq/dk/dv/dg after gate
parameter reductions and Q/K normalization backward. Each feature gradient
is cast to its required public dtype before compaction; casting commutes
with token selection and avoids compacting FP32 values that would ultimately
be returned as BF16. The existing beta gather and initial-state gradient
handling remain unchanged. CP and fixed-length calls keep their prior paths.

The compactor retains its existing hardware-sized head-group, width and
VMEM checks; unsupported or oversized inputs use gather. Sequence ownership,
padding masks and guard scratch are shared with forward output compaction.
The option defaults false pending TPU validation.

This change retains aligned gradient buffers and adds separate compaction
calls. It is not direct packed stores from the backward megakernel, nor does
it establish a peak-memory saving. Residuals and gate/normalization reduction
semantics remain unchanged.

Focused tests compare the complete gradient tree exactly with gather in
BF16/FP32 and saved/rematerialized modes, with packed input loading and
forward output compaction enabled. Additional cases omit normalization,
raw-gate activation and initial/final states and cover K=256. Fixed-length
calls verify fallback dispatch. Shared compactor tests cover poisoned tails,
empty batches and memory/shape fallbacks.

```bash
python -m pytest tokamax/_src/ops/kda/pallas_mosaic_tpu_packed_gradients_test.py -v
```

The benchmark's `mosaic_packed_gradients` differs from `mosaic_packed_backward`
only in gradient compaction. Compare forward+VJP with the default device timer
and record hardware/JAX/JAXLIB/libtpu versions and actual compactor dispatch.
TPU lowering, allocation, device correctness and timing remain pending. Keep
the option disabled by default and the PR in draft; no speedup is claimed.
