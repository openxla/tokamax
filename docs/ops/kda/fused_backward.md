# Non-CP saved-state KDA backward fusion

This is the backward-only increment after forward fusion. It follows the
saved-state path of the M5 backward in
[`primatrix/pallas-kernel` c82625a2bdadd5881b12f1ab37ec3fa2004c9780](https://github.com/primatrix/pallas-kernel/blob/c82625a2bdadd5881b12f1ab37ec3fa2004c9780/tops/ops/kda/chunk_bwd_mega.py).
The adaptation composes the existing Tokamax compute bodies rather than
importing manual DMA, native packed layouts or CP into the same change.

## Scope

For non-CP calls with saved chunk-start states and 128-aligned K/V,
`Config.fuse_backward=True` combines these stages in one Pallas call:

1. Recompute w/qg/kg/v_new from q/k/v/beta/Akk/g and saved h.
2. Compute dAqk and the initial dv contribution from Aqk and do.
3. Reverse state recurrence, WY and intra-chunk gradients, gate reverse sum.

The bridge tensors remain in VMEM. Their dtypes match the staged path's
HBM buffers, including FP32 dAqk and dv0. The existing matrix multiplication
precision and value-axis reduction order are retained. Gate activation,
gate-parameter gradients, Q/K normalization backward, and output unalignment
stay outside this fusion boundary.

CP and non-aligned K/V keep their existing implementations. The separate
[state-rematerialization increment](rematerialized_backward.md) adds opt-in
fusion for missing chunk states; its default remains the original staged
path. All previously supported modes remain callable.

## State and memory

The grid traverses chunks in reverse order independently for each batch
and head group. A persistent FP32 dh ref is initialized at each sequence
end and written to dh0 at the corresponding sequence start. Per-chunk
bridge refs are overwritten each iteration. No large forward residual
layout changes are introduced; the same KdaResiduals can feed either path.

The launcher reuses the existing backward's hardware-derived mini-batch
budget. Its conservative live-buffer estimate includes the bridge tensors
and compute temporaries even though those bridge tensors no longer have
HBM input buffers. TPU compiler allocation and spill behavior remain a
required validation step, not a conclusion from CPU interpretation.

The forward Op passes an explicit Config through to its automatically
created VJP, making `fuse_backward=False` an effective baseline selection.
An explicitly supplied VJP continues to take precedence.

## Validation

Focused tests compare staged/fused backward with both forward strategies,
BF16/FP32, fixed/packed inputs, initial/final states, and gate-parameter
gradients. The rematerialized-state test checks that it still selects the
staged backward. Gate gradients are also compared with XLA autodiff.

The separate [packed backward correctness fix](packed_gradients.md) extends
comparisons to all tokens and state slots and adds raw-gate regressions
against XLA autodiff, without accepting non-finite values.

```bash
python -m pytest tokamax/_src/ops/kda/pallas_mosaic_tpu_bwd_fused_test.py -v
python -m pytest tokamax/_src/ops/kda -v
python -m tokamax.benchmarks.kda --skip_implementations=xla
```

The benchmark exposes staged, forward-only fused and forward+backward fused
Mosaic variants on the same inputs and default device timer. Compare the
latter two forward+VJP times to isolate this PR's effect. TPU compilation,
correctness, memory use and device-time results must be recorded before
marking the PR ready; no performance gain is inferred from CPU execution.
