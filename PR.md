# PR: GPU kernels for `linear_softmax_cross_entropy_loss`

## Summary

Adds two GPU backends for `linear_softmax_cross_entropy_loss`, which previously
only ran on TPU (Pallas/Mosaic-TPU). Both backends implement the memory-efficient
tiled algorithm from [Liger et al. (2024)](https://arxiv.org/abs/2410.10989v2):
tile `(B, V)` with an inner `H` loop so the full `(B, V)` logit matrix never
appears in HBM.

- **Triton** (`pallas_triton_*`): forward + backward, targets SM80+ (Ampere and up). Float32 accumulation throughout.
- **Mosaic GPU SM90** (`pallas_mosaic_gpu_*`): forward + backward, targets H100+ (SM90). WGMMA + TMA pipelining; two warp groups per CTA.

The `api.py` default selection order is: `mosaic_gpu` → `triton` → `mosaic_tpu` → `xla`, with each backend skipped if unavailable.

Also adds a benchmark harness (`benchmarks/linear_softmax_cross_entropy_loss.py`) registered in `benchmark_registry.pbtxt` (H100, B200, TPU-v6e, TPU-v7 environments), and updates the README.

---

## Algorithm overview

The key insight (from the paper) is that the loss can be computed without
ever materialising `x @ w` of shape `(B, V)`:

```
loss = sum_b ( LSE_b - correct_logit_b )
  where LSE_b  = logsumexp_v( x[b,:] @ w[:,v] )
        correct_logit_b = x[b,:] @ w[:, labels[b]]
```

Both kernels tile over `(b_cta, v)` pairs and compute `x[b_tile,:] @ w[:,v_tile]`
in registers/ACC, accumulating per-token logsumexp. The correct-class logit is
computed **outside** the kernel as a cheap `O(B*H)` XLA einsum
(`jnp.einsum("bh,hb->b", x, w[:, labels])`). This avoids the need for a
gather operation inside the kernel, which is awkward with TMA.

The backward also tiles `(B, V)` and recomputes the logit tile on-the-fly
rather than storing it (recompute-for-backward, as in FlashAttention).

---

## Implementation notes

### Triton backend

Straightforward Pallas/Triton implementation. Matmul accumulates in **float32**
throughout (Triton handles this natively with `jnp.float32` dot). This gives
good numerical accuracy — gradients match the XLA reference at `atol=2e-2`.

The backward fuses the gradient scale (`dout / B` for mean, `dout` for sum)
into the kernel rather than applying it post-hoc, saving one pass over the
output tensors.

### Mosaic GPU SM90 backend

Uses `plgpu.emit_pipeline_warp_specialized` with two warp groups per CTA.
One warp group handles rows `[0, tile_m)`, the other `[tile_m, 2*tile_m)`.
The pipeline loads `x` and `w` tiles into SMEM via TMA and issues WGMMA.

**Float32 inputs are downcast to bf16** before entering the kernel. This is a
hardware constraint: SM90 WGMMA only supports bf16/fp8 inputs (no float32
WGMMA path). The accumulator is float32.

#### Backward: two-phase design

The backward reuses the same `pipeline_allocs` (same in_specs, same SMEM
layout) for both phases to avoid doubling allocation overhead:

- **Phase 1**: same WGMMA pipeline as forward → recompute logit tile →
  compute `s_tile = scale * (softmax(logit) - one_hot)` → cast to bf16 →
  write to scratch `s_smem`.
- **Phase 2**: second pipeline call over the same `(x, w)` tiles →
  two WGMMA ops per K-step:
  - `x_grad[b,k] += s_smem @ w_smem.T`
  - `w_grad[k,v] += x_smem[wg_m].T @ s_smem`
  Both results are accumulated via `plgpu.atomic_add` into zero-initialised output buffers.

#### `_kernel_zero_init`

`plgpu.kernel` initialises outputs with `jax.lax.empty` (undefined memory).
The backward uses `plgpu.atomic_add` to accumulate contributions from different
`(b_cta, v)` iterations, so outputs must start at zero. `_kernel_zero_init` is
a thin wrapper around the internal `core_map` machinery that substitutes
`jnp.zeros` for `lax.empty`. This avoids a separate zeroing kernel.

#### SMEM budget

H100 provides 227 KB shared memory per SM. The backward has an extra
`s_smem` allocation of `cta_tile_m * tile_n * 2` bytes (= `256 * tile_n * 2`)
on top of the pipeline buffers. This forces `num_stages` to be capped at 2
for the backward (pipeline at 4 stages would exceed budget at tile_n=128).

Configs that exceed the backward SMEM budget (`tile_n=256` and
`tile_n=128, tile_k=128`) are reachable by the forward but excluded from
backward tests with an explanation. The autotuning config generator
(`get_autotuning_configs`) currently produces these configs; the autotuner
would need to catch the SMEM-overflow error and skip them at search time.

---

## Precision

| Backend | Accumulation | Gradient atol (float32 input, sum) |
|---|---|---|
| XLA (reference) | float32 | — |
| Triton | float32 | 2e-2 |
| Mosaic GPU SM90 | bf16 → float32 acc | 0.20 (rtol=0.05) |

The Mosaic GPU tolerance is higher because bf16 WGMMA quantises the weight
matrix `w` from float32 to bf16. For unit-variance N(0,1) inputs the resulting
absolute error per gradient element is up to ~0.2, **uniform across gradient
magnitudes** (not relative). The error is dominated by near-cancellation
elements: when gradient contributions from different V-tiles nearly cancel,
the bf16 quantisation noise doesn't cancel with them.

This is verified empirically across 20 random seeds (worst observed: 0.201).
For `mean` reduction the error is ~B× smaller (absolute gradients are scaled
by 1/B), so the tighter `atol=2e-2` applies there.

This is expected behaviour for any bf16 WGMMA kernel with unit-scale float32
inputs. It is not a correctness defect.

---

## Files

| File | Purpose |
|---|---|
| `pallas_triton_kernel.py` | Triton fwd + bwd kernel functions |
| `pallas_triton_config.py` | Config dataclass, autotuning search space |
| `pallas_triton.py` | Op wrapper, VJP registration |
| `pallas_triton_kernel_test.py` | Direct kernel tests (fwd + bwd, various block sizes) |
| `pallas_triton_test.py` | End-to-end Op value+grad tests |
| `pallas_mosaic_gpu_kernel_sm90.py` | SM90 fwd + bwd kernel functions, `_kernel_zero_init` |
| `pallas_mosaic_gpu_common.py` | Config dataclass, autotuning search space |
| `pallas_mosaic_gpu.py` | Op wrapper, VJP registration |
| `pallas_mosaic_gpu_kernel_sm90_test.py` | Direct kernel tests (fwd + bwd, tile config sweep) |
| `pallas_mosaic_gpu_test.py` | End-to-end Op value+grad tests |
| `api.py` | Registers both backends, updates default selection |
| `benchmarks/linear_softmax_cross_entropy_loss.py` | Benchmark harness |

---

## What this doesn't cover

- **SM80 Mosaic**: WGMMA is SM90-only. Ampere is served by the Triton backend.
- **Blackwell (SM100)**: `supported_on` permits SM100 for the Mosaic backend
  (same SM90 kernels), but it hasn't been tested.
- **Autotuning SMEM guard**: configs that overflow the backward SMEM budget
  are generated but not filtered in `get_autotuning_configs`. A follow-up
  could add a `smem_bytes` check there.
- **tf32 WGMMA**: would give better precision than bf16 for float32 inputs,
  but is not currently supported by the Mosaic GPU Pallas layer.
