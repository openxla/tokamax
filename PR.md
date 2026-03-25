# PR: GPU kernels for `linear_softmax_cross_entropy_loss`

## Summary

Adds GPU backends for `linear_softmax_cross_entropy_loss`, which previously only ran on TPU. Both use the tiled algorithm from [Liger et al. (2024)](https://arxiv.org/abs/2410.10989v2).
Also adds a benchmark harness registered in `benchmark_registry.pbtxt` (H100, B200, TPU-v6e, TPU-v7) and updates the README.

### Triton (`pallas_triton_*`)
SM80+ (Ampere and up). Selected automatically on GPU when Triton is available. Forward and backward; float32 accumulation throughout.

### Mosaic GPU SM90 (`pallas_mosaic_gpu_*`)
H100+ (SM90). WGMMA + TMA pipelining; two warp groups per CTA. Not selected by default; its backward is 4–8x slower than XLA (chunked scan over V; see Performance). Use explicitly when the logit matrix would OOM.

Liger et al. benchmark claims in the paper were against a PyTorch baseline that materialises the full logit tensor.
Baselining against XLA, I found the naive implementation hard to match for speed, so the use of these kernels should be opt-in to address OOMs.

---

## Algorithm overview

Both kernels tile over `(b_cta, v)` pairs and compute `x[b_tile,:] @ w[:,v_tile]` in registers/ACC, accumulating per-token logsumexp:

```
loss = sum_b ( LSE_b - correct_logit_b )
  where LSE_b  = logsumexp_v( x[b,:] @ w[:,v] )
        correct_logit_b = x[b,:] @ w[:, labels[b]]
```

The correct-class logit is computed outside the kernel as a cheap `O(B*H)` XLA einsum (`jnp.einsum("bh,hb->b", x, w[:, labels])`),
avoiding a gather inside the kernel (awkward with TMA).
The backward recomputes logit tiles on-the-fly rather than storing them (recompute-for-backward, as in FlashAttention).

---

## Implementation notes

### Triton backend

Straightforward Pallas/Triton implementation.
Matmul accumulates in **float32** throughout (Triton handles this natively with `jnp.float32` dot).
This gives good numerical accuracy; gradients match the XLA reference at `atol=2e-2`.

The backward fuses the gradient scale (`dout / B` for mean, `dout` for sum) into the kernel rather than applying it post-hoc, saving one pass over the output tensors.

### Mosaic GPU SM90 backend

Uses `plgpu.emit_pipeline_warp_specialized` with two warp groups per CTA.
One warp group handles rows `[0, tile_m)`, the other `[tile_m, 2*tile_m)`.
The pipeline loads `x` and `w` tiles into SMEM via TMA and issues WGMMA.

**Float32 inputs are downcast to bf16** before entering the kernel.
This is a hardware constraint: SM90 WGMMA only supports bf16/fp8 inputs (no float32 WGMMA path). The accumulator is float32.

#### Forward

H100 provides 227 KB shared memory per SM.
The forward kernel at 4 stages and `tile_n=128`, `tile_k=64` uses ~129 KB.
Configs at `tile_n=256` or `tile_k=128` are reachable by the forward autotuner;
the backward is unaffected (it runs in XLA, not inside the SM90 kernel).
The autotuning config generator (`get_autotuning_configs`) does not currently filter configs by SMEM budget.

#### Backward

The backward does **not** use the SM90 WGMMA kernel. Instead it uses a `jax.lax.scan` over padded vocabulary chunks, issuing one pair of cuBLAS GEMMs per chunk:

```
for each chunk v_start..v_start+chunk_size:
    logit_chunk  = x @ w[:, v_start:v_start+chunk_size]   # recomputed, not stored
    s_chunk      = scale * (softmax(logit_chunk) - one_hot_chunk) * valid_mask
    x_grad      += s_chunk @ w_chunk.T
    w_grad_chunk = x.T @ s_chunk
```

The last chunk is zero-padded so `chunk_size` (4096) divides cleanly for any vocab size (including irregular sizes like V=128256).
Padded positions are masked by `valid = (col_idx < v_dim)` and contribute nothing.

This avoids the `atomic_add` serialisation of a naive in-kernel backward.
Total FLOP count matches XLA; overhead is 32–38 sequential cuBLAS launches vs XLA's 2 full-width matmuls.

---

## Performance

Benchmarked on H100 (bfloat16 inputs, `mean` reduction).
TODO: Triton numbers are not yet included; the benchmark was run before the autotuning configs were replaced with heuristics-based selection and the numbers need to be re-collected.

### Median wall-clock time (ms)

| Shape | `XLA` fwd | `mosaic_gpu` fwd | `XLA` fwd+vjp | `mosaic_gpu` fwd+vjp |
|---|---|---|---|---|
| qwen3-8b (B=4096, H=4096, V=151936) | 7.7 | 7.5 | 21.5 | 60 |
| gemma3-4b (B=4096, H=2560, V=262144) | 9.6 | 8.2 | 26 | 71 |
| gemma3-7b (B=4096, H=3840, V=262144) | 12.6 | 12.7 | 36 | 104 |
| llama3.1-8b (B=4096, H=4096, V=128256) | 6.5 | 6.3 | 18 | 54 |
| deepseek-v3-671b (B=8192, H=7168, V=128256) | 21.9 | 23.7 | 62 | 172 |
| gpt-oss-120b (B=4096, H=2880, V=201088) | 15.4 | 14.9 | 21 | 62 |

### Interpreting these numbers

Forward: `mosaic_gpu` is within ~5% of XLA across all shapes.

Backward: `mosaic_gpu` is 4–8x slower, scaling with `ceil(V / 4096)` (the number of sequential cuBLAS chunk iterations).
Total FLOP count is identical to XLA; the overhead is that XLA issues two full-width matmuls while the chunked scan issues 32–64 sequential ones.

For the shapes above on an H100 (80 GB), XLA fits comfortably.
At larger batch sizes, longer sequences, or on devices with smaller HBM (e.g. A100 40 GB), the logit tensor becomes the binding memory constraint.
During benchmarking, XLA's forward for qwen3-8b hit `RESOURCE_EXHAUSTED` (48 MB allocation failure) at high memory pressure, where `mosaic_gpu` succeeded.

---

## Precision

| Backend | Accumulation | Gradient atol (bf16 input, mean) | Gradient atol (float32 input, sum) |
|---|---|---|---|
| XLA (reference) | float32 | - | - |
| Triton | float32 | 2e-2 | 2e-2 |
| Mosaic GPU SM90 | bf16 -> float32 acc | 2e-2 | 0.40 (rtol=0.05) |

In practice, LLM training uses bfloat16 inputs and `mean` reduction, the common case in the first column, where all backends agree to `atol=2e-2`.

The float32/sum column is the worst case.
The SM90 forward kernel down-casts float32 inputs to bf16 for WGMMA (hardware requirement), introducing quantisation noise of up to ~0.4 per gradient element for unit-variance inputs, uniform across gradient magnitudes.
The backward uses cuBLAS in float32 throughout, so the full tolerance budget comes from the forward's bf16 down-cast.

The initial results led me down a few rabbit holes, but I've confirmed it's the bf16 down-cast that causes the sum accum tol discrepancy.

---

## Files

| File | Purpose |
|---|---|
| `pallas_triton_kernel.py` | Triton forward kernel |
| `pallas_triton_config.py` | Config dataclass, heuristics config |
| `pallas_triton.py` | Op wrapper, VJP (chunked-scan backward) |
| `pallas_triton_kernel_test.py` | Direct forward kernel tests (various block sizes) |
| `pallas_triton_test.py` | End-to-end Op value+grad tests |
| `pallas_mosaic_gpu_kernel_sm90.py` | SM90 forward kernel (WGMMA + TMA) |
| `pallas_mosaic_gpu_common.py` | Config dataclass, heuristics config |
| `pallas_mosaic_gpu.py` | Op wrapper, VJP (chunked-scan backward) |
| `pallas_mosaic_gpu_kernel_sm90_test.py` | Direct forward kernel tests (tile config sweep) |
| `pallas_mosaic_gpu_test.py` | End-to-end Op value+grad tests |
| `api.py` | Registers both backends, updates default selection |
| `benchmarks/linear_softmax_cross_entropy_loss.py` | Benchmark harness |

---

## What this doesn't cover

- Blackwell (SM100): `supported_on` permits SM100 for the Mosaic backend (same SM90 kernels), but it hasn't been tested.
- Autotuning SMEM guard: configs that overflow the SMEM budget are generated but not filtered in `get_autotuning_configs`. A follow-up could add a `smem_bytes` check there. TODO: follow up.
- tf32 WGMMA: would give better precision than bf16 for float32 inputs, but is not currently supported by the Mosaic GPU Pallas layer. TODO: follow up. 

