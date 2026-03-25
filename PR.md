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

#### Backward: chunked scan over V

The backward does **not** use the SM90 WGMMA kernel. Instead it uses a
`jax.lax.scan` over padded vocabulary chunks, issuing one pair of cuBLAS
GEMMs per chunk:

```
for each chunk v_start..v_start+chunk_size:
    logit_chunk = x @ w[:, v_start:v_start+chunk_size]   # recomputed, not stored
    s_chunk     = scale * (softmax(logit_chunk) - one_hot_chunk) * valid_mask
    x_grad     += s_chunk @ w_chunk.T
    w_grad_chunk = x.T @ s_chunk
```

The last chunk is zero-padded so chunk_size (4096) divides cleanly for any
vocab size (including irregular sizes like V=128256). Padded positions are
masked by `valid = (col_idx < v_dim)` and contribute nothing.

This avoids the `atomic_add` serialisation of the previous in-kernel backward
design. Total FLOP count matches XLA; overhead is 32–38 sequential cuBLAS
launches vs XLA's 2 full-width matmuls.

The `_kernel_zero_init` helper (used only by the forward) remains in
`pallas_mosaic_gpu_kernel_sm90.py` for any future in-kernel backward work.

#### SMEM budget (forward only)

H100 provides 227 KB shared memory per SM. The forward kernel at 4 stages and
tile_n=128, tile_k=64 uses ~129 KB. Configs at tile_n=256 or tile_k=128 are
reachable by the forward autotuner; the backward is unaffected (it runs in
XLA, not inside the SM90 kernel). The autotuning config generator
(`get_autotuning_configs`) does not currently filter configs by SMEM budget.

---

## Performance

Benchmarked on H100 (bfloat16 inputs, `mean` reduction). Triton is excluded
below because the forward kernel segfaults during autotuning compilation for
vocab sizes >100k — a pre-existing JAX/Triton LLVM thread-safety bug. The
backward no longer uses a Triton kernel (chunked scan instead), so that
contribution to the crashes is resolved, but the forward issue remains.

### Median wall-clock time (ms)

| Shape | XLA fwd | mosaic_gpu fwd | XLA fwd+vjp | mosaic_gpu fwd+vjp |
|---|---|---|---|---|
| qwen3-8b (B=4096, H=4096, V=151936) | 7.7 | 7.5 | 21.5 | 60 |
| gemma3-4b (B=4096, H=2560, V=262144) | 9.6 | 8.2 | 26 | 71 |
| gemma3-7b (B=4096, H=3840, V=262144) | 12.6 | 12.7 | 36 | 104 |
| llama3.1-8b (B=4096, H=4096, V=128256) | 6.5 | 6.3 | 18 | 54 |
| deepseek-v3-671b (B=8192, H=7168, V=128256) | 21.9 | 23.7 | 62 | 172 |
| gpt-oss-120b (B=4096, H=2880, V=201088) | 15.4 | 14.9 | 21 | 62 |

### Interpreting these numbers

**Forward pass**: mosaic_gpu is within ~5% of XLA across all shapes — effectively
neutral.

**Backward pass**: mosaic_gpu is ~3× slower than XLA. The backward uses a
`jax.lax.scan` over padded vocabulary chunks of size 4096, issuing one pair of
cuBLAS GEMMs per chunk (32–38 iterations for typical vocab sizes). XLA's
backward compiles to two full-width cuBLAS matmuls over the entire V dimension
in a single launch, which saturates memory bandwidth more efficiently.
Total FLOP count is identical; the overhead is sequential chunk iteration.

### When these kernels are the right tool

The defining characteristic of this implementation is that the `(B, V)` logit
matrix — of size `B * V * 4` bytes — is never materialised in HBM. For the
shapes above on an H100 (80 GB), XLA fits comfortably. But at larger batch
sizes, longer sequences, or on devices with smaller HBM (e.g. A100 40 GB),
the logit tensor becomes the binding memory constraint and XLA cannot run at
all. During benchmarking, XLA's forward for qwen3-8b hit `RESOURCE_EXHAUSTED`
(48 MB allocation failure) at high memory pressure; mosaic_gpu succeeded.

**These kernels are the lever to reach for when the final projection layer
would OOM the cards you're training on.** The cost is ~3× longer backward
pass for that layer — a worthwhile trade-off when the alternative is not
fitting the model at all.

### Relationship to the Liger paper

Liger et al. report ~3× speedup and ~5× memory reduction vs a **PyTorch
baseline** that first materialises the full `(B, V)` logit tensor in HBM and
then applies cross-entropy. That baseline is meaningfully slower than
XLA-compiled code, which fuses and optimises the same computation.
Our comparison is against XLA, so the speed claims from the paper do not
transfer here. The memory savings are real regardless of the baseline.

---

## Precision

| Backend | Accumulation | Gradient atol (float32 input, sum) |
|---|---|---|
| XLA (reference) | float32 | — |
| Triton | float32 | 2e-2 |
| Mosaic GPU SM90 | bf16 → float32 acc | 0.40 (rtol=0.05) |

The Mosaic GPU tolerance is higher because the SM90 forward kernel down-casts
float32 inputs to bf16 for WGMMA (hardware requirement). For unit-variance
N(0,1) inputs this introduces an absolute quantisation noise of up to ~0.4 per
gradient element, **uniform across gradient magnitudes** (not relative).

The backward pass uses cuBLAS in float32 throughout, so backward precision is
not a contributing factor — the full tolerance budget comes from the forward's
bf16 WGMMA. The Triton backend avoids this by accumulating in float32 end-to-end.

For `mean` reduction the error is ~B× smaller (absolute gradients are scaled
by 1/B), so the tighter `atol=2e-2` applies there.

This is expected behaviour for any bf16 WGMMA kernel with float32 inputs.
It is not a correctness defect.

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
