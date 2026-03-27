# PR: GPU kernels for `linear_softmax_cross_entropy_loss`

## Summary

Adds GPU backends for `linear_softmax_cross_entropy_loss`, which previously only ran on TPU. The motivation is memory, not speed.

XLA's implementation materialises the full `(B, V)` logit matrix. At LLM scale this is large:

| Shape | Logit matrix (float32) |
|---|---|
| qwen3-8b (B=4096, V=151936) | 2.5 GB |
| gemma3-4b (B=4096, V=262144) | 4.3 GB |
| deepseek-v3-671b (B=8192, V=128256) | 4.2 GB |

During training, this allocation sits alongside activations, weights, and optimiser state. Both kernels here use the tiled algorithm from [Liger et al. (2024)](https://arxiv.org/abs/2410.10989v2), which tiles over `(b_tile, v_tile)` pairs and keeps logits only in registers; peak logit memory drops from O(B*V) to O(b_block*v_block), a few KB regardless of vocab size.

The trade-off is speed: XLA's single cuBLAS GEMM is compute-bound and hard to match with a tiled kernel. These kernels are slower (see Performance) and should be used when the logit matrix is the binding memory constraint, not as a general replacement for XLA.

Also adds a benchmark harness registered in `benchmark_registry.pbtxt` (H100, B200, TPU-v6e, TPU-v7) and updates the README.

### Triton (`pallas_triton_*`)
SM80+ (Ampere and up). Selected automatically on GPU when Triton is available. Forward and backward; float32 accumulation throughout. ~2x XLA forward wall-clock time on LLM-scale shapes.

### Mosaic GPU SM90 (`pallas_mosaic_gpu_*`)
H100+ (SM90). WGMMA + TMA pipelining; two warp groups per CTA. Not selected by default. Forward within ~5% of XLA; backward 4-8x slower (chunked cuBLAS scan over V). Use explicitly when the logit matrix would OOM and the backward cost is acceptable.

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

## Memory

XLA allocates the full `(B, V)` logit tensor in HBM (float32 for numerical stability), then reads it again for the logsumexp and CE loss reduction. Both kernels here eliminate this:

Forward: each `(b_block, v_block)` logit tile lives in registers for the duration of one kernel invocation. No HBM allocation for logits at any point. The outputs written to HBM are `(B, num_v_blocks)`, a per-token, per-v-chunk logsumexp and correct-logit contribution, O(B) not O(B*V).

Backward: logit tiles are recomputed from `x` and `w` on the fly, one chunk at a time, and discarded. The peak extra allocation during the backward is one logit chunk `(B, chunk_size)`, a few MB, not `(B, V)`.

The residual saved from forward to backward is the per-token log-sum-exp `lse`, shape `(B,)`, negligible.

For reference, the `(B, V)` logit tensor that these kernels avoid:

| Shape | float32 logit tensor | bfloat16 equivalent |
|---|---|---|
| qwen3-8b (B=4096, V=151936) | 2.5 GB | 1.2 GB |
| gemma3-4b (B=4096, V=262144) | 4.3 GB | 2.1 GB |
| gemma3-7b (B=4096, V=262144) | 4.3 GB | 2.1 GB |
| llama3.1-8b (B=4096, V=128256) | 2.1 GB | 1.0 GB |
| deepseek-v3-671b (B=8192, V=128256) | 4.2 GB | 2.1 GB |
| gpt-oss-120b (B=4096, V=201088) | 3.3 GB | 1.6 GB |

XLA computes in float32 regardless of input dtype (bfloat16 inputs are upcast before the GEMM), so the relevant number is the float32 column. During benchmarking, XLA's forward for qwen3-8b hit `RESOURCE_EXHAUSTED` (48 MB allocation failure) at high memory pressure, where the tiled kernels succeeded.

---

## Implementation notes

### Triton backend

Straightforward Pallas/Triton implementation.
Matmul accumulates in **float32** throughout (Triton handles this natively with `jnp.float32` dot).
This gives good numerical accuracy; gradients match the XLA reference at `atol=2e-2`.

The backward fuses the gradient scale (`dout / B` for mean, `dout` for sum) into the kernel rather than applying it post-hoc, saving one pass over the output tensors.

#### Tiling heuristic

HBM traffic for the forward pass scales as:
- `x` traffic: `B * H * V / v_block` (x is re-read once per v-chunk tile)
- `w` traffic: `B * H * V / b_block` (w is re-read once per b-chunk tile)

Traffic is balanced when `b_block = v_block`. At `v_block=128` (the maximum safe value), the heuristic targets `b_block=128` when `B` is divisible by 128, which equalises x/w HBM reads and measurably improves performance (~4% on LLM-scale shapes).

Register budget on SM80+ (65536 regs/SM, `num_warps=4`, 128 threads/CTA):

| b | h | regs/thread | CTAs/SM |
|---|---|---|---|
| 128 | 64 | 256 (50%) | 2 |
| 64 | 128 | 256 (50%) | 2 |
| 64 | 64 | 160 (31%) | 2 |
| 32 | 128 | 192 (37%) | 2 |
| 128 | 128 | 384 (75%) | 1 (avoided) |

With `b=128`, `h` is capped at 64 to stay within the 50% budget (2 CTAs/SM).
With `b <= 64`, `h=128` is used when `H` is divisible by 128 for better tensor-core tile efficiency; `h_block` does not affect HBM traffic.

#### v_block_size cap at 128

`v_block_size=256` crashes the Triton-to-PTX compilation stage in JAX 0.9.2's bundled Triton with a C++ exception (segfault in `f.compile()`).
JAX's `pallas/triton/lowering.py` itself documents this: the power-of-2 tensor-size check (line 288-301) applies only to load/store ops and explicitly notes that for other ops "the Triton lowering will fail anyway but it will crash with a C++ exception".
With a (32, 256) accumulator tile, the load/store check passes (8192 = 2^13) but the Triton backend then crashes during instruction selection for `tl.dot`.

No tracked upstream issue was found for this specific case (float32 `tl.dot` with N=256 on SM80 in JAX's bundled Triton).
The closest related fix is [jax-ml/jax#35654](https://github.com/jax-ml/jax/pull/35654), which added an early guard for the same crash pattern in the fp64 MMA path; the fp32/n=256 case is not yet guarded.
The heuristic caps `v_block_size` at 128 and should be revisited when JAX upgrades its bundled Triton.

### Mosaic GPU SM90 backend

Uses `plgpu.emit_pipeline_warp_specialized` with two warp groups per CTA.
One warp group handles rows `[0, tile_m)`, the other `[tile_m, 2*tile_m)`.
The pipeline loads `x` and `w` tiles into SMEM via TMA and issues WGMMA.

Float32 inputs are downcast to bf16 before entering the kernel: SM90 WGMMA only supports bf16/fp8 inputs. The accumulator remains float32.

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
Total FLOP count matches XLA; overhead is 32-38 sequential cuBLAS launches vs XLA's 2 full-width matmuls.

---

## Performance

Benchmarked on H100 (bfloat16 inputs, `mean` reduction).
Triton forward numbers below are from RTX 3090 (same heuristic, same pattern expected on H100); H100 Triton numbers TBD.

### Median wall-clock time (ms)

H100 numbers (XLA and mosaic_gpu); RTX 3090 numbers (Triton, where available):

| Shape | `XLA` fwd | `mosaic_gpu` fwd | `triton` fwd | `XLA` fwd+vjp | `mosaic_gpu` fwd+vjp |
|---|---|---|---|---|---|
| qwen3-8b (B=4096, H=4096, V=151936) | 7.7 | 7.5 | TBD | 21.5 | 60 |
| gemma3-4b (B=4096, H=2560, V=262144) | 9.6 | 8.2 | TBD | 26 | 71 |
| gemma3-7b (B=4096, H=3840, V=262144) | 12.6 | 12.7 | TBD | 36 | 104 |
| llama3.1-8b (B=4096, H=4096, V=128256) | 6.5 | 6.3 | TBD | 18 | 54 |
| deepseek-v3-671b (B=8192, H=7168, V=128256) | 21.9 | 23.7 | TBD | 62 | 172 |
| gpt-oss-120b (B=4096, H=2880, V=201088) | 15.4 | 14.9 | TBD | 21 | 62 |

RTX 3090 Triton forward results (H100 benchmarks pending):

| Shape | `XLA` fwd (3090) | `triton` fwd (3090) | Ratio |
|---|---|---|---|
| qwen3-8b (B=4096, H=4096, V=151936) | 69.7 | 139.2 | 2.00x |
| llama3.1-8b (B=4096, H=4096, V=128256) | 58.9 | 116.9 | 1.98x |
| gpt-oss-120b (B=4096, H=2880, V=201088) | 66.7 | 130.3 | 1.95x |

### Interpreting these numbers

Forward: `mosaic_gpu` is within ~5% of XLA across all shapes.

`triton` forward runs at ~2x XLA wall-clock time. This is expected and close to the theoretical minimum for this tiling approach: Triton re-reads `x` once per v-chunk and `w` once per b-chunk, accumulating `B*H*V/128` elements from each, while XLA's cuBLAS reads `x` and `w` once in a single compute-bound GEMM. The heuristic balances x/w HBM traffic (`b_block = v_block = 128` when B is divisible by 128). Closing the gap further would require `v_block > 128`, which is blocked by the JAX 0.9.2 Triton compiler limitation described above.

Backward: `mosaic_gpu` is 4-8x slower, scaling with `ceil(V / 4096)` (the number of sequential cuBLAS chunk iterations).
Total FLOP count is identical to XLA; the overhead is that XLA issues two full-width matmuls while the chunked scan issues 32-64 sequential ones.

For the shapes above on an H100 (80 GB), XLA fits comfortably. On devices with smaller HBM (A100 40 GB, RTX 3090 24 GB) or at higher batch sizes the logit tensor becomes the binding constraint; see Memory.

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

