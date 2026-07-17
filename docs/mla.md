## Multi-Head Latent Attention (MLA)

Multi-Head Latent Attention is an experimental TPU-optimized attention kernel
implementing the attention mechanism from
[DeepSeek V3](https://arxiv.org/abs/2412.19437). MLA achieves efficient
inference by compressing key-value representations into a low-rank latent space.

### Overview

MLA separates the key representation into two components:

*   **Content-based KV (NOPE)** — latent key-value embeddings without
    positional encoding.
*   **Positional Encoding (ROPE)** — separate positional encoding for keys.

This decomposition enables efficient KV caching: instead of storing full
key-value pairs per head, MLA stores compressed latent representations shared
across heads, significantly reducing memory usage during inference.

### Status

MLA is currently **experimental** and located under
`tokamax._src.ops.experimental.mla`. The API may change without notice.

### Implementations

| Backend | Hardware | Status |
|---|---|---|
| `xla` | GPU + TPU | ✅ Available (reference implementation) |
| `mosaic_tpu` | TPU | ✅ Available (optimized Pallas kernel) |
| GPU (Triton/Mosaic) | GPU | ❌ Not yet implemented |

### Key Parameters (DeepSeek V3 Configuration)

| Parameter | Value | Description |
|---|---|---|
| `lkv_dim` | 512 | Latent KV dimension (NOPE dimension) |
| `r_dim` | 64 | ROPE dimension |
| `num_q_heads` | 128 | Number of query heads |
| `page_size` | Configurable | KV cache page size |

### Features

*   **Paged KV Cache** — efficient memory management using a paged cache
    structure with configurable page sizes.
*   **Sliding Window Attention** — optional sliding window for limiting
    attention span.
*   **Soft Capping** — optional logit capping via `soft_cap` parameter.
*   **FP8 Quantization** — supports `q_scale`, `k_scale`, and `v_scale`
    for quantized inference.
*   **Configurable Score Dtype** — `s_dtype` parameter controls the precision
    of attention logits.

### Source Files

| File | Description |
|---|---|
| `reference.py` | Reference JAX implementation with full documentation |
| `base.py` | Base operation class and `MultiHeadLatentAttention` definition |
| `pallas_mosaic_tpu.py` | TPU-optimized Pallas Mosaic kernel interface |
| `pallas_mosaic_tpu_kernel.py` | Core TPU kernel implementation (~92KB) |
| `utils.py` | Utility functions for MLA operations |

### References

*   DeepSeek V3 Paper: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
*   Internal source: `tokamax/_src/ops/experimental/mla/`
