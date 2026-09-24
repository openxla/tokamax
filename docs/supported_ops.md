## Supported Ops and Hardware

Tokamax provides optimized kernel implementations for a growing set of
fundamental operations used in modern ML models. Each operation supports
multiple backend implementations, allowing Tokamax to automatically select the
best kernel for your hardware, or letting you specify a specific implementation.

### Stable Operations

These operations are part of the public API (`tokamax.*`) and are recommended
for general use.

| Operation | GPU | TPU | Paper |
|---|---|---|---|
| `tokamax.dot_product_attention` | ✅ | ✅ | [FlashAttention](https://arxiv.org/abs/2205.14135) |
| `tokamax.gated_linear_unit` | ✅ | ❌ | [Gated Linear Units](https://arxiv.org/abs/2002.05202) (SwiGLU, etc.) |
| `tokamax.layer_norm` | ✅ | ❌ | [Layer Normalization](https://arxiv.org/abs/1607.06450) and [RMS Normalization](https://arxiv.org/abs/1910.07467) |
| `tokamax.ragged_dot` | ✅ | ✅ | [Mixture of Experts](https://arxiv.org/abs/2211.15841) |
| `tokamax.ragged_dot_general` | ✅ | ✅ | Generalized ragged dot with flexible contracting dimensions |
| `tokamax.linear_softmax_cross_entropy_loss` | ❌ | ✅ | [Memory Efficient Linear Cross Entropy Loss](https://arxiv.org/abs/2410.10989v2) |
| `tokamax.triangle_multiplication` | ❌ | ❌ (XLA only) | [AlphaFold](https://doi.org/10.1038/s41586-021-03819-2) (Supplementary Algorithm 11 & 12) |

### Experimental Operations

These operations are under active development in the `tokamax.experimental`
namespace. APIs may change without notice.

| Operation | GPU | TPU | Description |
|---|---|---|---|
| Splash Attention | ❌ | ✅ | Block-sparse attention with custom masking patterns for TPU |
| Multi-Head Latent Attention (MLA) | ❌ | ✅ | Latent attention mechanism used in [DeepSeek V3](https://arxiv.org/abs/2412.19437) |
| Flex Attention | ✅ | ❌ | Flexible attention with Pallas Triton backend |

### Implementation Backends

Each operation can be executed with different backend implementations. Use
`implementation=None` (the default) to let Tokamax automatically select the best
backend, or specify one explicitly:

| Backend | Hardware | Description |
|---|---|---|
| `"xla"` | GPU + TPU | Default XLA implementation. Always available as a fallback. |
| `"triton"` | GPU | [Pallas Triton](https://docs.jax.dev/en/latest/pallas/gpu/index.html) kernel. |
| `"mosaic"` | GPU | [Pallas Mosaic GPU](https://docs.jax.dev/en/latest/pallas/gpu/index.html) kernel. Uses CuTe DSL on SM90+. |
| `"mosaic_tpu"` | TPU | [Pallas Mosaic TPU](https://docs.jax.dev/en/latest/pallas/tpu/index.html) kernel. |
| `"xla_chunked"` | GPU + TPU | Chunked XLA implementation for memory-efficient attention. |

**Example — specifying an implementation:**

```python
# Let Tokamax choose the best backend
y = tokamax.dot_product_attention(q, k, v, implementation=None)

# Force Pallas Mosaic GPU kernel (will raise if unsupported)
y = tokamax.dot_product_attention(q, k, v, implementation="mosaic")

# Try Mosaic first, fall back to Triton, then XLA
y = tokamax.dot_product_attention(q, k, v, implementation=["mosaic", "triton", "xla"])
```

### Supported Data Types

Most kernels support the following data types:

| Dtype | GPU | TPU | Notes |
|---|---|---|---|
| `bfloat16` | ✅ | ✅ | Recommended for training |
| `float16` | ✅ | ❌ | GPU only |
| `float32` | ✅ | ✅ | Higher precision, slower |
| `float8` (FP8) | ✅ | ✅ | Ragged dot supports FP8 with scaling |

> **Note:** Not all dtype and implementation combinations are supported. If an
> unsupported combination is used with a specific implementation, a
> `NotImplementedError` is raised. Using `implementation=None` will always
> find a working backend.

### GPU Architecture Support

Tokamax includes architecture-specific optimized kernels:

| Architecture | Compute Capability | Supported | Notes |
|---|---|---|---|
| NVIDIA Ampere | SM80 | ✅ | A100 |
| NVIDIA Hopper | SM90 | ✅ | H100 — dedicated kernel files |
| NVIDIA Blackwell | SM100 | ✅ | B200 — dedicated kernel files with autotuning caches |