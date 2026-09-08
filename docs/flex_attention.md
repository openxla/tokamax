# FlexAttention

FlexAttention provides a flexible, composable attention mechanism inspired by
[PyTorch's FlexAttention](https://pytorch.org/blog/flexattention) with the
performance of FlashAttention.

It computes scaled dot-product attention from
["Attention is all you need"](https://arxiv.org/abs/1706.03762):

```
output = softmax(score_mod(Q @ K.T)) @ V
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Score modification** | Apply arbitrary transformations to attention logits before softmax (e.g., ALiBi, relative position bias) |
| **Mask modification** | Efficient boolean masking that can skip computation for masked regions (e.g., causal masks) |
| **Multi-query attention** | Supports grouped-query attention (GQA) where `num_heads_q` is a multiple of `num_heads_kv` |
| **Dropout** | Built-in attention dropout with configurable rate and mask |
| **Residuals** | Optionally return softmax residuals (max values and denominators) for custom backward passes |

## Usage

### Basic Self-Attention

```python
import jax
import jax.numpy as jnp
import tokamax

# Self-attention: Q, K, V with shape [batch, seq_len, num_heads, head_dim]
q = jax.random.normal(jax.random.key(0), (2, 512, 8, 64), dtype=jnp.bfloat16)
k = jax.random.normal(jax.random.key(1), (2, 512, 8, 64), dtype=jnp.bfloat16)
v = jax.random.normal(jax.random.key(2), (2, 512, 8, 64), dtype=jnp.bfloat16)

output = tokamax.flex_attention(q, k, v)
# output.shape == (2, 512, 8, 64)
```

### Causal Masking

```python
def causal_mask(shape):
    """Creates a causal (lower-triangular) mask."""
    # shape is (*B, H, T, t)
    T, t = shape[-2], shape[-1]
    return jnp.tril(jnp.ones((T, t), dtype=jnp.bool_))

output = tokamax.flex_attention(q, k, v, mask_mod=causal_mask)
```

### Score Modification (e.g., Scaling)

```python
import math

def scale_scores(scores):
    """Apply 1/sqrt(d) scaling to attention scores."""
    return scores / math.sqrt(64)  # head_dim = 64

output = tokamax.flex_attention(q, k, v, score_mod=scale_scores)
```

### Grouped-Query Attention (GQA)

```python
# 8 query heads, 2 KV heads -> each KV head serves 4 query heads
q = jax.random.normal(jax.random.key(0), (2, 512, 8, 64), dtype=jnp.bfloat16)
k = jax.random.normal(jax.random.key(1), (2, 512, 2, 64), dtype=jnp.bfloat16)
v = jax.random.normal(jax.random.key(2), (2, 512, 2, 64), dtype=jnp.bfloat16)

# Query heads [0,1,2,3] see KV head 0
# Query heads [4,5,6,7] see KV head 1
output = tokamax.flex_attention(q, k, v)
```

### With Dropout

```python
dropout_mask = jax.random.bernoulli(
    jax.random.key(42), 0.9, shape=(2, 8, 512, 512)
)
output = tokamax.flex_attention(
    q, k, v,
    dropout_mask=dropout_mask,
    dropout_rate=0.1,
)
```

## Implementations

| Implementation | Platform | Notes |
|----------------|----------|-------|
| `xla` | All (CPU, GPU, TPU) | Default. Uses `jnp.einsum` with JAX's standard XLA compilation. |
| `triton` | GPU (SM80+) | Uses Pallas/Triton kernels for fused attention. Requires GPU with compute capability ≥ 8.0. |

## API Reference

```python
tokamax.flex_attention(
    q,                      # Float[Array, "*B T H D"]   — queries
    k,                      # Float[Array, "*B t h D"]   — keys
    v,                      # Float[Array, "*B t h d"]   — values
    *,
    precision=None,         # PrecisionLike | tuple[PrecisionLike, PrecisionLike]
    score_mod=None,         # Callable[[scores], scores] | None
    mask_mod=None,          # Callable[[shape], bool_mask] | None
    dropout_mask=None,      # Bool[Array, "*#B #H #T #t"] | None
    dropout_rate=0.0,       # float
    q_sharding=None,        # jax.sharding.NamedSharding | None
    k_sharding=None,        # jax.sharding.NamedSharding | None
    normalize_output=True,  # bool — divide by softmax denominator
    return_residuals=False, # bool — return (output, residuals)
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | `Float[Array, "*B T H D"]` | required | Query tensor. `H` (num query heads) must be a multiple of `h` (num KV heads). |
| `k` | `Float[Array, "*B t h D"]` | required | Key tensor. |
| `v` | `Float[Array, "*B t h d"]` | required | Value tensor. |
| `precision` | `PrecisionLike` | `None` | Dot product precision. Can be a tuple `(qk_precision, pv_precision)` for separate control. |
| `score_mod` | `Callable` | `None` | Function applied to logits before softmax. Receives **unscaled** scores with accumulator dtype. |
| `mask_mod` | `Callable` | `None` | Function that takes `shape` and returns a boolean mask. `False` positions are masked to negative infinity. Preferred over `score_mod` for masking. |
| `dropout_mask` | `Bool[Array]` | `None` | Boolean mask applied after softmax. |
| `dropout_rate` | `float` | `0.0` | Dropout rate. Weights are scaled by `1 / (1 - dropout_rate)` when dropout is applied. |
| `q_sharding` | `NamedSharding` | `None` | Sharding specification for queries. KV sharding is inferred. ⚠️ Not yet implemented. |
| `k_sharding` | `NamedSharding` | `None` | Sharding specification for keys/values. ⚠️ Not yet implemented. |
| `normalize_output` | `bool` | `True` | If `True`, divide by softmax denominator (standard attention). If `False`, return unnormalized weighted sum. |
| `return_residuals` | `bool` | `False` | If `True`, return `(output, (max_logits, softmax_denom))` for custom backward passes. |

### Returns

- If `return_residuals=False`: `Float[Array, "*B T H d"]` — the attention output.
- If `return_residuals=True`: `tuple[Float[Array, "*B T H d"], tuple[Float[Array, "*B H T"], Float[Array, "*B H T"]]]` — the output and `(max_logits, softmax_denominator)`.

## Notes

- **Softmax stability**: The implementation uses numerically stable softmax with max-subtraction. Fully masked rows use `finfo.min` instead of `-inf` to avoid NaN propagation.
- **Automatic FP32 upcasting**: Softmax reductions are always performed in at least `float32` precision, regardless of input dtype.
- **Head interleaving for GQA**: Query heads `[0, 1]` attend to KV head `0`, query heads `[2, 3]` attend to KV head `1`, etc.
- **Custom JVP**: The softmax implementation includes a custom JVP rule for efficient gradient computation.
