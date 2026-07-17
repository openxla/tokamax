# Layer Normalization

The `layer_norm` function implements both LayerNorm
([Ba et al., 2016](https://arxiv.org/abs/1607.06450)) and RMSNorm
([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)).

These are essential normalization layers used in virtually all modern
Transformer architectures. RMSNorm (enabled via `subtract_mean=False`) is
the default choice in LLaMA, Gemini, Mistral, and DeepSeek models due to its
lower computational cost compared to full LayerNorm.

## Usage

```python
import jax.numpy as jnp
import tokamax

# Input: (batch, seq_len, hidden_dim)
x = jnp.ones((2, 1024, 4096), dtype=jnp.bfloat16)

# Scale and offset parameters
scale = jnp.ones((4096,), dtype=jnp.float32)
offset = jnp.zeros((4096,), dtype=jnp.float32)

# Standard LayerNorm
output = tokamax.layer_norm(x, scale, offset)

# RMSNorm (no mean subtraction — used in LLaMA, Gemini, etc.)
output = tokamax.layer_norm(x, scale, offset=None, subtract_mean=False)
```

### Numeric Precision

FP16 and BF16 inputs are automatically upcast to FP32 for all internal
computations (mean, variance, normalization). The result is then downcast back
to the input dtype. This ensures numeric stability without requiring manual
dtype management.

## Implementations

| Implementation | Hardware | Notes |
|---|---|---|
| `'xla'` | All platforms | Default fallback; uses `jax.nn.standardize` |
| `'triton'` | NVIDIA GPU | Fused Triton kernel; best GPU performance |

By default, Tokamax automatically selects the best available implementation.

```python
# Force Triton implementation (GPU only)
output = tokamax.layer_norm(x, scale, offset, implementation='triton')

# Try Triton first, fall back to XLA
output = tokamax.layer_norm(
    x, scale, offset,
    implementation=['triton', 'xla'],
)
```

## LayerNorm vs. RMSNorm

| Variant | `subtract_mean` | Formula | Models |
|---|---|---|---|
| **LayerNorm** | `True` (default) | `(x - mean) / sqrt(var + ε) * scale + offset` | GPT-2, BERT |
| **RMSNorm** | `False` | `x / sqrt(mean(x²) + ε) * scale` | LLaMA, Gemini, Mistral |

RMSNorm is ~10-15% faster than LayerNorm because it skips the mean
computation. It has become the de facto standard for large language models.

## API Reference

### `tokamax.layer_norm`

```python
tokamax.layer_norm(
    x: jax.Array,
    scale: jax.Array | None,
    offset: jax.Array | None,
    *,
    axis: int = -1,
    epsilon: float = 1e-06,
    scale_offset: float = 0.0,
    subtract_mean: bool = True,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> jax.Array
```

**Arguments:**

- **`x`**: The array to normalize.
- **`scale`**: Optional 1D array of length `x.shape[axis]`. Multiplicative
  scale factors applied after normalization.
- **`offset`**: Optional 1D array of length `x.shape[axis]`. Additive offset
  applied after scaling.
- **`axis`**: Axis along which to normalize. Default: `-1` (last axis).
- **`epsilon`**: Small constant added to the denominator for numeric stability.
  Default: `1e-6`.
- **`scale_offset`**: Offset added to scale factors before multiplication, i.e.
  the result is `normalized * (scale + scale_offset) + offset`. Default: `0.0`.
- **`subtract_mean`**: If `True`, computes standard LayerNorm (subtract mean
  before computing variance). If `False`, computes RMSNorm (assumes zero mean).
  Default: `True`.
- **`implementation`**: Which backend to use. `None` auto-selects.

**Returns:** Normalized array with the same shape and dtype as `x`.
