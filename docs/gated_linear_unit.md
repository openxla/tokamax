# Gated Linear Unit (GLU)

The `gated_linear_unit` function computes a gated linear unit
([Dauphin et al., 2017](https://arxiv.org/abs/1612.08083)):

```
output = activation(x @ w_gate) * (x @ w_up)
```

This is a key building block in modern Transformer feed-forward networks.
Depending on the choice of activation function, this implements:

| Activation | Name | Reference |
|---|---|---|
| `jax.nn.swish` | **SwiGLU** | [Shazeer, 2020](https://arxiv.org/abs/2002.05202) |
| `jax.nn.gelu` | **GEGLU** | [Shazeer, 2020](https://arxiv.org/abs/2002.05202) |
| `jax.nn.relu` | **REGLU** | [Shazeer, 2020](https://arxiv.org/abs/2002.05202) |
| `jax.nn.sigmoid` | **GLU** | [Dauphin et al., 2017](https://arxiv.org/abs/1612.08083) |
| `None` | Bilinear (no gate activation) | — |

SwiGLU is the most widely used variant, powering models like LLaMA, PaLM,
Gemini, and Mistral.

## Usage

```python
import jax
import jax.numpy as jnp
import tokamax

# Input: (batch, seq_len, hidden_dim)
x = jnp.ones((2, 1024, 4096), dtype=jnp.bfloat16)

# Fused weights: (hidden_dim, 2, intermediate_dim)
weights = jnp.ones((4096, 2, 11008), dtype=jnp.bfloat16)

# SwiGLU (default in LLaMA-style models)
output = tokamax.gated_linear_unit(x, weights, activation=jax.nn.swish)

# GEGLU
output = tokamax.gated_linear_unit(x, weights, activation=jax.nn.gelu)
```

### Unfused Weights

Weights can also be provided as a tuple of two separate arrays:

```python
w_gate = jnp.ones((4096, 11008), dtype=jnp.bfloat16)
w_up = jnp.ones((4096, 11008), dtype=jnp.bfloat16)

output = tokamax.gated_linear_unit(x, (w_gate, w_up), activation=jax.nn.swish)
```

## Implementations

| Implementation | Hardware | Notes |
|---|---|---|
| `'xla'` | All platforms | Default fallback; unfused matmuls via XLA |
| `'mosaic'` | NVIDIA GPU (SM90+) | Fused Mosaic GPU kernel; best performance on H100/B200 |
| `'triton'` | NVIDIA GPU | Fused Triton kernel |

By default, Tokamax automatically selects the best available implementation.
To override:

```python
# Force a specific implementation
output = tokamax.gated_linear_unit(
    x, weights,
    activation=jax.nn.swish,
    implementation='mosaic',
)

# Try multiple implementations in order
output = tokamax.gated_linear_unit(
    x, weights,
    activation=jax.nn.swish,
    implementation=['mosaic', 'triton', 'xla'],
)
```

## API Reference

### `tokamax.gated_linear_unit`

```python
tokamax.gated_linear_unit(
    x: Float[Array, '*B M K'],
    weights: FusedWeights | UnfusedWeights,
    *,
    activation: Callable[[Array], Array] | None = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> Float[Array, '*B M N']
```

**Arguments:**

- **`x`**: Input array of shape `(*B, M, K)`.
- **`weights`**: Either a fused array of shape `(K, 2, N)` or a tuple of two
  unfused arrays, each of shape `(K, N)`.
- **`activation`**: Activation function applied to the gate branch. Common
  choices: `jax.nn.swish`, `jax.nn.gelu`, `jax.nn.relu`, `jax.nn.sigmoid`.
  If `None`, no activation is applied (bilinear).
- **`precision`**: Matrix multiplication precision. `None` uses the backend
  default, or pass `jax.lax.Precision` / `jax.lax.DotAlgorithmPreset`.
- **`implementation`**: Which backend to use. `None` auto-selects.

**Returns:** Output array of shape `(*B, M, N)`.
