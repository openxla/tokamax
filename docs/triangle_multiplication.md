# Triangle Multiplication

Triangle Multiplication implements the **triangle multiplicative update** from
[AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2) (Supplementary
Algorithms 11 & 12, Jumper et al. 2021). This operation is a key component of
the Evoformer block used in protein structure prediction.

## Overview

Triangle multiplication updates pairwise representations by aggregating
information along shared edges in a triangle graph. There are two variants:

| Variant | Equation | Aggregation |
|---------|----------|-------------|
| **Outgoing** | `einsum("cik,cjk->ijc")` | Aggregates over shared column `k` with left projection on rows `i`, right on rows `j` |
| **Incoming** | `einsum("ckj,cki->ijc")` | Aggregates over shared row `k` with left projection on columns `j`, right on columns `i` |

### Algorithm

```
1. LayerNorm(x)                              # Input normalization
2. GLU(x, gate_weights, projection_weights)  # Gated projection → sigmoid gate × linear
3. Split into left_proj, right_proj          # Two projection streams
4. Apply mask                                # Boolean pair mask
5. Einsum (outgoing or incoming)             # Triangle aggregation
6. LayerNorm(result)                         # Output normalization
7. Linear projection                         # Project to output dim
8. Sigmoid gate × result                     # Output gating
```

## Usage

```python
import jax
import jax.numpy as jnp
import tokamax

# Pair representation: [N_residues, N_residues, C_channels]
N, C, H, D = 64, 128, 32, 128

x = jax.random.normal(jax.random.key(0), (N, N, C), dtype=jnp.bfloat16)
mask = jnp.ones((N, N), dtype=jnp.bool_)

# Weight shapes follow the AlphaFold2 convention
projection_in = jax.random.normal(jax.random.key(1), (C, 2, H))
gate_in = jax.random.normal(jax.random.key(2), (C, 2, H))
projection_out = jax.random.normal(jax.random.key(3), (H, D))
gate_out = jax.random.normal(jax.random.key(4), (C, D))
ln_in_scale = jnp.ones(C)
ln_in_offset = jnp.zeros(C)
ln_out_scale = jnp.ones(H)
ln_out_offset = jnp.zeros(H)

output = tokamax.triangle_multiplication(
    x, mask,
    projection_in, gate_in,
    projection_out, gate_out,
    ln_in_scale, ln_in_offset,
    ln_out_scale, ln_out_offset,
    triangle_type="outgoing",
)
# output.shape == (64, 64, 128)
```

## API Reference

```python
tokamax.triangle_multiplication(
    x,                        # Float[Array, "N N C"]    — pair representation
    mask,                     # Bool[Array, "N N"]       — pair mask
    projection_in_weights,    # Float[Array, "C 2 H"]    — input projection
    gate_in_weights,          # Float[Array, "C 2 H"]    — input gate
    projection_out_weights,   # Float[Array, "H D"]      — output projection
    gate_out_weights,         # Float[Array, "C D"]      — output gate
    layernorm_in_scale,       # Float[Array, "C"]        — input LN scale
    layernorm_in_offset,      # Float[Array, "C"]        — input LN offset
    layernorm_out_scale,      # Float[Array, "H"]        — output LN scale
    layernorm_out_offset,     # Float[Array, "H"]        — output LN offset
    triangle_type,            # Literal["incoming", "outgoing"]
    *,
    precision=None,           # jax.lax.PrecisionLike
    epsilon=1e-6,             # float — LayerNorm epsilon
    return_residuals=False,   # bool
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `Float[Array, "N N C"]` | Input pair representation. `N` is the number of residues, `C` is the channel dimension. |
| `mask` | `Bool[Array, "N N"]` | Boolean pair mask. Applied element-wise to projected activations before the triangle einsum. |
| `projection_in_weights` | `Float[Array, "C 2 H"]` | Input projection weights. The `2` dimension produces left and right projections. |
| `gate_in_weights` | `Float[Array, "C 2 H"]` | Input gating weights (sigmoid activation). Used with `GatedLinearUnit`. |
| `projection_out_weights` | `Float[Array, "H D"]` | Output projection from hidden dim `H` to output dim `D`. |
| `gate_out_weights` | `Float[Array, "C D"]` | Output gating weights (sigmoid). Applied after the triangle aggregation. |
| `layernorm_{in,out}_{scale,offset}` | `Float[Array, "..."]` | LayerNorm parameters for input and output normalizations. |
| `triangle_type` | `"incoming"` or `"outgoing"` | Determines aggregation direction. See table above. |
| `precision` | `PrecisionLike` | Matrix multiplication precision. |
| `epsilon` | `float` | Small constant for LayerNorm numerical stability. Default: `1e-6`. |

### Returns

`Float[Array, "N N D"]` — the updated pair representation.

## Context: AlphaFold2 Evoformer

In the AlphaFold2 architecture, triangle multiplication is part of the
**Evoformer** block, which processes MSA (Multiple Sequence Alignment)
representations and pair representations. The two variants work together:

- **Outgoing edges**: Information flows from residue `i` to residue `j` via
  shared residue `k` (edges `i→k` and `j→k`).
- **Incoming edges**: Information flows via shared residue `k` through edges
  `k→i` and `k→j`.

This is complemented by **triangle attention** (not implemented in Tokamax),
which applies attention along the rows/columns of the pair representation.

## References

- Jumper, J. et al. "Highly accurate protein structure prediction with
  AlphaFold." *Nature* 596, 583–589 (2021).
  [DOI: 10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
