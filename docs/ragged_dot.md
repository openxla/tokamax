# Ragged Dot (GMM) — Group Matrix Multiply for Mixture of Experts

## Overview

`tokamax.ragged_dot` implements the **Group Matrix Multiply (GMM)** operation
from the [Megablocks paper](https://arxiv.org/abs/2211.15841), which is the core
compute primitive for **Mixture of Experts (MoE)** models.

In an MoE model, tokens are routed to different "expert" sub-networks. The
ragged dot operation efficiently performs batched matrix multiplications where
each group (expert) processes a variable number of tokens.

```
out[i, j] = sum_k lhs[i, k] * rhs[g, k, j]
```

where `g` is the expert/group index determined by cumulative `group_sizes`.

## API

### `tokamax.ragged_dot`

```python
tokamax.ragged_dot(
    lhs,                        # (M, K) shaped array
    rhs,                        # (G, K, N) shaped array
    group_sizes,                # (G,) shaped integer array
    precision=None,             # jax.lax.PrecisionLike
    preferred_element_type=None,
    group_offset=None,          # Optional (1,) array for partial computation
    activation=None,            # Optional fused activation function
    implementation=None,        # 'xla', 'triton', 'mosaic', or None (auto)
)
```

### `tokamax.ragged_dot_general`

Generalized version with explicit dimension numbers, supporting flexible
contracting dimensions.

```python
tokamax.ragged_dot_general(
    lhs, rhs,
    group_sizes,
    ragged_dot_dimension_numbers,  # jax.lax.RaggedDotDimensionNumbers
    ...
)
```

## Implementations

| Implementation | Hardware | Backend | Notes |
|---|---|---|---|
| `xla` | CPU, GPU, TPU | XLA | Baseline, works everywhere |
| `triton` | GPU (SM80+) | Pallas Triton | Optimized for NVIDIA GPUs |
| `mosaic` | GPU (SM90+) | Pallas Mosaic GPU | Uses Mosaic GPU runtime |
| `mosaic_tpu` | TPU (v4+) | Pallas Mosaic TPU | V1 kernel |
| `mosaic_tpu_v2` | TPU (v6+) | Pallas Mosaic TPU | V2 kernel (see below) |

## GMM V2 Kernel (TPU)

The V2 kernel (`pallas_mosaic_tpu_v2.py`) is a rewrite of the V1 TPU kernel
with several improvements:

### Key Differences from V1

| Feature | V1 | V2 |
|---|---|---|
| TPU Generation | v4+ | v6+ only |
| DMA Strategy | Static | **Dynamic** — reduces redundant compute |
| Tile Selection | Manual | **Heuristic-based** VMEM-aware tiling |
| Fusion Options | Limited | Extensive (bias, scale, gate-up) |
| QArray Support | Yes | No (uses raw arrays with `rhs_scale`/`rhs_bias`) |
| Activation Fusion | Yes | Not yet supported |
| `manual_axis_type` | Yes | Not yet supported |

### V2-Specific Parameters

The V2 kernel accepts additional parameters not available in V1:

- **`rhs_scale`**: Per-group scale factors for FP8 quantized weights
- **`rhs_bias`**: Per-group bias tensors fused into the kernel
- **`maybe_quantize_lhs`**: Auto-quantize LHS to FP8 at kernel level
- **`zero_initialize`**: Control output buffer initialization
- **`fuse_gateup_activation`**: Fuse SwiGLU-style gate-up activation
- **`lhs_quantization_dtype`**: Specify LHS quantization dtype
- **`rhs_quantization_dtype`**: Specify RHS quantization dtype

### V2 Config

```python
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2

config = pallas_mosaic_tpu_v2.Config(
    tile_m=256,    # Tile size along M dimension (or None for auto)
    tile_k=256,    # Tile size along K dimension (or None for auto)
    tile_n=256,    # Tile size along N dimension (or None for auto)
)
```

When tile sizes are set to `None` (default), the kernel uses a VMEM-aware
heuristic (`calculate_tiling`) to automatically select optimal tile sizes.

### V2 Dimension Numbers

The V2 kernel supports three operation modes via `ragged_dot_dimension_numbers`:

| Mode | Dimension Numbers | Use Case |
|---|---|---|
| **GMM Forward** | `DEFAULT_RAGGED_DOT_DIM_NUMS` | Standard forward pass |
| **dLHS** | `TRANS_RHS_RAGGED_DOT_DIM_NUMS` | Gradient w.r.t. LHS |
| **dRHS (TGMM)** | `RAGGED_CONTRACTING_DOT_DIM_NUMS` | Gradient w.r.t. RHS (Transposed GMM) |

## Supported Data Types

| Input Type | Quantization | Hardware |
|---|---|---|
| `bfloat16` | None | GPU, TPU |
| `float32` | None | GPU, TPU |
| `float16` | None | GPU |
| `int8` | Per-tensor or per-tile | GPU, TPU |
| `int4` | Per-tensor or per-tile | GPU |
| `float8_e4m3fn` | Via `rhs_scale` (V2) | TPU v6+ |
| `float8_e5m2` | Via `rhs_scale` (V2) | TPU v6+ |

## Benchmark Specifications

Tokamax includes production-scale benchmark specs for ragged dot, derived from
real model configurations:

| Model | Experts | M | K | N |
|---|---|---|---|---|
| DeepSeek V3 | 256 | 131072–262144 | 256–7168 | 512–7168 |
| GPT-OSS (MaxText) | 128–256 | 65536–524288 | 768–7168 | 768–7168 |
| Mixtral 8x7B | 8 | 8192 | 14336 | 4096 |

## Usage Example

```python
import jax
import jax.numpy as jnp
import tokamax

# MoE configuration: 8 experts, 1024 tokens, hidden dim 512, output dim 256
num_experts = 8
num_tokens = 1024
hidden_dim = 512
output_dim = 256

# Create inputs
lhs = jax.random.normal(jax.random.key(0), (num_tokens, hidden_dim), jnp.bfloat16)
rhs = jax.random.normal(jax.random.key(1), (num_experts, hidden_dim, output_dim), jnp.bfloat16)

# Equal distribution of tokens across experts
group_sizes = jnp.array([num_tokens // num_experts] * num_experts, jnp.int32)

# Compute MoE output
output = tokamax.ragged_dot(lhs, rhs, group_sizes)
# output shape: (1024, 256)
```

## Source Files

| File | Description |
|---|---|
| `ragged_dot/api.py` | Public API |
| `ragged_dot/base.py` | Base class and reference implementation |
| `ragged_dot/pallas_triton.py` | GPU Triton kernel |
| `ragged_dot/pallas_mosaic_gpu.py` | GPU Mosaic kernel (SM90/SM100) |
| `ragged_dot/pallas_mosaic_tpu.py` | TPU V1 kernel |
| `ragged_dot/pallas_mosaic_tpu_v2.py` | TPU V2 kernel (GMM v2) |
| `ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py` | V2 GMM forward kernel |
| `ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py` | V2 TGMM (backward) kernel |
| `ragged_dot/arg_specs.py` | Benchmark specifications |
