# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dynamic activation quantization shared across routing_moe kernels."""

import functools

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import spec


def dtype_max(dtype: jax.typing.DTypeLike) -> float:
  """Returns the largest finite value of a floating-point or integer dtype."""
  if jnp.issubdtype(dtype, jnp.floating):
    return float(jnp.finfo(dtype).max)
  return float(jnp.iinfo(dtype).max)


def compute_scale(
    x: jax.Array, dtype: jax.typing.DTypeLike, quant: spec.QuantSpec
) -> tuple[jax.Array, jax.Array]:
  """Computes the scales of one quantization block and their reciprocals.

  Scale `i` maps the maximum magnitude of the elements sharing scale `i` in `x`
  to the largest finite value of `dtype`. All-zero scale blocks get scale 0 and
  reciprocal 0, so they quantize and dequantize to 0.

  Args:
    x: Unquantized `[..., cols]` values of one block. For
      `LaneStridedQuantSpec`, `cols` must equal `quant.block_col_size`. For
      `SubchannelQuantSpec`, `cols` is at most `quant.block_size`, which allows
      a shorter last block.
    dtype: Target quantized dtype.
    quant: Quantization spec specifying subchannel or lane-strided grouping.

  Returns:
    A `(scale, inv_scale)` pair of `[..., 1]` or `[..., 128]` arrays.
  """
  match quant:
    case spec.SubchannelQuantSpec(block_size=block_size):
      assert x.shape[-1] <= block_size
      absmax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    case spec.LaneStridedQuantSpec(block_col_size=block_col_size):
      lanes = 128
      assert x.shape[-1] == block_col_size
      chunks = jnp.split(x, block_col_size // lanes, axis=-1)
      absmax = functools.reduce(jnp.maximum, [jnp.abs(c) for c in chunks])
  scale = absmax / dtype_max(dtype)
  # 1 / 0 is inf, which would turn 0 into NaN when quantizing.
  inv_scale = jnp.where(scale == 0, 0, 1 / scale)
  return scale, inv_scale


def quantize(
    x: jax.Array, inv_scale: jax.Array, dtype: jax.typing.DTypeLike
) -> jax.Array:
  """Scales `x` by `inv_scale` and casts it to the quantized `dtype`."""
  return (x * inv_scale).astype(dtype)
