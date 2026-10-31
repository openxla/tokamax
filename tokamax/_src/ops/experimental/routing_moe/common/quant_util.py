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

import dataclasses
import functools

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class DynamicQuantConfig:
  """Dynamic quantization parameters for an activation tensor.

  Each scale covers `block_size` elements of a row spaced `stride` columns
  apart (`stride=1` for contiguous blocks, `stride=num_lanes` for per-lane
  scales).

  Attributes:
    dtype: Target quantized element dtype.
    block_size: Number of elements along the row sharing a single scale.
    stride: Column stride between elements sharing a scale.
  """

  dtype: jnp.dtype
  block_size: int
  stride: int = 1


def dtype_max(dtype: jax.typing.DTypeLike) -> float:
  """Returns the largest finite value of a floating-point or integer dtype."""
  if jnp.issubdtype(dtype, jnp.floating):
    return float(jnp.finfo(dtype).max)
  return float(jnp.iinfo(dtype).max)


def compute_scale(
    x: jax.Array, quant: DynamicQuantConfig
) -> tuple[jax.Array, jax.Array]:
  """Computes the scales of one quantization block and their reciprocals.

  Scale `l` maps the maximum magnitude of columns `l, l + stride, ...` of `x` to
  the largest finite value of `quant.dtype`. All-zero groups get scale 0 and
  reciprocal 0, so they quantize and dequantize to 0.

  Args:
    x: Unquantized `[..., cols]` values of one block. With `stride > 1`, `cols`
      is `block_size * stride`. With `stride == 1`, `cols` is at most
      `block_size`, which allows a shorter last block.
    quant: Dynamic quantization configuration.

  Returns:
    A `(scale, inv_scale)` pair of `[..., stride]` arrays.
  """
  if quant.stride == 1:
    assert x.shape[-1] <= quant.block_size
    absmax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
  else:
    assert x.shape[-1] == quant.block_size * quant.stride
    chunks = jnp.split(x, quant.block_size, axis=-1)
    absmax = functools.reduce(jnp.maximum, [jnp.abs(c) for c in chunks])
  scale = absmax / dtype_max(quant.dtype)
  # 1 / 0 is inf, which would turn 0 into NaN when quantizing.
  inv_scale = jnp.where(scale == 0, 0, 1 / scale)
  return scale, inv_scale


def quantize(
    x: jax.Array, inv_scale: jax.Array, dtype: jax.typing.DTypeLike
) -> jax.Array:
  """Scales `x` by `inv_scale` and casts it to the quantized `dtype`."""
  return (x * inv_scale).astype(dtype)
