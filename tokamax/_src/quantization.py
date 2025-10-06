# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""`QuantizedArray` class."""

from collections.abc import Callable, Sequence
import dataclasses

import jax
import jax.numpy as jnp


# TODO: Add support for offsets?
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QuantizedArray:
  """A quantized JAX array with a scale factor for each tile."""

  values: jax.Array
  scales: jax.Array

  def recompose(self) -> jax.Array:
    """Returns the original array values."""
    assert isinstance(self.values, jax.Array)
    assert isinstance(self.scales, jax.Array)
    scales = self.scales
    for i, tile_dim in enumerate(self.tile_shape):
      if tile_dim != 1:
        scales = jnp.repeat(scales, repeats=tile_dim, axis=i)
    return self.values.astype(self.dtype) * scales

  @property
  def shape(self) -> tuple[int, ...]:
    return self.values.shape

  @property
  def dtype(self) -> jnp.dtype:
    return self.scales.dtype

  @property
  def ndim(self) -> int:
    return len(self.shape)

  @property
  def size(self) -> int:
    return self.values.size

  @property
  def tile_shape(self) -> tuple[int, ...]:
    return tuple(d1 // d2 for d1, d2 in zip(self.shape, self.scales.shape))


def quantize_as(
    dtype: jnp.dtype,
    *,
    tile_shape: Sequence[int] | None = None,
    tile_preprocessor: Callable[[jax.Array], jax.Array] | None = None,
) -> Callable[[jax.Array], QuantizedArray]:
  """Returns a function that quantizes a JAX array as the given `dtype`."""
  # TODO: Support unsigned integers?
  if not (jnp.issubdtype(dtype, jnp.signedinteger)
          or jnp.issubdtype(dtype, jnp.floating)):
    raise ValueError("`dtype` must be a signed integer or a floating point"
                     f" type, but got {dtype}")

  info_fn = jnp.iinfo if jnp.issubdtype(dtype, jnp.integer) else jnp.finfo
  info = info_fn(dtype)

  def quantize_tile(tile):
    if tile_preprocessor is not None:
      tile = tile_preprocessor(tile)

    # Choose the smallest possible scale factor that allows that quantized
    # values to cover the full range.
    # finfo min/max can be in the queried dtype, so cast them to the tile dtype.
    min_val = jnp.array(info.min, dtype=tile.dtype)
    max_val = jnp.array(info.max, dtype=tile.dtype)
    scale = jnp.max(jnp.maximum(tile / max_val, tile / min_val), keepdims=True)
    return (tile / scale).astype(dtype), scale

  def quantize_array(values, tile_shape=tile_shape):
    if tile_shape is None:
      tile_shape = values.shape

    if len(tile_shape) != len(values.shape):
      raise ValueError("`tile_shape` must have same rank as `values` shape.")

    # Replace `-1` `tile_shape` dims with the full `values` dim.
    tile_shape = [t if t != -1 else s for t, s in zip(tile_shape, values.shape)]

    # Use nested `vmap` calls to apply `quantize_tile` to the correct tiles.
    # If a `tile_shape` dim is not equal to `1` or the full dim size, we split
    # the input dimension, then restore the original shape below.
    fn = jax.jit(quantize_tile)
    values_tiled_shape = []
    for dim, tile_dim in zip(reversed(values.shape), reversed(tile_shape)):
      if tile_dim == dim:
        values_tiled_shape.append(dim)
        continue  # No `vmap` needed.

      if tile_dim == 1:
        values_tiled_shape.append(dim)
      elif dim % tile_dim == 0:
        values_tiled_shape.extend((tile_dim, dim // tile_dim))
      else:
        raise ValueError("Input shape must divide exactly by `tile_shape`.")

      axis = -len(values_tiled_shape)
      fn = jax.vmap(fn, in_axes=axis, out_axes=(axis, axis))

    values_tiled_shape.reverse()

    quant_values, scales = fn(values.reshape(values_tiled_shape))
    scales_shape = [s // t for s, t in zip(values.shape, tile_shape)]
    return QuantizedArray(
        quant_values.reshape(values.shape), scales.reshape(scales_shape)
    )

  return quantize_array
