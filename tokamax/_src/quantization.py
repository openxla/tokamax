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

from collections.abc import Callable, Collection, Mapping, Sequence
import dataclasses
import functools

import immutabledict
import jax
import jax.numpy as jnp
import qwix
from tokamax._src import utils


QArray = qwix.QArray


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=("value",),
    meta_fields=(
        "qtype",
        "channelwise_axes",
        "tiled_axes",
        "calibration_method",
    ),
)
@dataclasses.dataclass(frozen=True, slots=True)
class AsQArray:
  """A JAX array that will be quantized before the computation.

  This allows for quantization to happen in-kernel.
  """

  value: jax.Array
  qtype: jax.typing.DTypeLike
  _: dataclasses.KW_ONLY
  channelwise_axes: Collection[int] = ()
  tiled_axes: Mapping[int, int | float] | None = None
  calibration_method: str = "absmax"

  def __post_init__(self):
    if self.channelwise_axes:
      object.__setattr__(self, "channelwise_axes", tuple(self.channelwise_axes))
    if self.tiled_axes is not None:
      tiled_axes = immutabledict.immutabledict(self.tiled_axes)
      object.__setattr__(self, "tiled_axes", tiled_axes)

  shape = property(lambda self: self.value.shape)
  dtype = property(lambda self: self.value.dtype)
  ndim = property(lambda self: self.value.ndim)
  size = property(lambda self: self.value.size)

  def as_qarray(self) -> QArray:
    return qwix.quantize(
        self.value,
        self.qtype,
        channelwise_axes=self.channelwise_axes,
        tiled_axes=self.tiled_axes,
        calibration_method=self.calibration_method,
    )


# TODO: Add support for offsets?
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
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


def as_qarray(x: QuantizedArray | QArray | AsQArray) -> QArray:
  """Converts value to a Qwix QArray."""
  if isinstance(x, QArray):
    return x
  if isinstance(x, QuantizedArray):
    return QArray(x.values, x.scales)
  return x.as_qarray()


def as_array(x: jax.Array | QuantizedArray | QArray | AsQArray) -> jax.Array:
  """Converts value to a JAX array."""
  return x if isinstance(x, jax.Array) else qwix.dequantize(as_qarray(x))


def as_array_or_qarray(
    x: jax.Array | QuantizedArray | QArray | AsQArray,
) -> jax.Array | QArray:
  """Converts value to a JAX array or Qwix QArray."""
  if isinstance(x, QuantizedArray):
    return QArray(x.values, x.scales)
  return x.as_qarray() if isinstance(x, AsQArray) else x


def as_array_or_qarray_without_zero_point(
    x: jax.Array | QuantizedArray | QArray | AsQArray,
) -> jax.Array | QArray:
  """Converts value to a JAX array or Qwix QArray without zero point."""
  y = as_array_or_qarray(x)
  if isinstance(y, QArray) and y.zero_point is not None:
    return qwix.dequantize(y)
  return y


def quantize_as(
    dtype: jax.typing.DTypeLike,
    *,
    tile_shape: Sequence[int] | None = None,
    tile_preprocessor: Callable[[jax.Array], jax.Array] | None = None,
    scale_dtype: jax.typing.DTypeLike | None = None,
) -> Callable[[jax.Array], QuantizedArray]:
  """Returns a function that quantizes a JAX array as the given `dtype`."""
  dtype = jnp.dtype(dtype)
  # TODO: Support unsigned integers?
  if not (
      jnp.issubdtype(dtype, jnp.signedinteger)
      or jnp.issubdtype(dtype, jnp.floating)
  ):
    raise ValueError(
        "`dtype` must be a signed integer or a floating point"
        f" type, but got {dtype}"
    )

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
    values = (tile / scale).astype(dtype)
    if scale_dtype is not None:
      scale = scale.astype(scale_dtype)
    return values, scale

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
      else:
        n = utils.exact_div((dim, "`values.shape`"), (tile_dim, "`tile_shape`"))
        values_tiled_shape.extend((tile_dim, n))

      axis = -len(values_tiled_shape)
      fn = jax.vmap(fn, in_axes=axis, out_axes=(axis, axis))

    values_tiled_shape.reverse()

    quant_values, scales = fn(values.reshape(values_tiled_shape))
    scales_shape = [s // t for s, t in zip(values.shape, tile_shape)]
    return QuantizedArray(
        quant_values.reshape(values.shape), scales.reshape(scales_shape)
    )

  return quantize_array
