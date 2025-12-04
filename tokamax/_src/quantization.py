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
"""Quantization utilities."""

from collections.abc import Collection, Mapping
import dataclasses
import functools

import immutabledict
import jax
import qwix


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


def as_array(x: jax.Array | QArray | AsQArray) -> jax.Array:
  """Converts value to a JAX array."""
  y = as_array_or_qarray(x)
  return y if isinstance(y, jax.Array) else qwix.dequantize(y)


def as_array_or_qarray(x: jax.Array | QArray | AsQArray) -> jax.Array | QArray:
  """Converts value to a JAX array or Qwix QArray."""
  return x.as_qarray() if isinstance(x, AsQArray) else x


def as_array_or_qarray_without_zero_point(
    x: jax.Array | QArray | AsQArray,
) -> jax.Array | QArray:
  """Converts value to a JAX array or Qwix QArray without zero point."""
  y = as_array_or_qarray(x)
  if isinstance(y, QArray) and y.zero_point is not None:
    return qwix.dequantize(y)
  return y
