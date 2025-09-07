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
"""Pallas-Triton normalization VJP configuration."""

import dataclasses
from typing import TypeAlias

import immutabledict
import jax
import pydantic
from tokamax._src import pydantic as pydantic_lib
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import pallas_triton_config


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  block_m: pydantic_lib.PowerOfTwo
  block_n: pydantic_lib.PowerOfTwo | None
  num_warps: pydantic_lib.PowerOfTwo


Key: TypeAlias = pallas_triton_config.Key


def get_heuristics_config(
    residuals: base.Residuals,
    out: jax.Array,
    dout: jax.Array,
    x: jax.Array,
    scale: jax.Array | None,
    offset: jax.Array | None,
    **kwargs,
) -> Config:
  """Returns a config based on heuristics."""
  del residuals, out, dout  # Unused.
  # Re-use the forwards heuristics.
  config = pallas_triton_config.get_heuristics_config(
      x, scale, offset, block_size_per_warp=2048, **kwargs
  )
  return Config(**dataclasses.asdict(config))


_canonicalize_shape = pallas_triton_config.canonicalize_shape


def _maybe_shape(x, axis) -> jax.ShapeDtypeStruct | None:
  if x is None:
    return None
  if axis is None:
    return jax.ShapeDtypeStruct(x.shape, x.dtype)
  return jax.ShapeDtypeStruct(_canonicalize_shape(x.shape, axis), x.dtype)


def get_key(
    residuals: base.Residuals,
    out: jax.Array,
    dout: jax.Array,
    x: jax.Array,
    scale: jax.Array | None,
    offset: jax.Array | None,
    *,
    axis: int,
    scale_offset: float,
    **kwargs,
) -> Key:
  """Returns the lookup key for the given args."""
  # TODO: Canonicalize shape to two dimensions?
  # TODO: Cap the shape at a given size?
  del out, dout  # Unused.
  if kwargs.pop('return_residuals'):
    raise NotImplementedError('`return_residuals` not supported.')
  kwargs.pop('epsilon')
  kwargs.pop('subtract_mean')

  mean, rstddev = residuals
  x_shape = _canonicalize_shape(x.shape, axis)
  rstddev_shape = _canonicalize_shape(rstddev.shape, axis)
  key = immutabledict.immutabledict(
      x=jax.ShapeDtypeStruct(x_shape, x.dtype),
      scale=_maybe_shape(scale, None),
      offset=_maybe_shape(offset, None),
      mean=_maybe_shape(mean, axis),
      rstddev=jax.ShapeDtypeStruct(rstddev_shape, rstddev.dtype),
      has_scale_offset=scale_offset != 0.0,
  )
  assert not kwargs, f'Unhandled kwargs: {kwargs}'
  return key
