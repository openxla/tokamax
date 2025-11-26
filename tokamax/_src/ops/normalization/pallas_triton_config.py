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
"""Pallas-Triton normalization op configuration."""

from collections.abc import Sequence
import math
from typing import Any, TypeAlias

import immutabledict
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import pydantic
from tokamax._src import gpu_utils
from tokamax._src import pydantic as pydantic_lib


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_m: pydantic_lib.PowerOfTwo
  block_n: pydantic_lib.PowerOfTwo | None
  num_warps: pydantic_lib.PowerOfTwo


Key: TypeAlias = immutabledict.immutabledict[str, Any]


def canonicalize_shape_3d(
    shape: Sequence[int], axis: int
) -> tuple[int, int, int]:
  return (math.prod(shape[:axis]), shape[axis], math.prod(shape[axis:][1:]))


def canonicalize_shape(
    shape: Sequence[int], axis: int
) -> tuple[int, int] | tuple[int, int, int]:
  if len(shape[axis:]) > 1:
    return canonicalize_shape_3d(shape, axis)
  return (math.prod(shape[:axis]), shape[axis])


def _maybe_shape(x) -> jax.ShapeDtypeStruct | None:
  return jax.ShapeDtypeStruct(x.shape, x.dtype) if x is not None else None


def get_key(
    x: jax.Array,
    scale: jax.Array | None,
    offset: jax.Array | None,
    *,
    axis: int,
    return_residuals: bool = False,
    **kwargs,
) -> Key:
  """Returns the lookup key for the given args."""
  # TODO: Cap the shape at a given size?
  key = immutabledict.immutabledict(
      x=jax.ShapeDtypeStruct(canonicalize_shape(x.shape, axis), x.dtype),
      scale=_maybe_shape(scale),
      offset=_maybe_shape(offset),
      has_epsilon=kwargs.pop('epsilon') != 0.0,
      has_scale_offset=kwargs.pop('scale_offset') != 0.0,
      subtract_mean=kwargs.pop('subtract_mean'),
      return_residuals=return_residuals,
  )
  assert not kwargs, f'Unhandled kwargs: {kwargs}'
  return key


def get_heuristics_config(
    x: jax.Array,
    scale: jax.Array | None,
    offset: jax.Array | None,
    *,
    axis: int,
    block_size_per_warp: int = 1024,
    vmap_axis_sizes: tuple[int, ...],
    **_,
) -> Config:
  """Returns a config based on heuristics."""
  x = jax.ShapeDtypeStruct(canonicalize_shape(x.shape, axis), x.dtype)
  # We get diminishing returns, and worse load-balancing, with `block_m > 32`.
  # `block_m == 1` appears best whenever not reducing in trailing axis.
  block_m = 32 if x.ndim == 2 else 1
  if scale is None and offset is None:
    block_m = 1  # There is no oportunity to re-use data.

  block_size = block_m * pl.next_power_of_2(x.shape[1])
  num_blocks = pl.cdiv(x.shape[0], block_m) * math.prod(vmap_axis_sizes)
  if x.ndim > 2:
    # Read full cache line at a time.
    els_per_cache_line = (
        gpu_utils.CACHE_LINE_SIZE_BYTES // jnp.dtype(x.dtype).itemsize
    )
    block_n = min(els_per_cache_line, pl.next_power_of_2(x.shape[2]))
    block_size *= block_n
    num_blocks *= pl.cdiv(x.shape[2], block_n)
  else:
    block_n = None

  # Pick a block size that fits into registers, and launch enough blocks to fill
  # the device.
  max_block_size = gpu_utils.NUM_REGISTERS_PER_SM // 4
  min_num_blocks = 4 * jax.devices()[0].core_count
  while (block_m > 1) and (
      (block_size > max_block_size) or (num_blocks < min_num_blocks)
  ):
    block_m //= 2
    block_size //= 2
    num_blocks *= 2

  num_warps = min(pl.cdiv(block_size, block_size_per_warp), 4)
  return Config(block_m=block_m, block_n=block_n, num_warps=num_warps)
