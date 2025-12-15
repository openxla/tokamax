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
"""Common utilities for Mosaic GPU attention implementations."""

from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import pydantic


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  # TODO: Relax constraints to multiple of 32.
  block_q: pydantic.conint(multiple_of=64, gt=0) = 64
  block_kv: pydantic.conint(multiple_of=64, gt=0) = 64
  num_stages: pydantic.conint(gt=1) = 2
  fold_q_sequence_heads: bool = False
  split_k: pydantic.PositiveInt = 1


def load_bcast(
    ref: Any,
    idx: tuple[int | jax.Array | pl.Slice, ...],
    *,
    layout: Any,
    optimized: bool = False,
) -> jax.Array:
  """Loads from a reference, with given index, broadcasting if needed."""
  new_idx = []
  shape = []
  bcast_dims = []
  # NOTE: We could add support for `idx` shorter than `ref.ndim`.
  for d, ix in zip(ref.shape, idx, strict=True):
    new_idx.append(0 if d == 1 else ix)

    if isinstance(ix, pl.Slice):
      if d == 1:
        layout = layout.reduce(len(shape))
      else:
        bcast_dims.append(len(shape))
      shape.append(ix.size)

  if not bcast_dims:
    return ref[tuple(new_idx)]  # Return a scalar value.
  value = plgpu.load(ref, tuple(new_idx), layout=layout, optimized=optimized)
  return jax.lax.broadcast_in_dim(value, shape, bcast_dims)


def num_bits(dtype: jax.typing.DTypeLike) -> int:
  fn = jnp.finfo if jnp.issubdtype(dtype, jnp.floating) else jnp.iinfo
  return fn(dtype).bits


def tile_swizzle_transforms(
    shape: tuple[int, ...], dtype: jax.typing.DTypeLike, what: str = ""
) -> tuple[plgpu.TilingTransform, plgpu.SwizzleTransform]:
  """Returns tiling and swizzling transforms."""
  elem_bits = num_bits(dtype)
  swizzle = plgpu.find_swizzle(shape[-1] * elem_bits, what)
  tiling = (8, 8 * swizzle // elem_bits)
  return plgpu.TilingTransform(tiling), plgpu.SwizzleTransform(swizzle)
