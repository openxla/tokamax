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
  """Configuration parameters for Pallas-Mosaic-GPU kernels.

  Attributes:
    block_q: Block size along Q sequence length.
    block_kv: Block size along KV sequence length.
    block_d: Block size along head_dim for updating accumulator.
    num_stages: Number of tma stages for loading KV.
    fold_q_sequence_heads: Whether to fold seq_q into num_q_heads.
    split_k: Number of chunks to split seq_len_k into to improve parallelism.
    num_tma_splits: Number of chunks to load each K/V - helpful to better hide
      GMEM load latences as we can notify TMA warp after part of the mma, thus
      giving more time to TMA loads.
    collective: if True - 2 CTA MMA will be run with M=256, N=128
  """
  # TODO: Relax block size constraints to multiple of 32.
  block_q: pydantic.conint(multiple_of=64, gt=0) = 64
  block_kv: pydantic.conint(multiple_of=64, gt=0) = 64
  num_stages: pydantic.PositiveInt = 2
  fold_q_sequence_heads: pydantic.StrictBool = False
  split_k: pydantic.PositiveInt = 1
  # sm100 specific
  block_d: pydantic.conint(multiple_of=8, gt=0) = 128
  num_tma_splits: pydantic.PositiveInt = 2
  collective: pydantic.StrictBool = True


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
