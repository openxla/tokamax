# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Pallas-Triton linear softmax cross-entropy loss configuration."""

from typing import Annotated, Any, TypeAlias

import immutabledict
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import pydantic
from tokamax._src import pydantic as pydantic_lib


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Tile-size configuration for the Pallas/Triton GPU kernel.

  All block sizes must evenly divide the corresponding tensor dimension.

  Attributes:
    b_block_size: Tile size over the batch/token (B) dimension.
    h_block_size: Tile size for the inner hidden (H) matmul loop. Each
      iteration loads a (b_block_size, h_block_size) slice of x and a
      (h_block_size, v_block_size) slice of w; total HBM data volume is the
      same regardless of this value. It controls register pressure and the
      matmul tile shape presented to tensor cores.
    v_block_size: Tile size over the vocabulary (V) dimension.
    num_warps: Number of Triton warps per program.
  """

  b_block_size: Annotated[int, pydantic.Field(ge=16, multiple_of=16)] = 32
  h_block_size: Annotated[int, pydantic.Field(ge=16, multiple_of=16)] = 64
  v_block_size: Annotated[int, pydantic.Field(ge=16, multiple_of=16)] = 128
  num_warps: pydantic_lib.PowerOfTwo = 4


Key: TypeAlias = immutabledict.immutabledict[str, Any]


def get_heuristics_config(
    x: jax.Array,
    w: jax.Array,
) -> Config:
  """Returns a reasonable default config based on the input shapes."""
  b_dim, h_dim = x.shape
  v_dim = w.shape[1]

  # Pick the largest power-of-2 block sizes that divide the dimensions,
  # capped at 1024 per the CLAUDE.md guideline.
  def best_block(dim: int, default: int, cap: int = 1024) -> int:
    size = default
    while size * 2 <= cap and dim % (size * 2) == 0:
      size *= 2
    return size if dim % size == 0 else default

  b_block_size = best_block(b_dim, 32)
  h_block_size = best_block(h_dim, 64)
  v_block_size = best_block(v_dim, 128)

  return Config(
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
      num_warps=4,
  )


def get_autotuning_configs(x: jax.Array, w: jax.Array) -> set[Config]:
  """Returns a bounded set of configs to try during autotuning."""
  b_dim, h_dim = x.shape
  v_dim = w.shape[1]

  sizes = lambda dim: [
      s for s in (16, 32, 64, 128, 256, 512, 1024) if dim % s == 0
  ]

  configs: set[Config] = set()
  for b_block in sizes(b_dim):
    for h_block in sizes(h_dim):
      for v_block in sizes(v_dim):
        for num_warps in (4, 8):
          configs.add(
              Config(
                  b_block_size=b_block,
                  h_block_size=h_block,
                  v_block_size=v_block,
                  num_warps=num_warps,
              )
          )
  return configs


def get_key(
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    *,
    reduction: str,
    **_kwargs,
) -> Key:
  """Returns the autotuning cache lookup key for the given arguments."""
  return immutabledict.immutabledict(
      x=jax.ShapeDtypeStruct(x.shape, x.dtype),
      labels=jax.ShapeDtypeStruct(labels.shape, labels.dtype),
      w=jax.ShapeDtypeStruct(w.shape, w.dtype),
      reduction=reduction,
  )
