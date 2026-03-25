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
"""Common definitions for Pallas-Mosaic-GPU linear softmax cross-entropy loss."""

from typing import Annotated, Any, TypeAlias

import immutabledict
import jax
import jax.numpy as jnp
import pydantic
from tokamax._src import pydantic as pydantic_lib


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Tile-size configuration for the Pallas/Mosaic-GPU kernel.

  The matmul is x[B, H] @ w[H, V] tiled as (B=M, H=K, V=N).

  Attributes:
    tile_m: Tile size over the batch/token (B) dimension. Each CTA handles
      2 * tile_m rows (two warp groups each covering tile_m rows). B must be
      divisible by 2 * tile_m.
    tile_n: Tile size over the vocabulary (V) dimension. V must be divisible
      by tile_n.
    tile_k: Tile size for the inner hidden (H/K) matmul loop. H must be
      divisible by tile_k.
    num_stages: Maximum number of concurrent pipeline stages for async
      TMA prefetch.
  """

  tile_m: Annotated[int, pydantic.Field(ge=128, multiple_of=64)] = 128
  tile_n: Annotated[int, pydantic.Field(ge=64, multiple_of=64)] = 128
  tile_k: Annotated[int, pydantic.Field(ge=16, multiple_of=16)] = 64
  num_stages: pydantic_lib.PowerOfTwo = 4


Key: TypeAlias = immutabledict.immutabledict[str, Any]


def get_heuristics_config(x: jax.Array, w: jax.Array) -> Config:
  """Returns a reasonable default config for H100 (sm90)."""
  del x, w  # shapes don't change the default for sm90
  return Config(tile_m=128, tile_n=128, tile_k=64, num_stages=4)


def get_autotuning_configs(x: jax.Array, w: jax.Array) -> set[Config]:
  """Returns a bounded set of configs to try during autotuning."""
  b_dim, h_dim = x.shape
  v_dim = w.shape[1]

  tile_ms = [t for t in (64, 128) if b_dim % (2 * t) == 0]
  tile_ns = [t for t in (64, 128, 256) if v_dim % t == 0]
  tile_ks = [t for t in (32, 64, 128) if h_dim % t == 0]
  num_stages_opts = [2, 4]

  configs: set[Config] = set()
  for tm in tile_ms:
    for tn in tile_ns:
      for tk in tile_ks:
        for ns in num_stages_opts:
          configs.add(Config(tile_m=tm, tile_n=tn, tile_k=tk, num_stages=ns))
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
