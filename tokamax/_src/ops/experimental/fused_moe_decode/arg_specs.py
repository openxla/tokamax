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
"""Benchmark and autotuning argument specifications for Fused MoE Decode."""

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec

_AXIS = "d"
_P = jax.sharding.PartitionSpec


class _RankInitializer(jax.ShapeDtypeStruct, numerics.ArrayInitializer):
  """Initializes per-rank mesh indices `[0, 1, ..., n_ranks - 1]`."""

  def __init__(
      self,
      shape: tuple[int, ...] = (1,),
      sharding: jax.sharding.Sharding | None = None,
  ):
    jax.ShapeDtypeStruct.__init__(self, shape, jnp.int32, sharding=sharding)

  def __call__(self, rng: np.random.Generator) -> np.ndarray:
    del rng
    return np.arange(self.shape[0], dtype=np.int32).reshape(self.shape)

  def with_sharding(
      self, sharding: jax.sharding.Sharding
  ) -> "_RankInitializer":
    return _RankInitializer(self.shape, sharding=sharding)

  def with_shape_and_sharding(
      self,
      shape: tuple[int, ...],
      sharding: jax.sharding.Sharding,
  ) -> "_RankInitializer":
    return _RankInitializer(shape, sharding=sharding)


def _make_spec(
    name: str,
    *,
    tokens: int,
    experts_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
    shared_intermediate_size: int,
    top_k: int,
    n_ranks: int = 8,
    shared_gated: bool = True,
    project: str,
    tags: tuple[arg_spec.Tag, ...] = ("forward_only",),
) -> arg_spec.ArgSpec:
  """Creates a `shard_map`-native `ArgSpec` for Fused MoE Decode."""
  h, i, i_s, e = (
      hidden_size,
      intermediate_size,
      shared_intermediate_size,
      experts_per_rank,
  )
  f8 = jnp.float8_e4m3fn
  bf16 = jnp.bfloat16
  f32 = jnp.float32
  sds = jax.ShapeDtypeStruct

  args = dict(
      rank=_RankInitializer((1,)),
      hidden=sds((tokens, h), bf16),
      w1=sds((e, h, 2 * i), f8),
      w2=sds((e, i, h), f8),
      w1_scale=sds((e, 1, 1, 2 * i), f32),
      w2_scale=sds((e, 1, 1, h), f32),
      shared_w1=sds((h, 2 * i_s), f8),
      shared_w1_scale=sds((1, 2 * i_s), f32),
      shared_w2=sds((i_s, h), f8),
      shared_w2_scale=sds((1, h), f32),
      shared_gate_weight=sds((h, 1), bf16) if shared_gated else None,
      topk_weights=numerics.FloatRangedArrayInitializer(
          (tokens, top_k), bf16, 0.0, 1.0
      ),
      topk_ids=numerics.RangedArrayInitializer(
          (tokens, top_k), jnp.int32, 0, e
      ),
      n_ranks=n_ranks,
      axis_name=_AXIS,
  )
  mesh = jax.sharding.AbstractMesh(
      (n_ranks,), (_AXIS,), axis_types=(jax.sharding.AxisType.Auto,)
  )
  shard_map = arg_spec.ShardMapSpec(
      mesh=mesh,
      in_specs={
          "rank": _P(_AXIS),
          "hidden": _P(),
          "w1": _P(_AXIS),
          "w2": _P(_AXIS),
          "w1_scale": _P(_AXIS),
          "w2_scale": _P(_AXIS),
          "shared_w1": _P(_AXIS),
          "shared_w1_scale": _P(_AXIS),
          "shared_w2": _P(_AXIS),
          "shared_w2_scale": _P(_AXIS),
          "shared_gate_weight": _P(),
          "topk_weights": _P(),
          "topk_ids": _P(),
      },
      out_specs=_P(_AXIS),
      check_vma=False,
  )
  return arg_spec.ArgSpec(
      args=args,
      project=project,
      name=name,
      tags=tags,
      excluded_platforms=("gpu",),
      shard_map=shard_map,
  )


ARG_SPECS: tuple[arg_spec.ArgSpec, ...] = (
    _make_spec(
        "decode_16_tp1",
        tokens=16,
        experts_per_rank=16,
        hidden_size=2048,
        intermediate_size=256,
        shared_intermediate_size=128,
        top_k=4,
        n_ranks=1,
        project="deepseek_v3",
    ),
    *(
        _make_spec(
            f"decode_{tokens}_tp8",
            tokens=tokens,
            experts_per_rank=32,
            hidden_size=7168,
            intermediate_size=256,
            shared_intermediate_size=256,
            top_k=8,
            n_ranks=8,
            project="deepseek_v3",
        )
        for tokens in (16, 64, 128)
    ),
)
