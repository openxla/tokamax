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
"""TPU harness and NumPy reference for metadata walk tests."""

import dataclasses
import jax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.gmm import config
from tokamax._src.ops.experimental.routing_moe.gmm import memory_ref
from tokamax._src.ops.experimental.routing_moe.gmm import metadata
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases
from tokamax._src.ops.experimental.routing_moe.test.common import test_inputs

# Columns of a tile table: local group, first sorted row, and rows in the tile.
_TILE_COLUMNS = ("group_id", "m_offset", "num_rows")


@dataclasses.dataclass(frozen=True)
class RoutingCase:
  """A routing and the shard whose groups the walk lays out.

  Attributes:
    case: Test case specifying the MoE shape and token count.
    group_offset: Starting global expert index of this shard.
    num_skipped: Trailing experts that receive no routed rows.
  """

  case: test_cases.Case
  group_offset: int = 0
  num_skipped: int = 0


# Every shared unit case, plus shards in the middle of the experts, a single
# token, groups spanning several tiles, and a shard that no token routes to.
ROUTINGS = (
    *(RoutingCase(c) for c in test_cases.UNIT_CASES),
    RoutingCase(test_cases.DECODE_64_EP8, group_offset=128),
    RoutingCase(test_cases.DECODE_1_EP8, group_offset=128),
    RoutingCase(test_cases.PREFILL_16E),
    RoutingCase(test_cases.DECODE_64_EP8, group_offset=448, num_skipped=64),
)

PLANS = (
    gmm_plan.RoutingGmmPlan(skip_metadata_size=0),
    gmm_plan.RoutingGmmPlan(skip_metadata_size=1),
    gmm_plan.RoutingGmmPlan(skip_metadata_size=2),
    gmm_plan.RoutingGmmPlan(skip_metadata_size=63),
    gmm_plan.RoutingGmmPlan(skip_metadata_size=1024),
    gmm_plan.RoutingGmmPlan(live_group_walk=True),
)


def make_gmm_config(
    *,
    num_rows: int,
    num_global_groups: int,
    num_local_groups: int,
    size_k: int,
    size_n: int,
    custom_plan: gmm_plan.RoutingGmmPlan | None = None,
) -> config.RoutingGmmConfig:
  """Builds a direct-mode bf16 config resolved the same way as `routing_gmm`."""
  x = arg.ActivationSpec(
      row_size=num_rows, col_size=size_k, dtype=jnp.dtype(jnp.bfloat16)
  )
  w = (
      arg.WeightSpec(
          group_size=num_local_groups,
          in_size=size_k,
          out_size=size_n,
          dtype=jnp.dtype(jnp.bfloat16),
      ),
  )
  tpu_info = pltpu.get_tpu_info()
  dim = config.Dimension.create(
      x=x,
      w=w,
      num_global_groups=num_global_groups,
      gather=None,
      scatter=None,
      fuse_act=None,
  )
  resolved_plan = gmm_plan.ResolvedGmmPlan.create(
      custom_plan,
      x=x,
      w=w,
      out_dtype=None,
      size_out_m=dim.size_out_m,
      size_global_groups=dim.size_global_groups,
      size_k=dim.size_k,
      size_out_n=dim.size_out_n,
      tpu_info=tpu_info,
  )
  return config.RoutingGmmConfig.create(
      x=x,
      w=w,
      dim=dim,
      plan=resolved_plan,
      gather=None,
      scatter=None,
      fuse_act=None,
      out_dtype=None,
      tpu_info=tpu_info,
  )


def _fill_kernel(
    metadata_in_ref: metadata.MetadataInputs[hijax.TransformedRef],
    tiles_out: hijax.TransformedRef,
    num_gm_out: hijax.TransformedRef,
    metadata_ref: metadata.SortedMetadataRef,
) -> None:
  """Fills the metadata scratch and copies its tables to the SMEM outputs."""
  num_gm_out[0] = metadata_ref.fill_metadata_and_return_num_gm(metadata_in_ref)
  num_columns = len(_TILE_COLUMNS)

  # SMEM outputs must be 1-D, so tiles are flattened row-major.
  def copy(gm_id: jax.Array, carry: None) -> None:
    base = gm_id * num_columns
    tiles_out[base] = metadata_ref.gm_id_to_group_id[gm_id]
    tiles_out[base + 1] = metadata_ref.gm_id_to_m_offset[gm_id]
    tiles_out[base + 2] = metadata_ref.gm_id_to_num_rows[gm_id]
    return carry

  jax.lax.fori_loop(0, metadata_ref.cfg.max_num_gm, copy, None)


def fill_metadata(
    routing: test_inputs.Routing, cfg: config.RoutingGmmConfig
) -> np.ndarray:
  """Runs the metadata walk of `cfg` on TPU and returns `[num_gm, 3]` tiles."""
  group_cum_sum = jnp.cumulative_sum(routing.group_sizes, include_initial=True)
  metadata_in, metadata_scratch = metadata.create_metadata(
      group_cum_sum.astype(jnp.int32),
      jnp.array([routing.group_offset], dtype=jnp.int32),
      cfg,
      row_group=routing.row_group if cfg.plan.live_group_walk else None,
  )

  # Memory space constraints are only honored under jit, as in `routing_gmm`.
  @jax.jit
  def run(
      metadata_in: metadata.MetadataInputs[jax.Array],
  ) -> tuple[jax.Array, jax.Array]:
    metadata_in = memory_ref.map_memory_constraint(metadata_in, pltpu.SMEM)
    return pl.kernel(
        _fill_kernel,
        mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=1),
        out_type=(
            pltpu.SMEM((cfg.max_num_gm * len(_TILE_COLUMNS),), jnp.int32),
            pltpu.SMEM((1,), jnp.int32),
        ),
        scratch_types=(metadata_scratch,),
    )(metadata_in)

  tiles, num_gm = run(metadata_in)
  tiles = np.asarray(tiles).reshape(cfg.max_num_gm, len(_TILE_COLUMNS))
  return tiles[: int(num_gm[0])]


def expected_metadata(routing: test_inputs.Routing, tile_m: int) -> np.ndarray:
  """Computes expected `[num_gm, 3]` tiles of `routing` with `tile_m` rows."""
  group_cum_sum = np.concatenate([[0], np.cumsum(routing.group_sizes)])
  tiles = []
  for local_group_id in range(routing.num_local_experts):
    group_id = routing.group_offset + local_group_id
    start, end = int(group_cum_sum[group_id]), int(group_cum_sum[group_id + 1])
    for offset in range(start, end, tile_m):
      tiles.append((local_group_id, offset, min(end - offset, tile_m)))
  return np.asarray(tiles, np.int32).reshape(-1, len(_TILE_COLUMNS))
