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

"""Module that encapsulates metadata used for DMA and scheduling."""

import abc
import dataclasses
import functools
from typing import Any, Callable

import jax
from jax import lax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.gmm import config


type WriteGroupFn = Callable[[jax.Array, jax.Array], jax.Array]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataInputs[T]:
  """Inputs required to calculate metadata that were passed as inputs."""

  group_cum_sum: T
  group_offset: T
  row_group: T | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AbstractMetadataRef(abc.ABC):
  """Stores scalar data used for routing and pipeline scheudling."""

  gm_id_to_group_id: hijax.TransformedRef
  gm_id_to_m_offset: hijax.TransformedRef
  gm_id_to_num_rows: hijax.TransformedRef
  num_gm: hijax.TransformedRef
  cfg: config.RoutingGmmConfig = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create_scratch_types(
      cls, cfg: config.RoutingGmmConfig
  ) -> Any:
    """Creates SMEM scratch specs for the metadata tables."""
    return cls(
        gm_id_to_group_id=pltpu.SMEM((cfg.max_num_gm,), jnp.int32),
        gm_id_to_m_offset=pltpu.SMEM((cfg.max_num_gm,), jnp.int32),
        gm_id_to_num_rows=pltpu.SMEM((cfg.max_num_gm,), jnp.int32),
        num_gm=pltpu.SMEM((1,), jnp.int32),
        cfg=cfg,
    )

  @property
  def sink_gm_id(self) -> int:
    return self.cfg.max_num_gm - 1

  @property
  def num_steps(self) -> jax.Array:
    return self.cfg.num_n * self.num_gm[0]

  def step(
      self, n_id: jax.Array | int, gm_id: jax.Array | int, k_id: jax.Array | int
  ) -> jax.Array:
    """Convert (n, gm, k) grid into a linear grid."""
    # TODO: Fold k into the step once num_k > 1 is supported.
    del k_id
    return n_id * self.num_gm[0] + gm_id

  def slot(
      self,
      n_id: jax.Array | int,
      gm_id: jax.Array | int,
      k_id: jax.Array | int,
      buffer_count: int,
  ) -> jax.Array:
    # Because gm_id is used for lookahead/lookback, the smallest possible
    # value is -buffer_count. To protect against uint underflow, we add
    # buffer_count first to ensure slot indexing is always correct.
    step_u32 = self.step(n_id, gm_id + buffer_count, k_id).astype(jnp.uint32)
    return lax.rem(step_u32, jnp.uint32(buffer_count))

  def resolve(
      self,
      n_id: jax.Array | int,
      gm_id: jax.Array | int,
      k_id: jax.Array | int,
      dtype: jax.typing.DTypeLike = jnp.int32,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Maps a possibly out-of-range (n, gm, k) to an in-range or the sink."""
    step = self.step(n_id, gm_id, k_id)
    valid = jnp.logical_and(0 <= step, step < self.num_steps)
    # TODO: Add equivalent logic for K tiling when the feature lands.
    if self.cfg.num_n > 1:
      # Because of trucated division, if signed integer is used, compiler
      # introduces various additional costly instructions to ensure division
      # returns correct number. This can be avoid if unsighed integer is used.
      num_gm_u32 = self.num_gm[0].astype(jnp.uint32)
      step_u32 = step.astype(jnp.uint32)
      # Even if step_u32 is negative and returns incorrect n_id, it will not
      # cause any issues as it means the step is invalid and issue no-op DMA.
      n_id = lax.div(step_u32, num_gm_u32)
      # Even if step_u32 is negative and returns incorrect gm_id, they will be
      # clamped into sink value anyway.
      gm_id = lax.rem(step_u32, num_gm_u32)
    gm_id = jnp.where(valid, gm_id, self.sink_gm_id)
    return (
        jnp.asarray(n_id, dtype),
        jnp.asarray(gm_id, dtype),
        jnp.asarray(k_id, dtype),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SortedMetadataRef(AbstractMetadataRef):
  """Metadata from routing sorted outside the kernel.

  XLA sorts the routed rows by group and counts the rows of each group
  (`group_cum_sum`); the kernel walks this shard's groups to lay out the tiles.
  """

  def fill_metadata_and_return_num_gm(
      self, metadata_in: MetadataInputs[hijax.TransformedRef]
  ) -> jax.Array:
    """Iterate group sizes to metadata about DMA and pipeline schedule logic."""

    # NOTE: Unlike GMM v2, there is no need for complex sublane alignment logic
    # as issuing one dma per row avoids any shared chunk or non-zero starting
    # offset issue that was found in GMM v2.

    group_start = metadata_in.group_offset[0]
    group_end = group_start + self.cfg.dim.size_local_groups
    self.gm_id_to_group_id[self.sink_gm_id] = 0
    self.gm_id_to_m_offset[self.sink_gm_id] = 0
    self.gm_id_to_num_rows[self.sink_gm_id] = 0

    @jax.named_scope("write_tile")
    def write_tile(
        gm_id: jax.Array,
        offset: jax.Array,
        *,
        local_group_id: jax.Array,
        end: jax.Array
    ):
      num_rows = jnp.minimum(end - offset, self.cfg.plan.tile_m)
      self.gm_id_to_group_id[gm_id] = local_group_id
      self.gm_id_to_m_offset[gm_id] = offset
      self.gm_id_to_num_rows[gm_id] = num_rows
      return offset + num_rows

    @jax.named_scope("write_group")
    def write_group(group_id: jax.Array, num_gm: jax.Array):
      start = metadata_in.group_cum_sum[group_id]
      end = metadata_in.group_cum_sum[group_id + 1]
      next_num_gm = num_gm + pl.cdiv(end - start, self.cfg.plan.tile_m)
      local_group_id = group_id - group_start

      jax.lax.fori_loop(
          num_gm,
          next_num_gm,
          functools.partial(write_tile, local_group_id=local_group_id, end=end),
          start,
      )
      return next_num_gm

    num_gm = self._walk_groups(metadata_in, write_group, group_start, group_end)
    self.num_gm[0] = num_gm
    return num_gm

  def _walk_groups(
      self,
      metadata_in: MetadataInputs[hijax.TransformedRef],
      write_group: WriteGroupFn,
      group_start: jax.Array,
      group_end: jax.Array,
  ) -> jax.Array:
    """Walks this shard's groups to lay out their tiles."""
    # Walk the groups, but skip over "skip" consecutive empty elements.
    skip = self.cfg.plan.skip_metadata_size

    def step(carry):
      group_id, num_gm = carry
      ahead = jnp.minimum(group_id + skip + 1, group_end)
      empty = (
          metadata_in.group_cum_sum[ahead]
          == metadata_in.group_cum_sum[group_id]
      )
      # If the cumulative sum hasn't changed, then we can skip these.
      num_gm = write_group(
          group_id, num_gm
      )  # if the group is empty, nothing is written.
      return jnp.where(empty, ahead, group_id + 1), num_gm

    if skip == 0:
      # There's overhead of our checks, so standard fori is better here.
      return jax.lax.fori_loop(group_start, group_end, write_group, 0)
    _, num_gm = jax.lax.while_loop(
        lambda carry: carry[0] < group_end,
        step,
        (group_start, jnp.int32(0)),
    )
    return num_gm


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LiveGroupMetadataRef(SortedMetadataRef):
  """Metadata from routing sorted outside the kernel, walking only live groups.

  Iterates only the local groups that have at least one routed row.
  """

  def _walk_groups(
      self,
      metadata_in: MetadataInputs[hijax.TransformedRef],
      write_group: WriteGroupFn,
      group_start: jax.Array,
      group_end: jax.Array,
  ) -> jax.Array:
    """Walks only the local groups that have at least one routed row."""
    assert metadata_in.row_group is not None
    row = metadata_in.group_cum_sum[group_start]
    row_end = metadata_in.group_cum_sum[group_end]

    def cond_fn(val: tuple[jax.Array, jax.Array]) -> jax.Array:
      row_idx, _ = val
      return row_idx < row_end

    def body_fn(
        val: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
      row_idx, num_gm = val
      group_id = metadata_in.row_group[row_idx]
      next_num_gm = write_group(group_id, num_gm)
      next_row = metadata_in.group_cum_sum[group_id + 1]
      return (next_row, next_num_gm)

    _, num_gm = jax.lax.while_loop(
        cond_fn, body_fn, (row, jnp.array(0, dtype=jnp.int32))
    )
    return num_gm


# Not a fan of return type Any but despite metadataref being flattened using
# register_dataclass, linter complains mismatch with scratch_typess :(
def create_metadata(
    group_cum_sum: jax.Array,
    group_offset: jax.Array,
    cfg: config.RoutingGmmConfig,
    *,
    row_group: jax.Array | None = None,
) -> tuple[MetadataInputs[jax.Array], Any]:
  """Returns the SMEM metadata inputs and scratch specs for this config."""
  if row_group is not None and cfg.plan.live_group_walk:
    metadata_in = MetadataInputs(
        group_cum_sum=group_cum_sum,
        group_offset=group_offset,
        row_group=row_group,
    )
    return metadata_in, LiveGroupMetadataRef.create_scratch_types(cfg)
  metadata_in = MetadataInputs(
      group_cum_sum=group_cum_sum,
      group_offset=group_offset,
  )
  return metadata_in, SortedMetadataRef.create_scratch_types(cfg)
