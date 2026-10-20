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
from typing import Any

import jax
from jax import lax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.gmm import config


@dataclasses.dataclass(frozen=True)
class AbstractMetadataRef(abc.ABC):
  """Stores scalar data used for routing and pipeline scheudling."""

  gm_id_to_group_id: jax.Array | hijax.TransformedRef
  gm_id_to_m_offset: jax.Array | hijax.TransformedRef
  gm_id_to_num_rows: jax.Array | hijax.TransformedRef
  num_gm: jax.Array | hijax.TransformedRef
  cfg: config.RoutingGmmConfig = dataclasses.field(metadata=dict(static=True))

  @staticmethod
  def _empty_tables(cfg: config.RoutingGmmConfig) -> dict[str, Any]:
    return dict(
        gm_id_to_group_id=jnp.empty((cfg.max_num_gm,), jnp.int32),
        gm_id_to_m_offset=jnp.empty((cfg.max_num_gm,), jnp.int32),
        gm_id_to_num_rows=jnp.empty((cfg.max_num_gm,), jnp.int32),
        num_gm=jnp.empty((1,), jnp.int32),
    )

  @property
  def idx_ref(self) -> jax.Array | hijax.TransformedRef | None:
    """SMEM-resident row-index table, or None when streamed from HBM."""
    return None

  @property
  def gather_idx_ref(self) -> jax.Array | hijax.TransformedRef | None:
    """SMEM-resident gather row-index table, or None when streamed from HBM."""
    return self.idx_ref

  @property
  def scatter_idx_ref(self) -> jax.Array | hijax.TransformedRef | None:
    """SMEM-resident scatter row-index table, or None when streamed from HBM."""
    return self.idx_ref

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
    if self.cfg.num_n > 1:
      num_gm_u32 = self.num_gm[0].astype(jnp.uint32)
      step_u32 = step.astype(jnp.uint32)
      # NOTE: This requires update when K tiling is added.
      n_id = lax.div(step_u32, num_gm_u32)
      gm_id = lax.rem(step_u32, num_gm_u32)
    gm_id = jnp.where(valid, gm_id, self.sink_gm_id)
    return (
        jnp.asarray(n_id, dtype),
        jnp.asarray(gm_id, dtype),
        jnp.asarray(k_id, dtype),
    )

  @abc.abstractmethod
  def fill_metadata_and_return_num_gm(self) -> jax.Array:
    """Fills the tables and returns the number of tiles."""
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SortedMetadataRef(AbstractMetadataRef):
  """Metadata from routing sorted outside the kernel.

  XLA sorts the routed rows by group and counts the rows of each group
  (`group_cum_sum`); the kernel walks this shard's groups to lay out the tiles.
  """

  group_cum_sum: jax.Array | hijax.TransformedRef
  group_offset: jax.Array | hijax.TransformedRef

  @classmethod
  def create(
      cls,
      group_cum_sum: jax.Array,
      group_offset: jax.Array,
      cfg: config.RoutingGmmConfig,
      row_group: jax.Array | None = None,
  ) -> "SortedMetadataRef":
    """Wraps scalar data for metadata."""
    del row_group
    return cls(
        **cls._empty_tables(cfg),
        group_cum_sum=group_cum_sum,
        group_offset=group_offset,
        cfg=cfg,
    )

  def fill_metadata_and_return_num_gm(self) -> jax.Array:
    """Iterate group sizes to metadata about DMA and pipeline schedule logic."""

    # NOTE: Unlike GMM v2, there is no need for complex sublane alignment logic
    # as issuing one dma per row avoids any shared chunk or non-zero starting
    # offset issue that was found in GMM v2.

    group_start = self.group_offset[0]
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
      start = self.group_cum_sum[group_id]
      end = self.group_cum_sum[group_id + 1]
      next_num_gm = num_gm + pl.cdiv(end - start, self.cfg.plan.tile_m)
      local_group_id = group_id - group_start

      jax.lax.fori_loop(
          num_gm,
          next_num_gm,
          functools.partial(write_tile, local_group_id=local_group_id, end=end),
          start,
      )
      return next_num_gm

    num_gm = self._walk_groups(write_group, group_start, group_end)
    self.num_gm[0] = num_gm
    return num_gm

  def _walk_groups(
      self,
      write_group: Any,
      group_start: jax.Array,
      group_end: jax.Array,
  ) -> jax.Array:
    """Walks this shard's groups to lay out their tiles."""
    # Walk the groups, but skip over "skip" consecutive empty elements.
    skip = self.cfg.plan.skip_metadata_size

    def step(carry):
      group_id, num_gm = carry
      ahead = jnp.minimum(group_id + skip + 1, group_end)
      empty = self.group_cum_sum[ahead] == self.group_cum_sum[group_id]
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

  # The global group of each sorted routed row.
  row_group: jax.Array | hijax.TransformedRef

  @classmethod
  def create(
      cls,
      group_cum_sum: jax.Array,
      group_offset: jax.Array,
      cfg: config.RoutingGmmConfig,
      row_group: jax.Array | None = None,
  ) -> "LiveGroupMetadataRef":
    """Wraps scalar data for metadata and the group of each sorted row."""
    assert row_group is not None
    return cls(
        **cls._empty_tables(cfg),
        group_cum_sum=group_cum_sum,
        group_offset=group_offset,
        row_group=row_group,
        cfg=cfg,
    )

  def _walk_groups(
      self,
      write_group: Any,
      group_start: jax.Array,
      group_end: jax.Array,
  ) -> jax.Array:
    """Walks only the local groups that have at least one routed row."""
    row = self.group_cum_sum[group_start]
    row_end = self.group_cum_sum[group_end]

    def cond_fn(val: tuple[jax.Array, jax.Array]) -> jax.Array:
      row_idx, _ = val
      return row_idx < row_end

    def body_fn(
        val: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
      row_idx, num_gm = val
      group_id = self.row_group[row_idx]
      next_num_gm = write_group(group_id, num_gm)
      next_row = self.group_cum_sum[group_id + 1]
      return (next_row, next_num_gm)

    _, num_gm = jax.lax.while_loop(
        cond_fn, body_fn, (row, jnp.array(0, dtype=jnp.int32))
    )
    return num_gm


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LowTokMetadataRef(AbstractMetadataRef):
  """Metadata built in the kernel from the raw top-k ids, for few tokens.

  Nothing is sorted outside the kernel. The fill visits the routed slots
  (tokens x topk) three times on the scalar core: it finds this shard's
  distinct groups, in order of first appearance, and counts their rows; lays
  out each group's rows contiguously and writes its tiles; then gives every slot
  the next row of its group and writes that row's gather index (slot // topk)
  or scatter index (slot). The cost grows with the number of slots, not with
  the number of groups.

  Rows of a group keep slot order, as a stable sort does, so the routing gmm
  computes the same rows as with `SortedMetadataRef`; only the order of the
  groups differs. The routing gmms of a layer must see the same slots, so that
  they agree on where each row is.
  """

  gather_idx: jax.Array | hijax.TransformedRef | None
  scatter_idx: jax.Array | hijax.TransformedRef | None
  slot_group: jax.Array | hijax.TransformedRef  # [slots]: global group ids.
  group_offset: jax.Array | hijax.TransformedRef
  topk: int = dataclasses.field(metadata=dict(static=True))

  @property
  def idx_ref(self) -> jax.Array | hijax.TransformedRef | None:
    return self.gather_idx if self.gather_idx is not None else self.scatter_idx

  @property
  def gather_idx_ref(self) -> jax.Array | hijax.TransformedRef | None:
    return self.gather_idx

  @property
  def scatter_idx_ref(self) -> jax.Array | hijax.TransformedRef | None:
    return self.scatter_idx

  @classmethod
  def create(
      cls,
      gather_slots: jax.Array | None,
      scatter_slots: jax.Array | None,
      group_offset: jax.Array,
      cfg: config.RoutingGmmConfig,
      topk: int,
  ) -> "LowTokMetadataRef":
    """Creates SMEM metadata and resident index table for LowTok routing."""
    if not isinstance(topk, int) or isinstance(topk, bool) or topk <= 0:
      raise ValueError(f"topk must be a positive int. Got {topk}.")
    if gather_slots is None and scatter_slots is None:
      raise ValueError("At least one of gather or scatter slots is required.")
    if cfg.out.mode == config.RoutingMode.REMOTE:
      raise NotImplementedError("LowTok does not support remote routing.")
    slots = gather_slots if gather_slots is not None else scatter_slots
    assert slots is not None
    # Filled by the kernel; padded by tile_m like the sorted indices.
    idx_shape = (slots.shape[0] + cfg.plan.tile_m,)
    gather_idx = (
        None if gather_slots is None else jnp.empty(idx_shape, jnp.int32)
    )
    scatter_idx = (
        None if scatter_slots is None else jnp.empty(idx_shape, jnp.int32)
    )
    return cls(
        **cls._empty_tables(cfg),
        gather_idx=gather_idx,
        scatter_idx=scatter_idx,
        slot_group=slots.astype(jnp.int32),
        group_offset=group_offset,
        cfg=cfg,
        topk=topk,
    )

  def fill_metadata_and_return_num_gm(self) -> jax.Array:
    num_slots = int(self.slot_group.shape[0])
    num_groups = int(self.cfg.dim.size_local_groups)
    tile_m = self.cfg.plan.tile_m
    max_distinct = min(num_slots, num_groups)
    group_start = self.group_offset[0]

    def body(seen, distinct, count, cursor, slot_distinct):
      # Pass 1: this shard's distinct groups and their number of rows.
      def find(slot_id, num_seen):
        group_id = self.slot_group[slot_id] - group_start
        local = (group_id >= 0) & (group_id < num_groups)
        group_id = jnp.clip(group_id, 0, num_groups - 1)
        # seen[group_id] is group_id's 1-based position in `distinct`, valid
        # only if `distinct` agrees, so that no pass has to clear the table.
        dist_id = seen[group_id] - 1
        known = (dist_id >= 0) & (dist_id < num_seen)
        known &= distinct[jnp.clip(dist_id, 0, max_distinct - 1)] == group_id
        new = local & jnp.logical_not(known)
        dist_id = jnp.where(known, dist_id, num_seen)

        @pl.when(new)
        def _():
          distinct[num_seen] = group_id
          seen[group_id] = num_seen + 1
          count[num_seen] = 0

        @pl.when(local)
        def _():
          count[dist_id] = count[dist_id] + 1

        slot_distinct[slot_id] = jnp.where(local, dist_id, -1)
        return num_seen + new.astype(jnp.int32)

      num_distinct = jax.lax.fori_loop(0, num_slots, find, jnp.int32(0))

      # Pass 2: lay out each group's rows contiguously and write its tiles.
      self.gm_id_to_group_id[self.sink_gm_id] = 0
      self.gm_id_to_m_offset[self.sink_gm_id] = 0
      self.gm_id_to_num_rows[self.sink_gm_id] = 0

      def lay_out(dist_id, carry):
        num_gm, start = carry
        group_id = distinct[dist_id]
        end = start + count[dist_id]
        cursor[dist_id] = start

        def write_tile(gm_id, offset):
          num_rows = jnp.minimum(end - offset, tile_m)
          self.gm_id_to_group_id[gm_id] = group_id
          self.gm_id_to_m_offset[gm_id] = offset
          self.gm_id_to_num_rows[gm_id] = num_rows
          return offset + num_rows

        next_num_gm = num_gm + pl.cdiv(end - start, tile_m)
        jax.lax.fori_loop(num_gm, next_num_gm, write_tile, start)
        return next_num_gm, end

      num_gm, num_rows = jax.lax.fori_loop(
          0, num_distinct, lay_out, (jnp.int32(0), jnp.int32(0))
      )
      self.num_gm[0] = num_gm

      # Pass 3: give each slot the next row of its group.
      def place(slot_id, carry):
        dist_id = slot_distinct[slot_id]

        @pl.when(dist_id >= 0)
        def _():
          row = cursor[dist_id]
          cursor[dist_id] = row + 1
          if self.gather_idx is not None:
            self.gather_idx[row] = slot_id // self.topk
          if self.scatter_idx is not None:
            self.scatter_idx[row] = slot_id

        return carry

      jax.lax.fori_loop(0, num_slots, place, 0)

      # The last tile reads up to tile_m indices past the last row.
      def clear(i, carry):
        if self.gather_idx is not None:
          self.gather_idx[num_rows + i] = 0
        if self.scatter_idx is not None:
          self.scatter_idx[num_rows + i] = 0
        return carry

      jax.lax.fori_loop(0, tile_m, clear, 0)
      return num_gm

    return pl.run_scoped(
        body,
        pltpu.SMEM((num_groups,), jnp.int32),
        pltpu.SMEM((max_distinct,), jnp.int32),
        pltpu.SMEM((max_distinct,), jnp.int32),
        pltpu.SMEM((max_distinct,), jnp.int32),
        pltpu.SMEM((num_slots,), jnp.int32),
    )


def create_metadata_ref(
    group_cum_sum: jax.Array | None,
    group_offset: jax.Array,
    cfg: config.RoutingGmmConfig,
    *,
    row_group: jax.Array | None = None,
    gather_idx: jax.Array | None = None,
    scatter_idx: jax.Array | None = None,
    topk: int | None = None,
) -> AbstractMetadataRef:
  """Returns the metadata ref that fills the tables for this config.

  Args:
    group_cum_sum: The cumulative rows of the groups, without `topk`.
    group_offset: The first group of this shard.
    cfg: The routing gmm config.
    row_group: Without `topk`, the group of each sorted routed row; if given
      and the plan selects it, only the groups with rows are walked.
    gather_idx: With `topk`, the group of each routed slot for gather.
    scatter_idx: With `topk`, the group of each routed slot for scatter.
    topk: Groups per token. If given, the routing is built in the kernel from
      the raw top-k ids (`LowTokMetadataRef`).
  """
  if topk is not None:
    return LowTokMetadataRef.create(
        gather_idx, scatter_idx, group_offset, cfg, topk
    )
  if group_cum_sum is None:
    raise ValueError("group_cum_sum is required when topk is None.")
  if row_group is not None and cfg.plan.live_group_walk:
    return LiveGroupMetadataRef.create(
        group_cum_sum, group_offset, cfg, row_group=row_group
    )
  return SortedMetadataRef.create(group_cum_sum, group_offset, cfg)
