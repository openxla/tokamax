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
from typing import Any, Self

import jax
from jax import lax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.gmm import config


def _searchsorted_left(
    sorted_ref: hijax.TransformedRef, size: int, target: jax.Array
) -> jax.Array:
  """Returns the first index in `[0, size]` with `sorted_ref[i] >= target`."""
  assert size > 0
  # Unrolled scalar SMEM bit-stepping avoids loading `sorted_ref` into VREGs
  # (as `jnp.searchsorted` would) or emitting a scalar `while_loop`.
  lo = jnp.int32(0)
  for bit in range(size.bit_length() - 1, -1, -1):
    probe = lo + (1 << bit)
    idx = jnp.minimum(probe, size) - 1
    lo = jnp.where(
        jnp.logical_and(probe <= size, sorted_ref[idx] < target),
        probe,
        lo,
    )
  return lo


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataInputs[T]:
  """Inputs required to calculate tile metadata."""

  group_sizes: T
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
  def create_scratch_types(cls, cfg: config.RoutingGmmConfig) -> Self:
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
      self,
      n_id: jax.Array | int,
      gm_id: jax.Array | int,
      k2_id: jax.Array | int,
      k1_id: jax.Array | int,
  ) -> jax.Array:
    """Convert (n, gm, k2, k1) grid into a linear grid over (n, gm)."""
    # TODO: Fold k into the step once num_k > 1 is supported.
    del k2_id, k1_id
    return n_id * self.num_gm[0] + gm_id

  def slot(
      self,
      n_id: jax.Array | int,
      gm_id: jax.Array | int,
      k2_id: jax.Array | int,
      k1_id: jax.Array | int,
      buffer_count: int,
  ) -> jax.Array:
    """Returns the buffer slot index for the given grid coordinates."""
    # Because gm_id is used for lookahead/lookback, the smallest possible
    # value is -buffer_count. To protect against uint underflow, we add
    # buffer_count first to ensure slot indexing is always correct.
    step = self.step(n_id, gm_id + buffer_count, k2_id, k1_id)
    return lax.rem(step.astype(jnp.uint32), jnp.uint32(buffer_count))

  def resolve(
      self,
      n_id: jax.Array | int,
      gm_id: jax.Array | int,
      k2_id: jax.Array | int,
      k1_id: jax.Array | int,
      dtype: jax.typing.DTypeLike = jnp.int32,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Maps a possibly out-of-range (n, gm, k2, k1) to in-range or the sink."""
    step = self.step(n_id, gm_id, k2_id, k1_id)
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
        jnp.asarray(k2_id, dtype),
        jnp.asarray(k1_id, dtype),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SortedMetadataRef(AbstractMetadataRef):
  """Metadata from routing sorted outside the kernel.

  XLA sorts the routed rows by group and counts the rows of each group
  (`group_sizes`); the kernel walks this shard's groups to lay out the tiles.
  """

  @jax.named_scope("write_tile")
  def _write_tile(
      self,
      gm_id: jax.Array,
      offset: jax.Array,
      *,
      local_group_id: jax.Array,
      end: jax.Array,
  ) -> jax.Array:
    """Writes metadata for one tile and returns the next row offset."""
    num_rows = jnp.minimum(end - offset, self.cfg.plan.tile_m)
    self.gm_id_to_group_id[gm_id] = local_group_id
    self.gm_id_to_m_offset[gm_id] = offset
    self.gm_id_to_num_rows[gm_id] = num_rows
    return offset + num_rows

  @jax.named_scope("write_group")
  def _write_group(
      self,
      group_id: jax.Array,
      start: jax.Array,
      count: jax.Array,
      num_gm: jax.Array,
      *,
      group_start: jax.Array,
  ) -> jax.Array:
    """Writes all tiles of one group and returns the next tile index."""
    end = start + count
    next_num_gm = num_gm + pl.cdiv(count, self.cfg.plan.tile_m)
    local_group_id = group_id - group_start
    jax.lax.fori_loop(
        num_gm,
        next_num_gm,
        functools.partial(
            self._write_tile, local_group_id=local_group_id, end=end
        ),
        start,
    )
    return next_num_gm

  def fill_metadata_and_return_num_gm(
      self, metadata_in: MetadataInputs[hijax.TransformedRef]
  ) -> jax.Array:
    """Iterate group sizes to metadata about DMA and pipeline schedule logic."""

    # NOTE: Unlike GMM v2, there is no need for complex sublane alignment logic
    # as issuing one dma per row avoids any shared chunk or non-zero starting
    # offset issue that was found in GMM v2.

    group_start = metadata_in.group_offset[0]
    group_end = group_start + self.cfg.spec.size_local_groups
    self.gm_id_to_group_id[self.sink_gm_id] = 0
    self.gm_id_to_m_offset[self.sink_gm_id] = 0
    self.gm_id_to_num_rows[self.sink_gm_id] = 0

    num_gm = self._walk_groups(metadata_in, group_start, group_end)
    self.num_gm[0] = num_gm
    return num_gm

  def _walk_groups(
      self,
      metadata_in: MetadataInputs[hijax.TransformedRef],
      group_start: jax.Array,
      group_end: jax.Array,
  ) -> jax.Array:
    """Walks every local group in order; empty groups write no tiles."""
    # Used when most groups have rows or `row_group` is not given.
    init_row = jnp.int32(0)
    if self.cfg.spec.size_local_groups < self.cfg.spec.num_global_groups:
      init_row = jax.lax.fori_loop(
          0,
          group_start,
          lambda g, acc: acc + metadata_in.group_sizes[g],
          init_row,
      )

    def step_fn(
        group_id: jax.Array, state: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
      row, num_gm = state
      count = metadata_in.group_sizes[group_id]
      next_num_gm = self._write_group(
          group_id, row, count, num_gm, group_start=group_start
      )
      return row + count, next_num_gm

    _, num_gm = jax.lax.fori_loop(
        group_start, group_end, step_fn, (init_row, jnp.int32(0))
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
      group_start: jax.Array,
      group_end: jax.Array,
  ) -> jax.Array:
    """Walks only the local groups that have at least one routed row."""
    row_group = metadata_in.row_group
    size_out_m = self.cfg.spec.size_out_m
    assert row_group is not None and size_out_m > 0

    def group_at(row_idx: jax.Array) -> jax.Array:
      return row_group[jnp.minimum(row_idx, size_out_m - 1)]

    row = jnp.int32(0)
    if self.cfg.spec.size_local_groups < self.cfg.spec.num_global_groups:
      row = _searchsorted_left(row_group, size_out_m, group_start)

    def cond_fn(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
      row_idx, group_id, _ = state
      return jnp.logical_and(row_idx < size_out_m, group_id < group_end)

    def body_fn(
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      row_idx, group_id, num_gm = state
      count = metadata_in.group_sizes[group_id]
      next_num_gm = self._write_group(
          group_id, row_idx, count, num_gm, group_start=group_start
      )
      next_row = row_idx + count
      return (next_row, group_at(next_row), next_num_gm)

    _, _, num_gm = jax.lax.while_loop(
        cond_fn, body_fn, (row, group_at(row), jnp.int32(0))
    )
    return num_gm


# Not a fan of return type Any but despite metadataref being flattened using
# register_dataclass, linter complains mismatch with scratch_typess :(
def create_metadata(
    group_sizes: jax.Array,
    group_offset: jax.Array,
    cfg: config.RoutingGmmConfig,
    *,
    row_group: jax.Array | None = None,
) -> tuple[MetadataInputs[jax.Array], Any]:
  """Returns the SMEM metadata inputs and scratch specs for this config."""
  use_live_walk = (
      row_group is not None
      and cfg.plan.live_group_walk
      and cfg.spec.size_out_m > 0
  )
  metadata_in = MetadataInputs(
      group_sizes=group_sizes,
      group_offset=group_offset,
      row_group=row_group if use_live_walk else None,
  )
  metadata_cls = LiveGroupMetadataRef if use_live_walk else SortedMetadataRef
  return metadata_in, metadata_cls.create_scratch_types(cfg)
