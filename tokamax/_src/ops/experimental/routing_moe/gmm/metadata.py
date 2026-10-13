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

import dataclasses
import functools

import jax
from jax.experimental import hijax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.gmm import config


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  """Stores scalar data used for routing and pipeline scheudling."""

  # NOTE: because gm_id is offsetted by num_buffers, it is not recommended
  # to directly access these values as it is error prone. Instead, it is
  # recommended to access them via get_* class methods.
  _gm_id_to_group_id: jax.Array | hijax.TransformedRef
  _gm_id_to_m_offset: jax.Array | hijax.TransformedRef
  _gm_id_to_num_rows: jax.Array | hijax.TransformedRef
  gather_idx: jax.Array | hijax.TransformedRef | None
  scatter_idx: jax.Array | hijax.TransformedRef | None
  group_cum_sum: jax.Array | hijax.TransformedRef
  group_offset: jax.Array | hijax.TransformedRef
  cfg: config.RoutingGmmConfig = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls,
      gather_idx: jax.Array | None,
      scatter_idx: jax.Array | None,
      group_cum_sum: jax.Array,
      group_offset: jax.Array,
      cfg: config.RoutingGmmConfig,
  ):
    """Performs simple preprocessing and wraps scalar data for metadata."""

    # Padd indices by tile_m to avoid any out of bound access at last steps.
    if gather_idx is not None:
      gather_idx = jnp.pad(gather_idx, (0, cfg.tile.tile_m))
    if scatter_idx is not None:
      scatter_idx = jnp.pad(scatter_idx, (0, cfg.tile.tile_m))

    return cls(
        _gm_id_to_group_id=jnp.empty((cfg.max_num_gm,), jnp.int32),
        _gm_id_to_m_offset=jnp.empty((cfg.max_num_gm,), jnp.int32),
        _gm_id_to_num_rows=jnp.empty((cfg.max_num_gm,), jnp.int32),
        gather_idx=gather_idx,
        scatter_idx=scatter_idx,
        group_cum_sum=group_cum_sum,
        group_offset=group_offset,
        cfg=cfg,
    )

  def _real_gm_id(self, gm_id: jax.Array | int) -> jax.Array | int:
    return gm_id + self.cfg.tile.num_max_buffers

  def get_num_rows(self, gm_id: jax.Array | int) -> jax.Array:
    return self._gm_id_to_num_rows[self._real_gm_id(gm_id)]

  def get_m_offset(self, gm_id: jax.Array | int) -> jax.Array:
    return self._gm_id_to_m_offset[self._real_gm_id(gm_id)]

  def get_group_id(self, gm_id: jax.Array | int) -> jax.Array:
    return self._gm_id_to_group_id[self._real_gm_id(gm_id)]

  def fill_metadata_and_return_num_gm(self) -> jax.Array:
    """Iterate group sizes to metadata about DMA and pipeline schedule logic."""

    # NOTE: Unlike GMM v2, there is no need for complex sublane alignment logic
    # as issuing one dma per row avoids any shared chunk or non-zero starting
    # offset issue that was found in GMM v2.

    group_start = self.group_offset[0]
    group_end = group_start + self.cfg.dim.size_local_groups
    num_buffers = self.cfg.tile.num_max_buffers

    # To avoid any runtime out-of-the-bounds check overhead, we intentially
    # start gm_id from num_buffers so that trying to DMA wait gm_id -
    # num_buffers does not trigger stall.
    for b in range(num_buffers):
      self._gm_id_to_group_id[b] = 0
      self._gm_id_to_m_offset[b] = 0
      self._gm_id_to_num_rows[b] = 0

    @jax.named_scope("write_tile")
    def write_tile(gm_id, offset, *, local_group_id, end):
      num_rows = jnp.minimum(end - offset, self.cfg.tile.tile_m)
      self._gm_id_to_group_id[gm_id] = local_group_id
      self._gm_id_to_m_offset[gm_id] = offset
      self._gm_id_to_num_rows[gm_id] = num_rows
      return offset + num_rows

    @jax.named_scope("write_group")
    def write_group(group_id, num_gm):
      start = self.group_cum_sum[group_id]
      end = self.group_cum_sum[group_id + 1]
      next_num_gm = num_gm + pl.cdiv(end - start, self.cfg.tile.tile_m)
      local_group_id = group_id - group_start

      jax.lax.fori_loop(
          num_gm,
          next_num_gm,
          functools.partial(write_tile, local_group_id=local_group_id, end=end),
          start,
      )
      return next_num_gm

    num_gm = jax.lax.fori_loop(group_start, group_end, write_group, num_buffers)

    # Similar to above, we zero out up-to last gm + num_buffers to ensure DMA
    # copy for gm_id + num_buffers does not tirgger out-of-the-bound errors.
    for b in range(num_buffers):
      self._gm_id_to_group_id[num_gm + b] = 0
      self._gm_id_to_m_offset[num_gm + b] = 0
      self._gm_id_to_num_rows[num_gm + b] = 0

    # Return real number of buffers minus left and right padding.
    return num_gm - num_buffers
