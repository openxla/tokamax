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
from jax import lax
from jax.experimental import hijax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.gmm import config


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  """Stores scalar data used for routing and pipeline scheudling."""

  gm_id_to_group_id: jax.Array | hijax.TransformedRef
  gm_id_to_m_offset: jax.Array | hijax.TransformedRef
  gm_id_to_num_rows: jax.Array | hijax.TransformedRef
  num_gm: jax.Array | hijax.TransformedRef
  group_cum_sum: jax.Array | hijax.TransformedRef
  group_offset: jax.Array | hijax.TransformedRef
  cfg: config.RoutingGmmConfig = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls,
      group_cum_sum: jax.Array,
      group_offset: jax.Array,
      cfg: config.RoutingGmmConfig,
  ):
    """Wraps scalar data for metadata."""
    return cls(
        gm_id_to_group_id=jnp.empty((cfg.max_num_gm,), jnp.int32),
        gm_id_to_m_offset=jnp.empty((cfg.max_num_gm,), jnp.int32),
        gm_id_to_num_rows=jnp.empty((cfg.max_num_gm,), jnp.int32),
        num_gm=jnp.empty((1,), jnp.int32),
        group_cum_sum=group_cum_sum,
        group_offset=group_offset,
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
      num_gm = jax.lax.fori_loop(group_start, group_end, write_group, 0)
    else:
      _, num_gm = jax.lax.while_loop(
          lambda carry: carry[0] < group_end,
          step,
          (group_start, jnp.int32(0)),
      )
    self.num_gm[0] = num_gm
    return num_gm
