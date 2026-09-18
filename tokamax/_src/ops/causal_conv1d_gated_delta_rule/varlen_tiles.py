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
"""Tiling plans for Pallas kernels that grid over (sequence, tile).

A ragged batch can be walked two ways. Flattening it onto one token axis makes
``BlockSpec`` indexing work -- a block index is a multiple of the block shape --
but only if every sequence starts on a block boundary, which costs a gather that
pads each sequence up to that boundary. Gridding over (sequence, tile) instead
lets a tile address arbitrary rows of the *unaligned* activations, so no padding
gather is needed; the kernel fetches ``r_size`` rows from ``r_base`` itself.

This module computes the second form. It is the arithmetic only: no VMEM
staging, no kernel, no config object -- so both the GDN and Kimi KDA kernels can
share it. Extracted from ``gdn/v3/metadata.py``, which now delegates here.

The key property for callers is that ``num_tiles`` is a *device* scalar that
tracks the batch's real content, while the grid stays static at ``max_tiles``.
A kernel that no-ops beyond ``num_tiles`` therefore costs nothing for sequences
that are absent -- which is what makes it affordable to hand a kernel a segment
list that is empty, the way GDN hands its prefill pass one on a decode step.
"""

import dataclasses

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class VarlenTilePlan:
  """Per-tile and per-sequence indices for a (sequence, tile) grid.

  ``p_id`` is the tile index, i.e. the grid position along the tile axis.
  Entries at ``p_id >= num_tiles`` are padding: `jnp.repeat` fills them by
  extending the last real value, so they are safe to read but meaningless.
  A kernel must bound itself by ``num_tiles`` rather than trusting them.

  Attributes:
      num_tiles: device scalar; how many tiles hold real tokens.
      p_id_to_s_idx: which sequence each tile belongs to.
      p_id_to_r_base: first row of the tile in the unaligned token axis.
      p_id_to_r_size: rows the tile holds, ``<= tile_size``. A sequence whose
        length is not a multiple of ``chunk_size`` has a short last tile.
      p_id_is_first_tile / p_id_is_last_tile: where a sequence's recurrent state
        has to be read in and written back.
      s_idx_has_initial_state: per sequence, whether it carries state in -- true
        iff it has previously computed context.
  """

  num_tiles: jax.Array
  p_id_to_s_idx: jax.Array
  p_id_to_r_base: jax.Array
  p_id_to_r_size: jax.Array
  p_id_is_first_tile: jax.Array
  p_id_is_last_tile: jax.Array
  s_idx_has_initial_state: jax.Array


def roll_to_start_seq(x: jax.Array, start_seq: jax.Array | int) -> jax.Array:
  """Rotate a per-sequence array so ``start_seq`` lands at index 0.

  :func:`plan_per_seq_tiles` does this to the arrays it reads. Callers with
  their own per-sequence payloads -- state slot ids, read offsets -- have to
  rotate them the same way to stay aligned with ``p_id_to_s_idx``.
  """
  return jnp.roll(x, shift=-start_seq)


def plan_per_seq_tiles(
    seq_lens: jax.Array,  # [max_seqs]
    query_start_loc: jax.Array,  # [max_seqs + 1]
    *,
    max_tiles: int,
    chunk_size: int,
    tile_size: int,
    start_seq: jax.Array | int = 0,
    end_seq: jax.Array | int | None = None,
) -> VarlenTilePlan:
  """Plan one tile per ``chunk_size`` tokens, per sequence.

  Args:
      seq_lens: total length of each sequence, context included. Only ``seq_lens
        - query_lens > 0`` is read from it, to decide carry-in.
      query_start_loc: cu_seqlens over this step's token axis.
      max_tiles: static tile-axis size, and so the grid extent. Must be at least
        the largest tile count any batch can produce; the token bucket works,
        since a tile holds at least one token.
      chunk_size: tokens per tile, as the kernel's math requires.
      tile_size: cap on ``p_id_to_r_size``. Equal to ``chunk_size`` unless a
        kernel fetches in a different granularity than it computes.
      start_seq, end_seq: the half-open sequence range to plan for, as device
        scalars. Sequences outside it get zero tiles and so cost nothing -- this
        is how a caller restricts one kernel to the prefill segment of a batch
        and another to the decode segment, without a host branch. ``end_seq``
        defaults to every sequence.

  Returns:
      A :class:`VarlenTilePlan`.
  """
  max_seqs = seq_lens.size
  if end_seq is None:
    end_seq = max_seqs
  all_seqs = jnp.arange(max_seqs)
  all_tiles = jnp.arange(max_tiles)

  # Shift so the first element is for start_seq.
  query_start_loc = roll_to_start_seq(query_start_loc, start_seq)
  seq_lens = roll_to_start_seq(seq_lens, start_seq)

  query_lens = query_start_loc[1:] - query_start_loc[:-1]
  # Only query_lens has to be masked: it decides the tile count, and nothing
  # else is read for a sequence that contributes no tiles.
  num_seqs = end_seq - start_seq
  query_lens = jnp.where(all_seqs < num_seqs, query_lens, 0)

  s_idx_to_num_tiles = pl.cdiv(query_lens, chunk_size)
  s_idx_to_start_p_id = jnp.cumulative_sum(
      s_idx_to_num_tiles, include_initial=True
  )
  # Tile -> sequence. With s_idx_to_num_tiles = [1 2 3 0 1] this gives
  # [0 1 1 2 2 2 4]: sequence 3 contributes nothing and is skipped.
  # `total_repeat_length` is what makes the repeat jittable; it pads the tail
  # past num_tiles, which is why callers must bound themselves by num_tiles.
  p_id_to_s_idx = jnp.repeat(
      all_seqs, s_idx_to_num_tiles, total_repeat_length=max_tiles
  )
  p_id_to_t_id = all_tiles - s_idx_to_start_p_id[p_id_to_s_idx]
  p_id_to_r_base = (
      query_start_loc[p_id_to_s_idx] + p_id_to_t_id * chunk_size
  )
  p_id_to_r_size = jnp.minimum(
      query_start_loc[p_id_to_s_idx + 1] - p_id_to_r_base,
      tile_size,
  )

  # A sequence whose query is shorter than its total length has context
  # behind it, so its first tile reads state in; its last tile writes back.
  has_initial_state = (seq_lens - query_lens) > 0
  p_id_is_first_tile = p_id_to_t_id == 0
  p_id_is_last_tile = p_id_to_t_id == (s_idx_to_num_tiles[p_id_to_s_idx] - 1)

  # Sequences outside [start_seq, end_seq) have query_lens 0 and so contribute
  # zero tiles, which is what makes this sum the real tile count.
  return VarlenTilePlan(
      num_tiles=s_idx_to_num_tiles.sum(),
      p_id_to_s_idx=p_id_to_s_idx,
      p_id_to_r_base=p_id_to_r_base,
      p_id_to_r_size=p_id_to_r_size,
      p_id_is_first_tile=p_id_is_first_tile,
      p_id_is_last_tile=p_id_is_last_tile,
      s_idx_has_initial_state=has_initial_state,
  )
