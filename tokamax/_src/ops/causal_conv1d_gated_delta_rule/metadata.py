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

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import memory_ref
from tokamax._src.ops.causal_conv1d_gated_delta_rule import varlen_tiles


def compute_batched_seq_metadata(
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    read_offsets: jax.Array,
    end_seq: jax.Array,
    ckpt_indices: jax.Array | None = None,
) -> memory_ref.MetadataRef:
  """Metadata for computing multiple sequences per tile.

  A sequence contributes exactly one tile holding all of its query tokens:
  a single decoded token, or a speculative verify window of up to
  `cfg.window_size` of them. The initial state is read from
  `state_indices[s] + read_offsets[s]` and one state checkpoint per window
  position is written back to `state_indices[s] + t`.
  """

  max_seqs = seq_lens.size
  all_seqs = jnp.arange(max_seqs)

  # NOTE: Only supports use case where query_lens[i] <= cfg.window_size where
  # i < end_seq. This must be guaranteed by the function caller.
  # TODO: Add error handling when above condition is not met.
  query_lens = query_start_loc[1:] - query_start_loc[:-1]
  is_valid_seqs = jnp.where(all_seqs < end_seq, True, False)
  has_initial_state = (seq_lens - query_lens) > 0
  all_valid_seqs = jnp.where(is_valid_seqs, all_seqs, 0)

  return memory_ref.MetadataRef.create(
      cfgs=cfg,
      num_tiles=pl.cdiv(end_seq, cfg.tile_size),
      p_id_to_s_idx=all_valid_seqs,
      p_id_to_r_base=query_start_loc[all_valid_seqs],
      p_id_to_r_size=jnp.where(is_valid_seqs, query_lens, 0),
      p_id_is_first_tile=is_valid_seqs,
      p_id_is_last_tile=is_valid_seqs,
      s_idx_has_initial_state=has_initial_state,
      s_idx_to_state_indices=state_indices,
      s_idx_to_read_offset=read_offsets,
      s_idx_to_ckpt_indices=ckpt_indices,
  )


def compute_per_seq_metadata(
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    read_offsets: jax.Array,
    start_seq: jax.Array,
    end_seq: jax.Array,
    ckpt_indices: jax.Array | None = None,
) -> memory_ref.MetadataRef:
  """Metadata for computing single sequence per tile.

  The tiling itself lives in `varlen_tiles.py` so the Kimi KDA kernels
  can share it; this wraps it in GDN's SMEM-resident MetadataRef and adds the
  per-sequence state addressing, which is GDN's own.

  `read_offsets` selects the checkpoint a resuming sequence's initial state
  is read from. It is 0 in the common cases (a fresh or chunked prefill
  resumes from the single state its previous chunk wrote), but a
  prefix-cache resume on the unified pool may land on a boundary state
  block whose committed state is a non-zero checkpoint from a verify step.
  The final state is always written to checkpoint 0.
  """
  plan = varlen_tiles.plan_per_seq_tiles(
      seq_lens,
      query_start_loc,
      max_tiles=cfg.batch_size,
      chunk_size=cfg.chunk_size,
      tile_size=cfg.tile_size,
      start_seq=start_seq,
      end_seq=end_seq,
  )
  # Rotated the same way the plan rotates what it reads, so the per-sequence
  # payloads stay aligned with `p_id_to_s_idx`.
  state_indices = varlen_tiles.roll_to_start_seq(state_indices, start_seq)
  read_offsets = varlen_tiles.roll_to_start_seq(read_offsets, start_seq)
  if ckpt_indices is not None:
    # Sequence-major axis only: the checkpoint axis must stay put or the
    # read offset would select another slot's block.
    ckpt_indices = jnp.roll(ckpt_indices, shift=-start_seq, axis=0)

  return memory_ref.MetadataRef.create(
      cfgs=cfg,
      num_tiles=plan.num_tiles,
      p_id_to_s_idx=plan.p_id_to_s_idx,
      p_id_to_r_base=plan.p_id_to_r_base,
      p_id_to_r_size=plan.p_id_to_r_size,
      p_id_is_first_tile=plan.p_id_is_first_tile,
      p_id_is_last_tile=plan.p_id_is_last_tile,
      s_idx_has_initial_state=plan.s_idx_has_initial_state,
      s_idx_to_state_indices=state_indices,
      s_idx_to_read_offset=read_offsets,
      s_idx_to_ckpt_indices=ckpt_indices,
  )
