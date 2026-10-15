# Copyright 2026 Google LLC. All Rights Reserved.
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
"""Calculates batch_tile_idx, bq_idx, and bkv_idx metadata

for 1D pipeline step schedule.
"""

import dataclasses
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# --------------------------------------------------------------------------- #
# Context-parallel helpers.
# --------------------------------------------------------------------------- #


def cp_local_to_global(local_idx, cp_rank, cp_size: int, interleave_c: int):
  """Map a rank-local compressed index to its global compressed position."""
  if cp_size == 1:
    return local_idx
  cycle_c = cp_size * interleave_c
  return (
      (local_idx // interleave_c) * cycle_c
      + cp_rank * interleave_c
      + local_idx % interleave_c
  )


def cp_local_length(global_len, cp_rank, cp_size: int, interleave_c: int):
  """Count compressed positions ``< global_len`` owned by `cp_rank`."""
  if cp_size == 1:
    return global_len
  cycle_c = cp_size * interleave_c
  full = global_len // cycle_c
  rem = global_len - full * cycle_c
  return full * interleave_c + jnp.clip(
      rem - cp_rank * interleave_c, 0, interleave_c
  )


def cp_owner_rank(global_idx, cp_size: int, interleave_c: int):
  """Which rank owns global compressed position `global_idx`."""
  if cp_size == 1:
    return jnp.zeros_like(global_idx)
  return (global_idx // interleave_c) % cp_size


def cp_global_to_local(global_idx, cp_size: int, interleave_c: int):
  """Inverse of `cp_local_to_global`, evaluated on the owning rank."""
  if cp_size == 1:
    return global_idx
  cycle_c = cp_size * interleave_c
  return (global_idx // cycle_c) * interleave_c + global_idx % interleave_c


def cp_rank_as_data(cp_axis_name: str, cp_size: int):
  """This shard's index along `cp_axis_name`, without `lax.axis_index`."""
  ranks = jnp.arange(cp_size, dtype=jnp.int32)
  exchanged = lax.all_to_all(ranks, cp_axis_name, 0, 0, tiled=True)
  return lax.dynamic_slice_in_dim(exchanged, 0, 1, axis=0)[0]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  """Ref containers for pipeline step schedule indexing arrays."""

  num_steps: Any
  batch_tile_idx: Any
  bq_idx: Any
  bkv_idx: Any

  @classmethod
  def create(
      cls,
      num_steps: Any,
      batch_tile_idx: Any,
      bq_idx: Any,
      bkv_idx: Any,
  ):
    return cls(
        num_steps=num_steps,
        batch_tile_idx=batch_tile_idx,
        bq_idx=bq_idx,
        bkv_idx=bkv_idx,
    )

  @property
  def start_seq_idx(self) -> Any:
    """Returns starting sequence index per step (alias for batch_tile_idx)."""
    return self.batch_tile_idx


def next_power_of_2(x):
  assert x > 0
  return 1 << (x - 1).bit_length()


def generate_max_steps(
    num_seqs: int,
    max_pages_per_seq: int,
) -> int:
  """Calculates upper bound on max_steps based on TPU SMEM capacity."""
  fixed_bytes = (
      num_seqs  # seq_lens (int32 per seq)
      + (num_seqs + 1)  # cu_q_lens (int32 prefix sum array)
      + (
          num_seqs * max_pages_per_seq
      )  # page_indices block table (int32 per page)
      + 11  # num_steps and other scalar inputs
  ) * 4  # 4 bytes per int32 element

  tpu_info = pltpu.get_tpu_info()
  smem_limit_bytes = tpu_info.smem_capacity_bytes - (32 * 1024)
  smem_limit_bytes //= 2  # Use half of the SMEM for metadata.
  available_bytes = max(0, smem_limit_bytes - fixed_bytes)

  # MetadataRef contains 3 int32 arrays (batch_tile_idx, bq_idx, bkv_idx)
  # = 12 bytes per step
  bytes_per_step = 12
  max_steps_ub = available_bytes // bytes_per_step

  num_lanes = tpu_info.num_lanes
  max_steps_ub = max(1, max_steps_ub // num_lanes) * num_lanes
  return max_steps_ub


def compute_batched_seq_metadata(
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    start_seq_idx: jax.Array | int,
    end_seq_idx: jax.Array | int,
    bq_sz: int,
    bkv_sz: int,
    pages_per_seq: int,
    page_size: int,
    max_num_tokens: int,
    compression_ratio: int = 1,
    static_q_len: int | None = None,
    seq_batch_size: int = 1,
    chunk_token_start: int | tuple[int | None, ...] | None = None,
    chunk_tokens: int | None = None,
    cp_rank: jax.Array | int = 0,
    cp_size: int = 1,
    interleave_c: int = 1,
) -> MetadataRef:
  max_num_seqs = seq_lens.shape[0]
  max_seq_tiles = pl.cdiv(max_num_seqs, seq_batch_size)

  chunk_starts = (
      chunk_token_start
      if isinstance(chunk_token_start, tuple)
      else (chunk_token_start,)
  )
  num_chunks = len(chunk_starts)
  chunked = chunk_tokens is not None and chunk_starts[0] is not None

  if static_q_len == 1:
    total_max_bq = max_seq_tiles
  elif static_q_len is not None:
    total_max_bq = max_seq_tiles * max(1, pl.cdiv(static_q_len, bq_sz))
  else:
    # Here, cdiv(max_q_len_ub, bq_sz) covers the total token volume, while
    # + max_seq_tiles accounts for up to one partially filled tail block per
    # sequence tile.
    max_q_len_ub = next_power_of_2(max_num_tokens)
    total_max_bq = max(1, pl.cdiv(max_q_len_ub, bq_sz)) + max_seq_tiles

  max_kv_pages_per_block = (
      max(1, bkv_sz // page_size) if bkv_sz >= page_size else 1
  )
  max_bkv = max(1, pl.cdiv(pages_per_seq, max_kv_pages_per_block))

  if chunked:
    # A chunk can cut across a sequence tile, in which case we compute the
    # block holding the seam twice: once for the tail of the first sequence,
    # once for the head of the next. That is one extra block per tile the
    # chunk touches, and it touches at most max_seq_tiles tiles, or at most
    # chunk_tokens of them since a touched tile owns at least one token.
    per_chunk_bq = min(
        total_max_bq,
        pl.cdiv(chunk_tokens, seq_batch_size * bq_sz)
        + min(max_seq_tiles, chunk_tokens),
    )
    req_max_steps = num_chunks * per_chunk_bq * max_bkv
  else:
    req_max_steps = total_max_bq * max_bkv
  req_max_steps = pl.cdiv(req_max_steps, 1024) * 1024

  max_steps = max(1024, req_max_steps)

  is_empty_pass = jnp.greater_equal(start_seq_idx, end_seq_idx)

  def _empty_schedule():
    return MetadataRef.create(
        num_steps=jnp.zeros((num_chunks,), dtype=jnp.int32),
        batch_tile_idx=jnp.zeros((max_steps,), dtype=jnp.int32),
        bq_idx=jnp.zeros((max_steps,), dtype=jnp.int32),
        bkv_idx=jnp.zeros((max_steps,), dtype=jnp.int32),
    )

  # Static shape bucketing for JAX XLA tracing on TPU:
  # Compute max_seq_tiles and padded arrays over max_num_seqs to maintain static
  # tensor shapes across kernel invocations. This prevents XLA recompilation per
  # request.
  # Sequences outside [start_seq_idx, end_seq_idx) are zero-masked upfront via
  # valid_seq_lens, while the Pallas attention scoring kernel strictly
  # executes num_steps active pipeline steps on the MXU.
  def _compute_schedule():
    all_steps = jnp.arange(max_steps, dtype=jnp.int32)
    all_seq_tiles = jnp.arange(max_seq_tiles, dtype=jnp.int32)
    padded_len = max_seq_tiles * seq_batch_size
    pad_size = padded_len - max_num_seqs
    padded_seq_lens = jnp.pad(seq_lens, (0, pad_size))
    # A sequence tile is a group of seq_batch_size sequences processed together
    # (e.g. 4 sequences per tile in decode mode, or 1 sequence per tile in
    # prefill mode).
    starting_seq_tile_idx = all_seq_tiles * seq_batch_size
    is_valid_seq_tile = (starting_seq_tile_idx >= start_seq_idx) & (
        starting_seq_tile_idx < end_seq_idx
    )

    seq_indices = jnp.arange(padded_len, dtype=jnp.int32)
    is_valid_seq = (seq_indices >= start_seq_idx) & (seq_indices < end_seq_idx)
    valid_seq_lens = jnp.where(is_valid_seq, padded_seq_lens, 0)

    kv_lens_all = (valid_seq_lens // compression_ratio).reshape(
        max_seq_tiles, seq_batch_size
    )
    if cp_size > 1:
      local_kv_lens_all = cp_local_length(
          kv_lens_all, cp_rank, cp_size, interleave_c
      )
    else:
      local_kv_lens_all = kv_lens_all
    max_kv_lens = jnp.max(local_kv_lens_all, axis=-1)
    # 1D array of shape (max_seq_tiles,)
    s_num_bkv = jnp.where(
        is_valid_seq_tile,
        jnp.maximum(1, pl.cdiv(max_kv_lens, bkv_sz)),
        0,
    ).astype(jnp.int32)

    # 1D array of shape (max_seq_tiles,)
    if static_q_len is not None:
      s_num_bq = jnp.full(
          (max_seq_tiles,),
          jnp.maximum(1, pl.cdiv(static_q_len, bq_sz)),
          dtype=jnp.int32,
      )
    else:
      q_lens_all = (cu_q_lens[1:] - cu_q_lens[:-1]).reshape(
          max_seq_tiles, seq_batch_size
      )
      max_q_lens = jnp.max(q_lens_all, axis=-1)
      s_num_bq = jnp.maximum(1, pl.cdiv(max_q_lens, bq_sz)).astype(jnp.int32)

    if chunked:
      bq_span = seq_batch_size * bq_sz
      tile_first_seq = jnp.minimum(starting_seq_tile_idx, max_num_seqs)
      tile_last_seq = jnp.minimum(
          starting_seq_tile_idx + seq_batch_size, max_num_seqs
      )
      tile_tok_start = cu_q_lens[tile_first_seq][None, :]
      tile_tok_end = cu_q_lens[tile_last_seq][None, :]
      # A tile keeps only the tokens its chunk owns, [lo, hi), and lays its
      # blocks out from lo.
      chunk_start = jnp.asarray(chunk_starts, dtype=jnp.int32)[:, None]
      lo = jnp.clip(chunk_start, tile_tok_start, tile_tok_end)
      hi = jnp.clip(chunk_start + chunk_tokens, tile_tok_start, tile_tok_end)
      num_bq = pl.cdiv(hi - lo, bq_span).astype(jnp.int32)
    else:
      num_bq = s_num_bq[None, :]

    # A segment is one (chunk, sequence tile) pair. Segments sit back to back
    # on one flat axis, so chunk m owns the max_seq_tiles segments starting at
    # m * max_seq_tiles, and everything below reads per segment exactly as it
    # used to read per sequence tile.
    seg_num_bkv = jnp.tile(s_num_bkv, num_chunks)
    seg_seq_tiles = jnp.tile(all_seq_tiles, num_chunks)
    seg_num_steps = num_bq.reshape(-1) * seg_num_bkv
    seg_start_step_id = jnp.cumulative_sum(seg_num_steps, include_initial=True)

    # 2D boolean ownership mask [num_chunks * max_seq_tiles, max_steps] where
    # entry (j, t) is True if flat pipeline step t belongs to segment j.
    is_step_in_seg = (all_steps[None, :] >= seg_start_step_id[:-1, None]) & (
        all_steps[None, :] < seg_start_step_id[1:, None]
    )

    # Map each flat pipeline step t to its sequence tile index (0, 1, ...) and
    # convert it to the starting sequence index in the batch (batch_tile_idx).
    repeat_seq_tiles = jnp.sum(seg_seq_tiles[:, None] * is_step_in_seg, axis=0)
    batch_tile_idx = repeat_seq_tiles * seq_batch_size
    # Holds the starting sequence index per flat pipeline step.
    start_step_id_repeat = jnp.sum(
        seg_start_step_id[:-1, None] * is_step_in_seg, axis=0
    )
    # Holds the local step index within the sequence tile.
    tile_step_id = all_steps - start_step_id_repeat
    # Holds the number of KV blocks processed per pipeline step.
    # Convert local 1D step index (tile_step_id) into 2D grid coordinates [row,
    # col] where width is nbkv_per_step_id:
    # row = tile_step_id // width, col = tile_step_id % width.
    # jnp.maximum(1, ...) prevents division by zero on inactive steps.
    nbkv_per_step_id = jnp.sum(seg_num_bkv[:, None] * is_step_in_seg, axis=0)

    bq_idx = jnp.maximum(0, tile_step_id // jnp.maximum(1, nbkv_per_step_id))
    bkv_idx = jnp.maximum(0, tile_step_id % jnp.maximum(1, nbkv_per_step_id))

    chunk_num_steps = jnp.sum(
        seg_num_steps.reshape(num_chunks, max_seq_tiles), axis=1
    )

    return MetadataRef.create(
        num_steps=chunk_num_steps,
        batch_tile_idx=batch_tile_idx,
        bq_idx=bq_idx,
        bkv_idx=bkv_idx,
    )

  return jax.lax.cond(is_empty_pass, _empty_schedule, _compute_schedule)


def compute_metadata(
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    start_seq_idx: jax.Array | int,
    end_seq_idx: jax.Array | int,
    bq_sz: int,
    bkv_sz: int,
    pages_per_seq: int,
    page_size: int,
    max_num_tokens: int,
    compression_ratio: int = 1,
    static_q_len: int | None = None,
    seq_batch_size: int = 1,
    chunk_token_start: int | tuple[int | None, ...] | None = None,
    chunk_tokens: int | None = None,
    cp_rank: jax.Array | int = 0,
    cp_size: int = 1,
    interleave_c: int = 1,
) -> MetadataRef:
  """Unified metadata calculation for batch_tile_idx, bq_idx, and bkv_idx.

  Passing a tuple of `chunk_token_start` values builds one schedule holding
  every chunk back to back, with `num_steps[m]` steps for chunk `m`.
  """
  return compute_batched_seq_metadata(
      seq_lens=seq_lens,
      cu_q_lens=cu_q_lens,
      start_seq_idx=start_seq_idx,
      end_seq_idx=end_seq_idx,
      bq_sz=bq_sz,
      bkv_sz=bkv_sz,
      pages_per_seq=pages_per_seq,
      page_size=page_size,
      max_num_tokens=max_num_tokens,
      compression_ratio=compression_ratio,
      static_q_len=static_q_len,
      seq_batch_size=seq_batch_size,
      chunk_token_start=chunk_token_start,
      chunk_tokens=chunk_tokens,
      cp_rank=cp_rank,
      cp_size=cp_size,
      interleave_c=interleave_c,
  )
