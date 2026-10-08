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
"""TPU-Friendly StreamIndex Top-K kernel."""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from tokamax._src.ops.experimental.lightning_indexer.kernel import (
    bref_override,
    config,
    dcp_sc_compact,
    metadata,
)
from tokamax._src.ops.experimental.tpu.topk import pallas_mosaic_tpu_kernel

BITS_NEG_INF = pallas_mosaic_tpu_kernel.BITS_NEG_INF
sparsecore_topk = pallas_mosaic_tpu_kernel.sparsecore_topk

MlaCase = config.MlaCase
KVLayout = config.KVLayout
DEFAULT_BUFFER_COUNT = config.DEFAULT_BUFFER_COUNT
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
DCP_AXIS_NAME = "dcp"

cp_local_to_global = metadata.cp_local_to_global
cp_local_length = metadata.cp_local_length
cp_owner_rank = metadata.cp_owner_rank
cp_global_to_local = metadata.cp_global_to_local
cp_rank_as_data = metadata.cp_rank_as_data


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def align_to(x, a):
  return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
  return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
  bits = get_dtype_bitwidth(dtype)
  return 32 // bits


def kernel(
    # Prefetch
    seq_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [3] (start_seq, end_seq, chunk_token_start)
    metadata_ref,  # MetadataRef
    cp_rank_ref,  # [1]
    # Input
    q_hbm_ref,  # Shape: [max_num_tokens, num_q_heads, head_dim], Memory: HBM
    indexer_weights_hbm_ref,  # Shape: [max_num_tokens, num_q_heads], Memory: HBM
    # HEAD_ALONG_SUBLANE:
    #     [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    # SEQ_ALONG_LANE: [total_num_pages, kv_sublane_groups, kv_packing, page_size]
    cache_kv_hbm_ref,
    # HBM scores, [chunk_tokens, num_sublanes_total, 128]: (out,), or
    # (donated_in, out) after the first pass. Only the last is ever read.
    *score_refs,
    compression_ratio: int,
    static_q_len: int | None,
    bkv_p: int,
    bq_sz: int,
    seq_batch_size: int,
    buffer_count: int = 3,
    case: MlaCase = MlaCase.MIXED,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    cp_size: int = 1,
    interleave_c: int = 1,
    chunk_idx: int = 0,
):
  """Executes the n-buffered Pallas scoring pipeline for StreamIndex Top-K."""
  del static_q_len, case
  scores_hbm_ref = score_refs[-1]
  buffer_cnt = buffer_count
  seq_along_lane = kv_layout == KVLayout.SEQ_ALONG_LANE

  _, num_q_heads, head_dim = q_hbm_ref.shape

  if seq_along_lane:
    # [total_num_pages, kv_sublane_groups, kv_packing, page_size] where the
    # leading `head_dim` sublane rows hold the FP8 key bytes and the next row
    # holds the per-token UE8M0 scale.
    _, kv_sublane_groups, kv_packing, page_size = cache_kv_hbm_ref.shape
    kv_head_dim_groups = head_dim // kv_packing
    page_size_per_kv_packing = page_size // kv_packing
  else:
    (
        _,
        page_size_per_kv_packing,
        kv_packing,
        _,
    ) = cache_kv_hbm_ref.shape
    page_size = page_size_per_kv_packing * kv_packing
    kv_sublane_groups = None
    kv_head_dim_groups = None

  max_num_seqs = seq_lens_ref.shape[0]
  num_page_indices = page_indices_ref.shape[0]

  pages_per_seq = num_page_indices // max_num_seqs

  bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
  bkv_sz = bkv_p * page_size
  num_sublanes_bkv = bkv_sz // 128
  shift_comp = (compression_ratio - 1).bit_length()

  # The schedule holds every chunk's steps back to back, so this chunk starts
  # where the chunks before it end and runs `num_steps[chunk_idx]` steps.
  step_offset = sum(metadata_ref.num_steps[m] for m in range(chunk_idx))

  def seq_tile(p_id):
    """Sequence tile scheduled at pipeline step `p_id`."""
    return metadata_ref.batch_tile_idx[step_offset + p_id]

  def bq_block(p_id):
    """Query block scheduled at pipeline step `p_id`."""
    return metadata_ref.bq_idx[step_offset + p_id]

  def bkv_block(p_id):
    """KV block scheduled at pipeline step `p_id`."""
    return metadata_ref.bkv_idx[step_offset + p_id]

  q_spec = pl.BlockSpec(
      (pl.BoundedSlice(seq_batch_size * bq_sz), num_q_heads, head_dim),
      lambda p_id: (seq_tile(p_id), bq_block(p_id), 0),
      pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
  )
  weights_spec = pl.BlockSpec(
      (pl.BoundedSlice(seq_batch_size * bq_sz), num_q_heads),
      lambda p_id: (seq_tile(p_id), bq_block(p_id), 0),
      pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
  )
  if seq_along_lane:
    kv_spec = pl.BlockSpec(
        (
            pl.BoundedSlice(seq_batch_size),
            kv_sublane_groups,
            kv_packing,
            bkv_sz,
        ),
        lambda p_id: (seq_tile(p_id), 0, 0, bkv_block(p_id)),
        pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
    )
  else:
    kv_spec = pl.BlockSpec(
        (
            pl.BoundedSlice(seq_batch_size),
            bkv_sz_per_kv_packing,
            kv_packing,
            cache_kv_hbm_ref.shape[-1],
        ),
        lambda p_id: (seq_tile(p_id), 0, bkv_block(p_id), 0),
        pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
    )
  out_spec = pl.BlockSpec(
      (pl.BoundedSlice(seq_batch_size * bq_sz), num_sublanes_bkv, 128),
      lambda p_id: (seq_tile(p_id), bq_block(p_id), bkv_block(p_id), 0),
      pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
  )

  q_alloc = bref_override.StreamIndexQBufferedRef.input(
      spec=q_spec,
      dtype_or_type=q_hbm_ref,
      buffer_count=buffer_cnt,
      use_lookahead=True,
      bq_sz=bq_sz,
      seq_batch_size=seq_batch_size,
      chunk_tokens=scores_hbm_ref.shape[0],
      chunk_idx=chunk_idx,
  )
  weights_alloc = bref_override.StreamIndexQBufferedRef.input(
      spec=weights_spec,
      dtype_or_type=indexer_weights_hbm_ref,
      buffer_count=buffer_cnt,
      use_lookahead=True,
      bq_sz=bq_sz,
      seq_batch_size=seq_batch_size,
      chunk_tokens=scores_hbm_ref.shape[0],
      chunk_idx=chunk_idx,
  )
  if seq_along_lane:
    kv_alloc = bref_override.StreamIndexKVSeqAlongLaneBufferedRef.input(
        spec=kv_spec,
        dtype_or_type=cache_kv_hbm_ref,
        buffer_count=buffer_cnt,
        use_lookahead=True,
        bkv_p=bkv_p,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        seq_batch_size=seq_batch_size,
        chunk_idx=chunk_idx,
    )
  else:
    kv_alloc = bref_override.StreamIndexKVBufferedRef.input(
        spec=kv_spec,
        dtype_or_type=cache_kv_hbm_ref,
        buffer_count=buffer_cnt,
        use_lookahead=True,
        bkv_p=bkv_p,
        page_size_per_kv_packing=page_size_per_kv_packing,
        kv_packing=kv_packing,
        pages_per_seq=pages_per_seq,
        seq_batch_size=seq_batch_size,
        chunk_idx=chunk_idx,
    )
  o_alloc = bref_override.StreamIndexOBufferedRef.output(
      spec=out_spec,
      dtype_or_type=scores_hbm_ref,
      buffer_count=2,
      use_lookahead=False,
      bq_sz=bq_sz,
      num_sublanes=num_sublanes_bkv,
      seq_batch_size=seq_batch_size,
      chunk_idx=chunk_idx,
  )

  def transpose_to_head_major(bq_rows, n_tok):
    """[n_tok*H, D] head-minor rows -> [H*n_tok, D] head-major rows."""
    return bq_rows.transpose(1, 0, 2).reshape(num_q_heads * n_tok, head_dim)

  def reduce_heads(bq_rows, bkv, bq_weights, head_major=False):
    """relu(q @ k) weighted-summed over heads.

    Args:
      bq_rows: [n_tok * num_q_heads, head_dim] queries, head-minor.
      bkv: the packed key block for this step.
      bq_weights: [n_tok, num_q_heads] indexer weights.

    Returns:
      [n_tok, bkv_sz] f32 weighted sum over heads.
    """
    n_tok = bq_weights.shape[0]
    w = bq_weights.astype(jnp.float32)

    def qk(rows):
      if seq_along_lane:
        # bkv is [head_dim, bkv_sz]: the MXU-native contraction layout.
        return jnp.einsum(
            "nd,dm->nm", rows, bkv, preferred_element_type=jnp.float32
        )
      # bkv is [bkv_sz, head_dim].
      return jnp.einsum(
          "nd,md->nm", rows, bkv, preferred_element_type=jnp.float32
      )

    if n_tok == 1:
      # Decode. With a single query token there is no token axis to trade
      # against the head axis, so the plain vectorized form is cheapest.
      st = qk(bq_rows).reshape(n_tok, num_q_heads, -1)
      st = jnp.maximum(st, 0.0) * w[:, :, None]
      return st.sum(axis=1)

    # Prefill. Transpose to head-major if needed.
    rows_hm = bq_rows if head_major else transpose_to_head_major(bq_rows, n_tok)
    st = qk(rows_hm).reshape(num_q_heads, n_tok, -1)
    acc = jnp.maximum(st[0], 0.0) * w[:, 0:1]
    # Loop over heads 1..H-1, accumulating the sum of relu(st[h] * w[:, h]).
    for h in range(1, num_q_heads):
      relu_st = jnp.maximum(st[h], 0.0)  # relu
      acc = acc + relu_st * w[:, h : h + 1]  # fused multiply-add.
    return acc

  def compute_scores(
      bq_vec,
      bkv_vec,
      scale_val_vec,
      bq_weights_vec,
      bq_pos_compressed_vec,
      bkv_idx,
      batch_start_seq_idx,
      cp_rank=0,
      q_is_head_major=False,
  ):
    assert len(bq_vec) == seq_batch_size
    assert len(bkv_vec) == seq_batch_size
    assert len(scale_val_vec) == seq_batch_size
    assert len(bq_weights_vec) == seq_batch_size
    assert len(bq_pos_compressed_vec) == seq_batch_size
    ret = []

    for batch_idx in range(seq_batch_size):
      bq = bq_vec[batch_idx].reshape(-1, head_dim)
      bkv = bkv_vec[batch_idx]
      scale_val = scale_val_vec[batch_idx]
      bq_weights = bq_weights_vec[batch_idx]
      bq_pos_compressed = bq_pos_compressed_vec[batch_idx]

      s_summed = reduce_heads(bq, bkv, bq_weights, head_major=q_is_head_major)
      s_summed = s_summed * scale_val
      k_local = bkv_idx * bkv_sz + lax.broadcasted_iota(
          jnp.int32, s_summed.shape, 1
      )
      k_span = cp_local_to_global(k_local, cp_rank, cp_size, interleave_c)
      seq_len = seq_lens_ref[batch_start_seq_idx + batch_idx]
      kv_len = seq_len >> shift_comp

      valid_mask = k_span < kv_len
      causal_mask = k_span <= bq_pos_compressed[:, None]
      mask = jnp.logical_and(valid_mask, causal_mask)
      s_summed = jnp.where(mask, s_summed, -jnp.inf)
      ret.append(s_summed.reshape(-1, num_sublanes_bkv, 128))
    return jnp.concatenate(ret, axis=0)

  def step_body(q_vmem, weights_vmem, bkv_vmem, scores_vmem, q_hm_ref=None):
    p_id = pl.program_id(0)
    batch_start_seq_idx = seq_tile(p_id)
    bq_idx = bq_block(p_id)
    bkv_idx = bkv_block(p_id)
    cp_rank = cp_rank_ref[0]

    if q_hm_ref is not None:
      # The head-major reordering depends only on the q block, but the grid
      # iterates kv innermost (`bkv_idx = step % num_bkv`), so bq_idx is held
      # fixed for a whole run of steps. Reordering once per q block instead of
      # once per step avoids redoing identical work num_bkv times; `bkv_idx
      # == 0` marks exactly the first step of each new q block.
      @pl.when(bkv_idx == 0)
      def _():
        q_hm_ref[...] = transpose_to_head_major(q_vmem[...], bq_sz)

      bq_vec = [q_hm_ref[...]]
    else:
      q_data = q_vmem[...].reshape(seq_batch_size, -1, head_dim)
      bq_vec = [q_data[batch_idx] for batch_idx in range(seq_batch_size)]
    w_data = weights_vmem[...].reshape(seq_batch_size, -1, num_q_heads)
    bq_weights_vec = [w_data[batch_idx] for batch_idx in range(seq_batch_size)]

    # Blocks are laid out from the chunk's first token, 0 when unchunked.
    chunk_start = start_end_seq_idx_ref[2]
    bq_pos_compressed_vec = []
    for batch_idx in range(seq_batch_size):
      s_l = seq_lens_ref[batch_start_seq_idx + batch_idx]
      q_s = cu_q_lens_ref[batch_start_seq_idx + batch_idx]
      q_e = cu_q_lens_ref[batch_start_seq_idx + batch_idx + 1]
      q_l = q_e - q_s
      # Offset of this block's first token within the sequence.
      block_start = bq_idx * bq_sz + jnp.maximum(0, chunk_start - q_s)
      q_pos = s_l - q_l + block_start + jnp.arange(bq_sz, dtype=jnp.int32)
      bq_pos_compressed_vec.append(q_pos >> shift_comp)

    bkvs = []
    bkv_scales = []
    for batch_idx in range(seq_batch_size):
      if seq_along_lane:
        # Keys: sublane rows [0, head_dim) of the packed block.
        key_bytes = bkv_vmem[batch_idx, :kv_head_dim_groups]
        fp8_val = pltpu.bitcast(
            key_bytes.reshape(head_dim, bkv_sz), jnp.float8_e4m3fn
        )
        # Scales: the first sublane row of the trailing packed group. This is
        # already laid out along lanes, so no transpose/relayout is needed.
        scale_bytes = bkv_vmem[
            batch_idx, kv_head_dim_groups : kv_head_dim_groups + 1
        ].reshape(kv_packing, bkv_sz)[0:1]
        scale_val = pltpu.bitcast(scale_bytes, jnp.float8_e8m0fnu).astype(
            jnp.bfloat16
        )
        bkvs.append(fp8_val)
        bkv_scales.append(scale_val)
      else:
        bkv = bkv_vmem[batch_idx, :bkv_sz_per_kv_packing][...]
        flat_bkv = bkv.reshape(-1, bkv.shape[-1])
        fp8_val = pltpu.bitcast(flat_bkv[:, :head_dim], jnp.float8_e4m3fn)
        scale_val = pltpu.bitcast(
            flat_bkv[:, head_dim : head_dim + 1].T, jnp.float8_e8m0fnu
        ).astype(jnp.bfloat16)
        bkvs.append(fp8_val.reshape(bkv_sz, head_dim))
        bkv_scales.append(scale_val)

    scores = compute_scores(
        bq_vec,
        bkvs,
        bkv_scales,
        bq_weights_vec,
        bq_pos_compressed_vec,
        bkv_idx,
        batch_start_seq_idx,
        q_is_head_major=q_hm_ref is not None,
        cp_rank=cp_rank,
    )
    scores_vmem[...] = pltpu.bitcast(scores, jnp.int32)

  def _run_all(q_hm_ref=None):
    @pl.with_scoped(final_allocs=(q_alloc, weights_alloc, kv_alloc, o_alloc))
    def _run_pipeline(final_allocs):
      q_buf, weights_buf, kv_buf, o_buf = final_allocs
      pipeline_fn = pltpu.emit_pipeline(
          functools.partial(step_body, q_hm_ref=q_hm_ref),
          grid=(metadata_ref.num_steps[chunk_idx],),
          in_specs=(q_buf.spec, weights_buf.spec, kv_buf.spec),
          out_specs=o_buf.spec,
      )
      window_refs = (cu_q_lens_ref, metadata_ref, start_end_seq_idx_ref)
      pipeline_fn(
          (q_hbm_ref, *window_refs),
          (indexer_weights_hbm_ref, *window_refs),
          (cache_kv_hbm_ref, page_indices_ref, metadata_ref),
          (scores_hbm_ref, *window_refs),
          allocations=(q_buf, weights_buf, kv_buf, o_buf),
      )

    _run_pipeline()

  # Only the multi-token path reorders q, and only a single sequence per tile
  # keeps the cached block unambiguous.
  if seq_batch_size == 1 and bq_sz > 1:
    pl.run_scoped(
        _run_all,
        pltpu.VMEM((num_q_heads * bq_sz, head_dim), q_hbm_ref.dtype),
    )
  else:
    _run_all()

  ### ------- Kernel end ------- ###


def prepare_q_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
  _, actual_num_q_heads, actual_head_dim = q.shape
  q_packing = get_dtype_packing(q.dtype)
  num_q_heads = align_to(actual_num_q_heads, q_packing)
  head_dim = align_to(actual_head_dim, 128)
  q = jnp.pad(
      q,
      (
          (0, 0),
          (0, num_q_heads - actual_num_q_heads),
          (0, head_dim - actual_head_dim),
      ),
      constant_values=0,
  )
  return q


def prepare_index_weights(
    index_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads],
    q_dtype,
):
  _, actual_num_q_heads = index_weights.shape
  index_weights = index_weights.astype(jnp.float32)
  num_q_heads = align_to(actual_num_q_heads, get_dtype_packing(q_dtype))
  index_weights = jnp.pad(
      index_weights,
      (
          (0, 0),
          (0, num_q_heads - actual_num_q_heads),
      ),
      constant_values=0,
  )
  return index_weights


def prepare_outputs(out):
  if out.ndim == 3:
    out = out.reshape(out.shape[0], -1)
  return out


# --------------------------------------------------------------------------
# KV cache layout helpers
# --------------------------------------------------------------------------
#
# The index cache stores, per compressed KV token, `head_dim` FP8 (e4m3) bytes
# followed by a single UE8M0 (e8m0) scale byte.
#
# HEAD_ALONG_SUBLANE puts the token on the sublane dimension, so the record
# must be padded out to a multiple of the 128-lane vector register width:
# `align_to(head_dim + 1, 128)`. For head_dim=128 that wastes 127 of every 256
# bytes of HBM traffic.
#
# SEQ_ALONG_LANE puts the token on the *lane* dimension, so the record only has
# to be padded out to a multiple of the sublane packing (4 for 8-bit data):
# `head_dim + kv_packing`. For head_dim=128 that is 132 bytes, a 1.94x
# reduction in HBM bytes per token.

# 8-bit KV data packs 4 values into every 32-bit sublane word.
KV_PACKING = 4


def get_kv_cache_shape(
    total_num_pages: int,
    page_size: int,
    actual_head_dim: int,
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
) -> tuple[int, ...]:
  """Returns the packed uint8 index KV cache shape for a given layout."""
  kv_layout = KVLayout.parse(kv_layout)
  head_dim = align_to(actual_head_dim, 128)
  if kv_layout == KVLayout.SEQ_ALONG_LANE:
    if page_size % 128 != 0:
      raise ValueError(
          "SEQ_ALONG_LANE requires page_size to be a multiple of the 128-lane"
          f" register width, got {page_size=}."
      )
    return (
        total_num_pages,
        head_dim // KV_PACKING + 1,
        KV_PACKING,
        page_size,
    )
  if page_size % KV_PACKING != 0:
    raise ValueError(
        "HEAD_ALONG_SUBLANE requires page_size to be a multiple of"
        f" {KV_PACKING}, got {page_size=}."
    )
  # `head_dim // 128` scale bytes are reserved to mirror the compressor's
  # record format, then the record is padded to the lane width.
  record_width = head_dim + head_dim // 128
  return (
      total_num_pages,
      page_size // KV_PACKING,
      KV_PACKING,
      align_to(record_width, 128),
  )


def convert_cache_to_seq_along_lane(
    cache_kv: jax.Array,  # uint8[pages, page_size // 4, 4, width]
    actual_head_dim: int,
) -> jax.Array:  # uint8[pages, head_dim // 4 + 1, 4, page_size]
  """Repacks a HEAD_ALONG_SUBLANE index cache into the SEQ_ALONG_LANE layout.

  This is a host-side (XLA) conversion intended for tests and for one-off
  migration of an existing cache; production callers should write the cache in
  the target layout directly.
  """
  total_num_pages, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
  page_size = page_size_per_kv_packing * kv_packing
  head_dim = align_to(actual_head_dim, 128)

  # [pages, page_size, width] -> keys and the single scale byte per token.
  flat = cache_kv.reshape(total_num_pages, page_size, -1)
  keys = flat[:, :, :head_dim]  # [pages, page_size, head_dim]
  scales = flat[:, :, head_dim : head_dim + 1]  # [pages, page_size, 1]
  # Pad the scale record out to a full packed sublane group.
  scales = jnp.pad(
      scales, ((0, 0), (0, 0), (0, KV_PACKING - 1)), constant_values=0
  )
  rows = jnp.concatenate([keys, scales], axis=-1)
  # [pages, page_size, head_dim + 4] -> [pages, head_dim + 4, page_size]
  rows = rows.swapaxes(1, 2)
  return rows.reshape(
      total_num_pages,
      (head_dim + KV_PACKING) // KV_PACKING,
      KV_PACKING,
      page_size,
  )


def _effective_row_lengths(
    seq_lens: jax.Array,  # i32[max_num_seqs]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    num_tokens: int,
    num_positions: int,
    compression_ratio: int,
    cp_rank: jax.Array | int = 0,
    cp_size: int = 1,
    interleave_c: int = 1,
) -> jax.Array:  # i32[num_tokens]
  """Visible compressed KV positions per token row of the score matrix.

  Under context parallelism the score matrix is rank-local, so the row length
  is the number of globally-visible positions that *this* rank owns.
  """
  max_num_seqs = seq_lens.shape[0]
  num_seqs = distribution[2]
  token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
  seq_mask = token_ids[:, None] >= cu_q_lens[None, 1 : max_num_seqs + 1]
  seq_mask = jnp.where(
      jnp.arange(max_num_seqs)[None, :] < num_seqs, seq_mask, False
  )
  seq_ids = jnp.sum(seq_mask, axis=1)

  q_start = cu_q_lens[seq_ids]
  q_len = cu_q_lens[seq_ids + 1] - q_start
  seq_len = seq_lens[seq_ids]
  token_pos = seq_len - q_len + (token_ids - q_start)
  kv_len = seq_len // compression_ratio
  visible = jnp.minimum(kv_len, token_pos // compression_ratio + 1)
  visible = cp_local_length(visible, cp_rank, cp_size, interleave_c)
  valid = token_ids < cu_q_lens[num_seqs]
  return jnp.where(valid, jnp.minimum(visible, num_positions), 0)


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "compression_ratio",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "buffer_count",
        "vmem_limit_bytes",
        "decode_req_batch_size",
        "enable_early_exit",
        "kv_layout",
        "chunk_tokens",
        "cp_size",
        "interleave_size",
        "return_scores",
    ),
)
def streamindex_topk(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    indexer_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads]
    # D = align_to(actual_head_dim, 128); W = align_to(D + D // 128, 128)
    # HEAD_ALONG_SUBLANE: uint8[pages, page_size // 4, 4, W]
    # SEQ_ALONG_LANE:     uint8[pages, D // 4 + 1, 4, page_size]
    cache_kv: jax.Array,
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    buffer_count: tuple[int, int, int] | int | None = DEFAULT_BUFFER_COUNT,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = False,
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
    chunk_tokens: int | None = None,
    cp_size: int = 1,
    cp_rank: jax.Array | int = 0,
    interleave_size: int = 1,
    return_scores: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """StreamIndex Top-K retrieval.

  Args:
    q: concatenated all sequences' queries.
    indexer_weights: concatenated all sequences' indexer weights.
    cache_kv: the current kv cache, packed as uint8. Its shape depends on
      `kv_layout`. With `D = align_to(actual_head_dim, 128)`: *
      `HEAD_ALONG_SUBLANE`: `[total_num_pages, page_size // 4, 4, align_to(D + D
      // 128, 128)]`. Tokens sit on the sublane dimension; each token's lane
      record holds its `D` FP8 key bytes followed by `D // 128` UE8M0 scale
      bytes, padded out to a multiple of the 128-lane register width. For
      `D=128` that is 129 useful bytes stored in 256. * `SEQ_ALONG_LANE`:
      `[total_num_pages, D // 4 + 1, 4, page_size]`. Tokens sit on the lane
      dimension; sublane rows `[0, D)` hold the FP8 key bytes, row `D` holds the
      per-token UE8M0 scale, and rows `D+1 .. D+3` are padding that completes
      the final packed sublane group. For `D=128` that is 132 bytes per token.
    seq_lens: the length of each sequence in the kv cache (uncompressed).
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    k: Number of top-K elements to retrieve.
    compression_ratio: KV cache compression ratio.
    num_kv_pages_per_block: number of kv pages to be processed in one block in
      the pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
    num_queries_per_block: number of queries to be processed in one block in the
      pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
    buffer_count: buffer count for the pallas kernel. This is a tuple of
      (decode, prefill, mixed) cases or a single integer. Defaults to (4, 3, 3).
    vmem_limit_bytes: the vmem limit for the pallas kernel.
    enable_early_exit: whether to enable early exit using jax.lax.cond when k >=
      kv_len for all sequences in the batch. Defaults to False.
    decode_req_batch_size: maximum decode batch size per iteration.
    kv_layout: memory layout of `cache_kv`. `SEQ_ALONG_LANE` removes the lane
      padding that per-token quantization forces on `HEAD_ALONG_SUBLANE` and is
      therefore substantially faster whenever the kernel is HBM bound.
    chunk_tokens: size, in tokens, of each piece to cut the flat token axis
      into. Each piece scores into its own buffer and runs its own SparseCore
      top-k, and consecutive pieces share an XLA scheduling group so chunk i's
      top-k overlaps chunk i+1's scoring. None, or any size that does not divide
      `max_num_tokens`, runs a single unpipelined pass. Defaults to None: the
      size that wins depends on the workload and the KV layout, so the caller
      picks it.
    cp_size: number of context-parallel ranks the compressed KV cache is sharded
      over. 1 (default) means no sharding and the whole CP path compiles away.
      When > 1, `cache_kv` / `page_indices` describe only this rank's shard, and
      the returned top-k is *local* -- correct only after the cross-rank merge
      in `streamindex_topk_dcp`.
    cp_rank: this rank's index in the CP group. Traced, so one compiled program
      serves every rank.
    interleave_size: CP chunk-interleave width in uncompressed tokens. Must be a
      multiple of `compression_ratio`.
    return_scores: also return the score of each selected position, which is
      what makes a cross-rank merge possible. A local top-k over a sharded KV
      axis is not the global top-k, so the merge needs the scores the same way
      the DCP attention kernels need an LSE output.

  Returns:
    Top-K indices in this rank's local compressed space, or
    `(indices, scores)` if `return_scores`. Unfilled slots are `-1` with
    score `-inf`.
  """
  # Scale factors for the FP8 index cache format are packed directly inside
  # `cache_kv`, keeping HBM transactions fused.

  kv_layout = KVLayout.parse(kv_layout)

  if cp_size < 1:
    raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
  if cp_size > 1:
    if interleave_size % compression_ratio != 0:
      raise ValueError(
          f"interleave_size ({interleave_size}) must be a multiple of "
          f"compression_ratio ({compression_ratio}) for the CP chunk "
          "boundary to fall on a compressed-row boundary."
      )
    if enable_early_exit:
      raise NotImplementedError(
          "enable_early_exit is not supported with cp_size > 1."
      )
  if enable_early_exit and return_scores:
    raise NotImplementedError(
        "return_scores is not supported with enable_early_exit."
    )
  interleave_c = interleave_size // compression_ratio if cp_size > 1 else 1

  if num_kv_pages_per_block is None or num_queries_per_block is None:
    raise ValueError(
        "num_kv_pages_per_block and num_queries_per_block must be specified."
    )
  if (
      compression_ratio < 1
      or (compression_ratio & (compression_ratio - 1)) != 0
  ):
    raise ValueError("compression_ratio must be a power of 2.")

  if isinstance(num_kv_pages_per_block, int):
    num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
  else:
    num_kv_pages_per_blocks = num_kv_pages_per_block

  if isinstance(num_queries_per_block, int):
    num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
  else:
    num_queries_per_blocks = num_queries_per_block

  if buffer_count is None:
    buffer_counts = list(DEFAULT_BUFFER_COUNT)
  elif isinstance(buffer_count, int):
    buffer_counts = [buffer_count for _ in range(3)]
  else:
    buffer_counts = list(buffer_count)

  if len(buffer_counts) != 3:
    raise ValueError(
        "buffer_count must be a 3-tuple or a single integer, got"
        f" {buffer_count}"
    )

  if chunk_tokens is not None and chunk_tokens % decode_req_batch_size:
    raise ValueError(
        f"chunk_tokens ({chunk_tokens}) must be a multiple of"
        f" decode_req_batch_size ({decode_req_batch_size}), or a cut can"
        " fall inside a decode block and split it across two chunks."
    )

  max_num_seqs = seq_lens.shape[0]

  original_dtype = q.dtype
  actual_head_dim = q.shape[-1]

  prepared_indexer_weights = prepare_index_weights(
      indexer_weights, original_dtype
  )
  q = prepare_q_inputs(q)
  head_dim = q.shape[-1]

  if kv_layout == KVLayout.SEQ_ALONG_LANE:
    total_num_pages, kv_sublane_groups, kv_packing, page_size = cache_kv.shape
    expected_shape = get_kv_cache_shape(
        total_num_pages, page_size, actual_head_dim, kv_layout
    )
    if cache_kv.shape != expected_shape:
      raise ValueError(
          "SEQ_ALONG_LANE expects cache_kv of shape"
          f" {expected_shape}, got {cache_kv.shape}."
      )
    if kv_sublane_groups * kv_packing != head_dim + kv_packing:
      raise ValueError(
          f"Inconsistent SEQ_ALONG_LANE cache: {kv_sublane_groups=},"
          f" {kv_packing=} does not match {head_dim=}."
      )
  else:
    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
  pages_per_seq = page_indices.shape[0] // max_num_seqs

  for bkv_p in num_kv_pages_per_blocks:
    bkv_sz = page_size * bkv_p
    if bkv_sz % 128 != 0:
      raise ValueError(
          f"bkv_sz ({page_size} * {bkv_p} = {bkv_sz}) must be a multiple of"
          " 128."
      )

  num_sublanes_total = max(
      align_to(pages_per_seq, bkv_p) * page_size // 128
      for bkv_p in num_kv_pages_per_blocks
  )

  def _block_sizes(num_queries_per_block, bkv_p, static_q_len):
    """Query and KV block sizes a pass runs with."""
    if static_q_len is not None:
      bq_sz = min(num_queries_per_block, static_q_len)
    else:
      bq_sz = num_queries_per_block
    bkv_sz = page_size * bkv_p
    return bq_sz, bkv_sz

  def run_scores_kernel(
      q,
      prepared_indexer_weights,
      cache_kv,
      scores_init,
      seq_lens,
      page_indices,
      cu_q_lens,
      start_seq_idx,
      end_seq_idx,
      static_q_len,
      num_kv_pages_per_block,
      num_queries_per_block,
      buffer_count,
      seq_batch_size,
      out_dtype,
      pass_meta,
      case=MlaCase.MIXED,
      chunk_token_start=None,
      chunk_tokens=None,
      scheduling_group_id=None,
      chunk_idx=0,
  ):
    max_num_tokens = q.shape[0]
    # Only support batching for decode sequences.
    # TODO: support batching for decode sequences with speculative decoding
    # enabled, e.g. static_q_len = gamma + 1.
    if seq_batch_size > 1:
      assert static_q_len == 1

    chunked = chunk_token_start is not None
    if not chunked:
      chunk_tokens = max_num_tokens

    bkv_p = num_kv_pages_per_block
    bq_sz, _ = _block_sizes(num_queries_per_block, bkv_p, static_q_len)

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    # Passes 2 and 3 donate the previous pass's HBM buffer so its rows survive
    # into this output. Only pass 1 has no donor, so Pallas allocates it there;
    # one buffer per chunk either way.
    aliased = scores_init is not None
    in_specs = [
        hbm_spec,  # q
        hbm_spec,  # prepared_indexer_weights
        hbm_spec,  # cache_kv
    ]
    if aliased:
      in_specs.append(hbm_spec)  # scores_init, aliased to out
    out_specs = hbm_spec

    scratch_shapes = []

    chunk_start = 0 if chunk_token_start is None else chunk_token_start
    scalar_prefetches = (
        seq_lens,
        page_indices,
        cu_q_lens,
        jnp.array([start_seq_idx, end_seq_idx, chunk_start], jnp.int32),
        pass_meta,
        jnp.asarray([cp_rank], jnp.int32),
    )

    num_scalar_prefetch = len(scalar_prefetches)
    num_flat_prefetches = len(jax.tree_util.tree_leaves(scalar_prefetches))
    # The aliased `scores_init` is the 4th operand after the flat prefetches.
    input_output_aliases = {num_flat_prefetches + 3: 0} if aliased else {}

    layout_tag = "" if kv_layout == KVLayout.HEAD_ALONG_SUBLANE else "-sal"
    scope_name = (
        f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}{layout_tag}"
    )
    if chunked:
      # The chunk count is part of the kernel's identity, so a sweep over chunk
      # sizes shows up in the profile instead of silently reusing one shape.
      scope_name += f"-c{max_num_tokens // chunk_tokens}"
    pallas_kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                kernel,
                compression_ratio=compression_ratio,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
                seq_batch_size=seq_batch_size,
                buffer_count=buffer_count,
                case=case,
                kv_layout=kv_layout,
                cp_size=cp_size,
                interleave_c=interleave_c,
                chunk_idx=chunk_idx,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=num_scalar_prefetch,
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(1,),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(
                shape=(chunk_tokens, num_sublanes_total, 128),
                dtype=out_dtype,
            ),
            input_output_aliases=input_output_aliases,
            name=scope_name,
        )
    )
    scores = pallas_kernel(
        *scalar_prefetches,
        q,
        prepared_indexer_weights,
        cache_kv,
        *((scores_init,) if aliased else ()),
    )
    if scheduling_group_id is not None:
      scores = xla_metadata.set_xla_metadata(
          scores, _scheduling_group_id=scheduling_group_id
      )
    return scores

  def _pass_specs(decode_batch_end):
    """`run_scores_kernel` kwargs for the three passes over the batch."""
    decode_cfg = dict(
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        buffer_count=buffer_counts[0],
        static_q_len=1,
        case=MlaCase.DECODE,
    )
    return (
        dict(
            start_seq_idx=jnp.array(0),
            end_seq_idx=decode_batch_end,
            seq_batch_size=decode_req_batch_size,
            **decode_cfg,
        ),
        # Handle num_decode_seqs % decode_req_batch_size != 0 case.
        dict(
            start_seq_idx=decode_batch_end,
            end_seq_idx=distribution[1],
            seq_batch_size=1,
            **decode_cfg,
        ),
        dict(
            num_kv_pages_per_block=num_kv_pages_per_blocks[2],
            num_queries_per_block=num_queries_per_blocks[2],
            buffer_count=buffer_counts[2],
            start_seq_idx=distribution[1],
            end_seq_idx=distribution[2],
            static_q_len=None,
            seq_batch_size=1,
            case=MlaCase.MIXED,
        ),
    )

  def _pass_metadata(spec, chunk_starts, chunk_tokens):
    """One schedule per pass, holding every chunk's steps back to back."""
    bq_sz, bkv_sz = _block_sizes(
        spec["num_queries_per_block"],
        spec["num_kv_pages_per_block"],
        spec["static_q_len"],
    )
    return metadata.compute_metadata(
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        start_seq_idx=spec["start_seq_idx"],
        end_seq_idx=spec["end_seq_idx"],
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        pages_per_seq=pages_per_seq,
        compression_ratio=compression_ratio,
        static_q_len=spec["static_q_len"],
        seq_batch_size=spec["seq_batch_size"],
        page_size=page_size,
        max_num_tokens=q.shape[0],
        chunk_token_start=chunk_starts,
        chunk_tokens=chunk_tokens,
        cp_rank=cp_rank,
        cp_size=cp_size,
        interleave_c=interleave_c,
    )

  def _scores_for_chunk(
      passes,
      chunk_idx,
      chunk_token_start,
      chunk_tokens,
      out_dtype,
      scheduling_group_id,
  ):
    """Scores for one window of the flat token axis, in a private buffer."""
    # `scores=None`: no -inf pre-fill. It would write all of HBM scores, and
    # XLA would common the identical per-chunk fills into one buffer then clone
    # it back out (~370us on 1GB of scores). `row_lengths` bounds the reads.
    scores = None
    for spec, pass_meta in passes:
      scores = run_scores_kernel(
          q,
          prepared_indexer_weights,
          cache_kv,
          scores,
          seq_lens,
          page_indices,
          cu_q_lens,
          pass_meta=pass_meta,
          chunk_idx=chunk_idx,
          chunk_token_start=chunk_token_start,
          chunk_tokens=chunk_tokens,
          out_dtype=out_dtype,
          scheduling_group_id=scheduling_group_id,
          **spec,
      )
    return scores

  def _common_path(_):
    # Raw f32 bits, not f32. The SparseCore top-k compares bits, and this is
    # the dtype it wants to read, so nothing has to reformat HBM scores.
    out_dtype = jnp.int32
    max_num_tokens = q.shape[0]

    # TODO: we shall sort the sequences by length, so that multiple decode
    # sequences in one batch have similar lengths to reduce waste of compute.
    # With the same batch size, the longest sequence will determine number of
    # blocks to run computation for.
    decode_batch_end = (
        distribution[0] // decode_req_batch_size * decode_req_batch_size
    )

    num_positions = max(num_sublanes_total * 128, k)
    eff_row_lengths = _effective_row_lengths(
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        num_tokens=max_num_tokens,
        num_positions=num_positions,
        compression_ratio=compression_ratio,
        cp_rank=cp_rank,
        cp_size=cp_size,
        interleave_c=interleave_c,
    )

    if chunk_tokens is None or max_num_tokens % chunk_tokens:
      num_chunks, chunk_len = 1, max_num_tokens
    else:
      num_chunks, chunk_len = max_num_tokens // chunk_tokens, chunk_tokens

    # Ids count from 1, so the 0 is never a real group: a single chunk has no
    # pair to annotate and takes the `None` branch below both times.
    group_base = 0
    if num_chunks > 1:
      group_base = config.reserve_scheduling_group_ids(num_chunks - 1)

    need_scores = cp_size > 1 or return_scores

    chunk_starts = (
        (None,)
        if num_chunks == 1
        else tuple(m * chunk_len for m in range(num_chunks))
    )
    passes = tuple(
        (spec, _pass_metadata(spec, chunk_starts, chunk_len))
        for spec in _pass_specs(decode_batch_end)
    )
    topk_idxs = []
    topk_scores_list = []
    for m in range(num_chunks):
      # Group g holds chunk g's SparseCore top-k and chunk g+1's TensorCore
      # scoring, so the scheduler emits `sc(g).start, tc(g+1), sc(g).done`.
      # Chunk 0's scoring and the last top-k have no partner.
      tc_group = None if m == 0 else group_base + m - 1
      sc_group = None if m == num_chunks - 1 else group_base + m
      # Stage 2 lands one group later than its own stage 1, so it overlaps
      # chunk m + 2's scoring. The last two chunks have no later group.
      sc2_group = None if m >= num_chunks - 2 else group_base + m + 1

      start = m * chunk_len
      scores = _scores_for_chunk(
          passes,
          m,
          chunk_starts[m],
          chunk_len,
          out_dtype,
          tc_group,
      )
      scores = scores.reshape(chunk_len, -1)
      if scores.shape[1] < k:
        scores = jnp.pad(
            scores,
            ((0, 0), (0, k - scores.shape[1])),
            constant_values=BITS_NEG_INF,
        )

      chunk_topk_idxs, chunk_topk_scores = sparsecore_topk(
          scores,
          k,
          row_lengths=eff_row_lengths[start : start + chunk_len],
          write_empty_rows=True,
          scheduling_group_id=sc_group,
          stage2_scheduling_group_id=sc2_group,
          return_scores=True,
      )
      topk_idxs.append(chunk_topk_idxs)
      if need_scores:
        topk_scores_list.append(chunk_topk_scores)

    out = topk_idxs[0] if len(topk_idxs) == 1 else jnp.concatenate(topk_idxs)
    out = out[:max_num_tokens, :k]

    if cp_size == 1 and not return_scores:
      return out

    scores_out = (
        topk_scores_list[0]
        if len(topk_scores_list) == 1
        else jnp.concatenate(topk_scores_list)
    )
    scores_out = scores_out[:max_num_tokens, :k]

    if not return_scores:
      return out
    return out, scores_out

  def _fast_path(_):
    token_idx = jnp.arange(q.shape[0])
    seq_idx = jnp.minimum(
        jnp.searchsorted(cu_q_lens[1:], token_idx, side="right"),
        seq_lens.shape[0] - 1,
    )
    seq_len = seq_lens[seq_idx]
    q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
    q_start = cu_q_lens[seq_idx]
    q_abs_pos = (seq_len - q_len) + (token_idx - q_start)
    max_valid_idx = jnp.minimum(
        seq_len // compression_ratio - 1,
        q_abs_pos // compression_ratio,
    )
    max_valid_idx = jnp.where(token_idx < cu_q_lens[-1], max_valid_idx, -1)
    s_idx = jnp.arange(k, dtype=jnp.int32)[None, :]
    return jnp.where(s_idx <= max_valid_idx[:, None], s_idx, -1)

  if not enable_early_exit:
    return _common_path(None)
  return jax.lax.cond(
      jnp.max(seq_lens) // compression_ratio <= k,
      _fast_path,
      _common_path,
      None,
  )


@functools.partial(
    jax.jit,
    static_argnames=(
        "mesh",
        "k",
        "compression_ratio",
        "dcp_size",
        "interleave_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "buffer_count",
        "vmem_limit_bytes",
        "decode_req_batch_size",
        "chunk_tokens",
        "dcp_axis_name",
    ),
)
def streamindex_topk_dcp(
    q: jax.Array,  # [padded_num_tokens, num_q_heads, head_dim], replicated
    indexer_weights: jax.Array,  # [padded_num_tokens, num_q_heads], replicated
    cache_kv: jax.Array,  # sharded on axis 0 over the DCP axis
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * virtual_pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    mesh,
    k: int,
    compression_ratio: int,
    dcp_size: int,
    interleave_size: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    buffer_count: tuple[int, int, int] | int | None = DEFAULT_BUFFER_COUNT,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    chunk_tokens: int | None = None,
    dcp_axis_name: str = DCP_AXIS_NAME,
) -> jax.Array:
  """Exact global top-k over a DCP-sharded KV cache, delivered rank-local.

  Architecture:
    1. Local Scoring: Every rank scores the replicated query tokens against its
       own local KV cache shard via `streamindex_topk`, producing candidate
       indices and scores of shape `[padded_num_tokens, k]`. Because DCP
       partitions the KV cache, a true global top-k winner is beaten by at most
       `k - 1` positions globally, and therefore by at most `k - 1` positions
       on its owner rank. Thus, every true global winner is guaranteed to
       survive into its owner's local candidate list.
    2. Candidate All-Gather: A single `all_gather` along the candidate axis
       (axis 1) exchanges candidate *scores* only, rank-major:
       `[T, k] -> [T, dcp_size * k]`. The indices are not exchanged, so they
       never leave the rank that produced them.
    3. SparseCore Merge: Each rank runs `sparsecore_topk` over the concatenated
       `[T, dcp_size * k]` scores to select the exact top-k global winners:
       `[T, dcp_size * k] -> [T, k]`.
    4. Rank-Local Filtering: `dcp_sc_compact.resolve_owned_winners` keeps the
       slots whose chunk came from `dcp_rank`, resolves them against this
       rank's own pre-gather indices at `slot % k`, and left-packs them into a
       `-1` padded prefix. Those indices are rank-local already -- they never
       crossed the gather -- so no coordinate conversion is needed and
       ownership is decided in exactly one place. The ranks' outputs form an
       exact partition of the global top-k.

  By gathering candidates along axis 1 upfront rather than partitioning tokens
  over an `all_to_all`, all communication is consolidated into a single upfront
  collective phase, avoiding the barrier latency of a multi-phase pipeline and
  preserving the token axis `T` unpartitioned.

  Args:
    q: replicated query tokens, in natural request-major order
      `[padded_num_tokens, num_q_heads, head_dim]`.
    indexer_weights: replicated query weights `[padded_num_tokens,
      num_q_heads]`.
    cache_kv: this rank's shard of the compressed KV cache.
    seq_lens: the length of each sequence in the kv cache (uncompressed).
    page_indices: replicated *virtual* page ordinals, resolved against each
      rank's own shard by `local = virtual % num_local_pages`.
    cu_q_lens: the cumulative sum of the effective query lengths.
    distribution: (i, j, k) partition indices.
    mesh: mesh containing `dcp_axis_name`.
    k: Number of top-K elements to retrieve.
    compression_ratio: KV cache compression ratio.
    dcp_size: number of context-parallel ranks.
    interleave_size: chunk-interleave width in uncompressed tokens.
    num_kv_pages_per_block: number of kv pages per block in pallas kernel.
    num_queries_per_block: number of queries per block in pallas kernel.
    buffer_count: buffer count for pallas kernel.
    vmem_limit_bytes: vmem limit for pallas kernel.
    decode_req_batch_size: maximum decode batch size per iteration.
    chunk_tokens: forwarded to `streamindex_topk` for the per-rank local scoring
      pass; see there for the semantics. Chunking cuts the replicated
      query-token axis, which is orthogonal to the KV axis DCP shards, so the
      two compose.
    dcp_axis_name: name of the DCP mesh axis.

  Returns:
    i32[padded_num_tokens, k] of this rank's own *local* cache indices for
    every token, packed into a prefix and `-1` padded, sharded over
    `dcp_axis_name` (global leading dim `dcp_size * padded_num_tokens`).
    Callers derive per-token counts from the `-1` padding; a row that is all
    `-1` means this rank owns none of that token's top-k, which the attention
    side must mask out of the LSE merge rather than attend over. Rows are
    `k` wide but hold `k / dcp_size` entries on average, so a caller that
    gathers the full width pays `dcp_size`x more than it needs to; the
    attention side should bound its gather by the per-row count.
  """
  if dcp_size <= 1:
    raise ValueError(
        f"streamindex_topk_dcp requires dcp_size > 1, got {dcp_size}; "
        "use streamindex_topk for the unsharded case."
    )
  if dcp_axis_name not in mesh.axis_names:
    raise ValueError(f"mesh {mesh.axis_names} has no {dcp_axis_name!r} axis.")
  if mesh.shape[dcp_axis_name] != dcp_size:
    raise ValueError(
        f"dcp_size={dcp_size} does not match mesh axis"
        f" {dcp_axis_name!r}={mesh.shape[dcp_axis_name]}."
    )
  if q.shape[0] % dcp_size != 0:
    raise ValueError(
        f"padded_num_tokens={q.shape[0]} must be divisible by"
        f" dcp_size={dcp_size}."
    )

  def _local(
      q,
      indexer_weights,
      cache_kv,
      seq_lens,
      page_indices,
      cu_q_lens,
      distribution,
  ):
    dcp_rank = cp_rank_as_data(dcp_axis_name, dcp_size)
    local_page_indices = jnp.mod(page_indices, jnp.int32(cache_kv.shape[0]))

    idxs, scores = streamindex_topk(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        local_page_indices,
        cu_q_lens,
        distribution,
        k=k,
        compression_ratio=compression_ratio,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        buffer_count=buffer_count,
        vmem_limit_bytes=vmem_limit_bytes,
        decode_req_batch_size=decode_req_batch_size,
        enable_early_exit=False,
        chunk_tokens=chunk_tokens,
        cp_size=dcp_size,
        cp_rank=dcp_rank,
        interleave_size=interleave_size,
        return_scores=True,
    )

    # Stage 1. All-gather the scores; the indices stay home.
    scores = lax.all_gather(scores, dcp_axis_name, axis=1, tiled=True)

    # Stage 2. Full width as the row length: -inf entries (unfilled local
    # slots, padding tokens) are never selected, so they need no masking.
    slots = sparsecore_topk(
        scores,
        k,
        row_lengths=jnp.full(
            (scores.shape[0],), scores.shape[1], dtype=jnp.int32
        ),
        write_empty_rows=True,
    )
    return dcp_sc_compact.resolve_owned_winners(idxs, slots, k, dcp_rank)

  replicated = P()
  return jax.shard_map(
      _local,
      mesh=mesh,
      in_specs=(
          replicated,  # q
          replicated,  # indexer_weights
          P(dcp_axis_name),  # cache_kv
          replicated,  # seq_lens
          replicated,  # page_indices
          replicated,  # cu_q_lens
          replicated,  # distribution
      ),
      out_specs=P(dcp_axis_name),
      check_vma=False,
  )(
      q,
      indexer_weights,
      cache_kv,
      seq_lens,
      page_indices,
      cu_q_lens,
      distribution,
  )
