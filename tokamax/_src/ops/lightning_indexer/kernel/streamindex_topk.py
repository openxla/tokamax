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
"""TPU-Friendly StreamIndex Top-K kernel.

Optimization Provenance:
  Blocked prefill scoring (`_BlockedPrefillConfig` with dual-MXU head-group
  splitting and strided VMEM score stores), 128-aligned metadata `max_steps`,
  zero-step `emit_pipeline` guard (`@pl.when(metadata_ref.num_steps[chunk_idx] >
  0)`),
  single-pass `uniform_q_len is None` dispatch fast paths, and single-stage
  `VMEM_SHARED` SparseCore top-k with single-mask (`key > lo_b`) candidate
  compaction and fused `sparsecore_topk_dcp_merge` developed via Jetski Teamwork
  + Auto evolutionary RSI optimization.
"""

import dataclasses
import functools
import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from tokamax._src.ops.lightning_indexer.kernel import (
    bref_override,
    config,
    metadata,
    tc_streamindex_topk,
    tc_topk as tc_topk_lib,
)
from tokamax._src.ops.topk import pallas_mosaic_tpu_kernel

BITS_NEG_INF = pallas_mosaic_tpu_kernel.BITS_NEG_INF
sparsecore_topk = pallas_mosaic_tpu_kernel.sparsecore_topk
sparsecore_topk_dcp_merge = pallas_mosaic_tpu_kernel.sparsecore_topk_dcp_merge

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


@dataclasses.dataclass(frozen=True)
class _BlockedPrefillConfig:
  """Multi-token (prefill / mixed) scoring layout and tiling configuration.

  The head reduction runs over blocks of `tok_blk` tokens and `lane_chunk` KV
  positions. Heads are dealt round-robin into one group per MXU (head `h` goes
  to group `h % G`), and the cached queries are ordered `(head_group,
  token_block, head, token)`: each chunk's `q @ k` is one matmul per head
  group, which the compiler spreads over the MXUs, and every MXU emits its rows
  in the exact order the per-head accumulation consumes them. The
  lane-broadcast indexer weights are cached once per q block instead of being
  rebuilt every step.
  """

  enabled: bool
  tok_blk: int
  num_tok_blks: int
  num_head_groups: int
  heads_per_group: int
  rows_per_group_tok_blk: int
  rows_per_group: int
  lane_chunk: int
  num_lane_chunks: int
  tiles_per_chunk: int
  strided_score_store: bool
  score_pitch: int
  bf16_reduce: bool
  wb_dtype: Any

  @classmethod
  def create(
      cls,
      *,
      bq_sz: int,
      bkv_sz: int,
      num_q_heads: int,
      q_dtype: Any,
      seq_batch_size: int,
  ) -> "_BlockedPrefillConfig":
    """Computes the blocked-prefill layout parameters for a kernel pass."""
    tok_blk = math.gcd(bq_sz, 32)
    num_tok_blks = bq_sz // tok_blk
    num_head_groups = math.gcd(num_q_heads, pltpu.get_tpu_info().num_mxus)
    heads_per_group = num_q_heads // num_head_groups

    lane_chunk = math.gcd(bkv_sz, 256)
    num_lane_chunks = bkv_sz // lane_chunk
    tiles_per_chunk = lane_chunk // 128

    q_sublane_tiling = 8 * get_dtype_packing(q_dtype)
    enabled = (
        seq_batch_size == 1 and bq_sz > 1 and tok_blk % q_sublane_tiling == 0
    )

    # Scores are written straight into the [bq, nsub, 128] block with strided
    # stores when every token owns whole (8, 128) tiles of it.
    num_sublanes_bkv = bkv_sz // 128
    strided_score_store = num_sublanes_bkv % 8 == 0
    # Rows between consecutive tokens of the VMEM score block. The sublanes of
    # one store must fall in distinct VMEM banks (row mod 32), so with a pitch
    # of nsub rows, a multiple of 16 for the usual block sizes, every strided
    # store splits into 2-row pieces (1-row at nsub = 32), and a bundle holding
    # two stores cannot also pop the MXU. A pitch of 8 mod 16 rows lets each
    # store cover 4 rows; the padding rows are never copied out.
    score_pitch = num_sublanes_bkv
    if enabled and strided_score_store and score_pitch % 16 == 0:
      score_pitch += 8

    # Reduced-precision head reduction (used for FP8 queries):
    # relu(q @ k) is rounded to bf16 and the weights are held in bf16, so each
    # weighted term is an exact bf16 x bf16 product accumulated in f32. The
    # compiler lowers that pattern to one multiply-accumulate per packed bf16
    # register, which halves the VPU work of the reduction; see
    # `reduce_token_block`. It needs whole packed (16, 128) tiles per token
    # block.
    bf16_reduce = get_dtype_bitwidth(q_dtype) == 8 and tok_blk % 16 == 0
    wb_dtype = jnp.bfloat16 if bf16_reduce else jnp.float32

    return cls(
        enabled=enabled,
        tok_blk=tok_blk,
        num_tok_blks=num_tok_blks,
        num_head_groups=num_head_groups,
        heads_per_group=heads_per_group,
        rows_per_group_tok_blk=heads_per_group * tok_blk,
        rows_per_group=heads_per_group * bq_sz,
        lane_chunk=lane_chunk,
        num_lane_chunks=num_lane_chunks,
        tiles_per_chunk=tiles_per_chunk,
        strided_score_store=strided_score_store,
        score_pitch=score_pitch,
        bf16_reduce=bf16_reduce,
        wb_dtype=wb_dtype,
    )

  def cache_row(self, h: int, tb: int) -> int:
    """First `q_hm_ref` row of head `h` in token block `tb`."""
    g, hg = h % self.num_head_groups, h // self.num_head_groups
    return (
        g * self.rows_per_group
        + tb * self.rows_per_group_tok_blk
        + hg * self.tok_blk
    )


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
    fused_topk_k: int | None = None,
    fused_return_scores: bool = False,
    num_sublanes_total: int = 1,
):
  """Executes the n-buffered Pallas scoring pipeline for StreamIndex Top-K."""
  del static_q_len, case
  fused_topk = fused_topk_k is not None
  if fused_topk:
    if fused_return_scores:
      _, _, topk_ids_out_ref, topk_scores_out_ref = score_refs
    else:
      _, topk_ids_out_ref = score_refs
      topk_scores_out_ref = None
    scores_hbm_ref = None
    chunk_tokens = topk_ids_out_ref.shape[0]
  else:
    del compression_ratio, cp_size, interleave_c, cp_rank_ref
    scores_hbm_ref = score_refs[-1]
    topk_ids_out_ref = None
    topk_scores_out_ref = None
    chunk_tokens = scores_hbm_ref.shape[0]
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

  prefill_cfg = _BlockedPrefillConfig.create(
      bq_sz=bq_sz,
      bkv_sz=bkv_sz,
      num_q_heads=num_q_heads,
      q_dtype=q_hbm_ref.dtype,
      seq_batch_size=seq_batch_size,
  )

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
      (pl.BoundedSlice(seq_batch_size * bq_sz), prefill_cfg.score_pitch, 128),
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
      chunk_tokens=chunk_tokens,
      chunk_idx=chunk_idx,
  )
  weights_alloc = bref_override.StreamIndexQBufferedRef.input(
      spec=weights_spec,
      dtype_or_type=indexer_weights_hbm_ref,
      buffer_count=buffer_cnt,
      use_lookahead=True,
      bq_sz=bq_sz,
      seq_batch_size=seq_batch_size,
      chunk_tokens=chunk_tokens,
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
  if fused_topk:
    o_alloc = None
  else:
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

  # ------------------------------------------------------------------------
  # SEQ_ALONG_LANE block access.
  #
  # The VMEM block is u8[seq_batch, kv_sublane_groups, kv_packing, bkv_sz]:
  # packed group g holds key dims [4g, 4g + 4) of every token, one byte per
  # dim, and group `kv_head_dim_groups` holds the UE8M0 scale in byte 0.
  # Viewed as u32, row `g * nt + j` (nt = bkv_sz // 128) is group g of lane
  # tile j, so a sublane-strided load of rows `j, j + nt, ...` yields the
  # [head_dim // 4, 128] words of lane tile j. Bitcasting those words to FP8
  # gives [head_dim, 128] with key dim 4g + i taken from byte i -- exactly the
  # MXU-native packed layout -- so no relayout is needed on the key path.
  # ------------------------------------------------------------------------
  num_lane_tiles_bkv = bkv_sz // 128

  def seq_along_lane_block_rows(bkv_vmem):
    return bkv_vmem.bitcast(jnp.uint32).reshape(
        seq_batch_size, kv_sublane_groups * num_lane_tiles_bkv, 128
    )

  def seq_along_lane_key_tile(rows, batch_idx, lane_tile_idx):
    """[head_dim, 128] FP8 keys of lane tile `lane_tile_idx`."""
    words = rows[
        batch_idx,
        pl.ds(lane_tile_idx, kv_head_dim_groups, stride=num_lane_tiles_bkv),
        :,
    ]
    return pltpu.bitcast(words, jnp.float8_e4m3fn)

  def seq_along_lane_scale_tile(rows, batch_idx, lane_tile_idx):
    """[1, 128] f32 dequantization scales of lane tile `lane_tile_idx`."""
    row = kv_head_dim_groups * num_lane_tiles_bkv + lane_tile_idx
    words = rows[batch_idx, pl.ds(row, 1), :]
    # UE8M0 holds a bare biased exponent, which is where f32 keeps its own:
    # moving byte 0 into the f32 exponent field yields 2^(e - 127) exactly.
    return pltpu.bitcast((words & 0xFF) << 23, jnp.float32)

  def transpose_to_head_major(bq_rows, n_tok):
    """[n_tok*H, D] head-minor rows -> [H*n_tok, D] head-major rows."""
    return bq_rows.transpose(1, 0, 2).reshape(num_q_heads * n_tok, head_dim)

  # Tokens per head in the cached head-major q block of the per-step path.
  # Whole 8-row sublane groups keep each head's [tokens, bkv] slice of q @ k
  # vreg-aligned: at the 7-token verify window a head straddles two groups,
  # and realigning every slice costs more VPU work than the reduction itself.
  # The zero pad rows are scored and dropped.
  hm_tok = align_to(bq_sz, 8)

  def pad_tokens(bq_rows, n_tok):
    """[n_tok, H, D] -> [hm_tok, H, D], zero rows appended."""
    if n_tok == hm_tok:
      return bq_rows
    pad = jnp.zeros((hm_tok - n_tok, *bq_rows.shape[1:]), bq_rows.dtype)
    return jnp.concatenate([bq_rows, pad], axis=0)

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
    if head_major:
      # The cached block carries `hm_tok` tokens per head.
      rows_hm, rows_tok = bq_rows, hm_tok
      w = jnp.pad(w, ((0, rows_tok - n_tok), (0, 0)))
    else:
      rows_hm, rows_tok = transpose_to_head_major(bq_rows, n_tok), n_tok
    st = qk(rows_hm).reshape(num_q_heads, rows_tok, -1)
    acc = jnp.maximum(st[0], 0.0) * w[:, 0:1]
    # Loop over heads 1..H-1, accumulating the sum of relu(st[h] * w[:, h]).
    for h in range(1, num_q_heads):
      relu_st = jnp.maximum(st[h], 0.0)  # relu
      acc = acc + relu_st * w[:, h : h + 1]  # fused multiply-add.
    return acc[:n_tok]

  def compute_scores(
      bq_vec,
      bkv_vec,
      scale_val_vec,
      bq_weights_vec,
      q_is_head_major=False,
  ):
    assert len(bq_vec) == seq_batch_size
    assert len(bkv_vec) == seq_batch_size
    assert len(scale_val_vec) == seq_batch_size
    assert len(bq_weights_vec) == seq_batch_size
    ret = []

    # Positions a token cannot see are left unmasked here because SparseCore
    # top-k bounds each score row by `row_lengths` (`_effective_row_lengths`).
    for batch_idx in range(seq_batch_size):
      bq = bq_vec[batch_idx].reshape(-1, head_dim)
      bkv = bkv_vec[batch_idx]
      scale_val = scale_val_vec[batch_idx]
      bq_weights = bq_weights_vec[batch_idx]

      s_summed = reduce_heads(bq, bkv, bq_weights, head_major=q_is_head_major)
      s_summed = s_summed * scale_val
      ret.append(s_summed.reshape(-1, num_sublanes_bkv, 128))
    return jnp.concatenate(ret, axis=0)

  r_raw = seq_batch_size * bq_sz
  r_pad = align_to(r_raw, 8)
  if not fused_return_scores and r_pad >= 32 and r_pad % 32 == 0:
    r_blk = 32
    grp_tiles = 4
  else:
    r_blk = 16 if (r_pad >= 16 and r_pad % 16 == 0) else 8
    grp_tiles = 8
  num_s_blks = r_pad // r_blk
  compact_grp = 4
  num_m_pad = align_to(num_sublanes_total, grp_tiles)
  k_aligned = align_to(fused_topk_k, 128) if fused_topk else 128
  num_k_tiles = k_aligned // 128

  def _monotone_key(bits):
    sign_mask = lax.shift_right_arithmetic(bits, jnp.int32(31))
    return lax.bitwise_xor(
        bits, lax.bitwise_and(sign_mask, jnp.int32(0x7FFFFFFF))
    )

  def _row_sum_bcast(x_i32):
    return jnp.broadcast_to(
        jnp.sum(x_i32.astype(jnp.float32), axis=1, keepdims=True),
        (r_blk, 128),
    ).astype(jnp.int32)

  def _fused_epilogue(
      p_id,
      bkv_idx,
      row_scores_vmem,
      out_ids_vmem,
      out_scores_vmem,
      out_sem,
  ):
    assert fused_topk_k is not None
    total_steps = metadata_ref.num_steps[chunk_idx]
    abs_step = step_offset + p_id
    next_abs = jnp.minimum(abs_step + 1, metadata_ref.bkv_idx.shape[0] - 1)
    is_last_bkv = (p_id + 1 >= total_steps) | (
        metadata_ref.bkv_idx[next_abs] == 0
    )

    @pl.when(is_last_bkv)
    def _():
      active_tiles = jnp.minimum(
          (bkv_idx + 1) * num_sublanes_bkv, num_sublanes_total
      )
      active_groups = (active_tiles + (grp_tiles - 1)) // grp_tiles
      compact_groups = (active_tiles + (compact_grp - 1)) // compact_grp
      start_seq = start_end_seq_idx_ref[0]
      end_seq = start_end_seq_idx_ref[1]
      chunk_start = start_end_seq_idx_ref[2]
      seq_idx = seq_tile(p_id)
      bq_idx = bq_block(p_id)
      cp_rank_val = cp_rank_ref[0]
      shift_comp = (compression_ratio - 1).bit_length()
      num_positions = max(num_sublanes_total * 128, fused_topk_k)

      s_len = jnp.int32(0)
      q_s = jnp.int32(0)
      q_l = jnp.int32(0)
      token_start = jnp.int32(0)
      if seq_batch_size > 1:
        lo = jnp.maximum(cu_q_lens_ref[seq_idx], chunk_start)
        dst_start = lo - chunk_start
        n_valid_seqs = jnp.maximum(
            0, jnp.minimum(seq_batch_size, end_seq - seq_idx)
        )
        total_sz = jnp.minimum(
            n_valid_seqs, jnp.maximum(0, chunk_tokens - dst_start)
        )
      else:
        seq_valid = (seq_idx >= start_seq) & (seq_idx < end_seq)
        seq_safe = jnp.minimum(seq_idx, max_num_seqs - 1)
        s_len = seq_lens_ref[seq_safe]
        q_s = cu_q_lens_ref[seq_safe]
        q_e = cu_q_lens_ref[seq_safe + 1]
        q_l = q_e - q_s
        lo = jnp.maximum(q_s, chunk_start)
        hi = jnp.minimum(q_e, chunk_start + chunk_tokens)
        token_start = lo + bq_idx * bq_sz
        dst_start = token_start - chunk_start
        total_sz = jnp.where(
            seq_valid,
            jnp.maximum(0, jnp.minimum(bq_sz, hi - token_start)),
            0,
        )

      lanes_2d = lax.broadcasted_iota(jnp.int32, (r_blk, 128), 1)
      upper_128 = (
          lax.broadcasted_iota(jnp.int32, (128, 128), 0)
          <= lax.broadcasted_iota(jnp.int32, (128, 128), 1)
      ).astype(jnp.bfloat16)
      key_neg_inf = jnp.int32(BITS_NEG_INF) ^ jnp.int32(0x7FFFFFFF)
      int32_min = jnp.int32(-0x80000000)

      for s_idx in range(num_s_blks):
        r0 = s_idx * r_blk
        if seq_batch_size > 1:
          # Decode passes hold one row per sequence, so row `r` of block
          # `s_idx` is sequence `seq_idx + r0 + r`. Blocks past the first exist
          # once `seq_batch_size > r_blk` (e.g. 24, or 32 with scores).
          r_iota_8 = r0 + lax.broadcasted_iota(jnp.int32, (r_blk, 128), 0)
          row_len_vec = jnp.zeros((r_blk, 128), dtype=jnp.int32)
          for b in range(r0, min(r0 + r_blk, seq_batch_size)):
            s = seq_idx + b
            s_in_pass = (s >= start_seq) & (s < end_seq)
            s_safe = jnp.minimum(s, max_num_seqs - 1)
            sb_len = seq_lens_ref[s_safe]
            qb_s = cu_q_lens_ref[s_safe]
            qb_e = cu_q_lens_ref[s_safe + 1]
            qb_l = qb_e - qb_s
            in_chunk = (
                (qb_s >= chunk_start)
                & (qb_s < chunk_start + chunk_tokens)
                & (qb_l > 0)
            )
            tok_pos = sb_len - qb_l
            kv_len = sb_len >> shift_comp
            vis = jnp.minimum(kv_len, (tok_pos >> shift_comp) + 1)
            if cp_size > 1:
              vis = cp_local_length(vis, cp_rank_val, cp_size, interleave_c)
            rl_b = jnp.where(
                s_in_pass & in_chunk, jnp.minimum(vis, num_positions), 0
            )
            row_len_vec = jnp.where(r_iota_8 == b, rl_b, row_len_vec)
        else:
          r_ids_2d = r0 + lax.broadcasted_iota(jnp.int32, (r_blk, 128), 0)
          tok_id_2d = token_start + r_ids_2d
          tok_pos_2d = s_len - q_l + (tok_id_2d - q_s)
          kv_len = s_len >> shift_comp
          vis_2d = jnp.minimum(kv_len, (tok_pos_2d >> shift_comp) + 1)
          if cp_size > 1:
            vis_2d = cp_local_length(vis_2d, cp_rank_val, cp_size, interleave_c)
          row_len_vec = jnp.where(
              r_ids_2d < total_sz, jnp.minimum(vis_2d, num_positions), 0
          )

        # Pass 1: grp_tiles contiguous monotone key conversion + causal/pad mask
        def _prep_group(g, elig_acc, s_val=s_idx, rl_val=row_len_vec):
          raw_g = row_scores_vmem[s_val, pl.ds(g * grp_tiles, grp_tiles), :, :]
          prepped = []
          for u in range(grp_tiles):
            m = g * grp_tiles + u
            cols = m * 128 + lanes_2d
            skey = _monotone_key(raw_g[u])
            elig = (m < active_tiles) & (cols < rl_val) & (skey > key_neg_inf)
            skey_m = jnp.where(elig, skey, int32_min)
            elig_acc = elig_acc + jnp.where(elig, jnp.int32(1), jnp.int32(0))
            prepped.append(skey_m)
          row_scores_vmem[s_val, pl.ds(g * grp_tiles, grp_tiles), :, :] = (
              jnp.stack(prepped, axis=0)
          )
          return elig_acc

        elig_lanes = lax.fori_loop(
            0,
            active_groups,
            _prep_group,
            jnp.zeros((r_blk, 128), dtype=jnp.int32),
        )
        quota_vec = jnp.minimum(
            _row_sum_bcast(elig_lanes), jnp.int32(fused_topk_k)
        )

        # Pass 2: 2-bit quaternary signed-int32 bisection (15 coarse + 1 final)
        def _bisect_quat(p, cur_base, s_val=s_idx, q_val=quota_vec):
          step_sz = lax.shift_left(jnp.int32(1), jnp.int32(30 - 2 * p))
          t1 = cur_base + step_sz
          t2 = t1 + step_sz
          t3 = t2 + step_sz

          def _cnt_grp(g, accs):
            a1, a2, a3 = accs
            sg = row_scores_vmem[s_val, pl.ds(g * grp_tiles, grp_tiles), :, :]
            for u in range(grp_tiles):
              sk = sg[u]
              a1 = a1 + jnp.where(sk >= t1, jnp.int32(1), jnp.int32(0))
              a2 = a2 + jnp.where(sk >= t2, jnp.int32(1), jnp.int32(0))
              a3 = a3 + jnp.where(sk >= t3, jnp.int32(1), jnp.int32(0))
            return (a1, a2, a3)

          z0 = jnp.zeros((r_blk, 128), dtype=jnp.int32)
          acc1, acc2, acc3 = lax.fori_loop(
              0, active_groups, _cnt_grp, (z0, z0, z0)
          )
          c1 = _row_sum_bcast(acc1)
          c2 = _row_sum_bcast(acc2)
          c3 = _row_sum_bcast(acc3)
          return jnp.where(
              c3 >= q_val,
              t3,
              jnp.where(c2 >= q_val, t2, jnp.where(c1 >= q_val, t1, cur_base)),
          )

        base_15 = lax.fori_loop(
            0,
            15,
            _bisect_quat,
            jnp.full((r_blk, 128), int32_min, dtype=jnp.int32),
        )

        # Final 2-bit step (step_sz = 1) fused with strictly-above count
        t1_f = base_15 + jnp.int32(1)
        t2_f = base_15 + jnp.int32(2)
        t3_f = base_15 + jnp.int32(3)

        def _cnt_final_grp(
            g,
            accs,
            s_val=s_idx,
            t1_val=t1_f,
            t2_val=t2_f,
            t3_val=t3_f,
        ):
          a1, a2, a3, agt3 = accs
          sg = row_scores_vmem[s_val, pl.ds(g * grp_tiles, grp_tiles), :, :]
          for u in range(grp_tiles):
            sk = sg[u]
            a1 = a1 + jnp.where(sk >= t1_val, jnp.int32(1), jnp.int32(0))
            a2 = a2 + jnp.where(sk >= t2_val, jnp.int32(1), jnp.int32(0))
            a3 = a3 + jnp.where(sk >= t3_val, jnp.int32(1), jnp.int32(0))
            agt3 = agt3 + jnp.where(sk > t3_val, jnp.int32(1), jnp.int32(0))
          return (a1, a2, a3, agt3)

        zf = jnp.zeros((r_blk, 128), dtype=jnp.int32)
        fa1, fa2, fa3, fagt3 = lax.fori_loop(
            0, active_groups, _cnt_final_grp, (zf, zf, zf, zf)
        )
        fc1 = _row_sum_bcast(fa1)
        fc2 = _row_sum_bcast(fa2)
        fc3 = _row_sum_bcast(fa3)
        fc_gt3 = _row_sum_bcast(fagt3)
        thresh_vec = jnp.where(
            fc3 >= quota_vec,
            t3_f,
            jnp.where(
                fc2 >= quota_vec,
                t2_f,
                jnp.where(fc1 >= quota_vec, t1_f, base_15),
            ),
        )
        above_cnt = jnp.where(
            fc3 >= quota_vec,
            fc_gt3,
            jnp.where(
                fc2 >= quota_vec,
                fc3,
                jnp.where(fc1 >= quota_vec, fc2, fc1),
            ),
        )
        tie_quota_vec = jnp.where(
            quota_vec > 0, quota_vec - above_cnt, jnp.int32(0)
        )

        # Pass 3: 4-tile unrolled lowest-column tie resolution + roll compaction
        init_ids = tuple(
            jnp.full((r_blk, 128), jnp.int32(-1), dtype=jnp.int32)
            for _ in range(num_k_tiles)
        )
        if fused_return_scores:
          init_sc = tuple(
              jnp.full((r_blk, 128), jnp.int32(BITS_NEG_INF), dtype=jnp.int32)
              for _ in range(num_k_tiles)
          )
        else:
          init_sc = ()

        def _compact_grp(
            cg,
            carry,
            s_val=s_idx,
            th_val=thresh_vec,
            q_val=quota_vec,
            tq_val=tie_quota_vec,
        ):
          written_cnt, ties_used, cur_out_ids, cur_out_sc = carry
          sg4 = row_scores_vmem[
              s_val, pl.ds(cg * compact_grp, compact_grp), :, :
          ]
          out_ids_list = list(cur_out_ids)
          out_sc_list = list(cur_out_sc)
          for u in range(compact_grp):
            m = cg * compact_grp + u
            skey = sg4[u]
            above_m = (q_val > 0) & (skey > th_val)
            tied_m = (q_val > 0) & (skey == th_val) & (skey > key_neg_inf)
            tie_rank = lax.dot_general(
                jnp.where(tied_m, jnp.bfloat16(1.0), jnp.bfloat16(0.0)),
                upper_128,
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(jnp.int32)
            won_tied = tied_m & (tie_rank <= (tq_val - ties_used))
            won_m = above_m | won_tied

            n_tied = _row_sum_bcast(
                jnp.where(won_tied, jnp.int32(1), jnp.int32(0))
            )
            win_rank = lax.dot_general(
                jnp.where(won_m, jnp.bfloat16(1.0), jnp.bfloat16(0.0)),
                upper_128,
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(jnp.int32)
            n_won = _row_sum_bcast(jnp.where(won_m, jnp.int32(1), jnp.int32(0)))

            rem_delta = jnp.where(
                won_m, lanes_2d - (win_rank - 1), jnp.int32(0)
            )
            cur_col = m * 128 + lanes_2d
            pkt = jnp.where(
                won_m,
                lax.bitwise_or(
                    jnp.int32(1 << 30),
                    lax.bitwise_or(
                        lax.shift_left(rem_delta, jnp.int32(23)), cur_col
                    ),
                ),
                jnp.int32(0),
            )
            cur_bits = _monotone_key(skey) if fused_return_scores else None
            for s in range(7):
              d = 1 << s
              move_bit = jnp.int32((1 << 30) | (d << 23))
              is_move = lax.bitwise_and(pkt, move_bit) == move_bit
              stay_pkt = jnp.where(is_move, jnp.int32(0), pkt)
              rolled_pkt = pltpu.roll(
                  jnp.where(is_move, pkt, jnp.int32(0)), 128 - d, axis=1
              )
              arrived = (rolled_pkt != 0) & (lanes_2d < (128 - d))
              pkt = jnp.where(arrived, rolled_pkt, stay_pkt)
              if cur_bits is not None:
                cur_bits = jnp.where(
                    arrived, pltpu.roll(cur_bits, 128 - d, axis=1), cur_bits
                )

            cur_col = lax.bitwise_and(pkt, jnp.int32(0x7FFFFF))
            rot = lax.bitwise_and(written_cnt, jnp.int32(127))
            for s in range(7):
              d = 1 << s
              do_rot = lax.bitwise_and(rot, jnp.int32(d)) != 0
              cur_col = jnp.where(
                  do_rot, pltpu.roll(cur_col, d, axis=1), cur_col
              )
              if cur_bits is not None:
                cur_bits = jnp.where(
                    do_rot, pltpu.roll(cur_bits, d, axis=1), cur_bits
                )

            offset_s0 = lax.bitwise_and(lanes_2d - rot, jnp.int32(127))
            valid_rot = offset_s0 < n_won
            slot_q = lax.shift_right_arithmetic(
                written_cnt + offset_s0, jnp.int32(7)
            )
            for q in range(num_k_tiles):
              wq = valid_rot & (slot_q == q)
              out_ids_list[q] = jnp.where(wq, cur_col, out_ids_list[q])
              if cur_bits is not None:
                out_sc_list[q] = jnp.where(wq, cur_bits, out_sc_list[q])

            written_cnt = written_cnt + n_won
            ties_used = ties_used + n_tied

          return (
              written_cnt,
              ties_used,
              tuple(out_ids_list),
              tuple(out_sc_list),
          )

        _, _, final_ids, final_sc = lax.fori_loop(
            0,
            compact_groups,
            _compact_grp,
            (
                jnp.zeros((r_blk, 128), dtype=jnp.int32),
                jnp.zeros((r_blk, 128), dtype=jnp.int32),
                init_ids,
                init_sc,
            ),
        )
        out_ids_vmem[pl.ds(r0, r_blk)] = jnp.stack(final_ids, axis=1)
        if fused_return_scores and out_scores_vmem is not None:
          out_scores_vmem[pl.ds(r0, r_blk)] = jnp.stack(final_sc, axis=1)

      @pl.when(total_sz > 0)
      def _():
        assert topk_ids_out_ref is not None
        id_copy = pltpu.make_async_copy(
            out_ids_vmem.at[pl.ds(0, total_sz)],
            topk_ids_out_ref.at[pl.ds(dst_start, total_sz)],
            out_sem,
        )
        id_copy.start()
        id_copy.wait()
        if (
            fused_return_scores
            and topk_scores_out_ref is not None
            and out_scores_vmem is not None
        ):
          sc_copy = pltpu.make_async_copy(
              out_scores_vmem.at[pl.ds(0, total_sz)],
              topk_scores_out_ref.at[pl.ds(dst_start, total_sz)],
              out_sem,
          )
          sc_copy.start()
          sc_copy.wait()

  def step_body(
      step,
      q_vmem,
      weights_vmem,
      bkv_vmem,
      scores_vmem: Any = None,
      q_hm_ref: Any = None,
      row_scores_vmem: Any = None,
      out_ids_vmem: Any = None,
      out_scores_vmem: Any = None,
      out_sem: Any = None,
  ):
    p_id = step.index[0]
    bkv_idx = bkv_block(p_id)

    if q_hm_ref is not None:
      @pl.when(bkv_idx == 0)
      def _():
        q_hm_ref[...] = transpose_to_head_major(
            pad_tokens(q_vmem[...], bq_sz), hm_tok
        )

      bq_vec = [q_hm_ref[...]]
    else:
      q_data = q_vmem[...].reshape(seq_batch_size, -1, head_dim)
      bq_vec = [q_data[batch_idx] for batch_idx in range(seq_batch_size)]
    w_data = weights_vmem[...].reshape(seq_batch_size, -1, num_q_heads)
    bq_weights_vec = [w_data[batch_idx] for batch_idx in range(seq_batch_size)]

    bkvs = []
    bkv_scales = []
    rows = seq_along_lane_block_rows(bkv_vmem) if seq_along_lane else None
    lane_tiles = range(num_lane_tiles_bkv)
    for batch_idx in range(seq_batch_size):
      if seq_along_lane:
        fp8_val = jnp.concatenate(
            [
                seq_along_lane_key_tile(rows, batch_idx, lane_tile_idx)
                for lane_tile_idx in lane_tiles
            ],
            axis=1,
        )
        scale_val = jnp.concatenate(
            [
                seq_along_lane_scale_tile(rows, batch_idx, lane_tile_idx)
                for lane_tile_idx in lane_tiles
            ],
            axis=1,
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
        q_is_head_major=q_hm_ref is not None,
    )
    if fused_topk:
      for jj in range(num_sublanes_bkv):
        s_tile = pltpu.bitcast(scores[:, jj, :], jnp.int32)
        if r_raw < r_pad:
          s_tile = jnp.pad(
              s_tile,
              ((0, r_pad - r_raw), (0, 0)),
              constant_values=BITS_NEG_INF,
          )
        safe_m = jnp.minimum(
            bkv_idx * num_sublanes_bkv + jj, num_sublanes_total - 1
        )
        for s_i in range(num_s_blks):
          row_scores_vmem[s_i, safe_m, :, :] = s_tile[
              s_i * r_blk : (s_i + 1) * r_blk
          ]
      _fused_epilogue(
          p_id,
          bkv_idx,
          row_scores_vmem,
          out_ids_vmem,
          out_scores_vmem,
          out_sem,
      )
    else:
      scores_vmem[...] = pltpu.bitcast(scores, jnp.int32)

  def blocked_prefill_step(
      step,
      q_vmem,
      weights_vmem,
      bkv_vmem,
      scores_vmem: Any = None,
      q_hm_ref: Any = None,
      wb_ref: Any = None,
      row_scores_vmem: Any = None,
      out_ids_vmem: Any = None,
      out_scores_vmem: Any = None,
      out_sem: Any = None,
  ):
    """Scores one (q block, kv block) step of the multi-token path.

    Args:
      step: Pipeline step.
      q_vmem: [bq, H, D] query block.
      weights_vmem: [bq, H] indexer weights.
      bkv_vmem: the packed key block of this step.
      scores_vmem: [bq, score_pitch, 128] i32 output block, of which the first
        nsub rows of each token are copied out.
      q_hm_ref: [bq * H, D] cache of the q block in (head group, token block,
        head, token) row order.
      wb_ref: [num_tok_blks, H, tok_blk, 128] cache of the indexer weights,
        broadcast along lanes; bf16 for FP8 queries, f32 otherwise.
      row_scores_vmem: Optional VMEM scratch holding row scores for fused top-k.
      out_ids_vmem: Optional VMEM output staging buffer for top-k indices.
      out_scores_vmem: Optional VMEM output staging buffer for top-k scores.
      out_sem: Optional DMA semaphore for writing fused top-k outputs to HBM.
    """
    p_id = step.index[0]
    bkv_idx = bkv_block(p_id)
    tok_blk = prefill_cfg.tok_blk
    num_tok_blks = prefill_cfg.num_tok_blks
    num_head_groups = prefill_cfg.num_head_groups
    heads_per_group = prefill_cfg.heads_per_group
    rows_per_group_tok_blk = prefill_cfg.rows_per_group_tok_blk
    rows_per_group = prefill_cfg.rows_per_group
    lane_chunk = prefill_cfg.lane_chunk
    num_lane_chunks = prefill_cfg.num_lane_chunks
    tiles_per_chunk = prefill_cfg.tiles_per_chunk
    score_pitch = prefill_cfg.score_pitch

    # The caches depend only on the q block, but the grid iterates kv
    # innermost (`bkv_idx = step % num_bkv`), so bq_idx is held fixed for a
    # whole run of steps. Rebuilding them once per q block instead of once per
    # step avoids redoing identical work num_bkv times; `bkv_idx == 0` marks
    # exactly the first step of each new q block.
    @pl.when(bkv_idx == 0)
    def _():
      w = weights_vmem[...].astype(jnp.float32)
      for tb in range(num_tok_blks):
        t0 = tb * tok_blk
        q_rows = transpose_to_head_major(q_vmem[pl.ds(t0, tok_blk)], tok_blk)
        for h in range(num_q_heads):
          q_hm_ref[pl.ds(prefill_cfg.cache_row(h, tb), tok_blk)] = q_rows[
              h * tok_blk : (h + 1) * tok_blk
          ]
          wb_ref[tb, h] = jnp.broadcast_to(
              w[t0 : t0 + tok_blk, h : h + 1], (tok_blk, 128)
          ).astype(wb_ref.dtype)

    # Positions a token cannot see are left unmasked. The SparseCore top-k
    # reads each score row only up to its `row_lengths` entry, which is
    # exactly the causal and end-of-sequence limit (`_effective_row_lengths`);
    # that same bound already keeps it off the KV blocks the schedule skips,
    # so masking here would only cost a pass over every diagonal block.

    if seq_along_lane:
      rows = seq_along_lane_block_rows(bkv_vmem)

      def load_chunk(c):
        tiles = range(c * tiles_per_chunk, (c + 1) * tiles_per_chunk)
        keys = jnp.concatenate(
            [
                seq_along_lane_key_tile(rows, 0, lane_tile_idx)
                for lane_tile_idx in tiles
            ],
            1,
        )
        scales = jnp.concatenate(
            [
                seq_along_lane_scale_tile(rows, 0, lane_tile_idx)
                for lane_tile_idx in tiles
            ],
            1,
        )
        return keys, scales

      def qk(q_rows, keys):
        # keys is [head_dim, lane_chunk]: the MXU-native contraction layout.
        return jnp.einsum(
            "nd,dm->nm", q_rows, keys, preferred_element_type=jnp.float32
        )

    else:
      groups_per_chunk = lane_chunk // kv_packing

      def load_chunk(c):
        blk = bkv_vmem[0, pl.ds(c * groups_per_chunk, groups_per_chunk)]
        flat = blk.reshape(lane_chunk, blk.shape[-1])
        keys = pltpu.bitcast(flat[:, :head_dim], jnp.float8_e4m3fn)
        scales = pltpu.bitcast(
            flat[:, head_dim : head_dim + 1].T, jnp.float8_e8m0fnu
        ).astype(jnp.bfloat16)
        return keys, scales.astype(jnp.float32)

      def qk(q_rows, keys):
        # keys is [lane_chunk, head_dim].
        return jnp.einsum(
            "nd,md->nm", q_rows, keys, preferred_element_type=jnp.float32
        )

    def reduce_token_block(sts, tb):
      """sum_h relu(st_h) * w_h over the rows of token block `tb`.

      Args:
        sts: per head group, the [rows_per_group, lane_chunk] q @ k block.
        tb: token block index.

      Returns:
        The [tok_blk, lane_chunk] accumulated score tile for token block `tb`.
      """
      acc = None
      for i in range(heads_per_group):
        # Consecutive heads come from different groups, and so from different
        # MXUs, so the accumulation drains every MXU at the rate it produces
        # results.
        for g, st in enumerate(sts):
          r0 = tb * rows_per_group_tok_blk + i * tok_blk
          st_h = st[r0 : r0 + tok_blk]
          w_h = wb_ref[tb, i * num_head_groups + g]
          if tiles_per_chunk > 1:
            w_h = jnp.concatenate([w_h] * tiles_per_chunk, axis=1)
          if prefill_cfg.bf16_reduce:
            # Rounding before the relu is the same as rounding after it. The
            # chain has to start from a product and add one unpacked bf16 x
            # bf16 product per head for the compiler to fuse each packed pair
            # of links into a single multiply-accumulate.
            r = jnp.maximum(st_h.astype(jnp.bfloat16), 0)
            term = r.astype(jnp.float32) * w_h.astype(jnp.float32)
          else:
            term = jnp.maximum(st_h, 0.0) * w_h
          acc = term if acc is None else acc + term
      return acc

    if fused_topk:
      sub_per_tb = tok_blk // r_blk
      for c in range(num_lane_chunks):
        keys, scales = load_chunk(c)
        sts = [
            qk(q_hm_ref[pl.ds(g * rows_per_group, rows_per_group)], keys)
            for g in range(num_head_groups)
        ]
        for tb in range(num_tok_blks):
          acc = reduce_token_block(sts, tb) * scales
          for jj in range(tiles_per_chunk):
            lane_tile = c * tiles_per_chunk + jj
            safe_m = jnp.minimum(
                bkv_idx * num_sublanes_bkv + lane_tile,
                num_sublanes_total - 1,
            )
            tile_i32 = pltpu.bitcast(
                acc[:, jj * 128 : (jj + 1) * 128], jnp.int32
            )
            for sub in range(sub_per_tb):
              row_scores_vmem[tb * sub_per_tb + sub, safe_m, :, :] = tile_i32[
                  sub * r_blk : (sub + 1) * r_blk
              ]
      _fused_epilogue(
          p_id,
          bkv_idx,
          row_scores_vmem,
          out_ids_vmem,
          out_scores_vmem,
          out_sem,
      )
      return

    # Row `t * score_pitch + j` holds lane tile j of token t.
    scores_rows = (
        scores_vmem.reshape(bq_sz * score_pitch, 128)
        if prefill_cfg.strided_score_store
        else None
    )
    chunk_scores = []
    for c in range(num_lane_chunks):
      keys, scales = load_chunk(c)
      # Reload the cached rows for every chunk instead of keeping the whole q
      # block live in registers across chunks. Each head group is its own
      # matmul so that the groups can run on different MXUs.
      sts = [
          qk(q_hm_ref[pl.ds(g * rows_per_group, rows_per_group)], keys)
          for g in range(num_head_groups)
      ]
      tb_scores = []
      for tb in range(num_tok_blks):
        acc = reduce_token_block(sts, tb) * scales
        if scores_rows is None:
          tb_scores.append(acc)
          continue
        for jj in range(tiles_per_chunk):
          lane_tile = c * tiles_per_chunk + jj
          scores_rows[
              pl.ds(
                  tb * tok_blk * score_pitch + lane_tile,
                  tok_blk,
                  stride=score_pitch,
              )
          ] = pltpu.bitcast(acc[:, jj * 128 : (jj + 1) * 128], jnp.int32)
      if scores_rows is None:
        chunk_scores.append(jnp.concatenate(tb_scores, axis=0))

    if scores_rows is None:
      s_summed = jnp.concatenate(chunk_scores, axis=1)
      scores_vmem[...] = pltpu.bitcast(
          s_summed.reshape(bq_sz, num_sublanes_bkv, 128), jnp.int32
      )

  def _run_all(
      q_hm_ref=None,
      wb_ref=None,
      row_scores_vmem=None,
      out_ids_vmem=None,
      out_scores_vmem=None,
      out_sem=None,
  ):
    if fused_topk:
      if prefill_cfg.enabled:
        body = functools.partial(
            blocked_prefill_step,
            scores_vmem=None,
            q_hm_ref=q_hm_ref,
            wb_ref=wb_ref,
            row_scores_vmem=row_scores_vmem,
            out_ids_vmem=out_ids_vmem,
            out_scores_vmem=out_scores_vmem,
            out_sem=out_sem,
        )
      else:
        body = functools.partial(
            step_body,
            scores_vmem=None,
            q_hm_ref=q_hm_ref,
            row_scores_vmem=row_scores_vmem,
            out_ids_vmem=out_ids_vmem,
            out_scores_vmem=out_scores_vmem,
            out_sem=out_sem,
        )

      @pl.with_scoped(final_allocs=(q_alloc, weights_alloc, kv_alloc))
      def _run_fused_pipeline(final_allocs):
        q_buf, weights_buf, kv_buf = final_allocs
        pipeline_fn = pltpu.emit_pipeline(
            body,
            grid=(metadata_ref.num_steps[chunk_idx],),
            in_specs=(q_buf.spec, weights_buf.spec, kv_buf.spec),
            out_specs=(),
            _explicit_indices=True,
        )

        window_refs = (cu_q_lens_ref, metadata_ref, start_end_seq_idx_ref)

        @pl.when(metadata_ref.num_steps[chunk_idx] > 0)
        def _():
          pipeline_fn(
              (q_hbm_ref, *window_refs),
              (indexer_weights_hbm_ref, *window_refs),
              (cache_kv_hbm_ref, page_indices_ref, metadata_ref),
              allocations=(q_buf, weights_buf, kv_buf),
          )

      _run_fused_pipeline()
      return

    if prefill_cfg.enabled:
      body = functools.partial(
          blocked_prefill_step, q_hm_ref=q_hm_ref, wb_ref=wb_ref
      )
    else:
      body = functools.partial(step_body, q_hm_ref=q_hm_ref)

    @pl.with_scoped(final_allocs=(q_alloc, weights_alloc, kv_alloc, o_alloc))
    def _run_pipeline(final_allocs):
      q_buf, weights_buf, kv_buf, o_buf = final_allocs
      pipeline_fn = pltpu.emit_pipeline(
          body,
          grid=(metadata_ref.num_steps[chunk_idx],),
          in_specs=(q_buf.spec, weights_buf.spec, kv_buf.spec),
          out_specs=o_buf.spec,
          _explicit_indices=True,
      )

      window_refs = (cu_q_lens_ref, metadata_ref, start_end_seq_idx_ref)
      @pl.when(metadata_ref.num_steps[chunk_idx] > 0)
      def _():
        pipeline_fn(
            (q_hbm_ref, *window_refs),
            (indexer_weights_hbm_ref, *window_refs),
            (cache_kv_hbm_ref, page_indices_ref, metadata_ref),
            (scores_hbm_ref, *window_refs),
            allocations=(q_buf, weights_buf, kv_buf, o_buf),
        )

    _run_pipeline()

  if fused_topk:
    score_scratch = (
        (pltpu.VMEM((r_pad, num_k_tiles, 128), jnp.int32),)
        if fused_return_scores
        else ()
    )
    fused_scratches = (
        pltpu.VMEM((num_s_blks, num_m_pad, r_blk, 128), jnp.int32),
        pltpu.VMEM((r_pad, num_k_tiles, 128), jnp.int32),
        *score_scratch,
        pltpu.SemaphoreType.DMA,
    )
  else:
    fused_scratches = ()

  def _dispatch_scoped(*scratches):
    if fused_topk:
      if fused_return_scores:
        *q_scratches, rs_vmem, oi_vmem, os_vmem, o_sem = scratches
      else:
        *q_scratches, rs_vmem, oi_vmem, o_sem = scratches
        os_vmem = None
    else:
      q_scratches = scratches
      rs_vmem = oi_vmem = os_vmem = o_sem = None
    q_hm = q_scratches[0] if len(q_scratches) >= 1 else None
    wb = q_scratches[1] if len(q_scratches) >= 2 else None
    _run_all(
        q_hm_ref=q_hm,
        wb_ref=wb,
        row_scores_vmem=rs_vmem,
        out_ids_vmem=oi_vmem,
        out_scores_vmem=os_vmem,
        out_sem=o_sem,
    )

  # Only the multi-token path reorders q, and only a single sequence per tile
  # keeps the cached block unambiguous.
  q_scratches_spec: tuple[Any, ...] = ()
  if prefill_cfg.enabled:
    q_scratches_spec = (
        pltpu.VMEM((num_q_heads * bq_sz, head_dim), q_hbm_ref.dtype),
        pltpu.VMEM(
            (
                prefill_cfg.num_tok_blks,
                num_q_heads,
                prefill_cfg.tok_blk,
                128,
            ),
            prefill_cfg.wb_dtype,
        ),
    )
  elif seq_batch_size == 1 and bq_sz > 1:
    q_scratches_spec = (
        pltpu.VMEM((num_q_heads * hm_tok, head_dim), q_hbm_ref.dtype),
    )

  all_scratches = (*q_scratches_spec, *fused_scratches)
  if all_scratches:
    pl.run_scoped(_dispatch_scoped, *all_scratches)
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
  if num_q_heads != actual_num_q_heads or head_dim != actual_head_dim:
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
  if num_q_heads != actual_num_q_heads:
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
  if max_num_seqs == 1:
    q_len = cu_q_lens[1]
    seq_len = seq_lens[0]
    token_pos = (seq_len - q_len) + token_ids
    kv_len = seq_len // compression_ratio
    visible = jnp.minimum(kv_len, token_pos // compression_ratio + 1)
    if cp_size > 1:
      visible = cp_local_length(visible, cp_rank, cp_size, interleave_c)
    valid = token_ids < jnp.where(num_seqs > 0, q_len, 0)
    return jnp.where(valid, jnp.minimum(visible, num_positions), 0)
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


_TC_TOPK_PREFIX_BUCKETS: tuple[int, ...] = (
    1024,
    2048,
    4096,
    16384,
    32768,
    65536,
)
_TC_TOPK_TOURNAMENT_LEAF_WIDTH = 4096
_TC_TOPK_VMEM_SUB_BATCH = 32


def _f32_to_desc_key(x: jax.Array) -> jax.Array:
  """Maps f32 values to int32 keys ordered descending (+0.0 > -0.0)."""
  bits = jax.lax.bitcast_convert_type(x, jnp.int32)
  sign_mask = jax.lax.shift_right_arithmetic(bits, jnp.int32(31))
  return ~((bits & jnp.int32(0x7FFFFFFF)) ^ sign_mask)


def _desc_key_to_f32(desc_key: jax.Array) -> jax.Array:
  """Inverts `_f32_to_desc_key` back to exact IEEE float32 values."""
  u = ~desc_key
  sign_mask = jax.lax.shift_right_arithmetic(u, jnp.int32(31))
  bits = u ^ (sign_mask & jnp.int32(0x7FFFFFFF))
  return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _tournament_topk(
    scores: jax.Array,
    k: int,
    leaf_width: int = _TC_TOPK_TOURNAMENT_LEAF_WIDTH,
) -> tuple[jax.Array, jax.Array]:
  """Exact tournament top-k via leaf `lax.top_k` and 2-key sort."""
  batch_rows, width = scores.shape
  pad_w = (-width) % leaf_width
  if pad_w > 0:
    scores = jnp.pad(scores, ((0, 0), (0, pad_w)), constant_values=-jnp.inf)
  num_blocks = scores.shape[1] // leaf_width
  leaf_vals, leaf_local = jax.lax.top_k(
      scores.reshape(batch_rows * num_blocks, leaf_width), k
  )
  offsets = (jnp.arange(num_blocks, dtype=jnp.int32) * leaf_width)[
      None, :, None
  ]
  cand_vals = leaf_vals.reshape(batch_rows, num_blocks * k)
  cand_ids = (
      leaf_local.reshape(batch_rows, num_blocks, k).astype(jnp.int32) + offsets
  ).reshape(batch_rows, num_blocks * k)

  cand_desc = _f32_to_desc_key(cand_vals)
  sorted_desc, sorted_ids = jax.lax.sort(
      (cand_desc, cand_ids),
      dimension=1,
      is_stable=False,
      num_keys=2,
  )
  return _desc_key_to_f32(sorted_desc[:, :k]), sorted_ids[:, :k].astype(
      jnp.int32
  )


def _exact_tournament_topk(
    scores: jax.Array,
    k: int,
    leaf_width: int = _TC_TOPK_TOURNAMENT_LEAF_WIDTH,
) -> tuple[jax.Array, jax.Array]:
  """Exact top-k via strided k=16 VPU filter with certified fallback."""
  batch_rows, width = scores.shape
  if k >= leaf_width or width <= leaf_width:
    vals, ids = jax.lax.top_k(scores, k)
    return vals, ids.astype(jnp.int32)

  num_lanes = 128
  k_lane = 16
  if k <= 512 and width >= 8192 and width % num_lanes == 0:
    lane_len = width // num_lanes
    if lane_len >= k_lane:
      lanes = (
          scores.reshape(batch_rows, lane_len, num_lanes)
          .swapaxes(1, 2)
          .reshape(batch_rows * num_lanes, lane_len)
      )
      leaf_vals, leaf_q = jax.lax.top_k(lanes, k_lane)
      leaf_vals = leaf_vals.reshape(batch_rows, num_lanes, k_lane)
      lane_ids = jnp.arange(num_lanes, dtype=jnp.int32)[None, :, None]
      leaf_ids = (
          leaf_q.reshape(batch_rows, num_lanes, k_lane).astype(jnp.int32)
          * num_lanes
          + lane_ids
      )
      leaf_desc = _f32_to_desc_key(leaf_vals)
      drop_desc = leaf_desc[:, :, k_lane - 1]
      drop_id = leaf_ids[:, :, k_lane - 1]
      cand_desc = leaf_desc.reshape(batch_rows, num_lanes * k_lane)
      cand_ids = leaf_ids.reshape(batch_rows, num_lanes * k_lane)

      sorted_desc, sorted_ids = jax.lax.sort(
          (cand_desc, cand_ids),
          dimension=1,
          is_stable=False,
          num_keys=2,
      )
      fast_desc = sorted_desc[:, :k]
      fast_ids = sorted_ids[:, :k]
      fast_vals = _desc_key_to_f32(fast_desc)
      kth_desc = fast_desc[:, k - 1 : k]
      kth_id = fast_ids[:, k - 1 : k]
      neg_inf_desc = jnp.int32(0x7F800000)
      certified = jnp.all(
          (kth_desc < drop_desc)
          | ((kth_desc == drop_desc) & (kth_id <= drop_id))
          | (drop_desc >= neg_inf_desc)
      )
      return jax.lax.cond(
          certified,
          lambda _: (fast_vals, fast_ids),
          lambda _: _tournament_topk(scores, k, leaf_width),
          None,
      )

  return _tournament_topk(scores, k, leaf_width)


def _tensorcore_topk_select_tile(
    sliced_bits: jax.Array,
    k: int,
    row_lengths: jax.Array,
    *,
    return_scores: bool,
    use_tournament: bool = True,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Selects exact top-k columns for one sub-batch tile on TensorCore."""
  scores = jax.lax.bitcast_convert_type(sliced_bits, jnp.float32)
  cols = jnp.arange(scores.shape[1], dtype=jnp.int32)[None, :]
  scores = jnp.where(cols < row_lengths[:, None], scores, -jnp.inf)
  if scores.shape[1] < k:
    scores = jnp.pad(
        scores, ((0, 0), (0, k - scores.shape[1])), constant_values=-jnp.inf
    )
  if use_tournament:
    values, ids = _exact_tournament_topk(scores, k)
  else:
    values, ids = jax.lax.top_k(scores, k)
  valid = values > -jnp.inf
  ids = jnp.where(valid, ids.astype(jnp.int32), jnp.int32(-1))
  if not return_scores:
    return ids
  values = jnp.where(valid, values, -jnp.inf)
  return ids, jax.lax.bitcast_convert_type(values, jnp.int32)


def _tensorcore_topk_select_width(
    score_bits: jax.Array,
    k: int,
    row_lengths: jax.Array,
    width: int,
    *,
    return_scores: bool,
    use_tournament: bool = True,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Selects exact top-k over a static prefix width using VMEM sub-batching."""
  sliced_bits = score_bits[:, :width]
  batch_rows = sliced_bits.shape[0]
  sub_batch = _TC_TOPK_VMEM_SUB_BATCH
  if use_tournament and width >= 8192 and batch_rows > sub_batch:
    num_sub = batch_rows // sub_batch
    main_rows = num_sub * sub_batch
    sub_bits = sliced_bits[:main_rows].reshape(num_sub, sub_batch, width)
    sub_lens = row_lengths[:main_rows].reshape(num_sub, sub_batch)
    neg_inf_bits = jax.lax.bitcast_convert_type(
        jnp.float32(-jnp.inf), jnp.int32
    )

    def _scan_body(carry, xs):
      b_bits, b_lens = xs

      def _empty_tile(_):
        empty_ids = jnp.full((sub_batch, k), jnp.int32(-1), dtype=jnp.int32)
        if not return_scores:
          return empty_ids
        empty_scores = jnp.full((sub_batch, k), neg_inf_bits, dtype=jnp.int32)
        return empty_ids, empty_scores

      def _active_tile(_):
        return _tensorcore_topk_select_tile(
            b_bits,
            k,
            b_lens,
            return_scores=return_scores,
            use_tournament=True,
        )

      return carry, jax.lax.cond(
          jnp.max(b_lens) <= 0, _empty_tile, _active_tile, None
      )

    _, out = jax.lax.scan(_scan_body, None, (sub_bits, sub_lens))
    if not return_scores:
      assert isinstance(out, jax.Array)
      main_ids = out.reshape(main_rows, k)
      if main_rows == batch_rows:
        return main_ids
      tail_ids = _tensorcore_topk_select_tile(
          sliced_bits[main_rows:],
          k,
          row_lengths[main_rows:],
          return_scores=False,
          use_tournament=True,
      )
      assert isinstance(tail_ids, jax.Array)
      return jnp.concatenate([main_ids, tail_ids], axis=0)
    assert isinstance(out, tuple)
    out_ids, out_scores = out
    main_ids = out_ids.reshape(main_rows, k)
    main_scores = out_scores.reshape(main_rows, k)
    if main_rows == batch_rows:
      return main_ids, main_scores
    tail_out = _tensorcore_topk_select_tile(
        sliced_bits[main_rows:],
        k,
        row_lengths[main_rows:],
        return_scores=True,
        use_tournament=True,
    )
    assert isinstance(tail_out, tuple)
    tail_ids, tail_scores = tail_out
    return (
        jnp.concatenate([main_ids, tail_ids], axis=0),
        jnp.concatenate([main_scores, tail_scores], axis=0),
    )

  return _tensorcore_topk_select_tile(
      sliced_bits,
      k,
      row_lengths,
      return_scores=return_scores,
      use_tournament=use_tournament,
  )


def tensorcore_topk_select(
    score_bits: jax.Array,
    k: int,
    row_lengths: jax.Array,
    *,
    enable_prefix_bucketing: bool = True,
    return_scores: bool = True,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Selects top-k indices on TensorCore with optional prefix bucketing.

  Args:
    score_bits: `int32[batch_rows, width]` raw IEEE float32 score bits.
    k: Number of top entries to select per row.
    row_lengths: `int32[batch_rows]` valid prefix length per row.
    enable_prefix_bucketing: Whether to dispatch to the smallest prefix bucket
      covering `max(row_lengths)` with tournament selection.
    return_scores: Whether to also return `int32[batch_rows, k]` bitcast float32
      scores alongside indices.

  Returns:
    `indices` (`int32[batch_rows, k]`) or `(indices, score_bits)` if
    `return_scores` is True. Unfilled entries are `-1` with `-inf` score bits.
  """
  full_width = score_bits.shape[1]
  buckets = tuple(b for b in _TC_TOPK_PREFIX_BUCKETS if k <= b < full_width) + (
      full_width,
  )
  if not enable_prefix_bucketing:
    return _tensorcore_topk_select_width(
        score_bits,
        k,
        row_lengths,
        full_width,
        return_scores=return_scores,
        use_tournament=False,
    )
  if len(buckets) == 1:
    return _tensorcore_topk_select_width(
        score_bits,
        k,
        row_lengths,
        full_width,
        return_scores=return_scores,
        use_tournament=True,
    )
  max_len = jnp.maximum(jnp.max(row_lengths), k)
  branch_idx = jnp.sum(jnp.asarray(buckets[:-1], dtype=jnp.int32) < max_len)
  branches = [
      (
          lambda w: lambda bits, lens: _tensorcore_topk_select_width(
              bits,
              k,
              lens,
              w,
              return_scores=return_scores,
              use_tournament=True,
          )
      )(w)
      for w in buckets
  ]
  return jax.lax.switch(branch_idx, branches, score_bits, row_lengths)


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
        "uniform_q_len",
        "tc_topk",
        "topk_mode",
        "decode_only",
        "max_score_sublanes",
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
    uniform_q_len: int | None = None,
    tc_topk: bool = False,
    topk_mode: str = "sparsecore",
    decode_only: bool = False,
    max_score_sublanes: int | None = None,
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
    uniform_q_len: the query length every sequence in `[distribution[0],
      distribution[1])` has -- speculative-decoding verify windows, for
      instance. That segment is then scored by its own pass with a static q
      length and the (decode, prefill, mixed) block sizes' prefill entry,
      instead of the decode passes, which assume one query token. None keeps the
      segment on the decode passes.
    return_scores: also return the score of each selected position, which is
      what makes a cross-rank merge possible. A local top-k over a sharded KV
      axis is not the global top-k, so the merge needs the scores the same way
      the DCP attention kernels need an LSE output.
    tc_topk: whether to run scoring and top-k on TensorCore
      (`tc_streamindex_topk`) instead of SparseCore (`sparsecore_topk`).
      Defaults to False.
    topk_mode: top-k selector. "sparsecore" (default) runs the SparseCore top-k
      on the HBM score buffer. "tensorcore_full" runs `lax.top_k` over the full
      width of the HBM score buffer. "tensorcore_bounded" runs an exact
      tournament `lax.top_k` over the smallest prefix bucket that covers the
      longest row. "tensorcore_fused" selects inside the scoring kernel from
      VMEM-resident score tiles, with no HBM score buffer, and returns each
      row's ids in ascending column order; when the tiles do not fit in VMEM it
      runs "tensorcore_bounded" instead.
    decode_only: whether every sequence is a decode (`distribution[0] ==
      distribution[2]`, or `distribution[0] == 0` and `distribution[1] ==
      distribution[2]` when `uniform_q_len` is set), so only the decode passes
      (or only the `uniform_q_len` pass) run.
    max_score_sublanes: upper bound, in 128-position tiles, on the compressed
      positions any row can see. Caps the score buffer (HBM or VMEM) below the
      `pages_per_seq` extent. Must be a positive multiple of
      `num_kv_pages_per_block * page_size // 128` for every pass, and raises
      `ValueError` when `cp_size > 1`. For `cp_size == 1`, runtime metadata
      scheduling clamps `seq_lens` to `num_sublanes_total * 128` compressed
      positions (`* compression_ratio` uncompressed tokens), so positions past
      the cap are not scored. None (default) sizes the buffer from
      `pages_per_seq`.

  Returns:
    Top-K indices in this rank's local compressed space, or
    `(indices, scores)` if `return_scores`. Unfilled slots are `-1` with
    score `-inf`.
  """
  # Scale factors for the FP8 index cache format are packed directly inside
  # `cache_kv`, keeping HBM transactions fused.
  if tc_topk:
    if topk_mode != "sparsecore":
      raise ValueError(
          f"Cannot set both tc_topk=True and topk_mode={topk_mode!r}."
      )
    return tc_streamindex_topk.tc_streamindex_topk(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=k,
        compression_ratio=compression_ratio,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        buffer_count=buffer_count,
        vmem_limit_bytes=vmem_limit_bytes,
        decode_req_batch_size=decode_req_batch_size,
        enable_early_exit=enable_early_exit,
        kv_layout=kv_layout,
        chunk_tokens=chunk_tokens,
        cp_size=cp_size,
        cp_rank=cp_rank,
        interleave_size=interleave_size,
        return_scores=return_scores,
        uniform_q_len=uniform_q_len,
    )

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
  if 2 <= max_num_seqs < decode_req_batch_size:
    decode_req_batch_size = 1 << (max_num_seqs.bit_length() - 1)
  elif q.shape[1] >= 64 and max_num_seqs <= 32 and decode_req_batch_size > 4:
    decode_req_batch_size = 4

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
  if max_score_sublanes is not None:
    if cp_size > 1:
      raise ValueError("max_score_sublanes is not supported with cp_size > 1.")
    # A KV block writes `bkv_p * page_size // 128` score tiles at once. An
    # unaligned cap ends the buffer inside a block, so the fused path would
    # clamp the block's extra tiles onto the last tile and the HBM path would
    # write past the buffer.
    for bkv_p in num_kv_pages_per_blocks:
      sublanes_per_block = bkv_p * page_size // 128
      if max_score_sublanes <= 0 or max_score_sublanes % sublanes_per_block:
        raise ValueError(
            f"max_score_sublanes ({max_score_sublanes}) must be a positive"
            " multiple of num_kv_pages_per_block * page_size // 128"
            f" ({bkv_p} * {page_size} // 128 = {sublanes_per_block})."
        )
  if max_score_sublanes is not None and max_score_sublanes < num_sublanes_total:
    num_sublanes_total = max_score_sublanes

  if topk_mode not in (
      "sparsecore",
      "tensorcore_full",
      "tensorcore_bounded",
      "tensorcore_fused",
  ):
    raise ValueError(f"Unsupported topk_mode: {topk_mode}")

  try:
    vmem_cap = pltpu.get_tpu_info().vmem_capacity_bytes
  except Exception:  # pylint: disable=broad-except
    vmem_cap = vmem_limit_bytes
  if vmem_limit_bytes == DEFAULT_VMEM_LIMIT_BYTES:
    eff_vmem_limit = (
        min(vmem_cap - (16 << 20), 160 << 20)
        if vmem_cap >= (128 << 20)
        else max(48 << 20, vmem_cap - (8 << 20))
    )
  else:
    eff_vmem_limit = min(vmem_limit_bytes, max(48 << 20, vmem_cap - (8 << 20)))

  def _capped_nq(nq: int) -> int:
    if (
        topk_mode == "tensorcore_fused"
        and nq > 64
        and num_sublanes_total * align_to(nq, 8) * 128 * 4
        > int(0.45 * eff_vmem_limit)
    ):
      return 64
    return nq

  max_cand_r_pad = max(
      align_to(decode_req_batch_size, 8),
      *(align_to(_capped_nq(nq), 8) for nq in num_queries_per_blocks),
  )
  use_fused_vmem_topk = (
      topk_mode == "tensorcore_fused"
      and num_sublanes_total * max_cand_r_pad * 128 * 4 <= int(
          0.75 * eff_vmem_limit
      )
  )

  def _block_sizes(
      num_queries_per_block, bkv_p, static_q_len, seq_batch_size=1
  ):
    """Query and KV block sizes a pass runs with."""
    if static_q_len is not None:
      bq_sz = min(num_queries_per_block, static_q_len)
    else:
      bq_sz = num_queries_per_block
    if (
        use_fused_vmem_topk
        and seq_batch_size == 1
        and bq_sz > 64
        and num_sublanes_total * align_to(bq_sz, 8) * 128 * 4
        > int(0.45 * eff_vmem_limit)
    ):
      bq_sz = 64
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
    bq_sz, _ = _block_sizes(
        num_queries_per_block, bkv_p, static_q_len, seq_batch_size
    )

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

  def run_fused_topk_kernel(
      q,
      prepared_indexer_weights,
      cache_kv,
      topk_ids_init,
      topk_scores_init,
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
      pass_meta,
      need_scores: bool,
      case=MlaCase.MIXED,
      chunk_token_start=None,
      chunk_tokens=None,
      chunk_idx=0,
  ):
    max_num_tokens = q.shape[0]
    if seq_batch_size > 1:
      assert static_q_len == 1

    chunked = chunk_token_start is not None
    if not chunked:
      chunk_tokens = max_num_tokens

    bkv_p = num_kv_pages_per_block
    bq_sz, _ = _block_sizes(
        num_queries_per_block, bkv_p, static_q_len, seq_batch_size
    )
    k_aligned = align_to(k, 128)
    num_k_tiles = k_aligned // 128

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [
        hbm_spec,  # q
        hbm_spec,  # prepared_indexer_weights
        hbm_spec,  # cache_kv
        hbm_spec,  # topk_ids_init (aliased to output 0)
    ]
    if need_scores:
      in_specs.append(hbm_spec)  # topk_scores_init (aliased to output 1)
      out_specs = (hbm_spec, hbm_spec)
      out_shape = (
          jax.ShapeDtypeStruct((chunk_tokens, num_k_tiles, 128), jnp.int32),
          jax.ShapeDtypeStruct((chunk_tokens, num_k_tiles, 128), jnp.int32),
      )
    else:
      out_specs = hbm_spec
      out_shape = jax.ShapeDtypeStruct(
          (chunk_tokens, num_k_tiles, 128), jnp.int32
      )

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
    input_output_aliases = {num_flat_prefetches + 3: 0}
    if need_scores:
      input_output_aliases[num_flat_prefetches + 4] = 1

    layout_tag = "" if kv_layout == KVLayout.HEAD_ALONG_SUBLANE else "-sal"
    scope_name = (
        f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}{layout_tag}-ftk{k}"
    )
    if chunked:
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
                fused_topk_k=k,
                fused_return_scores=need_scores,
                num_sublanes_total=num_sublanes_total,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=num_scalar_prefetch,
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(1,),
                scratch_shapes=[],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=eff_vmem_limit,
                disable_bounds_checks=True,
            ),
            out_shape=out_shape,
            input_output_aliases=input_output_aliases,
            name=scope_name,
        )
    )
    return pallas_kernel(
        *scalar_prefetches,
        q,
        prepared_indexer_weights,
        cache_kv,
        topk_ids_init,
        *((topk_scores_init,) if need_scores else ()),
    )

  def _pass_specs(decode_batch_end):
    """`run_scores_kernel` kwargs for the passes over the batch."""
    decode_cfg = dict(
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        buffer_count=buffer_counts[0],
        static_q_len=1,
        case=MlaCase.DECODE,
    )
    mixed_spec = dict(
        num_kv_pages_per_block=num_kv_pages_per_blocks[2],
        num_queries_per_block=num_queries_per_blocks[2],
        buffer_count=buffer_counts[2],
        start_seq_idx=distribution[1],
        end_seq_idx=distribution[2],
        static_q_len=None,
        seq_batch_size=1,
        case=MlaCase.MIXED,
    )
    uniform = (
        ()
        if uniform_q_len is None
        else (
            dict(
                num_kv_pages_per_block=num_kv_pages_per_blocks[1],
                num_queries_per_block=num_queries_per_blocks[1],
                buffer_count=buffer_counts[1],
                start_seq_idx=distribution[0],
                end_seq_idx=distribution[1],
                static_q_len=uniform_q_len,
                seq_batch_size=1,
                case=MlaCase.PREFILL,
            ),
        )
    )
    if uniform_q_len is None:
      if max_num_seqs == 1 and q.shape[0] == 1:
        return (
            dict(
                start_seq_idx=jnp.array(0),
                end_seq_idx=distribution[2],
                seq_batch_size=1,
                **decode_cfg,
            ),
        )
      if (max_num_seqs == 1 and q.shape[0] > 1) or (
          num_queries_per_blocks[2] >= 64 and q.shape[0] >= max_num_seqs * 64
      ):
        return (
            dict(
                num_kv_pages_per_block=num_kv_pages_per_blocks[2],
                num_queries_per_block=num_queries_per_blocks[2],
                buffer_count=buffer_counts[2],
                start_seq_idx=jnp.array(0),
                end_seq_idx=distribution[2],
                static_q_len=None,
                seq_batch_size=1,
                case=MlaCase.MIXED,
            ),
        )
    single_batched_decode = (
        decode_only
        and uniform_q_len is None
        and max_num_seqs >= decode_req_batch_size
        and max_num_seqs % decode_req_batch_size == 0
        and q.shape[0] >= max_num_seqs
    )
    if max_num_seqs < decode_req_batch_size:
      batched_decode = ()
    else:
      batched_decode = (
          dict(
              start_seq_idx=jnp.array(0),
              end_seq_idx=(
                  distribution[0] if single_batched_decode else decode_batch_end
              ),
              seq_batch_size=decode_req_batch_size,
              **decode_cfg,
          ),
      )
    if decode_only and uniform_q_len is None:
      if single_batched_decode:
        return batched_decode
      return (
          *batched_decode,
          dict(
              start_seq_idx=decode_batch_end,
              end_seq_idx=distribution[1],
              seq_batch_size=1,
              **decode_cfg,
          ),
      )
    if decode_only and uniform_q_len is not None:
      return uniform
    return (
        *batched_decode,
        # Handle num_decode_seqs % decode_req_batch_size != 0 case.
        dict(
            start_seq_idx=decode_batch_end,
            end_seq_idx=distribution[0 if uniform else 1],
            seq_batch_size=1,
            **decode_cfg,
        ),
        *uniform,
        mixed_spec,
    )

  if max_score_sublanes is not None:
    meta_seq_lens = jnp.minimum(
        seq_lens,
        jnp.asarray(
            min(num_sublanes_total * 128 * compression_ratio, 0x7FFFFFFF),
            dtype=seq_lens.dtype,
        ),
    )
  else:
    meta_seq_lens = seq_lens

  def _pass_metadata(spec, chunk_starts, chunk_tokens):
    """One schedule per pass, holding every chunk's steps back to back."""
    bq_sz, bkv_sz = _block_sizes(
        spec["num_queries_per_block"],
        spec["num_kv_pages_per_block"],
        spec["static_q_len"],
        spec["seq_batch_size"],
    )
    return metadata.compute_metadata(
        seq_lens=meta_seq_lens,
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

  def _fused_topk_for_chunk(
      passes,
      chunk_idx,
      chunk_token_start,
      chunk_tokens,
      need_scores: bool,
  ):
    k_aligned = align_to(k, 128)
    num_k_tiles = k_aligned // 128
    topk_ids = jnp.full(
        (chunk_tokens, num_k_tiles, 128), jnp.int32(-1), dtype=jnp.int32
    )
    if need_scores:
      topk_scores = jnp.full(
          (chunk_tokens, num_k_tiles, 128),
          jnp.int32(BITS_NEG_INF),
          dtype=jnp.int32,
      )
    else:
      topk_scores = None
    for spec, pass_meta in passes:
      res = run_fused_topk_kernel(
          q,
          prepared_indexer_weights,
          cache_kv,
          topk_ids,
          topk_scores,
          seq_lens,
          page_indices,
          cu_q_lens,
          pass_meta=pass_meta,
          need_scores=need_scores,
          chunk_idx=chunk_idx,
          chunk_token_start=chunk_token_start,
          chunk_tokens=chunk_tokens,
          **spec,
      )
      if need_scores:
        topk_ids, topk_scores = res
      else:
        topk_ids = res
    ids_flat = topk_ids.reshape(chunk_tokens, k_aligned)[:, :k]
    if not need_scores:
      return ids_flat, None
    assert topk_scores is not None
    return ids_flat, topk_scores.reshape(chunk_tokens, k_aligned)[:, :k]

  out_dtype = jnp.int32
  max_num_tokens = q.shape[0]
  decode_batch_end = (
      jnp.array(0)
      if max_num_seqs < decode_req_batch_size
      else distribution[0] // decode_req_batch_size * decode_req_batch_size
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
  chunk_starts = (
      (None,)
      if num_chunks == 1
      else tuple(m * chunk_len for m in range(num_chunks))
  )
  if num_chunks == 1 or topk_mode != "sparsecore":
    hoisted_passes = tuple(
        (spec, _pass_metadata(spec, chunk_starts, chunk_len))
        for spec in _pass_specs(decode_batch_end)
    )
  else:
    hoisted_passes = None

  def _common_path(_):
    # Ids count from 1, so the 0 is never a real group: a single chunk has no
    # pair to annotate and takes the `None` branch below both times.
    group_base = 0
    if num_chunks > 1 and topk_mode == "sparsecore":
      group_base = config.reserve_scheduling_group_ids(num_chunks - 1)

    need_scores = return_scores

    if hoisted_passes is not None:
      passes = hoisted_passes
    else:
      passes = tuple(
          (spec, _pass_metadata(spec, chunk_starts, chunk_len))
          for spec in _pass_specs(decode_batch_end)
      )
    topk_idxs = []
    topk_scores_list = []
    for m in range(num_chunks):
      if use_fused_vmem_topk:
        chunk_topk_idxs, chunk_topk_scores = _fused_topk_for_chunk(
            passes,
            m,
            chunk_starts[m],
            chunk_len,
            need_scores=need_scores,
        )
        topk_idxs.append(chunk_topk_idxs)
        if need_scores:
          assert chunk_topk_scores is not None
          topk_scores_list.append(chunk_topk_scores)
        continue

      # Group g holds chunk g's SparseCore top-k and chunk g+1's TensorCore
      # scoring, so the scheduler emits `sc(g).start, tc(g+1), sc(g).done`.
      # Chunk 0's scoring and the last top-k have no partner.
      tc_group = (
          None if (m == 0 or topk_mode != "sparsecore") else group_base + m - 1
      )
      sc_group = (
          None
          if (m == num_chunks - 1 or topk_mode != "sparsecore")
          else group_base + m
      )

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

      if topk_mode == "sparsecore":
        if need_scores:
          chunk_topk_idxs, chunk_topk_scores = sparsecore_topk(
              scores,
              k,
              row_lengths=eff_row_lengths[start : start + chunk_len],
              write_empty_rows=True,
              scheduling_group_id=sc_group,
              return_scores=True,
          )
        else:
          chunk_topk_idxs = sparsecore_topk(
              scores,
              k,
              row_lengths=eff_row_lengths[start : start + chunk_len],
              write_empty_rows=True,
              scheduling_group_id=sc_group,
              return_scores=False,
          )
          chunk_topk_scores = None
      elif topk_mode in (
          "tensorcore_full",
          "tensorcore_bounded",
          "tensorcore_fused",
      ):
        tc_out = tensorcore_topk_select(
            scores,
            k,
            eff_row_lengths[start : start + chunk_len],
            enable_prefix_bucketing=topk_mode != "tensorcore_full",
            return_scores=need_scores,
        )
        if need_scores:
          assert isinstance(tc_out, tuple)
          chunk_topk_idxs, chunk_topk_scores = tc_out
        else:
          assert isinstance(tc_out, jax.Array)
          chunk_topk_idxs, chunk_topk_scores = tc_out, None
      else:
        raise ValueError(f"Unsupported topk_mode: {topk_mode}")
      topk_idxs.append(chunk_topk_idxs)
      if need_scores:
        assert chunk_topk_scores is not None
        topk_scores_list.append(chunk_topk_scores)

    out = topk_idxs[0] if len(topk_idxs) == 1 else jnp.concatenate(topk_idxs)
    out = out[:max_num_tokens, :k]

    if not return_scores:
      return out

    scores_out = (
        topk_scores_list[0]
        if len(topk_scores_list) == 1
        else jnp.concatenate(topk_scores_list)
    )
    scores_out = scores_out[:max_num_tokens, :k]
    return out, scores_out

  def _fast_path(_):
    s_idx = jnp.arange(k, dtype=jnp.int32)[None, :]
    return jnp.where(s_idx < eff_row_lengths[:, None], s_idx, -1)

  if not enable_early_exit:
    return _common_path(None)
  return jax.lax.cond(
      jnp.max(eff_row_lengths) <= k,
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
        "kv_layout",
        "chunk_tokens",
        "dcp_axis_name",
        "tc_topk",
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
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
    chunk_tokens: int | None = None,
    dcp_axis_name: str = DCP_AXIS_NAME,
    tc_topk: bool = False,
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
       (axis 1) exchanges only the candidate *scores*:
       `[T, k] -> [T, dcp_size * k]`. The local candidate indices stay home.
    3. Fused SparseCore Merge, Owner Filter & Prefix Compaction: Each rank runs
       `sparsecore_topk_dcp_merge` over the gathered `[T, dcp_size * k]` scores
       and its pre-gather `idxs[T, k]`. Inside a single SparseCore pass, the
       kernel finds the exact global top-k threshold across all `dcp_size * k`
       scores, scans only `[dcp_rank * k, (dcp_rank + 1) * k)` during emission,
       resolves surviving slots directly to `idxs`, and left-packs them into a
       `-1`-padded prefix of width `k`. The ranks' outputs form an exact
       partition of the global top-k.

  When `tc_topk=True`, steps 1, 3, and 4 run on TensorCore via
  `tc_streamindex_topk` and `tc_topk.dcp_merge_and_compact_tc`, with the same
  tie-breaking and output contract.

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
    kv_layout: physical layout of `cache_kv`, a `KVLayout` or its string value.
    chunk_tokens: forwarded to `streamindex_topk` for the per-rank local scoring
      pass; see there for the semantics. Chunking cuts the replicated
      query-token axis, which is orthogonal to the KV axis DCP shards, so the
      two compose.
    dcp_axis_name: name of the DCP mesh axis.
    tc_topk: whether to run local top-k and the cross-rank merge/compaction on
      TensorCore instead of SparseCore. Defaults to False.

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
        kv_layout=kv_layout,
        chunk_tokens=chunk_tokens,
        cp_size=dcp_size,
        cp_rank=dcp_rank,
        interleave_size=interleave_size,
        return_scores=True,
        tc_topk=tc_topk,
    )

    # Stage 1. All-gather the scores; the indices stay home.
    scores = lax.all_gather(scores, dcp_axis_name, axis=1, tiled=True)

    if tc_topk:
      return tc_topk_lib.dcp_merge_and_compact_tc(
          scores,
          idxs,
          k=k,
          dcp_rank=dcp_rank,
          dcp_size=dcp_size,
          inputs_sorted=True,
      )

    # Stage 2. Fused SparseCore global top-k threshold selection, owner
    # filtering, local index resolution, and prefix compaction.
    return sparsecore_topk_dcp_merge(
        scores,
        idxs,
        k,
        dcp_rank=dcp_rank,
        dcp_size=dcp_size,
    )

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
