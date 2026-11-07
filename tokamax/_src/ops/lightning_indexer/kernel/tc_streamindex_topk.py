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
"""Fused StreamIndex scoring + TensorCore Top-K kernel (no SparseCore needed).

Used on TPUs with fewer than 32 SparseCore subcores.
"""

import dataclasses
import functools
import math
import operator
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.lightning_indexer.kernel import bref_override
from tokamax._src.ops.lightning_indexer.kernel import config
from tokamax._src.ops.lightning_indexer.kernel import metadata
from tokamax._src.ops.lightning_indexer.kernel import tc_topk

BITS_NEG_INF = tc_topk.BITS_NEG_INF
MlaCase = config.MlaCase
KVLayout = config.KVLayout
DEFAULT_BUFFER_COUNT = config.DEFAULT_BUFFER_COUNT
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

cp_local_length = metadata.cp_local_length
KV_PACKING = 4

# Top-K implementations of the fused kernel:
#   `stream`: per-lane sorted candidate lists + collision-driven staging.
#   `select`: VMEM-resident order keys + threshold selection
#     (`tc_topk.select_topk_vmem`) once per query block.
#   `auto`: `select` where it fits in VMEM and is expected to be faster.
TOPK_IMPL_AUTO = "auto"
TOPK_IMPL_STREAM = "stream"
TOPK_IMPL_SELECT = "select"
_TOPK_IMPLS = (TOPK_IMPL_AUTO, TOPK_IMPL_STREAM, TOPK_IMPL_SELECT)
_MAX_UNROLLED_INSERT_CHUNKS = 2
# Multi-KV-block passes use `stream` when the total number of score tiles
# across all KV blocks is at most this multiple of `num_blocks`.
_STREAM_MAX_TOTAL_TILES_PER_TOPK_BLOCK = 12


def cdiv(a: int, b: int) -> int:
  assert b != 0
  return (a + b - 1) // b


def align_to(x: int, a: int) -> int:
  return cdiv(x, a) * a


def get_dtype_bitwidth(dtype: Any) -> int:
  return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype: Any) -> int:
  return 32 // get_dtype_bitwidth(dtype)


def prepare_q_inputs(q: jax.Array) -> jax.Array:
  """Zero-pads `q` heads to the dtype packing and `head_dim` to 128 lanes."""
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


def prepare_index_weights(index_weights: jax.Array, q_dtype: Any) -> jax.Array:
  _, actual_num_q_heads = index_weights.shape
  index_weights = index_weights.astype(jnp.float32)
  num_q_heads = align_to(actual_num_q_heads, get_dtype_packing(q_dtype))
  if num_q_heads != actual_num_q_heads:
    index_weights = jnp.pad(
        index_weights,
        ((0, 0), (0, num_q_heads - actual_num_q_heads)),
        constant_values=0,
    )
  return index_weights


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
  record_width = head_dim + head_dim // 128
  return (
      total_num_pages,
      page_size // KV_PACKING,
      KV_PACKING,
      align_to(record_width, 128),
  )


# --------------------------------------------------------------------------- #
# Custom BufferedRefs for `qblock_schedule=True` MetadataRef.
# --------------------------------------------------------------------------- #


def _step_offset(meta_ref: Any, chunk_idx: int):
  return sum(meta_ref.num_steps[m] for m in range(chunk_idx))


def _qblock_token_window(
    refs: tuple[Any, ...],
    p_id: int | jax.Array,
    chunk_tokens: int,
    *,
    bq_sz: int,
    seq_batch_size: int,
    chunk_idx: int = 0,
):
  """Flat-token rows `(token_start, total_sz)` for step `p_id`."""
  _, cu_q_lens_ref, meta_ref, start_end_seq_idx_ref = refs
  chunk_start = start_end_seq_idx_ref[2]
  target_step = p_id + _step_offset(meta_ref, chunk_idx)
  seq_idx, bq_idx, _, _, _ = metadata.lookup_qblock_step(meta_ref, target_step)
  max_seq = cu_q_lens_ref.shape[0] - 1
  seq_clamped = jnp.minimum(seq_idx, max_seq - 1)
  lo = jnp.maximum(cu_q_lens_ref[seq_clamped], chunk_start)

  if seq_batch_size > 1:
    return lo, seq_batch_size
  token_start = lo + bq_idx * bq_sz
  if bq_sz == 1:
    return token_start, 1
  hi = jnp.minimum(cu_q_lens_ref[seq_clamped + 1], chunk_start + chunk_tokens)
  return token_start, jnp.maximum(0, jnp.minimum(bq_sz, hi - token_start))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QBlockKVBufferedRef(bref_override.StreamIndexKVBufferedRef):
  """Paged KV BufferedRef (`HEAD_ALONG_SUBLANE`) using `lookup_qblock_step`."""

  def copy_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, Any],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    cache_kv_hbm, page_indices_ref, meta_ref = src_ref
    target_step = grid_indices[0] + _step_offset(meta_ref, self.chunk_idx)
    batch_tile_idx, _, bkv_idx, _, _ = metadata.lookup_qblock_step(
        meta_ref, target_step
    )

    assert self.sem_recvs is not None
    assert self.window_ref is not None
    assert isinstance(cache_kv_hbm, jax.Ref)
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_dst: Any = self.window_ref.at[slot]

    reshaped_cache_hbm_ref: Any = (
        cache_kv_hbm.reshape(  # pyrefly: ignore[missing-attribute]
            cache_kv_hbm.shape[0] * self.page_size_per_kv_packing,
            self.kv_packing,
            cache_kv_hbm.shape[-1],
        )
    )
    max_hbm_pages = reshaped_cache_hbm_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    for batch_idx in range(self.seq_batch_size):
      kv_p_start = bkv_idx * self.bkv_p
      page_indices_offset = (
          batch_tile_idx + batch_idx
      ) * self.pages_per_seq + kv_p_start
      for i in range(self.bkv_p):
        page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
        safe_page_offset = jnp.minimum(
            page_indices_ref[page_idx] * self.page_size_per_kv_packing,
            jnp.maximum(0, max_hbm_pages - self.page_size_per_kv_packing),
        )
        pltpu.make_async_copy(
            reshaped_cache_hbm_ref.at[
                pl.ds(safe_page_offset, self.page_size_per_kv_packing)
            ],
            vmem_dst.at[
                batch_idx,
                pl.ds(
                    i * self.page_size_per_kv_packing,
                    self.page_size_per_kv_packing,
                ),
            ],
            sem,
        ).start()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QBlockKVSeqAlongLaneBufferedRef(
    bref_override.StreamIndexKVSeqAlongLaneBufferedRef
):
  """Paged KV BufferedRef (`SEQ_ALONG_LANE`) using `lookup_qblock_step`."""

  def copy_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, Any],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    cache_kv_hbm, page_indices_ref, meta_ref = src_ref
    target_step = grid_indices[0] + _step_offset(meta_ref, self.chunk_idx)
    batch_tile_idx, _, bkv_idx, _, _ = metadata.lookup_qblock_step(
        meta_ref, target_step
    )

    assert self.sem_recvs is not None
    assert self.window_ref is not None
    assert isinstance(cache_kv_hbm, jax.Ref)
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_dst: Any = self.window_ref.at[slot]

    max_hbm_pages = cache_kv_hbm.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    for batch_idx in range(self.seq_batch_size):
      kv_p_start = bkv_idx * self.bkv_p
      page_indices_offset = (
          batch_tile_idx + batch_idx
      ) * self.pages_per_seq + kv_p_start
      for i in range(self.bkv_p):
        page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
        safe_page = jnp.minimum(page_indices_ref[page_idx], max_hbm_pages - 1)
        dst_lane_start = i * self.page_size
        pltpu.make_async_copy(
            cache_kv_hbm.at[safe_page],
            vmem_dst.at[
                batch_idx,
                :,
                :,
                pl.ds(dst_lane_start, self.page_size),
            ],
            sem,
        ).start()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QBlockQBufferedRef(bref_override.StreamIndexQBufferedRef):
  """Query / weights BufferedRef using `_qblock_token_window`."""

  def copy_in(
      self,
      src_ref: tuple[Any, ...],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    q_hbm = src_ref[0]
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_dst: Any = self.window_ref.at[slot]

    q_len_start, total_sz = _qblock_token_window(
        src_ref,
        grid_indices[0],
        self.chunk_tokens,
        bq_sz=self.bq_sz,
        seq_batch_size=self.seq_batch_size,
        chunk_idx=self.chunk_idx,
    )
    safe_start = jnp.minimum(
        q_len_start, jnp.maximum(0, q_hbm.shape[0] - jnp.maximum(1, total_sz))
    )
    pltpu.make_async_copy(
        q_hbm.at[pl.ds(safe_start, total_sz)],
        vmem_dst.at[pl.ds(0, total_sz)],
        sem,
    ).start()

  def wait_in(
      self,
      src_ref: tuple[Any, ...],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    assert self.is_input
    if not self.is_buffered:
      return
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_dst: Any = self.window_ref.at[slot]
    _, total_sz = _qblock_token_window(
        src_ref,
        grid_indices[0],
        self.chunk_tokens,
        bq_sz=self.bq_sz,
        seq_batch_size=self.seq_batch_size,
        chunk_idx=self.chunk_idx,
    )
    pltpu.make_async_copy(
        vmem_dst.at[pl.ds(0, total_sz)],
        vmem_dst.at[pl.ds(0, total_sz)],
        sem,
    ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QBlockTopKOBufferedRef(pltpu.BufferedRef):
  """Writes finalized per-query-block Top-K indices (+ scores) to HBM."""

  bq_sz: int = dataclasses.field(default=16, metadata=dict(static=True))
  seq_batch_size: int = dataclasses.field(default=1, metadata=dict(static=True))
  chunk_idx: int = dataclasses.field(default=0, metadata=dict(static=True))

  @classmethod
  def output(  # pyrefly: ignore[bad-override]
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_count: int,
      use_lookahead: bool,
      bq_sz: int,
      seq_batch_size: int = 1,
      grid_rank: int = 1,
      chunk_idx: int = 0,
      **kwargs: Any,
  ):
    del kwargs
    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=pltpu.BufferType.OUTPUT,
        buffer_count=buffer_count,
        grid_rank=grid_rank,
        use_lookahead=use_lookahead,
    )
    return cls(
        bq_sz=bq_sz,
        seq_batch_size=seq_batch_size,
        chunk_idx=chunk_idx,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )

  def copy_out(
      self,
      dst_ref: tuple[Any, ...],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    out_hbm, _, _, start_end_seq_idx_ref = dst_ref
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]
    vmem_src: Any = self.window_ref.at[slot]

    token_start, total_sz = _qblock_token_window(
        dst_ref,
        grid_indices[0],
        out_hbm.shape[0],
        bq_sz=self.bq_sz,
        seq_batch_size=self.seq_batch_size,
        chunk_idx=self.chunk_idx,
    )
    dst_start = jnp.minimum(
        jnp.maximum(0, token_start - start_end_seq_idx_ref[2]),
        jnp.maximum(0, out_hbm.shape[0] - jnp.maximum(1, total_sz)),
    )
    pltpu.make_async_copy(
        vmem_src.at[pl.ds(0, total_sz)],
        out_hbm.at[pl.ds(dst_start, total_sz)],
        sem,
    ).start()

  def wait_out(
      self,
      dst_ref: tuple[Any, ...],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    assert self.is_output
    if not self.is_buffered:
      return
    assert self.sem_sends is not None
    assert self.window_ref is not None
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    vmem_src: Any = self.window_ref.at[slot]
    _, total_sz = _qblock_token_window(
        dst_ref,
        grid_indices[0],
        dst_ref[0].shape[0],
        bq_sz=self.bq_sz,
        seq_batch_size=self.seq_batch_size,
        chunk_idx=self.chunk_idx,
    )
    sliced = vmem_src.at[pl.ds(0, total_sz)]
    pltpu.make_async_copy(sliced, sliced, sem).wait()


# --------------------------------------------------------------------------- #
# Fused scoring + in-VMEM TensorCore Top-K kernel.
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class _BlockedPrefillConfig:
  """Accumulator-fitted multi-MXU prefill scoring layout."""

  enabled: bool
  tok_blk: int
  num_tok_blks: int
  tbs_per_batch: int
  num_head_groups: int
  heads_per_group: int
  rows_per_group_tok_blk: int
  rows_per_group: int
  lane_chunk: int
  num_lane_chunks: int
  tiles_per_chunk: int
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
    """Derives the blocked prefill scoring layout for one block shape."""
    tpu_info = pltpu.get_tpu_info()
    tok_blk = math.gcd(bq_sz, 32)
    num_tok_blks = bq_sz // tok_blk
    num_head_groups = math.gcd(num_q_heads, tpu_info.num_mxus)
    heads_per_group = num_q_heads // num_head_groups
    rows_per_group_tok_blk = heads_per_group * tok_blk
    rows_per_group = heads_per_group * bq_sz

    lane_chunk = math.gcd(bkv_sz, 256)
    num_lane_chunks = bkv_sz // lane_chunk
    tiles_per_chunk = lane_chunk // 128

    # Size each MXU contraction batch to fit hardware MXU accumulators
    # (e.g. `256` `(8, 128)` accumulators = `1024` rows at `lane_chunk = 256`)
    # so `sts` never spills to VMEM when `H * bq` is large.
    max_accum_rows = max(
        rows_per_group_tok_blk,
        (tpu_info.num_accumulators // max(1, tiles_per_chunk)) * 8,
    )
    tbs_per_batch = math.gcd(
        num_tok_blks, max(1, max_accum_rows // rows_per_group_tok_blk)
    )

    q_sublane_tiling = 8 * get_dtype_packing(q_dtype)
    enabled = (
        seq_batch_size == 1 and bq_sz > 1 and tok_blk % q_sublane_tiling == 0
    )

    bf16_reduce = get_dtype_bitwidth(q_dtype) == 8 and tok_blk % 16 == 0
    wb_dtype = jnp.bfloat16 if bf16_reduce else jnp.float32

    return cls(
        enabled=enabled,
        tok_blk=tok_blk,
        num_tok_blks=num_tok_blks,
        tbs_per_batch=tbs_per_batch,
        num_head_groups=num_head_groups,
        heads_per_group=heads_per_group,
        rows_per_group_tok_blk=rows_per_group_tok_blk,
        rows_per_group=rows_per_group,
        lane_chunk=lane_chunk,
        num_lane_chunks=num_lane_chunks,
        tiles_per_chunk=tiles_per_chunk,
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


def tc_kernel(
    # Scalar prefetches (SMEM)
    seq_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [3] (start_seq, end_seq, chunk_token_start)
    metadata_ref,  # MetadataRef (qblock_schedule=True)
    cp_rank_ref,  # [1]
    # HBM refs
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    indexer_weights_hbm_ref,  # [max_num_tokens, num_q_heads]
    cache_kv_hbm_ref,
    *out_refs,
    k: int,
    need_scores: bool,
    num_positions: int,
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
    topk_impl: str = TOPK_IMPL_STREAM,
):
  """Executes fused MXU scoring + in-VMEM TensorCore Top-K."""
  del case
  out_hbm_ref = out_refs[-1]
  chunk_tokens = out_hbm_ref.shape[0]
  buffer_cnt = buffer_count
  seq_along_lane = kv_layout == KVLayout.SEQ_ALONG_LANE

  _, num_q_heads, head_dim = q_hbm_ref.shape
  if seq_along_lane:
    _, kv_sublane_groups, kv_packing, page_size = cache_kv_hbm_ref.shape
    kv_head_dim_groups = head_dim // kv_packing
    page_size_per_kv_packing = page_size // kv_packing
  else:
    _, page_size_per_kv_packing, kv_packing, _ = cache_kv_hbm_ref.shape
    page_size = page_size_per_kv_packing * kv_packing
    kv_sublane_groups = None
    kv_head_dim_groups = None

  max_num_seqs = seq_lens_ref.shape[0]
  pages_per_seq = page_indices_ref.shape[0] // max_num_seqs

  bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
  bkv_sz = bkv_p * page_size
  num_sublanes_bkv = bkv_sz // 128

  _, num_blocks = tc_topk.topk_capacity(k)
  num_out_blocks = (2 * num_blocks) if need_scores else num_blocks

  bq_total = seq_batch_size * bq_sz
  bq_alloc = align_to(bq_total, 8)
  num_q_groups = bq_alloc // 8

  prefill_cfg = _BlockedPrefillConfig.create(
      bq_sz=bq_sz,
      bkv_sz=bkv_sz,
      num_q_heads=num_q_heads,
      q_dtype=q_hbm_ref.dtype,
      seq_batch_size=seq_batch_size,
  )
  num_partitions = (
      1
      if prefill_cfg.enabled
      else tc_topk.sublane_partition_count(bq_total, bkv_sz)
  )
  single_bkv = pages_per_seq <= bkv_p and num_partitions == 1
  use_select = topk_impl == TOPK_IMPL_SELECT
  num_kv_blocks = cdiv(pages_per_seq, bkv_p)
  select_idx_bits = max(1, (num_positions - 1).bit_length())
  step_offset = sum(metadata_ref.num_steps[m] for m in range(chunk_idx))
  # Lists never need more levels than a lane can see tiles per query block.
  max_tiles_per_lane = cdiv(
      align_to(pages_per_seq, bkv_p) * page_size, 128 * num_partitions
  )
  lane_depth = min(
      tc_topk.lane_list_depth(num_blocks),
      align_to(max_tiles_per_lane, num_blocks),
  )
  ins_w = math.gcd(num_sublanes_bkv // num_partitions, 8)
  num_ins_chunks = (num_sublanes_bkv // num_partitions) // ins_w
  # Lane-list and evicted-entry VMEM refs, bound by `_run_all`.
  lane_refs: dict[str, Any] = {}

  def lookup_step(p_id):
    return metadata.lookup_qblock_step(metadata_ref, step_offset + p_id)

  def q_blk_id(p_id):
    return lookup_step(p_id)[4]

  def kv_sal_idx(p_id):
    bt, _, cbkv, _, _ = lookup_step(p_id)
    return (bt, 0, 0, cbkv)

  def kv_has_idx(p_id):
    bt, _, cbkv, _, _ = lookup_step(p_id)
    return (bt, 0, cbkv, 0)

  q_spec = pl.BlockSpec(
      (pl.BoundedSlice(bq_total), num_q_heads, head_dim),
      lambda p_id: (q_blk_id(p_id), 0, 0),
      pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
  )
  weights_spec = pl.BlockSpec(
      (pl.BoundedSlice(bq_total), num_q_heads),
      lambda p_id: (q_blk_id(p_id), 0),
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
        kv_sal_idx,
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
        kv_has_idx,
        pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
    )
  out_spec = pl.BlockSpec(
      (pl.BoundedSlice(bq_alloc), num_out_blocks, 128),
      lambda p_id: (q_blk_id(p_id), 0, 0),
      pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
  )

  q_alloc = QBlockQBufferedRef.input(
      spec=q_spec,
      dtype_or_type=q_hbm_ref,
      buffer_count=buffer_cnt,
      use_lookahead=True,
      bq_sz=bq_sz,
      seq_batch_size=seq_batch_size,
      chunk_tokens=chunk_tokens,
      chunk_idx=chunk_idx,
  )
  weights_alloc = QBlockQBufferedRef.input(
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
    kv_alloc = QBlockKVSeqAlongLaneBufferedRef.input(
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
    kv_alloc = QBlockKVBufferedRef.input(
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
  o_alloc = QBlockTopKOBufferedRef.output(
      spec=out_spec,
      dtype_or_type=out_hbm_ref,
      buffer_count=2,
      use_lookahead=False,
      bq_sz=bq_sz,
      seq_batch_size=seq_batch_size,
      chunk_idx=chunk_idx,
  )

  num_lane_tiles_bkv = bkv_sz // 128

  def seq_along_lane_block_rows(bkv_vmem):
    return bkv_vmem.bitcast(jnp.uint32).reshape(
        seq_batch_size, kv_sublane_groups * num_lane_tiles_bkv, 128
    )

  def seq_along_lane_key_tile(rows, batch_idx, lane_tile_idx):
    words = rows[
        batch_idx,
        pl.ds(lane_tile_idx, kv_head_dim_groups, stride=num_lane_tiles_bkv),
        :,
    ]
    return pltpu.bitcast(words, jnp.float8_e4m3fn)

  def seq_along_lane_scale_tile(rows, batch_idx, lane_tile_idx):
    row = kv_head_dim_groups * num_lane_tiles_bkv + lane_tile_idx
    words = rows[batch_idx, pl.ds(row, 1), :]
    return pltpu.bitcast((words & 0xFF) << 23, jnp.float32)

  def transpose_to_head_major(bq_rows, n_tok):
    return bq_rows.transpose(1, 0, 2).reshape(num_q_heads * n_tok, head_dim)

  hm_tok = align_to(bq_sz, 8)

  def pad_tokens(bq_rows, n_tok):
    if n_tok == hm_tok:
      return bq_rows
    pad = jnp.zeros((hm_tok - n_tok, *bq_rows.shape[1:]), bq_rows.dtype)
    return jnp.concatenate([bq_rows, pad], axis=0)

  def reduce_heads(bq_rows, bkv, bq_weights, head_major=False):
    n_tok = bq_weights.shape[0]
    w = bq_weights.astype(jnp.float32)

    def qk(rows):
      if seq_along_lane:
        return jnp.einsum(
            "nd,dm->nm", rows, bkv, preferred_element_type=jnp.float32
        )
      return jnp.einsum(
          "nd,md->nm", rows, bkv, preferred_element_type=jnp.float32
      )

    if n_tok == 1:
      st = qk(bq_rows).reshape(n_tok, num_q_heads, -1)
      st = jnp.maximum(st, 0.0) * w[:, :, None]
      return st.sum(axis=1)

    if head_major:
      rows_hm, rows_tok = bq_rows, hm_tok
      w = jnp.pad(w, ((0, rows_tok - n_tok), (0, 0)))
    else:
      rows_hm, rows_tok = transpose_to_head_major(bq_rows, n_tok), n_tok
    st = qk(rows_hm).reshape(num_q_heads, rows_tok, -1)
    acc = jnp.maximum(st[0], 0.0) * w[:, 0:1]
    for h in range(1, num_q_heads):
      relu_st = jnp.maximum(st[h], 0.0)
      acc = acc + relu_st * w[:, h : h + 1]
    return acc[:n_tok]

  def compute_scores(
      bq_vec, bkv_vec, scale_val_vec, bq_weights_vec, q_is_head_major=False
  ):
    ret = []
    for batch_idx in range(seq_batch_size):
      bq = bq_vec[batch_idx].reshape(-1, head_dim)
      bkv = bkv_vec[batch_idx]
      scale_val = scale_val_vec[batch_idx]
      bq_weights = bq_weights_vec[batch_idx]
      s_summed = reduce_heads(bq, bkv, bq_weights, head_major=q_is_head_major)
      s_summed = s_summed * scale_val
      ret.append(s_summed.reshape(-1, num_sublanes_bkv, 128))
    return jnp.concatenate(ret, axis=0)

  def _init_query_block(
      seq_idx,
      bq_idx,
      top_o_ref,
      top_i_ref,
      cand_o_ref,
      cand_i_ref,
      tau_ref,
      eff_len_ref,
  ):
    if top_o_ref is not None:
      tc_topk.init_topk_vmem(
          top_o_ref,
          top_i_ref,
          cand_o_ref,
          cand_i_ref,
          tau_ref,
          num_q_groups=num_q_groups,
          num_blocks=num_blocks,
      )

      def _init_lanes(g, _):
        tc_topk.init_lane_lists_vmem(
            lane_refs["lst_o"],
            lane_refs["lst_i"],
            q_group=g,
            depth=lane_depth,
        )
        return None

      if num_q_groups == 1:
        _init_lanes(0, None)
      else:
        lax.fori_loop(0, num_q_groups, _init_lanes, None)
    end_seq = start_end_seq_idx_ref[1]
    cp_rank_val = cp_rank_ref[0] if cp_size > 1 else 0

    if seq_batch_size > 1:
      for g in range(num_q_groups):
        row_chunks = []
        num_slots = 8 // num_partitions
        for slot in range(num_slots):
          b = slot if num_partitions > 1 else (g * 8 + slot)
          if b >= seq_batch_size:
            row_chunks.append(jnp.zeros((num_partitions, 128), dtype=jnp.int32))
            continue
          s = seq_idx + b
          s_clamped = jnp.minimum(s, max_num_seqs - 1)
          q_len_s = cu_q_lens_ref[s_clamped + 1] - cu_q_lens_ref[s_clamped]
          seq_len_s = seq_lens_ref[s_clamped]
          kv_len_s = seq_len_s // compression_ratio
          tok_pos_s = seq_len_s - q_len_s
          vis_s = jnp.clip(tok_pos_s // compression_ratio + 1, 0, kv_len_s)
          if cp_size > 1:
            vis_s = cp_local_length(vis_s, cp_rank_val, cp_size, interleave_c)
          valid_s = (s < end_seq) & (s < max_num_seqs) & (q_len_s > 0)
          eff_s = lax.select(
              valid_s,
              jnp.minimum(vis_s, jnp.int32(num_positions)),
              jnp.int32(0),
          )
          row_chunks.append(
              jnp.full((num_partitions, 128), eff_s, dtype=jnp.int32)
          )
        eff_len_ref[g] = (
            row_chunks[0]
            if len(row_chunks) == 1
            else jnp.concatenate(row_chunks, axis=0)
        )
    else:
      s_clamped = jnp.minimum(seq_idx, max_num_seqs - 1)
      seq_vec = jnp.full((8, 128), seq_idx, jnp.int32)
      end_vec = jnp.full((8, 128), end_seq, jnp.int32)
      chunk_start_vec = jnp.full((8, 128), start_end_seq_idx_ref[2], jnp.int32)
      seq_q_start_vec = jnp.full((8, 128), cu_q_lens_ref[s_clamped], jnp.int32)
      seq_q_end_vec = jnp.full(
          (8, 128), cu_q_lens_ref[s_clamped + 1], jnp.int32
      )
      lo_vec = jnp.maximum(seq_q_start_vec, chunk_start_vec)
      hi_vec = jnp.minimum(seq_q_end_vec, chunk_start_vec + chunk_tokens)
      tok_start_vec = lo_vec + jnp.full((8, 128), bq_idx * bq_sz, jnp.int32)
      seq_len_vec = jnp.full((8, 128), seq_lens_ref[s_clamped], jnp.int32)
      kv_len_vec = seq_len_vec // compression_ratio
      cp_vec = (
          jnp.full((8, 128), cp_rank_val, jnp.int32) if cp_size > 1 else None
      )

      sublanes = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
      part_shift = (num_partitions - 1).bit_length()

      def _init_eff_group(g, _):
        r_off = (
            lax.shift_right_arithmetic(sublanes, jnp.int32(part_shift))
            if num_partitions > 1
            else (g * jnp.int32(8) + sublanes)
        )
        tok_id = tok_start_vec + r_off
        row_valid = (
            (seq_vec < end_vec)
            & (seq_vec < max_num_seqs)
            & (r_off < bq_sz)
            & (tok_id < hi_vec)
        )
        tok_pos = seq_len_vec - (seq_q_end_vec - tok_id)
        vis = jnp.clip(tok_pos // compression_ratio + 1, 0, kv_len_vec)
        if cp_size > 1:
          vis = cp_local_length(vis, cp_vec, cp_size, interleave_c)
        eff_len_ref[g] = jnp.where(
            row_valid, jnp.minimum(vis, num_positions), 0
        )
        return None

      if num_q_groups == 1:
        _init_eff_group(jnp.int32(0), None)
      else:
        lax.fori_loop(0, num_q_groups, _init_eff_group, None)

  def _active_q_groups(p_id):
    """Number of query groups holding rows of this query block.

    Static when every group can hold valid rows; otherwise computed from the
    block's token window so empty trailing groups skip their top-k work.

    Args:
      p_id: Pipeline step index.

    Returns:
      The active query-group count, a Python int when static.
    """
    if (
        num_q_groups == 1
        or seq_batch_size > 1
        or (
            static_q_len is not None
            and static_q_len % bq_sz == 0
            and chunk_tokens % bq_sz == 0
        )
    ):
      return num_q_groups
    _, total_sz = _qblock_token_window(
        (None, cu_q_lens_ref, metadata_ref, start_end_seq_idx_ref),
        p_id,
        chunk_tokens,
        bq_sz=bq_sz,
        seq_batch_size=1,
        chunk_idx=chunk_idx,
    )
    return (total_sz + 7) // 8

  def _for_each_group(n_act, fn, carry=None):
    """Runs `carry = fn(g, carry)` over the `n_act` active query groups."""
    if isinstance(n_act, int) and n_act == 1:
      return fn(0, carry)
    return lax.fori_loop(0, n_act, fn, carry)

  def _for_group_pairs(n_act, fn):
    """Runs `fn(groups)` over the active query groups, two at a time.

    With an odd count the last pair repeats its group; `fn` is idempotent per
    group, and lock-step processing overlaps latency-bound sorting networks.

    Args:
      n_act: Number of active query groups.
      fn: Callback taking a tuple of query group indices.
    """
    if isinstance(n_act, int) and n_act == 1:
      fn((jnp.int32(0),))
      return

    def _pair(p, _):
      fn((2 * p, jnp.minimum(2 * p + 1, n_act - 1)))
      return None

    lax.fori_loop(0, (n_act + 1) // 2, _pair, None)

  def _write_group_out(out_vmem, g, fin_o, fin_i):
    """Writes one query group's final Top-K indices (+ score bits)."""
    idx_2d = jnp.concatenate(fin_i, axis=1)
    if need_scores:
      scores_2d = jnp.concatenate(
          [
              tc_topk.ord_i32_to_f32_bits(o, i >= 0)
              for o, i in zip(fin_o, fin_i)
          ],
          axis=1,
      )
      group_out = jnp.concatenate([idx_2d, scores_2d], axis=1).reshape(
          8, num_out_blocks, 128
      )
    else:
      group_out = idx_2d.reshape(8, num_out_blocks, 128)
    out_vmem[pl.ds(g * 8, 8)] = group_out

  def _finalize_query_block(
      out_vmem, top_o_ref, top_i_ref, cand_o_ref, cand_i_ref, tau_ref, n_act
  ):
    def _finalize_groups(groups):
      results = tc_topk.finalize_lane_lists_vmem(
          top_o_ref,
          top_i_ref,
          cand_o_ref,
          cand_i_ref,
          tau_ref,
          lane_refs["lst_o"],
          lane_refs["lst_i"],
          q_groups=groups,
          k=k,
          num_blocks=num_blocks,
          depth=lane_depth,
          num_partitions=num_partitions,
      )
      for g, (fin_o, fin_i) in zip(groups, results):
        _write_group_out(out_vmem, g, fin_o, fin_i)

    # Pairs of query groups are finalized together to overlap their
    # latency-bound sorting networks.
    _for_group_pairs(n_act, _finalize_groups)

  m_part = num_sublanes_bkv // num_partitions
  # Two-level select (per-block lane maxima first) needs enough cells per row
  # and enough tiles per block to pay off.
  use_two_level = (
      use_select and m_part >= 4 and num_kv_blocks >= 2 * num_blocks
  )
  subbatch_w = math.gcd(m_part, min(num_blocks, 4))
  num_subbatches = m_part // subbatch_w
  direct_blocks = num_blocks
  num_direct_chunks = cdiv(m_part, direct_blocks)

  def _direct_single_bkv_topk(scores_tile_ref, eff_len_ref, out_vmem, n_act):
    lanes = lax.broadcasted_iota(jnp.int32, (8, 128), 1)
    empty_o = jnp.full((8, 128), tc_topk.EMPTY_ORD_SENTINEL, dtype=jnp.int32)
    empty_i = jnp.full((8, 128), -1, dtype=jnp.int32)

    def _load_direct_chunk(g, eff_len_g, ch_idx: int):
      ords = []
      idxs = []
      for r in range(direct_blocks):
        m_p = ch_idx * direct_blocks + r
        if m_p < m_part:
          s_tile = scores_tile_ref[g, m_p]
          idx_tile = jnp.int32(m_p * 128) + lanes
          valid = idx_tile < eff_len_g
          ords.append(tc_topk.f32_to_ord_i32(s_tile, valid))
          idxs.append(jnp.where(valid, idx_tile, empty_i))
        else:
          ords.append(empty_o)
          idxs.append(empty_i)
      return tuple(ords), tuple(idxs)

    def _direct_groups(groups):
      # Query groups run in lock-step so their latency-bound sorting
      # networks overlap.
      eff_lens = [eff_len_ref[g] for g in groups]

      def _tree(g, eff_len_g, lo, hi, descending):
        # Balanced merge tree over chunks `[lo, hi)`: independent sorts and
        # merges expose more parallelism than a sequential chain.
        if hi - lo == 1:
          ch_o, ch_i = _load_direct_chunk(g, eff_len_g, lo)
          return tc_topk.bitonic_sort_blocks(
              ch_o, ch_i, descending=descending
          )
        mid = (lo + hi) // 2
        l_o, l_i = _tree(g, eff_len_g, lo, mid, True)
        r_o, r_i = _tree(g, eff_len_g, mid, hi, False)
        return tc_topk.bitonic_topk_merge_blocks(
            l_o, l_i, r_o, r_i, descending=descending
        )

      tops = [
          _tree(g, eff_len_g, 0, num_direct_chunks, True)
          for g, eff_len_g in zip(groups, eff_lens)
      ]

      for g, (top_o, top_i) in zip(groups, tops):
        _write_group_out(out_vmem, g, top_o[:num_blocks], top_i[:num_blocks])

    _for_group_pairs(n_act, _direct_groups)

  def _stream_scores_from_vmem(
      cur_bkv,
      scores_tile_ref,
      top_o_ref,
      top_i_ref,
      cand_o_ref,
      cand_i_ref,
      tau_ref,
      eff_len_ref,
      n_act,
  ):
    """Streams one KV block of score tiles into the per-lane candidate lists.

    Entries that fall off a lane list are written to VMEM. Only if one of them
    reaches the row's k-th-best lower bound (rare) are they staged into the
    collision-driven top-k state, behind a single scalar sync per KV block.

    Args:
      cur_bkv: Index of the current KV block within the query block.
      scores_tile_ref: `[groups, m_part, 8, 128]` float32 score tile ref.
      top_o_ref: VMEM ref of the running descending top-k order keys.
      top_i_ref: VMEM ref of the indices matching `top_o_ref`.
      cand_o_ref: VMEM ref of the staged candidate order keys.
      cand_i_ref: VMEM ref of the indices matching `cand_o_ref`.
      tau_ref: VMEM ref of each row's k-th best staged order key.
      eff_len_ref: VMEM ref of each row's valid KV length.
      n_act: Number of active query groups.
    """
    lst_o_ref, lst_i_ref = lane_refs["lst_o"], lane_refs["lst_i"]
    ev_o_ref, ev_i_ref = lane_refs["ev_o"], lane_refs["ev_i"]
    lanes = lax.broadcasted_iota(jnp.int32, (8, 128), 1)
    if num_partitions > 1:
      sublanes = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
      part_col_offset = (
          lax.bitwise_and(sublanes, jnp.int32(num_partitions - 1))
          * jnp.int32(m_part * 128)
          + lanes
      )
    else:
      part_col_offset = lanes
    bkv_base_vec = jnp.full((8, 128), cur_bkv * bkv_sz, dtype=jnp.int32)

    def _tau_floor(g):
      return tc_topk.lane_lists_tau_floor(
          lst_o_ref, q_group=g, num_blocks=num_blocks
      )

    def _insert_group(g, hot):
      eff_len_g = eff_len_ref[g]
      tau_x = tc_topk.exclusive_threshold(
          jnp.maximum(tau_ref[g], _tau_floor(g))
      )

      def _chunk(c, hot):
        t_ords = []
        t_idxs = []
        m_ps = [c * ins_w + w for w in range(ins_w)]
        for m_p in m_ps:
          idx_tile = (bkv_base_vec + m_p * jnp.int32(128)) + part_col_offset
          valid = idx_tile < eff_len_g
          t_ords.append(tc_topk.f32_to_ord_i32(scores_tile_ref[g, m_p], valid))
          t_idxs.append(idx_tile)
        ev_o, ev_i = tc_topk.lane_lists_insert_vmem(
            lst_o_ref,
            lst_i_ref,
            q_group=g,
            tile_ords=tuple(t_ords),
            tile_idxs=tuple(t_idxs),
            depth=lane_depth,
        )
        for w, m_p in enumerate(m_ps):
          ev_o_ref[g, m_p] = ev_o[w]
          ev_i_ref[g, m_p] = ev_i[w]
          hot = hot | jnp.where(ev_o[w] > tau_x, 1, 0)
        return hot

      if num_ins_chunks <= _MAX_UNROLLED_INSERT_CHUNKS:
        # Unrolled so the threshold reduction overlaps the insertion.
        for c in range(num_ins_chunks):
          hot = _chunk(c, hot)
        return hot
      return lax.fori_loop(0, num_ins_chunks, _chunk, hot)

    hot = _for_each_group(
        n_act, _insert_group, jnp.zeros((8, 128), jnp.int32)
    )

    @pl.when(jnp.any(hot != 0))
    def _stage_evicted():
      def _stage_group(g, _):
        floor = _tau_floor(g)

        def _stage_sb(sb, _):
          m0 = sb * subbatch_w
          tc_topk.accumulate_subbatch_vmem(
              top_o_ref,
              top_i_ref,
              cand_o_ref,
              cand_i_ref,
              tau_ref,
              q_group=g,
              tile_ords=tuple(
                  ev_o_ref[g, m0 + w] for w in range(subbatch_w)
              ),
              tile_idxs=tuple(
                  ev_i_ref[g, m0 + w] for w in range(subbatch_w)
              ),
              sub_step=sb,
              k=k,
              num_blocks=num_blocks,
              tau_floor=floor,
              inclusive=True,
          )
          return None

        def _stage_all():
          if num_subbatches == 1:
            _stage_sb(0, None)
          else:
            lax.fori_loop(0, num_subbatches, _stage_sb, None)

        if num_q_groups == 1:
          _stage_all()
        else:
          # Skip query groups without a qualifying evicted entry.
          tau_x = tc_topk.exclusive_threshold(jnp.maximum(tau_ref[g], floor))
          g_hot = functools.reduce(
              operator.or_, [ev_o_ref[g, m] > tau_x for m in range(m_part)]
          )
          pl.when(jnp.any(g_hot))(_stage_all)
        return None

      _for_each_group(n_act, _stage_group)

  def _part_col_offset():
    lanes = lax.broadcasted_iota(jnp.int32, (8, 128), 1)
    if num_partitions == 1:
      return lanes
    sublanes = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
    return (
        lax.bitwise_and(sublanes, jnp.int32(num_partitions - 1))
        * jnp.int32(m_part * 128)
        + lanes
    )

  def _select_ingest_tile(select_refs, eff_len_g, g, cur_bkv, m_p, s_tile):
    """Stores one score tile of block `cur_bkv` as masked order keys."""
    idx_tile = cur_bkv * bkv_sz + m_p * 128 + _part_col_offset()
    select_refs[0][g, cur_bkv * m_part + m_p] = tc_topk.f32_to_ord_i32(
        s_tile, idx_tile < eff_len_g
    )

  def _select_summarize(cur_bkv, select_refs, n_act):
    """Records each lane's max key (+ its tile) of block `cur_bkv`."""
    if not use_two_level:
      return
    keys_ref, cell_max_ref, cell_arg_ref = select_refs

    def _summarize_group(g, _):
      best, arg = tc_topk.summarize_key_block(
          lambda t: keys_ref[g, cur_bkv * m_part + t], m_part
      )
      cell_max_ref[g, cur_bkv] = best
      cell_arg_ref[g, cur_bkv] = arg
      return None

    _for_each_group(n_act, _summarize_group)

  def _select_finalize(
      n_act, n_kv_blocks, select_refs, top_o_ref, top_i_ref, out_vmem
  ):
    """Exact Top-K of the query block's resident keys, written to `out_vmem`."""
    keys_ref = select_refs[0]
    common = dict(k=k, num_blocks=num_blocks, idx_bits=select_idx_bits)

    def _idx_tile(b, t):
      return b * bkv_sz + t * 128 + _part_col_offset()

    def _finalize_group(g, _):
      def _load_tile(b, t):
        return keys_ref[g, b * m_part + t]

      if use_two_level:
        tc_topk.select_topk_two_level(
            _load_tile,
            _idx_tile,
            lambda b: select_refs[1][g, b],
            lambda b: select_refs[2][g, b],
            n_kv_blocks,
            m_part,
            top_o_ref,
            top_i_ref,
            q_group=g,
            **common,
        )
      else:
        tc_topk.select_topk_vmem(
            _load_tile,
            _idx_tile,
            n_kv_blocks,
            m_part,
            top_o_ref,
            top_i_ref,
            q_group=g,
            **common,
        )
      fin_o = tuple(top_o_ref[g, r] for r in range(num_blocks))
      fin_i = tuple(top_i_ref[g, r] for r in range(num_blocks))
      if num_partitions > 1:
        fin_o, fin_i = tc_topk.reduce_sublane_partitions(
            fin_o, fin_i, num_partitions=num_partitions
        )
      _write_group_out(out_vmem, g, fin_o, fin_i)
      return None

    _for_each_group(n_act, _finalize_group)

  def step_body(
      step,
      q_vmem,
      weights_vmem,
      bkv_vmem,
      out_vmem,
      *,
      top_o_ref,
      top_i_ref,
      cand_o_ref,
      cand_i_ref,
      tau_ref,
      eff_len_ref,
      scores_tile_ref=None,
      q_hm_ref=None,
      select_refs=None,
  ):
    p_id = step.index[0]
    seq_idx, bq_idx, cur_bkv, is_last_bkv, _ = lookup_step(p_id)

    @pl.when(cur_bkv == 0)
    def _on_first():
      _init_query_block(
          seq_idx,
          bq_idx,
          None if use_select else top_o_ref,
          top_i_ref,
          cand_o_ref,
          cand_i_ref,
          tau_ref,
          eff_len_ref,
      )
      if q_hm_ref is not None:
        q_hm_ref[...] = transpose_to_head_major(
            pad_tokens(q_vmem[...], bq_sz), hm_tok
        )

    if q_hm_ref is not None:
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

    if num_partitions > 1:
      active_sublanes = bq_total * num_partitions
      scores_part = scores.reshape(active_sublanes, m_part, 128)
      if active_sublanes < 8:
        scores_part = jnp.pad(
            scores_part,
            ((0, 8 - active_sublanes), (0, 0), (0, 0)),
            constant_values=-jnp.inf,
        )
      scores_padded = scores_part
    elif bq_alloc > bq_total:
      scores_padded = jnp.pad(
          scores,
          ((0, bq_alloc - bq_total), (0, 0), (0, 0)),
          constant_values=-jnp.inf,
      )
    else:
      scores_padded = scores

    n_act = _active_q_groups(p_id)
    if use_select:
      for g in range(num_q_groups):
        grp_scores = scores_padded[g * 8 : (g + 1) * 8]
        eff_len_g = eff_len_ref[g]
        for m_p in range(m_part):
          _select_ingest_tile(
              select_refs, eff_len_g, g, cur_bkv, m_p, grp_scores[:, m_p, :]
          )
      _select_summarize(cur_bkv, select_refs, n_act)

      @pl.when(is_last_bkv)
      def _on_last_select():
        _select_finalize(
            n_act, cur_bkv + 1, select_refs, top_o_ref, top_i_ref, out_vmem
        )

      return

    assert scores_tile_ref is not None
    for g in range(num_q_groups):
      grp_scores = scores_padded[g * 8 : (g + 1) * 8]
      for m_p in range(m_part):
        scores_tile_ref[g, m_p] = grp_scores[:, m_p, :]
    if single_bkv:
      _direct_single_bkv_topk(scores_tile_ref, eff_len_ref, out_vmem, n_act)
    else:
      _stream_scores_from_vmem(
          cur_bkv,
          scores_tile_ref,
          top_o_ref,
          top_i_ref,
          cand_o_ref,
          cand_i_ref,
          tau_ref,
          eff_len_ref,
          n_act,
      )

      @pl.when(is_last_bkv)
      def _on_last():
        _finalize_query_block(
            out_vmem,
            top_o_ref,
            top_i_ref,
            cand_o_ref,
            cand_i_ref,
            tau_ref,
            n_act,
        )

  def blocked_prefill_step(
      step,
      q_vmem,
      weights_vmem,
      bkv_vmem,
      out_vmem,
      *,
      top_o_ref,
      top_i_ref,
      cand_o_ref,
      cand_i_ref,
      tau_ref,
      eff_len_ref,
      scores_tile_ref,
      q_hm_ref,
      wb_ref,
      select_refs=None,
  ):
    p_id = step.index[0]
    seq_idx, bq_idx, cur_bkv, is_last_bkv, _ = lookup_step(p_id)

    tok_blk = prefill_cfg.tok_blk
    num_tok_blks = prefill_cfg.num_tok_blks
    tbs_per_batch = prefill_cfg.tbs_per_batch
    num_head_groups = prefill_cfg.num_head_groups
    heads_per_group = prefill_cfg.heads_per_group
    rows_per_group_tok_blk = prefill_cfg.rows_per_group_tok_blk
    rows_per_group = prefill_cfg.rows_per_group
    lane_chunk = prefill_cfg.lane_chunk
    num_lane_chunks = prefill_cfg.num_lane_chunks
    tiles_per_chunk = prefill_cfg.tiles_per_chunk
    groups_per_tb = tok_blk // 8
    batch_rows = tbs_per_batch * rows_per_group_tok_blk

    @pl.when(cur_bkv == 0)
    def _on_first():
      _init_query_block(
          seq_idx,
          bq_idx,
          None if use_select else top_o_ref,
          top_i_ref,
          cand_o_ref,
          cand_i_ref,
          tau_ref,
          eff_len_ref,
      )
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
        return jnp.einsum(
            "nd,md->nm", q_rows, keys, preferred_element_type=jnp.float32
        )

    def reduce_token_block(sts, tb: int, local_tb: int) -> jax.Array:
      acc = None
      for i in range(heads_per_group):
        for g_h, st in enumerate(sts):
          r0 = local_tb * rows_per_group_tok_blk + i * tok_blk
          st_h = st[r0 : r0 + tok_blk]
          w_h = wb_ref[tb, i * num_head_groups + g_h]
          if tiles_per_chunk > 1:
            w_h = jnp.concatenate([w_h] * tiles_per_chunk, axis=1)
          if prefill_cfg.bf16_reduce:
            r = jnp.maximum(st_h.astype(jnp.bfloat16), 0)
            term = r.astype(jnp.float32) * w_h.astype(jnp.float32)
          else:
            term = jnp.maximum(st_h, 0.0) * w_h
          acc = term if acc is None else acc + term
      assert acc is not None
      return acc

    for c in range(num_lane_chunks):
      keys, scales = load_chunk(c)
      for tb0 in range(0, num_tok_blks, tbs_per_batch):
        sts = [
            qk(
                q_hm_ref[
                    pl.ds(
                        g_h * rows_per_group + tb0 * rows_per_group_tok_blk,
                        batch_rows,
                    )
                ],
                keys,
            )
            for g_h in range(num_head_groups)
        ]
        for local_tb in range(tbs_per_batch):
          tb = tb0 + local_tb
          acc = reduce_token_block(sts, tb, local_tb) * scales
          for jj in range(tiles_per_chunk):
            acc_tile = acc[:, jj * 128 : (jj + 1) * 128]
            for sg in range(groups_per_tb):
              grp = tb * groups_per_tb + sg
              m_p = c * tiles_per_chunk + jj
              tile = acc_tile[sg * 8 : (sg + 1) * 8]
              if use_select:
                _select_ingest_tile(
                    select_refs, eff_len_ref[grp], grp, cur_bkv, m_p, tile
                )
              else:
                scores_tile_ref[grp, m_p] = tile

    n_act = _active_q_groups(p_id)
    if use_select:
      _select_summarize(cur_bkv, select_refs, n_act)

      @pl.when(is_last_bkv)
      def _on_last_select():
        _select_finalize(
            n_act, cur_bkv + 1, select_refs, top_o_ref, top_i_ref, out_vmem
        )

    elif single_bkv:
      _direct_single_bkv_topk(scores_tile_ref, eff_len_ref, out_vmem, n_act)
    else:
      _stream_scores_from_vmem(
          cur_bkv,
          scores_tile_ref,
          top_o_ref,
          top_i_ref,
          cand_o_ref,
          cand_i_ref,
          tau_ref,
          eff_len_ref,
          n_act,
      )

      @pl.when(is_last_bkv)
      def _on_last():
        _finalize_query_block(
            out_vmem,
            top_o_ref,
            top_i_ref,
            cand_o_ref,
            cand_i_ref,
            tau_ref,
            n_act,
        )

  def _run_all(
      top_o_ref,
      top_i_ref,
      cand_o_ref,
      cand_i_ref,
      tau_ref,
      eff_len_ref,
      scores_tile_ref=None,
      q_hm_ref=None,
      wb_ref=None,
      select_refs=None,
  ):
    if prefill_cfg.enabled:
      body = functools.partial(
          blocked_prefill_step,
          top_o_ref=top_o_ref,
          top_i_ref=top_i_ref,
          cand_o_ref=cand_o_ref,
          cand_i_ref=cand_i_ref,
          tau_ref=tau_ref,
          eff_len_ref=eff_len_ref,
          scores_tile_ref=scores_tile_ref,
          q_hm_ref=q_hm_ref,
          wb_ref=wb_ref,
          select_refs=select_refs,
      )
    else:
      body = functools.partial(
          step_body,
          top_o_ref=top_o_ref,
          top_i_ref=top_i_ref,
          cand_o_ref=cand_o_ref,
          cand_i_ref=cand_i_ref,
          tau_ref=tau_ref,
          eff_len_ref=eff_len_ref,
          scores_tile_ref=scores_tile_ref,
          q_hm_ref=q_hm_ref,
          select_refs=select_refs,
      )

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
            (out_hbm_ref, *window_refs),
            allocations=(q_buf, weights_buf, kv_buf, o_buf),
        )

    if not use_select and not single_bkv:
      lane_list_vmem = pltpu.VMEM(
          (num_q_groups, lane_depth + num_blocks, 8, 128), jnp.int32
      )
      evicted_vmem = pltpu.VMEM((num_q_groups, m_part, 8, 128), jnp.int32)

      @pl.with_scoped(
          lst_o=lane_list_vmem,
          lst_i=lane_list_vmem,
          ev_o=evicted_vmem,
          ev_i=evicted_vmem,
      )
      def _with_lane(lst_o, lst_i, ev_o, ev_i):
        lane_refs["lst_o"] = lst_o
        lane_refs["lst_i"] = lst_i
        lane_refs["ev_o"] = ev_o
        lane_refs["ev_i"] = ev_i
        _run_pipeline()

      _with_lane()
    else:
      _run_pipeline()

  vec_vmem = pltpu.VMEM((num_q_groups, 8, 128), jnp.int32)
  scores_vmem = pltpu.VMEM((num_q_groups, m_part, 8, 128), jnp.float32)

  if use_select:
    topk_vmem = pltpu.VMEM((num_q_groups, num_blocks, 8, 128), jnp.int32)
    keys_vmem = pltpu.VMEM(
        (num_q_groups, num_kv_blocks * m_part, 8, 128), jnp.int32
    )
    cells_vmem = pltpu.VMEM(
        (num_q_groups, num_kv_blocks if use_two_level else 1, 8, 128),
        jnp.int32,
    )
    extra_scratches = ()
    if prefill_cfg.enabled:
      extra_scratches = (
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
      extra_scratches = (
          pltpu.VMEM((num_q_heads * hm_tok, head_dim), q_hbm_ref.dtype),
      )

    def _run_select(
        top_o, top_i, eff_r, keys_r, cmax_r, carg_r, q_hm=None, wb_r=None
    ):
      _run_all(
          top_o, top_i, None, None, None, eff_r, None, q_hm, wb_r,
          select_refs=(keys_r, cmax_r, carg_r),
      )

    pl.run_scoped(
        _run_select,
        topk_vmem,
        topk_vmem,
        vec_vmem,
        keys_vmem,
        cells_vmem,
        cells_vmem,
        *extra_scratches,
    )
  elif single_bkv:
    if prefill_cfg.enabled:
      pl.run_scoped(
          lambda eff_r, sc_r, q_hm, wb_r: _run_all(
              None, None, None, None, None, eff_r, sc_r, q_hm, wb_r
          ),
          vec_vmem,
          scores_vmem,
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
      pl.run_scoped(
          lambda eff_r, sc_r, q_hm: _run_all(
              None, None, None, None, None, eff_r, sc_r, q_hm
          ),
          vec_vmem,
          scores_vmem,
          pltpu.VMEM((num_q_heads * hm_tok, head_dim), q_hbm_ref.dtype),
      )
    else:
      pl.run_scoped(
          lambda eff_r, sc_r: _run_all(
              None, None, None, None, None, eff_r, sc_r
          ),
          vec_vmem,
          scores_vmem,
      )
  else:
    topk_vmem = pltpu.VMEM((num_q_groups, num_blocks, 8, 128), jnp.int32)
    base_scratches = (
        topk_vmem,
        topk_vmem,
        topk_vmem,
        topk_vmem,
        vec_vmem,
        vec_vmem,
        scores_vmem,
    )
    if prefill_cfg.enabled:
      pl.run_scoped(
          _run_all,
          *base_scratches,
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
      pl.run_scoped(
          _run_all,
          *base_scratches,
          pltpu.VMEM((num_q_heads * hm_tok, head_dim), q_hbm_ref.dtype),
      )
    else:
      pl.run_scoped(_run_all, *base_scratches)


# --------------------------------------------------------------------------- #
# Host / XLA driver matching `streamindex_topk` API & invariants.
# --------------------------------------------------------------------------- #


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
        "topk_impl",
    ),
)
def tc_streamindex_topk(
    q: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
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
    topk_impl: str = TOPK_IMPL_AUTO,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Fused TensorCore StreamIndex Top-K retrieval (no SparseCore needed).

  Drop-in replacement for `streamindex_topk.streamindex_topk` on TPUs whose
  SparseCore has fewer than 32 subcores. Scoring and the exact top-k selection
  both run in one TensorCore Pallas kernel per pass, with the same tie-breaking
  as `jax.lax.top_k` (higher score first, then lower index).

  Args:
    q: Concatenated queries of all sequences, `[num_tokens, num_heads, D]`.
    indexer_weights: Concatenated indexer weights, `[num_tokens, num_heads]`.
    cache_kv: Packed `uint8` FP8 index cache; its shape depends on
      `kv_layout` (see `streamindex_topk.streamindex_topk`).
    seq_lens: Uncompressed length of each sequence in the KV cache.
    page_indices: Flattened page table indexed by `(seq_id, page_id)`.
    cu_q_lens: Cumulative sum of the per-sequence query lengths.
    distribution: `(i, j, k)`: sequences `[0, i)` are decode-only, `[i, j)`
      are chunked-prefill-only, and `[j, k)` are mixed.
    k: Number of top-k positions to return per query token.
    compression_ratio: KV cache compression ratio (a power of 2).
    num_kv_pages_per_block: KV pages per block, as a `(decode, prefill,
      mixed)` tuple or a single integer.
    num_queries_per_block: Queries per block, as a `(decode, prefill, mixed)`
      tuple or a single integer.
    buffer_count: Pipeline buffer count, as a `(decode, prefill, mixed)`
      tuple or a single integer.
    vmem_limit_bytes: Scoped VMEM limit for the Pallas kernel.
    decode_req_batch_size: Maximum number of decode sequences per block.
    enable_early_exit: Whether to skip the kernel when `k` covers every
      sequence's compressed KV length.
    kv_layout: Memory layout of `cache_kv`.
    chunk_tokens: Size, in tokens, of each piece the flat token axis is cut
      into, or None for a single pass.
    cp_size: Number of context-parallel ranks the compressed KV cache is
      sharded over.
    cp_rank: This rank's index in the context-parallel group.
    interleave_size: Context-parallel interleave width in uncompressed
      tokens; must be a multiple of `compression_ratio`.
    return_scores: Whether to also return the score of each selected
      position, as needed for the cross-rank merge.
    uniform_q_len: Static query length shared by every sequence in
      `[distribution[0], distribution[1])`, or None.
    topk_impl: Top-K implementation: `"stream"`, `"select"` (VMEM-resident
      keys + threshold selection), or `"auto"` to pick per pass. All give
      identical results.

  Returns:
    `int32[num_tokens, k]` top-k indices in this rank's local compressed
    space, or `(indices, scores)` if `return_scores`, where `scores` is
    `int32[num_tokens, k]` holding the raw IEEE-754 `float32` bits of each
    score. Unfilled slots have index `-1` and score bits of `-inf`.
  """
  kv_layout = KVLayout.parse(kv_layout)
  if topk_impl not in _TOPK_IMPLS:
    raise ValueError(f"topk_impl must be one of {_TOPK_IMPLS}, got {topk_impl}")

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
  elif q.shape[1] >= 64 and max_num_seqs <= 32 and decode_req_batch_size > 8:
    decode_req_batch_size = 8

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
  num_positions = max(num_sublanes_total * 128, k)
  k_cap, num_blocks = tc_topk.topk_capacity(k)
  num_out_blocks = (2 * num_blocks) if return_scores else num_blocks
  select_budget = (
      min(vmem_limit_bytes, pltpu.get_tpu_info().vmem_capacity_bytes) // 2
  )

  def _select_group_bytes(bkv_p):
    """VMEM bytes of one query group's resident `select` keys."""
    return cdiv(pages_per_seq, bkv_p) * page_size * bkv_p * 8 * 4

  def _block_sizes(num_queries_per_block, bkv_p, static_q_len, chunk_len):
    if static_q_len is not None:
      bq_sz = min(num_queries_per_block, static_q_len)
    else:
      bq_sz = num_queries_per_block
    if bq_sz > 32 and chunk_len < bq_sz:
      chunk_cap = max(32, 1 << (max(1, chunk_len) - 1).bit_length())
      bq_sz = min(bq_sz, chunk_cap)
    if topk_impl != TOPK_IMPL_STREAM and bq_sz > 8 and pages_per_seq > bkv_p:
      # Shrink the query block until its resident `select` keys fit in VMEM:
      # streaming Top-K over many KV blocks is far slower than `select`.
      max_groups = select_budget // _select_group_bytes(bkv_p)
      if 1 <= max_groups < cdiv(bq_sz, 8):
        bq_sz = min(bq_sz, 1 << ((8 * max_groups).bit_length() - 1))
    return bq_sz, page_size * bkv_p

  def _pass_topk_impl(bq_sz, bkv_p, seq_batch_size):
    """Resolves `topk_impl` for one fused pass."""
    if topk_impl != TOPK_IMPL_AUTO:
      return topk_impl
    bq_total = seq_batch_size * bq_sz
    bkv_sz = page_size * bkv_p
    prefill_enabled = _BlockedPrefillConfig.create(
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        num_q_heads=q.shape[1],
        q_dtype=q.dtype,
        seq_batch_size=seq_batch_size,
    ).enabled
    num_partitions = (
        1
        if prefill_enabled
        else tc_topk.sublane_partition_count(bq_total, bkv_sz)
    )
    tiles_per_block = bkv_sz // 128 // num_partitions
    if pages_per_seq <= bkv_p:
      if num_partitions == 1:
        return TOPK_IMPL_STREAM
      return (
          TOPK_IMPL_SELECT
          if tiles_per_block > num_blocks
          else TOPK_IMPL_STREAM
      )
    num_kv_blocks = cdiv(pages_per_seq, bkv_p)
    if (
        num_kv_blocks * tiles_per_block
        <= _STREAM_MAX_TOTAL_TILES_PER_TOPK_BLOCK * num_blocks
        or tiles_per_block < 4
    ):
      return TOPK_IMPL_STREAM
    key_bytes = cdiv(bq_total, 8) * _select_group_bytes(bkv_p)
    return TOPK_IMPL_SELECT if key_bytes <= select_budget else TOPK_IMPL_STREAM

  def run_fused_pass(
      out_init,
      *,
      start_seq_idx,
      end_seq_idx,
      static_q_len,
      num_kv_pages_per_block,
      num_queries_per_block,
      buffer_count,
      seq_batch_size,
      pass_meta,
      case=MlaCase.MIXED,
      chunk_token_start=None,
      chunk_len=None,
      scheduling_group_id=None,
      chunk_idx=0,
  ):
    max_num_tokens = q.shape[0]
    if seq_batch_size > 1:
      assert static_q_len == 1

    chunked = chunk_token_start is not None
    eff_chunk_len = (
        max_num_tokens if not chunked or chunk_len is None else chunk_len
    )

    bkv_p = num_kv_pages_per_block
    bq_sz, _ = _block_sizes(
        num_queries_per_block, bkv_p, static_q_len, eff_chunk_len
    )
    pass_topk_impl = _pass_topk_impl(bq_sz, bkv_p, seq_batch_size)

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    aliased = out_init is not None
    in_specs = [hbm_spec, hbm_spec, hbm_spec]
    if aliased:
      in_specs.append(hbm_spec)

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
    input_output_aliases = {num_flat_prefetches + 3: 0} if aliased else {}

    layout_tag = "" if kv_layout == KVLayout.HEAD_ALONG_SUBLANE else "-sal"
    scope_name = (
        f"StreamIdxFusedTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}{layout_tag}"
    )
    if chunked:
      scope_name += f"-c{max_num_tokens // eff_chunk_len}"
    if pass_topk_impl == TOPK_IMPL_SELECT:
      scope_name += "-sel"

    pallas_kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                tc_kernel,
                k=k,
                need_scores=return_scores,
                num_positions=num_positions,
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
                topk_impl=pass_topk_impl,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=num_scalar_prefetch,
                in_specs=in_specs,
                out_specs=hbm_spec,
                grid=(1,),
                scratch_shapes=[],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(
                shape=(eff_chunk_len, num_out_blocks, 128),
                dtype=jnp.int32,
            ),
            input_output_aliases=input_output_aliases,
            name=scope_name,
        )
    )
    out = pallas_kernel(
        *scalar_prefetches,
        q,
        prepared_indexer_weights,
        cache_kv,
        *((out_init,) if aliased else ()),
    )
    if scheduling_group_id is not None:
      out = xla_metadata.set_xla_metadata(
          out, _scheduling_group_id=scheduling_group_id
      )
    return out

  def _pass_specs(decode_batch_end):
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
    if uniform_q_len is None:
      uniform = ()
    else:
      uniform = (
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
    if max_num_seqs < decode_req_batch_size:
      batched_decode = ()
    else:
      batched_decode = (
          dict(
              start_seq_idx=jnp.array(0),
              end_seq_idx=decode_batch_end,
              seq_batch_size=decode_req_batch_size,
              **decode_cfg,
          ),
      )
    return (
        *batched_decode,
        dict(
            start_seq_idx=decode_batch_end,
            end_seq_idx=distribution[0 if uniform else 1],
            seq_batch_size=1,
            **decode_cfg,
        ),
        *uniform,
        mixed_spec,
    )

  def _pass_metadata(spec, chunk_starts, chunk_len):
    bq_sz, bkv_sz = _block_sizes(
        spec["num_queries_per_block"],
        spec["num_kv_pages_per_block"],
        spec["static_q_len"],
        chunk_len,
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
        chunk_tokens=chunk_len,
        cp_rank=cp_rank,
        cp_size=cp_size,
        interleave_c=interleave_c,
        qblock_schedule=True,
    )

  def _common_path(_):
    max_num_tokens = q.shape[0]
    decode_batch_end = (
        jnp.array(0)
        if max_num_seqs < decode_req_batch_size
        else distribution[0] // decode_req_batch_size * decode_req_batch_size
    )

    if chunk_tokens is None or max_num_tokens % chunk_tokens:
      num_chunks, chunk_len = 1, max_num_tokens
    else:
      num_chunks, chunk_len = max_num_tokens // chunk_tokens, chunk_tokens

    group_base = 0
    if num_chunks > 1:
      group_base = config.reserve_scheduling_group_ids(num_chunks - 1)

    chunk_starts = (
        (None,)
        if num_chunks == 1
        else tuple(m * chunk_len for m in range(num_chunks))
    )
    passes = tuple(
        (spec, _pass_metadata(spec, chunk_starts, chunk_len))
        for spec in _pass_specs(decode_batch_end)
    )

    chunk_outs = []
    for m in range(num_chunks):
      tc_group = None if m == 0 else group_base + m - 1
      out_chunk = None
      for spec, pass_meta in passes:
        out_chunk = run_fused_pass(
            out_chunk,
            pass_meta=pass_meta,
            chunk_idx=m,
            chunk_token_start=chunk_starts[m],
            chunk_len=chunk_len,
            scheduling_group_id=tc_group,
            **spec,
        )
      chunk_outs.append(out_chunk)

    raw_out = (
        chunk_outs[0]
        if len(chunk_outs) == 1
        else jnp.concatenate(chunk_outs, axis=0)
    )
    num_active_seqs = jnp.clip(distribution[2], 0, max_num_seqs)
    active_tokens = cu_q_lens[num_active_seqs]
    valid_row = (
        jnp.arange(max_num_tokens, dtype=jnp.int32)[:, None] < active_tokens
    )

    top_idx = raw_out[:, :num_blocks, :].reshape(max_num_tokens, k_cap)[:, :k]
    top_idx = jnp.where(valid_row, top_idx, jnp.int32(-1))

    if not return_scores:
      return top_idx

    top_scores = raw_out[:, num_blocks:, :].reshape(max_num_tokens, k_cap)[
        :, :k
    ]
    top_scores = jnp.where(valid_row, top_scores, jnp.int32(BITS_NEG_INF))
    return top_idx, top_scores

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
