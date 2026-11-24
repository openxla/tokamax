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

from collections.abc import Callable, Sequence
import dataclasses
import functools
import math
import operator
from typing import Any, override

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
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


def _kv_page_size(cache_kv_shape: Sequence[int], kv_layout: KVLayout) -> int:
  """Returns the number of compressed KV positions per cache page."""
  if kv_layout == KVLayout.SEQ_ALONG_LANE:
    return cache_kv_shape[-1]
  return cache_kv_shape[1] * cache_kv_shape[2]


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
# In-kernel query-block schedule.
# --------------------------------------------------------------------------- #

# Largest per-step query-block table (int32 SMEM entries) a pass builds. Passes
# with more possible steps binary-search the query-block end steps instead.
_MAX_STEP_TABLE_ENTRIES = 32768

# A sequence bound of a fused pass: `None` for 0, or `(i, align)` for
# `distribution[i] // align * align`.
SeqBound = tuple[int, int] | None
_DECODE_END: SeqBound = (0, 1)
_PREFILL_END: SeqBound = (1, 1)
_MIXED_END: SeqBound = (2, 1)

_INFO_START_SEQ = 0
_INFO_END_SEQ = 1
_INFO_NUM_STEPS = 2


def _seq_bound(dist_ref: Any, bound: SeqBound) -> jax.Array:
  if bound is None:
    return jnp.int32(0)
  idx, align = bound
  value = dist_ref[idx]
  if align > 1:
    value = value // align * align
  return value


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ScheduleRef:
  """SMEM query-block schedule of one fused pass, built inside the kernel.

  Query blocks are numbered in pipeline order; query block `qb` covers the
  pipeline steps `[q_start[qb], q_end[qb])`, one per KV block.

  Attributes:
    info: `[3]` SMEM `int32` ref holding `(start_seq, end_seq, num_steps)`.
    batch_tile_idx: `[num_q_blocks]` first sequence index of each block's tile.
    bq_idx: `[num_q_blocks]` query-block index within its sequence tile.
    q_start: `[num_q_blocks]` first pipeline step of each query block.
    q_end: `[num_q_blocks]` exclusive end step of each query block.
    step_qb: `[num_steps_cap]` query-block index of each step, or `None` when
      binary search over `q_end` is used instead.
    chunk_start: First flat token index of the current chunk.
    chunk_tokens: Number of flat tokens in the current chunk (`0` if unchunked).
  """

  info: Any
  batch_tile_idx: Any
  bq_idx: Any
  q_start: Any
  q_end: Any
  step_qb: Any
  chunk_start: int = dataclasses.field(default=0, metadata=dict(static=True))
  chunk_tokens: int = dataclasses.field(default=0, metadata=dict(static=True))

  @property
  def start_seq(self) -> jax.Array:
    """First sequence index of this pass."""
    return self.info[_INFO_START_SEQ]

  @property
  def end_seq(self) -> jax.Array:
    """Exclusive end sequence index of this pass."""
    return self.info[_INFO_END_SEQ]

  @property
  def num_steps(self) -> jax.Array:
    """Number of pipeline steps in this pass."""
    return self.info[_INFO_NUM_STEPS]

  def lookup(
      self, step: int | jax.Array
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Returns `(batch_tile_idx, bq_idx, cur_bkv, is_last_bkv, q_blk_idx)`."""
    step = jnp.asarray(step, dtype=jnp.int32)
    if self.step_qb is None:
      return metadata.lookup_qblock_step(
          metadata.MetadataRef.create(
              num_steps=None,
              batch_tile_idx=self.batch_tile_idx,
              bq_idx=self.bq_idx,
              bkv_idx=self.q_end,
          ),
          step,
      )
    qb = self.step_qb[step]
    return (
        self.batch_tile_idx[qb],
        self.bq_idx[qb],
        step - self.q_start[qb],
        step + 1 >= self.q_end[qb],
        qb,
    )


def schedule_capacity(
    *,
    max_num_seqs: int,
    max_num_tokens: int,
    seq_batch_size: int,
    bq_sz: int,
    static_q_len: int | None,
    chunk_tokens: int | None,
    num_kv_blocks: int,
) -> tuple[int, int]:
  """Static `(num_q_blocks, num_step_table_entries)` capacity of a pass.

  The query-block bound matches `metadata.compute_batched_seq_metadata` for
  one chunk. The step table is omitted (0 entries) when it would be too large.

  Args:
    max_num_seqs: Number of sequence slots.
    max_num_tokens: Number of query tokens.
    seq_batch_size: Sequences per query block.
    bq_sz: Query tokens per query block.
    static_q_len: Static query length of every sequence of the pass, or None.
    chunk_tokens: Chunk length when the token axis is chunked, else None.
    num_kv_blocks: Maximum number of KV blocks per query block.

  Returns:
    `(num_q_blocks, num_step_table_entries)`.
  """
  max_seq_tiles = cdiv(max_num_seqs, seq_batch_size)
  if static_q_len == 1:
    q_blocks = max_seq_tiles
  elif static_q_len is not None:
    q_blocks = max_seq_tiles * max(1, cdiv(static_q_len, bq_sz))
  else:
    q_blocks = (
        max(1, cdiv(metadata.next_power_of_2(max_num_tokens), bq_sz))
        + max_seq_tiles
    )
  if chunk_tokens is not None:
    q_blocks = min(
        q_blocks,
        cdiv(chunk_tokens, seq_batch_size * bq_sz)
        + min(max_seq_tiles, chunk_tokens),
    )
  q_blocks = max(1, q_blocks)
  table = q_blocks * num_kv_blocks
  return q_blocks, (table if table <= _MAX_STEP_TABLE_ENTRIES else 0)


def build_schedule(
    sched: ScheduleRef,
    seq_lens_ref: Any,
    cu_q_lens_ref: Any,
    dist_ref: Any,
    cp_rank: Any,
    *,
    start_bound: SeqBound,
    end_bound: SeqBound,
    seq_batch_size: int,
    bq_sz: int,
    bkv_sz: int,
    num_kv_blocks: int,
    compression_ratio: int,
    static_q_len: int | None,
    chunk_tokens: int | None,
    cp_size: int,
    interleave_c: int,
) -> None:
  """Fills `sched` with the query-block schedule of one fused pass.

  Scalar-core equivalent of `metadata.compute_batched_seq_metadata(...,
  qblock_schedule=True)` for a single chunk: the same query blocks, in the same
  order, with the same number of KV blocks each.

  Args:
    sched: SMEM refs to fill.
    seq_lens_ref: `[max_num_seqs]` uncompressed sequence lengths.
    cu_q_lens_ref: `[max_num_seqs + 1]` cumulative query lengths.
    dist_ref: `[3]` request distribution.
    cp_rank: This rank's context-parallel index.
    start_bound: First sequence of the pass (`None` means sequence `0`).
    end_bound: End sequence (exclusive) of the pass (`None` means sequence `0`).
    seq_batch_size: Sequences per query block.
    bq_sz: Query tokens per query block.
    bkv_sz: KV positions per KV block.
    num_kv_blocks: Maximum number of KV blocks per query block.
    compression_ratio: KV compression ratio.
    static_q_len: Static query length of every sequence of the pass, or None.
    chunk_tokens: Chunk length when the token axis is chunked, else None.
    cp_size: Number of context-parallel ranks.
    interleave_c: Context-parallel interleave in compressed positions.
  """
  num_seqs = seq_lens_ref.shape[0]
  num_tiles = cdiv(num_seqs, seq_batch_size)
  q_cap = sched.q_end.shape[0]
  start = _seq_bound(dist_ref, start_bound)
  end = _seq_bound(dist_ref, end_bound)
  sched.info[_INFO_START_SEQ] = start
  sched.info[_INFO_END_SEQ] = end
  # Tile `t` is in the pass iff `start <= t * seq_batch_size < end`.
  t_lo = jnp.clip(
      (start + (seq_batch_size - 1)) // seq_batch_size, 0, num_tiles
  )
  t_hi_raw = (end + (seq_batch_size - 1)) // seq_batch_size
  t_hi = jnp.maximum(t_lo, jnp.clip(t_hi_raw, 0, num_tiles))
  chunked = chunk_tokens is not None
  span = seq_batch_size * bq_sz

  def _local_len(kv_len):
    return metadata.cp_local_length(kv_len, cp_rank, cp_size, interleave_c)

  def _num_bkv(kv_len):
    return jnp.minimum(
        num_kv_blocks, jnp.maximum(1, pl.cdiv(_local_len(kv_len), bkv_sz))
    )

  def _tile(t, carry):
    tile_seq = t * seq_batch_size
    lo = hi = None
    if chunked:
      tile_tok_start = cu_q_lens_ref[jnp.minimum(tile_seq, num_seqs)]
      tile_tok_end = cu_q_lens_ref[
          jnp.minimum(tile_seq + seq_batch_size, num_seqs)
      ]
      lo = jnp.clip(sched.chunk_start, tile_tok_start, tile_tok_end)
      hi = jnp.clip(
          sched.chunk_start + chunk_tokens, tile_tok_start, tile_tok_end
      )
      num_bq = pl.cdiv(hi - lo, span)
    elif static_q_len is not None:
      num_bq = max(1, cdiv(static_q_len, bq_sz))
    else:
      assert seq_batch_size == 1
      q_len = cu_q_lens_ref[t + 1] - cu_q_lens_ref[t]
      num_bq = jnp.maximum(1, pl.cdiv(q_len, bq_sz))

    tile_bkv = None
    if seq_batch_size > 1:
      max_seq = jnp.int32(0)
      for j in range(seq_batch_size):
        s = tile_seq + j
        valid = (s >= start) & (s < end) & (s < num_seqs)
        seq_len = jnp.where(
            valid, seq_lens_ref[jnp.minimum(s, num_seqs - 1)], 0
        )
        max_seq = jnp.maximum(max_seq, seq_len)
      tile_bkv = _num_bkv(max_seq // compression_ratio)

    def _block(b, carry):
      qb, step = carry
      if tile_bkv is not None:
        nbkv = tile_bkv
      else:
        seq = jnp.minimum(tile_seq, num_seqs - 1)
        seq_q_end = cu_q_lens_ref[seq + 1]
        if lo is not None and hi is not None:
          seg_lo, seg_hi = lo, hi
        else:
          seg_lo, seg_hi = cu_q_lens_ref[seq], seq_q_end
        blk_start = seg_lo + b * bq_sz
        blk_end = jnp.minimum(seg_hi, blk_start + bq_sz)
        seq_len = seq_lens_ref[seq]
        kv_len = seq_len // compression_ratio
        max_q_abs_pos = seq_len - (seq_q_end - (blk_end - 1))
        vis = jnp.where(
            blk_end > blk_start,
            jnp.clip(max_q_abs_pos // compression_ratio + 1, 0, kv_len),
            0,
        )
        nbkv = _num_bkv(vis)
      in_cap = qb < q_cap
      step_end = step + nbkv

      @pl.when(in_cap)
      def _():
        sched.batch_tile_idx[qb] = tile_seq
        sched.bq_idx[qb] = b
        sched.q_start[qb] = step
        sched.q_end[qb] = step_end
        if sched.step_qb is not None:

          def _fill(s, _):
            sched.step_qb[s] = qb
            return None

          lax.fori_loop(step, step_end, _fill, None)

      return qb + 1, jnp.where(in_cap, step_end, step)

    return lax.fori_loop(0, num_bq, _block, carry)

  num_qb, num_steps = lax.fori_loop(
      t_lo, t_hi, _tile, (jnp.int32(0), jnp.int32(0))
  )
  sched.info[_INFO_NUM_STEPS] = num_steps
  if sched.step_qb is None:
    # The binary search reads every slot; mark unused ones as never reached.
    def _seal(qb, _):
      sched.q_end[qb] = jnp.int32(metadata.QBLOCK_END_SENTINEL)
      return None

    lax.fori_loop(jnp.minimum(num_qb, q_cap), q_cap, _seal, None)


# --------------------------------------------------------------------------- #
# Custom BufferedRefs driven by a `ScheduleRef`.
# --------------------------------------------------------------------------- #


def _qblock_token_window(
    cu_q_lens_ref: Any,
    sched: ScheduleRef,
    p_id: int | jax.Array,
    chunk_tokens: int,
    *,
    bq_sz: int,
    seq_batch_size: int,
):
  """Flat-token rows `(token_start, total_sz)` for step `p_id`."""
  chunk_start = sched.chunk_start
  seq_idx, bq_idx, _, _, _ = sched.lookup(p_id)
  max_seq = cu_q_lens_ref.shape[0] - 1
  seq_clamped = jnp.minimum(seq_idx, max_seq - 1)
  lo = jnp.maximum(cu_q_lens_ref[seq_clamped], chunk_start)

  if seq_batch_size > 1:
    return lo, seq_batch_size
  token_start = lo + bq_idx * bq_sz
  hi = jnp.minimum(cu_q_lens_ref[seq_clamped + 1], chunk_start + chunk_tokens)
  return token_start, jnp.maximum(0, jnp.minimum(bq_sz, hi - token_start))


def _contiguous_pages(page_indices_ref, first_entry, num_pages, total_pages):
  """Returns `(first_page, contiguous)` for `num_pages` page-table entries."""
  num_entries = page_indices_ref.shape[0]

  def _page(i):
    return page_indices_ref[jnp.minimum(first_entry + i, num_entries - 1)]

  first = _page(0)
  # Four independent OR chains of `page_i - (first + i)` keep dependency chains
  # short without holding every entry live in scalar registers.
  num_chains = 4
  chains = [jnp.int32(0)] * num_chains
  for i in range(1, num_pages):
    c = i % num_chains
    chains[c] = chains[c] | (_page(i) - (first + i))
  mismatch = functools.reduce(operator.or_, chains)
  in_range = (first >= 0) & (first <= total_pages - num_pages)
  return first, in_range & (mismatch == 0)


def _copy_kv_block(
    page_indices_ref: Any,
    *,
    first_entry: Callable[[int], jax.Array],
    seq_batch_size: int,
    bkv_p: int,
    pages_per_seq: int,
    total_pages: int,
    copy_pages: Callable[[int, int, jax.Array], None],
    copy_page: Callable[[int, int, jax.Array], None],
) -> None:
  """Starts the DMAs of one KV block (`seq_batch_size` x `bkv_p` pages).

  Runs of consecutive cache pages are fetched with one DMA: the whole block
  when every sequence's single KV block is contiguous with the next, else one
  per sequence whose pages are consecutive. Everything else falls back to one
  DMA per page. Every path moves the same bytes to the same place.

  Args:
    page_indices_ref: Flat SMEM page table.
    first_entry: `b -> page-table index` of sequence `b`'s first page.
    seq_batch_size: Sequences in the block.
    bkv_p: Pages per sequence in the block.
    pages_per_seq: Page-table entries per sequence.
    total_pages: Number of pages in the KV cache.
    copy_pages: `(b, num_seqs, first_page)` starts one DMA of `num_seqs * bkv_p`
      pages into sequences `[b, b + num_seqs)`.
    copy_page: `(b, i, page)` starts the DMA of page `i` of sequence `b`.
  """
  num_entries = page_indices_ref.shape[0]

  def _copy_one(b, i):
    entry = jnp.minimum(first_entry(b) + i, num_entries - 1)
    copy_page(b, i, page_indices_ref[entry])

  # The per-page fallbacks are rolled loops with zero trips when a run DMA
  # covered the pages, not branches: the compiler would otherwise predicate
  # (and pay for) every per-page DMA on the fast path too.
  if seq_batch_size > 1 and pages_per_seq == bkv_p:
    first, contiguous = _contiguous_pages(
        page_indices_ref, first_entry(0), seq_batch_size * bkv_p, total_pages
    )

    @pl.when(contiguous)
    def _():
      copy_pages(0, seq_batch_size, first)

    def _seq_pages(b, _):
      for i in range(bkv_p):
        _copy_one(b, i)
      return None

    lax.fori_loop(
        0,
        jnp.where(contiguous, 0, seq_batch_size).astype(jnp.int32),
        _seq_pages,
        None,
    )
    return

  chunk = math.gcd(bkv_p, 8)
  for b in range(seq_batch_size):
    if bkv_p == 1:
      _copy_one(b, 0)
      continue
    first, contiguous = _contiguous_pages(
        page_indices_ref, first_entry(b), bkv_p, total_pages
    )

    @pl.when(contiguous)
    def _(b=b, first=first):
      copy_pages(b, 1, first)

    def _chunk_pages(c, _, b=b):
      for j in range(chunk):
        _copy_one(b, c * chunk + j)
      return None

    lax.fori_loop(
        0,
        jnp.where(contiguous, 0, bkv_p // chunk).astype(jnp.int32),
        _chunk_pages,
        None,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QBlockKVBufferedRef(pltpu.BufferedRef):
  """Paged KV `BufferedRef` (HAS and SAL layouts) driven by a `ScheduleRef`.

  Attributes:
    bkv_p: Number of KV pages per sequence in one pipeline block.
    pages_per_seq: Page-table entries per sequence.
    seq_batch_size: Number of sequences fetched per block.
    seq_along_lane: True for `SEQ_ALONG_LANE`, False for `HEAD_ALONG_SUBLANE`.
  """

  bkv_p: int = dataclasses.field(default=1, metadata=dict(static=True))
  pages_per_seq: int = dataclasses.field(default=1, metadata=dict(static=True))
  seq_batch_size: int = dataclasses.field(default=1, metadata=dict(static=True))
  seq_along_lane: bool = dataclasses.field(
      default=False, metadata=dict(static=True)
  )

  @override
  def copy_in(
      self,
      src_ref: tuple[Any, ...],
      grid_indices: tuple[int | jax.Array, ...],
  ) -> None:
    cache_kv_hbm, page_indices_ref, sched = src_ref
    batch_tile_idx, _, bkv_idx, _, _ = sched.lookup(grid_indices[0])
    assert self.sem_recvs is not None and self.window_ref is not None
    slot = self.current_copy_in_slot
    vmem_dst: Any = self.window_ref.at[slot]
    sem = self.sem_recvs.at[slot]
    total_pages = cache_kv_hbm.shape[0]

    if self.seq_along_lane:

      def _copy_pages(b, num_seqs, first):
        dst = (
            vmem_dst.at[b]
            if num_seqs == 1
            else vmem_dst.reshape(num_seqs * self.bkv_p, *vmem_dst.shape[2:])
        )
        pltpu.make_async_copy(
            cache_kv_hbm.at[pl.ds(first, num_seqs * self.bkv_p)], dst, sem
        ).start()

      def _copy_page(b, i, page):
        safe_page = jnp.clip(page, 0, max(0, total_pages - 1))
        pltpu.make_async_copy(
            cache_kv_hbm.at[safe_page],
            vmem_dst.at[b, i],
            sem,
        ).start()

    else:
      packed_rows_per_page = cache_kv_hbm.shape[1]
      rows = self.bkv_p * packed_rows_per_page
      flat_hbm: Any = cache_kv_hbm.reshape(
          total_pages * packed_rows_per_page, *cache_kv_hbm.shape[2:]
      )

      def _copy_pages(b, num_seqs, first):
        dst = (
            vmem_dst.at[b, pl.ds(0, rows)]
            if num_seqs == 1
            else vmem_dst.reshape(num_seqs * rows, *vmem_dst.shape[2:])
        )
        pltpu.make_async_copy(
            flat_hbm.at[pl.ds(first * packed_rows_per_page, num_seqs * rows)],
            dst,
            sem,
        ).start()

      def _copy_page(b, i, page):
        safe_page = jnp.clip(page, 0, max(0, total_pages - 1))
        pltpu.make_async_copy(
            flat_hbm.at[
                pl.ds(safe_page * packed_rows_per_page, packed_rows_per_page)
            ],
            vmem_dst.at[
                b, pl.ds(i * packed_rows_per_page, packed_rows_per_page)
            ],
            sem,
        ).start()

    _copy_kv_block(
        page_indices_ref,
        first_entry=lambda b: (batch_tile_idx + b) * self.pages_per_seq
        + bkv_idx * self.bkv_p,
        seq_batch_size=self.seq_batch_size,
        bkv_p=self.bkv_p,
        pages_per_seq=self.pages_per_seq,
        total_pages=total_pages,
        copy_pages=_copy_pages,
        copy_page=_copy_page,
    )

  @override
  def wait_in(
      self,
      src_ref: tuple[Any, ...],
      grid_indices: tuple[int | jax.Array, ...],
  ) -> None:
    del src_ref, grid_indices
    assert self.is_input
    if not self.is_buffered:
      return
    assert self.sem_recvs is not None and self.window_ref is not None
    slot = self.current_wait_in_slot
    vmem_dst: Any = self.window_ref.at[slot]
    pltpu.make_async_copy(vmem_dst, vmem_dst, self.sem_recvs.at[slot]).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QBlockTokenBufferedRef(pltpu.BufferedRef):
  """Flat-token input (`q`, `weights`) and output (`out`) BufferedRef.

  Attributes:
    bq_sz: Query tokens per query block.
    seq_batch_size: Sequences per query block.
  """

  bq_sz: int = dataclasses.field(default=16, metadata=dict(static=True))
  seq_batch_size: int = dataclasses.field(default=1, metadata=dict(static=True))

  def _window(self, refs: tuple[Any, ...], p_id: int | jax.Array):
    hbm_ref, cu_q_lens_ref, sched = refs
    chunk_tokens = sched.chunk_tokens or hbm_ref.shape[0]
    return _qblock_token_window(
        cu_q_lens_ref,
        sched,
        p_id,
        chunk_tokens,
        bq_sz=self.bq_sz,
        seq_batch_size=self.seq_batch_size,
    )

  @override
  def copy_in(
      self,
      src_ref: tuple[Any, ...],
      grid_indices: tuple[int | jax.Array, ...],
  ) -> None:
    q_hbm = src_ref[0]
    assert self.sem_recvs is not None and self.window_ref is not None
    slot = self.current_copy_in_slot
    q_len_start, total_sz = self._window(src_ref, grid_indices[0])
    safe_start = jnp.minimum(
        q_len_start, jnp.maximum(0, q_hbm.shape[0] - jnp.maximum(1, total_sz))
    )
    pltpu.make_async_copy(
        q_hbm.at[pl.ds(safe_start, total_sz)],
        self.window_ref.at[slot].at[pl.ds(0, total_sz)],
        self.sem_recvs.at[slot],
    ).start()

  @override
  def wait_in(
      self,
      src_ref: tuple[Any, ...],
      grid_indices: tuple[int | jax.Array, ...],
  ) -> None:
    assert self.is_input
    if not self.is_buffered:
      return
    assert self.sem_recvs is not None and self.window_ref is not None
    slot = self.current_wait_in_slot
    _, total_sz = self._window(src_ref, grid_indices[0])
    sliced = self.window_ref.at[slot].at[pl.ds(0, total_sz)]
    pltpu.make_async_copy(sliced, sliced, self.sem_recvs.at[slot]).wait()

  @override
  def copy_out(
      self,
      dst_ref: tuple[Any, ...],
      grid_indices: tuple[int | jax.Array, ...],
  ) -> None:
    out_hbm, _, sched = dst_ref
    assert self.sem_sends is not None and self.window_ref is not None
    slot = self.current_copy_out_slot
    token_start, total_sz = self._window(dst_ref, grid_indices[0])
    dst_start = jnp.minimum(
        jnp.maximum(0, token_start - sched.chunk_start),
        jnp.maximum(0, out_hbm.shape[0] - jnp.maximum(1, total_sz)),
    )
    pltpu.make_async_copy(
        self.window_ref.at[slot].at[pl.ds(0, total_sz)],
        out_hbm.at[pl.ds(dst_start, total_sz)],
        self.sem_sends.at[slot],
    ).start()

  @override
  def wait_out(
      self,
      dst_ref: tuple[Any, ...],
      grid_indices: tuple[int | jax.Array, ...],
  ) -> None:
    assert self.is_output
    if not self.is_buffered:
      return
    assert self.sem_sends is not None and self.window_ref is not None
    slot = self.current_wait_out_slot
    _, total_sz = self._window(dst_ref, grid_indices[0])
    sliced = self.window_ref.at[slot].at[pl.ds(0, total_sz)]
    pltpu.make_async_copy(sliced, sliced, self.sem_sends.at[slot]).wait()


# --------------------------------------------------------------------------- #
# Fused scoring + in-VMEM TensorCore Top-K kernel.
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class _PassConfig:
  """Static parameters of one fused pass inside `tc_kernel`."""

  k: int
  need_scores: bool
  num_positions: int
  compression_ratio: int
  static_q_len: int | None
  bq_sz: int
  bkv_p: int
  seq_batch_size: int
  buffer_count: int
  kv_layout: KVLayout
  cp_size: int
  interleave_c: int
  topk_impl: str
  start_bound: SeqBound
  end_bound: SeqBound
  chunk_start: int | None


def _is_blocked_prefill_enabled(
    bq_sz: int, q_dtype: Any, seq_batch_size: int
) -> bool:
  """Returns whether multi-MXU blocked prefill scoring applies to a block."""
  q_sublane_tiling = 8 * get_dtype_packing(q_dtype)
  return (
      seq_batch_size == 1
      and bq_sz > 1
      and math.gcd(bq_sz, 32) % q_sublane_tiling == 0
  )


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

    enabled = _is_blocked_prefill_enabled(bq_sz, q_dtype, seq_batch_size)
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


def _fori(
    n: int | jax.Array,
    fn: Callable[[int | jax.Array, Any], Any],
    carry: Any = None,
) -> Any:
  """Runs `carry = fn(i, carry)` for `i` in `range(n)`, unrolling `n == 1`."""
  if isinstance(n, int) and n == 1:
    return fn(0, carry)
  return lax.fori_loop(0, n, fn, carry)


def _sublane_transpose_sum(parts: Sequence[jax.Array]) -> jax.Array:
  """Reduces 8 `(8, n)` parts into one `(8, n)` tile (`out[r] = sum(parts[r])`).

  Uses a 3-stage (`d = 4, 2, 1`) sublane butterfly roll-and-blend so all 8 rows
  reduce across their 8 sublanes concurrently without scalar gathers.

  Args:
    parts: Eight `(8, n)` arrays, one per output sublane `r`.

  Returns:
    `(8, n)` array whose sublane `r` equals `parts[r].sum(axis=0)`.
  """
  sub = lax.broadcasted_iota(jnp.int32, parts[0].shape, 0)
  d = 4
  while d >= 1:
    half = len(parts) // 2
    keep = (sub & d) == 0
    parts = [
        jnp.where(
            keep,
            a + pltpu.roll(a, 8 - d, 0),
            b + pltpu.roll(b, d, 0),
        )
        for a, b in zip(parts[:half], parts[half:], strict=True)
    ]
    d //= 2
  return parts[0]


def _tc_kernel_body(
    seq_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    sched: ScheduleRef,
    cp_rank_ref,  # [1]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    indexer_weights_hbm_ref,  # [max_num_tokens, num_q_heads]
    cache_kv_hbm_ref,
    out_hbm_ref,
    cfg: _PassConfig,
):
  """Executes fused MXU scoring + in-VMEM TensorCore Top-K."""
  k = cfg.k
  need_scores = cfg.need_scores
  num_positions = cfg.num_positions
  compression_ratio = cfg.compression_ratio
  static_q_len = cfg.static_q_len
  bkv_p = cfg.bkv_p
  bq_sz = cfg.bq_sz
  seq_batch_size = cfg.seq_batch_size
  buffer_count = cfg.buffer_count
  cp_size = cfg.cp_size
  interleave_c = cfg.interleave_c
  seq_along_lane = cfg.kv_layout == KVLayout.SEQ_ALONG_LANE
  chunk_tokens = out_hbm_ref.shape[0]

  _, num_q_heads, head_dim = q_hbm_ref.shape
  page_size = _kv_page_size(cache_kv_hbm_ref.shape, cfg.kv_layout)
  kv_packing = cache_kv_hbm_ref.shape[2]
  kv_sublane_groups = cache_kv_hbm_ref.shape[1]
  kv_head_dim_groups = head_dim // kv_packing
  page_size_per_kv_packing = page_size // kv_packing

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
  hm_tok = align_to(bq_sz, 8)

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
  # HEAD_ALONG_SUBLANE single-token rows (`bq_sz == 1`) are scored straight
  # into `(8, 128)` top-k tiles one key chunk at a time (see
  # `_decode_score_tiles`), so per-row keys never stay live across the block.
  decode_tiles = (
      not seq_along_lane
      and bq_sz == 1
      and not prefill_cfg.enabled
      and num_q_heads % 8 == 0
  )

  single_bkv = pages_per_seq <= bkv_p
  use_select = cfg.topk_impl == TOPK_IMPL_SELECT
  num_kv_blocks = cdiv(pages_per_seq, bkv_p)
  select_idx_bits = max(1, (num_positions - 1).bit_length())
  m_part = num_sublanes_bkv // num_partitions
  use_two_level = use_select and m_part >= 4 and num_kv_blocks >= 4 * num_blocks
  subbatch_w = math.gcd(m_part, min(num_blocks, 4))
  num_subbatches = m_part // subbatch_w
  num_direct_chunks = cdiv(m_part, num_blocks)

  # Lists never need more levels than a lane can see tiles per query block.
  max_tiles_per_lane = cdiv(
      align_to(pages_per_seq, bkv_p) * page_size, 128 * num_partitions
  )
  lane_depth = min(
      tc_topk.lane_list_depth(num_blocks),
      align_to(max_tiles_per_lane, num_blocks),
  )
  ins_w = math.gcd(m_part, 8)
  num_ins_chunks = m_part // ins_w

  def q_blk_id(p_id):
    return sched.lookup(p_id)[4]

  def kv_sal_idx(p_id):
    bt, _, cbkv, _, _ = sched.lookup(p_id)
    return (bt, cbkv, 0, 0, 0)

  def kv_has_idx(p_id):
    bt, _, cbkv, _, _ = sched.lookup(p_id)
    return (bt, 0, cbkv, 0)

  in_mode = pl.Buffered(buffer_count=buffer_count, use_lookahead=True)
  q_spec = pl.BlockSpec(
      (pl.BoundedSlice(bq_total), num_q_heads, head_dim),
      lambda p_id: (q_blk_id(p_id), 0, 0),
      pipeline_mode=in_mode,
  )
  weights_spec = pl.BlockSpec(
      (pl.BoundedSlice(bq_total), num_q_heads),
      lambda p_id: (q_blk_id(p_id), 0),
      pipeline_mode=in_mode,
  )
  if seq_along_lane:
    kv_spec = pl.BlockSpec(
        (
            pl.BoundedSlice(seq_batch_size),
            bkv_p,
            kv_sublane_groups,
            kv_packing,
            page_size,
        ),
        kv_sal_idx,
        pipeline_mode=in_mode,
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
        pipeline_mode=in_mode,
    )
  out_spec = pl.BlockSpec(
      (pl.BoundedSlice(bq_alloc), num_out_blocks, 128),
      lambda p_id: (q_blk_id(p_id), 0, 0),
      pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
  )

  tok_fields = dict(bq_sz=bq_sz, seq_batch_size=seq_batch_size)
  q_alloc = dataclasses.replace(
      QBlockTokenBufferedRef.input(
          q_spec,
          q_hbm_ref,
          buffer_count=buffer_count,
          grid_rank=1,
          use_lookahead=True,
      ),
      **tok_fields,
  )
  weights_alloc = dataclasses.replace(
      QBlockTokenBufferedRef.input(
          weights_spec,
          indexer_weights_hbm_ref,
          buffer_count=buffer_count,
          grid_rank=1,
          use_lookahead=True,
      ),
      **tok_fields,
  )
  kv_alloc = dataclasses.replace(
      QBlockKVBufferedRef.input(
          kv_spec,
          cache_kv_hbm_ref,
          buffer_count=buffer_count,
          grid_rank=1,
          use_lookahead=True,
      ),
      bkv_p=bkv_p,
      pages_per_seq=pages_per_seq,
      seq_batch_size=seq_batch_size,
      seq_along_lane=seq_along_lane,
  )
  o_alloc = dataclasses.replace(
      QBlockTokenBufferedRef.output(
          out_spec,
          out_hbm_ref,
          buffer_count=2,
          grid_rank=1,
          use_lookahead=False,
      ),
      **tok_fields,
  )

  # Unified KV block view + tile loader for both SEQ_ALONG_LANE and
  # HEAD_ALONG_SUBLANE layouts.
  num_lane_tiles_bkv = bkv_sz // 128
  lane_tiles_per_page = page_size // 128
  has_rec_tiles = cache_kv_hbm_ref.shape[-1] // 128
  has_key_tiles = head_dim // 128
  groups_per_tile = 128 // kv_packing
  qk_spec = "nd,dm->nm" if seq_along_lane else "nd,md->nm"

  def qk(q_rows, keys):
    return jnp.einsum(qk_spec, q_rows, keys, preferred_element_type=jnp.float32)

  def kv_block_rows(bkv_vmem):
    if seq_along_lane:
      return bkv_vmem.bitcast(jnp.uint32).reshape(
          seq_batch_size, kv_sublane_groups * num_lane_tiles_bkv, 128
      )
    return bkv_vmem.bitcast(jnp.uint32).reshape(
        seq_batch_size, bkv_sz_per_kv_packing * has_rec_tiles, 128
    )

  def _has_words(
      rows, batch_idx, tile_start: int, num_tiles: int, tile_idx: int
  ):
    group_start = tile_start * groups_per_tile
    groups = num_tiles * groups_per_tile
    return rows[
        batch_idx,
        pl.ds(
            group_start * has_rec_tiles + tile_idx,
            groups,
            stride=has_rec_tiles,
        ),
        :,
    ]

  def load_kv_keys(rows, batch_idx, tile_start: int, num_tiles: int):
    if seq_along_lane:
      tiles = []
      for t in range(tile_start, tile_start + num_tiles):
        page, tile = divmod(t, lane_tiles_per_page)
        base = page * kv_sublane_groups * lane_tiles_per_page + tile
        words = rows[
            batch_idx,
            pl.ds(base, kv_head_dim_groups, stride=lane_tiles_per_page),
            :,
        ]
        tiles.append(pltpu.bitcast(words, jnp.float8_e4m3fn))
      return jnp.concatenate(tiles, axis=1)
    return jnp.concatenate(
        [
            pltpu.bitcast(
                _has_words(rows, batch_idx, tile_start, num_tiles, t),
                jnp.float8_e4m3fn,
            )
            for t in range(has_key_tiles)
        ],
        axis=1,
    )

  def load_kv_scales(rows, batch_idx, tile_start: int, num_tiles: int):
    # `float8_e8m0fnu` stores an unsigned biased exponent `e` representing
    # `2**(e - 127)` (`e == 255` is NaN). Shifting `1 <= e <= 254` into
    # IEEE-754 float32 exponent bits `[23:31]` (`e << 23`) decodes it directly;
    # HAS also preserves exact `e == 0` (`2**-127`, subnormal in float32) and
    # `e == 255` (`NaN`, non-zero mantissa in float32).
    if seq_along_lane:
      tiles = []
      for t in range(tile_start, tile_start + num_tiles):
        page, tile = divmod(t, lane_tiles_per_page)
        scale_row = (
            page * kv_sublane_groups + kv_head_dim_groups
        ) * lane_tiles_per_page + tile
        words = rows[batch_idx, pl.ds(scale_row, 1), :]
        tiles.append(pltpu.bitcast((words & 0xFF) << 23, jnp.float32))
      return jnp.concatenate(tiles, axis=1)
    scale_bytes = pltpu.bitcast(
        _has_words(rows, batch_idx, tile_start, num_tiles, has_key_tiles),
        jnp.uint8,
    )
    e = scale_bytes[:, 0:1].T.astype(jnp.int32)
    scales = pltpu.bitcast(e << 23, jnp.float32)
    scales = jnp.where(e == 0, jnp.float32(2.0**-127), scales)
    return jnp.where(e == 255, jnp.float32(jnp.nan), scales)

  def load_kv_tiles(rows, batch_idx, tile_start: int, num_tiles: int):
    """Loads `(keys, scales)` for 128-token tiles `[tile_start, +num_tiles)`."""
    return (
        load_kv_keys(rows, batch_idx, tile_start, num_tiles),
        load_kv_scales(rows, batch_idx, tile_start, num_tiles),
    )

  def transpose_to_head_major(bq_rows, n_tok):
    return bq_rows.transpose(1, 0, 2).reshape(num_q_heads * n_tok, head_dim)

  def pad_tokens(bq_rows, n_tok):
    if n_tok == hm_tok:
      return bq_rows
    pad = jnp.zeros((hm_tok - n_tok, *bq_rows.shape[1:]), bq_rows.dtype)
    return jnp.concatenate([bq_rows, pad], axis=0)

  def reduce_heads(bq_rows, bkv, bq_weights, head_major=False):
    n_tok = bq_weights.shape[0]
    w = bq_weights.astype(jnp.float32)
    if n_tok == 1:
      st = qk(bq_rows, bkv).reshape(n_tok, num_q_heads, -1)
      st = jnp.maximum(st, 0.0) * w[:, :, None]
      return st.sum(axis=1)

    if head_major:
      rows_hm, rows_tok = bq_rows, hm_tok
      w = jnp.pad(w, ((0, rows_tok - n_tok), (0, 0)))
    else:
      rows_hm, rows_tok = transpose_to_head_major(bq_rows, n_tok), n_tok
    st = qk(rows_hm, bkv).reshape(num_q_heads, rows_tok, -1)
    acc = jnp.maximum(st[0], 0.0) * w[:, 0:1]
    for h in range(1, num_q_heads):
      acc = acc + jnp.maximum(st[h], 0.0) * w[:, h : h + 1]
    return acc[:n_tok]

  def _active_q_groups(p_id):
    """Number of query groups holding rows of this query block."""
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
        cu_q_lens_ref,
        sched,
        p_id,
        chunk_tokens,
        bq_sz=bq_sz,
        seq_batch_size=1,
    )
    return (total_sz + 7) // 8

  def _for_group_pairs(n_act, fn):
    """Runs `fn(groups)` over the active query groups, two at a time."""
    if isinstance(n_act, int) and n_act == 1:
      fn((jnp.int32(0),))
      return
    lax.fori_loop(
        0,
        (n_act + 1) // 2,
        lambda p, _: fn((2 * p, jnp.minimum(2 * p + 1, n_act - 1))),
        None,
    )

  def _write_group_out(out_vmem, g, fin_o, fin_i):
    """Writes one query group's final Top-K indices (+ score bits)."""
    idx_2d = jnp.concatenate(fin_i, axis=1)
    if need_scores:
      scores_2d = jnp.concatenate(
          [
              tc_topk.ord_i32_to_f32_bits(o, i >= 0)
              for o, i in zip(fin_o, fin_i, strict=True)
          ],
          axis=1,
      )
      group_out = jnp.concatenate([idx_2d, scores_2d], axis=1).reshape(
          8, num_out_blocks, 128
      )
    else:
      group_out = idx_2d.reshape(8, num_out_blocks, 128)
    out_vmem[pl.ds(g * 8, 8)] = group_out

  def _finish_lane_major_group(out_vmem, g, fin_o, fin_i):
    """Reduces sublane partitions, transposes to block-major, and writes `g`."""
    if num_partitions > 1:
      fin_o, fin_i = tc_topk.reduce_sublane_partitions(
          fin_o, fin_i, num_partitions=num_partitions
      )
    fin_i = tc_topk.lane_major_to_block_major(fin_i)
    if need_scores:
      fin_o = tc_topk.lane_major_to_block_major(fin_o)
    _write_group_out(out_vmem, g, fin_o, fin_i)

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

  # Build a single PyTree of scoped VMEM scratches (`None` leaves are skipped).
  use_stream = not use_select and not single_bkv
  vec_vmem = pltpu.VMEM((num_q_groups, 8, 128), jnp.int32)
  topk_vmem = pltpu.VMEM((num_q_groups, num_blocks, 8, 128), jnp.int32)
  lane_vmem = pltpu.VMEM(
      (num_q_groups, lane_depth + num_blocks, 8, 128), jnp.int32
  )
  ev_vmem = pltpu.VMEM((num_q_groups, m_part, 8, 128), jnp.int32)
  cells_vmem = pltpu.VMEM((num_q_groups, num_kv_blocks, 8, 128), jnp.int32)

  q_hm_spec = None
  wb_spec = None
  if prefill_cfg.enabled:
    q_hm_spec = pltpu.VMEM((num_q_heads * bq_sz, head_dim), q_hbm_ref.dtype)
    wb_spec = pltpu.VMEM(
        (prefill_cfg.num_tok_blks, num_q_heads, prefill_cfg.tok_blk, 128),
        prefill_cfg.wb_dtype,
    )
  elif seq_batch_size == 1 and bq_sz > 1:
    q_hm_spec = pltpu.VMEM((num_q_heads * hm_tok, head_dim), q_hbm_ref.dtype)

  scores_vmem = pltpu.VMEM((num_q_groups, m_part, 8, 128), jnp.float32)
  keys_vmem = pltpu.VMEM(
      (num_q_groups, num_kv_blocks * m_part, 8, 128), jnp.int32
  )
  scratch_spec = dict(
      eff_len=vec_vmem,
      scores=None if use_select else scores_vmem,
      top_o=topk_vmem if (use_select or use_stream) else None,
      top_i=topk_vmem if (use_select or use_stream) else None,
      cand_o=topk_vmem if use_stream else None,
      cand_i=topk_vmem if use_stream else None,
      tau=vec_vmem if use_stream else None,
      lst_o=lane_vmem if use_stream else None,
      lst_i=lane_vmem if use_stream else None,
      ev_o=ev_vmem if use_stream else None,
      ev_i=ev_vmem if use_stream else None,
      keys=keys_vmem if use_select else None,
      cell_max=cells_vmem if use_two_level else None,
      cell_arg=cells_vmem if use_two_level else None,
      q_hm=q_hm_spec,
      wb=wb_spec,
  )

  def _run_with_scratches(scratch, final_allocs):
    eff_len_ref = scratch["eff_len"]
    scores_tile_ref = scratch["scores"]
    top_o_ref = scratch["top_o"]
    top_i_ref = scratch["top_i"]
    cand_o_ref = scratch["cand_o"]
    cand_i_ref = scratch["cand_i"]
    tau_ref = scratch["tau"]
    lst_o_ref = scratch["lst_o"]
    lst_i_ref = scratch["lst_i"]
    ev_o_ref = scratch["ev_o"]
    ev_i_ref = scratch["ev_i"]
    keys_ref = scratch["keys"]
    cell_max_ref = scratch["cell_max"]
    cell_arg_ref = scratch["cell_arg"]
    q_hm_ref = scratch["q_hm"]
    wb_ref = scratch["wb"]

    def _init_query_block(seq_idx, bq_idx):
      if use_stream:
        tc_topk.init_topk_vmem(
            top_o_ref,
            top_i_ref,
            cand_o_ref,
            cand_i_ref,
            tau_ref,
            num_q_groups=num_q_groups,
            num_blocks=num_blocks,
        )
        _fori(
            num_q_groups,
            lambda g, _: tc_topk.init_lane_lists_vmem(
                lst_o_ref, lst_i_ref, q_group=g, depth=lane_depth
            ),
        )
      end_seq = sched.end_seq
      cp_rank_val = cp_rank_ref[0] if cp_size > 1 else 0

      if seq_batch_size > 1:
        # Gather per-sequence scalars into vectors and compute `eff_len` once
        # per query group on the VPU (`num_partitions` sublanes per sequence).
        sublanes = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
        seqs_per_group = 8 // num_partitions
        seq_slot = (
            sublanes if num_partitions == 1 else sublanes // num_partitions
        )
        for g in range(num_q_groups):
          zeros = jnp.zeros((8, 128), jnp.int32)
          q_lo, q_hi, seq_len = zeros, zeros, zeros
          for slot in range(
              min(seqs_per_group, seq_batch_size - g * seqs_per_group)
          ):
            s_clamped = jnp.minimum(
                seq_idx + (g * seqs_per_group + slot), max_num_seqs - 1
            )
            at_slot = seq_slot == slot
            q_lo = jnp.where(at_slot, cu_q_lens_ref[s_clamped], q_lo)
            q_hi = jnp.where(at_slot, cu_q_lens_ref[s_clamped + 1], q_hi)
            seq_len = jnp.where(at_slot, seq_lens_ref[s_clamped], seq_len)
          b = g * seqs_per_group + seq_slot
          s = seq_idx + b
          q_len = q_hi - q_lo
          kv_len = seq_len // compression_ratio
          vis = jnp.clip((seq_len - q_len) // compression_ratio + 1, 0, kv_len)
          if cp_size > 1:
            vis = cp_local_length(vis, cp_rank_val, cp_size, interleave_c)
          valid = (
              (b < seq_batch_size)
              & (s < end_seq)
              & (s < max_num_seqs)
              & (q_len > 0)
          )
          eff_len_ref[g] = jnp.where(
              valid, jnp.minimum(vis, jnp.int32(num_positions)), 0
          )
      else:
        s_clamped = jnp.minimum(seq_idx, max_num_seqs - 1)
        seq_q_start = cu_q_lens_ref[s_clamped]
        seq_q_end = cu_q_lens_ref[s_clamped + 1]
        lo = jnp.maximum(0, jnp.maximum(seq_q_start, sched.chunk_start))
        hi = jnp.where(
            (seq_idx < end_seq) & (seq_idx < max_num_seqs),
            jnp.minimum(seq_q_end, sched.chunk_start + chunk_tokens),
            0,
        )
        tok_start = lo + bq_idx * bq_sz
        seq_len = seq_lens_ref[s_clamped]
        kv_len = seq_len // compression_ratio
        pos_offset = seq_len - seq_q_end
        sublanes = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
        part_shift = (num_partitions - 1).bit_length()

        def _init_eff_group(g, _):
          r_off = (
              lax.shift_right_arithmetic(sublanes, jnp.int32(part_shift))
              if num_partitions > 1
              else (g * jnp.int32(8) + sublanes)
          )
          tok_id = tok_start + r_off
          row_valid = (r_off < bq_sz) & (tok_id < hi)
          vis = jnp.clip(
              (pos_offset + tok_id) // compression_ratio + 1, 0, kv_len
          )
          if cp_size > 1:
            vis = cp_local_length(vis, cp_rank_val, cp_size, interleave_c)
          eff_len_ref[g] = jnp.where(
              row_valid, jnp.minimum(vis, num_positions), 0
          )
          return None

        _fori(num_q_groups, _init_eff_group)

    def _emit_score_tile(g, cur_bkv, m_p, tile):
      if use_select:
        idx_tile = cur_bkv * bkv_sz + m_p * 128 + _part_col_offset()
        keys_ref[g, cur_bkv * m_part + m_p] = tc_topk.f32_to_ord_i32(
            tile, idx_tile < eff_len_ref[g]
        )
      else:
        scores_tile_ref[g, m_p] = tile

    def _decode_score_tiles(q_vmem, weights_vmem, kv_rows, cur_bkv):
      """Scores `bq_sz == 1` HAS rows directly into `(8, 128)` top-k tiles."""

      def _run(dec_wb_ref):
        w_rows = weights_vmem[...].astype(jnp.float32)
        for b in range(bq_total):
          dec_wb_ref[b] = jnp.broadcast_to(
              w_rows[b, :, None], (num_q_heads, 128)
          )
        num_rows = bq_total * num_partitions
        tpc = math.gcd(m_part, 2)
        n = tpc * 128
        for g in range(num_q_groups):
          for c0 in range(0, m_part, tpc):
            parts = []
            scales = []
            for r in range(8):
              row = g * 8 + r
              if row >= num_rows:
                parts.append(jnp.zeros((8, n), jnp.float32))
                scales.append(jnp.ones((1, n), jnp.float32))
                continue
              b, p = divmod(row, num_partitions)
              keys = load_kv_keys(kv_rows, b, p * m_part + c0, tpc)
              st = qk(q_vmem[b], keys)
              part = None
              for hv in range(num_q_heads // 8):
                w8 = dec_wb_ref[b, pl.ds(hv * 8, 8), :]
                if tpc > 1:
                  w8 = jnp.concatenate([w8] * tpc, axis=1)
                term = jnp.maximum(st[hv * 8 : (hv + 1) * 8], 0.0) * w8
                part = term if part is None else part + term
              parts.append(part)
              scales.append(
                  load_kv_scales(kv_rows, b, p * m_part + c0, tpc).astype(
                      jnp.float32
                  )
              )
            tile = _sublane_transpose_sum(parts) * jnp.concatenate(
                scales, axis=0
            )
            if num_rows < (g + 1) * 8:
              sub = lax.broadcasted_iota(jnp.int32, (8, n), 0)
              tile = jnp.where(sub < num_rows - g * 8, tile, -jnp.inf)
            for j in range(tpc):
              _emit_score_tile(
                  g, cur_bkv, c0 + j, tile[:, j * 128 : (j + 1) * 128]
              )

      pl.run_scoped(_run, pltpu.VMEM((bq_total, num_q_heads, 128), jnp.float32))

    def _score_blocked_prefill(kv_rows, cur_bkv):
      """Scores multi-MXU blocked prefill chunks into `(8, 128)` tiles."""
      tok_blk = prefill_cfg.tok_blk
      tbs_per_batch = prefill_cfg.tbs_per_batch
      num_head_groups = prefill_cfg.num_head_groups
      rows_per_group_tok_blk = prefill_cfg.rows_per_group_tok_blk
      tiles_per_chunk = prefill_cfg.tiles_per_chunk
      groups_per_tb = tok_blk // 8
      batch_rows = tbs_per_batch * rows_per_group_tok_blk

      def _reduce_token_block(sts, tb: int, local_tb: int) -> jax.Array:
        acc = None
        for i in range(prefill_cfg.heads_per_group):
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

      eff_lens = (
          [eff_len_ref[g] for g in range(num_q_groups)] if use_select else None
      )
      part_col_base = (
          cur_bkv * bkv_sz + _part_col_offset() if use_select else None
      )
      empty_ord = jnp.int32(tc_topk.EMPTY_ORD_SENTINEL)

      for c in range(prefill_cfg.num_lane_chunks):
        keys, scales = load_kv_tiles(
            kv_rows, 0, c * tiles_per_chunk, tiles_per_chunk
        )
        scales = scales.astype(jnp.float32)
        for tb0 in range(0, prefill_cfg.num_tok_blks, tbs_per_batch):
          sts = [
              qk(
                  q_hm_ref[
                      pl.ds(
                          g_h * prefill_cfg.rows_per_group
                          + tb0 * rows_per_group_tok_blk,
                          batch_rows,
                      )
                  ],
                  keys,
              )
              for g_h in range(num_head_groups)
          ]
          for local_tb in range(tbs_per_batch):
            tb = tb0 + local_tb
            acc = _reduce_token_block(sts, tb, local_tb) * scales
            if use_select:
              assert (
                  eff_lens is not None
                  and part_col_base is not None
                  and keys_ref is not None
              )
              bits = lax.bitcast_convert_type(acc, jnp.int32)
              sign_mask = lax.shift_right_arithmetic(bits, jnp.int32(31))
              ord_acc = lax.bitwise_xor(
                  bits, lax.bitwise_and(sign_mask, jnp.int32(0x7FFFFFFF))
              )
              for jj in range(tiles_per_chunk):
                m_p = c * tiles_per_chunk + jj
                idx_tile = part_col_base + m_p * 128
                ord_chunk = ord_acc[:, jj * 128 : (jj + 1) * 128]
                for sg in range(groups_per_tb):
                  grp = tb * groups_per_tb + sg
                  ord_tile = ord_chunk[sg * 8 : (sg + 1) * 8]
                  keys_ref[grp, cur_bkv * m_part + m_p] = jnp.where(
                      idx_tile < eff_lens[grp], ord_tile, empty_ord
                  )
            else:
              for jj in range(tiles_per_chunk):
                acc_tile = acc[:, jj * 128 : (jj + 1) * 128]
                m_p = c * tiles_per_chunk + jj
                for sg in range(groups_per_tb):
                  grp = tb * groups_per_tb + sg
                  scores_tile_ref[grp, m_p] = acc_tile[sg * 8 : (sg + 1) * 8]

    def _score_general(q_vmem, weights_vmem, kv_rows, cur_bkv):
      """Fallback block scoring (SAL decode, small-head, or unblocked bq)."""
      q_is_hm = q_hm_ref is not None
      if q_is_hm:
        bq_vec = [q_hm_ref[...]]
      else:
        q_data = q_vmem[...].reshape(seq_batch_size, -1, head_dim)
        bq_vec = [q_data[b] for b in range(seq_batch_size)]
      w_data = weights_vmem[...].reshape(seq_batch_size, -1, num_q_heads)

      per_seq = []
      for b in range(seq_batch_size):
        keys, scales = load_kv_tiles(kv_rows, b, 0, num_lane_tiles_bkv)
        s_summed = reduce_heads(
            bq_vec[b].reshape(-1, head_dim), keys, w_data[b], head_major=q_is_hm
        )
        per_seq.append((s_summed * scales).reshape(-1, num_sublanes_bkv, 128))
      scores = jnp.concatenate(per_seq, axis=0)

      if num_partitions > 1:
        active_sublanes = bq_total * num_partitions
        scores_padded = scores.reshape(active_sublanes, m_part, 128)
        if active_sublanes < 8:
          scores_padded = jnp.pad(
              scores_padded,
              ((0, 8 - active_sublanes), (0, 0), (0, 0)),
              constant_values=-jnp.inf,
          )
      elif bq_alloc > bq_total:
        scores_padded = jnp.pad(
            scores,
            ((0, bq_alloc - bq_total), (0, 0), (0, 0)),
            constant_values=-jnp.inf,
        )
      else:
        scores_padded = scores

      for g in range(num_q_groups):
        grp_scores = scores_padded[g * 8 : (g + 1) * 8]
        for m_p in range(m_part):
          _emit_score_tile(g, cur_bkv, m_p, grp_scores[:, m_p, :])

    def _direct_single_bkv_topk(out_vmem, n_act):
      part_col_offset = _part_col_offset()
      empty_i = jnp.full((8, 128), -1, dtype=jnp.int32)

      def _load_slice(g, eff_len_g, m_lo: int, m_hi: int):
        ords, idxs = [], []
        for m_p in range(m_lo, m_hi):
          idx_tile = jnp.int32(m_p * 128) + part_col_offset
          valid = idx_tile < eff_len_g
          ords.append(tc_topk.f32_to_ord_i32(scores_tile_ref[g, m_p], valid))
          idxs.append(jnp.where(valid, idx_tile, empty_i))
        return tuple(ords), tuple(idxs)

      def _load_chunk(g, eff_len_g, ch_idx: int):
        m_lo = ch_idx * num_blocks
        return _load_slice(g, eff_len_g, m_lo, min(m_lo + num_blocks, m_part))

      def _finish_group(g, top_o, top_i):
        _finish_lane_major_group(
            out_vmem, g, top_o[:num_blocks], top_i[:num_blocks]
        )

      def _direct_groups(groups):
        eff_lens = [eff_len_ref[g] for g in groups]
        if (
            num_direct_chunks == 1
            and len(groups) == 2
            and num_blocks <= 4
            and m_part == num_blocks
        ):
          g0, g1 = groups
          a_o, a_i = _load_chunk(g0, eff_lens[0], 0)
          b_o, b_i = _load_chunk(g1, eff_lens[1], 0)
          (top0_o, top0_i), (top1_o, top1_i) = tc_topk.bitonic_sort_blocks_pair(
              a_o, a_i, b_o, b_i, desc_a=True, desc_b=True
          )
          _finish_group(g0, top0_o, top0_i)
          _finish_group(g1, top1_o, top1_i)
          return

        def _tree(g, eff_len_g, lo, hi, descending):
          if hi - lo == 1:
            ch_o, ch_i = _load_chunk(g, eff_len_g, lo)
            return tc_topk.sort_blocks_padded(
                ch_o, ch_i, num_blocks, descending=descending
            )
          if hi - lo == 2 and (
              num_blocks <= 4 or (num_blocks == 8 and len(groups) == 1)
          ):
            a_o, a_i = _load_chunk(g, eff_len_g, lo)
            b_o, b_i = _load_chunk(g, eff_len_g, lo + 1)
            if len(b_o) == num_blocks:
              return tc_topk.bitonic_topk_2r_to_r(
                  a_o, a_i, b_o, b_i, descending=descending
              )
            a_o, a_i = tc_topk.bitonic_sort_blocks(a_o, a_i, descending=True)
            b_o, b_i = tc_topk.sort_blocks_padded(
                b_o, b_i, num_blocks, descending=False
            )
            return tc_topk.bitonic_topk_merge_blocks(
                a_o, a_i, b_o, b_i, descending=descending
            )
          if (
              hi - lo == 4
              and num_blocks <= 4
              and len(groups) == 1
              and (lo + 4) * num_blocks <= m_part
          ):
            all_o, all_i = _load_slice(
                g, eff_len_g, lo * num_blocks, (lo + 4) * num_blocks
            )
            return tc_topk.bitonic_topk_4r_to_r(
                all_o, all_i, descending=descending
            )
          mid = lo + (1 << ((hi - lo - 1).bit_length() - 1))
          l_o, l_i = _tree(g, eff_len_g, lo, mid, True)
          r_o, r_i = _tree(g, eff_len_g, mid, hi, False)
          return tc_topk.bitonic_topk_merge_blocks(
              l_o, l_i, r_o, r_i, descending=descending
          )

        tops = [
            _tree(g, eff_len_g, 0, num_direct_chunks, True)
            for g, eff_len_g in zip(groups, eff_lens, strict=True)
        ]
        for g, (top_o, top_i) in zip(groups, tops, strict=True):
          _finish_group(g, top_o, top_i)

      _for_group_pairs(n_act, _direct_groups)

    def _stream_scores_from_vmem(cur_bkv, n_act):
      """Streams one KV block of score tiles into per-lane candidate lists."""
      part_col_offset = _part_col_offset()
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
          t_ords, t_idxs = [], []
          m_ps = [c * ins_w + w for w in range(ins_w)]
          for m_p in m_ps:
            idx_tile = (bkv_base_vec + m_p * jnp.int32(128)) + part_col_offset
            valid = idx_tile < eff_len_g
            t_ords.append(
                tc_topk.f32_to_ord_i32(scores_tile_ref[g, m_p], valid)
            )
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
          for c in range(num_ins_chunks):
            hot = _chunk(c, hot)
          return hot
        return lax.fori_loop(0, num_ins_chunks, _chunk, hot)

      hot = _fori(n_act, _insert_group, jnp.zeros((8, 128), jnp.int32))

      @pl.when(jnp.any(hot != 0))
      def _stage_evicted():
        def _stage_group(g, _):
          floor = _tau_floor(g)

          def _stage_subbatch(sub_idx, _):
            m0 = sub_idx * subbatch_w
            tc_topk.accumulate_subbatch_vmem(
                top_o_ref,
                top_i_ref,
                cand_o_ref,
                cand_i_ref,
                tau_ref,
                q_group=g,
                tile_ords=tuple(ev_o_ref[g, m0 + w] for w in range(subbatch_w)),
                tile_idxs=tuple(ev_i_ref[g, m0 + w] for w in range(subbatch_w)),
                sub_step=sub_idx,
                k=k,
                num_blocks=num_blocks,
                tau_floor=floor,
                inclusive=True,
            )
            return None

          def _stage_all():
            _fori(num_subbatches, _stage_subbatch)

          if num_q_groups == 1:
            _stage_all()
          else:
            tau_x = tc_topk.exclusive_threshold(jnp.maximum(tau_ref[g], floor))
            g_hot = functools.reduce(
                operator.or_, [ev_o_ref[g, m] > tau_x for m in range(m_part)]
            )
            pl.when(jnp.any(g_hot))(_stage_all)
          return None

        _fori(n_act, _stage_group)

    def _select_finalize(n_act, n_kv_blocks, out_vmem):
      """Selects exact Top-K of the resident keys into `out_vmem`."""
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
              lambda b: cell_max_ref[g, b],
              lambda b: cell_arg_ref[g, b],
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
        _finish_lane_major_group(out_vmem, g, fin_o, fin_i)
        return None

      _fori(n_act, _finalize_group)

    def _finish_step_topk(p_id, cur_bkv, is_last_bkv, out_vmem):
      n_act = _active_q_groups(p_id)
      if use_select:
        if use_two_level:

          def _summarize_group(g, _):
            best, arg = tc_topk.summarize_key_block(
                lambda t: keys_ref[g, cur_bkv * m_part + t], m_part
            )
            cell_max_ref[g, cur_bkv] = best
            cell_arg_ref[g, cur_bkv] = arg
            return None

          _fori(n_act, _summarize_group)

        @pl.when(is_last_bkv)
        def _on_last_select():
          _select_finalize(n_act, cur_bkv + 1, out_vmem)

        return

      if single_bkv:
        _direct_single_bkv_topk(out_vmem, n_act)
        return

      _stream_scores_from_vmem(cur_bkv, n_act)

      @pl.when(is_last_bkv)
      def _on_last():
        def _finalize_groups(groups):
          results = tc_topk.finalize_lane_lists_vmem(
              top_o_ref,
              top_i_ref,
              cand_o_ref,
              cand_i_ref,
              tau_ref,
              lst_o_ref,
              lst_i_ref,
              q_groups=groups,
              k=k,
              num_blocks=num_blocks,
              depth=lane_depth,
              num_partitions=num_partitions,
          )
          for g, (fin_o, fin_i) in zip(groups, results, strict=True):
            _write_group_out(out_vmem, g, fin_o, fin_i)

        _for_group_pairs(n_act, _finalize_groups)

    def step_body(step, q_vmem, weights_vmem, bkv_vmem, out_vmem):
      p_id = step.index[0]
      seq_idx, bq_idx, cur_bkv, is_last_bkv, _ = sched.lookup(p_id)

      @pl.when(cur_bkv == 0)
      def _on_first():
        _init_query_block(seq_idx, bq_idx)
        if prefill_cfg.enabled:
          tok_blk = prefill_cfg.tok_blk
          w = weights_vmem[...].astype(jnp.float32)
          for tb in range(prefill_cfg.num_tok_blks):
            t0 = tb * tok_blk
            q_rows = transpose_to_head_major(
                q_vmem[pl.ds(t0, tok_blk)], tok_blk
            )
            for h in range(num_q_heads):
              q_hm_ref[pl.ds(prefill_cfg.cache_row(h, tb), tok_blk)] = q_rows[
                  h * tok_blk : (h + 1) * tok_blk
              ]
              wb_ref[tb, h] = jnp.broadcast_to(
                  w[t0 : t0 + tok_blk, h : h + 1], (tok_blk, 128)
              ).astype(wb_ref.dtype)
        elif q_hm_ref is not None:
          q_hm_ref[...] = transpose_to_head_major(
              pad_tokens(q_vmem[...], bq_sz), hm_tok
          )

      rows = kv_block_rows(bkv_vmem)
      if prefill_cfg.enabled:
        _score_blocked_prefill(rows, cur_bkv)
      elif decode_tiles:
        _decode_score_tiles(q_vmem, weights_vmem, rows, cur_bkv)
      else:
        _score_general(q_vmem, weights_vmem, rows, cur_bkv)

      _finish_step_topk(p_id, cur_bkv, is_last_bkv, out_vmem)

    q_buf, weights_buf, kv_buf, o_buf = final_allocs
    pipeline_fn = pltpu.emit_pipeline(
        step_body,
        grid=(sched.num_steps,),
        in_specs=(q_buf.spec, weights_buf.spec, kv_buf.spec),
        out_specs=o_buf.spec,
        _explicit_indices=True,
    )
    window_refs = (cu_q_lens_ref, sched)

    @pl.when(sched.num_steps > 0)
    def _():
      pipeline_fn(
          (q_hbm_ref, *window_refs),
          (indexer_weights_hbm_ref, *window_refs),
          (cache_kv_hbm_ref, page_indices_ref, sched),
          (out_hbm_ref, *window_refs),
          allocations=(q_buf, weights_buf, kv_buf, o_buf),
      )

  pl.run_scoped(
      _run_with_scratches,
      scratch_spec,
      (q_alloc, weights_alloc, kv_alloc, o_alloc),
  )


def _fill_rows_after(out_hbm_ref, first_row, *, num_idx_blocks: int):
  """Writes empty Top-K rows (index -1, score -inf) to `out[first_row:]`."""
  num_rows, num_out_blocks, _ = out_hbm_ref.shape
  first_row = jnp.clip(first_row, 0, num_rows)
  num_fill = num_rows - first_row
  fill_rows = max(1, min(256, (1 << 20) // (4 * 128 * num_out_blocks)))

  @pl.when(num_fill > 0)
  def _():
    def _scoped(buf, sem):
      blocks = lax.broadcasted_iota(jnp.int32, buf.shape, 1)
      buf[...] = jnp.where(
          blocks < num_idx_blocks, jnp.int32(-1), jnp.int32(BITS_NEG_INF)
      )

      def _chunk(c, _):
        row = first_row + c * fill_rows
        size = jnp.minimum(fill_rows, num_rows - row)
        copy = pltpu.make_async_copy(
            buf.at[pl.ds(0, size)], out_hbm_ref.at[pl.ds(row, size)], sem
        )
        copy.start()
        copy.wait()
        return None

      lax.fori_loop(0, pl.cdiv(num_fill, fill_rows), _chunk, None)

    pl.run_scoped(
        _scoped,
        pltpu.VMEM((fill_rows, num_out_blocks, 128), jnp.int32),
        pltpu.SemaphoreType.DMA(()),
    )


def tc_kernel(
    seq_lens_ref: Any,
    page_indices_ref: Any,
    cu_q_lens_ref: Any,
    dist_ref: Any,
    cp_rank_ref: Any,
    q_hbm_ref: Any,
    indexer_weights_hbm_ref: Any,
    cache_kv_hbm_ref: Any,
    *out_refs: Any,
    cfg: _PassConfig,
    fill_tail: bool,
) -> None:
  """Builds the pass schedule in SMEM, then runs `_tc_kernel_body`.

  Args:
    seq_lens_ref: `[max_num_seqs]` SMEM uncompressed sequence lengths.
    page_indices_ref: `[max_num_seqs * pages_per_seq]` SMEM page table.
    cu_q_lens_ref: `[max_num_seqs + 1]` SMEM cumulative query lengths.
    dist_ref: `[3]` SMEM request distribution `(decode, prefill, total)`.
    cp_rank_ref: `[1]` SMEM context-parallel rank.
    q_hbm_ref: `[max_num_tokens, num_q_heads, head_dim]` HBM queries.
    indexer_weights_hbm_ref: `[max_num_tokens, num_q_heads]` HBM weights.
    cache_kv_hbm_ref: Packed `uint8` HBM KV cache.
    *out_refs: HBM output ref(s); the last ref holds the packed Top-K output.
    cfg: Static pass configuration.
    fill_tail: Whether to fill rows after the active tokens with `-1` / `-inf`.
  """
  out_hbm_ref = out_refs[-1]
  page_size = _kv_page_size(cache_kv_hbm_ref.shape, cfg.kv_layout)
  max_num_seqs = seq_lens_ref.shape[0]
  num_kv_blocks = cdiv(page_indices_ref.shape[0] // max_num_seqs, cfg.bkv_p)
  chunk_tokens = None if cfg.chunk_start is None else out_hbm_ref.shape[0]
  q_cap, table_cap = schedule_capacity(
      max_num_seqs=max_num_seqs,
      max_num_tokens=q_hbm_ref.shape[0],
      seq_batch_size=cfg.seq_batch_size,
      bq_sz=cfg.bq_sz,
      static_q_len=cfg.static_q_len,
      chunk_tokens=chunk_tokens,
      num_kv_blocks=num_kv_blocks,
  )

  def _smem(n: int):
    return pltpu.SMEM((n,), jnp.int32)

  sched_spec = ScheduleRef(
      info=_smem(3),
      batch_tile_idx=_smem(q_cap),
      bq_idx=_smem(q_cap),
      q_start=_smem(q_cap),
      q_end=_smem(q_cap),
      step_qb=_smem(table_cap) if table_cap else None,
      chunk_start=0 if cfg.chunk_start is None else cfg.chunk_start,
      chunk_tokens=out_hbm_ref.shape[0],
  )

  def _scoped(sched: ScheduleRef):
    build_schedule(
        sched,
        seq_lens_ref,
        cu_q_lens_ref,
        dist_ref,
        cp_rank_ref[0] if cfg.cp_size > 1 else 0,
        start_bound=cfg.start_bound,
        end_bound=cfg.end_bound,
        seq_batch_size=cfg.seq_batch_size,
        bq_sz=cfg.bq_sz,
        bkv_sz=cfg.bkv_p * page_size,
        num_kv_blocks=num_kv_blocks,
        compression_ratio=cfg.compression_ratio,
        static_q_len=cfg.static_q_len,
        chunk_tokens=chunk_tokens,
        cp_size=cfg.cp_size,
        interleave_c=cfg.interleave_c,
    )
    _tc_kernel_body(
        seq_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        sched,
        cp_rank_ref,
        q_hbm_ref,
        indexer_weights_hbm_ref,
        cache_kv_hbm_ref,
        out_hbm_ref,
        cfg,
    )

  pl.run_scoped(_scoped, sched_spec)

  if fill_tail:
    num_seqs_active = jnp.clip(dist_ref[2], 0, max_num_seqs)
    _, num_blocks = tc_topk.topk_capacity(cfg.k)
    _fill_rows_after(
        out_hbm_ref,
        cu_q_lens_ref[num_seqs_active],
        num_idx_blocks=num_blocks,
    )


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
  distribution = jnp.asarray(distribution, jnp.int32)
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

  def _as_triple(val):
    return [val] * 3 if isinstance(val, int) else list(val)

  num_kv_pages_per_blocks = _as_triple(num_kv_pages_per_block)
  num_queries_per_blocks = _as_triple(num_queries_per_block)
  buffer_counts = _as_triple(
      DEFAULT_BUFFER_COUNT if buffer_count is None else buffer_count
  )

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

  page_size = _kv_page_size(cache_kv.shape, kv_layout)
  if kv_layout == KVLayout.SEQ_ALONG_LANE:
    expected_shape = get_kv_cache_shape(
        cache_kv.shape[0], page_size, actual_head_dim, kv_layout
    )
    if cache_kv.shape != expected_shape:
      raise ValueError(
          "SEQ_ALONG_LANE expects cache_kv of shape"
          f" {expected_shape}, got {cache_kv.shape}."
      )
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

  def _select_bq_sz(num_queries_per_block, bkv_p, static_q_len, chunk_len):
    bq_sz = (
        min(num_queries_per_block, static_q_len)
        if static_q_len is not None
        else num_queries_per_block
    )
    if bq_sz > 32 and chunk_len < bq_sz:
      chunk_cap = max(32, 1 << (max(1, chunk_len) - 1).bit_length())
      bq_sz = min(bq_sz, chunk_cap)
    if topk_impl != TOPK_IMPL_STREAM and bq_sz > 8 and pages_per_seq > bkv_p:
      # Shrink the query block until its resident `select` keys fit in VMEM:
      # streaming Top-K over many KV blocks is far slower than `select`.
      max_groups = select_budget // _select_group_bytes(bkv_p)
      if 1 <= max_groups < cdiv(bq_sz, 8):
        bq_sz = min(bq_sz, 1 << ((8 * max_groups).bit_length() - 1))
    return bq_sz

  def _pass_topk_impl(bq_sz, bkv_p, seq_batch_size):
    """Resolves `topk_impl` for one fused pass."""
    if topk_impl != TOPK_IMPL_AUTO:
      return topk_impl
    if pages_per_seq <= bkv_p:
      return TOPK_IMPL_STREAM
    bq_total = seq_batch_size * bq_sz
    bkv_sz = page_size * bkv_p
    num_partitions = (
        1
        if _is_blocked_prefill_enabled(bq_sz, q.dtype, seq_batch_size)
        else tc_topk.sublane_partition_count(bq_total, bkv_sz)
    )
    tiles_per_block = bkv_sz // 128 // num_partitions
    num_kv_blocks = cdiv(pages_per_seq, bkv_p)
    if (
        num_kv_blocks * tiles_per_block
        <= _STREAM_MAX_TOTAL_TILES_PER_TOPK_BLOCK * num_blocks
        or tiles_per_block < 4
    ):
      return TOPK_IMPL_STREAM
    key_bytes = cdiv(bq_total, 8) * _select_group_bytes(bkv_p)
    return TOPK_IMPL_SELECT if key_bytes <= select_budget else TOPK_IMPL_STREAM

  def _pass_config(
      *,
      start_bound,
      end_bound,
      static_q_len,
      num_kv_pages_per_block,
      num_queries_per_block,
      buffer_count,
      seq_batch_size,
      case,
      chunk_token_start: int | None,
      chunk_len: int,
  ) -> tuple[str, _PassConfig]:
    """Returns `(name, _PassConfig)` of one fused pass."""
    if seq_batch_size > 1:
      assert static_q_len == 1
    eff_chunk_len = q.shape[0] if chunk_token_start is None else chunk_len
    bkv_p = num_kv_pages_per_block
    bq_sz = _select_bq_sz(
        num_queries_per_block, bkv_p, static_q_len, eff_chunk_len
    )
    pass_topk_impl = _pass_topk_impl(bq_sz, bkv_p, seq_batch_size)

    layout_tag = "" if kv_layout == KVLayout.HEAD_ALONG_SUBLANE else "-sal"
    name = f"{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}{layout_tag}"
    if pass_topk_impl == TOPK_IMPL_SELECT:
      name += "-sel"
    return name, _PassConfig(
        k=k,
        need_scores=return_scores,
        num_positions=num_positions,
        compression_ratio=compression_ratio,
        static_q_len=static_q_len,
        bq_sz=bq_sz,
        bkv_p=bkv_p,
        seq_batch_size=seq_batch_size,
        buffer_count=buffer_count,
        kv_layout=kv_layout,
        cp_size=cp_size,
        interleave_c=interleave_c,
        topk_impl=pass_topk_impl,
        start_bound=start_bound,
        end_bound=end_bound,
        chunk_start=chunk_token_start,
    )

  def run_fused_passes(
      pass_specs,
      *,
      chunk_token_start: int | None,
      chunk_len: int,
      scheduling_group_id: int | None,
  ):
    """Runs every pass of one chunk, in order, in a single kernel."""
    max_num_tokens = q.shape[0]
    chunked = chunk_token_start is not None
    eff_chunk_len = max_num_tokens if not chunked else chunk_len
    named = [
        _pass_config(
            chunk_token_start=chunk_token_start, chunk_len=chunk_len, **spec
        )
        for spec in pass_specs
    ]
    cfgs = tuple(c for _, c in named)

    def kernel(*refs):
      for i, pass_cfg in enumerate(cfgs):
        # Only an unchunked call's last pass writes the empty tail rows.
        tc_kernel(
            *refs, cfg=pass_cfg, fill_tail=not chunked and i == len(cfgs) - 1
        )

    scope_name = "StreamIdxFusedTC-" + "+".join(n for n, _ in named)
    if chunked:
      scope_name += f"-c{max_num_tokens // eff_chunk_len}"

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    scalar_prefetches = (
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        jnp.asarray([cp_rank], jnp.int32),
    )
    pallas_kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=[hbm_spec, hbm_spec, hbm_spec],
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
            name=scope_name,
        )
    )
    out = pallas_kernel(
        *scalar_prefetches, q, prepared_indexer_weights, cache_kv
    )
    if scheduling_group_id is not None:
      out = xla_metadata.set_xla_metadata(
          out, _scheduling_group_id=scheduling_group_id
      )
    return out

  # Sequence bounds of the passes, as `SeqBound`s over `distribution`.
  decode_batch_end = (
      None
      if max_num_seqs < decode_req_batch_size
      else (0, decode_req_batch_size)
  )

  def _pass_specs():
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
        start_bound=_PREFILL_END,
        end_bound=_MIXED_END,
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
              start_bound=_DECODE_END,
              end_bound=_PREFILL_END,
              static_q_len=uniform_q_len,
              seq_batch_size=1,
              case=MlaCase.PREFILL,
          ),
      )
    if uniform_q_len is None:
      if max_num_seqs == 1 and q.shape[0] == 1:
        return (
            dict(
                start_bound=None,
                end_bound=_MIXED_END,
                seq_batch_size=1,
                **decode_cfg,
            ),
        )
      if (max_num_seqs == 1 and q.shape[0] > 1) or (
          num_queries_per_blocks[2] >= 64 and q.shape[0] >= max_num_seqs * 64
      ):
        return (dict(mixed_spec, start_bound=None),)
    if max_num_seqs < decode_req_batch_size:
      batched_decode = ()
    else:
      batched_decode = (
          dict(
              start_bound=None,
              end_bound=decode_batch_end,
              seq_batch_size=decode_req_batch_size,
              **decode_cfg,
          ),
      )
    return (
        *batched_decode,
        dict(
            start_bound=decode_batch_end,
            end_bound=_DECODE_END if uniform else _PREFILL_END,
            seq_batch_size=1,
            **decode_cfg,
        ),
        *uniform,
        mixed_spec,
    )

  def _common_path(_):
    max_num_tokens = q.shape[0]

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
    passes = _pass_specs()

    chunk_outs = [
        run_fused_passes(
            passes,
            chunk_token_start=chunk_starts[m],
            chunk_len=chunk_len,
            scheduling_group_id=None if m == 0 else group_base + m - 1,
        )
        for m in range(num_chunks)
    ]

    def _unpack(raw, first_block):
      return raw[:, first_block : first_block + num_blocks, :].reshape(
          raw.shape[0], k_cap
      )[:, :k]

    if num_chunks == 1:
      # The kernel already wrote empty rows past the active tokens.
      top_idx = _unpack(chunk_outs[0], 0)
      if not return_scores:
        return top_idx
      return top_idx, _unpack(chunk_outs[0], num_blocks)

    raw_out = jnp.concatenate(chunk_outs, axis=0)
    num_active_seqs = jnp.clip(distribution[2], 0, max_num_seqs)
    active_tokens = cu_q_lens[num_active_seqs]
    valid_row = (
        jnp.arange(max_num_tokens, dtype=jnp.int32)[:, None] < active_tokens
    )

    top_idx = jnp.where(valid_row, _unpack(raw_out, 0), jnp.int32(-1))

    if not return_scores:
      return top_idx

    top_scores = jnp.where(
        valid_row, _unpack(raw_out, num_blocks), jnp.int32(BITS_NEG_INF)
    )
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
