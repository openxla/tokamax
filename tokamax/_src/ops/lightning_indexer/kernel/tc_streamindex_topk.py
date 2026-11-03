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
# Blocked prefill fuses scoring with the single-KV-block top-k of whole token
# blocks when each row has at most this many score tiles.
_MAX_FUSED_PREFILL_TILES = 8
# Rows of at most this many score tiles rank two token blocks per iteration;
# more groups per basic block overflow the vector registers.
_MAX_FUSED_PREFILL_TILES_2TB = 2
_MAX_FUSED_PREFILL_TOKEN_BLOCKS = 2
# A single-KV-block query-block pass is split into up to this many extra
# passes, with power-of-two KV blocks of at least `_MIN_KV_BUCKET_POSITIONS`
# positions, that run the query blocks whose visible KV fits in them.
_MAX_KV_BUCKET_PASSES = 3
_MIN_KV_BUCKET_POSITIONS = 512
# Smaller query blocks are not split into KV-length buckets.
_KV_BUCKET_MIN_QUERIES_PER_BLOCK = 64
# Nor are larger `k`: their single-KV-block top-k code is large enough that
# the extra passes slow the kernel down even when they run no query blocks.
_KV_BUCKET_MAX_K = 1024


def _fused_prefill_token_blocks(m_part: int, num_tok_blks: int) -> int:
  """Token blocks ranked per iteration of the fused blocked prefill loop."""
  if m_part <= _MAX_FUSED_PREFILL_TILES_2TB and num_tok_blks % 2 == 0:
    return _MAX_FUSED_PREFILL_TOKEN_BLOCKS
  return 1


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
# In-kernel query-block schedule.
# --------------------------------------------------------------------------- #

# Largest per-step query-block table (int32 SMEM entries) a pass builds. Passes
# with more possible steps binary-search the query-block end steps instead.
_MAX_STEP_TABLE_ENTRIES = 32768

# A sequence bound of a fused pass: `None` for 0, or `(i, align)` for
# `distribution[i] // align * align`.
SeqBound = tuple[int, int] | None

# Visible local KV length range `(lo, hi)` of the query blocks a fused pass
# admits: `lo < visible` and, unless `hi` is None, `visible <= hi`. None
# admits every query block.
VisRange = tuple[int, int | None] | None


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
  """

  info: Any  # [3]: (start_seq, end_seq, num_steps).
  batch_tile_idx: Any  # [num_q_blocks]: first sequence of the block's tile.
  bq_idx: Any  # [num_q_blocks]: query block index within its tile.
  q_start: Any  # [num_q_blocks]: first pipeline step.
  q_end: Any  # [num_q_blocks]: exclusive end step.
  step_qb: Any  # [num_steps_cap]: query block of each step, or None.
  chunk_start: int = dataclasses.field(
      default=0, metadata=dict(static=True)
  )

  @property
  def num_steps(self) -> jax.Array:
    return self.info[2]

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
    vis_range: VisRange = None,
):
  """Fills `sched` with the query-block schedule of one fused pass.

  Scalar-core equivalent of `metadata.compute_batched_seq_metadata(...,
  qblock_schedule=True)` for a single chunk: the same query blocks, in the same
  order, with the same number of KV blocks each. With `vis_range`, only the
  query blocks whose visible local KV length falls in that range are kept.

  Args:
    sched: SMEM refs to fill.
    seq_lens_ref: `[max_num_seqs]` uncompressed sequence lengths.
    cu_q_lens_ref: `[max_num_seqs + 1]` cumulative query lengths.
    dist_ref: `[3]` request distribution.
    cp_rank: This rank's context-parallel index.
    start_bound: First sequence of the pass.
    end_bound: End sequence (exclusive) of the pass.
    seq_batch_size: Sequences per query block.
    bq_sz: Query tokens per query block.
    bkv_sz: KV positions per KV block.
    num_kv_blocks: Maximum number of KV blocks per query block.
    compression_ratio: KV compression ratio.
    static_q_len: Static query length of every sequence of the pass, or None.
    chunk_tokens: Chunk length when the token axis is chunked, else None.
    cp_size: Number of context-parallel ranks.
    interleave_c: Context-parallel interleave in compressed positions.
    vis_range: Optional `(lo, hi)`: keep only the query blocks whose visible
      local KV length `v` satisfies `lo < v` and (unless `hi` is None)
      `v <= hi`. Requires `seq_batch_size == 1`.
  """
  num_seqs = seq_lens_ref.shape[0]
  sb = seq_batch_size
  assert vis_range is None or sb == 1, (vis_range, sb)
  num_tiles = cdiv(num_seqs, sb)
  q_cap = sched.q_end.shape[0]
  start = _seq_bound(dist_ref, start_bound)
  end = _seq_bound(dist_ref, end_bound)
  sched.info[0] = start
  sched.info[1] = end
  # Tile `t` is in the pass iff `start <= t * sb < end`.
  t_lo = jnp.clip((start + (sb - 1)) // sb, 0, num_tiles)
  t_hi = jnp.maximum(t_lo, jnp.clip((end + (sb - 1)) // sb, 0, num_tiles))
  chunked = chunk_tokens is not None
  span = sb * bq_sz

  def _local_len(kv_len):
    return metadata.cp_local_length(kv_len, cp_rank, cp_size, interleave_c)

  def _num_bkv(kv_len):
    return jnp.minimum(
        num_kv_blocks, jnp.maximum(1, pl.cdiv(_local_len(kv_len), bkv_sz))
    )

  def _tile(t, carry):
    tile_seq = t * sb
    lo = hi = None
    if chunked:
      tile_tok_start = cu_q_lens_ref[jnp.minimum(tile_seq, num_seqs)]
      tile_tok_end = cu_q_lens_ref[jnp.minimum(tile_seq + sb, num_seqs)]
      lo = jnp.clip(sched.chunk_start, tile_tok_start, tile_tok_end)
      hi = jnp.clip(
          sched.chunk_start + chunk_tokens, tile_tok_start, tile_tok_end
      )
      num_bq = pl.cdiv(hi - lo, span)
    elif static_q_len is not None:
      num_bq = max(1, cdiv(static_q_len, bq_sz))
    else:
      assert sb == 1
      q_len = cu_q_lens_ref[t + 1] - cu_q_lens_ref[t]
      num_bq = jnp.maximum(1, pl.cdiv(q_len, bq_sz))

    tile_bkv = None
    if sb > 1:
      max_kv = jnp.int32(0)
      for j in range(sb):
        s = tile_seq + j
        valid = (s >= start) & (s < end) & (s < num_seqs)
        seq_len = jnp.where(
            valid, seq_lens_ref[jnp.minimum(s, num_seqs - 1)], 0
        )
        max_kv = jnp.maximum(
            max_kv, _local_len(seq_len // compression_ratio)
        )
      tile_bkv = jnp.minimum(
          num_kv_blocks, jnp.maximum(1, pl.cdiv(max_kv, bkv_sz))
      )

    def _block(b, carry):
      qb, step = carry
      admit = None
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
        if vis_range is not None:
          vis_lo, vis_hi = vis_range
          local_vis = _local_len(vis)
          admit = local_vis > vis_lo
          if vis_hi is not None:
            admit = admit & (local_vis <= vis_hi)
      in_cap = qb < q_cap
      if admit is not None:
        in_cap = in_cap & admit
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

      next_qb = qb + 1 if admit is None else qb + admit.astype(jnp.int32)
      return next_qb, jnp.where(in_cap, step_end, step)

    return lax.fori_loop(0, num_bq, _block, carry)

  num_qb, num_steps = lax.fori_loop(
      t_lo, t_hi, _tile, (jnp.int32(0), jnp.int32(0))
  )
  sched.info[2] = num_steps
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
    refs: tuple[Any, ...],
    p_id: int | jax.Array,
    chunk_tokens: int,
    *,
    bq_sz: int,
    seq_batch_size: int,
):
  """Flat-token rows `(token_start, total_sz)` for step `p_id`."""
  _, cu_q_lens_ref, sched = refs
  chunk_start = sched.chunk_start
  seq_idx, bq_idx, _, _, _ = sched.lookup(p_id)
  max_seq = cu_q_lens_ref.shape[0] - 1
  seq_clamped = jnp.minimum(seq_idx, max_seq - 1)
  lo = jnp.maximum(cu_q_lens_ref[seq_clamped], chunk_start)

  if seq_batch_size > 1:
    return lo, seq_batch_size * bq_sz
  token_start = lo + bq_idx * bq_sz
  if bq_sz == 1:
    return token_start, 1
  hi = jnp.minimum(cu_q_lens_ref[seq_clamped + 1], chunk_start + chunk_tokens)
  return token_start, jnp.maximum(0, jnp.minimum(bq_sz, hi - token_start))


def _contiguous_pages(page_indices_ref, first_entry, num_pages, total_pages):
  """Returns `(first_page, contiguous)` for `num_pages` page-table entries.

  `contiguous` is True iff the entries name consecutive in-range cache pages,
  so that one DMA of `num_pages` pages from `first_page` moves exactly the
  bytes of the per-page DMAs.

  Args:
    page_indices_ref: Flat SMEM page table.
    first_entry: Page-table index of the first entry (clamped like the
      per-page path).
    num_pages: Number of entries.
    total_pages: Number of pages in the KV cache.
  """
  num_entries = page_indices_ref.shape[0]

  def _page(i):
    return page_indices_ref[jnp.minimum(first_entry + i, num_entries - 1)]

  first = _page(0)
  # A few independent OR chains of `page_i - (first + i)`: short dependency
  # chains without keeping every entry live in scalar registers.
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
    first_entry: Any,
    seq_batch_size: int,
    bkv_p: int,
    pages_per_seq: int,
    total_pages: int,
    copy_pages: Any,
    copy_page: Any,
):
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
    copy_pages: `(b, num_seqs, first_page)` starts one DMA of
      `num_seqs * bkv_p` pages into sequences `[b, b + num_seqs)`.
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
class QBlockKVBufferedRef(bref_override.StreamIndexKVBufferedRef):
  """Paged KV BufferedRef (`HEAD_ALONG_SUBLANE`) driven by a `ScheduleRef`."""

  def copy_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, Any],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    cache_kv_hbm, page_indices_ref, sched = src_ref
    batch_tile_idx, _, bkv_idx, _, _ = sched.lookup(grid_indices[0])

    assert self.sem_recvs is not None
    assert self.window_ref is not None
    assert isinstance(cache_kv_hbm, jax.Ref)
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_dst: Any = self.window_ref.at[slot]

    pspk = self.page_size_per_kv_packing
    rows = self.bkv_p * pspk
    reshaped_cache_hbm_ref: Any = (
        cache_kv_hbm.reshape(  # pyrefly: ignore[missing-attribute]
            cache_kv_hbm.shape[0] * pspk,
            self.kv_packing,
            cache_kv_hbm.shape[-1],
        )
    )
    max_hbm_pages = reshaped_cache_hbm_ref.shape[0]

    def _copy_pages(b, num_seqs, first):
      if num_seqs == 1:
        dst = vmem_dst.at[b, pl.ds(0, rows)]
      else:
        dst = vmem_dst.reshape(num_seqs * rows, *vmem_dst.shape[2:])
      pltpu.make_async_copy(
          reshaped_cache_hbm_ref.at[pl.ds(first * pspk, num_seqs * rows)],
          dst,
          sem,
      ).start()

    def _copy_page(b, i, page):
      safe_page_offset = jnp.minimum(
          page * pspk, jnp.maximum(0, max_hbm_pages - pspk)
      )
      pltpu.make_async_copy(
          reshaped_cache_hbm_ref.at[pl.ds(safe_page_offset, pspk)],
          vmem_dst.at[b, pl.ds(i * pspk, pspk)],
          sem,
      ).start()

    _copy_kv_block(
        page_indices_ref,
        first_entry=lambda b: (batch_tile_idx + b) * self.pages_per_seq
        + bkv_idx * self.bkv_p,
        seq_batch_size=self.seq_batch_size,
        bkv_p=self.bkv_p,
        pages_per_seq=self.pages_per_seq,
        total_pages=cache_kv_hbm.shape[0],
        copy_pages=_copy_pages,
        copy_page=_copy_page,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QBlockKVSeqAlongLaneBufferedRef(
    bref_override.StreamIndexKVSeqAlongLaneBufferedRef
):
  """Paged KV BufferedRef (`SEQ_ALONG_LANE`) driven by a `ScheduleRef`.

  The VMEM block is page-major, `[seq_batch, bkv_p, groups, packing, page]`,
  so a run of consecutive cache pages lands with a single DMA.
  """

  def copy_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, Any],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    cache_kv_hbm, page_indices_ref, sched = src_ref
    batch_tile_idx, _, bkv_idx, _, _ = sched.lookup(grid_indices[0])

    assert self.sem_recvs is not None
    assert self.window_ref is not None
    assert isinstance(cache_kv_hbm, jax.Ref)
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_dst: Any = self.window_ref.at[slot]

    max_hbm_pages = cache_kv_hbm.shape[0]

    def _copy_pages(b, num_seqs, first):
      if num_seqs == 1:
        dst = vmem_dst.at[b]
      else:
        dst = vmem_dst.reshape(num_seqs * self.bkv_p, *vmem_dst.shape[2:])
      pltpu.make_async_copy(
          cache_kv_hbm.at[pl.ds(first, num_seqs * self.bkv_p)], dst, sem
      ).start()

    def _copy_page(b, i, page):
      pltpu.make_async_copy(
          cache_kv_hbm.at[jnp.minimum(page, max_hbm_pages - 1)],
          vmem_dst.at[b, i],
          sem,
      ).start()

    _copy_kv_block(
        page_indices_ref,
        first_entry=lambda b: (batch_tile_idx + b) * self.pages_per_seq
        + bkv_idx * self.bkv_p,
        seq_batch_size=self.seq_batch_size,
        bkv_p=self.bkv_p,
        pages_per_seq=self.pages_per_seq,
        total_pages=max_hbm_pages,
        copy_pages=_copy_pages,
        copy_page=_copy_page,
    )


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
    out_hbm, _, sched = dst_ref
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
    )
    dst_start = jnp.minimum(
        jnp.maximum(0, token_start - sched.chunk_start),
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
    )
    sliced = vmem_src.at[pl.ds(0, total_sz)]
    pltpu.make_async_copy(sliced, sliced, sem).wait()


# --------------------------------------------------------------------------- #
# Fused scoring + in-VMEM TensorCore Top-K kernel.
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class _DirectScratch:
  """VMEM refs of the deferred lane-overflow repair of the direct top-k.

  Attributes:
    top_o: `[groups, R, 8, 128]` top-k order keys kept for overflow repair.
    top_i: `[groups, R, 8, 128]` top-k indices kept for overflow repair.
    ovf: `[groups, 8, 128]` nonzero where a lane may have overflowed.
  """

  top_o: Any
  top_i: Any
  ovf: Any


@dataclasses.dataclass(frozen=True)
class _BlockedPrefillConfig:
  """Accumulator-fitted multi-MXU prefill scoring layout."""

  enabled: bool
  tok_blk: int
  num_tok_blks: int
  seqs_per_tb: int
  src_step: int
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
    if bq_sz == 16 and seq_batch_size > 1 and seq_batch_size % 2 == 0:
      # Two sequences per token block double the rows of each MXU pass and
      # rank two query group pairs per token block.
      tok_blk = 32
    elif bq_sz < 8:
      tok_blk = 8
    elif bq_sz <= 16:
      tok_blk = 16 if (bq_sz == 16 or seq_batch_size > 1) else 8
    else:
      tok_blk = math.gcd(bq_sz, 32)
    seqs_per_tb = max(1, tok_blk // bq_sz) if seq_batch_size > 1 else 1
    if seq_batch_size > 1:
      num_tok_blks = max(1, seq_batch_size // seqs_per_tb)
      src_step = seqs_per_tb * bq_sz
    else:
      num_tok_blks = max(1, bq_sz // tok_blk)
      src_step = min(bq_sz, tok_blk)
    num_head_groups = math.gcd(num_q_heads, tpu_info.num_mxus)
    heads_per_group = num_q_heads // num_head_groups
    rows_per_group_tok_blk = heads_per_group * tok_blk
    rows_per_group = num_tok_blks * rows_per_group_tok_blk

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
    if seq_batch_size > 1:
      tbs_per_batch = 1
    else:
      tbs_per_batch = math.gcd(
          num_tok_blks, max(1, max_accum_rows // rows_per_group_tok_blk)
      )

    q_sublane_tiling = 8 * get_dtype_packing(q_dtype)
    enabled = (
        bq_sz > 1
        and (
            seq_batch_size == 1
            or (bq_sz in (2, 4, 8, 16) and seq_batch_size % seqs_per_tb == 0)
        )
        and tok_blk % 8 == 0
        and rows_per_group_tok_blk % q_sublane_tiling == 0
    )

    bf16_reduce = get_dtype_bitwidth(q_dtype) == 8 and tok_blk % 16 == 0
    wb_dtype = jnp.bfloat16 if bf16_reduce else jnp.float32

    return cls(
        enabled=enabled,
        tok_blk=tok_blk,
        num_tok_blks=num_tok_blks,
        seqs_per_tb=seqs_per_tb,
        src_step=src_step,
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


def _tc_kernel_body(
    # Scalar prefetches (SMEM)
    seq_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    sched,  # ScheduleRef
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
    topk_impl: str = TOPK_IMPL_STREAM,
    single_kv_block: bool = False,
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
  # With `single_kv_block`, the schedule only holds query blocks whose visible
  # KV fits in the first KV block of their sequence, so every query block takes
  # the single-KV-block path even if `pages_per_seq > bkv_p`.
  single_bkv = single_kv_block or pages_per_seq <= bkv_p
  use_select = topk_impl == TOPK_IMPL_SELECT
  assert not (single_kv_block and use_select)
  num_kv_blocks = 1 if single_kv_block else cdiv(pages_per_seq, bkv_p)
  select_idx_bits = max(1, (num_positions - 1).bit_length())
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
  # `_DirectScratch` of lane-pruned direct top-k, bound by `_run_all`.
  direct_refs: dict[str, Any] = {}

  def lookup_step(p_id):
    return sched.lookup(p_id)

  def q_blk_id(p_id):
    return lookup_step(p_id)[4]

  def kv_sal_idx(p_id):
    bt, _, cbkv, _, _ = lookup_step(p_id)
    return (bt, cbkv, 0, 0, 0)

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
            bkv_p,
            kv_sublane_groups,
            kv_packing,
            page_size,
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
  )
  weights_alloc = QBlockQBufferedRef.input(
      spec=weights_spec,
      dtype_or_type=indexer_weights_hbm_ref,
      buffer_count=buffer_cnt,
      use_lookahead=True,
      bq_sz=bq_sz,
      seq_batch_size=seq_batch_size,
      chunk_tokens=chunk_tokens,
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
    )
  o_alloc = QBlockTopKOBufferedRef.output(
      spec=out_spec,
      dtype_or_type=out_hbm_ref,
      buffer_count=2,
      use_lookahead=False,
      bq_sz=bq_sz,
      seq_batch_size=seq_batch_size,
  )

  num_lane_tiles_bkv = bkv_sz // 128
  lane_tiles_per_page = page_size // 128

  def seq_along_lane_block_rows(bkv_vmem):
    # The block is page-major: lane tile `t` of sublane group `g` of page `p`
    # is word row `(p * kv_sublane_groups + g) * lane_tiles_per_page + t`.
    return bkv_vmem.bitcast(jnp.uint32).reshape(
        seq_batch_size, kv_sublane_groups * num_lane_tiles_bkv, 128
    )

  def seq_along_lane_row(lane_tile_idx, group):
    page, tile = divmod(lane_tile_idx, lane_tiles_per_page)
    return (page * kv_sublane_groups + group) * lane_tiles_per_page + tile

  def seq_along_lane_key_tile(rows, batch_idx, lane_tile_idx):
    words = rows[
        batch_idx,
        pl.ds(
            seq_along_lane_row(lane_tile_idx, 0),
            kv_head_dim_groups,
            stride=lane_tiles_per_page,
        ),
        :,
    ]
    return pltpu.bitcast(words, jnp.float8_e4m3fn)

  def seq_along_lane_scale_tile(rows, batch_idx, lane_tile_idx):
    row = seq_along_lane_row(lane_tile_idx, kv_head_dim_groups)
    words = rows[batch_idx, pl.ds(row, 1), :]
    return pltpu.bitcast((words & 0xFF) << 23, jnp.float32)

  # HEAD_ALONG_SUBLANE packs `kv_packing` tokens per 32-bit word, so every
  # 128-lane tile of a token group's record is one `(1, 128)` uint32 row.
  # Strided row loads then give keys (and scales) in native vreg tiling with
  # no relayout.
  has_rec_tiles = cache_kv_hbm_ref.shape[-1] // 128
  has_key_tiles = head_dim // 128

  def head_along_sublane_block_rows(bkv_vmem):
    return bkv_vmem.bitcast(jnp.uint32).reshape(
        seq_batch_size, bkv_sz_per_kv_packing * has_rec_tiles, 128
    )

  def head_along_sublane_words(rows, batch_idx, tile_idx, group_start, groups):
    return rows[
        batch_idx,
        pl.ds(group_start * has_rec_tiles + tile_idx, groups,
              stride=has_rec_tiles),
        :,
    ]

  def head_along_sublane_keys(rows, batch_idx, group_start, groups):
    """`(groups * kv_packing, head_dim)` fp8 keys, one token per row."""
    return jnp.concatenate(
        [
            pltpu.bitcast(
                head_along_sublane_words(
                    rows, batch_idx, t, group_start, groups
                ),
                jnp.float8_e4m3fn,
            )
            for t in range(has_key_tiles)
        ],
        axis=1,
    )

  def head_along_sublane_scales(rows, batch_idx, group_start, groups):
    """`(1, groups * kv_packing)` float32 per-token scales.

    Decodes e8m0 straight to float32 bits (`e << 23`), matching the bfloat16
    conversion, including `e == 0` (2**-127) and `e == 255` (NaN).

    Args:
      rows: `head_along_sublane_block_rows` view of the KV block.
      batch_idx: Row of the block.
      group_start: First `kv_packing`-token group.
      groups: Number of groups.

    Returns:
      A `(1, groups * kv_packing)` float32 array.
    """
    scale_bytes = pltpu.bitcast(
        head_along_sublane_words(
            rows, batch_idx, has_key_tiles, group_start, groups
        ),
        jnp.uint8,
    )
    e = scale_bytes[:, 0:1].T.astype(jnp.int32)
    scales = pltpu.bitcast(e << 23, jnp.float32)
    scales = jnp.where(e == 0, jnp.float32(2.0**-127), scales)
    return jnp.where(e == 255, jnp.float32(jnp.nan), scales)

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
      bq = bq_vec[batch_idx]
      bkv = bkv_vec[batch_idx]
      scale_val = scale_val_vec[batch_idx]
      bq_weights = bq_weights_vec[batch_idx]
      if q_is_head_major or bq_weights.shape[0] == 1:
        bq = bq.reshape(-1, head_dim)
      else:
        bq = bq.reshape(-1, num_q_heads, head_dim)
      s_summed = reduce_heads(bq, bkv, bq_weights, head_major=q_is_head_major)
      s_summed = s_summed * scale_val
      ret.append(s_summed.reshape(-1, num_sublanes_bkv, 128))
    return jnp.concatenate(ret, axis=0)

  def _sublane_transpose_sum(parts):
    """Returns `t` with `t[r] = parts[r].sum(axis=0)` for 8 `(8, n)` parts.

    A 3-level butterfly: each level adds sublane pairs `d` apart and keeps
    half of each of two parts, so row `r` of the result ends up holding the
    full sublane sum of part `r`.

    Args:
      parts: 8 `(8, n)` float32 arrays.

    Returns:
      An `(8, n)` float32 array.
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
          for a, b in zip(parts[:half], parts[half:])
      ]
      d //= 2
    return parts[0]

  # HEAD_ALONG_SUBLANE single-token rows (`bq_sz == 1`) are scored straight
  # into `(8, 128)` top-k tiles one key chunk at a time (see
  # `_decode_score_tiles`), so per-row keys are loaded per chunk instead of
  # being kept live (and spilled) for the whole block.
  decode_tiles = (
      not seq_along_lane
      and bq_sz == 1
      and not prefill_cfg.enabled
      and num_q_heads % 8 == 0
  )

  def _decode_score_tiles(q_vmem, wb_ref, kv_rows, emit):
    """Scores `bq_sz == 1` rows directly into `(8, 128)` top-k tiles.

    Tile row `r` of group `g` is query `(g * 8 + r) // num_partitions` against
    KV partition `(g * 8 + r) % num_partitions`. For each 256-key chunk, each
    row's relu(q.k) * w is summed over heads down to 8 sublanes, and a
    butterfly over the group's 8 rows finishes the head sum with row `r` in
    sublane `r`. This keeps the live set to one chunk, so the full
    `(heads, bkv)` scores never spill and no `(1, bkv)` row needs a relayout.

    Args:
      q_vmem: `[bq_total, num_q_heads, head_dim]` query block.
      wb_ref: `[bq_total, num_q_heads, 128]` lane-broadcast head weights.
      kv_rows: `head_along_sublane_block_rows` view of the KV block.
      emit: `(g, m_p, tile)` consumer of each `(8, 128)` score tile.
    """
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
          t0 = (p * m_part + c0) * 128
          g0, ng = t0 // kv_packing, n // kv_packing
          keys = head_along_sublane_keys(kv_rows, b, g0, ng)
          st = jnp.einsum(
              "nd,md->nm", q_vmem[b], keys, preferred_element_type=jnp.float32
          )
          part = None
          for hv in range(num_q_heads // 8):
            w8 = wb_ref[b, pl.ds(hv * 8, 8), :]
            if tpc > 1:
              w8 = jnp.concatenate([w8] * tpc, axis=1)
            term = jnp.maximum(st[hv * 8 : (hv + 1) * 8], 0.0) * w8
            part = term if part is None else part + term
          parts.append(part)
          scales.append(
              head_along_sublane_scales(kv_rows, b, g0, ng).astype(jnp.float32)
          )
        tile = _sublane_transpose_sum(parts) * jnp.concatenate(scales, axis=0)
        if num_rows < (g + 1) * 8:
          sub = lax.broadcasted_iota(jnp.int32, (8, n), 0)
          tile = jnp.where(sub < num_rows - g * 8, tile, -jnp.inf)
        for j in range(tpc):
          emit(g, c0 + j, tile[:, j * 128 : (j + 1) * 128])

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
    end_seq = sched.info[1]
    cp_rank_val = cp_rank_ref[0] if cp_size > 1 else 0

    if seq_batch_size > 1 and num_partitions == 1:
      # Equally sized sequences packed into query groups (`seqs_per_group` per
      # 8-sublane group, or `groups_per_seq` groups per sequence when
      # `bq_sz >= 8`): gather the per-sequence scalars and compute `eff_len`
      # per query group on the VPU.
      sublanes = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
      if bq_sz >= 8:
        groups_per_seq = bq_sz // 8
        end_vec = jnp.full((8, 128), end_seq, jnp.int32)
        for g in range(num_q_groups):
          b = g // groups_per_seq
          s = seq_idx + b
          s_clamped = jnp.minimum(s, max_num_seqs - 1)
          s_vec = jnp.full((8, 128), s, jnp.int32)
          q_len = jnp.full(
              (8, 128),
              cu_q_lens_ref[s_clamped + 1] - cu_q_lens_ref[s_clamped],
              jnp.int32,
          )
          seq_len = jnp.full((8, 128), seq_lens_ref[s_clamped], jnp.int32)
          kv_len = seq_len // compression_ratio
          tok_in_seq = (g % groups_per_seq) * 8 + sublanes
          vis = jnp.clip(
              (seq_len - (q_len - tok_in_seq)) // compression_ratio + 1,
              0,
              kv_len,
          )
          if cp_size > 1:
            vis = cp_local_length(vis, cp_rank_val, cp_size, interleave_c)
          valid = (
              (b < seq_batch_size)
              & (s_vec < end_vec)
              & (s_vec < max_num_seqs)
              & (tok_in_seq < q_len)
          )
          eff_len_ref[g] = jnp.where(
              valid, jnp.minimum(vis, jnp.int32(num_positions)), 0
          )
      else:
        seqs_per_group = 8 // bq_sz
        seq_slot = sublanes if bq_sz == 1 else sublanes // bq_sz
        tok_in_seq = 0 if bq_sz == 1 else sublanes % bq_sz
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
          vis = jnp.clip(
              (seq_len - (q_len - tok_in_seq)) // compression_ratio + 1,
              0,
              kv_len,
          )
          if cp_size > 1:
            vis = cp_local_length(vis, cp_rank_val, cp_size, interleave_c)
          valid = (
              (b < seq_batch_size)
              & (s < end_seq)
              & (s < max_num_seqs)
              & (tok_in_seq < q_len)
          )
          eff_len_ref[g] = jnp.where(
              valid, jnp.minimum(vis, jnp.int32(num_positions)), 0
          )
    elif seq_batch_size > 1:
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
      chunk_start_vec = jnp.full((8, 128), sched.chunk_start, jnp.int32)
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
        (None, cu_q_lens_ref, sched),
        p_id,
        chunk_tokens,
        bq_sz=bq_sz,
        seq_batch_size=1,
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
      fn((0,))
      return
    if isinstance(n_act, int) and n_act == 2:
      fn((0, 1))
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

  # Rows of at least 4 chunks with more than 8 output blocks first keep each
  # lane's top `2 * num_blocks` entries (see `_direct_one_group`).
  lane_pruned = (
      num_direct_chunks > 1
      and m_part // direct_blocks >= 4
      and direct_blocks > 8
  )

  def _with_direct_scratch(scores_tile_ref, eff_len_ref, out_vmem, n_act, body):
    """Runs `body(scratch)` for the active query groups' direct top-k.

    `scratch` is a `_DirectScratch` for lane-pruned rows and None otherwise.
    For lane-pruned rows the rare lane-overflow repairs run afterwards, outside
    any hot loop, so loop bodies that rank groups stay one basic block.

    Args:
      scores_tile_ref: `[groups, m_part, 8, 128]` float32 score tile ref.
      eff_len_ref: VMEM ref of each row's valid KV length.
      out_vmem: Output block ref.
      n_act: Number of active query groups.
      body: Callback ranking all active groups via `_direct_topk_groups`.
    """
    if not lane_pruned:
      body(None)
      return

    scratch = direct_refs["scratch"]
    body(scratch)

    def _fixup(g, carry):
      _direct_topk_groups(
          scores_tile_ref,
          eff_len_ref,
          out_vmem,
          (g,),
          scratch=scratch,
          fixup_only=True,
      )
      return carry

    _for_each_group(n_act, _fixup)

  def _direct_single_bkv_topk(scores_tile_ref, eff_len_ref, out_vmem, n_act):
    _with_direct_scratch(
        scores_tile_ref,
        eff_len_ref,
        out_vmem,
        n_act,
        lambda scratch: _for_group_pairs(
            n_act,
            functools.partial(
                _direct_topk_groups,
                scores_tile_ref,
                eff_len_ref,
                out_vmem,
                scratch=scratch,
            ),
        ),
    )

  def _direct_topk_groups(
      scores_tile_ref,
      eff_len_ref,
      out_vmem,
      groups,
      *,
      scratch=None,
      fixup_only=False,
  ):
    """Direct single-KV-block top-k of the query groups in `groups`.

    Reads only the `scores_tile_ref` tiles of `groups` and writes only their
    `out_vmem` rows, so callers may overlap it with scoring other groups.

    Args:
      scores_tile_ref: `[groups, m_part, 8, 128]` float32 score tile ref.
      eff_len_ref: VMEM ref of each row's valid KV length.
      out_vmem: Output block ref.
      groups: Tuple of one or two query group indices.
      scratch: Optional `_DirectScratch` for lane-pruned rows. When given, the
        rare lane-overflow repair is deferred: each group's top-k and overflow
        flag are stored for a later `fixup_only` call.
      fixup_only: Only run the deferred overflow repair of `groups`.
    """
    part_col_offset = _part_col_offset()
    empty_i = jnp.full((8, 128), -1, dtype=jnp.int32)

    def _load_direct_slice(
        g, eff_len_g, m_lo: int, m_hi: int, *, mask_idx: bool = True
    ):
      ords = []
      idxs = []
      for m_p in range(m_lo, m_hi):
        s_tile = scores_tile_ref[g, m_p]
        idx_tile = jnp.int32(m_p * 128) + part_col_offset
        valid = idx_tile < eff_len_g
        ords.append(tc_topk.f32_to_ord_i32(s_tile, valid))
        idxs.append(
            jnp.where(valid, idx_tile, empty_i) if mask_idx else idx_tile
        )
      return tuple(ords), tuple(idxs)

    def _load_direct_chunk_valid(
        g, eff_len_g, ch_idx: int, *, mask_idx: bool = True
    ):
      m_lo = ch_idx * direct_blocks
      m_hi = min(m_lo + direct_blocks, m_part)
      return _load_direct_slice(g, eff_len_g, m_lo, m_hi, mask_idx=mask_idx)

    def _finish_direct_group(
        g, top_o, top_i, *, mask_invalid_idx: bool = False
    ):
      fin_o, fin_i = top_o[:num_blocks], top_i[:num_blocks]
      if num_partitions > 1:
        fin_o, fin_i = tc_topk.reduce_sublane_partitions(
            fin_o, fin_i, num_partitions=num_partitions
        )
      if mask_invalid_idx:
        fin_i = tuple(
            jnp.where(o == tc_topk.EMPTY_ORD_SENTINEL, empty_i, i)
            for o, i in zip(fin_o, fin_i)
        )
      fin_i = tc_topk.lane_major_to_block_major(fin_i)
      if need_scores:
        fin_o = tc_topk.lane_major_to_block_major(fin_o)
      _write_group_out(out_vmem, g, fin_o, fin_i)

    def _finish_direct_unpadded_group(g, sorted_o, sorted_i):
      if num_partitions > 1 or direct_blocks % m_part != 0:
        top_o, top_i = tc_topk.pad_sorted_blocks(
            sorted_o, sorted_i, direct_blocks, descending=True
        )
        _finish_direct_group(g, top_o, top_i)
        return
      fin_i = tc_topk.lane_major_to_block_major(sorted_i)
      pad_n = num_blocks - m_part
      if pad_n > 0:
        fin_i = fin_i + (empty_i,) * pad_n
      if need_scores:
        fin_o = tc_topk.lane_major_to_block_major(sorted_o)
        if pad_n > 0:
          empty_o = jnp.full(
              (8, 128), tc_topk.EMPTY_ORD_SENTINEL, dtype=jnp.int32
          )
          fin_o = fin_o + (empty_o,) * pad_n
      else:
        fin_o = sorted_o
      _write_group_out(out_vmem, g, fin_o, fin_i)

    if (
        num_direct_chunks == 1
        and len(groups) == 2
        and m_part <= 8
        and direct_blocks % m_part == 0
    ):
      g0, g1 = groups
      a_o, a_i = _load_direct_chunk_valid(g0, eff_len_ref[g0], 0)
      b_o, b_i = _load_direct_chunk_valid(g1, eff_len_ref[g1], 0)
      (a_o, a_i), (b_o, b_i) = tc_topk.bitonic_sort_blocks_pair(
          a_o,
          a_i,
          b_o,
          b_i,
          desc_a=True,
          desc_b=True,
          ordered_inputs=True,
      )
      _finish_direct_unpadded_group(g0, a_o, a_i)
      _finish_direct_unpadded_group(g1, b_o, b_i)
      return

    def _repair_lane_overflow(g, eff_len_g, top_o, top_i):
      """Merges the top-k of the lane-pruning losers into `top`."""
      r0_o, r0_i = _load_direct_chunk_valid(g, eff_len_g, 0, mask_idx=False)
      r1_o, r1_i = _load_direct_chunk_valid(g, eff_len_g, 1, mask_idx=False)
      r2_o, r2_i = _load_direct_chunk_valid(g, eff_len_g, 2, mask_idx=False)
      r3_o, r3_i = _load_direct_chunk_valid(g, eff_len_g, 3, mask_idx=False)
      l_o, l_i = tc_topk.reduce_4r_to_2r_lane_losers(
          r0_o,
          r0_i,
          r1_o,
          r1_i,
          r2_o,
          r2_i,
          r3_o,
          r3_i,
          ordered_inputs=True,
      )
      cand_o, cand_i = tc_topk.bitonic_topk_2r_to_r(
          l_o[:direct_blocks],
          l_i[:direct_blocks],
          l_o[direct_blocks:],
          l_i[direct_blocks:],
          descending=False,
          bitonic_lanes=True,
      )
      fixed_o, fixed_i = tc_topk.bitonic_topk_merge_blocks(
          top_o, top_i, cand_o, cand_i, descending=True
      )
      _finish_direct_group(g, fixed_o, fixed_i, mask_invalid_idx=True)

    if fixup_only:
      assert scratch is not None
      ovf_ref, top_o_ref, top_i_ref = scratch.ovf, scratch.top_o, scratch.top_i
      for g in groups:

        @pl.when(jnp.any(ovf_ref[g] != 0))
        def _repair_deferred(g=g):
          top_o = tuple(top_o_ref[g, b] for b in range(direct_blocks))
          top_i = tuple(top_i_ref[g, b] for b in range(direct_blocks))
          _repair_lane_overflow(g, eff_len_ref[g], top_o, top_i)

      return

    def _direct_one_group(g):
      eff_len_g = eff_len_ref[g]
      if num_direct_chunks == 1:
        ch_o, ch_i = _load_direct_chunk_valid(g, eff_len_g, 0)
        if direct_blocks % m_part == 0:
          s_o, s_i = tc_topk.bitonic_sort_blocks(
              ch_o, ch_i, descending=True, ordered_inputs=True
          )
          _finish_direct_unpadded_group(g, s_o, s_i)
        else:
          top_o, top_i = tc_topk.sort_blocks_padded(
              ch_o,
              ch_i,
              direct_blocks,
              descending=True,
              ordered_inputs=True,
          )
          _finish_direct_group(g, top_o, top_i)
        return
      num_full_chunks = m_part // direct_blocks
      bot_max_ord = None
      if num_full_chunks >= 4 and direct_blocks <= 8:
        c0_o, c0_i = _load_direct_chunk_valid(g, eff_len_g, 0, mask_idx=False)
        c1_o, c1_i = _load_direct_chunk_valid(g, eff_len_g, 1, mask_idx=False)
        c2_o, c2_i = _load_direct_chunk_valid(g, eff_len_g, 2, mask_idx=False)
        c3_o, c3_i = _load_direct_chunk_valid(g, eff_len_g, 3, mask_idx=False)
        top_o, top_i = tc_topk.bitonic_topk_4r_to_r(
            c0_o + c1_o + c2_o + c3_o,
            c0_i + c1_i + c2_i + c3_i,
            descending=True,
            ordered_inputs=True,
        )
        ch = 4
      elif num_full_chunks >= 4:
        c0_o, c0_i = _load_direct_chunk_valid(g, eff_len_g, 0, mask_idx=False)
        c1_o, c1_i = _load_direct_chunk_valid(g, eff_len_g, 1, mask_idx=False)
        c2_o, c2_i = _load_direct_chunk_valid(g, eff_len_g, 2, mask_idx=False)
        c3_o, c3_i = _load_direct_chunk_valid(g, eff_len_g, 3, mask_idx=False)
        w_o, w_i, bot_max_ord = tc_topk.reduce_4r_to_2r_lane_bitonic(
            c0_o,
            c0_i,
            c1_o,
            c1_i,
            c2_o,
            c2_i,
            c3_o,
            c3_i,
            ordered_inputs=True,
        )
        top_o, top_i = tc_topk.bitonic_topk_2r_to_r(
            w_o[:direct_blocks],
            w_i[:direct_blocks],
            w_o[direct_blocks:],
            w_i[direct_blocks:],
            descending=True,
            bitonic_lanes=True,
        )
        ch = 4
      elif num_full_chunks >= 2:
        c0_o, c0_i = _load_direct_chunk_valid(g, eff_len_g, 0, mask_idx=False)
        c1_o, c1_i = _load_direct_chunk_valid(g, eff_len_g, 1, mask_idx=False)
        top_o, top_i = tc_topk.bitonic_topk_2r_to_r(
            c0_o, c0_i, c1_o, c1_i, descending=True, ordered_inputs=True
        )
        ch = 2
      else:
        c0_o, c0_i = _load_direct_chunk_valid(g, eff_len_g, 0, mask_idx=False)
        top_o, top_i = tc_topk.bitonic_sort_blocks(
            c0_o, c0_i, descending=True, ordered_inputs=True
        )
        ch = 1

      while ch + 4 <= num_full_chunks and direct_blocks <= 8:
        ca_o, ca_i = _load_direct_chunk_valid(g, eff_len_g, ch, mask_idx=False)
        cb_o, cb_i = _load_direct_chunk_valid(
            g, eff_len_g, ch + 1, mask_idx=False
        )
        cc_o, cc_i = _load_direct_chunk_valid(
            g, eff_len_g, ch + 2, mask_idx=False
        )
        cd_o, cd_i = _load_direct_chunk_valid(
            g, eff_len_g, ch + 3, mask_idx=False
        )
        cand_o, cand_i = tc_topk.bitonic_topk_4r_to_r(
            ca_o + cb_o + cc_o + cd_o,
            ca_i + cb_i + cc_i + cd_i,
            descending=False,
            ordered_inputs=True,
        )
        top_o, top_i = tc_topk.bitonic_topk_merge_blocks(
            top_o, top_i, cand_o, cand_i, descending=True
        )
        ch += 4

      while ch + 2 <= num_full_chunks:
        ca_o, ca_i = _load_direct_chunk_valid(g, eff_len_g, ch, mask_idx=False)
        cb_o, cb_i = _load_direct_chunk_valid(
            g, eff_len_g, ch + 1, mask_idx=False
        )
        cand_o, cand_i = tc_topk.bitonic_topk_2r_to_r(
            ca_o, ca_i, cb_o, cb_i, descending=False, ordered_inputs=True
        )
        top_o, top_i = tc_topk.bitonic_topk_merge_blocks(
            top_o, top_i, cand_o, cand_i, descending=True
        )
        ch += 2

      if ch < num_full_chunks:
        ch_o, ch_i = _load_direct_chunk_valid(g, eff_len_g, ch, mask_idx=False)
        cand_o, cand_i = tc_topk.bitonic_sort_blocks(
            ch_o, ch_i, descending=False, ordered_inputs=True
        )
        top_o, top_i = tc_topk.bitonic_topk_merge_blocks(
            top_o, top_i, cand_o, cand_i, descending=True
        )

      if m_part > num_full_chunks * direct_blocks:
        rem_o, rem_i = _load_direct_chunk_valid(
            g, eff_len_g, num_full_chunks, mask_idx=False
        )
        cand_o, cand_i = tc_topk.sort_blocks_padded(
            rem_o,
            rem_i,
            direct_blocks,
            descending=False,
            ordered_inputs=True,
        )
        top_o, top_i = tc_topk.bitonic_topk_merge_blocks(
            top_o, top_i, cand_o, cand_i, descending=True
        )

      if bot_max_ord is not None and scratch is not None:
        # Keep the top-k and the overflow flag for the deferred repair.
        for b in range(direct_blocks):
          scratch.top_o[g, b] = top_o[b]
          scratch.top_i[g, b] = top_i[b]
        tau_x = tc_topk.exclusive_threshold(
            tc_topk.extract_row_kth_threshold(top_o, 128 * direct_blocks)
        )
        scratch.ovf[g] = jnp.where(bot_max_ord > tau_x, 1, 0).astype(
            jnp.int32
        )
        _finish_direct_group(g, top_o, top_i, mask_invalid_idx=True)
        return
      _finish_direct_group(g, top_o, top_i, mask_invalid_idx=True)
      if bot_max_ord is not None:
        tau_x = tc_topk.exclusive_threshold(
            tc_topk.extract_row_kth_threshold(top_o, 128 * direct_blocks)
        )

        @pl.when(jnp.any(bot_max_ord > tau_x))
        def _fixup_rare_lane_overflow():
          _repair_lane_overflow(g, eff_len_g, top_o, top_i)

    for g in groups:
      _direct_one_group(g)

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

  def _select_ingest_tile(
      select_refs, eff_len_g, g, cur_bkv, m_p, s_tile, fold_cell=None
  ):
    """Stores one score tile of block `cur_bkv` as masked order keys."""
    idx_tile = cur_bkv * bkv_sz + m_p * 128 + _part_col_offset()
    keys = tc_topk.f32_to_ord_i32(s_tile, idx_tile < eff_len_g)
    select_refs[0][g, cur_bkv * m_part + m_p] = keys
    if fold_cell is not None:
      fold_cell(g, m_p, keys)

  def _select_cell_folder(cur_bkv, select_refs):
    """Returns `fold(g, m_p, keys)` summarizing block `cur_bkv` on the fly.

    Tracks each lane's max key of query group `g` (and the first tile holding
    it) in registers as the tiles arrive, which must be in ascending `m_p`
    order per group, and writes the cells after the group's last tile, so the
    keys are never re-read from VMEM.

    Args:
      cur_bkv: KV block being ingested.
      select_refs: `(keys_ref, cell_max_ref, cell_arg_ref)`.

    Returns:
      The fold callback, or `None` when no cells are needed.
    """
    if not use_two_level:
      return None
    state = {}

    def fold(g, m_p, keys):
      if m_p == 0:
        best, arg = keys, jnp.zeros((8, 128), jnp.int32)
      else:
        best, arg, prev = state.pop(g)
        assert prev == m_p - 1, (g, prev, m_p)
        better = keys > best
        best = jnp.where(better, keys, best)
        arg = jnp.where(better, jnp.int32(m_p), arg)
      if m_p == m_part - 1:
        select_refs[1][g, cur_bkv] = best
        select_refs[2][g, cur_bkv] = arg
      else:
        state[g] = (best, arg, m_p)

    return fold

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
            max_blocks=num_kv_blocks,
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
      fin_i = tc_topk.lane_major_to_block_major(fin_i)
      if need_scores:
        fin_o = tc_topk.lane_major_to_block_major(fin_o)
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

    rows = (
        seq_along_lane_block_rows(bkv_vmem)
        if seq_along_lane
        else head_along_sublane_block_rows(bkv_vmem)
    )

    fold_cell = (
        _select_cell_folder(cur_bkv, select_refs) if use_select else None
    )

    def emit(g, m_p, tile):
      if use_select:
        _select_ingest_tile(
            select_refs, eff_len_ref[g], g, cur_bkv, m_p, tile, fold_cell
        )
      else:
        assert scores_tile_ref is not None
        scores_tile_ref[g, m_p] = tile

    if decode_tiles:

      def _score_decode(wb_ref):
        w_rows = weights_vmem[...].astype(jnp.float32)
        for b in range(bq_total):
          wb_ref[b] = jnp.broadcast_to(
              w_rows[b : b + 1].reshape(1, num_q_heads, 1),
              (1, num_q_heads, 128),
          )[0]
        _decode_score_tiles(q_vmem, wb_ref, rows, emit)

      pl.run_scoped(
          _score_decode,
          pltpu.VMEM((bq_total, num_q_heads, 128), jnp.float32),
      )
    else:
      if q_hm_ref is not None:
        bq_vec = [q_hm_ref[...]]
      else:
        q_data = q_vmem[...].reshape(seq_batch_size, -1, head_dim)
        bq_vec = [q_data[batch_idx] for batch_idx in range(seq_batch_size)]
      w_data = weights_vmem[...].reshape(seq_batch_size, -1, num_q_heads)
      bq_weights_vec = [
          w_data[batch_idx] for batch_idx in range(seq_batch_size)
      ]

      bkvs = []
      bkv_scales = []
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
        else:
          fp8_val = head_along_sublane_keys(
              rows, batch_idx, 0, bkv_sz_per_kv_packing
          )
          scale_val = head_along_sublane_scales(
              rows, batch_idx, 0, bkv_sz_per_kv_packing
          )
        bkvs.append(fp8_val)
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

      for g in range(num_q_groups):
        grp_scores = scores_padded[g * 8 : (g + 1) * 8]
        for m_p in range(m_part):
          emit(g, m_p, grp_scores[:, m_p, :])

    n_act = _active_q_groups(p_id)
    if use_select:

      @pl.when(is_last_bkv)
      def _on_last_select():
        _select_finalize(
            n_act, cur_bkv + 1, select_refs, top_o_ref, top_i_ref, out_vmem
        )

      return

    assert scores_tile_ref is not None
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
    seqs_per_tb = prefill_cfg.seqs_per_tb
    src_step = prefill_cfg.src_step
    tbs_per_batch = prefill_cfg.tbs_per_batch
    num_head_groups = prefill_cfg.num_head_groups
    heads_per_group = prefill_cfg.heads_per_group
    rows_per_group_tok_blk = prefill_cfg.rows_per_group_tok_blk
    rows_per_group = prefill_cfg.rows_per_group
    lane_chunk = prefill_cfg.lane_chunk
    num_lane_chunks = prefill_cfg.num_lane_chunks
    tiles_per_chunk = prefill_cfg.tiles_per_chunk
    groups_per_tb = min(num_q_groups, tok_blk // 8)
    batch_rows = tbs_per_batch * rows_per_group_tok_blk
    # Single-KV-block direct top-k runs per token block, fused with staging and
    # scoring the next token block so both share VLIW bundles. The token
    # block's query groups are ranked in one basic block, which overlaps their
    # latency-bound sorting networks; with many tiles per row their register
    # pressure serializes them instead, so only short rows are fused.
    fuse_topk = (
        single_bkv
        and not use_select
        and seq_batch_size == 1
        and num_tok_blks > 1
        and m_part <= _MAX_FUSED_PREFILL_TILES
    )
    # Longer rows rank one query group pair per basic block, next to scoring
    # half of the next token block, when top-k uses at most 8 output blocks or
    # defers its rare lane-overflow repair out of the loop.
    fuse_pairs = (
        single_bkv
        and not use_select
        and seq_batch_size == 1
        and num_tok_blks > 1
        and not fuse_topk
        and (num_blocks <= _MAX_FUSED_PREFILL_TILES or lane_pruned)
        and groups_per_tb == 4
        and num_lane_chunks >= 2
    )

    def stage_token_block(tb, w=None, q_all=None):
      """Writes token block `tb`'s head-major queries and lane weights."""
      t0 = tb * src_step
      if w is None:
        w_tb = weights_vmem[pl.ds(t0, src_step)].astype(jnp.float32)
      else:
        w_tb = w[t0 : t0 + src_step]
      q_tb = (
          q_vmem[pl.ds(t0, src_step)]
          if q_all is None
          else q_all[t0 : t0 + src_step]
      )
      if src_step < tok_blk:
        pad_n = tok_blk - src_step
        w_tb = jnp.pad(w_tb, ((0, pad_n), (0, 0)))
        q_tb = jnp.pad(q_tb, ((0, pad_n), (0, 0), (0, 0)))
      q_rows = transpose_to_head_major(q_tb, tok_blk)
      for g_h in range(num_head_groups):
        q_hm_ref[
            pl.ds(
                g_h * rows_per_group + tb * rows_per_group_tok_blk,
                rows_per_group_tok_blk,
            )
        ] = q_rows[
            g_h * rows_per_group_tok_blk : (g_h + 1) * rows_per_group_tok_blk
        ]
      for h in range(num_q_heads):
        wb_ref[tb, h] = jnp.broadcast_to(
            w_tb[:, h : h + 1], (tok_blk, 128)
        ).astype(wb_ref.dtype)

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
      if not fuse_topk:
        w = weights_vmem[...].astype(jnp.float32)
        q_all = q_vmem[...] if src_step < tok_blk else None
        for tb in range(num_tok_blks):
          stage_token_block(tb, w, q_all)

    if seq_along_lane:
      rows = seq_along_lane_block_rows(bkv_vmem)

      def load_chunk(c, batch_idx=0):
        tiles = [c * tiles_per_chunk + t for t in range(tiles_per_chunk)]
        keys = jnp.concatenate(
            [
                seq_along_lane_key_tile(rows, batch_idx, lane_tile_idx)
                for lane_tile_idx in tiles
            ],
            1,
        )
        scales = jnp.concatenate(
            [
                seq_along_lane_scale_tile(rows, batch_idx, lane_tile_idx)
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
      has_rows = head_along_sublane_block_rows(bkv_vmem)

      def load_chunk(c, batch_idx=0):
        g0 = c * groups_per_chunk
        keys = head_along_sublane_keys(
            has_rows, batch_idx, g0, groups_per_chunk
        )
        scales = head_along_sublane_scales(
            has_rows, batch_idx, g0, groups_per_chunk
        )
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
          w_h = wb_ref[tb, g_h * heads_per_group + i]
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

    def score_token_block(tb, dst_tb, chunks=None):
      """Scores token block `tb` into the tiles of token block `dst_tb`.

      Args:
        tb: Token block whose staged queries are scored.
        dst_tb: Token block whose score tiles receive the result.
        chunks: Optional `(first, count)` lane chunk range; `first` may be
          traced. Defaults to all lane chunks.
      """
      first, count = (0, num_lane_chunks) if chunks is None else chunks
      for j in range(count):
        c = first + j
        keys, scales = load_chunk(c)
        sts = [
            qk(
                q_hm_ref[
                    pl.ds(
                        g_h * rows_per_group + tb * rows_per_group_tok_blk,
                        rows_per_group_tok_blk,
                    )
                ],
                keys,
            )
            for g_h in range(num_head_groups)
        ]
        acc = reduce_token_block(sts, tb, 0) * scales
        for jj in range(tiles_per_chunk):
          m_p = c * tiles_per_chunk + jj
          for sg in range(groups_per_tb):
            scores_tile_ref[dst_tb * groups_per_tb + sg, m_p] = acc[
                sg * 8 : (sg + 1) * 8, jj * 128 : (jj + 1) * 128
            ]

    n_act = _active_q_groups(p_id)
    if fuse_topk:
      # Iteration `it` runs the top-k of token blocks `[it * tpi, (it + 1) *
      # tpi)` and stages and scores the next `tpi` token blocks; the last
      # iteration scores into spare tiles past the query block. Score writes
      # never alias the tiles being ranked.
      tpi = _fused_prefill_token_blocks(m_part, num_tok_blks)
      for l in range(tpi):
        stage_token_block(l)
        score_token_block(l, l)
      last_group = n_act - 1

      def _fused(it, _):
        for l in range(tpi):
          for p in range(0, groups_per_tb, 2):
            g0 = (it * tpi + l) * groups_per_tb + p
            groups = (jnp.minimum(g0, last_group),)
            if groups_per_tb > 1:
              groups += (jnp.minimum(g0 + 1, last_group),)
            _direct_topk_groups(scores_tile_ref, eff_len_ref, out_vmem, groups)
        for l in range(tpi):
          dst = (it + 1) * tpi + l
          nxt = jnp.minimum(dst, num_tok_blks - 1)
          stage_token_block(nxt)
          score_token_block(nxt, dst)
        return None

      lax.fori_loop(0, pl.cdiv(n_act, tpi * groups_per_tb), _fused, None)
      return

    if fuse_pairs:
      # Iteration `i` runs the top-k of query group pair `i` and scores half of
      # the lane chunks of the next token block, so one pair's sorting network
      # shares a basic block with scoring.
      half_chunks = cdiv(num_lane_chunks, 2)
      rem_chunks = num_lane_chunks - half_chunks
      score_token_block(0, 0)
      last_group = n_act - 1

      def _run_pairs(scratch):
        def _paired(i, _):
          groups = (
              jnp.minimum(2 * i, last_group),
              jnp.minimum(2 * i + 1, last_group),
          )
          _direct_topk_groups(
              scores_tile_ref, eff_len_ref, out_vmem, groups, scratch=scratch
          )
          tb = i // 2
          nxt = jnp.minimum(tb + 1, num_tok_blks - 1)
          score_token_block(nxt, tb + 1, ((i % 2) * rem_chunks, half_chunks))
          return None

        lax.fori_loop(0, pl.cdiv(n_act, 2), _paired, None)

      _with_direct_scratch(
          scores_tile_ref, eff_len_ref, out_vmem, n_act, _run_pairs
      )
      return

    fold_cell = (
        _select_cell_folder(cur_bkv, select_refs) if use_select else None
    )
    if seq_batch_size == 1:
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
                      select_refs,
                      eff_len_ref[grp],
                      grp,
                      cur_bkv,
                      m_p,
                      tile,
                      fold_cell,
                  )
                else:
                  scores_tile_ref[grp, m_p] = tile
    else:
      slot_st = slot_sc = None
      if seqs_per_tb > 1:
        st_rows = lax.broadcasted_iota(
            jnp.int32, (rows_per_group_tok_blk, lane_chunk), 0
        )
        slot_st = (st_rows % tok_blk) // bq_sz
        slot_sc = (
            lax.broadcasted_iota(jnp.int32, (tok_blk, lane_chunk), 0) // bq_sz
        )
      for tb in range(num_tok_blks):
        q_tb_groups = [
            q_hm_ref[
                pl.ds(
                    g_h * rows_per_group + tb * rows_per_group_tok_blk,
                    rows_per_group_tok_blk,
                )
            ]
            for g_h in range(num_head_groups)
        ]
        for c in range(num_lane_chunks):
          keys, scales = load_chunk(c, tb * seqs_per_tb)
          sts = [qk(q_g, keys) for q_g in q_tb_groups]
          if seqs_per_tb > 1:
            scales = jnp.broadcast_to(scales, (tok_blk, lane_chunk))
            for slot in range(1, seqs_per_tb):
              keys_s, scales_s = load_chunk(c, tb * seqs_per_tb + slot)
              sts = [
                  jnp.where(slot_st == slot, qk(q_g, keys_s), st)
                  for q_g, st in zip(q_tb_groups, sts)
              ]
              scales = jnp.where(slot_sc == slot, scales_s, scales)
          acc = reduce_token_block(sts, tb, 0) * scales
          for jj in range(tiles_per_chunk):
            acc_tile = acc[:, jj * 128 : (jj + 1) * 128]
            for sg in range(groups_per_tb):
              grp = tb * groups_per_tb + sg
              m_p = c * tiles_per_chunk + jj
              tile = acc_tile[sg * 8 : (sg + 1) * 8]
              if use_select:
                _select_ingest_tile(
                    select_refs,
                    eff_len_ref[grp],
                    grp,
                    cur_bkv,
                    m_p,
                    tile,
                    fold_cell,
                )
              else:
                scores_tile_ref[grp, m_p] = tile
        if single_bkv and not use_select and seqs_per_tb > 1:
          if groups_per_tb == 2:
            _direct_topk_groups(
                scores_tile_ref, eff_len_ref, out_vmem, (2 * tb, 2 * tb + 1)
            )
          elif groups_per_tb == 4:
            for g0 in (4 * tb, 4 * tb + 2):
              _direct_topk_groups(
                  scores_tile_ref, eff_len_ref, out_vmem, (g0, g0 + 1)
              )
          elif groups_per_tb == 1 and tb % 2 == 1:
            _direct_topk_groups(
                scores_tile_ref, eff_len_ref, out_vmem, (tb - 1, tb)
            )
      if (
          single_bkv
          and not use_select
          and seqs_per_tb > 1
          and groups_per_tb in (1, 2, 4)
      ):
        if groups_per_tb == 1 and num_tok_blks % 2 == 1:
          _direct_topk_groups(
              scores_tile_ref, eff_len_ref, out_vmem, (num_tok_blks - 1,)
          )
        return

    n_act = _active_q_groups(p_id)
    if use_select:

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
    elif single_bkv and lane_pruned and not use_select:
      top_vmem = pltpu.VMEM((num_q_groups, direct_blocks, 8, 128), jnp.int32)

      @pl.with_scoped(
          top_o=top_vmem,
          top_i=top_vmem,
          ovf=pltpu.VMEM((num_q_groups, 8, 128), jnp.int32),
      )
      def _with_direct(top_o, top_i, ovf):
        direct_refs["scratch"] = _DirectScratch(top_o, top_i, ovf)
        _run_pipeline()

      _with_direct()
    else:
      _run_pipeline()

  vec_vmem = pltpu.VMEM((num_q_groups, 8, 128), jnp.int32)
  scores_vmem = pltpu.VMEM((num_q_groups, m_part, 8, 128), jnp.float32)
  q_hm_rows = num_q_heads * prefill_cfg.num_tok_blks * prefill_cfg.tok_blk

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
          pltpu.VMEM((q_hm_rows, head_dim), q_hbm_ref.dtype),
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
      # The fused prefill loop scores token blocks past the query block.
      spare_groups = _MAX_FUSED_PREFILL_TOKEN_BLOCKS * prefill_cfg.tok_blk // 8
      pl.run_scoped(
          lambda eff_r, sc_r, q_hm, wb_r: _run_all(
              None, None, None, None, None, eff_r, sc_r, q_hm, wb_r
          ),
          vec_vmem,
          pltpu.VMEM(
              (num_q_groups + spare_groups, m_part, 8, 128), jnp.float32
          ),
          pltpu.VMEM((q_hm_rows, head_dim), q_hbm_ref.dtype),
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
          pltpu.VMEM((q_hm_rows, head_dim), q_hbm_ref.dtype),
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
    # Scalar prefetches (SMEM)
    seq_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    dist_ref,  # [3] request distribution
    cp_rank_ref,  # [1]
    # HBM refs
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    indexer_weights_hbm_ref,  # [max_num_tokens, num_q_heads]
    cache_kv_hbm_ref,
    *out_refs,
    start_bound: SeqBound,
    end_bound: SeqBound,
    chunk_start: int | None,
    fill_tail: bool,
    vis_range: VisRange = None,
    **kwargs: Any,
):
  """Builds the pass schedule in SMEM, then runs `_tc_kernel_body`.

  Args:
    seq_lens_ref: `[max_num_seqs]` uncompressed sequence lengths.
    page_indices_ref: Flat page table.
    cu_q_lens_ref: `[max_num_seqs + 1]` cumulative query lengths.
    dist_ref: `[3]` request distribution.
    cp_rank_ref: `[1]` context-parallel rank.
    q_hbm_ref: Queries.
    indexer_weights_hbm_ref: Indexer weights.
    cache_kv_hbm_ref: Packed index KV cache.
    *out_refs: Output (and its aliased input, if any) HBM refs.
    start_bound: First sequence of the pass.
    end_bound: End sequence (exclusive) of the pass.
    chunk_start: First token of this chunk of the token axis, or None when
      the token axis is not chunked.
    fill_tail: Whether this is the last pass, which writes empty rows for
      every token past the active ones.
    vis_range: Visible local KV length range of the query blocks this pass
      runs (see `VisRange`); the other passes run the rest. When its upper
      bound fits in one KV block, the pass is single-KV-block.
    **kwargs: Static arguments of `_tc_kernel_body`.
  """
  out_hbm_ref = out_refs[-1]
  seq_batch_size = kwargs["seq_batch_size"]
  bq_sz = kwargs["bq_sz"]
  bkv_p = kwargs["bkv_p"]
  cp_size = kwargs.get("cp_size", 1)
  if kwargs.get("kv_layout", KVLayout.HEAD_ALONG_SUBLANE) == (
      KVLayout.SEQ_ALONG_LANE
  ):
    page_size = cache_kv_hbm_ref.shape[-1]
  else:
    page_size = cache_kv_hbm_ref.shape[1] * cache_kv_hbm_ref.shape[2]
  max_num_seqs = seq_lens_ref.shape[0]
  num_kv_blocks = cdiv(page_indices_ref.shape[0] // max_num_seqs, bkv_p)
  single_kv_block = (
      vis_range is not None
      and vis_range[1] is not None
      and vis_range[1] <= bkv_p * page_size
  )
  if single_kv_block:
    num_kv_blocks = 1
  chunk_tokens = None if chunk_start is None else out_hbm_ref.shape[0]
  q_cap, table_cap = schedule_capacity(
      max_num_seqs=max_num_seqs,
      max_num_tokens=q_hbm_ref.shape[0],
      seq_batch_size=seq_batch_size,
      bq_sz=bq_sz,
      static_q_len=kwargs["static_q_len"],
      chunk_tokens=chunk_tokens,
      num_kv_blocks=num_kv_blocks,
  )

  def _scoped(info, bt, bq, q_start, q_end, step_qb=None):
    sched = ScheduleRef(
        info=info,
        batch_tile_idx=bt,
        bq_idx=bq,
        q_start=q_start,
        q_end=q_end,
        step_qb=step_qb,
        chunk_start=0 if chunk_start is None else chunk_start,
    )
    build_schedule(
        sched,
        seq_lens_ref,
        cu_q_lens_ref,
        dist_ref,
        cp_rank_ref[0] if cp_size > 1 else 0,
        start_bound=start_bound,
        end_bound=end_bound,
        seq_batch_size=seq_batch_size,
        bq_sz=bq_sz,
        bkv_sz=bkv_p * page_size,
        num_kv_blocks=num_kv_blocks,
        compression_ratio=kwargs["compression_ratio"],
        static_q_len=kwargs["static_q_len"],
        chunk_tokens=chunk_tokens,
        cp_size=cp_size,
        interleave_c=kwargs.get("interleave_c", 1),
        vis_range=vis_range,
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
        *out_refs,
        single_kv_block=single_kv_block,
        **kwargs,
    )

  smem = lambda n: pltpu.SMEM((n,), jnp.int32)
  start = _seq_bound(dist_ref, start_bound)
  end = _seq_bound(dist_ref, end_bound)
  t_lo = (start + (seq_batch_size - 1)) // seq_batch_size
  t_hi = (end + (seq_batch_size - 1)) // seq_batch_size

  @pl.when(t_hi > t_lo)
  def _run_pass():
    pl.run_scoped(
        _scoped,
        smem(3),
        *(smem(q_cap) for _ in range(4)),
        *((smem(table_cap),) if table_cap else ()),
    )

  if fill_tail:
    num_seqs_active = jnp.clip(dist_ref[2], 0, max_num_seqs)
    _, num_blocks = tc_topk.topk_capacity(kwargs["k"])
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

  def _block_sizes(
      num_queries_per_block,
      bkv_p,
      static_q_len,
      chunk_len,
      pages=pages_per_seq,
  ):
    if static_q_len is not None:
      bq_sz = min(num_queries_per_block, static_q_len)
    else:
      bq_sz = num_queries_per_block
    if bq_sz > 32 and chunk_len < bq_sz:
      chunk_cap = max(32, 1 << (max(1, chunk_len) - 1).bit_length())
      bq_sz = min(bq_sz, chunk_cap)
    if topk_impl != TOPK_IMPL_STREAM and bq_sz > 8 and pages > bkv_p:
      # Shrink the query block until its resident `select` keys fit in VMEM:
      # streaming Top-K over many KV blocks is far slower than `select`.
      max_groups = select_budget // _select_group_bytes(bkv_p)
      if 1 <= max_groups < cdiv(bq_sz, 8):
        bq_sz = min(bq_sz, 1 << ((8 * max_groups).bit_length() - 1))
    return bq_sz, page_size * bkv_p

  def _pass_topk_impl(bq_sz, bkv_p, seq_batch_size, pages=pages_per_seq):
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
    if pages <= bkv_p:
      return TOPK_IMPL_STREAM
    num_kv_blocks = cdiv(pages_per_seq, bkv_p)
    if (
        num_kv_blocks * tiles_per_block
        <= _STREAM_MAX_TOTAL_TILES_PER_TOPK_BLOCK * num_blocks
        or tiles_per_block < 4
    ):
      return TOPK_IMPL_STREAM
    key_bytes = cdiv(bq_total, 8) * _select_group_bytes(bkv_p)
    return TOPK_IMPL_SELECT if key_bytes <= select_budget else TOPK_IMPL_STREAM

  def _pass_kernel_args(
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
      vis_range: VisRange = None,
  ):
    """Returns `(name, tc_kernel kwargs)` of one fused pass."""
    max_num_tokens = q.shape[0]
    if seq_batch_size > 1:
      assert static_q_len == 1 or (
          static_q_len is not None and static_q_len <= num_queries_per_block
      )

    chunked = chunk_token_start is not None
    eff_chunk_len = max_num_tokens if not chunked else chunk_len

    bkv_p = num_kv_pages_per_block
    # A pass whose query blocks see at most one KV block of KV is single-block.
    pages = pages_per_seq
    if (
        vis_range is not None
        and vis_range[1] is not None
        and vis_range[1] <= page_size * bkv_p
    ):
      pages = min(pages, bkv_p)
    bq_sz, _ = _block_sizes(
        num_queries_per_block, bkv_p, static_q_len, eff_chunk_len, pages
    )
    pass_topk_impl = _pass_topk_impl(bq_sz, bkv_p, seq_batch_size, pages)

    layout_tag = "" if kv_layout == KVLayout.HEAD_ALONG_SUBLANE else "-sal"
    name = f"{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}{layout_tag}"
    if vis_range is not None and vis_range[1] is not None:
      name += f"-v{vis_range[1]}"
    if pass_topk_impl == TOPK_IMPL_SELECT:
      name += "-sel"
    return name, dict(
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
        topk_impl=pass_topk_impl,
        start_bound=start_bound,
        end_bound=end_bound,
        chunk_start=chunk_token_start,
        vis_range=vis_range,
    )

  def run_fused_passes(
      pass_specs,
      *,
      uni_fast_spec: dict[str, Any] | None = None,
      chunk_token_start: int | None,
      chunk_len: int,
      scheduling_group_id: int | None,
  ):
    """Runs every pass of one chunk, in order, in a single kernel."""
    max_num_tokens = q.shape[0]
    chunked = chunk_token_start is not None
    eff_chunk_len = max_num_tokens if not chunked else chunk_len
    named = [
        _pass_kernel_args(
            chunk_token_start=chunk_token_start, chunk_len=chunk_len, **spec
        )
        for spec in pass_specs
    ]
    pass_kwargs = tuple(kw for _, kw in named)
    num_passes = len(pass_kwargs)
    uni_fast_kw = None
    if uni_fast_spec is not None:
      _, uni_fast_kw = _pass_kernel_args(
          chunk_token_start=chunk_token_start,
          chunk_len=chunk_len,
          **uni_fast_spec,
      )

    def kernel(*refs):
      if uni_fast_kw is None:
        for i, kw in enumerate(pass_kwargs):
          # Only an unchunked call's last pass writes the empty tail rows.
          tc_kernel(*refs, fill_tail=not chunked and i == num_passes - 1, **kw)
        return
      dist_ref = refs[3]
      pure_uni = (dist_ref[0] == 0) & (dist_ref[1] == dist_ref[2])

      @pl.when(pure_uni)
      def _():
        tc_kernel(*refs, fill_tail=not chunked, **uni_fast_kw)

      @pl.when(~pure_uni)
      def _():
        for i, kw in enumerate(pass_kwargs):
          tc_kernel(*refs, fill_tail=not chunked and i == num_passes - 1, **kw)

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

  def _uni_fast_spec() -> dict[str, Any] | None:
    if (
        uniform_q_len is None
        or uniform_q_len <= 1
        or q.shape[0] != max_num_seqs * uniform_q_len
    ):
      return None
    bq_u = min(num_queries_per_blocks[1], uniform_q_len)
    bkv_p_u = num_kv_pages_per_blocks[1]
    bkv_u = bkv_p_u * page_size
    if (
        bq_u != uniform_q_len
        or bq_u not in (2, 4, 8, 16)
        or pages_per_seq > bkv_p_u
        or bkv_u > 4096
    ):
      return None
    if bq_u == 2:
      target_sb = 16
    elif bq_u <= 8:
      target_sb = 8
    else:
      target_sb = 4
    cand_sb = math.gcd(max_num_seqs, target_sb)
    prefill_u = _BlockedPrefillConfig.create(
        bq_sz=bq_u,
        bkv_sz=bkv_u,
        num_q_heads=q.shape[1],
        q_dtype=q.dtype,
        seq_batch_size=cand_sb,
    )
    if (
        cand_sb <= 1
        or not prefill_u.enabled
        or cand_sb % max(prefill_u.seqs_per_tb, 8 // bq_u) != 0
        or (chunk_tokens is not None and chunk_tokens % (cand_sb * bq_u) != 0)
    ):
      return None
    return dict(
        num_kv_pages_per_block=bkv_p_u,
        num_queries_per_block=num_queries_per_blocks[1],
        buffer_count=buffer_counts[1],
        start_bound=None,
        end_bound=(1, 1),
        static_q_len=uniform_q_len,
        seq_batch_size=cand_sb,
        case=MlaCase.PREFILL,
    )

  def _kv_bucket_passes(spec: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Splits a single-KV-block query-block pass into KV-length buckets.

    Each bucket pass has a smaller static KV block and runs only the query
    blocks whose visible KV fits in it, so their single-KV-block top-k sorts
    fewer (and a power of two of) score tiles. `spec` itself runs the
    remaining query blocks, last. The routing is decided per query block when
    the in-kernel schedule is built.

    Args:
      spec: Pass spec of a `seq_batch_size == 1` query-block pass.

    Returns:
      The bucket passes followed by `spec`.
    """
    bkv_p = spec["num_kv_pages_per_block"]
    top = min(pages_per_seq, bkv_p)
    if (
        topk_impl == TOPK_IMPL_SELECT
        or spec["seq_batch_size"] != 1
        or spec["static_q_len"] is not None
        or spec["num_queries_per_block"] < _KV_BUCKET_MIN_QUERIES_PER_BLOCK
        or k > _KV_BUCKET_MAX_K
        or pages_per_seq > bkv_p
        or top <= 1
        or _uni_fast_spec() is not None
    ):
      return (spec,)
    bucket_pages = []
    # Buckets smaller than `k` mostly sort padding blocks; their passes cost
    # more instruction memory than they save.
    min_positions = max(_MIN_KV_BUCKET_POSITIONS, k)
    p = 1 << ((top - 1).bit_length() - 1)
    while (
        p >= 1
        and len(bucket_pages) < _MAX_KV_BUCKET_PASSES
        and p * page_size >= min_positions
        and (p * page_size) % 128 == 0
    ):
      bucket_pages.append(p)
      p //= 2
    if not bucket_pages:
      return (spec,)
    bucket_pages.sort()
    passes = []
    lo = -1
    for p in bucket_pages:
      hi = p * page_size
      passes.append(
          dict(spec, num_kv_pages_per_block=p, vis_range=(lo, hi))
      )
      lo = hi
    passes.append(dict(spec, vis_range=(lo, None)))
    return tuple(passes)

  def _pass_specs():
    passes = []
    for spec in _base_pass_specs():
      if spec["case"] == MlaCase.MIXED:
        passes.extend(_kv_bucket_passes(spec))
      else:
        passes.append(spec)
    return tuple(passes)

  def _base_pass_specs():
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
        start_bound=(1, 1),
        end_bound=(2, 1),
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
              start_bound=(0, 1),
              end_bound=(1, 1),
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
                end_bound=(2, 1),
                seq_batch_size=1,
                **decode_cfg,
            ),
        )
      if (max_num_seqs == 1 and q.shape[0] > 1) or (
          num_queries_per_blocks[2] >= 64 and q.shape[0] >= max_num_seqs * 256
      ):
        return (
            dict(
                num_kv_pages_per_block=num_kv_pages_per_blocks[2],
                num_queries_per_block=num_queries_per_blocks[2],
                buffer_count=buffer_counts[2],
                start_bound=None,
                end_bound=(2, 1),
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
            end_bound=(0 if uniform else 1, 1),
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
    uni_fast_spec = _uni_fast_spec()

    chunk_outs = [
        run_fused_passes(
            passes,
            uni_fast_spec=uni_fast_spec,
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
