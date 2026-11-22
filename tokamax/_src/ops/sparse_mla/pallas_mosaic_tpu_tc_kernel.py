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
"""TPU-Friendly MLA Ragged Paged Attention TensorCore kernel."""

import functools
import math

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.dsv4_compressor import csa_cache_layout
from tokamax._src.ops.sparse_mla import pallas_mosaic_tpu_kernel as sc_kernel

_HI16_MASK = -65536  # int32 0xFFFF0000: the high bf16 half of a word.


def _pow2_floor(x: int) -> int:
  """Returns the largest power of two <= `x`."""
  return 1 << (x.bit_length() - 1)


def _divmod_static(x: jax.Array, d: int) -> tuple[jax.Array, jax.Array]:
  """Returns `(x // d, x % d)` for non-negative `x`, using shifts if possible."""
  if d & (d - 1) == 0:
    return lax.shift_right_logical(x, d.bit_length() - 1), lax.bitwise_and(
        x, d - 1
    )
  return x // d, x % d


def _pad_rows(x: jax.Array, num_rows: int, value=0) -> jax.Array:
  """Pads axis 0 of `x` up to `num_rows` rows with `value`."""
  pad = num_rows - x.shape[0]
  if pad == 0:
    return x
  return jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1), constant_values=value)


def _zero_non_finite(x: jax.Array) -> jax.Array:
  """Zeroes NaN / inf values decoded from unwritten or overlaid cache bytes."""
  return jnp.where(jnp.abs(x) <= jnp.finfo(x.dtype).max, x, 0)


def _pack_bf16_pair(lo: jax.Array, hi: jax.Array) -> jax.Array:
  """Packs two bf16 arrays into int32 words (`lo` low half, `hi` high half)."""
  lo_bits = lax.shift_right_logical(
      pltpu.bitcast(lo.astype(jnp.float32), jnp.int32), 16
  )
  hi_bits = jnp.bitwise_and(
      pltpu.bitcast(hi.astype(jnp.float32), jnp.int32), _HI16_MASK
  )
  return jnp.bitwise_or(lo_bits, hi_bits)


def _unpack_bf16_pair(words: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Inverse of `_pack_bf16_pair`."""
  lo = pltpu.bitcast(jnp.left_shift(words, 16), jnp.float32)
  hi = pltpu.bitcast(jnp.bitwise_and(words, _HI16_MASK), jnp.float32)
  return lo.astype(jnp.bfloat16), hi.astype(jnp.bfloat16)


def _rope_words_to_f32_bits(w: jax.Array) -> jax.Array:
  """Expands RoPE words (token in lanes 0..31) to f32 bits in lanes 0..63."""
  lane = lax.broadcasted_iota(jnp.int32, w.shape, 1)
  return jnp.where(
      lane < 32,
      jnp.left_shift(w, 16),
      jnp.bitwise_and(pltpu.roll(w, 32, 1), _HI16_MASK),
  )


def _decode_rope_page(rope_i32: jax.Array, page_size: int) -> jax.Array:
  """Unpacks an int32[page_size // 4, 128] RoPE page to bf16[page_size, 64]."""
  u32_parts = [
      _rope_words_to_f32_bits(
          rope_i32 if j == 0 else pltpu.roll(rope_i32, 128 - 32 * j, 1)
      )
      for j in range(4)
  ]
  u32_page = jnp.concatenate(u32_parts, axis=1).reshape(page_size, 128)
  return pltpu.bitcast(u32_page, jnp.float32).astype(jnp.bfloat16)[:, :64]


def _decode_rope_tile(
    bkv_rope_i32: jax.Array, pq_slice: jax.Array, batch_idx: jax.Array
) -> jax.Array:
  """Unpacks gathered int32[bkv_tile, 128] RoPE rows using compact q_k."""
  qk_8xk = lax.bitwise_and(
      lax.shift_right_logical(pq_slice, 2 * batch_idx),
      3,
  )
  q_k = jnp.tile(qk_8xk, (16, 1)).T
  w = bkv_rope_i32
  for j in (1, 2, 3):
    w = jnp.where(q_k == j, pltpu.roll(bkv_rope_i32, 128 - 32 * j, 1), w)
  rope_u32 = _rope_words_to_f32_bits(w)
  return pltpu.bitcast(rope_u32, jnp.float32).astype(jnp.bfloat16)[:, :64]


def _compute_token_counts(
    idx_rows_ref,
    c_vmem_ref,
    *,
    b,
    c_row,
    topk: int,
    num_g: int,
) -> None:
  """Computes per-token top-k multiplicity counts with one-hot MXU matmuls.

  `count[128 * g + r] = sum_k [idx[k] >> 7 == g] * [idx[k] & 127 == r]` is an
  NT matmul of two one-hot matrices whose contraction dim `k` lies along lanes,
  so both are built from one sublane-broadcast index row without any lane
  broadcasts. Negative (invalid) indices match no group and contribute nothing.

  Args:
    idx_rows_ref: int32[batch, topk // 128, 128] logical top-k indices (VMEM).
    c_vmem_ref: float32[batch_or_1, num_g, 8, 128] count scratch (VMEM).
    b: Row of `idx_rows_ref` to read.
    c_row: Row of `c_vmem_ref` to write.
    topk: Number of top-k indices per token.
    num_g: Number of 128-position groups to count.
  """
  num_chunks = topk // 128
  step = 2 if num_chunks % 2 == 0 else 1
  k_chunk = step * 128
  num_g_mm = sc_kernel.align_to(num_g, 16)
  r_iota = lax.broadcasted_iota(jnp.int32, (128, k_chunk), 0)
  g_iota = lax.broadcasted_iota(jnp.int32, (num_g_mm, k_chunk), 0)
  counts = jnp.zeros((num_g_mm, 128), jnp.float32)
  for c in range(0, num_chunks, step):
    idx_k = jnp.concatenate(
        [idx_rows_ref[b, pl.ds(c + s, 1), :] for s in range(step)], axis=1
    )
    b_mat = (jnp.bitwise_and(idx_k, 127) == r_iota).astype(jnp.bfloat16)
    a_mat = (jnp.right_shift(idx_k, 7) == g_iota).astype(jnp.bfloat16)
    counts += jnp.einsum(
        "gk,rk->gr", a_mat, b_mat, preferred_element_type=jnp.float32
    )
  for g in range(num_g):
    c_vmem_ref[c_row, g] = jnp.broadcast_to(counts[g : g + 1, :], (8, 128))


def _load_count_tile(c_vmem_ref, b, t, num_q_heads: int) -> jax.Array:
  """Loads float32[num_q_heads, 256] multiplicity counts for 256-token tile."""
  c_8x256 = jnp.concatenate(
      [c_vmem_ref[b, 2 * t], c_vmem_ref[b, 2 * t + 1]], axis=1
  )
  return jnp.tile(c_8x256, (num_q_heads // 8, 1))


def _tc_gather_attention_kernel(
    # Scalar prefetch (SMEM)
    kv_lens_ref,  # int32[padded_tokens] or int32[1]
    # int32[3] (num_valid_tokens, active_seq_pages, active_kv_tiles)
    runtime_counts_ref,
    page_indices_ref,  # int32[total_seq_pages] or int32[1]
    seq_ids_ref,  # int32[padded_tokens] or int32[1]
    active_pages_ref,  # int32[padded_tokens] or int32[1]
    # Inputs
    attention_sinks_ref,  # float32[num_q_heads] (VMEM)
    q_ref,  # [batch_size, num_q_heads, head_dim] (VMEM)
    cache_kv_nope_hbm_ref,  # int32[total_num_pages, page_size, 128] (HBM)
    cache_kv_rope_hbm_ref,  # int32[total_num_pages, page_size // 4, 128] (HBM)
    gather_indices_smem_ref,  # int32[batch_size * topk] (SMEM) or None
    packed_q_vmem_ref,  # int32[1, 8, topk] (VMEM) or None
    idx_rows_ref,  # int32[batch_size, topk // 128, 128] (VMEM) or None
    swa_acc_ref,  # [batch_size, num_q_heads, head_dim] (VMEM)
    swa_l_ref,  # [batch_size, num_l_heads] or [padded_tokens, num_l_heads]
    swa_m_ref,  # [batch_size, num_l_heads] or [padded_tokens, num_l_heads]
    # Output
    o_ref,  # [batch_size, num_q_heads, head_dim] (VMEM)
    # Scratch
    # int32[4, 2 * bkv_tile, 128], int32[4, bkv_tile, 128], or None
    bkv_a_x4_ref,
    bkv_b_x4_ref,  # int32[4, bkv_tile, 128] or None
    vmem_cache_a_ref,  # bf16 dense KV, packed dequant KV, or quant NoPE in VMEM
    vmem_cache_b_ref,  # quant RoPE pages in VMEM, or None
    stage_nope_x2_ref,  # int32[2, page_size, 128] or None
    stage_rope_x2_ref,  # int32[2, page_size // 4, 128] or None
    c_vmem_ref,  # float32[batch_or_1, padded_num_g, 8, 128] or None
    s_scratch_ref,  # float32[batch_size, num_q_heads, 256] or None
    p_scratch_ref,  # q_dtype[batch_size, num_q_heads, 256] or None
    acc_scratch_ref,  # float32[batch_or_1, num_q_heads, head_dim]
    m_scratch_ref,  # float32[batch_or_1, num_q_heads, 128]
    l_scratch_ref,  # float32[batch_or_1, num_q_heads, 128]
    sems,  # DMA((4, 4))
    *,
    sm_scale: float,
    batch_size: int,
    topk: int,
    bkv_tile: int,
    page_size: int,
    pages_per_seq: int,
    max_num_seqs: int,
    num_g: int,
    can_flatten_kv: bool,
    kv_cache_mode: str,
    swa_lm_pipelined: bool,
):
  """TensorCore-native SparseMLA gather and FlashAttention kernel."""
  _, num_q_heads, head_dim = q_ref.shape
  q_dtype = q_ref.dtype
  head_reps = head_dim // 128

  num_valid_tokens = runtime_counts_ref[0]
  active_seq_pages = runtime_counts_ref[1]
  active_kv_tiles = runtime_counts_ref[2]
  batch_start_seq_idx = pl.program_id(0) * batch_size
  active_bs = jnp.clip(num_valid_tokens - batch_start_seq_idx, 0, batch_size)

  hbm_nope_u8 = csa_cache_layout.as_u8_slabs(cache_kv_nope_hbm_ref)
  hbm_rope_u8 = csa_cache_layout.as_u8_slabs(cache_kv_rope_hbm_ref)
  max_phys_page = hbm_nope_u8.shape[0] - 1
  f32_min = jnp.finfo(jnp.float32).min

  def _bcast_heads(x):
    return jnp.broadcast_to(x[..., None], (num_q_heads, 128))

  attention_sinks = _bcast_heads(attention_sinks_ref[...])

  def _mxu_kv(kv):
    return kv.astype(jnp.float32) if q_dtype == jnp.float32 else kv

  def _qk(q, kv):
    return (
        jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        * sm_scale
    )

  # ---- Online-softmax state (m, l, acc) in scratch slot `slot`. ----

  def _init_state(b, slot):
    """Seeds the state with the SWA partial result and the attention sink."""
    lm_row = b if swa_lm_pipelined else batch_start_seq_idx + b
    swa_l = _bcast_heads(swa_l_ref[lm_row, :num_q_heads])
    swa_m = _bcast_heads(swa_m_ref[lm_row, :num_q_heads])
    m_init = jnp.maximum(jnp.maximum(swa_m, attention_sinks), f32_min)
    exp_swa = jnp.exp(swa_m - m_init)
    m_scratch_ref[slot] = m_init
    l_scratch_ref[slot] = swa_l * exp_swa + jnp.exp(attention_sinks - m_init)
    acc_scratch_ref[slot] = swa_acc_ref[b].astype(jnp.float32) * jnp.tile(
        exp_swa, (1, head_reps)
    )

  def _softmax_update(s, slot, weight=None):
    """Updates m and l of `slot`; returns (p, alpha) to rescale acc by."""
    m_prev = m_scratch_ref[slot]
    m_tile = jnp.broadcast_to(
        jnp.max(s, axis=1, keepdims=True), (num_q_heads, 128)
    )
    m_new = jnp.maximum(m_prev, m_tile)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(s - jnp.tile(m_new, (1, s.shape[1] // 128)))
    if weight is not None:
      p = weight * p
    p_sum = jnp.broadcast_to(
        jnp.sum(p, axis=1, keepdims=True), (num_q_heads, 128)
    )
    l_scratch_ref[slot] = l_scratch_ref[slot] * alpha + p_sum
    m_scratch_ref[slot] = m_new
    return p, alpha

  def _flash_step(s, bkv, weight=None):
    """One FlashAttention-2 step over a KV tile into state slot 0."""
    p, alpha = _softmax_update(s, 0, weight)
    pv = jnp.einsum(
        "nm,md->nd",
        p.astype(bkv.dtype),
        bkv,
        preferred_element_type=jnp.float32,
    )
    acc_scratch_ref[0] = (
        acc_scratch_ref[0] * jnp.tile(alpha, (1, head_reps)) + pv
    )

  def _finalize(slot):
    acc = acc_scratch_ref[slot]
    l = jnp.tile(l_scratch_ref[slot], (1, head_reps))
    if q_dtype == jnp.float32:
      return lax.div(acc, l)
    return (acc * pltpu.reciprocal(l, approx=True)).astype(q_dtype)

  def _count_tokens(b, c_row):
    _compute_token_counts(
        idx_rows_ref, c_vmem_ref, b=b, c_row=c_row, topk=topk, num_g=num_g
    )

  # ---- Double-buffered whole-page HBM -> VMEM staging. ----

  # (HBM source, VMEM stage, semaphore row) for the NoPE and RoPE pages.
  stage_copies = ()
  if stage_nope_x2_ref is not None:
    stage_copies = (
        (hbm_nope_u8, csa_cache_layout.as_u8_slabs(stage_nope_x2_ref), 2),
        (hbm_rope_u8, csa_cache_layout.as_u8_slabs(stage_rope_x2_ref), 3),
    )

  def _start_page(lp_global, buf):
    phys_p = jnp.clip(page_indices_ref[lp_global], 0, max_phys_page)
    for src, dst, sem_row in stage_copies:
      pltpu.make_async_copy(
          src.at[phys_p], dst.at[buf], sems.at[sem_row, buf]
      ).start()

  def _wait_page(buf):
    for _, dst, sem_row in stage_copies:
      pltpu.make_async_copy(
          dst.at[buf], dst.at[buf], sems.at[sem_row, buf]
      ).wait()

  def _stream_pages(
      num_pages,
      page_base,
      body,
      *,
      start_buf=0,
      first_started=False,
      then_start=None,
  ):
    """Streams pages `page_base + [0, num_pages)`; calls body(lp, kv_bf16).

    Args:
      num_pages: Number of pages to stream.
      page_base: Index into `page_indices_ref` of the first page.
      body: Called with the local page index and the decoded bf16 page.
      start_buf: Stage buffer holding (or to hold) the first page.
      first_started: Whether the first page DMA was already started.
      then_start: Optional `(cond, lp_global)`; if `cond`, the last page
        prefetches page `lp_global` into the next stage buffer.
    """
    if not first_started:
      _start_page(page_base, start_buf)

    @pl.loop(0, num_pages)
    def _page_loop(lp):
      buf = lax.bitwise_and(start_buf + lp, 1)
      next_buf = lax.bitwise_and(start_buf + lp + 1, 1)

      @pl.when(lp + 1 < num_pages)
      def _prefetch_next_page():
        _start_page(page_base + lp + 1, next_buf)

      if then_start is not None:
        cond, next_lp_global = then_start

        @pl.when((lp + 1 == num_pages) & cond)
        def _prefetch_after_last_page():
          _start_page(next_lp_global, next_buf)

      _wait_page(buf)
      nope_bf16 = sc_kernel.dequant_dsv4_fp8(stage_nope_x2_ref.at[buf][...])
      rope_bf16 = _decode_rope_page(stage_rope_x2_ref.at[buf][...], page_size)
      kv_bf16 = jnp.concatenate([nope_bf16, rope_bf16], axis=-1)
      body(lp, _zero_non_finite(kv_bf16))

  # ---- Mode: VMEM-resident dense attention weighted by top-k counts. ----

  def _dense_vmem_body():
    def _load_dense_kv(t):
      # Re-read from VMEM at each use rather than keeping the tile live.
      return _mxu_kv(vmem_cache_a_ref[pl.ds(t * 256, 256), :])

    @pl.when(pl.program_id(0) == 0)
    def _preload_dense_pages():
      @pl.when(
          (active_kv_tiles > 0)
          & (active_seq_pages * page_size < active_kv_tiles * 256)
      )
      def _zero_partial_tail_tile():
        vmem_cache_a_ref[pl.ds((active_kv_tiles - 1) * 256, 256), :] = (
            jnp.zeros((256, head_dim), jnp.bfloat16)
        )

      @pl.when(active_seq_pages > 0)
      def _load_and_dequant_dense():
        def _store_page(lp, kv_bf16):
          vmem_cache_a_ref[pl.ds(lp * page_size, page_size), :] = kv_bf16

        _stream_pages(active_seq_pages, 0, _store_page)

    @pl.when(active_bs > 0)
    def _run_dense_batch():
      @pl.loop(0, active_bs)
      def _init_and_count_loop(b):
        _init_state(b, b)

        @pl.when(active_kv_tiles > 0)
        def _():
          _count_tokens(b, b)

      @pl.loop(active_bs, batch_size)
      def _zero_inactive_scratch(b):
        m_scratch_ref[b] = jnp.zeros((num_q_heads, 128), jnp.float32)
        l_scratch_ref[b] = jnp.ones((num_q_heads, 128), jnp.float32)
        p_scratch_ref[b] = jnp.zeros((num_q_heads, 256), p_scratch_ref.dtype)
        acc_scratch_ref[b] = jnp.zeros((num_q_heads, head_dim), jnp.float32)

      # Inactive token rows (`b >= active_bs`) only feed their own discarded
      # rows of `s` and are never written by `_write_valid_out`, so `q` needs
      # no masking: `batch_size <= 4` zeroes their counts (`p = 0, alpha = 1`),
      # and `batch_size > 4` keeps `p = 0` from `_zero_inactive_scratch` and
      # sets `alpha = 1`.
      @pl.loop(0, active_kv_tiles)
      def _dense_kv_loop(t):
        q_stacked = q_ref[...].reshape(batch_size * num_q_heads, head_dim)
        s_all = _qk(q_stacked, _load_dense_kv(t))
        s_scratch_ref[...] = s_all.reshape(batch_size, num_q_heads, 256)

        def _softmax_one(b, *, mask_inactive: bool):
          c_b = _load_count_tile(c_vmem_ref, b, t, num_q_heads)
          if mask_inactive:
            c_b = jnp.where(b < active_bs, c_b, 0.0)
          s_b = jnp.where(c_b > 0, s_scratch_ref[b], f32_min)
          p_b, alpha = _softmax_update(s_b, b, weight=c_b)
          # `s_b` is dead now; stash `alpha` in its slot so the accumulator is
          # rescaled in the same pass that adds `p @ v` below.
          s_scratch_ref[b, :, :128] = alpha
          p_scratch_ref[b] = p_b.astype(p_scratch_ref.dtype)

        if batch_size <= 4:
          for b in range(batch_size):
            _softmax_one(b, mask_inactive=True)
        else:

          @pl.loop(0, active_bs)
          def _softmax_b_loop(b):
            _softmax_one(b, mask_inactive=False)

          @pl.loop(active_bs, batch_size)
          def _fill_inactive_alpha(b):
            s_scratch_ref[b, :, :128] = jnp.ones(
                (num_q_heads, 128), jnp.float32
            )

        p_stacked = p_scratch_ref[...].reshape(batch_size * num_q_heads, 256)
        pv_all = jnp.einsum(
            "nm,md->nd",
            p_stacked,
            _load_dense_kv(t),
            preferred_element_type=jnp.float32,
        )
        alpha = jnp.tile(s_scratch_ref[:, :, :128], (1, 1, head_reps))
        acc_scratch_ref[...] = acc_scratch_ref[...] * alpha + pv_all.reshape(
            batch_size, num_q_heads, head_dim
        )

      @pl.loop(0, active_bs)
      def _write_valid_out(b):
        o_ref[b] = _finalize(b)

  # ---- Mode: per-token whole-page streaming of the token's sequence. ----

  def _seq_pages_body():
    num_seq_tokens = active_pages_ref.shape[0]

    # Pages stream through the two stage buffers in one sequence that spans
    # token boundaries: the last page of token `b` prefetches the first page of
    # token `b + 1`, so `start_buf` (carried) is the buffer of a token's page 0.
    def _seq_token_body(b, start_buf):
      tok_idx = batch_start_seq_idx + b
      seq_id = jnp.clip(seq_ids_ref[tok_idx], 0, max_num_seqs - 1)
      seq_page_base = seq_id * pages_per_seq
      act_pages = active_pages_ref[tok_idx]
      prev_pages = active_pages_ref[jnp.maximum(tok_idx - 1, 0)]
      first_page_in_flight = (b > 0) & (prev_pages > 0)
      next_tok_idx = jnp.minimum(tok_idx + 1, num_seq_tokens - 1)
      next_seq_id = jnp.clip(seq_ids_ref[next_tok_idx], 0, max_num_seqs - 1)
      prefetch_next_tok = (b + 1 < active_bs) & (
          active_pages_ref[next_tok_idx] > 0
      )

      # Start the first page DMA early so it overlaps the state init.
      @pl.when((act_pages > 0) & jnp.logical_not(first_page_in_flight))
      def _fetch_first_page():
        _start_page(seq_page_base, start_buf)

      _init_state(b, 0)

      @pl.when(act_pages > 0)
      def _run_seq_pages():
        _count_tokens(b, 0)

        def _attend_page(lp, kv_bf16):
          bkv = _mxu_kv(kv_bf16)
          c_b = _load_count_tile(c_vmem_ref, 0, lp, num_q_heads)
          s = jnp.where(c_b > 0, _qk(q_ref[b], bkv), f32_min)
          _flash_step(s, bkv, weight=c_b)

        _stream_pages(
            act_pages,
            seq_page_base,
            _attend_page,
            start_buf=start_buf,
            first_started=True,
            then_start=(prefetch_next_tok, next_seq_id * pages_per_seq),
        )

      o_ref[b] = _finalize(0)
      return lax.bitwise_and(start_buf + act_pages, 1)

    lax.fori_loop(0, active_bs, _seq_token_body, jnp.int32(0))

  # ---- Modes: 4-buffer lookahead-2 per-row gather of the top-k KV. ----

  def _gather_body():
    num_kv_tiles = topk // bkv_tile
    total_tiles = active_bs * num_kv_tiles

    if kv_cache_mode == "dequant_vmem":

      @pl.when((pl.program_id(0) == 0) & (active_seq_pages > 0))
      def _preload_dequant_pages():
        def _store_packed_page(lp, kv_bf16):
          part_a = _pack_bf16_pair(kv_bf16[:, :128], kv_bf16[:, 256:384])
          part_b = _pack_bf16_pair(kv_bf16[:, 128:256], kv_bf16[:, 384:512])
          interleaved = jnp.concatenate([part_a, part_b], axis=1).reshape(
              2 * page_size, 128
          )
          vmem_cache_a_ref[0, pl.ds(lp * 2 * page_size, 2 * page_size), :] = (
              interleaved
          )

        _stream_pages(active_seq_pages, 0, _store_packed_page)

    elif kv_cache_mode == "quant_vmem":

      @pl.when((pl.program_id(0) == 0) & (active_seq_pages > 0))
      def _preload_quant_pages():
        vmem_nope_u8 = csa_cache_layout.as_u8_slabs(vmem_cache_a_ref)
        vmem_rope_u8 = csa_cache_layout.as_u8_slabs(vmem_cache_b_ref)
        copies = (
            (hbm_nope_u8, vmem_nope_u8, 2),
            (hbm_rope_u8, vmem_rope_u8, 3),
        )

        @pl.loop(0, active_seq_pages)
        def _load_page_loop(lp):
          phys_p = jnp.clip(page_indices_ref[lp], 0, max_phys_page)
          for src, dst, sem_row in copies:
            pltpu.make_async_copy(
                src.at[phys_p], dst.at[lp], sems.at[sem_row, 0]
            ).start()

        for _, dst, sem_row in copies:
          loaded = dst.at[pl.ds(0, active_seq_pages)]
          pltpu.make_async_copy(loaded, loaded, sems.at[sem_row, 0]).wait()

    # Per-row gather sources. `_row_srcs(idx)` returns one source row view per
    # destination buffer in `dst_bufs`; `sem_rows` are their semaphore rows.
    if kv_cache_mode == "dequant_vmem":
      # Packed bf16 rows: 2 x 128 words per token.
      src_deq = vmem_cache_a_ref.bitcast(jnp.uint8).reshape(
          1, vmem_cache_a_ref.shape[1] // 2, 8, 128
      )
      dst_bufs = (bkv_a_x4_ref,)
      dst_views = (
          bkv_a_x4_ref.bitcast(jnp.uint8).reshape(4, bkv_tile, 8, 128),
      )
      sem_rows = (0,)

      def _row_srcs(idx):
        return (src_deq.at[0, idx],)

    else:
      dst_bufs = (bkv_a_x4_ref, bkv_b_x4_ref)
      dst_views = tuple(csa_cache_layout.as_u8_slabs(r) for r in dst_bufs)
      sem_rows = (0, 1)
      in_vmem = kv_cache_mode == "quant_vmem"
      src_nope = vmem_cache_a_ref if in_vmem else cache_kv_nope_hbm_ref
      src_rope = vmem_cache_b_ref if in_vmem else cache_kv_rope_hbm_ref

      if can_flatten_kv:
        nope_flat = src_nope.bitcast(jnp.uint8).reshape(
            1, src_nope.shape[0] * page_size, 4, 128
        )
        rope_flat = src_rope.bitcast(jnp.uint8).reshape(
            1, src_rope.shape[0] * (page_size // 4), 4, 128
        )

        def _row_srcs(idx):
          return (
              nope_flat.at[0, idx],
              rope_flat.at[0, lax.shift_right_logical(idx, 2)],
          )

      else:
        nope_u8 = csa_cache_layout.as_u8_slabs(src_nope)
        rope_u8 = csa_cache_layout.as_u8_slabs(src_rope)

        def _row_srcs(idx):
          p, s = _divmod_static(idx, page_size)
          return (
              nope_u8.at[p, s],
              rope_u8.at[p, lax.shift_right_logical(s, 2)],
          )

    dma_unroll = math.gcd(bkv_tile, 32)
    num_dma_chunks = bkv_tile // dma_unroll

    def _issue_bkv_chunk(step, buf_idx, chunk: int):
      """Issues the gather DMAs for keys [chunk, chunk + 1) * dma_unroll."""
      k0 = step * bkv_tile + chunk * dma_unroll
      idxs = [gather_indices_smem_ref[k0 + j] for j in range(dma_unroll)]
      for j, idx in enumerate(idxs):
        dst_k = chunk * dma_unroll + j
        for src, dst, sem_row in zip(_row_srcs(idx), dst_views, sem_rows):
          pltpu.make_async_copy(
              src, dst.at[buf_idx, dst_k], sems.at[sem_row, buf_idx]
          ).start()

    def _issue_bkv_tile(step, buf_idx):
      for chunk in range(num_dma_chunks):
        _issue_bkv_chunk(step, buf_idx, chunk)

    def _wait_bkv_tile(buf_idx):
      for buf, sem_row in zip(dst_bufs, sem_rows):
        dst = buf.at[buf_idx]
        pltpu.make_async_copy(dst, dst, sems.at[sem_row, buf_idx]).wait()

    def _decode_bkv_chunk(buf_idx, chunk: int) -> jax.Array:
      """Decodes the NoPE part (or full bf16 row) of one chunk of a KV tile."""
      if kv_cache_mode == "dequant_vmem":
        words = bkv_a_x4_ref[
            buf_idx, pl.ds(chunk * 2 * dma_unroll, 2 * dma_unroll), :
        ].reshape(dma_unroll, 256)
        return jnp.concatenate(_unpack_bf16_pair(words), axis=-1)
      return sc_kernel.dequant_dsv4_fp8(
          bkv_a_x4_ref[buf_idx, pl.ds(chunk * dma_unroll, dma_unroll), :]
      )

    @pl.when(total_tiles > 0)
    def _run_active_tiles():
      # Look-ahead DMAs past the last tile are clamped to re-fetch it into an
      # idle buffer, so the issue never sits behind a branch and can share a
      # basic block with vector work. The two extra tiles are drained at the
      # end.
      last_step = total_tiles - 1
      _issue_bkv_tile(0, 0)
      _issue_bkv_tile(jnp.minimum(1, last_step), 1)

      @pl.loop(0, total_tiles)
      def _tile_loop(step):
        buf_idx = lax.bitwise_and(step, 3)
        pf_step = jnp.minimum(step + 2, last_step)
        pf_buf_idx = lax.bitwise_and(step + 2, 3)
        b, t = _divmod_static(step, num_kv_tiles)
        _wait_bkv_tile(buf_idx)

        @pl.when(t == 0)
        def _init_token_state():
          _init_state(b, 0)

        kv_len = kv_lens_ref[batch_start_seq_idx + b]
        k_base = t * bkv_tile

        # Interleave this tile's decode with the look-ahead DMA issue so the
        # scalar descriptor pushes co-issue with vector dequant work.
        # VMEM-sourced gathers compete with the decode for VMEM, so issue those
        # up front.
        interleave_issue = kv_cache_mode != "quant_vmem"
        if not interleave_issue:
          _issue_bkv_tile(pf_step, pf_buf_idx)
        chunks = []
        for chunk in range(num_dma_chunks):
          chunks.append(_decode_bkv_chunk(buf_idx, chunk))
          if interleave_issue:
            _issue_bkv_chunk(pf_step, pf_buf_idx, chunk)
        bkv_all = jnp.concatenate(chunks, axis=0)
        if kv_cache_mode != "dequant_vmem":
          pq_slice = packed_q_vmem_ref[0, :, pl.ds(t * bkv_tile, bkv_tile)]
          bkv_rope = _decode_rope_tile(
              bkv_b_x4_ref.at[buf_idx][...], pq_slice, b
          )
          bkv_all = jnp.concatenate([bkv_all, bkv_rope], axis=-1)

        def _update_tile(masked: bool):
          bkv = bkv_all
          if masked:
            k_span_kv = k_base + lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
            bkv = jnp.where(k_span_kv < kv_len, bkv, 0)
          bkv = _mxu_kv(bkv)
          s = _qk(q_ref[b], bkv)
          if masked:
            k_span_s = k_base + lax.broadcasted_iota(jnp.int32, s.shape, 1)
            s = jnp.where(k_span_s < kv_len, s, f32_min)
          _flash_step(s, bkv)

        @pl.when(k_base + bkv_tile <= kv_len)
        def _full_tile():
          _update_tile(masked=False)

        @pl.when((k_base < kv_len) & (k_base + bkv_tile > kv_len))
        def _partial_tile():
          _update_tile(masked=True)

        @pl.when(t == num_kv_tiles - 1)
        def _finalize_token():
          o_ref[b] = _finalize(0)

      _wait_bkv_tile(lax.bitwise_and(total_tiles, 3))
      _wait_bkv_tile(lax.bitwise_and(total_tiles + 1, 3))

  if kv_cache_mode == "dense_vmem":
    _dense_vmem_body()
  elif kv_cache_mode == "seq_pages":
    _seq_pages_body()
  else:
    _gather_body()

  @pl.loop(active_bs, batch_size)
  def _copy_inactive_rows(b):
    o_ref[b] = q_ref[b]


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
        "vmem_limit_bytes",
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # int32[total_num_pages, page_size, 128]
    cache_kv_rope: jax.Array,  # int32[total_num_pages, page_size // 4, 128]
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    swa_accumution: jax.Array,  # bf16[max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # float32[max_num_tokens, num_q_heads]
    swa_m: jax.Array,  # float32[max_num_tokens, num_q_heads]
    *,
    sm_scale: float = 1.0,
    gather_and_attention_chunk_size: int = 128,
    attention_kernel_batch_size: int = 16,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
  """TensorCore MLA ragged paged attention supporting prefill and decode.

  Args:
    q: Query tensor of shape `[max_num_tokens, actual_num_q_heads, head_dim]`.
    cache_kv_nope: Compressed NoPE KV cache of shape `[total_num_pages,
      page_size, 128]` (see `csa_cache_layout`).
    cache_kv_rope: Compressed RoPE KV cache of shape `[total_num_pages,
      page_size // 4, 128]`.
    topk_indices: Top-k KV token indices per query token of shape
      `[max_num_tokens, csa_topk]`.
    page_indices: Flattened page table of shape `[max_num_seqs *
      pages_per_seq]`.
    cu_q_lens: Cumulative query lengths of shape `[max_num_seqs + 1]`.
    distribution: 3-element array `(num_decode, num_decode_and_prefill,
      num_seqs)`.
    attention_sinks: Attention sink logits of shape `[actual_num_q_heads]`.
    swa_accumution: Sliding-window attention accumulator.
    swa_l: Sliding-window attention softmax denominator `l`.
    swa_m: Sliding-window attention running row-max `m`.
    sm_scale: Softmax scale factor applied to `Q @ K^T`.
    gather_and_attention_chunk_size: Unused; accepted for API compatibility with
      the SparseCore kernel.
    attention_kernel_batch_size: Maximum query batch size per Pallas grid step.
    vmem_limit_bytes: Optional VMEM limit in bytes for the Pallas kernel. If
      `None`, defaults to 95% of the device VMEM capacity.

  Returns:
    Attention output tensor of shape
    `[max_num_tokens, actual_num_q_heads, actual_head_dim]`.
  """
  del gather_and_attention_chunk_size
  assert cache_kv_nope.dtype == jnp.int32
  assert cache_kv_rope.dtype == jnp.int32

  tpu_info = pltpu.get_tpu_info()
  _, actual_num_q_heads, actual_head_dim = q.shape

  q = sc_kernel.prepare_q_inputs(q, min_head_align=8)
  max_num_tokens, num_q_heads, head_dim = q.shape
  if num_q_heads != actual_num_q_heads:
    attention_sinks = jnp.pad(
        attention_sinks,
        (0, num_q_heads - actual_num_q_heads),
        constant_values=jnp.finfo(attention_sinks.dtype).min,
    )
  assert swa_accumution.dtype == q.dtype
  swa_accumution, swa_l, swa_m = sc_kernel.prepare_swa_inputs(
      swa_accumution, swa_l, swa_m, min_head_align=8
  )
  num_l_heads = swa_l.shape[1]

  total_num_pages, page_size, _ = cache_kv_nope.shape
  max_num_seqs = cu_q_lens.shape[0] - 1
  total_seq_pages = page_indices.shape[0]
  assert total_seq_pages % max_num_seqs == 0
  pages_per_seq = total_seq_pages // max_num_seqs
  total_seq_tokens = total_seq_pages * page_size
  topk = topk_indices.shape[-1]
  assert topk >= 128 and topk % 128 == 0

  # ---- KV execution strategy selection. ----
  num_sublanes = tpu_info.num_sublanes
  effective_vmem_limit = (
      int(tpu_info.vmem_capacity_bytes * 0.95)
      if vmem_limit_bytes is None
      else vmem_limit_bytes
  )
  q_elem_bytes = q.dtype.itemsize
  page_table_fits_smem = page_indices.nbytes <= 256 * 1024
  can_use_count_mxu = (
      num_q_heads % 8 == 0 and num_q_heads <= 256 and page_table_fits_smem
  )
  has_kv_reuse = (
      max_num_tokens * topk >= total_seq_tokens * 2
      and page_size % num_sublanes == 0
      and page_table_fits_smem
  )

  dense_bs = _pow2_floor(
      min(
          attention_kernel_batch_size,
          16,
          max_num_tokens,
          max(1, 256 // num_q_heads),
      )
  )
  padded_dense_tokens = max(256, sc_kernel.align_to(total_seq_tokens, 256))
  dense_num_g = sc_kernel.align_to(padded_dense_tokens // 128, 8)
  idx_rows_bytes_per_tok = (
      sc_kernel.align_to(topk // 128, num_sublanes) * 128 * 4
  )
  dense_total_vmem = (
      3 * (2 * dense_bs * num_q_heads * head_dim * q_elem_bytes)
      + 2 * sc_kernel.align_to(max_num_tokens, dense_bs) * num_l_heads * 4
      + 2 * dense_bs * idx_rows_bytes_per_tok
      + padded_dense_tokens * head_dim * 2
      + 4 * page_size * 128 * 4
      + dense_bs * dense_num_g * 8 * 128 * 4
      + dense_bs * num_q_heads * 256 * (4 + q_elem_bytes)
      + dense_bs * num_q_heads * (head_dim + 256) * 4
      + 4 * 1024 * 1024
  )
  can_use_dense_vmem = (
      has_kv_reuse
      and can_use_count_mxu
      and total_seq_tokens <= 4096
      and page_size % (2 * num_sublanes) == 0
      and dense_total_vmem <= effective_vmem_limit
  )

  if can_use_dense_vmem:
    batch_size = dense_bs
  else:
    batch_size = _pow2_floor(
        min(attention_kernel_batch_size, 16, max_num_tokens)
    )
    while batch_size > 1 and batch_size * topk * 8 > 256 * 1024:
      batch_size //= 2
  padded_tokens = sc_kernel.align_to(max_num_tokens, batch_size)
  num_batches = padded_tokens // batch_size
  swa_lm_pipelined = (batch_size % num_sublanes == 0) or (
      batch_size == padded_tokens
  )

  if can_use_dense_vmem:
    kv_cache_mode = "dense_vmem"
    bkv_tile = 256
  else:
    use_wide_bkv = (
        (num_q_heads <= 64 or tpu_info.generation >= 8)
        and num_q_heads <= 128
        and topk % 256 == 0
    )
    bkv_tile = min(topk, 256 if use_wide_bkv else 128)

    pipelined_io_bytes = 3 * (
        2 * batch_size * num_q_heads * head_dim * q_elem_bytes
    )
    swa_lm_bytes = (
        (2 * 2 * batch_size * num_l_heads * 4)
        if swa_lm_pipelined
        else (2 * padded_tokens * num_l_heads * 4)
    )
    packed_q_bytes = 2 * 8 * topk * 4
    scratch_fixed_bytes = (
        (4 * bkv_tile * 256 * 4)
        + (num_q_heads * (head_dim + 256) * 4)
        + 64 * 1024
    )
    cache_vmem_budget = effective_vmem_limit - (
        pipelined_io_bytes
        + swa_lm_bytes
        + packed_q_bytes
        + scratch_fixed_bytes
        + 4 * 1024 * 1024
    )

    dequant_cache_bytes = sc_kernel.align_to(total_seq_tokens, 8) * 256 * 4 + (
        4 * page_size * 128 * 4
    )
    quant_cache_bytes = total_seq_pages * (page_size + page_size // 4) * 128 * 4
    seq_pages_vmem = (
        2 * batch_size * idx_rows_bytes_per_tok
        + 4 * page_size * 128 * 4
        + sc_kernel.align_to(pages_per_seq * 2, 8) * 8 * 128 * 4
    )
    est_seq_pages = pages_per_seq - 2 if pages_per_seq > 2 else pages_per_seq
    can_use_seq_pages = (
        can_use_count_mxu
        and page_size == 256
        and pages_per_seq <= 16
        and 2 * est_seq_pages * page_size <= 3 * topk
        and seq_pages_vmem <= cache_vmem_budget
    )

    if has_kv_reuse and dequant_cache_bytes <= cache_vmem_budget:
      kv_cache_mode = "dequant_vmem"
    elif has_kv_reuse and quant_cache_bytes <= cache_vmem_budget:
      kv_cache_mode = "quant_vmem"
    elif can_use_seq_pages:
      kv_cache_mode = "seq_pages"
      bkv_tile = 256
    else:
      kv_cache_mode = "hbm"

  # ---- Mode-specific kernel inputs. ----
  q_in = _pad_rows(q, padded_tokens)
  swa_acc_in = _pad_rows(swa_accumution, padded_tokens)
  swa_l_in = _pad_rows(swa_l, padded_tokens)
  swa_m_in = _pad_rows(swa_m, padded_tokens)

  tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
  seq_ids_segment = jnp.repeat(
      jnp.arange(max_num_seqs, dtype=jnp.int32),
      tokens_per_seq,
      total_repeat_length=max_num_tokens,
  )
  num_valid_tokens = cu_q_lens[distribution[2]]
  is_padding = topk_indices == -1

  unused_smem = jnp.zeros((1,), jnp.int32)
  kv_lens_arg = seq_ids_arg = active_pages_arg = unused_smem
  page_indices_arg = page_indices
  active_seq_pages = jnp.int32(0)
  active_kv_tiles = jnp.int32(0)
  padded_num_g = 8
  gather_indices_arg = gather_indices_spec = None
  packed_q_arg = packed_q_spec = None
  idx_rows_arg = idx_rows_spec = None

  if kv_cache_mode in ("dense_vmem", "seq_pages"):
    is_invalid = is_padding | (
        jnp.arange(max_num_tokens, dtype=jnp.int32)[:, None] >= num_valid_tokens
    )
    seq_tok_idx = jnp.clip(
        topk_indices // page_size, 0, pages_per_seq - 1
    ) * page_size + (topk_indices % page_size)
    if kv_cache_mode == "dense_vmem":
      logical_idx = jnp.where(
          is_invalid,
          -1,
          seq_ids_segment[:, None] * (pages_per_seq * page_size) + seq_tok_idx,
      )
      max_dense_idx = jnp.max(logical_idx)
      active_kv_tiles = jnp.where(
          max_dense_idx >= 0,
          jnp.minimum(padded_dense_tokens // 256, (max_dense_idx // 256) + 1),
          0,
      )
      active_seq_pages = jnp.where(
          max_dense_idx >= 0,
          jnp.minimum(
              total_seq_pages, sc_kernel.cdiv(active_kv_tiles * 256, page_size)
          ),
          0,
      )
      padded_num_g = dense_num_g
    else:
      logical_idx = jnp.where(is_invalid, -1, seq_tok_idx)
      max_seq_tok = jnp.max(logical_idx, axis=-1)
      active_pages_per_tok = jnp.where(
          max_seq_tok >= 0,
          jnp.minimum(pages_per_seq, (max_seq_tok // page_size) + 1),
          0,
      )
      seq_ids_arg = _pad_rows(seq_ids_segment, padded_tokens)
      active_pages_arg = _pad_rows(active_pages_per_tok, padded_tokens)
      padded_num_g = sc_kernel.align_to(pages_per_seq * 2, 8)

    idx_rows_arg = _pad_rows(logical_idx, padded_tokens, value=-1).reshape(
        padded_tokens, topk // 128, 128
    )
    idx_rows_spec = pl.BlockSpec(
        (batch_size, topk // 128, 128),
        lambda i, *_: (i, 0, 0),
        memory_space=pltpu.VMEM,
    )
  else:
    # 104729 and 15485863 are large primes that scatter the gather indices of
    # "-1" padding slots to avoid gather memory access hotspotting.
    flat_element_index = jnp.arange(
        max_num_tokens * topk, dtype=jnp.int32
    ).reshape(max_num_tokens, topk)
    if kv_cache_mode in ("dequant_vmem", "quant_vmem"):
      active_seq_pages = jnp.clip(distribution[2], 0, max_num_seqs) * (
          pages_per_seq
      )
      active_seq_tokens = jnp.maximum(1, active_seq_pages * page_size)
      scattered_idx = (flat_element_index * 104729) % active_seq_tokens
      logical_idx = seq_ids_segment[:, None] * (
          pages_per_seq * page_size
      ) + jnp.clip(topk_indices, 0, pages_per_seq * page_size - 1)
      gather_indices_2d = jnp.where(is_padding, scattered_idx, logical_idx)
      token_offset = gather_indices_2d % page_size
    else:
      seq_page_ids = topk_indices // page_size
      token_offset = topk_indices % page_size
      page_ids = sc_kernel.gather_page_ids(
          page_indices, seq_page_ids, seq_ids_segment, max_num_seqs
      )
      scattered_page_ids = (flat_element_index * 104729) % total_num_pages
      scattered_token_offset = (flat_element_index * 15485863) % page_size
      page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
      token_offset = jnp.where(is_padding, scattered_token_offset, token_offset)
      gather_indices_2d = page_ids * page_size + token_offset
      page_indices_arg = unused_smem

    kv_lens_arg = _pad_rows(jnp.sum(topk_indices != -1, axis=-1), padded_tokens)
    gather_indices_2d = _pad_rows(gather_indices_2d, padded_tokens)

    if kv_cache_mode != "dequant_vmem":
      # Packs each token's in-row RoPE slot (token_offset % 4) as 2 bits.
      token_offset = _pad_rows(token_offset, padded_tokens)
      q_shifts = 2 * jnp.arange(batch_size, dtype=jnp.int32)[:, None]
      packed_q_1d = jnp.sum(
          (token_offset.reshape(num_batches, batch_size, topk) & 3) << q_shifts,
          axis=1,
      )
      packed_q_arg = jnp.broadcast_to(
          packed_q_1d[:, None, :], (num_batches, 8, topk)
      )
      packed_q_spec = pl.BlockSpec(
          (1, 8, topk), lambda i, *_: (i, 0, 0), memory_space=pltpu.VMEM
      )

    gather_block_len = batch_size * topk
    gather_indices_arg = gather_indices_2d.reshape(
        num_batches, 1, gather_block_len
    )
    gather_indices_spec = pl.BlockSpec(
        (None, None, gather_block_len),
        lambda i, *_: (i, 0, 0),
        memory_space=pltpu.SMEM,
    )

  if swa_lm_pipelined:
    swa_lm_spec = pl.BlockSpec(
        (batch_size, num_l_heads),
        lambda i, *_: (i, 0),
        memory_space=pltpu.VMEM,
    )
  else:
    swa_lm_spec = pl.BlockSpec(memory_space=pltpu.VMEM)

  # ---- Mode-specific scratch. ----
  bkv_a_x4 = bkv_b_x4 = None
  vmem_cache_a = vmem_cache_b = None
  stage_nope_x2 = stage_rope_x2 = None
  c_vmem = s_scratch = p_scratch = None
  state_bs = 1

  if kv_cache_mode in ("dense_vmem", "seq_pages", "dequant_vmem"):
    stage_nope_x2 = pltpu.VMEM((2, page_size, 128), jnp.int32)
    stage_rope_x2 = pltpu.VMEM((2, page_size // 4, 128), jnp.int32)
  if kv_cache_mode in ("quant_vmem", "hbm"):
    bkv_a_x4 = pltpu.VMEM((4, bkv_tile, 128), jnp.int32)
    bkv_b_x4 = pltpu.VMEM((4, bkv_tile, 128), jnp.int32)

  if kv_cache_mode == "dense_vmem":
    vmem_cache_a = pltpu.VMEM((padded_dense_tokens, head_dim), jnp.bfloat16)
    c_vmem = pltpu.VMEM((batch_size, padded_num_g, 8, 128), jnp.float32)
    s_scratch = pltpu.VMEM((batch_size, num_q_heads, 256), jnp.float32)
    p_scratch = pltpu.VMEM((batch_size, num_q_heads, 256), q.dtype)
    state_bs = batch_size
  elif kv_cache_mode == "seq_pages":
    c_vmem = pltpu.VMEM((1, padded_num_g, 8, 128), jnp.float32)
  elif kv_cache_mode == "dequant_vmem":
    padded_seq_tokens = sc_kernel.align_to(total_seq_tokens, 8)
    bkv_a_x4 = pltpu.VMEM((4, 2 * bkv_tile, 128), jnp.int32)
    vmem_cache_a = pltpu.VMEM((1, 2 * padded_seq_tokens, 128), jnp.int32)
  elif kv_cache_mode == "quant_vmem":
    vmem_cache_a = pltpu.VMEM((total_seq_pages, page_size, 128), jnp.int32)
    vmem_cache_b = pltpu.VMEM((total_seq_pages, page_size // 4, 128), jnp.int32)

  scratch_shapes = [
      bkv_a_x4,
      bkv_b_x4,
      vmem_cache_a,
      vmem_cache_b,
      stage_nope_x2,
      stage_rope_x2,
      c_vmem,
      s_scratch,
      p_scratch,
      pltpu.VMEM((state_bs, num_q_heads, head_dim), jnp.float32),
      pltpu.VMEM((state_bs, num_q_heads, 128), jnp.float32),
      pltpu.VMEM((state_bs, num_q_heads, 128), jnp.float32),
      pltpu.SemaphoreType.DMA((4, 4)),
  ]

  scalar_prefetches = (
      kv_lens_arg,
      jnp.array(
          [num_valid_tokens, active_seq_pages, active_kv_tiles], jnp.int32
      ),
      page_indices_arg,
      seq_ids_arg,
      active_pages_arg,
  )
  in_specs = [
      pl.BlockSpec(memory_space=pltpu.VMEM),  # attention_sinks
      pl.BlockSpec(
          (batch_size, num_q_heads, head_dim),
          lambda i, *_: (i, 0, 0),
          memory_space=pltpu.VMEM,
      ),  # q
      pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_nope
      pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_rope
      gather_indices_spec,
      packed_q_spec,
      idx_rows_spec,
      pl.BlockSpec(
          (batch_size, num_q_heads, head_dim),
          lambda i, *_: (i, 0, 0),
          memory_space=pltpu.VMEM,
      ),  # swa_accumution
      swa_lm_spec,  # swa_l
      swa_lm_spec,  # swa_m
  ]
  out_specs = pl.BlockSpec(
      (batch_size, num_q_heads, head_dim),
      lambda i, *_: (i, 0, 0),
      memory_space=pltpu.VMEM,
  )
  grid = (
      jnp.clip(sc_kernel.cdiv(num_valid_tokens, batch_size), 1, num_batches),
  )
  can_flatten_kv = page_size % (4 * num_sublanes) == 0
  scope_name = (
      f"SparseMLA-TCGather-{kv_cache_mode}-p_{topk}-t_{bkv_tile}"
      f"-bz_{batch_size}"
  )
  out = jax.named_scope(scope_name)(
      pl.pallas_call(
          functools.partial(
              _tc_gather_attention_kernel,
              sm_scale=sm_scale,
              batch_size=batch_size,
              topk=topk,
              bkv_tile=bkv_tile,
              page_size=page_size,
              pages_per_seq=pages_per_seq,
              max_num_seqs=max_num_seqs,
              num_g=padded_num_g,
              can_flatten_kv=can_flatten_kv,
              kv_cache_mode=kv_cache_mode,
              swa_lm_pipelined=swa_lm_pipelined,
          ),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=len(scalar_prefetches),
              in_specs=in_specs,
              out_specs=out_specs,
              grid=grid,
              scratch_shapes=scratch_shapes,
          ),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("arbitrary",),
              vmem_limit_bytes=effective_vmem_limit,
              disable_bounds_checks=True,
          ),
          out_shape=jax.ShapeDtypeStruct(shape=q_in.shape, dtype=q_in.dtype),
          input_output_aliases={
              len(scalar_prefetches) + 1: 0,  # Alias output activation with q
          },
          name=scope_name,
      )
  )(
      *scalar_prefetches,
      attention_sinks,
      q_in,
      cache_kv_nope,
      cache_kv_rope,
      gather_indices_arg,
      packed_q_arg,
      idx_rows_arg,
      swa_acc_in,
      swa_l_in,
      swa_m_in,
  )
  return sc_kernel.prepare_outputs(
      out[:max_num_tokens], actual_num_q_heads, actual_head_dim
  )
