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
"""Masked-dense MLA ragged paged attention kernel.

Streams the whole KV cache in `bkv_sz`-token blocks, as an ordinary dense paged
attention would, and applies the indexer's top-k selection as a mask over the
attention scores.

Both NoPE and RoPE are read from `uint8` caches:

    nope: uint8[total_pages, page_size,     4, 128]
          token `t` of page `p` is row `p * page_size + t`, holding 512
          contiguous float8_e4m3fn values.
    rope: uint8[total_pages, page_size // 4, 4, 128]
          the same row addressing after flattening the middle two dims,
          holding float8_e4m3fn values in the 128-byte lane.
"""

import enum
import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.masked_dense_mla import csa_mask


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def align_to(x, a):
  return cdiv(x, a) * a


def get_dtype_packing(dtype):
  return 32 // jax.dtypes.itemsize_bits(dtype)


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


class MlaCase(enum.Enum):
  """Represents the different cases for MLA.

  - DECODE: Sequences are in decode-only mode (q_len = 1).
  - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
  - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
  """

  DECODE = 0
  PREFILL = 1
  MIXED = 2

  @property
  def symbol(self):
    return {
        MlaCase.DECODE: "d",
        MlaCase.PREFILL: "p",
        MlaCase.MIXED: "m",
    }[self]


def _mla_ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bo_ids_ref,
    # Input
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    cache_kv_nope_hbm_ref,  # [total_num_pages, page_size, nope_dim]
    cache_kv_rope_hbm_ref,  # [total_num_pages, page_size_rope, rope_dim]
    mask_hbm_ref,  # [max_num_tokens, num_bkv_blocks, bkv_sz]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # Scratch
    bkv_nope_x2_ref,
    bkv_rope_x2_ref,
    bq_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    bo_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    mask_hbm_x2_ref,  # [2, bq_sz, 1, bkv_sz]
    sems,  # [5, 2]
    l_ref,  # [bq_sz * num_q_heads, 128],
    m_ref,  # [bq_sz * num_q_heads, 128],
    acc_ref,  # [bq_sz * num_q_heads, head_dim],
    *,
    static_q_len: int | None,
    sm_scale: float,
    k_scale: float,
    mask_value: float,
    bkv_p: int,
    bq_sz: int,
    analytic_mask: bool,
):
  """Pallas kernel for masked-dense MLA ragged paged attention."""
  assert q_hbm_ref.shape == o_hbm_ref.shape

  _, num_q_heads, head_dim = q_hbm_ref.shape
  total_num_pages, page_size = cache_kv_nope_hbm_ref.shape[:2]
  kv_packing = 1
  max_num_seqs = kv_lens_ref.shape[0]
  num_page_indices = page_indices_ref.shape[0]

  assert num_page_indices % max_num_seqs == 0
  pages_per_seq = num_page_indices // max_num_seqs
  q_dtype = q_hbm_ref.dtype
  q_packing = get_dtype_packing(q_dtype)
  # Validate against the KV dtype.
  assert o_hbm_ref.dtype == q_dtype
  assert cache_kv_nope_hbm_ref.shape[-1] % 128 == 0
  assert head_dim % 128 == 0
  bkv_sz_per_kv_packing = bkv_p * page_size
  bkv_sz = bkv_sz_per_kv_packing * kv_packing
  assert num_q_heads % q_packing == 0
  num_q_heads_per_q_packing = num_q_heads // q_packing

  start_seq_idx = start_end_seq_idx_ref[0]
  end_seq_idx = start_end_seq_idx_ref[1]
  seq_idx = pl.program_id(0) + start_seq_idx
  q_start = cu_q_lens_ref[seq_idx]
  q_end = cu_q_lens_ref[seq_idx + 1]
  q_len = q_end - q_start
  kv_len = kv_lens_ref[seq_idx]

  # Analytic replacement for the CSA mask (True where masked out).
  # Valid only when every token has at most `topk` causal candidates: the
  # indexer scores non-causal and past-`kv_len` positions as -inf, takes an
  # exact top-k, and maps -inf back to -1, so with `pos + 1 <= topk` its
  # selection is exactly `[0, pos]` and the CSA bitmap degenerates to the
  # causal mask. The caller guarantees that with `max_kv_len <= topk`.
  def causal_mask(bq_idx, bkv_idx):
    shape = (bq_sz, 1, bkv_sz)
    # Absolute position of the query token in row `t` of this bq block:
    # the sequence's KV minus this step's chunk, plus the row's offset.
    q_pos = (
        kv_len
        - q_len
        + bq_idx * bq_sz
        + lax.broadcasted_iota(jnp.int32, shape, 0)
    )
    k_pos = bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, shape, 2)
    return k_pos > q_pos

  def flash_attention(
      q,  # [bq_sz * num_q_heads, head_dim]
      kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_x2_ref
      *,
      bq_idx,
      bkv_idx,
      bq_mask_in,
  ):
    assert len(q.shape) == 2
    assert len(kv.shape) == 2
    assert q.shape[0] % num_q_heads == 0
    assert q.shape[1] == head_dim
    assert kv.shape == (bkv_sz, head_dim)
    head_l_ref = l_ref.at[: q.shape[0]]
    head_m_ref = m_ref.at[: q.shape[0]]
    head_acc_ref = acc_ref.at[: q.shape[0]]

    # Follow FlashAttention-2 forward pass.
    s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
    s *= sm_scale
    if k_scale != 1.0:
      s *= k_scale

    if analytic_mask:
      mask = causal_mask(bq_idx, bkv_idx)  # (bq_sz, 1, bkv_sz)
    else:
      mask = ~(bq_mask_in[:, None, :].astype(jnp.bool_))  # (bq_sz, 1, bkv_sz)
    s_3d = s.reshape(bq_sz, num_q_heads, bkv_sz)
    s_3d = jnp.where(mask, mask_value, s_3d)
    s = s_3d.reshape(s.shape)
    s_rowmax = jnp.max(s, axis=1, keepdims=True)
    m_prev = head_m_ref[...]
    m_curr = jnp.maximum(m_prev, s_rowmax)
    head_m_ref[...] = m_curr
    p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

    pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)
    # We use `k_scale` for the PV step as well since MLA uses a shared
    # compressed latent for k/v and k_scale ~= v_scale.
    # TODO: Add a separate v_scale param and apply it to the
    # PV step.
    if k_scale != 1.0:
      pv *= k_scale

    p_rowsum = jnp.sum(p, axis=1, keepdims=True)
    exp_m_diff = jnp.exp(m_prev - m_curr)
    l_prev = head_l_ref[...]
    l_curr = exp_m_diff * l_prev + p_rowsum
    head_l_ref[...] = l_curr
    o_prev = head_acc_ref[...]
    o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
    head_acc_ref[...] = o_curr

  def _async_copy(src, dst, sem, wait):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
      cp.wait()
    else:
      cp.start()

  def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
    sem_nope = sems.at[0, bkv_sem_idx]
    sem_rope = sems.at[4, bkv_sem_idx]

    # bkv_nope_x2_ref shape: [2, bkv_p * page_size, 4, 128]
    bkv_nope_vmem_ref = bkv_nope_x2_ref.at[bkv_sem_idx]
    # bkv_rope_x2_ref shape: [2, bkv_p * page_size, 128]
    bkv_rope_vmem_ref = bkv_rope_x2_ref.at[bkv_sem_idx]

    # reshaped_cache_nope_hbm_ref shape: [total_num_pages * page_size, 4, 128]
    reshaped_cache_nope_hbm_ref = cache_kv_nope_hbm_ref.reshape(
        total_num_pages * page_size,
        *cache_kv_nope_hbm_ref.shape[2:],
    )
    reshaped_cache_rope_hbm_ref = cache_kv_rope_hbm_ref.reshape(
        total_num_pages * page_size,
        *cache_kv_rope_hbm_ref.shape[2:],
    )

    kv_p_start = bkv_idx * bkv_p
    page_indices_offset = seq_idx * pages_per_seq + kv_p_start

    if not wait:
      # Fetch effective kv from kv cache. To pipeline multiple DMA calls,
      # we utilize static for loop instead of dynamic for loop.
      # Loop through all pages in a block
      for i in range(bkv_p):
        # If the page index is out of bound, we clamp page_idx to the
        # last valid page. This forces a safe static-sized DMA copy of
        # garbage data, which is safely masked out later by bq_mask_in.
        page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)

        _async_copy(
            reshaped_cache_nope_hbm_ref.at[
                pl.ds(
                    page_indices_ref[page_idx] * page_size,
                    page_size,
                ),
            ],
            bkv_nope_vmem_ref.at[pl.ds(i * page_size, page_size)],
            sem_nope,
            wait,
        )
        _async_copy(
            reshaped_cache_rope_hbm_ref.at[
                pl.ds(
                    page_indices_ref[page_idx] * page_size,
                    page_size,
                ),
            ],
            bkv_rope_vmem_ref.at[pl.ds(i * page_size, page_size)],
            sem_rope,
            wait,
        )
    else:
      # When we wait, we can use a dummy copy to wait for DMAs to complete
      # where src == dst. However, the dma size must be correct.
      dst_nope_kv = bkv_nope_vmem_ref.at[pl.ds(0, bkv_sz_per_kv_packing)]
      _async_copy(src=dst_nope_kv, dst=dst_nope_kv, sem=sem_nope, wait=True)
      dst_rope_kv = bkv_rope_vmem_ref.at[pl.ds(0, bkv_sz_per_kv_packing)]
      _async_copy(src=dst_rope_kv, dst=dst_rope_kv, sem=sem_rope, wait=True)

  def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
    sem = sems.at[1, bq_sem_idx]
    bq_vmem_ref = bq_x2_ref.at[bq_sem_idx]

    q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
    q_end = cu_q_lens_ref[seq_idx + 1]
    sz = jnp.minimum(bq_sz, q_end - q_len_start)

    _async_copy(
        q_hbm_ref.at[pl.ds(q_len_start, sz)],
        bq_vmem_ref.at[pl.ds(0, sz)],
        sem,
        wait,
    )

  def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
    sem = sems.at[2, bo_sem_idx]
    vmem_ref = bo_x2_ref.at[bo_sem_idx]
    q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
    q_end = cu_q_lens_ref[seq_idx + 1]
    sz = jnp.minimum(bq_sz, q_end - q_len_start)

    _async_copy(
        vmem_ref.at[pl.ds(0, sz)],
        o_hbm_ref.at[pl.ds(q_len_start, sz)],
        sem,
        wait,
    )

  def _fetch_mask(seq_idx, bq_idx, bkv_idx, mask_sem_idx, *, wait=False):
    if analytic_mask:
      return
    sem = sems.at[3, mask_sem_idx]
    vmem_ref = mask_hbm_x2_ref.at[mask_sem_idx]

    q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
    q_end = cu_q_lens_ref[seq_idx + 1]
    # Copy only the rows this bq block actually owns. Rows [sz, bq_sz) of
    # the VMEM buffer keep whatever the previous block left there and are
    # read anyway by `flash_attention`; that is harmless because those
    # query rows are padding whose output is discarded.
    sz = jnp.minimum(bq_sz, q_end - q_len_start)

    _async_copy(
        mask_hbm_ref.at[pl.ds(q_len_start, sz), bkv_idx, pl.ds(0, bkv_sz)],
        vmem_ref.at[pl.ds(0, sz), 0, pl.ds(0, bkv_sz)],
        sem,
        wait,
    )

  def start_fetch_bkv(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
    _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)
    _fetch_mask(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

  def wait_fetch_bkv(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
    _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)
    _fetch_mask(seq_idx, bq_idx, bkv_idx, bkv_sem_idx, wait=True)

  def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
    return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

  def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
    return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

  def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
    bo_ids_ref[bo_sem_idx] = seq_idx
    bo_ids_ref[bo_sem_idx + 2] = bo_idx
    _send_bo(seq_idx, bo_idx, bo_sem_idx)

  def wait_send_bo(bo_sem_idx):
    old_seq_idx = bo_ids_ref[bo_sem_idx]
    old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

    @pl.when(jnp.logical_and(0 <= old_seq_idx, old_seq_idx <= seq_idx))
    def _():
      _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

  def load_bq(bq_sem_idx):
    q_ref = (
        bq_x2_ref.bitcast(jnp.uint32)
        .at[bq_sem_idx]
        .reshape(bq_sz * num_q_heads_per_q_packing, head_dim)
    )
    q = pltpu.bitcast(
        q_ref[: bq_sz * num_q_heads_per_q_packing],
        q_dtype,
    ).reshape(bq_sz * num_q_heads, head_dim)
    return q

  def load_bkv(bkv_sem_idx, bkv_idx):
    bkv_nope = bkv_nope_x2_ref[bkv_sem_idx, :bkv_sz, ...]
    bkv_nope = bkv_nope.reshape(bkv_sz, 512)
    bkv_nope = pltpu.bitcast(bkv_nope, jnp.float8_e4m3fn)

    bkv_rope = bkv_rope_x2_ref[bkv_sem_idx, :bkv_sz, :]
    bkv_rope = pltpu.bitcast(bkv_rope, jnp.float8_e4m3fn)

    bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)

    if bkv.shape[-1] < head_dim:
      bkv = jnp.pad(bkv, ((0, 0), (0, head_dim - bkv.shape[-1])))

    # Multiple caches may overlay on the same KV tensor. For example,
    # compressor state caches write data in bfloat16 / float32 format, where
    # certain byte patterns are interpreted as NaN in FP8.
    # Mask out the data by the actual kv_len to avoid NaN propagating
    # to the downstream computation.
    k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
    bkv = jnp.where(k_span < kv_len, bkv, 0)
    return bkv

  def broadcast_minor(src, shape):
    if src.shape == shape:
      return src
    assert src.shape[:-1] == shape[:-1]
    assert src.shape[-1] % 128 == 0
    target_minor = align_to(shape[-1], src.shape[-1])
    # no-op concatenation.
    return jnp.concatenate(
        [src for _ in range(target_minor // src.shape[-1])], axis=-1
    )[..., : shape[-1]]

  def process():
    # Force at least one bkv block and one bq block per sequence: the
    # double-buffered DMA pipeline hands the bkv and bq semaphore across
    # sequence boundaries and assumes every sequence runs >=1 bkv and bq
    # iteration.
    num_bkv = jnp.maximum(1, cdiv(kv_len, bkv_sz))
    if static_q_len is None:
      num_bq = jnp.maximum(1, cdiv(q_len, bq_sz))
    else:
      num_bq = jnp.maximum(1, cdiv(static_q_len, bq_sz))

    def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
      next_bq_idx = bq_idx + 1
      is_last_bq = next_bq_idx == num_bq
      next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
      next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
      next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
      return next_seq_idx, next_bq_idx, next_bq_sem_idx

    def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
      next_bkv_idx = bkv_idx + 1
      is_last_bkv = next_bkv_idx == num_bkv
      next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
      next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
      is_last_bq = next_bq_idx == num_bq
      next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
      next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
      next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
      return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

    def compute_with_bq(bq_idx, _):
      bq_sem_idx = sem_ids_ref[0]
      next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
          seq_idx, bq_idx, bq_sem_idx
      )

      # Prefetch next bq
      @pl.when(next_seq_idx < end_seq_idx)
      def prefetch_next_bq():
        sem_ids_ref[0] = next_bq_sem_idx
        start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

      l_ref[...] = jnp.zeros_like(l_ref)
      m_ref[...] = jnp.full_like(m_ref, jnp.finfo(jnp.float32).min)
      acc_ref[...] = jnp.zeros_like(acc_ref)

      def compute_with_bkv(bkv_idx, _):
        # Get next bkv ids.
        bkv_sem_idx = sem_ids_ref[1]
        next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx = (
            get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)
        )

        # Prefetch next bkv
        @pl.when(next_seq_idx < end_seq_idx)
        def prefetch_next_bkv():
          sem_ids_ref[1] = next_bkv_sem_idx
          start_fetch_bkv(
              next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx
          )

        # Wait for cur bkv
        wait_fetch_bkv(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)
        bq_mask_in = (
            None
            if analytic_mask
            else mask_hbm_x2_ref[bkv_sem_idx, ...][:bq_sz, 0, :]
        )

        # `load_bkv` zeroes rows past `kv_len` so that bytes belonging
        # to another cache overlaid on the same tensor cannot decode to
        # NaN and poison the softmax; the CSA mask then removes those
        # columns from the scores.
        bkv = load_bkv(bkv_sem_idx, bkv_idx)
        bq = load_bq(bq_sem_idx)

        flash_attention(
            bq,
            bkv,
            bq_idx=bq_idx,
            bkv_idx=bkv_idx,
            bq_mask_in=bq_mask_in,
        )

      # Wait for cur bq if not ready yet
      wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

      jax.lax.fori_loop(
          0,
          num_bkv,
          compute_with_bkv,
          None,
          unroll=False,
      )

      # Load acc and calculate final output.
      acc = acc_ref[...]
      denom = broadcast_minor(l_ref[...], acc.shape)
      out = (
          lax.div(acc, denom)
          if q_dtype == jnp.float32
          else (acc * pl.reciprocal(denom, approx=True)).astype(q_dtype)
      )

      # Wait for previous bo to be fully sent before storing new bo.
      bo_sem_idx = sem_ids_ref[2]
      sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
      wait_send_bo(bo_sem_idx)

      # Store output from acc to bo.
      bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
          bq_sz * num_q_heads_per_q_packing,
          head_dim,
      )[...] = pltpu.bitcast(out, jnp.int32)

      # Send cur bo
      start_send_bo(seq_idx, bq_idx, bo_sem_idx)

    lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

  ### ------- Kernel start ------- ###

  @pl.when(seq_idx == start_seq_idx)
  def prologue():
    start_fetch_bq(start_seq_idx, 0, 0)
    start_fetch_bkv(start_seq_idx, 0, 0, 0)

  process()

  @pl.when(seq_idx == end_seq_idx - 1)
  def epilogue():
    for i in range(2):
      wait_send_bo(i)

  ### ------- Kernel end ------- ###


def prepare_q_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
  """Pads query heads and head dimension to hardware alignment."""
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


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
  """Slices padded output back to actual query heads and head dimension."""
  return out[:, :actual_num_q_heads, :actual_head_dim]


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "k_scale",
        "mask_value",
        "max_kv_len",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
)
def masked_dense_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # uint8[total_num_pages, page_size, 4, 128]
    cache_kv_rope: jax.Array,  # uint8[total_num_pages, page_size // 4, 4, 128]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    k_scale: float = 1.0,
    mask_value: float | None = None,
    # Kernel optimization params.
    max_kv_len: int | None = None,
    chunk_prefill_size: int | None = None,
    # Kernel tuning params for decode, prefill, and mixed cases.
    # If passed in as int, all cases are the same.
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    sequence_start: jax.Array | None = None,
) -> jax.Array:
  """Masked-dense MLA ragged paged attention supporting mixed batches.

  Streams the paged KV cache block by block and masks the scores with the
  indexer's selection, instead of gathering the selected rows.

  Note that the compressed kv tokens of the current forward pass have already
  been written to the caches by the time this is called, so `kv_lens` is the
  length *after* that write.

  Args:
    q: concatenated all sequences' queries, `[max_num_tokens, num_q_heads,
      head_dim]` where head_dim is nope + rope, bf16.
    cache_kv_nope: uint8 nope cache; row `p * page_size + t` holds one token's
      512 float8_e4m3fn latents.
    cache_kv_rope: uint8 rope cache, same row addressing after flattening its
      middle two dims.
    kv_lens: per-**sequence** total KV length including this step's inserted
      tokens, `i32[max_num_seqs]`. Not to be confused with the sparse kernel's
      per-**token** count of valid top-k entries.
    topk_indices: for each query token, the KV positions the indexer selected,
      `-1` padded.
    page_indices: flattened page table, `(seq_id, logical_page) -> physical
      page`.
    cu_q_lens: cumulative sum of the effective query lengths; only the first
      num_seqs+1 values are valid.
    distribution: (i, j, k): sequences[0:i] are decode-only, sequences[i:j] are
      chunked-prefill-only, sequences[j:k] are mixed. k is the total number of
      sequences. The decode segment's launch is compiled with `static_q_len=1`,
      so `sequences[0:i]` must be 1-token requests -- speculative-verify windows
      belong in the mixed segment.
    sm_scale: softmax scale applied to Q@K^T.
    k_scale: per-tensor dequantization scale for the fp8 caches, applied to both
      the QK and the PV product.
    mask_value: score written where the mask excludes a KV position.
    max_kv_len: upper bound on every sequence's `kv_len` in this call. Defaults
      to the full page-table stride. When it is at most
      `topk_indices.shape[-1]`, the indexer necessarily selected every causal
      position, so the CSA mask degenerates to the causal mask: the kernel then
      computes it from `lax.broadcasted_iota` and neither builds nor reads the
      materialized bitmap. The caller must guarantee the bound -- a longer
      sequence would silently attend to KV rows the indexer did not select.
    chunk_prefill_size: static query length for the prefill-only segment. If
      None that segment is not run, which is correct only when `distribution[0]
      == distribution[1]`.
    num_kv_pages_per_block: KV pages per streamed block, per (decode, prefill,
      mixed) case.
    num_queries_per_block: query tokens per block, per case.
    vmem_limit_bytes: the vmem limit for the pallas kernel.
    sequence_start: first sequence to attend, as a device scalar. When set,
      sequences `[0, sequence_start)` are assumed already attended by the
      caller: the decode and prefill-only launches are skipped, the mixed launch
      covers `[sequence_start, distribution[2])`, and the rows of the earlier
      sequences are returned unchanged.

  Returns:
    The output of attention, `[max_num_tokens, num_q_heads, head_dim]`.
  """
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE

  assert cache_kv_nope.dtype == jnp.uint8
  assert cache_kv_rope.dtype == jnp.uint8

  if num_kv_pages_per_block is None or num_queries_per_block is None:
    raise ValueError(
        "num_kv_pages_per_block and num_queries_per_block must be specified."
    )

  if isinstance(num_kv_pages_per_block, int):
    num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
  else:
    num_kv_pages_per_blocks = num_kv_pages_per_block

  if isinstance(num_queries_per_block, int):
    num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
  else:
    num_queries_per_blocks = num_queries_per_block

  total_pages, page_size = cache_kv_nope.shape[0], cache_kv_nope.shape[1]
  cache_kv_rope = cache_kv_rope.reshape(total_pages, page_size, 128)

  _, actual_num_q_heads, actual_head_dim = q.shape

  q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
  head_dim = q.shape[-1]

  max_num_seqs = cu_q_lens.shape[0] - 1
  # `kv_lens` is per sequence here, unlike the sparse kernel's per-token
  # `kv_lens`. Feeding that one in would silently mis-derive `pages_per_seq`.
  assert kv_lens.shape[0] == max_num_seqs, (
      f"kv_lens must be per-sequence: got {kv_lens.shape[0]} entries for "
      f"{max_num_seqs} sequences"
  )
  num_page_indices = page_indices.shape[0]
  assert num_page_indices % max_num_seqs == 0
  pages_per_seq = num_page_indices // max_num_seqs
  page_table_span = pages_per_seq * page_size
  if max_kv_len is None:
    max_kv_len = page_table_span
  assert 0 < max_kv_len <= page_table_span, (
      f"max_kv_len={max_kv_len} must be within the page-table stride "
      f"{page_table_span}"
  )

  assert (
      topk_indices is not None
  ), "topk_indices must be provided for masked dense CSA"
  analytic_mask = max_kv_len <= topk_indices.shape[-1]
  if analytic_mask:
    # Nothing reads it; a minimal operand keeps the pallas_call's operand
    # layout identical in both modes.
    mask_hbm = jnp.zeros((1, 1, 1), jnp.int32)
  else:
    mask_hbm = csa_mask.generate_mask_sc(topk_indices, max_kv_len)[:, None, :]

  _, num_q_heads, _ = q.shape

  def run_mla_kernel(
      q: jax.Array,
      cache_kv_nope: jax.Array,
      cache_kv_rope: jax.Array,
      kv_lens: jax.Array,  # i32[max_num_seqs]
      mask_hbm: jax.Array,  # i32[max_num_tokens, 1, max_kv_len]
      page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
      cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
      start_seq_idx: jax.Array,  # i32
      end_seq_idx: jax.Array,  # i32
      static_q_len: int | None,
      num_kv_pages_per_block: int,
      num_queries_per_block: int,
      case: MlaCase = MlaCase.MIXED,
  ):
    bkv_p = num_kv_pages_per_block
    bkv_sz = bkv_p * page_size
    if analytic_mask:
      mask_hbm_reshaped = mask_hbm
    else:
      assert mask_hbm.shape[-1] % bkv_sz == 0, (
          f"mask width {mask_hbm.shape[-1]} must be a multiple of the "
          f"streamed block size {bkv_sz}"
      )
      mask_hbm_reshaped = mask_hbm.reshape(
          (mask_hbm.shape[0], mask_hbm.shape[-1] // bkv_sz, bkv_sz)
      )
    if static_q_len is not None:
      bq_sz = min(num_queries_per_block, static_q_len)
    else:
      bq_sz = num_queries_per_block

    grid = (end_seq_idx - start_seq_idx,)
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),  # q
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_nope
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_rope
        pl.BlockSpec(memory_space=pltpu.HBM),  # mask_hbm
    ]
    out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # o

    bkv_nope_double_buf = pltpu.VMEM(
        (2, bkv_p * cache_kv_nope.shape[1], *cache_kv_nope.shape[2:]),
        cache_kv_nope.dtype,
    )
    bkv_rope_double_buf = pltpu.VMEM(
        (2, bkv_p * cache_kv_rope.shape[1], *cache_kv_rope.shape[2:]),
        cache_kv_rope.dtype,
    )

    bq_double_buf = pltpu.VMEM(
        (2, bq_sz, num_q_heads, head_dim),
        q.dtype,
    )

    bo_double_buf = bq_double_buf

    # In analytic mode the mask lives in registers, so the double buffer
    # shrinks to a placeholder -- at 64 heads the int32 buffer is 256 KiB
    # of VMEM, which is what caps `num_queries_per_block` at 32.
    mask_hbm_double_buf = pltpu.VMEM(
        (2, 1, 1, 128) if analytic_mask else (2, bq_sz, 1, bkv_sz),
        mask_hbm_reshaped.dtype,
    )

    l_scratch = pltpu.VMEM(
        (bq_sz * num_q_heads, 128),
        jnp.float32,
    )
    m_scratch = l_scratch

    acc_scratch = pltpu.VMEM(
        (bq_sz * num_q_heads, head_dim),
        jnp.float32,
    )

    # Semaphores for double buffering of bkv, bq, bo, mask.
    # Intermediate buffers per kv head for flash attention.
    scratch_shapes = [
        bkv_nope_double_buf,
        bkv_rope_double_buf,
        bq_double_buf,
        bo_double_buf,
        mask_hbm_double_buf,
        pltpu.SemaphoreType.DMA((5, 2)),
        l_scratch,
        m_scratch,
        acc_scratch,
    ]

    scalar_prefetches = (
        kv_lens,
        page_indices,
        cu_q_lens,
        jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
        # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
        jnp.zeros((3,), jnp.int32),
        # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx,
        #  bo_sem_1_bo_idx)
        jnp.full((4,), -1, jnp.int32),
    )

    scope_name = f"MLA-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}"
    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _mla_ragged_paged_attention_kernel,
                sm_scale=sm_scale,
                k_scale=k_scale,
                static_q_len=static_q_len,
                mask_value=mask_value,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
                analytic_mask=analytic_mask,
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
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
            input_output_aliases={
                # Alias the output activation with q. Operand indices count
                # the scalar prefetches, so q is the first one after them.
                len(scalar_prefetches): 0,
            },
            name=scope_name,
        )
    )
    return kernel(
        *scalar_prefetches,
        q,
        cache_kv_nope,
        cache_kv_rope,
        mask_hbm_reshaped,
    )

  if sequence_start is None:
    # Decode-only
    q = run_mla_kernel(
        q,
        cache_kv_nope,
        cache_kv_rope,
        kv_lens,
        mask_hbm,
        page_indices,
        cu_q_lens,
        start_seq_idx=jnp.array(0, dtype=jnp.int32),
        end_seq_idx=distribution[0],
        static_q_len=1,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        case=MlaCase.DECODE,
    )

  if sequence_start is not None:
    # The caller has already attended sequences `[0, sequence_start)`.
    # Only the mixed launch runs, starting there; earlier rows keep their
    # values because the output aliases `q`.
    mixed_start = sequence_start
  elif chunk_prefill_size is None:
    # Without a static prefill length there is no prefill-only launch, so
    # the mixed launch has to start where the decode segment ended.
    # Starting it at `distribution[1]` instead would silently skip every
    # sequence in `[distribution[0], distribution[1])` and leave those
    # tokens' outputs as whatever `q` held, since the output aliases `q`.
    mixed_start = distribution[0]
  else:
    mixed_start = distribution[1]
    # Handle prefill where the query length is fixed per sequence.
    q = run_mla_kernel(
        q,
        cache_kv_nope,
        cache_kv_rope,
        kv_lens,
        mask_hbm,
        page_indices,
        cu_q_lens,
        start_seq_idx=distribution[0],
        end_seq_idx=distribution[1],
        static_q_len=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_blocks[1],
        num_queries_per_block=num_queries_per_blocks[1],
        case=MlaCase.PREFILL,
    )

  # Handle mixed case where the query length per sequence is variable.
  q = run_mla_kernel(
      q,
      cache_kv_nope,
      cache_kv_rope,
      kv_lens,
      mask_hbm,
      page_indices,
      cu_q_lens,
      start_seq_idx=mixed_start,
      end_seq_idx=distribution[2],
      static_q_len=None,
      num_kv_pages_per_block=num_kv_pages_per_blocks[2],
      num_queries_per_block=num_queries_per_blocks[2],
      case=MlaCase.MIXED,
  )
  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
  return prepare_outputs(q, actual_num_q_heads, actual_head_dim)
