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
"""Flash Attention compute primitives for Multi-Head Latent Attention (MLA).

Implements online softmax rescaling, QK dot products (non-positional Q_nope *
C_kv +
RoPE Q_pe * K_pe), causal & sliding window masking, and PV accumulation
(accumulating strictly against latent C_kv without RoPE key dimension).
"""

from collections.abc import Sequence
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import utils


# TODO: Generalize flash attention helpers between BatchedRPA and
# MLAv3/create a shared utility.
def flash_attention_qk_softmax(
    q: jax.Array,  # [B, bq_sz * H_q, d_q]
    k: jax.Array,  # [B, d_kv, S] (SEQ_ALONG_LANE)
    m_prev: jax.Array,  # [bq_sz * H_q, 128]
    l_prev: jax.Array,  # [bq_sz * H_q, 128]
    is_last_k: jax.Array | Sequence[jax.Array] | None = None,  # [B]
    *,
    q_abs_start: jax.Array | Sequence[jax.Array] | None = None,  # [B]
    kv_abs_start: jax.Array | Sequence[jax.Array] | None = None,  # [B]
    cfgs: configs.MlaConfigs,
    bq_start: int | jax.Array = 0,
) -> tuple[jax.Array, list[jax.Array], jax.Array, jax.Array]:
  """Computes QK matrix multiplications, masking, and online softmax step for MLA.

  MLA QK score is decomposed into:
    S = sm_scale * (Q_nope @ C_kv.T + Q_pe @ K_pe.T)
  which is equivalent to a single dot product of concatenated Q and K.

  Args:
    q: Combined query tensor [Q_nope, Q_pe] of width aligned_q_dim.
    k: Combined key/value tensor [C_kv, K_pe] of width aligned_kv_dim.
    m_prev: Previous running maximum logits for online softmax [bq_sz * H_q,
      128].
    l_prev: Previous running sum of exponentials (denominator) [bq_sz * H_q,
      128].
    is_last_k: Optional boolean flags indicating whether this is the last K
      tile.
    q_abs_start: Per lane, the absolute position on the sequence's KV timeline
      of query row 0 of this q block: `historical_kv_len + q_idx * bq_sz`, where
      `historical_kv_len = kv_len - q_len` is the cached prefix.
    kv_abs_start: Per lane, the absolute position on the same timeline of key
      column 0 of this K block: `k_idx * bkv_sz`.
    cfgs: MLA configuration parameters.
    bq_start: Row offset of this chunk within the q block (`q_idx * bq_c_sz`
      when chunking).

  Returns:
    Tuple of (p, alpha_list, m_carry, l_next).

    `p` is the softmax probability tensor and `alpha_list` the per-batch
    rescaling factors.

    `m_carry` is **not** `m_next`. It is the running max left over after the
    last lane of this block, *after* the end-of-sequence reset: when
    `is_last_k[b]` is set, lane b's carry is forced to `-inf` so the next block
    starts a fresh sequence rather than inheriting the finished one's max. The
    unreset per-lane maxima used to normalize `p` are internal and are not
    returned. `l_next` is the stacked per-lane denominator, and the caller
    keeps only its last lane (`l_next[-1]`) as the carry.
  """
  b = q.shape[0]

  num_q_heads = cfgs.aligned_num_q_heads
  n_q = q.shape[1]  # bq_sz * aligned_num_q_heads
  assert (
      n_q % num_q_heads == 0
  ), f"Q block rows {n_q} not divisible by aligned head count {num_q_heads}"

  # 1. Compute QK dot products: S = Q @ K.T

  s_dim = k.shape[-1]
  s = lax.dot(
      q,
      k,
      dimension_numbers=(([2], [1]), ([0], [0])),
      preferred_element_type=jnp.float32,
  )

  s *= cfgs.model.sm_scale

  if cfgs.serve.scale_k is not None:
    s *= cfgs.serve.scale_k
  if cfgs.serve.scale_q is not None:
    s *= cfgs.serve.scale_q

  # 2. Soft-capping (Gemma / Grok style)
  if cfgs.model.soft_cap is not None:
    s = cfgs.model.soft_cap * jnp.tanh(s / cfgs.model.soft_cap)

  # 3. Causal & Sliding-Window Masking
  assert (q_abs_start is None) == (
      kv_abs_start is None
  ), "q_abs_start and kv_abs_start must either both be provided or both be None"
  if q_abs_start is not None and kv_abs_start is not None:
    sliding_window = cfgs.model.sliding_window

    n_tokens = n_q // num_q_heads

    if n_tokens == 1:
      # One token per block, so q_iota is identically 0 and the predicate
      # collapses to `kv_iota <= -offset`: a [1, s_dim] iota broadcast over
      # rows.
      kv_iota_1d = lax.broadcasted_iota(jnp.int32, (1, s_dim), 1)
      s_masked = []
      for b_idx in range(b):
        offset = kv_abs_start[b_idx] - (bq_start + q_abs_start[b_idx])
        mask_b = kv_iota_1d <= -offset
        if sliding_window is not None:
          # -kv_iota < sliding_window + offset <=> kv_iota > -offset - sw
          mask_b = jnp.logical_and(
              mask_b, kv_iota_1d > -offset - sliding_window
          )
        s_masked.append(jnp.where(mask_b, s[b_idx], cfgs.model.mask_value))
      s = jnp.stack(s_masked, axis=0)
    else:
      q_iota = lax.broadcasted_iota(jnp.int32, (n_tokens, 1), 0)
      kv_iota = lax.broadcasted_iota(jnp.int32, (1, s_dim), 1)
      q_kv_diff = q_iota - kv_iota

      s_masked = []
      for b_idx in range(b):
        offset = kv_abs_start[b_idx] - (bq_start + q_abs_start[b_idx])
        mask_b = q_kv_diff >= offset

        if sliding_window is not None:
          mask_b = jnp.logical_and(mask_b, q_kv_diff < sliding_window + offset)

        mask_b = jnp.expand_dims(mask_b, axis=1)
        s_b = s[b_idx].reshape(n_tokens, num_q_heads, s_dim)
        s_masked.append(
            jnp.where(mask_b, s_b, cfgs.model.mask_value).reshape(n_q, s_dim)
        )
      s = jnp.stack(s_masked, axis=0)

  # 4. Online Softmax Running Statistics
  s_curr_max = jnp.max(s, axis=-1, keepdims=True)

  alpha_list = []
  m_next_list = []

  for b_idx in range(b):
    m_curr_b = s_curr_max[b_idx]
    m_next_b = jnp.maximum(m_prev, m_curr_b)
    alpha_b = jnp.exp(m_prev - m_next_b)
    alpha_list.append(alpha_b)
    m_next_list.append(m_next_b)
    if is_last_k is not None:
      m_prev = jnp.where(is_last_k[b_idx], -jnp.inf, m_next_b)
    else:
      m_prev = m_next_b

  m_next = jnp.stack(m_next_list, axis=0)

  # 5. Softmax Probabilities
  p = jnp.exp(s - utils.broadcast_minor(m_next, s.shape))
  p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=jnp.float32)

  l_next_list = []
  for b_idx in range(b):
    l_prev_b = l_prev
    l_next_b = alpha_list[b_idx] * l_prev_b + p_rowsum[b_idx]

    l_next_list.append(l_next_b)
    l_prev = l_next_b

  l_next = jnp.stack(l_next_list, axis=0)

  # `m_prev` here is the post-reset carry for the next block, not `m_next`.
  m_carry = m_prev
  return p, alpha_list, m_carry, l_next


def flash_attention_pv(
    p: jax.Array,  # [B, bq_sz * H_q, S]
    v: jax.Array,  # [B, d_nope, S]
    alpha_list: Sequence[jax.Array],  # B * [bq_sz * H_q, 128]
    o_prev: jax.Array,  # [bq_sz * H_q, d_nope]
    cfgs: configs.MlaConfigs,
) -> jax.Array:
  """Accumulates P @ V where V is strictly latent C_kv (excluding RoPE key K_pe).

  Args:
    p: Softmax probabilities tensor [B, bq_sz * H_q, S].
    v: Latent KV cache tensor (C_kv) with width d_nope (e.g. 512).
    alpha_list: Per-batch rescaling factor to update previous accumulator [bq_sz
      * H_q, 128].
    o_prev: Previous unnormalized accumulator [bq_sz * H_q, d_nope].
    cfgs: MLA configuration parameters.

  Returns:
    Updated unnormalized output accumulator [B, bq_sz * H_q, d_nope].
  """
  b = p.shape[0]

  if cfgs.p_same_dtype_as_v:
    p = p.astype(v.dtype)

  pv = lax.dot(
      p,
      v,
      dimension_numbers=(([2], [2]), ([0], [0])),
      preferred_element_type=jnp.float32,
  )
  if cfgs.serve.scale_v is not None:
    pv *= cfgs.serve.scale_v

  o_next_list = []
  for b_idx in range(b):
    alpha_b = utils.broadcast_minor(alpha_list[b_idx], o_prev.shape)
    o_next_b = alpha_b * o_prev + pv[b_idx]
    o_next_list.append(o_next_b)
    o_prev = o_next_b

  return jnp.stack(o_next_list, axis=0)


def _normalize(
    acc: jax.Array, l_val: jax.Array, *, cfgs: configs.MlaConfigs
) -> jax.Array:
  """Normalizes accumulated attention output by denominator l."""
  l_val = utils.broadcast_minor(l_val, acc.shape)
  if cfgs.serve.dtype_out == jnp.float32:
    result = lax.div(acc, l_val)
  else:
    result = acc * pl.reciprocal(l_val, approx=True)
  return result.astype(cfgs.serve.dtype_out)


def _flash_attention_decode(
    q: Any,  # Ref or Array [B, H_q, d_q]
    k: Any,  # Ref or Array [B, d_kv, S]
    m_scratch_ref: Any,  # Ref [H_q, 128]
    l_scratch_ref: Any,  # Ref [H_q, 128]
    acc_scratch_ref: Any,  # Ref [H_q, d_nope]
    o_vref: Any,  # Ref [B, 1, H_q, d_nope]
    is_last_k: jax.Array | Sequence[jax.Array],  # [B]
    *,
    q_abs_start: jax.Array | Sequence[jax.Array] | None = None,
    kv_abs_start: jax.Array | Sequence[jax.Array] | None = None,
    cfgs: configs.MlaConfigs,
) -> None:
  """Executes unchunked Flash Attention for decode (`q_split == 1`)."""
  # Because lanes are chained serially within a step and scratch refs only
  # persist the final lane's carry (`[-1]`), earlier lanes `b < B - 1` that
  # finish a sequence (`is_last_k[b] == 1`) only hold their final accumulator in
  # `batched_output[b]` right here, so all `B` lanes are normalized and stored
  # to `o_vref[...]` in one shot (`BatchingORef.copy_out` only DMAs lanes with
  # `is_last_k[b] == 1`).
  assert cfgs.q_split == 1, f"Decode requires q_split == 1, got {cfgs.q_split}"
  k_arr = k[...]
  k_nope_arr = k_arr[:, : cfgs.aligned_lkv_dim, :]

  p, alpha, m_carry, l_next = flash_attention_qk_softmax(
      q[...],
      k_arr,
      m_scratch_ref[...],
      l_scratch_ref[...],
      is_last_k=is_last_k,
      q_abs_start=q_abs_start,
      kv_abs_start=kv_abs_start,
      cfgs=cfgs,
  )
  batched_output = flash_attention_pv(
      p, k_nope_arr, alpha, acc_scratch_ref[...], cfgs=cfgs
  )
  m_scratch_ref[...] = m_carry
  l_scratch_ref[...] = l_next[-1]
  acc_scratch_ref[...] = batched_output[-1]
  o_vref[...] = _normalize(batched_output, l_next, cfgs=cfgs).reshape(
      o_vref.shape
  )


def _flash_attention_prefill(
    q: Any,  # Ref or Array [1, bq_sz * H_q, d_q]
    k: Any,  # Ref or Array [1, d_kv, S]
    m_scratch_ref: Any,  # Ref [bq_sz * H_q, 128]
    l_scratch_ref: Any,  # Ref [bq_sz * H_q, 128]
    acc_scratch_ref: Any,  # Ref [bq_sz * H_q, d_nope]
    o_vref: Any,  # Ref [1, bq_sz, H_q, d_nope]
    is_last_k: jax.Array | Sequence[jax.Array],  # [1]
    *,
    q_abs_start: jax.Array | Sequence[jax.Array] | None = None,
    kv_abs_start: jax.Array | Sequence[jax.Array] | None = None,
    cfgs: configs.MlaConfigs,
) -> None:
  """Executes software-pipelined chunked Flash Attention for prefill/mixed."""
  # Software-pipelines QK+Softmax of chunk `q_idx` with PV of chunk `q_idx - 1`
  # across `q_split` query sub-chunks of size `bq_c_sz`. Because there is only
  # one lane per step (`batch_size == 1`), each chunk writes to a disjoint row
  # slice `r_slice` of `acc_scratch_ref` and `l_scratch_ref`, so normalization
  # can be deferred to a single end-of-step `is_last_k[0]` check over the full
  # buffer.
  assert (
      cfgs.batch_size == 1
  ), f"Prefill/mixed requires batch_size == 1, got {cfgs.batch_size}"
  q_split = cfgs.q_split
  q_chunk_len = q.shape[1] // q_split
  bq_sz_chunk = cfgs.bq_c_sz

  k_arr = k[...]
  k_nope_arr = k_arr[:, : cfgs.aligned_lkv_dim, :]

  prev_pv_args = None
  for q_idx in range(q_split):
    r_slice = slice(q_idx * q_chunk_len, (q_idx + 1) * q_chunk_len)
    cur_p, cur_alpha, cur_m_carry, cur_l_next = flash_attention_qk_softmax(
        q[:, r_slice],
        k_arr,
        m_scratch_ref[r_slice],
        l_scratch_ref[r_slice],
        is_last_k=is_last_k,
        q_abs_start=q_abs_start,
        kv_abs_start=kv_abs_start,
        cfgs=cfgs,
        bq_start=q_idx * bq_sz_chunk,
    )
    m_scratch_ref[r_slice] = cur_m_carry
    l_scratch_ref[r_slice] = cur_l_next[0]

    if prev_pv_args is not None:
      prev_slice, prev_p, prev_alpha = prev_pv_args
      acc_scratch_ref[prev_slice] = flash_attention_pv(
          prev_p,
          k_nope_arr,
          prev_alpha,
          acc_scratch_ref[prev_slice],
          cfgs=cfgs,
      )[0]
    prev_pv_args = (r_slice, cur_p, cur_alpha)

  if prev_pv_args is not None:
    prev_slice, prev_p, prev_alpha = prev_pv_args
    acc_scratch_ref[prev_slice] = flash_attention_pv(
        prev_p,
        k_nope_arr,
        prev_alpha,
        acc_scratch_ref[prev_slice],
        cfgs=cfgs,
    )[0]

  @pl.when(is_last_k[0])
  def _write_out_b0():
    o_vref[...] = _normalize(
        acc_scratch_ref[...], l_scratch_ref[...], cfgs=cfgs
    ).reshape(o_vref.shape)


def _run_pv_fallback_chunk(
    q_idx: int,
    r_slice: slice,
    p: jax.Array,
    alpha: Sequence[jax.Array],
    l_next: jax.Array,
    k_nope_arr: jax.Array,
    acc_scratch_ref: Any,
    o_vref: Any,
    *,
    cfgs: configs.MlaConfigs,
) -> None:
  """Runs PV for one query chunk and stores per-lane slices to o_vref."""
  batched_output = flash_attention_pv(
      p, k_nope_arr, alpha, acc_scratch_ref[r_slice], cfgs=cfgs
  )
  acc_scratch_ref[r_slice] = batched_output[-1]
  bq_sz_chunk = cfgs.bq_c_sz
  out = _normalize(batched_output, l_next, cfgs=cfgs).reshape(
      cfgs.batch_size,
      bq_sz_chunk,
      cfgs.aligned_num_q_heads,
      cfgs.aligned_lkv_dim,
  )
  bq_slice = slice(q_idx * bq_sz_chunk, (q_idx + 1) * bq_sz_chunk)
  for b in range(cfgs.batch_size):
    o_vref[b, bq_slice, ...] = out[b]


def _flash_attention_fallback(
    q: Any,  # Ref or Array [B, bq_sz * H_q, d_q]
    k: Any,  # Ref or Array [B, d_kv, S]
    m_scratch_ref: Any,  # Ref [bq_sz * H_q, 128]
    l_scratch_ref: Any,  # Ref [bq_sz * H_q, 128]
    acc_scratch_ref: Any,  # Ref [bq_sz * H_q, d_nope]
    o_vref: Any,  # Ref [B, bq_sz, H_q, d_nope]
    is_last_k: jax.Array | Sequence[jax.Array],  # [B]
    *,
    q_abs_start: jax.Array | Sequence[jax.Array] | None = None,
    kv_abs_start: jax.Array | Sequence[jax.Array] | None = None,
    cfgs: configs.MlaConfigs,
) -> None:
  """Executes fallback chunked Flash Attention for `batch_size > 1` and `q_split > 1`."""
  # Normalizes all B lanes per chunk and stores per-lane slices to `o_vref`
  # (Mosaic requires a scalar index on major axis 0 when slicing axis 1).
  q_split = cfgs.q_split
  q_chunk_len = q.shape[1] // q_split
  bq_sz_chunk = cfgs.bq_c_sz

  k_arr = k[...]
  k_nope_arr = k_arr[:, : cfgs.aligned_lkv_dim, :]

  prev_pv_args = None
  for q_idx in range(q_split):
    r_slice = slice(q_idx * q_chunk_len, (q_idx + 1) * q_chunk_len)
    cur_p, cur_alpha, cur_m_carry, cur_l_next = flash_attention_qk_softmax(
        q[:, r_slice],
        k_arr,
        m_scratch_ref[r_slice],
        l_scratch_ref[r_slice],
        is_last_k=is_last_k,
        q_abs_start=q_abs_start,
        kv_abs_start=kv_abs_start,
        cfgs=cfgs,
        bq_start=q_idx * bq_sz_chunk,
    )
    m_scratch_ref[r_slice] = cur_m_carry
    l_scratch_ref[r_slice] = cur_l_next[-1]

    if prev_pv_args is not None:
      _run_pv_fallback_chunk(
          *prev_pv_args,
          k_nope_arr,
          acc_scratch_ref,
          o_vref,
          cfgs=cfgs,
      )
    prev_pv_args = (q_idx, r_slice, cur_p, cur_alpha, cur_l_next)

  if prev_pv_args is not None:
    _run_pv_fallback_chunk(
        *prev_pv_args,
        k_nope_arr,
        acc_scratch_ref,
        o_vref,
        cfgs=cfgs,
    )


def chunked_flash_attention(
    q: Any,  # Ref or Array [B, bq_sz * H_q, d_q]
    k: Any,  # Ref or Array [B, d_kv, S]
    m_scratch_ref: Any,  # Ref [bq_sz * H_q, 128]
    l_scratch_ref: Any,  # Ref [bq_sz * H_q, 128]
    acc_scratch_ref: Any,  # Ref [bq_sz * H_q, d_nope]
    o_vref: Any,  # Ref [B, bq_sz, H_q, d_nope]
    is_last_k: jax.Array | Sequence[jax.Array],  # [B]
    *,
    q_abs_start: jax.Array | Sequence[jax.Array] | None = None,
    kv_abs_start: jax.Array | Sequence[jax.Array] | None = None,
    cfgs: configs.MlaConfigs,
) -> None:
  """Executes Flash Attention, dispatching between decode, prefill/mixed, and fallback.

  Dispatches to `_flash_attention_decode` when in decode mode
  (`cfgs.one_new_token`), to `_flash_attention_prefill` when `cfgs.batch_size ==
  1`
  (which makes certain optimizations to lower register spills), and to
  `_flash_attention_fallback` for general `batch_size > 1` with `q_split > 1`.

  Args:
    q: Full combined query ref or array [Q_nope, Q_pe] [B, bq_sz * H_q,
      aligned_q_dim].
    k: Full combined key ref or array [C_kv, K_pe] [B, aligned_kv_dim, S].
    m_scratch_ref: Scratch ref for running max logits [bq_sz * H_q, 128].
    l_scratch_ref: Scratch ref for running sum of exponentials [bq_sz * H_q,
      128].
    acc_scratch_ref: Scratch ref for output accumulator [bq_sz * H_q, d_nope].
    o_vref: Output accumulator ref [B, bq_sz, H_q, d_nope].
    is_last_k: Boolean flags (per batch lane) indicating whether this is the
      last K tile.
    q_abs_start: Per lane, absolute KV-timeline position of query row 0 of this
      q block (`historical_kv_len + q_idx * bq_sz`). See
      `flash_attention_qk_softmax`.
    kv_abs_start: Per lane, absolute KV-timeline position of key column 0 of
      this K block (`k_idx * bkv_sz`). See `flash_attention_qk_softmax`.
    cfgs: MLA configuration parameters (containing bq_sz and bq_c_sz).
  """
  if cfgs.one_new_token:
    attention_fn = _flash_attention_decode
  elif cfgs.batch_size == 1:
    attention_fn = _flash_attention_prefill
  else:
    attention_fn = _flash_attention_fallback
  attention_fn(
      q,
      k,
      m_scratch_ref,
      l_scratch_ref,
      acc_scratch_ref,
      o_vref,
      is_last_k,
      q_abs_start=q_abs_start,
      kv_abs_start=kv_abs_start,
      cfgs=cfgs,
  )
