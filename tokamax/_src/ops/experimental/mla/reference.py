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
"""Base reference implementation for Multi-Head Latent Attention."""

import functools
import math

import jax
from jax import lax
import jax.numpy as jnp


def align_to(a, b):
  return ((a + b - 1) // b) * b


def unsigned_cdiv(a, b):
  exponent = int(math.log2(b))
  if b == int(math.pow(2, exponent)):
    # Use bit shift instead of division for efficiency.
    return (a + b - 1) >> exponent
  return (a + b - 1) // b


@jax.jit(donate_argnames="cache_kv")
def update_kv_cache(
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    batch_size: int,
) -> jax.Array:
  """Updates the paged KV cache with new key and value per batch.

  cache_kv Structure:
    - total_num_pages: The pool of allocated physical pages managed dynamically
      via page indices.
    - page_size // packing: The number of elements of KV_dtype held per page.
    - packing: The number of individual elements bundled into a single hardware
      word (32-bit).
    - padded_kv_dim: The total embedding features per token (latent content-NOPE
      + positional encodings-ROPE), padded to a multiple of 128 for TPU
      alignment.
  Args:
    new_kv_c: The new content-based KV embeddings.
    new_k_pe: The new positional encoding for keys.
    cache_kv: The existing KV cache.
    kv_lens: The current length of each sequence in the cache.
    page_indices: The page indices mapping sequence tokens to cache pages.
    cu_q_lens: Cumulative lengths of queries, used to slice `new_kv_c` and
      `new_k_pe`.
    batch_size: The total batch size.

  Returns:
    The updated KV cache.
  """
  actual_r_dim = new_k_pe.shape[-1]
  r_dim = align_to(actual_r_dim, 128)
  if actual_r_dim != r_dim:
    new_k_pe = jnp.pad(
        new_k_pe, ((0, 0), (0, r_dim - actual_r_dim)), constant_values=0
    )
  actual_lkv_dim = new_kv_c.shape[-1]
  lkv_dim = align_to(actual_lkv_dim, 128)
  if actual_lkv_dim != lkv_dim:
    new_kv_c = jnp.pad(
        new_kv_c, ((0, 0), (0, lkv_dim - actual_lkv_dim)), constant_values=0
    )
  _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
  page_size = page_size_per_kv_packing * kv_packing

  num_page_indices = page_indices.shape[0]
  pages_per_seq = num_page_indices // batch_size

  def per_batch_loop(i, cache_kv):
    q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]

    def per_token_loop(j, cache_kv_):
      token_idx_in_seq = kv_len - q_len + j
      page_num_in_seq = token_idx_in_seq // page_size
      page_indices_start = i * pages_per_seq
      page_idx = page_indices[page_indices_start + page_num_in_seq]
      row = (token_idx_in_seq % page_size) // kv_packing
      col = (token_idx_in_seq % page_size) % kv_packing

      cache_kv_ = cache_kv_.at[page_idx, row, col, ..., :lkv_dim].set(
          new_kv_c[q_start + j]
      )
      cache_kv_ = cache_kv_.at[page_idx, row, col, ..., lkv_dim:].set(
          new_k_pe[q_start + j]
      )
      return cache_kv_

    return lax.fori_loop(0, q_len, per_token_loop, cache_kv)

  return lax.fori_loop(0, batch_size, per_batch_loop, cache_kv)


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "q_scale",
        "k_scale",
        "v_scale",
        "s_dtype",
    ),
)
def mla_attention(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    s_dtype: jnp.dtype = jnp.bfloat16,
):
  """Performs Multi-Head Latent Attention.

  This function updates the KV cache with new key and value embeddings and then
  computes attention outputs in a paged structure as the inputs are paged.

  Args:
    ql_nope: Query latent without positional encoding projections.
    q_pe: Query positional encoding projection.
    new_kv_c: New content-based KV embeddings to be added to the cache.
    new_k_pe: New positional encoding for keys to be added to the cache.
    cache_kv: The existing paged KV cache.
    kv_lens: The current length of each sequence in the cache.
    page_indices: The page indices mapping sequence tokens to cache pages.
    cu_q_lens: Cumulative lengths of queries for each sequence in the batch.
    distribution: Distribution array, likely related to batching.
    sm_scale: Scaling factor for logits before softmax.
    sliding_window: Optional sliding window size for attention.
    soft_cap: Optional soft capping value for logits.
    mask_value: Optional value to use for masked positions in attention.
    q_scale: Optional scaling for queries.
    k_scale: Optional scaling for keys.
    v_scale: Optional scaling for values (applied to output).
    s_dtype: Data type of the Logits.

  Returns:
    A tuple containing:
      - The attention outputs.
      - The updated KV cache.
  """
  q_kv_dtype = ql_nope.dtype
  if mask_value is None:
    mask_value = float(jnp.finfo(s_dtype).min)
  batch_size = kv_lens.shape[0]
  updated_cache_kv = update_kv_cache(
      new_kv_c,
      new_k_pe,
      cache_kv,
      kv_lens,
      page_indices,
      cu_q_lens,
      batch_size,
  )
  actual_lkv_dim = ql_nope.shape[-1]
  lkv_dim = align_to(actual_lkv_dim, 128)
  if lkv_dim != actual_lkv_dim:
    ql_nope = jnp.pad(
        ql_nope,
        ((0, 0), (0, 0), (0, lkv_dim - actual_lkv_dim)),
        constant_values=0,
    )
  actual_r_dim = q_pe.shape[-1]
  r_dim = align_to(actual_r_dim, 128)
  if actual_r_dim != r_dim:
    q_pe = jnp.pad(
        q_pe, ((0, 0), (0, 0), (0, r_dim - actual_r_dim)), constant_values=0
    )

  q = jnp.concatenate([ql_nope, q_pe], axis=-1)
  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  pages_per_seq = num_page_indices // max_num_seqs

  total_num_pages, page_size_per_kv_packing, kv_packing, _ = (
      updated_cache_kv.shape
  )
  page_size = page_size_per_kv_packing * kv_packing

  kv_c_cache = updated_cache_kv[..., :lkv_dim].reshape(
      total_num_pages, page_size, lkv_dim
  )
  k_pe_cache = updated_cache_kv[..., lkv_dim:].reshape(
      total_num_pages, page_size, r_dim
  )

  def _run_per_batch(start, end):
    per_batch_outputs = []
    for i in range(start, end):
      q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
      q_len = q_end - q_start
      kv_len = kv_lens[i]
      # q_i = q[q_start:q_end]
      static_q_len = q.shape[0] // batch_size
      q_i = jax.lax.dynamic_slice(
          q, (q_start, 0, 0), (static_q_len, q.shape[1], q.shape[2])
      )

      indices_start = i * pages_per_seq
      # jax.jit requires static shapes, so we use dynamic slice instead of
      # slicing with static values.
      indices = jax.lax.dynamic_slice(
          page_indices, (indices_start,), (pages_per_seq,)
      )

      gathered_kv_c = kv_c_cache[indices]
      gathered_k_pe = k_pe_cache[indices]

      flat_kv_c = gathered_kv_c.reshape(-1, lkv_dim)
      flat_k_pe = gathered_k_pe.reshape(-1, r_dim)

      k_c = jax.lax.dynamic_slice(
          flat_kv_c, (0, 0), (flat_kv_c.shape[0], lkv_dim)
      )
      k_pe = jax.lax.dynamic_slice(
          flat_k_pe, (0, 0), (flat_k_pe.shape[0], r_dim)
      )
      k_i = jnp.concatenate([k_c, k_pe], axis=-1)
      v_i = k_c

      # MQA attention:
      # q:[q_len, actual_num_q_heads, lkv_dim+r_dim]
      # k:[kv_len, lkv_dim+r_dim]
      # v:[kv_len, lkv_dim]
      # attn: [q_len, actual_num_q_heads, kv_len]
      logits = jnp.einsum(
          "qnh,kh->nqk",
          q_i.astype(q_kv_dtype),
          k_i.astype(q_kv_dtype),
          preferred_element_type=jnp.float32,
      )
      logits *= sm_scale
      if k_scale is not None:
        logits *= k_scale
      if q_scale is not None:
        logits *= q_scale

      q_indices = jnp.arange(static_q_len) + (kv_len - q_len)
      k_indices = jnp.arange(flat_kv_c.shape[0])
      mask = jnp.logical_or(k_indices > q_indices[:, None], k_indices >= kv_len)
      if sliding_window is not None:
        mask = jnp.logical_or(
            mask,
            q_indices[:, None] - sliding_window >= k_indices[None, :],
        )
      if soft_cap is not None:
        logits = jnp.tanh(logits / soft_cap) * soft_cap
      logits = jnp.where(mask[None, :, :], mask_value, logits)
      logits = logits.astype(s_dtype)
      probs = jax.nn.softmax(logits, axis=-1).astype(v_i.dtype)

      # out_i: [total_q_len, actual_num_q_heads, lkv_dim]
      out_i = jnp.einsum(
          "nqk,kl->qnl",
          probs,
          v_i.astype(cache_kv.dtype),
          preferred_element_type=jnp.float32,
      )
      if v_scale is not None:
        out_i *= v_scale
      per_batch_outputs.append(out_i.astype(q_kv_dtype))
    return per_batch_outputs

  outputs = _run_per_batch(0, batch_size)

  return (
      jnp.concatenate(outputs, axis=0),
      updated_cache_kv,
  )
