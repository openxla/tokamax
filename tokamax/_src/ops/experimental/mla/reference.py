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

# Query rows attended per step of `mla_attention`'s per-sequence loop. Bounds
# the logits to `heads x _Q_CHUNK_ROWS x max_kv_len` per step.
_Q_CHUNK_ROWS = 256


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
    distribution: jax.Array | None = None,
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
    distribution: Optional `(i, j, k)` sequence distribution; only the first `k`
      sequence slots are written. If None, all `batch_size` slots are.

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

  if distribution is None:
    num_seqs = batch_size
  else:
    num_seqs = jnp.clip(distribution[2], 0, batch_size)
  return lax.fori_loop(0, num_seqs, per_batch_loop, cache_kv)


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
    s_dtype: jnp.dtype = jnp.bfloat16,  # pyrefly: ignore[bad-function-definition]
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
    distribution: `(i, j, k)`: sequences [0, i) are decode-only, [i, j) are
      chunked-prefill-only and [j, k) are mixed; `k` is the number of valid
      sequences. Slots at or past `k` are skipped and their rows are zero.
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
      distribution,
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

  # Queries are ragged: sequence i owns rows [cu_q_lens[i], cu_q_lens[i + 1]).
  # Each sequence is processed in fixed-size chunks of rows up to its own
  # `q_len`. `q` and the output are padded by one chunk so every chunk's
  # `dynamic_slice` stays in bounds; an out-of-bounds start would be clamped,
  # silently shifting the window onto the wrong rows.
  num_tokens, num_heads, _ = q.shape
  chunk = max(1, min(_Q_CHUNK_ROWS, num_tokens))
  q = jnp.pad(q, ((0, chunk), (0, 0), (0, 0)))
  # Rows of padding tokens (past `cu_q_lens[-1]`) stay zero.
  outputs = jnp.zeros((num_tokens + chunk, num_heads, lkv_dim), q_kv_dtype)

  # Only the first `distribution[2]` sequence slots are valid; the remaining
  # slots up to `batch_size` are padding and are skipped entirely (no page
  # gather, no attention).
  num_seqs = jnp.clip(distribution[2], 0, batch_size)

  def seq_body(i, outputs):
    q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]

    indices_start = i * pages_per_seq
    # jax.jit requires static shapes, so we use dynamic slice instead of
    # slicing with static values.
    indices = jax.lax.dynamic_slice(
        page_indices, (indices_start,), (pages_per_seq,)
    )
    k_c = kv_c_cache[indices].reshape(-1, lkv_dim)
    k_pe = k_pe_cache[indices].reshape(-1, r_dim)
    k_i = jnp.concatenate([k_c, k_pe], axis=-1)
    v_i = k_c
    k_indices = jnp.arange(k_c.shape[0])

    def chunk_body(c, outputs):
      row0 = c * chunk
      q_c = jax.lax.dynamic_slice(
          q, (q_start + row0, 0, 0), (chunk, num_heads, q.shape[2])
      )

      # MQA attention:
      # q:[chunk, actual_num_q_heads, lkv_dim+r_dim]
      # k:[kv_len, lkv_dim+r_dim]
      # v:[kv_len, lkv_dim]
      # attn: [actual_num_q_heads, chunk, kv_len]
      logits = jnp.einsum(
          "qnh,kh->nqk",
          q_c.astype(q_kv_dtype),
          k_i.astype(q_kv_dtype),
          preferred_element_type=jnp.float32,
      )
      logits *= sm_scale
      if k_scale is not None:
        logits *= k_scale
      if q_scale is not None:
        logits *= q_scale

      rows = jnp.arange(chunk) + row0
      q_indices = rows + (kv_len - q_len)
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

      # out_c: [chunk, actual_num_q_heads, lkv_dim]
      out_c = jnp.einsum(
          "nqk,kl->qnl",
          probs,
          v_i.astype(cache_kv.dtype),
          preferred_element_type=jnp.float32,
      )
      if v_scale is not None:
        out_c *= v_scale

      # Only this sequence's rows; the rest of the chunk belongs to the next.
      prev = jax.lax.dynamic_slice(
          outputs, (q_start + row0, 0, 0), (chunk, num_heads, lkv_dim)
      )
      out_c = jnp.where(
          (rows < q_len)[:, None, None], out_c.astype(q_kv_dtype), prev
      )
      return jax.lax.dynamic_update_slice(
          outputs, out_c, (q_start + row0, 0, 0)
      )

    return jax.lax.fori_loop(
        0, (q_len + chunk - 1) // chunk, chunk_body, outputs
    )

  outputs = jax.lax.fori_loop(0, num_seqs, seq_body, outputs)

  return outputs[:num_tokens], updated_cache_kv
