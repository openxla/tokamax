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


def _pad_last_dim_to_128(x: jax.Array) -> jax.Array:
  """Zero-pads the last axis of `x` up to a multiple of 128."""
  dim = x.shape[-1]
  padded_dim = align_to(dim, 128)
  if padded_dim == dim:
    return x
  return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, padded_dim - dim)])


def unsigned_cdiv(a, b):
  exponent = int(math.log2(b))
  if b == int(math.pow(2, exponent)):
    # Use bit shift instead of division for efficiency.
    return (a + b - 1) >> exponent
  return (a + b - 1) // b


@jax.jit(donate_argnames="cache_kv")
def update_kv_cache(
    new_kv: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array | int,
) -> jax.Array:
  """Writes the new tokens of the first `num_seqs` sequences into the cache.

  `cache_kv` is `[total_num_pages, kv_dim, page_size]`, with tokens along the
  last (lane) axis. `kv_dim` is the latent content (NOPE) features followed by
  the positional (ROPE) features, each padded to a multiple of 128.

  Args:
    new_kv: `[tokens, kv_dim]` new keys, already padded and concatenated to
      match the cache's `kv_dim`.
    cache_kv: The existing KV cache.
    kv_lens: `[max_num_seqs]` current length of each sequence in the cache.
    page_indices: `[max_num_seqs * pages_per_seq]` page indices mapping sequence
      tokens to cache pages.
    cu_q_lens: Cumulative lengths of queries, used to slice `new_kv`.
    num_seqs: Number of valid sequences; only the first `num_seqs` slots are
      written.

  Returns:
    The updated KV cache.
  """
  page_size = cache_kv.shape[-1]
  max_num_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_num_seqs

  def seq_body(i, cache_kv):
    q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]

    def token_body(j, cache_kv_):
      token_idx_in_seq = kv_len - q_len + j
      page_num_in_seq = token_idx_in_seq // page_size
      page_indices_start = i * pages_per_seq
      page_idx = page_indices[page_indices_start + page_num_in_seq]
      slot = token_idx_in_seq % page_size
      return cache_kv_.at[page_idx, :, slot].set(new_kv[q_start + j])

    return lax.fori_loop(0, q_len, token_body, cache_kv)

  return lax.fori_loop(0, num_seqs, seq_body, cache_kv)


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
    ql_nope: `[heads, tokens, lkv_dim]` query latent without positional encoding
      projections (head-major).
    q_pe: `[tokens, heads, r_dim]` query positional encoding projection.
    new_kv_c: `[tokens, lkv_dim]` new content-based KV embeddings to be added to
      the cache.
    new_k_pe: `[tokens, r_dim]` new positional encoding for keys to be added to
      the cache.
    cache_kv: `[pages, kv_dim, page_size]` paged KV cache, tokens along the last
      axis; `kv_dim` is `lkv_dim` and `r_dim`, each aligned to 128.
    kv_lens: `[max_num_seqs]` current length of each sequence in the cache.
    page_indices: `[max_num_seqs * pages_per_seq]` page indices mapping sequence
      tokens to cache pages.
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
      - The `[heads, tokens, align_to(lkv_dim, 128)]` attention outputs.
      - The updated `[pages, kv_dim, page_size]` KV cache.
  """
  # The attention below works token-major.
  ql_nope = jnp.swapaxes(ql_nope, 0, 1)
  q_kv_dtype = ql_nope.dtype
  if mask_value is None:
    mask_value = float(jnp.finfo(s_dtype).min)

  # The queries and the new keys must agree on the NOPE and ROPE feature dims;
  # they are contracted against each other in the logits.
  if ql_nope.shape[-1] != new_kv_c.shape[-1]:
    raise ValueError(
        f"ql_nope lkv_dim {ql_nope.shape[-1]} != new_kv_c lkv_dim"
        f" {new_kv_c.shape[-1]}"
    )
  if q_pe.shape[-1] != new_k_pe.shape[-1]:
    raise ValueError(
        f"q_pe r_dim {q_pe.shape[-1]} != new_k_pe r_dim {new_k_pe.shape[-1]}"
    )
  # Pad every feature dim to the cache's 128-aligned `lkv_dim` and `r_dim`.
  ql_nope = _pad_last_dim_to_128(ql_nope)
  q_pe = _pad_last_dim_to_128(q_pe)
  new_kv_c = _pad_last_dim_to_128(new_kv_c)
  new_k_pe = _pad_last_dim_to_128(new_k_pe)
  lkv_dim = ql_nope.shape[-1]
  r_dim = q_pe.shape[-1]

  q = jnp.concatenate([ql_nope, q_pe], axis=-1)
  new_kv = jnp.concatenate([new_kv_c, new_k_pe], axis=-1)
  if new_kv.shape[-1] != cache_kv.shape[1]:
    raise ValueError(
        f"Padded new key dim {new_kv.shape[-1]} (lkv_dim={lkv_dim},"
        f" r_dim={r_dim}) != cache_kv kv_dim {cache_kv.shape[1]}"
    )

  max_num_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_num_seqs

  # Only the first `distribution[2]` sequence slots are valid; the remaining
  # slots up to `max_num_seqs` are padding and are skipped entirely (no cache
  # update, no page gather, no attention).
  num_seqs = jnp.clip(distribution[2], 0, max_num_seqs)

  updated_cache_kv = update_kv_cache(
      new_kv, cache_kv, kv_lens, page_indices, cu_q_lens, num_seqs
  )
  kv_dim = updated_cache_kv.shape[1]

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
    # Only this sequence's pages, one key per row: `[keys, kv_dim]`. The
    # first `lkv_dim` features are also the value.
    k_i = jnp.swapaxes(updated_cache_kv[indices], 1, 2).reshape(-1, kv_dim)
    v_i = k_i[:, :lkv_dim]
    k_indices = jnp.arange(k_i.shape[0])

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

  return jnp.swapaxes(outputs[:num_tokens], 0, 1), updated_cache_kv
