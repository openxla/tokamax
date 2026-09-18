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
"""Pure JAX reference implementation for batched ragged paged attention."""

import functools
from typing import Any
import jax
from jax import lax
import jax.numpy as jnp


def _cdiv(a: int | jax.Array, b: int | jax.Array) -> int | jax.Array:
  return (a + b - 1) // b


def _align_to(val: int, align: int) -> int:
  return (val + align - 1) // align * align


def _merge_kv(k: jax.Array, v: jax.Array) -> jax.Array:
  """Concatenate and format key and value tensors."""
  max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
  actual_num_kv_heads_x2 = actual_num_kv_heads * 2
  head_dim_aligned = _align_to(actual_head_dim, 128)
  kv_concat = jnp.concatenate([k, v], axis=-1).reshape(
      max_num_tokens, actual_num_kv_heads_x2, actual_head_dim
  )
  kv_padded = jnp.pad(
      kv_concat,
      ((0, 0), (0, 0), (0, head_dim_aligned - actual_head_dim)),
      constant_values=0,
  )
  return kv_padded


@functools.partial(
    jax.jit,
    static_argnames=(
        "use_causal_mask",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "out_dtype",
        "q_scale",
        "k_scale",
        "v_scale",
        "decode_query_size",
        "skip_kv_update",
        "kv_layout",
        "cp_group_size",
        "attention_scope",
        "return_lse",
    ),
)
def batched_ragged_paged_attention_reference(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    use_causal_mask: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    out_dtype: Any = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    decode_query_size: int = 1,
    skip_kv_update: bool = True,
    kv_layout: Any = None,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: Any = None,
    return_lse: bool = False,
    **unused_kwargs: Any,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
  """Reference multi-head / grouped-query ragged paged attention in pure JAX.

  Args:
    queries: [total_q_tokens, num_q_heads, head_dim]
    keys: [total_q_tokens, num_kv_heads, head_dim]
    values: [total_q_tokens, num_kv_heads, head_dim]
    kv_cache: [total_pages, page_size, num_kv_heads * 2, head_dim_aligned]
    kv_lens: [max_num_seqs] sequence lengths for each request.
    page_indices: [max_num_seqs * pages_per_seq] page allocation mapping.
    cu_q_lens: [max_num_seqs + 1] cumulative query tokens per request.
    distribution: [3] active counts for [prefill, decode, mixed] batch splits.
    use_causal_mask: Whether to apply lower-triangular causal masking.
    sm_scale: Softmax scaling factor (typically 1 / sqrt(head_dim)).
    sliding_window: Optional local attention window size.
    soft_cap: Optional tanh logit soft-capping factor.
    out_dtype: Desired output data type.
    q_scale: Optional query quantization scale.
    k_scale: Optional key quantization scale.

  Returns:
    outputs: [total_q_tokens, num_q_heads, head_dim] attention outputs.
    updated_kv_cache: Updated KV cache buffer.
  """
  if out_dtype is None:
    out_dtype = queries.dtype

  actual_head_dim = queries.shape[-1]
  actual_num_q_heads = queries.shape[1]
  actual_num_kv_heads = keys.shape[1]
  actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

  merged_kv = _merge_kv(keys, values)
  total_pages, page_size, num_kv_heads_x2, head_dim_aligned = kv_cache.shape
  max_num_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_num_seqs
  max_kv_len = pages_per_seq * page_size
  total_q_tokens = queries.shape[0]

  if total_q_tokens == 0:
    return queries.astype(out_dtype), kv_cache

  # Step 1: Map each token t in [0, total_q_tokens) to its sequence index.
  token_indices = jnp.arange(total_q_tokens)
  seq_idx = jnp.searchsorted(cu_q_lens, token_indices, side="right") - 1
  seq_idx = jnp.clip(seq_idx, 0, max_num_seqs - 1)

  q_start = cu_q_lens[seq_idx]
  q_end = cu_q_lens[seq_idx + 1]
  q_len = q_end - q_start
  q_pos = token_indices - q_start
  token_kv_pos = kv_lens[seq_idx] - q_len + q_pos

  # Determine target page and page offset in kv_cache for each token.
  page_num = token_kv_pos // page_size
  page_idx = page_indices[seq_idx * pages_per_seq + page_num]
  page_offset = token_kv_pos % page_size

  # Mask out invalid / padding tokens beyond active sequence count.
  active_count = distribution[-1]
  valid_token = (token_indices < cu_q_lens[active_count]) & (seq_idx < active_count)

  # Append a dummy page to absorb invalid token writes safely.
  dummy_page = jnp.zeros(
      (1, page_size, num_kv_heads_x2, head_dim_aligned), dtype=kv_cache.dtype
  )
  padded_cache = jnp.concatenate([kv_cache, dummy_page], axis=0)

  safe_page_idx = jnp.where(valid_token, page_idx, total_pages)
  safe_page_offset = jnp.where(valid_token, page_offset, 0)
  padded_cache = padded_cache.at[safe_page_idx, safe_page_offset].set(merged_kv)
  updated_kv_cache = padded_cache[:total_pages]

  # Step 2: Unpage full K and V tensors for each sequence.
  page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
  seq_kv_pages = updated_kv_cache[page_table]
  flat_kv = seq_kv_pages.reshape(
      max_num_seqs, max_kv_len, actual_num_kv_heads * 2, head_dim_aligned
  )

  all_k = flat_kv[:, :, 0::2, :actual_head_dim]
  all_v = flat_kv[:, :, 1::2, :actual_head_dim]

  all_k = jnp.repeat(all_k, actual_num_q_heads_per_kv_head, axis=2)
  all_v = jnp.repeat(all_v, actual_num_q_heads_per_kv_head, axis=2)

  scaled_queries = queries / q_scale if q_scale is not None else queries
  if k_scale is not None:
    all_k = all_k / k_scale
  if v_scale is not None:
    all_v = all_v / v_scale

  # Step 3: Compute attention per query token.
  def compute_single_token(t: jax.Array) -> tuple[jax.Array, jax.Array]:
    s = seq_idx[t]
    q_t = scaled_queries[t]
    k_s = all_k[s]
    v_s = all_v[s]

    scores = jnp.einsum(
        "hd,khd->hk", q_t, k_s, preferred_element_type=jnp.float32
    ) * sm_scale

    if soft_cap is not None:
      scores = soft_cap * jnp.tanh(scores / soft_cap)

    k_positions = jnp.arange(max_kv_len)
    mask = k_positions < kv_lens[s]
    if use_causal_mask:
      mask = mask & (k_positions <= token_kv_pos[t])
    if sliding_window is not None:
      mask = mask & (k_positions > token_kv_pos[t] - sliding_window)

    scores = jnp.where(mask[None, :], scores, -1e30)
    lse = jax.nn.logsumexp(scores, axis=-1).astype(jnp.float32)
    weights = jax.nn.softmax(scores, axis=-1).astype(out_dtype)

    token_out = jnp.einsum(
        "hk,khd->hd", weights, v_s, preferred_element_type=jnp.float32
    ).astype(out_dtype)
    return token_out, lse

  token_outputs, token_lses = jax.vmap(compute_single_token)(token_indices)
  outputs = jnp.where(valid_token[:, None, None], token_outputs, jnp.zeros_like(token_outputs))

  if return_lse:
    lse = jnp.where(valid_token[:, None], token_lses, jnp.full_like(token_lses, -jnp.inf))
    return outputs, updated_kv_cache, lse

  return outputs, updated_kv_cache
