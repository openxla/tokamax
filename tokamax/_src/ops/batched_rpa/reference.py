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

_MAX_QUERY_CHUNK_SIZE = 64


def _cdiv(
    numerator: int | jax.Array, denominator: int | jax.Array
) -> int | jax.Array:
  return (numerator + denominator - 1) // denominator


def _align_to(val: int, align: int) -> int:
  return (val + align - 1) // align * align


def _is_head_along_sublane(kv_layout: Any) -> bool:
  """Returns whether `kv_layout` selects HEAD_ALONG_SUBLANE (the default)."""
  if kv_layout is None:
    return True
  if isinstance(kv_layout, str):
    return kv_layout.lower() == "head_along_sublane"
  return False


def _unfold_kv_cache_if_needed(
    kv_cache: jax.Array,
    num_kv_heads: int,
    head_dim: int,
    kv_layout: Any,
) -> tuple[jax.Array, int]:
  """Unfolds lane-folded HEAD_ALONG_SUBLANE KV cache to `[..., head_dim_aligned]`."""
  if (
      not _is_head_along_sublane(kv_layout)
      or kv_cache.dtype == jnp.uint8
      or jax.dtypes.itemsize_bits(kv_cache.dtype) > 8
  ):
    return kv_cache, 1
  folded_head_dim = kv_cache.shape[-1]
  if folded_head_dim >= head_dim or head_dim % folded_head_dim != 0:
    return kv_cache, 1
  fold = head_dim // folded_head_dim
  num_pages, page_size, _, kv_packing, _ = kv_cache.shape
  num_kv_heads_x2 = num_kv_heads * 2
  kv = kv_cache.reshape(num_pages, page_size, -1, folded_head_dim)
  kv = kv[:, :, : num_kv_heads_x2 * fold, :folded_head_dim]
  kv = (
      kv.reshape(num_pages, page_size, fold, num_kv_heads_x2, folded_head_dim)
      .transpose(0, 1, 3, 2, 4)
      .reshape(num_pages, page_size, num_kv_heads_x2, head_dim)
  )
  num_kv_heads_x2_aligned = _align_to(num_kv_heads_x2, kv_packing)
  head_dim_aligned = _align_to(head_dim, 128)
  kv = jnp.pad(
      kv,
      (
          (0, 0),
          (0, 0),
          (0, num_kv_heads_x2_aligned - num_kv_heads_x2),
          (0, head_dim_aligned - head_dim),
      ),
      constant_values=0,
  )
  unfolded = kv.reshape(
      num_pages,
      page_size,
      num_kv_heads_x2_aligned // kv_packing,
      kv_packing,
      head_dim_aligned,
  )
  return unfolded, fold


def _refold_kv_cache_if_needed(
    kv_cache: jax.Array,
    orig_shape: tuple[int, ...],
    num_kv_heads: int,
    head_dim: int,
    fold: int,
) -> jax.Array:
  """Re-folds an unfolded HEAD_ALONG_SUBLANE KV cache back to `orig_shape`."""
  if fold <= 1:
    return kv_cache
  num_pages, page_size, orig_stride, kv_packing, folded_head_dim = orig_shape
  sub_head_dim = head_dim // fold
  num_kv_heads_x2 = num_kv_heads * 2
  kv = kv_cache.reshape(num_pages, page_size, -1, kv_cache.shape[-1])
  kv = kv[:, :, :num_kv_heads_x2, :head_dim]
  kv = (
      kv.reshape(num_pages, page_size, num_kv_heads_x2, fold, sub_head_dim)
      .transpose(0, 1, 3, 2, 4)
      .reshape(num_pages, page_size, num_kv_heads_x2 * fold, sub_head_dim)
  )
  kv = jnp.pad(
      kv,
      (
          (0, 0),
          (0, 0),
          (0, orig_stride * kv_packing - num_kv_heads_x2 * fold),
          (0, folded_head_dim - sub_head_dim),
      ),
      constant_values=0,
  )
  return kv.reshape(orig_shape)


def _merge_kv(
    k: jax.Array, val_tensor: jax.Array, kv_packing: int = 1
) -> jax.Array:
  """Concatenate and format key and value tensors into 5D packed VREG slice."""
  max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
  actual_num_kv_heads_x2 = actual_num_kv_heads * 2
  head_dim_aligned = _align_to(actual_head_dim, 128)
  num_kv_heads_x2_aligned = _align_to(actual_num_kv_heads_x2, kv_packing)
  kv_concat = jnp.concatenate([k, val_tensor], axis=-1).reshape(
      max_num_tokens, actual_num_kv_heads_x2, actual_head_dim
  )
  kv_padded = jnp.pad(
      kv_concat,
      (
          (0, 0),
          (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
          (0, head_dim_aligned - actual_head_dim),
      ),
      constant_values=0,
  ).reshape(
      max_num_tokens,
      num_kv_heads_x2_aligned // kv_packing,
      kv_packing,
      head_dim_aligned,
  )
  return kv_padded


def _quantize_queries(
    queries: jax.Array, q_scale: float | None, kv_dtype: Any
) -> jax.Array:
  """Quantizes queries into `kv_dtype` when `q_scale` is provided."""
  if q_scale is None:
    return queries
  scaled = queries / q_scale
  if jnp.issubdtype(kv_dtype, jnp.floating):
    dtype_info = jnp.finfo(kv_dtype)
    scaled = jnp.clip(
        scaled, min=float(dtype_info.min), max=float(dtype_info.max)
    )
  return scaled.astype(kv_dtype)


def _update_kv_cache(
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
) -> jax.Array:
  """Writes new keys and values for active tokens into the paged KV cache."""
  (
      total_pages,
      page_size,
      num_kv_heads_x2_per_kv_packing,
      kv_packing,
      head_dim_aligned,
  ) = kv_cache.shape
  merged_kv = _merge_kv(keys, values, kv_packing=kv_packing)
  max_num_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_num_seqs
  total_q_tokens = keys.shape[0]

  token_indices = jnp.arange(total_q_tokens)
  seq_idx = jnp.searchsorted(cu_q_lens, token_indices, side="right") - 1
  seq_idx = jnp.clip(seq_idx, 0, max_num_seqs - 1)

  q_start = cu_q_lens[seq_idx]
  q_end = cu_q_lens[seq_idx + 1]
  q_len = q_end - q_start
  q_pos = token_indices - q_start
  token_kv_pos = kv_lens[seq_idx] - q_len + q_pos

  page_num = token_kv_pos // page_size
  page_idx = page_indices[seq_idx * pages_per_seq + page_num]
  page_offset = token_kv_pos % page_size

  active_count = distribution[-1]
  valid_token = (token_indices < cu_q_lens[active_count]) & (
      seq_idx < active_count
  )

  # Allocate ceil(total_q_tokens / page_size) dummy pages so every invalid
  # token scatters to a unique (page_idx, offset) slot under deterministic GPU
  # execution while keeping dummy storage O(total_q_tokens).
  num_dummy_pages = _cdiv(total_q_tokens, page_size)
  dummy_pages = jnp.zeros(
      (
          num_dummy_pages,
          page_size,
          num_kv_heads_x2_per_kv_packing,
          kv_packing,
          head_dim_aligned,
      ),
      dtype=kv_cache.dtype,
  )
  padded_cache = jnp.concatenate([kv_cache, dummy_pages], axis=0)
  safe_page_idx = jnp.where(
      valid_token, page_idx, total_pages + token_indices // page_size
  )
  safe_page_offset = jnp.where(
      valid_token, page_offset, token_indices % page_size
  )
  padded_cache = padded_cache.at[safe_page_idx, safe_page_offset].set(merged_kv)
  return padded_cache[:total_pages]


def _unpage_sequence_kv(
    updated_kv_cache: jax.Array,
    seq_page_indices: jax.Array,
    actual_num_kv_heads: int,
    actual_head_dim: int,
) -> tuple[jax.Array, jax.Array]:
  """Unpages and formats K and V tensors for a single sequence."""
  (
      _,
      page_size,
      num_kv_heads_x2_per_kv_packing,
      kv_packing,
      head_dim_aligned,
  ) = updated_kv_cache.shape
  pages_per_seq = seq_page_indices.shape[0]
  max_kv_len = pages_per_seq * page_size
  num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing

  seq_kv_pages = updated_kv_cache[seq_page_indices]
  flat_kv = seq_kv_pages.reshape(max_kv_len, num_kv_heads_x2, head_dim_aligned)[
      :, : actual_num_kv_heads * 2
  ]
  seq_k = flat_kv[:, 0::2, :actual_head_dim].astype(jnp.float32)
  seq_v = flat_kv[:, 1::2, :actual_head_dim].astype(jnp.float32)
  return seq_k, seq_v


def _softmax_and_lse(
    scores: jax.Array,
    attention_sink: jax.Array | None,
    out_dtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes unnormalized softmax probabilities, denominator, and LSE."""
  if attention_sink is not None:
    sink = attention_sink.astype(scores.dtype)[:, None]
    scores_with_sink = jnp.concatenate([sink, scores], axis=-1)
    max_score = jnp.max(scores_with_sink, axis=-1, keepdims=True)
    probs = jnp.exp(scores - max_score)
    sum_probs = jnp.sum(probs, axis=-1, keepdims=True) + jnp.exp(
        sink - max_score
    )
  else:
    max_score = jnp.max(scores, axis=-1, keepdims=True)
    probs = jnp.exp(scores - max_score)
    sum_probs = jnp.sum(probs, axis=-1, keepdims=True)
  lse = jnp.squeeze(max_score + jnp.log(sum_probs), axis=-1).astype(out_dtype)
  return probs, sum_probs, lse


def _attend_single_token(
    q_token: jax.Array,
    token_kv_pos: jax.Array,
    seq_k: jax.Array,
    seq_v: jax.Array,
    kv_len_s: jax.Array,
    *,
    sm_scale: float,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    use_causal_mask: bool,
    sliding_window: int | None,
    soft_cap: float | None,
    mask_value: float,
    attention_sink: jax.Array | None,
    quantize_p: bool,
    kv_cache_dtype: Any,
    out_dtype: Any,
) -> tuple[jax.Array, jax.Array]:
  """Computes attention output and LSE for one query token."""
  actual_num_q_heads, actual_head_dim = q_token.shape
  max_kv_len, actual_num_kv_heads, _ = seq_k.shape
  q_heads_per_kv = actual_num_q_heads // actual_num_kv_heads
  q_grouped = q_token.astype(jnp.float32).reshape(
      actual_num_kv_heads, q_heads_per_kv, actual_head_dim
  )
  scores = (
      jnp.einsum(
          "gqd,kgd->gqk", q_grouped, seq_k, preferred_element_type=jnp.float32
      ).reshape(actual_num_q_heads, max_kv_len)
      * sm_scale
  )
  if q_scale is not None:
    scores = scores * q_scale
  if k_scale is not None:
    scores = scores * k_scale

  if soft_cap is not None:
    scores = soft_cap * jnp.tanh(scores / soft_cap)

  k_positions = jnp.arange(max_kv_len)
  mask = k_positions < kv_len_s
  if use_causal_mask:
    mask = mask & (k_positions <= token_kv_pos)
  if sliding_window is not None:
    mask = mask & (k_positions > token_kv_pos - sliding_window)

  scores = jnp.where(mask[None, :], scores, mask_value)
  probs, sum_probs, lse = _softmax_and_lse(scores, attention_sink, out_dtype)

  if quantize_p:
    probs = probs.astype(kv_cache_dtype).astype(jnp.float32)

  probs_grouped = probs.reshape(actual_num_kv_heads, q_heads_per_kv, max_kv_len)
  token_out = (
      jnp.einsum(
          "gqk,kgd->gqd",
          probs_grouped,
          seq_v,
          preferred_element_type=jnp.float32,
      ).reshape(actual_num_q_heads, actual_head_dim)
      / sum_probs
  )
  if v_scale is not None:
    token_out = token_out * v_scale
  return token_out.astype(out_dtype), lse


def _attend_query_chunk(
    chunk_idx: jax.Array,
    carry: tuple[jax.Array, jax.Array],
    *,
    padded_queries: jax.Array,
    seq_k: jax.Array,
    seq_v: jax.Array,
    q_start: jax.Array,
    q_len: jax.Array,
    kv_len_s: jax.Array,
    active_q_limit: jax.Array,
    total_q_tokens: int,
    chunk_size: int,
    sm_scale: float,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    use_causal_mask: bool,
    sliding_window: int | None,
    soft_cap: float | None,
    mask_value: float,
    attention_sink: jax.Array | None,
    quantize_p: bool,
    kv_cache_dtype: Any,
    out_dtype: Any,
) -> tuple[jax.Array, jax.Array]:
  """Computes attention for one fixed-size query chunk of a sequence."""
  padded_outputs, padded_lses = carry
  chunk_q_pos = chunk_idx * chunk_size + jnp.arange(chunk_size)
  token_idx = q_start + chunk_q_pos
  chunk_start = jnp.clip(q_start + chunk_idx * chunk_size, 0, total_q_tokens)
  q_chunk = lax.dynamic_slice_in_dim(
      padded_queries, chunk_start, chunk_size, axis=0
  )
  token_kv_pos = kv_len_s - q_len + chunk_q_pos
  attend_fn = functools.partial(
      _attend_single_token,
      sm_scale=sm_scale,
      q_scale=q_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      use_causal_mask=use_causal_mask,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      attention_sink=attention_sink,
      quantize_p=quantize_p,
      kv_cache_dtype=kv_cache_dtype,
      out_dtype=out_dtype,
  )
  chunk_out, chunk_lse = jax.vmap(attend_fn, in_axes=(0, 0, None, None, None))(
      q_chunk, token_kv_pos, seq_k, seq_v, kv_len_s
  )

  valid_in_seq = (
      (chunk_q_pos < q_len) & (token_idx >= 0) & (token_idx < active_q_limit)
  )
  cur_out = lax.dynamic_slice_in_dim(
      padded_outputs, chunk_start, chunk_size, axis=0
  )
  upd_out = jnp.where(valid_in_seq[:, None, None], chunk_out, cur_out)
  padded_outputs = lax.dynamic_update_slice_in_dim(
      padded_outputs, upd_out, chunk_start, axis=0
  )
  cur_lse = lax.dynamic_slice_in_dim(
      padded_lses, chunk_start, chunk_size, axis=0
  )
  upd_lse = jnp.where(valid_in_seq[:, None], chunk_lse, cur_lse)
  padded_lses = lax.dynamic_update_slice_in_dim(
      padded_lses, upd_lse, chunk_start, axis=0
  )
  return padded_outputs, padded_lses


def _attend_sequence(
    seq_idx: jax.Array,
    carry: tuple[jax.Array, jax.Array],
    *,
    padded_queries: jax.Array,
    updated_kv_cache: jax.Array,
    page_table: jax.Array,
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    active_q_limit: jax.Array,
    total_q_tokens: int,
    chunk_size: int,
    actual_num_kv_heads: int,
    actual_head_dim: int,
    sm_scale: float,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    use_causal_mask: bool,
    sliding_window: int | None,
    soft_cap: float | None,
    mask_value: float,
    attention_sink: jax.Array | None,
    quantize_p: bool,
    kv_cache_dtype: Any,
    out_dtype: Any,
) -> tuple[jax.Array, jax.Array]:
  """Unpages KV once for `seq_idx` and processes its query tokens in chunks."""
  seq_k, seq_v = _unpage_sequence_kv(
      updated_kv_cache=updated_kv_cache,
      seq_page_indices=page_table[seq_idx],
      actual_num_kv_heads=actual_num_kv_heads,
      actual_head_dim=actual_head_dim,
  )
  q_start = cu_q_lens[seq_idx]
  q_end = cu_q_lens[seq_idx + 1]
  q_len = q_end - q_start
  kv_len_s = kv_lens[seq_idx]
  effective_q_len = jnp.maximum(0, jnp.minimum(q_end, active_q_limit) - q_start)
  num_chunks = _cdiv(effective_q_len, chunk_size)

  chunk_body = functools.partial(
      _attend_query_chunk,
      padded_queries=padded_queries,
      seq_k=seq_k,
      seq_v=seq_v,
      q_start=q_start,
      q_len=q_len,
      kv_len_s=kv_len_s,
      active_q_limit=active_q_limit,
      total_q_tokens=total_q_tokens,
      chunk_size=chunk_size,
      sm_scale=sm_scale,
      q_scale=q_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      use_causal_mask=use_causal_mask,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      attention_sink=attention_sink,
      quantize_p=quantize_p,
      kv_cache_dtype=kv_cache_dtype,
      out_dtype=out_dtype,
  )
  return lax.fori_loop(0, num_chunks, chunk_body, carry)


def _validate_reference_inputs(
    kv_cache: jax.Array,
    use_per_token_scale: bool,
    dynamic_k_scale: jax.Array | None,
    dynamic_v_scale: jax.Array | None,
) -> None:
  """Validates unsupported options and KV cache rank for the reference op."""
  if (
      use_per_token_scale
      or dynamic_k_scale is not None
      or dynamic_v_scale is not None
  ):
    raise NotImplementedError(
        "Per-token scale is not supported in"
        " batched_ragged_paged_attention_reference."
    )
  if kv_cache.ndim != 5:
    raise ValueError(
        "kv_cache must be 5D, matching get_kv_cache_shape. Got"
        f" {kv_cache.ndim}D with shape {kv_cache.shape}"
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "use_causal_mask",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "out_dtype",
        "use_per_token_scale",
        "per_token_scale_dtype",
        "q_scale",
        "k_scale",
        "v_scale",
        "decode_query_size",
        "skip_kv_update",
        "kv_layout",
        "cp_group_size",
        "attention_scope",
        "return_lse",
        "quantize_p",
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
    attention_sink: jax.Array | None = None,
    *,
    use_causal_mask: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    out_dtype: Any = None,
    use_per_token_scale: bool = False,
    per_token_scale_dtype: Any = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    dynamic_k_scale: jax.Array | None = None,
    dynamic_v_scale: jax.Array | None = None,
    decode_query_size: int = 1,
    skip_kv_update: bool = False,
    kv_layout: Any = None,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: Any = None,
    return_lse: bool = False,
    quantize_p: bool = False,
    **unused_kwargs: Any,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
  """Reference multi-head / grouped-query ragged paged attention in pure JAX.

  Args:
    queries: [total_q_tokens, num_q_heads, head_dim]
    keys: [total_q_tokens, num_kv_heads, head_dim]
    values: [total_q_tokens, num_kv_heads, head_dim]
    kv_cache: [total_pages, page_size, num_kv_heads_x2 // kv_packing,
      kv_packing, head_dim_aligned]
    kv_lens: [max_num_seqs] sequence lengths for each request.
    page_indices: [max_num_seqs * pages_per_seq] page allocation mapping.
    cu_q_lens: [max_num_seqs + 1] cumulative query tokens per request.
    distribution: [3] active counts for [prefill, decode, mixed] batch splits.
    attention_sink: Optional [num_q_heads] sink logits included in softmax
      normalization.
    use_causal_mask: Whether to apply lower-triangular causal masking.
    sm_scale: Softmax scaling factor (typically 1 / sqrt(head_dim)).
    sliding_window: Optional local attention window size.
    soft_cap: Optional tanh logit soft-capping factor.
    mask_value: Value for masked logits. Defaults to `jnp.finfo(out_dtype).min`.
    out_dtype: Desired output data type.
    q_scale: Optional query quantization scale.
    k_scale: Optional key quantization scale.
    v_scale: Optional value quantization scale.

  Returns:
    outputs: [total_q_tokens, num_q_heads, head_dim] attention outputs.
    updated_kv_cache: Updated KV cache buffer.
    lse: Optional [total_q_tokens, num_q_heads] log-sum-exp in `out_dtype`.
  """
  del per_token_scale_dtype
  _validate_reference_inputs(
      kv_cache, use_per_token_scale, dynamic_k_scale, dynamic_v_scale
  )
  if out_dtype is None:
    out_dtype = queries.dtype

  actual_head_dim = queries.shape[-1]
  actual_num_q_heads = queries.shape[1]
  actual_num_kv_heads = keys.shape[1]

  orig_kv_cache = kv_cache
  orig_kv_cache_shape = kv_cache.shape
  total_q_tokens = queries.shape[0]
  if total_q_tokens == 0:
    empty_out = queries.astype(out_dtype)
    if return_lse:
      return (
          empty_out,
          orig_kv_cache,
          jnp.empty((0, actual_num_q_heads), dtype=out_dtype),
      )
    return empty_out, orig_kv_cache

  kv_cache, lane_fold = _unfold_kv_cache_if_needed(
      kv_cache, actual_num_kv_heads, actual_head_dim, kv_layout
  )

  max_num_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_num_seqs

  if skip_kv_update:
    updated_kv_cache = kv_cache
  else:
    updated_kv_cache = _update_kv_cache(
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
    )

  page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
  scaled_queries = _quantize_queries(queries, q_scale, kv_cache.dtype)
  effective_mask_value = (
      float(jnp.finfo(jnp.dtype(out_dtype)).min)
      if mask_value is None
      else mask_value
  )
  chunk_size = min(total_q_tokens, _MAX_QUERY_CHUNK_SIZE)

  padded_queries = jnp.pad(
      scaled_queries, ((0, chunk_size), (0, 0), (0, 0)), constant_values=0
  )
  padded_outputs = jnp.zeros(
      (total_q_tokens + chunk_size, actual_num_q_heads, actual_head_dim),
      dtype=out_dtype,
  )
  padded_lses = jnp.full(
      (total_q_tokens + chunk_size, actual_num_q_heads),
      -jnp.inf,
      dtype=out_dtype,
  )

  active_count = jnp.clip(distribution[-1], 0, max_num_seqs)
  active_q_limit = jnp.minimum(cu_q_lens[active_count], total_q_tokens)

  seq_body = functools.partial(
      _attend_sequence,
      padded_queries=padded_queries,
      updated_kv_cache=updated_kv_cache,
      page_table=page_table,
      kv_lens=kv_lens,
      cu_q_lens=cu_q_lens,
      active_q_limit=active_q_limit,
      total_q_tokens=total_q_tokens,
      chunk_size=chunk_size,
      actual_num_kv_heads=actual_num_kv_heads,
      actual_head_dim=actual_head_dim,
      sm_scale=sm_scale,
      q_scale=q_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      use_causal_mask=use_causal_mask,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=effective_mask_value,
      attention_sink=attention_sink,
      quantize_p=quantize_p,
      kv_cache_dtype=kv_cache.dtype,
      out_dtype=out_dtype,
  )
  padded_outputs, padded_lses = lax.fori_loop(
      0, active_count, seq_body, (padded_outputs, padded_lses)
  )
  outputs = padded_outputs[:total_q_tokens]

  returned_kv_cache = (
      orig_kv_cache
      if skip_kv_update
      else _refold_kv_cache_if_needed(
          updated_kv_cache,
          orig_kv_cache_shape,
          actual_num_kv_heads,
          actual_head_dim,
          lane_fold,
      )
  )

  if return_lse:
    lse = padded_lses[:total_q_tokens]
    return outputs, returned_kv_cache, lse

  return outputs, returned_kv_cache
