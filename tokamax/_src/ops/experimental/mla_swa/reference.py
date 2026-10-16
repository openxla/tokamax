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
"""Pure JAX reference implementation of MLA sliding window attention."""

import functools

import jax
from jax import lax
import jax.numpy as jnp
from tokamax._src.ops.dsv4_compressor import csa_cache_layout

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def align_to(x: int, a: int) -> int:
  return ((x + a - 1) // a) * a


def encode_kv_tokens(kv: jax.Array) -> jax.Array:
  """Packs bfloat16 KV tokens into interleaved little-endian uint8 sublanes."""
  tokens, head_dim = kv.shape
  kv_u16 = lax.bitcast_convert_type(kv, jnp.uint16)
  kv_hi = ((kv_u16 >> 8) & 0xFF).astype(jnp.uint8)
  kv_lo = (kv_u16 & 0xFF).astype(jnp.uint8)
  nb = head_dim // 128
  kv_hi = kv_hi.reshape(tokens, nb, 128)
  kv_lo = kv_lo.reshape(tokens, nb, 128)
  return jnp.stack([kv_lo, kv_hi], axis=2).reshape(tokens, head_dim * 2)


def decode_kv_slabs(
    cache_slabs: jax.Array,
    *,
    logical_page_size: int,
    head_dim: int,
) -> jax.Array:
  """Unpacks uint8 slab cache pages into [num_pages, logical_page_size, head_dim] bf16."""
  total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
      cache_slabs.shape
  )
  slot_bytes = kv_packing * lkv_dim
  token_bytes = head_dim * 2
  slots_per_token = token_bytes // slot_bytes
  phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token
  nb = head_dim // 128

  cache_tokens = cache_slabs.reshape(
      total_num_pages,
      phys_tokens_per_page,
      slots_per_token,
      kv_packing,
      lkv_dim,
  )
  logical_bytes = cache_tokens[:, :logical_page_size].reshape(
      total_num_pages, logical_page_size, nb, 2, 128
  )
  lo = logical_bytes[..., 0, :].astype(jnp.uint16)
  hi = logical_bytes[..., 1, :].astype(jnp.uint16)
  kv_u16 = ((hi << 8) | lo).reshape(
      total_num_pages, logical_page_size, head_dim
  )
  return lax.bitcast_convert_type(kv_u16, jnp.bfloat16)


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "logical_page_size",
        "chunk_prefill_size",
        "unnormalized_output",
        "non_causal_block",
        "mask_value",
    ),
)
def mla_sliding_window_ragged_paged_attention(
    q: jax.Array,
    new_kv: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sinks: jax.Array,
    *,
    sliding_window: int,
    logical_page_size: int,
    sm_scale: float = 1.0,
    chunk_prefill_size: int | None = None,
    unnormalized_output: bool = False,
    non_causal_block: bool = False,
    mask_value: float | None = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """JAX reference impl for MLA sliding-window ragged paged attention.

  Args:
    q: Query tensor
    new_kv: New KV tokens.
    cache_kv: Paged SWA KV cache.
    kv_lens: Per-sequence KV lengths.
    page_indices: Flattened page table.
    cu_q_lens: Cumulative query token lengths.
    distribution: Batch distribution.
    attention_sinks: Attention sink logits.
    sliding_window: Sliding window size in tokens.
    logical_page_size: Logical page capacity in tokens.
    sm_scale: Softmax scale.
    chunk_prefill_size: Optional static query length for the prefill-only
      segment.
    unnormalized_output: Whether to return the unnormalized numerator
      accumulator instead of dividing by the softmax denominator and sinks.
    non_causal_block: If true, disables causal masking.
    mask_value: Score assigned to masked KV positions.

  Returns:
    Tuple (output, updated_cache_kv, out_l, out_m).
  """
  del chunk_prefill_size
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE

  max_num_tokens, _, actual_head_dim = q.shape
  head_dim = align_to(actual_head_dim, 128)
  if actual_head_dim != head_dim:
    q_padded = jnp.pad(
        q, ((0, 0), (0, 0), (0, head_dim - actual_head_dim)), constant_values=0
    )
    new_kv_padded = jnp.pad(
        new_kv, ((0, 0), (0, head_dim - actual_head_dim)), constant_values=0
    )
  else:
    q_padded = q
    new_kv_padded = new_kv

  cache_is_words = csa_cache_layout.is_word_array(cache_kv.dtype)
  cache_slabs = (
      csa_cache_layout.words_to_slabs(cache_kv) if cache_is_words else cache_kv
  )
  total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
      cache_slabs.shape
  )
  slot_bytes = kv_packing * lkv_dim
  token_bytes = head_dim * 2
  slots_per_token = token_bytes // slot_bytes
  phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token

  new_kv_bytes = encode_kv_tokens(new_kv_padded).reshape(
      max_num_tokens, slots_per_token, kv_packing, lkv_dim
  )
  cache_tokens = cache_slabs.reshape(
      total_num_pages,
      phys_tokens_per_page,
      slots_per_token,
      kv_packing,
      lkv_dim,
  )

  max_num_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_num_seqs

  def seq_loop_body(i, cache_):
    q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]

    def token_loop_body(j, c_):
      token_idx_in_seq = kv_len - q_len + j
      page_num_in_seq = token_idx_in_seq // logical_page_size
      in_page_off = token_idx_in_seq % logical_page_size
      page_idx = page_indices[i * pages_per_seq + page_num_in_seq]
      return c_.at[page_idx, in_page_off].set(new_kv_bytes[q_start + j])

    return lax.fori_loop(0, q_len, token_loop_body, cache_)

  updated_cache_tokens = lax.fori_loop(
      0, distribution[2], seq_loop_body, cache_tokens
  )
  updated_cache_slabs = updated_cache_tokens.reshape(cache_slabs.shape)
  updated_cache_kv = (
      csa_cache_layout.slabs_to_words(updated_cache_slabs)
      if cache_is_words
      else updated_cache_slabs
  )

  kv_c_cache = decode_kv_slabs(
      updated_cache_slabs,
      logical_page_size=logical_page_size,
      head_dim=head_dim,
  )

  page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
  valid_seq = jnp.arange(max_num_seqs, dtype=jnp.int32) < distribution[2]
  q_lens = jnp.where(
      valid_seq, jnp.maximum(cu_q_lens[1:] - cu_q_lens[:-1], 0), 0
  )
  seq_ids = jnp.repeat(
      jnp.arange(max_num_seqs, dtype=jnp.int32),
      q_lens,
      total_repeat_length=max_num_tokens,
  )
  token_indices = jnp.arange(max_num_tokens, dtype=jnp.int32)
  valid_token = token_indices < cu_q_lens[distribution[2]]
  kv_len_per_token = kv_lens[seq_ids]
  q_len_per_token = q_lens[seq_ids]
  local_q_idx = token_indices - cu_q_lens[seq_ids]
  q_pos = kv_len_per_token - q_len_per_token + local_q_idx

  window_span = sliding_window + (max_num_tokens if non_causal_block else 0)
  k_start = jnp.maximum(q_pos - sliding_window + 1, 0)
  k_pos = k_start[:, None] + jnp.arange(window_span, dtype=jnp.int32)[None, :]

  in_seq = (
      valid_token[:, None] & (k_pos >= 0) & (k_pos < kv_len_per_token[:, None])
  )
  in_window = (q_pos[:, None] >= k_pos) & (
      q_pos[:, None] - k_pos < sliding_window
  )
  if non_causal_block:
    in_block = k_pos >= (kv_len_per_token - q_len_per_token)[:, None]
    keep = in_seq & (in_window | in_block)
  else:
    keep = in_seq & in_window

  safe_k_pos = jnp.where(keep, k_pos, 0)
  seq_page_ids = jnp.clip(safe_k_pos // logical_page_size, 0, pages_per_seq - 1)
  token_offset = safe_k_pos % logical_page_size
  phys_page_ids = jnp.take_along_axis(
      page_table[seq_ids], seq_page_ids, axis=-1
  )
  kv_gathered = kv_c_cache[phys_page_ids, token_offset]
  kv_gathered = jnp.where(keep[..., None], kv_gathered, 0)

  attn = jnp.einsum(
      "tnh,tkh->tnk",
      q_padded,
      kv_gathered,
      preferred_element_type=jnp.float32,
  )
  attn *= sm_scale
  attn = jnp.where(keep[:, None, :], attn, mask_value)

  m = jnp.max(attn, axis=-1)
  p = jnp.exp(attn - m[..., None])
  l_sum = jnp.sum(p, axis=-1)
  acc = jnp.einsum(
      "tnk,tkh->tnh",
      p,
      kv_gathered,
      preferred_element_type=jnp.float32,
  )
  if unnormalized_output:
    out = acc.astype(q.dtype)
  else:
    l_sinks = jnp.exp(attention_sinks[None, :] - m)
    l_final = l_sum + l_sinks
    out = (acc / l_final[..., None]).astype(q.dtype)
  out = out[..., :actual_head_dim]

  out = jnp.where(valid_token[:, None, None], out, 0)
  l_sum = jnp.where(valid_token[:, None], l_sum, 0)
  m = jnp.where(valid_token[:, None], m, 0)

  return out, updated_cache_kv, l_sum, m
