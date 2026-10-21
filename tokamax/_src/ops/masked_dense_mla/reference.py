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
"""Pure JAX reference implementation of masked-dense MLA attention."""

import functools

import jax
from jax import lax
import jax.numpy as jnp

NOPE_DIM = 512
ROPE_STORAGE_DIM = 128
CACHE_HEAD_DIM = NOPE_DIM + ROPE_STORAGE_DIM

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "k_scale",
        "mask_value",
        "max_kv_len",
        "chunk_prefill_size",
    ),
)
def masked_dense_ragged_paged_attention(
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    kv_lens: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    k_scale: float = 1.0,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    max_kv_len: int | None = None,
    chunk_prefill_size: int | None = None,
    sequence_start: jax.Array | None = None,
) -> jax.Array:
  """Pure JAX reference for masked-dense MLA ragged paged attention.

  Args:
    q: Query tensor of shape `(max_num_tokens, num_q_heads, head_dim)`.
    cache_kv_nope: Quantized NoPE KV cache of shape `(total_num_pages,
      page_size, 4, 128)` with dtype `uint8`.
    cache_kv_rope: Quantized RoPE KV cache of shape `(total_num_pages,
      page_size // 4, 4, 128)` with dtype `uint8`.
    kv_lens: Per-sequence KV lengths of shape `(max_num_seqs,)`.
    topk_indices: Selected top-k KV token indices per query token of shape
      `(max_num_tokens, topk)`, `-1` padded.
    page_indices: Flattened page table of shape `(max_num_seqs *
      pages_per_seq,)`.
    cu_q_lens: Cumulative query token lengths of shape `(max_num_seqs + 1,)`.
    distribution: Batch distribution `(num_decode, num_prefill_end, num_seqs)`.
    sm_scale: Softmax scale applied to `Q @ K^T`.
    k_scale: Per-tensor dequantization scale for the FP8 caches.
    mask_value: Score written where the mask excludes a KV position.
    max_kv_len: Optional upper bound on every sequence's `kv_len`.
    chunk_prefill_size: Optional static query length for the prefill-only
      segment.
    sequence_start: Optional first sequence index to attend; sequences `[0,
      sequence_start)` keep their rows in `q` unchanged.

  Returns:
    Output attention tensor of shape `(max_num_tokens, num_q_heads, head_dim)`.
  """
  del chunk_prefill_size
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE

  max_num_tokens, _, actual_head_dim = q.shape
  total_num_pages, page_size, _, _ = cache_kv_nope.shape
  max_num_seqs = cu_q_lens.shape[0] - 1
  pages_per_seq = page_indices.shape[0] // max_num_seqs
  page_table_span = pages_per_seq * page_size
  if max_kv_len is None:
    max_kv_len = page_table_span

  nope_fp8 = lax.bitcast_convert_type(
      cache_kv_nope.reshape(total_num_pages, page_size, NOPE_DIM),
      jnp.float8_e4m3fn,
  )
  rope_fp8 = lax.bitcast_convert_type(
      cache_kv_rope.reshape(total_num_pages, page_size, ROPE_STORAGE_DIM),
      jnp.float8_e4m3fn,
  )
  kv_cache = jnp.concatenate([nope_fp8, rope_fp8], axis=-1)

  padded_head_dim = max(CACHE_HEAD_DIM, ((actual_head_dim + 127) // 128) * 128)
  if actual_head_dim < padded_head_dim:
    q_padded = jnp.pad(
        q, ((0, 0), (0, 0), (0, padded_head_dim - actual_head_dim))
    )
  else:
    q_padded = q
  if kv_cache.shape[-1] < padded_head_dim:
    kv_cache = jnp.pad(
        kv_cache,
        ((0, 0), (0, 0), (0, padded_head_dim - kv_cache.shape[-1])),
    )

  page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
  seq_kv = kv_cache[page_table].reshape(
      max_num_seqs, page_table_span, padded_head_dim
  )[:, :max_kv_len, :]

  valid_seq = jnp.arange(max_num_seqs, dtype=jnp.int32) < distribution[2]
  tokens_per_seq = jnp.where(valid_seq, cu_q_lens[1:] - cu_q_lens[:-1], 0)
  seq_ids = jnp.repeat(
      jnp.arange(max_num_seqs, dtype=jnp.int32),
      tokens_per_seq,
      total_repeat_length=max_num_tokens,
  )

  token_indices = jnp.arange(max_num_tokens, dtype=jnp.int32)
  kv_len_per_token = kv_lens[seq_ids]
  q_len_per_token = cu_q_lens[seq_ids + 1] - cu_q_lens[seq_ids]
  local_q_idx = token_indices - cu_q_lens[seq_ids]

  k_positions = jnp.arange(max_kv_len, dtype=jnp.int32)
  in_bounds = k_positions[None, :] < kv_len_per_token[:, None]
  kv_per_token = jnp.where(in_bounds[..., None], seq_kv[seq_ids], 0)

  if max_kv_len <= topk_indices.shape[-1]:
    q_pos = kv_len_per_token - q_len_per_token + local_q_idx
    valid_mask = k_positions[None, :] <= q_pos[:, None]
  else:
    valid_mask = jnp.any(
        topk_indices[:, :, None] == k_positions[None, None, :], axis=1
    )

  attn = jnp.einsum(
      "tnh,tkh->tnk",
      q_padded,
      kv_per_token,
      preferred_element_type=jnp.float32,
  )
  attn = attn * sm_scale * k_scale
  attn = jnp.where(valid_mask[:, None, :], attn, mask_value)

  m = jnp.max(attn, axis=-1, keepdims=True)
  p = jnp.exp(attn - m)
  l = jnp.sum(p, axis=-1, keepdims=True)
  acc = (
      jnp.einsum(
          "tnk,tkh->tnh",
          p,
          kv_per_token,
          preferred_element_type=jnp.float32,
      )
      * k_scale
  )
  out = (acc / l).astype(q.dtype)[..., :actual_head_dim]
  if sequence_start is not None:
    skip_token = token_indices < cu_q_lens[sequence_start]
    out = jnp.where(skip_token[:, None, None], q, out)
  return out
