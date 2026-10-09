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
"""Pure JAX reference implementation of Sparse MLA attention."""

import functools

import jax
from jax import lax
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.compress_store import csa_cache_layout

ROPE_WORDS = csa_cache_layout.ROPE_WORDS
ROPE_DIM = csa_cache_layout.ROPE_DIM
NOPE_DIM = 448
NOPE_SCALE_PERIOD = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM


def decode_rope(cache_kv_rope: jax.Array) -> jax.Array:
  """Unpacks packed int32 RoPE cache words into bfloat16 RoPE values."""
  total_pages, rope_rows, _ = cache_kv_rope.shape
  page_size = rope_rows * 4
  words = lax.bitcast_convert_type(
      cache_kv_rope.reshape(total_pages, page_size, ROPE_WORDS), jnp.uint32
  )
  bits = jnp.concatenate([words & 0xFFFF, words >> 16], axis=-1).astype(
      jnp.uint16
  )
  return lax.bitcast_convert_type(bits, jnp.bfloat16)


def dequantize_kv_cache(
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
) -> jax.Array:
  """Unpacks and dequantizes the NoPE and RoPE int32 caches to bfloat16."""
  total_num_pages, page_size, _ = cache_kv_nope.shape
  nope_bytes = csa_cache_layout.words_to_slabs(cache_kv_nope).reshape(
      total_num_pages, page_size, 512
  )
  fp8_quant = lax.bitcast_convert_type(
      nope_bytes[..., :NOPE_DIM], jnp.float8_e4m3fn
  ).reshape(
      total_num_pages,
      page_size,
      NOPE_DIM // NOPE_SCALE_PERIOD,
      NOPE_SCALE_PERIOD,
  )
  scales_quant = lax.bitcast_convert_type(
      nope_bytes[..., NOPE_DIM:], jnp.float8_e8m0fnu
  )
  fp8_dequant = (
      fp8_quant.astype(jnp.bfloat16)
      * scales_quant[..., None, :].astype(jnp.bfloat16)
  ).reshape(total_num_pages, page_size, NOPE_DIM)

  bf16_rope = decode_rope(cache_kv_rope)
  return jnp.concatenate([fp8_dequant, bf16_rope], axis=-1)


@functools.partial(jax.jit, static_argnames=("sm_scale",))
def sparse_ragged_paged_attention(
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sinks: jax.Array,
    swa_accumution: jax.Array,
    swa_l: jax.Array,
    swa_m: jax.Array,
    *,
    sm_scale: float = 1.0,
) -> jax.Array:
  """Pure JAX reference for Sparse MLA ragged paged attention.

  Args:
    q: queries
    cache_kv_nope: quantized NoPE KV cache
    cache_kv_rope: packed RoPE KV cache
    topk_indices: logical KV token indices per query token within its sequence
      (-1 for padding)
    page_indices: flattened page table
    cu_q_lens: cumulative query token lengths
    distribution: batch distribution (num_decode, num_prefill_end, num_seqs)
    attention_sinks: attention sink logits
    swa_accumution: sliding window attention numerator accumulator
    swa_l: sliding window attention denominator sum
    swa_m: sliding window attention row max
    sm_scale: Softmax scale applied to Q @ K^T

  Returns:
    Output attention tensor.
  """
  max_num_tokens = q.shape[0]
  _, page_size, _ = cache_kv_nope.shape
  max_num_seqs = cu_q_lens.shape[0] - 1
  pages_per_seq = page_indices.shape[0] // max_num_seqs
  page_table = page_indices.reshape(max_num_seqs, pages_per_seq)

  kv_c_cache = dequantize_kv_cache(cache_kv_nope, cache_kv_rope)

  valid_seq = jnp.arange(max_num_seqs, dtype=jnp.int32) < distribution[2]
  tokens_per_seq = jnp.where(valid_seq, cu_q_lens[1:] - cu_q_lens[:-1], 0)
  seq_ids_segment = jnp.repeat(
      jnp.arange(max_num_seqs, dtype=jnp.int32),
      tokens_per_seq,
      total_repeat_length=max_num_tokens,
  )

  valid_mask = topk_indices != -1
  safe_topk = jnp.where(valid_mask, topk_indices, 0)
  seq_page_ids = jnp.clip(safe_topk // page_size, 0, pages_per_seq - 1)
  token_offset = safe_topk % page_size
  phys_page_ids = jnp.take_along_axis(
      page_table[seq_ids_segment], seq_page_ids, axis=-1
  )

  kv_gathered = kv_c_cache[phys_page_ids, token_offset]
  kv_gathered = jnp.where(valid_mask[..., None], kv_gathered, 0)

  attn = jnp.einsum(
      "tnh,tkh->tnk",
      q,
      kv_gathered,
      preferred_element_type=jnp.float32,
  )
  attn *= sm_scale
  attn = jnp.where(valid_mask[:, None, :], attn, jnp.finfo(jnp.float32).min)

  m_2 = jnp.max(attn, axis=-1)
  m_1 = swa_m
  m = jnp.maximum(m_1, m_2)

  l_1_scaled = swa_l * jnp.exp(m_1 - m)
  p_2 = jnp.exp(attn - m[..., None])
  l_2 = jnp.sum(p_2, axis=-1)
  l_sinks = jnp.exp(attention_sinks[None, :] - m)
  l = l_1_scaled + l_2 + l_sinks

  acc_2 = jnp.einsum(
      "tnk,tkh->tnh",
      p_2,
      kv_gathered,
      preferred_element_type=jnp.float32,
  )
  acc_1_scaled = (
      swa_accumution.astype(jnp.float32) * jnp.exp(m_1 - m)[..., None]
  )
  acc = acc_1_scaled + acc_2
  return (acc / l[..., None]).astype(q.dtype)
