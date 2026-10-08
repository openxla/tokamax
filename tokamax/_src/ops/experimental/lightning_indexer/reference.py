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
"""Pure JAX reference implementation for Lightning Indexer."""

import functools

import jax
from jax import lax
import jax.numpy as jnp
from tokamax._src.ops.experimental.lightning_indexer.kernel import config as kernel_config
from tokamax._src.ops.experimental.lightning_indexer.kernel import metadata

KVLayout = kernel_config.KVLayout


def _align_to(x: int, a: int) -> int:
  return ((x + a - 1) // a) * a


def _dequantize_cache_kv(
    cache_kv: jax.Array,
    actual_head_dim: int,
    kv_layout: KVLayout,
) -> tuple[jax.Array, jax.Array]:
  """Unpacks packed uint8 `cache_kv` into `(fp8_keys, scales)` per page/token.

  Args:
    cache_kv: Packed uint8 KV cache in `HEAD_ALONG_SUBLANE` or `SEQ_ALONG_LANE`.
    actual_head_dim: Unpadded query/key head dimension.
    kv_layout: Memory layout of `cache_kv`.

  Returns:
    keys: `float8_e4m3fn[total_num_pages, page_size, actual_head_dim]`
    scales: `bfloat16[total_num_pages, page_size]`
  """
  head_dim = _align_to(actual_head_dim, 128)
  if kv_layout == KVLayout.SEQ_ALONG_LANE:
    total_num_pages, _, _, page_size = cache_kv.shape
    flat = cache_kv.reshape(total_num_pages, -1, page_size).swapaxes(1, 2)
  else:
    total_num_pages, page_size_per_packing, kv_packing, width = cache_kv.shape
    page_size = page_size_per_packing * kv_packing
    flat = cache_kv.reshape(total_num_pages, page_size, width)

  key_bytes = flat[:, :, :actual_head_dim]
  scale_bytes = flat[:, :, head_dim]
  keys = lax.bitcast_convert_type(key_bytes, jnp.float8_e4m3fn)
  scales = lax.bitcast_convert_type(scale_bytes, jnp.float8_e8m0fnu).astype(
      jnp.bfloat16
  )
  return keys, scales


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "compression_ratio",
        "kv_layout",
        "cp_size",
        "interleave_size",
        "return_scores",
    ),
)
def lightning_indexer(
    q: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    compression_ratio: int = 1,
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
    cp_size: int = 1,
    cp_rank: jax.Array | int = 0,
    interleave_size: int = 1,
    return_scores: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Pure JAX reference implementation for Lightning Indexer retrieval.

  Args:
    q: Query tensor of shape [max_num_tokens, num_q_heads, head_dim].
    indexer_weights: Indexer head weights of shape [max_num_tokens,
      num_q_heads].
    cache_kv: Packed uint8 KV cache based on kv_layout
    seq_lens: Uncompressed KV sequence lengths of shape [max_num_seqs]
    page_indices: Flattened page table
    cu_q_lens: Cumulative query token counts
    distribution: Batch split counts
    k: Number of top-K compressed KV positions to retrieve per query token
    compression_ratio: KV cache compression ratio (power of 2)
    kv_layout: Memory layout of cache_kv
    cp_size: Context-parallel world size
    cp_rank: Context-parallel rank of the current shard
    interleave_size: Context-parallel chunk-interleave width in uncompressed
      tokens
    return_scores: If True, also return the winner scores as int32 holding raw
      float32 bits

  Returns
    top_indices of shape [max_num_tokens, k]
  """
  kv_layout = KVLayout.parse(kv_layout)
  if (
      compression_ratio < 1
      or (compression_ratio & (compression_ratio - 1)) != 0
  ):
    raise ValueError("compression_ratio must be a power of 2.")
  if cp_size < 1:
    raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
  if cp_size > 1 and interleave_size % compression_ratio != 0:
    raise ValueError(
        f"interleave_size ({interleave_size}) must be a multiple of "
        f"compression_ratio ({compression_ratio})."
    )
  if cp_size > 1:
    interleave_c = interleave_size // cp_size
  else:
    interleave_c = 1

  num_tokens, _, actual_head_dim = q.shape
  max_num_seqs = seq_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_num_seqs

  keys, scales = _dequantize_cache_kv(cache_kv, actual_head_dim, kv_layout)
  total_num_pages, page_size, _ = keys.shape
  max_kv_len = pages_per_seq * page_size

  num_seqs = distribution[2]
  token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
  seq_mask = token_ids[:, None] >= cu_q_lens[None, 1 : max_num_seqs + 1]
  seq_mask = jnp.where(
      jnp.arange(max_num_seqs)[None, :] < num_seqs, seq_mask, False
  )
  seq_ids = jnp.sum(seq_mask, axis=1)
  valid_token = (token_ids < cu_q_lens[num_seqs]) & (seq_ids < num_seqs)

  page_table = jnp.clip(
      page_indices.reshape(max_num_seqs, pages_per_seq),
      0,
      total_num_pages - 1,
  )
  k_local = jnp.arange(max_kv_len, dtype=jnp.int32)
  k_global = metadata.cp_local_to_global(
      k_local, cp_rank, cp_size, interleave_c
  )

  def _score_token(t: jax.Array) -> jax.Array:
    s = seq_ids[t]
    seq_pages = page_table[s]
    seq_k = keys[seq_pages].reshape(max_kv_len, actual_head_dim)
    seq_scale = scales[seq_pages].reshape(max_kv_len).astype(jnp.float32)

    q_t = q[t].astype(jnp.float32)
    w_t = indexer_weights[t].astype(jnp.float32)
    dots = jnp.einsum(
        "hd,kd->hk",
        q_t,
        seq_k.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    weighted = (jnp.maximum(dots, 0.0) * w_t[:, None]).sum(axis=0)
    raw_scores = weighted * seq_scale

    q_start = cu_q_lens[s]
    q_len = cu_q_lens[s + 1] - q_start
    seq_len = seq_lens[s]
    q_pos_compressed = (seq_len - q_len + (t - q_start)) // compression_ratio
    kv_len = seq_len // compression_ratio

    mask = valid_token[t] & (k_global < kv_len) & (k_global <= q_pos_compressed)
    return jnp.where(mask, raw_scores, -jnp.inf)

  scores = jax.vmap(_score_token)(token_ids)
  if scores.shape[1] < k:
    scores = jnp.pad(
        scores,
        ((0, 0), (0, k - scores.shape[1])),
        constant_values=-jnp.inf,
    )

  top_scores, top_indices = lax.top_k(scores, k)
  top_indices = jnp.where(
      jnp.isfinite(top_scores), top_indices.astype(jnp.int32), -1
  )
  if return_scores:
    top_scores_bits = lax.bitcast_convert_type(top_scores, jnp.int32)
    return top_indices, top_scores_bits
  return top_indices
