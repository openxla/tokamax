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
"""Pure JAX reference implementation of the DeepSeek-V4 compressor projection.

The compressor of a DeepSeek-V4 CSA or HCA layer (and of the CSA indexer) keeps,
for every token, an f32 state of `2 * state_width` values: the projected `kv`
followed by the projected `score` plus its absolute position embedding (APE).
`state_width` is the head dimension, doubled when compression windows overlap
(CSA and the indexer).

The state is stored in a paged cache of uint8 slabs,
`uint8[num_pages, page_size, 4, lanes]`, where one row (a `(4, lanes)` slab)
holds `lanes` f32 values, value `l` being bytes `(0..3, l)` of the slab. A token
occupies `2 * state_width // lanes` consecutive rows starting at its
`slot_mapping` entry. CSA allocates its cache as `int32[num_pages, page_size,
lanes]` instead, where word `l` of a row is the f32 bit pattern of value `l`.
"""

import functools

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.compress_store import csa_cache_layout


def ref_wkv_proj_and_save_state(
    hidden_states: jax.Array,  # [num_tokens, hidden_size]
    wkv_wgate: jax.Array,  # [hidden_size, 2 * state_width]
    ape: jax.Array,  # [compress_ratio, state_width]
    positions: jax.Array,  # [num_tokens]
    slot_mapping: jax.Array,  # [num_tokens]
    cache: jax.Array,  # [num_pages, page_size, 4, 128] uint8
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> jax.Array:
  """Golden JAX reference for Kernel 1 (proj + save state)."""
  coff = 1 + int(overlap)
  state_width = coff * head_dim
  state_dim = 2 * state_width

  # 1. Project & APE
  kv_score = jnp.matmul(hidden_states, wkv_wgate)
  kv = kv_score[:, :state_width]
  score = kv_score[:, state_width:]

  ape_rows = jnp.mod(positions, compress_ratio)
  score_state = score + ape[ape_rows]
  packed = jnp.concatenate([kv, score_state], axis=-1)

  # 2. Geometry setup
  num_pages, page_size, d1, d2 = cache.shape
  slot_bytes = d1 * d2
  slots_per_token = state_dim * 4 // slot_bytes
  f32_per_slot = state_dim // slots_per_token
  bytes_per_slot = f32_per_slot * 4
  assert bytes_per_slot == slot_bytes
  tokens_per_page = page_size // slots_per_token
  assert state_block_size <= tokens_per_page, (
      f"state_block_size {state_block_size} exceeds the {tokens_per_page} "
      f"token states a {page_size}-row page holds"
  )

  # 3. Unpack inline
  cache_reshaped = cache.reshape(
      num_pages, tokens_per_page, slots_per_token, d1, d2
  )
  cache_t = cache_reshaped.transpose(0, 1, 2, 4, 3)
  cache_bitcast_shape = cache_t.reshape(
      num_pages, tokens_per_page, slots_per_token, (d2 * d1) // 4, 4
  )
  f32_flat = jax.lax.bitcast_convert_type(cache_bitcast_shape, jnp.float32)
  flat = f32_flat.reshape(num_pages * tokens_per_page, state_dim)

  # 4. Scatter
  num_tokens_flat = num_pages * tokens_per_page
  valid = slot_mapping >= 0
  slots = jnp.where(valid, slot_mapping // slots_per_token, num_tokens_flat)
  flat_padded = jnp.concatenate([flat, jnp.zeros((1, state_dim))], axis=0)
  flat_padded = flat_padded.at[slots].set(packed)
  flat = flat_padded[:-1]

  # 5. Pack inline back
  chunk = flat.reshape(
      num_pages, tokens_per_page, slots_per_token, f32_per_slot, 1
  )
  chunk_bytes = jax.lax.bitcast_convert_type(chunk, jnp.uint8)
  chunk_bytes_reshaped = chunk_bytes.reshape(
      num_pages, tokens_per_page, slots_per_token, d2, d1
  )
  chunk_bytes_t = chunk_bytes_reshaped.transpose(0, 1, 2, 4, 3)
  cache_view = cache.reshape(
      num_pages, tokens_per_page, slots_per_token, d1, d2
  )
  cache_view = cache_view.at[:].set(chunk_bytes_t)
  new_cache = cache_view.reshape(num_pages, page_size, d1, d2)

  return new_cache


@functools.partial(jax.jit, static_argnames=("compress_ratio",))
def proj_and_save_state(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    ape: jax.Array,
    positions: jax.Array,
    slot_mapping: jax.Array,
    cache: jax.Array,
    *,
    compress_ratio: int,
) -> jax.Array:
  """Pure JAX reference for the compressor projection, in the kernel's signature.

  Wraps `ref_wkv_proj_and_save_state`, accepting both cache declarations and
  deriving the static arguments the kernel does not take: `state_width` comes
  from `wkv_wgate` (passed as `head_dim` with `overlap=False`, which gives the
  same width), and `state_block_size` is the page's full token capacity (it
  only bounds an assertion).

  Args:
    hidden_states: `(num_tokens, hidden_size)` hidden states.
    wkv_wgate: `(hidden_size, 2 * state_width)` fused `kv` and `score`
      projection weights.
    ape: `(compress_ratio, state_width)` absolute position embeddings, added to
      `score`.
    positions: `(num_tokens,)` int32 token positions; token `t` adds APE row
      `positions[t] % compress_ratio`.
    slot_mapping: `(num_tokens,)` int32 first cache row (`page * page_size +
      row`) of each token's state, a multiple of the rows per token. Negative
      entries skip the token.
    cache: `(num_pages, page_size, 4, lanes)` uint8 or `(num_pages, page_size,
      lanes)` int32 state cache.
    compress_ratio: Number of APE rows.

  Returns:
    The updated cache, with the shape and dtype of `cache`.
  """
  state_width = wkv_wgate.shape[1] // 2
  is_words = csa_cache_layout.is_word_array(cache.dtype)
  slabs = csa_cache_layout.words_to_slabs(cache) if is_words else cache
  _, page_size, d1, d2 = slabs.shape
  rows_per_token = 2 * state_width * 4 // (d1 * d2)
  new_slabs = ref_wkv_proj_and_save_state(
      hidden_states,
      wkv_wgate,
      ape,
      positions,
      slot_mapping,
      slabs,
      state_block_size=page_size // rows_per_token,
      head_dim=state_width,
      compress_ratio=compress_ratio,
      overlap=False,
  )
  return csa_cache_layout.slabs_to_words(new_slabs) if is_words else new_slabs
