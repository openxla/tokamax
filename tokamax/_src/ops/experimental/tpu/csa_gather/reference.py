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
"""Pure JAX reference implementation of the DeepSeek-V4 CSA cache gather.

A CSA layer keeps its compressed KV cache as two int32 arrays with 128 lanes.
For a page of `T` compressed tokens:

  NoPE  `int32[num_pages, T, 128]`: one row per token, the `(4, 128)` uint8
        slab packed by the compressor, byte `b` of word `w` holding slab byte
        `(b, w)`.
  RoPE  `int32[num_pages, T // 4, 128]`: 4 tokens per row, token `t` owning
        words `[32 (t % 4), 32 (t % 4) + 32)` of row `t // 4`. Word `k` of a
        token holds bf16 value `k` in its low half and bf16 value `k + 32` in
        its high half.

`csa_gather` gathers the NoPE row and the RoPE words of every requested token
and lays them out for the sparse MLA kernel that consumes them.
"""

import functools

import jax
from jax import lax
import jax.numpy as jnp

ROW_WORDS = 128  # int32 words per cache row.
ROPE_WORDS = 32  # int32 RoPE words per token.
ROPE_DIM = 2 * ROPE_WORDS  # bf16 RoPE values per token.


@functools.partial(jax.jit, static_argnames=("top_k",))
def csa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    num_valid_indices: jax.Array | None = None,
    *,
    top_k: int = 1024,
) -> tuple[jax.Array, jax.Array]:
  """Pure JAX reference implementation of the CSA cache gather.

  Args:
    nope_cache: `(num_pages, page_size, 128)` int32 NoPE cache.
    rope_cache: `(num_pages, page_size // 4, 128)` int32 RoPE cache.
    indices: `(N,)` int32 token indices into the caches. `N` is a multiple of
      `top_k`.
    num_valid_indices: Optional number of valid leading indices. Output rows for
      the remaining indices are unspecified; this reference gathers them anyway.
    top_k: The consumer's row block (the attention kernel's top-k), a multiple
      of 128.

  Returns:
    A tuple `(nope_out, rope_out)` of int32 arrays with 128 lanes.

    - `nope_out`: `(N, 128)`. Row `i` is the NoPE row of token `indices[i]`.
    - `rope_out`: `(N // 4, 128)`, i.e. `(N // 2, 128)` bf16 once bitcast. For
      `i < top_k // 2`, bf16 row `p * (top_k // 2) + i` holds entry `i` of
      period `p` in lanes `0:64` and entry `i + top_k // 2` in lanes `64:128`.
      bf16 rows `2r` and `2r + 1` are the low and high halves of int32 row `r`.
  """
  del num_valid_indices  # Rows past the valid prefix are unspecified.
  nope_out = nope_cache.reshape(-1, ROW_WORDS)[indices]

  # Bit-level throughout: arbitrary cache words are NaN-laden as bf16.
  words = lax.bitcast_convert_type(
      rope_cache.reshape(-1, ROPE_WORDS)[indices], jnp.uint32
  )
  rope = jnp.concatenate([words & 0xFFFF, words >> 16], axis=-1)  # (N, 64)
  rope = rope.reshape(-1, 2, top_k // 2, ROPE_DIM)
  rows = jnp.concatenate([rope[:, 0], rope[:, 1]], axis=-1)
  rows = rows.reshape(-1, 2, ROW_WORDS)
  rope_out = lax.bitcast_convert_type(
      rows[:, 0] | (rows[:, 1] << 16), jnp.int32
  )
  return nope_out, rope_out
