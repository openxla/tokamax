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
"""DeepSeek-V4 CSA compressed-KV cache layout: SparseCore-native NoPE + RoPE.

A CSA layer keeps two int32 arrays with 128 lanes, for a page of `T`
compressed tokens:

  NoPE  `int32[num_pages, T, 128]`: one row per token, the `(4, 128)`
        uint8 slab the compressor packs (448 fp8 + lane-periodic e8m0 scales
        + padding), byte `b` of word `w` holding slab byte `(b, w)`.
  RoPE  `int32[num_pages, T // 4, 128]`: 4 tokens per row, token `t`
        owning words `[32 (t % 4), 32 (t % 4) + 32)` of row `t // 4`.
        Word `k` of a token holds bf16 `k` in its low half and bf16
        `k + 32` in its high half.

The SparseCore CSA gather fetches a token's 512-byte NoPE row and its own 128
RoPE bytes, where a TensorCore-tiled layout makes it fetch the whole 512-byte
RoPE row. A 32-word request is only expressible on a linear, untiled array
(`use_tc_tiling_on_sc=False`), which XLA hands over copy-free only for a 32-bit
array with 128 lanes -- a uint8 declaration makes XLA relayout the whole cache
on every call.

The NoPE array also hosts the SWA cache and the compressor states, and the
compressor writes both arrays; those TensorCore kernels address them as
`uint8[num_pages, rows, 4, 128]` slabs. Both declarations are byte-identical
(`T(8,128)` int32 vs `T(4,128)(4,1)` uint8), so the kernels take the int32
arrays and view them through `as_u8_slabs`, a free in-kernel bitcast. In that
view a RoPE token `j` of a row is lanes `[32 j, 32 j + 32)` of all 4 sub-rows,
not a sub-row of its own.
"""

import jax
from jax import lax
import jax.numpy as jnp

ROW_WORDS = 128
SLAB_ROWS = 4  # uint8 sub-rows per 128-word row
ROPE_WORDS = 32  # RoPE words per token
ROPE_DIM = 2 * ROPE_WORDS  # bf16 values per token
ROPE_TOKENS_PER_ROW = ROW_WORDS // ROPE_WORDS


def is_word_array(dtype) -> bool:
  """Whether a cache is declared as 32-bit words rather than uint8 slabs."""
  return jnp.dtype(dtype).itemsize == 4


def u8_slab_shape(shape: tuple[int, ...], dtype) -> tuple[int, ...]:
  """The `(num_pages, rows, 4, lanes)` uint8 view of a cache array."""
  if not is_word_array(dtype):
    return tuple(shape)
  num_pages, rows, lanes = shape
  return (num_pages, rows, SLAB_ROWS, lanes)


def as_u8_slabs(ref):
  """View a cache ref as `uint8[num_pages, rows, 4, lanes]` slabs.

  For an int32 `(num_pages, rows, lanes)` ref this is a free bitcast:
  Mosaic packs 4 consecutive uint8 sub-rows into one 32-bit row, little
  endian, which is exactly how XLA lays out the uint8 slab array. uint8 refs
  pass through unchanged.

  Args:
    ref: The cache ref, int32 `(num_pages, rows, lanes)` or uint8 slabs.

  Returns:
    The `uint8[num_pages, rows, 4, lanes]` view of `ref`.
  """
  if not is_word_array(ref.dtype):
    return ref
  num_pages, rows, lanes = ref.shape
  return ref.bitcast(jnp.uint8).reshape(num_pages, rows, SLAB_ROWS, lanes)


# Host-side converters between the int32 words and their uint8-slab / bf16
# forms. Not on the serving path: the kernels view the int32 arrays through
# `as_u8_slabs` instead, and only `reference.py` and the tests call the
# functions below.


def slabs_to_words(x: jax.Array) -> jax.Array:
  """uint8 `[..., 4, n]` -> int32 `[..., n]`; byte b from sub-row b."""
  x = x.astype(jnp.uint32)
  words = x[..., 0, :] | (x[..., 1, :] << 8) | (x[..., 2, :] << 16)
  words = words | (x[..., 3, :] << 24)
  return lax.bitcast_convert_type(words, jnp.int32)


def words_to_slabs(w: jax.Array) -> jax.Array:
  """int32 `[..., n]` -> uint8 `[..., 4, n]`; inverse of the above."""
  w = lax.bitcast_convert_type(w, jnp.uint32)
  return jnp.stack(
      [((w >> (8 * b)) & 0xFF).astype(jnp.uint8) for b in range(SLAB_ROWS)],
      axis=-2,
  )


def rope_words(rope: jax.Array) -> jax.Array:
  """bf16 `[..., 64]` -> int32 `[..., 32]`: word k = bf16 k | bf16 k+32."""
  bits = lax.bitcast_convert_type(rope, jnp.uint16).astype(jnp.uint32)
  return lax.bitcast_convert_type(
      bits[..., :ROPE_WORDS] | (bits[..., ROPE_WORDS:] << 16), jnp.int32
  )


def encode_rope(rope: jax.Array) -> jax.Array:
  """bf16 `[num_tokens, 64]` -> the int32 `[num_tokens // 4, 128]` rows."""
  return rope_words(rope).reshape(-1, ROW_WORDS)
