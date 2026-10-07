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
"""Shared correctness tests for compressor projection implementations."""

import dataclasses

from absl.testing import parameterized
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.proj_and_save_state import reference

# Numerics, measured on TPU7x. The state is an f32 matmul over `hidden_size`,
# and the kernel and the reference run it at the same default TPU matmul
# precision: both differ from a `Precision.HIGHEST` reference by about 1e-2,
# but from each other by at most 2e-6 (with states of magnitude up to about
# 8). They are not bit-exact because the kernel sums over `hidden_size` in
# `tile_k` chunks in its own order, so written rows are compared within the
# tolerance below, with a 5x margin. All other cache rows must be bit-exact.
ATOL = RTOL = 1e-5
# With bf16 inputs the reference rounds the projection and the APE sum to bf16,
# while the kernel accumulates them in f32.
BF16_ATOL = BF16_RTOL = 1e-2


@dataclasses.dataclass(frozen=True)
class Geometry:
  """A compressor's state and cache geometry.

  Attributes:
    state_width: Width of one state field, the head dimension doubled when
      compression windows overlap.
    compress_ratio: Number of APE rows.
    lanes: Lanes per cache row; a row holds `lanes` f32 values.
    page_size: Cache rows per page.
    words: Whether the cache is CSA's int32 word array rather than uint8 slabs.
  """

  state_width: int
  compress_ratio: int
  lanes: int
  page_size: int
  words: bool = False

  @property
  def rows_per_token(self) -> int:
    return 2 * self.state_width // self.lanes

  @property
  def tokens_per_page(self) -> int:
    return self.page_size // self.rows_per_token

  def cache_shape(self, num_pages: int) -> tuple[int, ...]:
    if self.words:
      return (num_pages, self.page_size, self.lanes)
    return (num_pages, self.page_size, 4, self.lanes)


# DeepSeek-V4's three compressors, with the page sizes of the upstream tests.
# CSA: head_dim 512 with overlapping windows, its state hosted in the int32 NoPE
# array.
CSA = Geometry(
    state_width=1024, compress_ratio=4, lanes=128, page_size=256, words=True
)
# CSA with the state in a uint8 slab array.
CSA_U8 = dataclasses.replace(CSA, words=False)
# HCA: head_dim 512, no overlap, compress ratio 128.
HCA = Geometry(state_width=512, compress_ratio=128, lanes=128, page_size=256)
# The CSA indexer: head_dim 128 with overlapping windows, 256-lane rows.
INDEXER = Geometry(state_width=256, compress_ratio=4, lanes=256, page_size=32)

GEOMETRIES = dict(csa=CSA, csa_u8=CSA_U8, hca=HCA, indexer=INDEXER)


def random_cache(key: jax.Array, geometry: Geometry, num_pages: int):
  """Returns a cache of random bytes, to catch writes outside the slots."""
  shape = geometry.cache_shape(num_pages)
  if geometry.words:
    bits = jax.random.bits(key, shape, dtype=jnp.uint32)
    return lax.bitcast_convert_type(bits, jnp.int32)
  return jax.random.bits(key, shape, dtype=jnp.uint8)


def make_inputs(
    geometry: Geometry,
    num_tokens: int,
    *,
    hidden_size: int = 4096,
    num_pages: int | None = None,
    num_padding: int = 0,
    dtype: jax.typing.DTypeLike = jnp.float32,
    seed: int = 0,
) -> tuple[jax.Array, ...]:
  """Returns random op inputs for `num_tokens` tokens.

  Every token gets a distinct token slot, drawn at random from the cache's
  pages, and the last `num_padding` tokens get slot -1, like the padding tokens
  of a serving batch.

  Args:
    geometry: The compressor geometry.
    num_tokens: Number of tokens.
    hidden_size: Hidden state width.
    num_pages: Number of cache pages. Defaults to enough pages for every token
      plus one spare page.
    num_padding: Number of trailing tokens to skip with slot -1.
    dtype: Dtype of the hidden states, weights and APE.
    seed: Random seed.

  Returns:
    `(hidden_states, wkv_wgate, ape, positions, slot_mapping, cache)`.
  """
  if num_pages is None:
    num_pages = -(-num_tokens // geometry.tokens_per_page) + 1
  keys = jax.random.split(jax.random.key(seed), 6)
  state_dim = 2 * geometry.state_width
  hidden_states = jax.random.normal(keys[0], (num_tokens, hidden_size), dtype)
  # Scaled so that a projected value is about N(0, 1).
  wkv_wgate = jax.random.normal(keys[1], (hidden_size, state_dim), dtype)
  wkv_wgate = (wkv_wgate / np.sqrt(hidden_size)).astype(dtype)
  ape = jax.random.normal(
      keys[2], (geometry.compress_ratio, geometry.state_width), dtype
  )
  positions = jax.random.randint(keys[3], (num_tokens,), 0, 1 << 16, jnp.int32)
  token_slots = jax.random.permutation(
      keys[4], num_pages * geometry.tokens_per_page
  )[:num_tokens]
  slot_mapping = (token_slots * geometry.rows_per_token).astype(jnp.int32)
  if num_padding:
    slot_mapping = slot_mapping.at[num_tokens - num_padding :].set(-1)
  cache = random_cache(keys[5], geometry, num_pages)
  return hidden_states, wkv_wgate, ape, positions, slot_mapping, cache


def _as_words(cache: jax.Array) -> np.ndarray:
  """Returns the cache as int32 `(num_pages, page_size, lanes)` words."""
  if cache.dtype == jnp.uint8:
    cache = reference.slabs_to_words(cache)
  return np.asarray(cache)


def assert_caches_match(
    actual: jax.Array,
    expected: jax.Array,
    slot_mapping: jax.Array,
    rows_per_token: int,
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
):
  """Checks the written rows within tolerance and every other row bit-exactly."""
  assert actual.shape == expected.shape, (actual.shape, expected.shape)
  assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
  actual_words, expected_words = _as_words(actual), _as_words(expected)
  num_pages, page_size, lanes = actual_words.shape
  actual_words = actual_words.reshape(num_pages * page_size, lanes)
  expected_words = expected_words.reshape(num_pages * page_size, lanes)

  written = np.zeros(num_pages * page_size, dtype=bool)
  for slot in np.asarray(slot_mapping):
    if slot >= 0:
      written[slot : slot + rows_per_token] = True

  np.testing.assert_array_equal(
      actual_words[~written], expected_words[~written]
  )
  np.testing.assert_allclose(
      actual_words[written].view(np.float32),
      expected_words[written].view(np.float32),
      atol=atol,
      rtol=rtol,
  )


class ProjAndSaveStateTestBase(parameterized.TestCase):
  """Correctness suite shared by all compressor projection implementations.

  Subclasses pass the implementation under test as `proj_fn`. Each token's state
  rows must match `reference.proj_and_save_state` within `ATOL` / `RTOL`, and
  all other cache rows must be left bit-exactly unchanged.
  """

  def __init__(self, *args, proj_fn):
    super().__init__(*args)
    self._proj_fn = proj_fn

  def _check(
      self,
      geometry: Geometry,
      num_tokens: int,
      *,
      atol: float = ATOL,
      rtol: float = RTOL,
      **kwargs,
  ):
    inputs = make_inputs(geometry, num_tokens, **kwargs)
    slot_mapping = inputs[4]
    expected = reference.proj_and_save_state(
        *inputs, compress_ratio=geometry.compress_ratio
    )
    actual = self._proj_fn(*inputs, compress_ratio=geometry.compress_ratio)
    assert_caches_match(
        actual,
        expected,
        slot_mapping,
        geometry.rows_per_token,
        atol=atol,
        rtol=rtol,
    )

  @parameterized.product(
      geometry=tuple(GEOMETRIES),
      # 13 and 200 are not multiples of the 8-row or 128-row token tile.
      num_tokens=(8, 13, 128, 200),
  )
  def test_correctness(self, geometry, num_tokens):
    """Checks every geometry for decode and prefill sized token counts."""
    self._check(GEOMETRIES[geometry], num_tokens)

  @parameterized.parameters(*GEOMETRIES)
  def test_long_prefill(self, geometry):
    """Checks many token tiles, the last one partial."""
    self._check(GEOMETRIES[geometry], 1000, num_padding=10)

  @parameterized.parameters(
      ("csa", 4096),
      ("csa", 7168),
      ("hca", 7168),
      ("indexer", 7168),
  )
  def test_hidden_size(self, geometry, hidden_size):
    """Checks DeepSeek-V4-Flash and full-size hidden widths."""
    self._check(GEOMETRIES[geometry], 72, hidden_size=hidden_size)

  @parameterized.parameters((16, 5), (200, 100), (8, 8))
  def test_padding_slots(self, num_tokens, num_padding):
    """Checks tokens with slot -1 are skipped, including all of them."""
    self._check(CSA, num_tokens, num_padding=num_padding)

  @parameterized.parameters(*GEOMETRIES)
  def test_full_cache(self, geometry):
    """Checks a cache with every slot written, so the first and last too."""
    geometry = GEOMETRIES[geometry]
    num_pages = 3
    self._check(
        geometry, num_pages * geometry.tokens_per_page, num_pages=num_pages
    )

  def test_bf16_inputs(self):
    """Checks bf16 hidden states, weights and APE."""
    self._check(CSA, 40, dtype=jnp.bfloat16, atol=BF16_ATOL, rtol=BF16_RTOL)
