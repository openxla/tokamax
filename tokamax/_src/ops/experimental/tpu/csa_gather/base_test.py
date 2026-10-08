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
"""Tests for the baseline JAX implementation of CSA Gather."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.compress_store import csa_cache_layout
from tokamax._src.ops.experimental.tpu.csa_gather import base
from tokamax._src.ops.experimental.tpu.csa_gather import test_base

jax.config.parse_flags_with_absl()


def _rope_values(rope_cache: np.ndarray, token: int) -> np.ndarray:
  """Returns the 64 bf16 RoPE values of `token` as uint16 bit patterns."""
  words = rope_cache.reshape(-1, csa_cache_layout.ROPE_WORDS)[token].view(
      np.uint32
  )
  return np.concatenate([words & 0xFFFF, words >> 16]).astype(np.uint16)


class BaseCsaGatherTest(test_base.CsaGatherTestBase):

  def __init__(self, *args):
    super().__init__(*args, gather_fn=base.CsaGather())

  @parameterized.parameters((256, 128), (512, 256), (1024, 256))
  def test_matches_layout_spec(self, num_indices, top_k):
    num_pages, page_size = 8, 64
    nope_key, rope_key, idx_key = jax.random.split(jax.random.key(0), 3)
    nope_cache = test_base.random_words(nope_key, (num_pages, page_size, 128))
    rope_cache = test_base.random_words(
        rope_key, (num_pages, page_size // 4, 128)
    )
    indices = jax.random.randint(
        idx_key, (num_indices,), 0, num_pages * page_size, jnp.int32
    )

    nope_out, rope_out = base.CsaGather()(
        nope_cache, rope_cache, indices, top_k=top_k
    )

    nope_cache, rope_cache = np.asarray(nope_cache), np.asarray(rope_cache)
    indices = np.asarray(indices)
    self.assertEqual(nope_out.shape, (num_indices, 128))
    self.assertEqual(rope_out.shape, (num_indices // 4, 128))
    np.testing.assert_array_equal(
        nope_out, nope_cache.reshape(-1, 128)[indices]
    )

    # bf16 rows 2r and 2r + 1 are the low and high halves of int32 row r.
    rope_bf16 = np.asarray(rope_out).view(np.uint16).reshape(-1, 128, 2)
    rope_bf16 = rope_bf16.transpose(0, 2, 1).reshape(-1, 128)
    half = top_k // 2
    for period in range(num_indices // top_k):
      for i in range(half):
        row = rope_bf16[period * half + i]
        lo = indices[period * top_k + i]
        hi = indices[period * top_k + half + i]
        np.testing.assert_array_equal(row[:64], _rope_values(rope_cache, lo))
        np.testing.assert_array_equal(row[64:], _rope_values(rope_cache, hi))

  def test_rejects_non_int32_caches(self):
    cache = jnp.zeros((2, 16, 128), jnp.int16)
    rope = jnp.zeros((2, 4, 128), jnp.int16)
    with self.assertRaisesRegex(ValueError, "int32"):
      base.CsaGather()(cache, rope, jnp.zeros((128,), jnp.int32), top_k=128)

  def test_rejects_bad_top_k(self):
    cache = jnp.zeros((2, 16, 128), jnp.int32)
    rope = jnp.zeros((2, 4, 128), jnp.int32)
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      base.CsaGather()(cache, rope, jnp.zeros((192,), jnp.int32), top_k=192)

  def test_rejects_ragged_num_indices(self):
    cache = jnp.zeros((2, 16, 128), jnp.int32)
    rope = jnp.zeros((2, 4, 128), jnp.int32)
    with self.assertRaisesRegex(ValueError, "multiple of top_k"):
      base.CsaGather()(cache, rope, jnp.zeros((192,), jnp.int32), top_k=128)


if __name__ == "__main__":
  absltest.main()
