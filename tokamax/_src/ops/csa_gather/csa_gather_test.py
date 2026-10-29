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
"""The upstream test suite of the CSA Gather SparseCore kernel, ported.

The tests are kept as they are upstream: same pure-JAX oracle, cache shapes,
seeds and named cases, all checked bit-exact, calling the raw kernel with its
default `num_streams`. Upstream's `csa_cache_layout` constants come from
`dsv4_compressor.csa_cache_layout`. The kernel needs a TPU SparseCore, so
unlike upstream the suite skips on other backends.
"""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.csa_gather import pallas_mosaic_tpu_kernel
from tokamax._src.ops.dsv4_compressor import csa_cache_layout

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


csa_gather = pallas_mosaic_tpu_kernel.csa_gather


def gather_nope_ref_impl(cache, indices):
  return cache.reshape(-1, csa_cache_layout.ROW_WORDS)[indices]


def gather_rope_ref_impl(cache, indices, top_k=1024):
  # Bit-level throughout: random words are NaN-laden as bf16.
  words = lax.bitcast_convert_type(
      cache.reshape(-1, csa_cache_layout.ROPE_WORDS)[indices], jnp.uint32
  )
  rope = jnp.concatenate([words & 0xFFFF, words >> 16], axis=-1)  # (n, 64)
  rope = rope.reshape(-1, 2, top_k // 2, 64)
  rows = jnp.concatenate([rope[:, 0], rope[:, 1]], axis=-1).reshape(-1, 2, 128)
  # bf16 rows 2t and 2t + 1 are the low and high halves of int32 row t.
  return lax.bitcast_convert_type(rows[:, 0] | (rows[:, 1] << 16), jnp.int32)


@functools.partial(jax.jit, static_argnames=("top_k",))
def gather_ref_impl(nope_cache, rope_cache, indices, top_k=1024):
  nope_out = gather_nope_ref_impl(nope_cache, indices)
  rope_out = gather_rope_ref_impl(rope_cache, indices, top_k=top_k)
  return nope_out, rope_out


def _random_words(key, shape):
  bits = jax.random.bits(key, shape, dtype=jnp.uint32)
  return lax.bitcast_convert_type(bits, jnp.int32)


@jax.jit
def create_nope_cache():
  key = jax.random.key(0)
  key, perm_key, cache_key = jax.random.split(key, 3)
  num_pages = 1000
  page_size = 256
  cache = _random_words(cache_key, (num_pages, page_size, 128))
  return cache, perm_key


@jax.jit
def create_rope_cache():
  key = jax.random.key(41)
  key, perm_key, cache_key = jax.random.split(key, 3)
  num_pages = 1000
  page_size = 256
  cache = _random_words(cache_key, (num_pages, page_size // 4, 128))
  return cache, perm_key


class GatherTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Not upstream: the kernel needs a TPU SparseCore and, like the rest of the
    # op's tests, is only validated on TPU7x.
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

  @parameterized.parameters(
      (256, 256),
      (4096, 256),
      (4096, 512),
      (128 * 1024, 1024),
      (128 * 1024, 2048),
      # Not a whole number of atom batches: exercises the padding.
      (25 * 512, 512),
  )
  def test_correctness(self, n, top_k):
    nope_cache, perm_key = create_nope_cache()
    rope_cache, _ = create_rope_cache()

    max_index = nope_cache.shape[0] * nope_cache.shape[1]
    indices = jax.random.randint(perm_key, (n,), 0, max_index, dtype=jnp.int32)

    nope_ref, rope_ref = gather_ref_impl(
        nope_cache, rope_cache, indices, top_k=top_k
    )
    nope_sc, rope_sc = csa_gather(nope_cache, rope_cache, indices, top_k=top_k)

    np.testing.assert_array_equal(nope_ref, nope_sc)
    np.testing.assert_array_equal(rope_ref, rope_sc)

  @parameterized.parameters(
      (4096, 1024),
      (128 * 1024, 16 * 1024),
      (128 * 1024, 1024),
  )
  def test_num_valid_indices(self, n, num_valid):
    nope_cache, perm_key = create_nope_cache()
    rope_cache, _ = create_rope_cache()

    max_index = nope_cache.shape[0] * nope_cache.shape[1]
    indices = jax.random.randint(perm_key, (n,), 0, max_index, dtype=jnp.int32)

    nope_ref, rope_ref = gather_ref_impl(
        nope_cache, rope_cache, indices[:num_valid]
    )
    nope_sc, rope_sc = csa_gather(
        nope_cache, rope_cache, indices, num_valid_indices=num_valid
    )

    np.testing.assert_array_equal(nope_ref, nope_sc[:num_valid])
    np.testing.assert_array_equal(rope_ref, rope_sc[: num_valid // 4])


if __name__ == "__main__":
  absltest.main()
