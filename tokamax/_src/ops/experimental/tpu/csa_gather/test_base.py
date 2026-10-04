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
"""Shared correctness tests for CSA Gather implementations."""

import functools

from absl.testing import parameterized
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.csa_gather import reference

_NUM_PAGES = 1000
_PAGE_SIZE = 256


def random_words(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
  """Returns random int32 words covering the full 32-bit range."""
  bits = jax.random.bits(key, shape, dtype=jnp.uint32)
  return lax.bitcast_convert_type(bits, jnp.int32)


@functools.cache
def make_inputs(num_indices: int) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns random `(nope_cache, rope_cache, indices)` for the test suite."""
  nope_key, rope_key, idx_key = jax.random.split(jax.random.key(0), 3)
  nope_cache = random_words(nope_key, (_NUM_PAGES, _PAGE_SIZE, 128))
  rope_cache = random_words(rope_key, (_NUM_PAGES, _PAGE_SIZE // 4, 128))
  indices = jax.random.randint(
      idx_key, (num_indices,), 0, _NUM_PAGES * _PAGE_SIZE, dtype=jnp.int32
  )
  return nope_cache, rope_cache, indices


class CsaGatherTestBase(parameterized.TestCase):
  """Correctness suite shared by all CSA Gather implementations.

  Subclasses pass the implementation under test as `gather_fn`. Results must be
  bit-exact against `reference.csa_gather` over the valid prefix of indices.
  """

  def __init__(self, *args, gather_fn):
    super().__init__(*args)
    self._gather_fn = gather_fn

  @parameterized.parameters(
      (256, 256),
      (4096, 256),
      (4096, 512),
      (128 * 1024, 1024),
      (128 * 1024, 2048),
      # Not a whole number of blocks: exercises the index padding.
      (25 * 512, 512),
  )
  def test_correctness(self, num_indices, top_k):
    """Checks both outputs are bit-exact against the reference."""
    nope_cache, rope_cache, indices = make_inputs(num_indices)

    nope_ref, rope_ref = reference.csa_gather(
        nope_cache, rope_cache, indices, top_k=top_k
    )
    nope_out, rope_out = self._gather_fn(
        nope_cache, rope_cache, indices, top_k=top_k
    )

    np.testing.assert_array_equal(nope_out, nope_ref)
    np.testing.assert_array_equal(rope_out, rope_ref)

  @parameterized.parameters(
      (4096, 1024),
      (128 * 1024, 16 * 1024),
      (128 * 1024, 1024),
  )
  def test_num_valid_indices(self, num_indices, num_valid):
    """Checks the valid prefix matches when trailing indices are skipped."""
    top_k = 1024
    nope_cache, rope_cache, indices = make_inputs(num_indices)

    nope_ref, rope_ref = reference.csa_gather(
        nope_cache, rope_cache, indices[:num_valid], top_k=top_k
    )
    nope_out, rope_out = self._gather_fn(
        nope_cache,
        rope_cache,
        indices,
        jnp.array([num_valid], jnp.int32),
        top_k=top_k,
    )

    np.testing.assert_array_equal(nope_out[:num_valid], nope_ref)
    np.testing.assert_array_equal(rope_out[: num_valid // 4], rope_ref)
