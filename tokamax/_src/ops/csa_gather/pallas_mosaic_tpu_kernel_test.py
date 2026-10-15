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
"""Hypothesis tests for the CSA Gather SparseCore Pallas kernel on TPU.

These sweep `top_k`, `num_streams`, ragged period counts (which exercise the
index padding), cache shapes, index patterns and the optional
`num_valid_indices` prefix through the raw kernel entry point the Pallas op
calls, checking it bit-exact against the reference.
"""

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.csa_gather import pallas_mosaic_tpu_kernel
from tokamax._src.ops.csa_gather import reference
from tokamax._src.ops.csa_gather import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


hp.settings.register_profile(
    name="deterministic",
    database=None,
    derandomize=True,
    deadline=None,
    max_examples=10,
    print_blob=True,
    verbosity=hp.Verbosity.verbose,
)
hp.settings.load_profile(name="deterministic")

_INDEX_PATTERNS = ("random", "constant", "arange", "reversed", "extremes")


def _make_indices(
    pattern: str, key: jax.Array, num_indices: int, num_tokens: int
) -> jax.Array:
  """Returns `(num_indices,)` int32 token indices in `[0, num_tokens)`."""
  match pattern:
    case "random":
      return jax.random.randint(key, (num_indices,), 0, num_tokens, jnp.int32)
    case "constant":
      # Every entry gathers the same token: all DMAs hit one cache row.
      token = jax.random.randint(key, (), 0, num_tokens, jnp.int32)
      return jnp.full((num_indices,), token, jnp.int32)
    case "arange":
      return jnp.arange(num_indices, dtype=jnp.int32) % num_tokens
    case "reversed":
      arange = jnp.arange(num_indices, dtype=jnp.int32) % num_tokens
      return num_tokens - 1 - arange
    case "extremes":
      # Only the first and last tokens of the caches.
      bits = jax.random.bernoulli(key, shape=(num_indices,))
      return jnp.where(bits, num_tokens - 1, 0).astype(jnp.int32)
    case _:
      raise ValueError(f"Unknown index pattern: {pattern}")


class PallasMosaicTpuKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Matches `pallas_mosaic_tpu_test`: the kernel is only validated on TPU7x.
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

  @hp.given(hps.data())
  def test_kernel_matches_reference(self, data):
    """Checks the raw kernel is bit-exact against the reference."""
    num_lanes = pltpu.get_tpu_info().sparse_core.num_lanes
    top_k = data.draw(
        hps.sampled_from([128, 256, 512, 1024, 2048]), label="top_k"
    )
    # The kernel requires `top_k // 2` to be a multiple of
    # `num_streams * num_lanes`, and `num_streams <= MAX_NUM_STREAMS`.
    num_streams = data.draw(
        hps.sampled_from([
            n
            for n in range(1, pallas_mosaic_tpu_kernel.MAX_NUM_STREAMS + 1)
            if (top_k // 2) % (n * num_lanes) == 0
        ]),
        label="num_streams",
    )
    # Period counts that are not a multiple of the subcore count exercise the
    # index padding.
    num_periods = data.draw(hps.integers(1, 40), label="num_periods")
    num_indices = num_periods * top_k
    pass_num_valid = data.draw(hps.booleans(), label="pass_num_valid")
    num_valid = (
        data.draw(hps.integers(1, num_periods), label="num_valid_periods")
        * top_k
    )
    scalar_num_valid = data.draw(hps.booleans(), label="scalar_num_valid")
    num_pages = data.draw(hps.integers(1, 64), label="num_pages")
    # The RoPE cache packs 4 tokens per row, so `page_size % 4 == 0`.
    page_size = data.draw(hps.sampled_from([4, 64, 256]), label="page_size")
    pattern = data.draw(hps.sampled_from(_INDEX_PATTERNS), label="pattern")
    seed = data.draw(hps.integers(0, 2**31 - 1), label="seed")

    nope_key, rope_key, idx_key = jax.random.split(jax.random.key(seed), 3)
    nope_cache = test_base.random_words(nope_key, (num_pages, page_size, 128))
    rope_cache = test_base.random_words(
        rope_key, (num_pages, page_size // 4, 128)
    )
    indices = _make_indices(
        pattern, idx_key, num_indices, num_pages * page_size
    )

    if not pass_num_valid:
      num_valid_arg = None
    elif scalar_num_valid:
      num_valid_arg = jnp.array(num_valid, jnp.int32)
    else:
      num_valid_arg = jnp.array([num_valid], jnp.int32)
    nope_out, rope_out = pallas_mosaic_tpu_kernel.csa_gather(
        nope_cache,
        rope_cache,
        indices,
        num_valid_arg,
        top_k=top_k,
        num_streams=num_streams,
    )
    nope_ref, rope_ref = reference.csa_gather(
        nope_cache, rope_cache, indices, top_k=top_k
    )

    self.assertEqual(nope_out.shape, (num_indices, 128))
    self.assertEqual(rope_out.shape, (num_indices // 4, 128))
    self.assertEqual(nope_out.dtype, jnp.int32)
    self.assertEqual(rope_out.dtype, jnp.int32)
    # Rows past the valid prefix are left unwritten by the kernel.
    n = num_valid if pass_num_valid else num_indices
    np.testing.assert_array_equal(nope_out[:n], nope_ref[:n])
    np.testing.assert_array_equal(rope_out[: n // 4], rope_ref[: n // 4])

  @parameterized.parameters(0, pallas_mosaic_tpu_kernel.MAX_NUM_STREAMS + 1, 8)
  def test_kernel_rejects_invalid_num_streams(self, num_streams):
    cache = jnp.zeros((2, 64, 128), jnp.int32)
    rope = jnp.zeros((2, 16, 128), jnp.int32)
    with self.assertRaisesRegex(ValueError, "num_streams must be in"):
      pallas_mosaic_tpu_kernel.csa_gather(
          cache,
          rope,
          jnp.zeros((1024,), jnp.int32),
          top_k=1024,
          num_streams=num_streams,
      )

  def test_kernel_rejects_top_k_not_divisible_by_streams_and_lanes(self):
    num_lanes = pltpu.get_tpu_info().sparse_core.num_lanes
    top_k = num_lanes  # `top_k // 2 < num_lanes`, so no stream count fits.
    cache = jnp.zeros((2, 64, 128), jnp.int32)
    rope = jnp.zeros((2, 16, 128), jnp.int32)
    with self.assertRaisesRegex(
        ValueError, "must be a multiple of num_streams"
    ):
      pallas_mosaic_tpu_kernel.csa_gather(
          cache,
          rope,
          jnp.zeros((8 * top_k,), jnp.int32),
          top_k=top_k,
          num_streams=1,
      )


if __name__ == "__main__":
  absltest.main()
