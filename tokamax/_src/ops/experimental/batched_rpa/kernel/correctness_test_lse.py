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
"""Correctness test for the LSE output of batched RPA.

Checks the returned LSE and output against a NumPy reference for multiple KV
heads, small and non-power-of-two Q-head groups, and unequal request lengths,
under both KV layouts.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tokamax._src.ops.experimental.batched_rpa.kernel import configs
from tokamax._src.ops.experimental.batched_rpa.kernel import wrapper

jax.config.parse_flags_with_absl()


class LseCorrectnessTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update("jax_numpy_dtype_promotion", "standard")
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    try:
      if not pltpu.get_tpu_info().generation >= 4:
        self.skipTest("Expect TPUv4+")
    except (ValueError, RuntimeError, AttributeError):
      self.skipTest("Failed to get TPU info.")

  @parameterized.product(
      kv_layout=list(configs.KVLayout),
      q_heads_per_kv=[3, 10],
  )
  def test_return_lse_preserves_gqa_heads(self, kv_layout, q_heads_per_kv):
    """LSE writeback preserves GQA head mapping and logical output shapes.

    Exercise multiple KV heads, small and non-power-of-two Q groups, and
    unequal request lengths through both decode and prefill scheduling.
    """
    rng = np.random.default_rng(42)
    q_lens = (1, 3, 2)
    starts = np.cumsum((0, *q_lens)).astype(np.int32)
    num_kv_heads, head_dim = 2, 128
    num_q_heads = num_kv_heads * q_heads_per_kv
    num_tokens = sum(q_lens)

    def random_array(shape, dtype):
      return jnp.asarray(rng.normal(0, 0.5, shape), dtype)

    query = random_array((num_tokens, num_q_heads, head_dim), jnp.bfloat16)
    key = random_array((num_tokens, num_kv_heads, head_dim), jnp.float8_e4m3fn)
    value = random_array(
        (num_tokens, num_kv_heads, head_dim), jnp.float8_e4m3fn
    )
    # Reserve two pages per request for the default 256-token KV block.
    shape = wrapper.get_kv_cache_shape(
        6,
        128,
        num_kv_heads,
        head_dim,
        key.dtype,
        kv_layout=kv_layout,
    )

    def run(return_lse):
      return wrapper.ragged_paged_attention(
          query,
          key,
          value,
          jnp.zeros(shape, key.dtype),
          jnp.asarray(q_lens, jnp.int32),
          jnp.array([4, 0, 2, 5, 1, 3], jnp.int32),
          jnp.asarray(starts),
          jnp.array([1, 1, 3], jnp.int32),
          sm_scale=head_dim**-0.5,
          kv_layout=kv_layout,
          return_lse=return_lse,
      )

    out_without_lse, cache_without_lse = run(False)
    output, cache, lse = run(True)
    np.testing.assert_allclose(
        np.asarray(output, np.float32),
        np.asarray(out_without_lse, np.float32),
        atol=3e-3,
        rtol=1e-2,
    )
    np.testing.assert_array_equal(
        np.asarray(cache, np.float32),
        np.asarray(cache_without_lse, np.float32),
    )

    queries = np.asarray(query, np.float32)
    keys = np.repeat(np.asarray(key, np.float32), q_heads_per_kv, axis=1)
    values = np.repeat(np.asarray(value, np.float32), q_heads_per_kv, axis=1)
    expected_out, expected_lse = [], []
    for start, end in zip(starts[:-1], starts[1:]):
      for pos in range(start, end):
        scores = (
            np.einsum("hd,thd->ht", queries[pos], keys[start : pos + 1])
            * head_dim**-0.5
        )
        maximum = scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores - maximum)
        denominator = weights.sum(axis=-1, keepdims=True)
        expected_out.append(
            np.einsum(
                "ht,thd->hd", weights / denominator, values[start : pos + 1]
            )
        )
        expected_lse.append((maximum + np.log(denominator))[:, 0])

    self.assertEqual(output.shape, query.shape)
    self.assertEqual(lse.shape, query.shape[:2])
    np.testing.assert_allclose(
        np.asarray(output, np.float32),
        expected_out,
        atol=3e-3,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        np.asarray(lse, np.float32),
        expected_lse,
        atol=4e-2,
        rtol=1e-2,
    )


if __name__ == "__main__":
  absltest.main()
