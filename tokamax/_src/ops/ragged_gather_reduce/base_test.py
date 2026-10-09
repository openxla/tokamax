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
"""Tests for the baseline JAX implementation of Ragged Gather Reduce."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_gather_reduce import base
from tokamax._src.ops.ragged_gather_reduce import test_base

jax.config.parse_flags_with_absl()


class BaseRaggedGatherReduceTest(test_base.RaggedGatherReduceTestBase):

  def __init__(self, *args):
    super().__init__(*args, gather_fn=base.RaggedGatherReduce())

  @parameterized.product(
      reduce_group_size=[1, 2, 4, 8],
      valid_mode=test_base.VALID_MODES,
      dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_matches_loop(self, reduce_group_size, valid_mode, dtype):
    """Checks the op against a direct loop over tokens and routes."""
    if valid_mode == "two_per_token" and reduce_group_size < 2:
      self.skipTest("Needs at least two routes per token.")
    if valid_mode == "uneven_pairs" and reduce_group_size < 3:
      self.skipTest("Needs at least three routes per token.")
    num_tokens, num_rows, hidden_size = 24, 37, 128
    input_size = num_tokens * reduce_group_size
    x_key, idx_key, w_key, valid_key = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(x_key, (num_rows, hidden_size), dtype)
    indices = jax.random.randint(idx_key, (input_size,), 0, num_rows)
    topk_weights = jax.random.normal(w_key, (input_size,), dtype)
    valid_rows_mask = test_base.make_valid_mask(
        valid_key, num_tokens, reduce_group_size, valid_mode
    )

    out = base.RaggedGatherReduce()(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )

    x_np = np.asarray(x, np.float32)
    w_np = np.asarray(topk_weights, np.float32)
    idx_np = np.asarray(indices)
    valid_np = np.asarray(valid_rows_mask)
    expected = np.zeros((num_tokens, hidden_size), np.float32)
    for t in range(num_tokens):
      for r in range(t * reduce_group_size, (t + 1) * reduce_group_size):
        if valid_np[r]:
          expected[t] += w_np[r] * x_np[idx_np[r]]
    self.assertEqual(out.dtype, dtype)
    tol = 1e-5 if dtype == jnp.float32 else test_base.ATOL
    np.testing.assert_allclose(
        np.asarray(out, np.float32), expected, atol=tol, rtol=tol
    )

  def test_rejects_ragged_routes(self):
    x = jnp.zeros((16, 128), jnp.bfloat16)
    routes = jnp.zeros((12,), jnp.int32)
    with self.assertRaisesRegex(ValueError, "multiple of reduce_group_size"):
      base.RaggedGatherReduce()(
          x, routes, routes.astype(x.dtype), routes > 0, reduce_group_size=8
      )

  def test_rejects_non_positive_reduce_group_size(self):
    x = jnp.zeros((16, 128), jnp.bfloat16)
    routes = jnp.zeros((16,), jnp.int32)
    with self.assertRaisesRegex(ValueError, "must be positive"):
      base.RaggedGatherReduce()(
          x, routes, routes.astype(x.dtype), routes > 0, reduce_group_size=0
      )


if __name__ == "__main__":
  absltest.main()
