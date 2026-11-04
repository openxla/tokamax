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
"""Tests for the baseline JAX implementation of Ragged Gather."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_gather import base

jax.config.parse_flags_with_absl()


class BaseRaggedGatherTest(absltest.TestCase):

  def test_base_running_correctly(self):
    in_size = 512
    out_size = 400
    hidden_size = 128
    start = 3
    end = 338
    x = jax.random.normal(
        jax.random.PRNGKey(0), (in_size, hidden_size), jnp.float32
    )
    indices = jax.random.randint(
        jax.random.PRNGKey(1), (out_size,), 0, in_size, jnp.int32
    )
    start_arr = jnp.array([start], jnp.int32)
    end_arr = jnp.array([end], jnp.int32)

    op = base.RaggedGather()

    @jax.jit
    def run_op(x, indices, start_arr, end_arr):
      return op(x, indices, start_arr, end_arr) + 1.0

    actual = run_op(x, indices, start_arr, end_arr)
    desired = np.asarray(x)[np.asarray(indices)] + 1.0
    np.testing.assert_allclose(
        actual[start:end], desired[start:end], rtol=1e-5, atol=1e-5
    )


if __name__ == "__main__":
  absltest.main()
