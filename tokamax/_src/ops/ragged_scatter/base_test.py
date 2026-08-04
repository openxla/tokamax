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
"""Tests for the baseline JAX implementation of Ragged Scatter."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_scatter import base

jax.config.parse_flags_with_absl()


class BaseRaggedScatterTest(absltest.TestCase):

  def test_base_running_correctly(self):
    out_size = 512
    hidden_size = 128
    start = 10
    end = 400

    key = jax.random.key(0)
    x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
    indices = jax.random.permutation(key, out_size)
    start_arr = jnp.array([start], jnp.int32)
    end_arr = jnp.array([end], jnp.int32)

    op = base.RaggedScatter()
    actual = op(x, indices, start_arr, end_arr)
    desired = base.ragged_scatter(x, indices, start_arr, end_arr)

    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
