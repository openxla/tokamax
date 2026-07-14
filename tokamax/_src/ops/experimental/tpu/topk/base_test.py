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
"""Tests for base TopK operator."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.topk import base

# TODO: Add more tests.


class BaseTopKTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only tested on TPU.")

  def test_base_topk(self):
    op = base.TopK()
    keys = jnp.array([[3.0, 1.0, 4.0, 2.0], [5.0, 9.0, 2.0, 6.0]])
    k = 2
    topk_keys, topk_vals = op(keys, k)
    np.testing.assert_allclose(topk_keys, np.array([[4.0, 3.0], [9.0, 6.0]]))
    np.testing.assert_array_equal(topk_vals, np.array([[2, 0], [1, 3]]))

  def test_base_topk_with_values(self):
    op = base.TopK()
    keys = jnp.array([[3.0, 1.0, 4.0, 2.0]])
    values = jnp.array([[10, 20, 30, 40]], dtype=jnp.int32)
    k = 2
    topk_keys, topk_vals = op(keys, k, values)
    np.testing.assert_allclose(topk_keys, np.array([[4.0, 3.0]]))
    np.testing.assert_array_equal(topk_vals, np.array([[30, 10]]))


if __name__ == "__main__":
  absltest.main()
