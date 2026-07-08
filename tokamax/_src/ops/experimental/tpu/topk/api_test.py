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
"""Tests for TopK API."""

from absl.testing import absltest
from absl.testing import parameterized
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.topk import api


# TODO: add more robust tests for the API.
class ApiTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if backend.get_default_device().device_kind != "TPU7x":
      self.skipTest("Only tested on TPU7x.")

  def test_topk_api_xla(self):
    keys = jnp.array([[3.0, 1.0, 4.0, 2.0], [5.0, 9.0, 2.0, 6.0]])
    k = 2
    res_k, res_v = api.top_k(keys, k, implementation="xla")
    np.testing.assert_allclose(res_k, np.array([[4.0, 3.0], [9.0, 6.0]]))
    np.testing.assert_array_equal(res_v, np.array([[2, 0], [1, 3]]))

  def test_topk_api_mosaic_tpu(self):
    keys = jnp.array([
        [1.0, 5.0, 2.0, 7.0, 3.0, 9.0, 4.0, 8.0, 0.0, 6.0, 11.0, 15.0, 12.0, 14.0, 13.0, 10.0],
        [15.0, 11.0, 14.0, 12.0, 13.0, 10.0, 9.0, 5.0, 8.0, 7.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    ])
    k = 4
    res_k, res_v = api.top_k(keys, k, implementation="mosaic_tpu")
    ref_k, ref_v = api.top_k(keys, k, implementation="xla")
    np.testing.assert_allclose(res_k, ref_k)
    np.testing.assert_array_equal(res_v, ref_v)


if __name__ == "__main__":
  absltest.main()
