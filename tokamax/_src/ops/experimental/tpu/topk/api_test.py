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
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src.ops.experimental.tpu.topk import api
from tokamax._src.ops.experimental.tpu.topk import test_base


class ApiTest(parameterized.TestCase):

  def test_topk_api_xla(self):
    scores = jnp.array([[3.0, 1.0, 4.0, 2.0], [5.0, 9.0, 2.0, 6.0]])
    k = 2

    @jax.jit
    def f(scores):
      return api.top_k(scores, k, implementation="xla")

    res_idx = f(scores)
    np.testing.assert_array_equal(res_idx, np.array([[2, 0], [1, 3]]))

    with self.subTest("correct_implementation_used"):
      opspecs = hlo_utils.get_opspecs(
          f.lower(scores), include_xla_kernels=False
      )
      self.assertEmpty(opspecs)

  def test_topk_api_mosaic_tpu(self):
    if backend.get_default_device().device_kind != "TPU7x":
      self.skipTest("Only tested on TPU7x.")

    rng = np.random.default_rng(0)
    scores_np = rng.standard_normal((32, 2048), dtype=np.float32)
    scores = jnp.asarray(scores_np)
    k = 16

    @jax.jit
    def f(scores):
      return api.top_k(scores, k, implementation="mosaic_tpu")

    res_idx = f(scores)
    test_base.assert_topk_matches_reference(scores_np, res_idx, k)

    with self.subTest("correct_implementation_used"):
      opspecs = hlo_utils.get_opspecs(
          f.lower(scores), include_xla_kernels=False
      )
      self.assertNotEmpty(opspecs)
      self.assertIsInstance(
          opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic_tpu"])
      )


if __name__ == "__main__":
  absltest.main()
