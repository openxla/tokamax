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
"""Tests for TopK Pallas TPU kernel."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.topk import base
from tokamax._src.ops.experimental.tpu.topk import pallas_mosaic_tpu_kernel


# TODO: Add more tests.
class PallasMosaicTpuKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if not pltpu.get_tpu_info().generation >= 6:
      self.skipTest("Pallas TPU kernel requires TPU v6 or newer.")

  def test_kernel_small_k(self):
    keys = jnp.arange(128, 0, -1.0, dtype=jnp.float32)[None, :]
    k = 4
    res_k, res_v = pallas_mosaic_tpu_kernel.top_k(
        keys, k=k, num_seq_windows=1, digit_width=4, num_digits=8
    )
    ref_k, ref_v = base.topk(keys, k=k)
    np.testing.assert_allclose(res_k, ref_k)
    np.testing.assert_array_equal(res_v, ref_v)


if __name__ == "__main__":
  absltest.main()
