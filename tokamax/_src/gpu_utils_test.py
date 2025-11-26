# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
from absl.testing import absltest
import jax
from tokamax._src import gpu_utils


class GpuUtilsTest(absltest.TestCase):

  def test_triton_support(self):
    is_gpu = jax.default_backend() == "gpu"
    self.assertEqual(gpu_utils.has_triton_support(), is_gpu)

  def test_mosaic_gpu_support(self):
    if jax.default_backend() != "gpu":
      self.assertFalse(gpu_utils.has_mosaic_gpu_support())

    device_kind = jax.devices()[0].device_kind
    expected = "H100" in device_kind or "B200" in device_kind
    self.assertEqual(gpu_utils.has_mosaic_gpu_support(), expected)

  def test_is_sm80(self):
    self.assertEqual(
        gpu_utils.is_sm80(), "A100" in jax.devices()[0].device_kind
    )

  def test_is_sm90(self):
    self.assertEqual(
        gpu_utils.is_sm90(), "H100" in jax.devices()[0].device_kind
    )

  def test_is_sm100(self):
    self.assertEqual(
        gpu_utils.is_sm100(), "B200" in jax.devices()[0].device_kind
    )


if __name__ == "__main__":
  absltest.main()
