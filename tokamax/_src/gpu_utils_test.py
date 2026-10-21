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
from unittest import mock

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
    expected = any(k in device_kind for k in ("H100", "B200", "GB300"))
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
    device_kind = jax.devices()[0].device_kind
    self.assertEqual(
        gpu_utils.is_sm100(),
        any(k in device_kind for k in ("B200", "GB300")),
    )

  def test_is_sm103(self):
    self.assertEqual(
        gpu_utils.is_sm103(), "GB300" in jax.devices()[0].device_kind
    )
    sm100_dev = mock.Mock(platform="gpu", compute_capability="10.0")
    sm103_dev = mock.Mock(platform="gpu", compute_capability="10.3")
    self.assertFalse(gpu_utils.is_sm103(sm100_dev))
    self.assertTrue(gpu_utils.is_sm103(sm103_dev))
    self.assertTrue(gpu_utils.is_sm100(sm103_dev))

  def test_is_sm107(self):
    sm100_dev = mock.Mock(platform="gpu", compute_capability="10.0")
    sm107_dev = mock.Mock(platform="gpu", compute_capability="10.7")
    self.assertFalse(gpu_utils.is_sm107(sm100_dev))
    self.assertTrue(gpu_utils.is_sm107(sm107_dev))
    self.assertTrue(gpu_utils.is_sm100(sm107_dev))

  def test_has_tcgen05_int8_support(self):
    self.assertEqual(
        gpu_utils.has_tcgen05_int8_support(),
        "B200" in jax.devices()[0].device_kind,
    )
    sm100_dev = mock.Mock(platform="gpu", compute_capability="10.0")
    sm103_dev = mock.Mock(platform="gpu", compute_capability="10.3")
    sm107_dev = mock.Mock(platform="gpu", compute_capability="10.7")
    self.assertTrue(gpu_utils.has_tcgen05_int8_support(sm100_dev))
    self.assertFalse(gpu_utils.has_tcgen05_int8_support(sm103_dev))
    self.assertFalse(gpu_utils.has_tcgen05_int8_support(sm107_dev))


if __name__ == "__main__":
  absltest.main()
