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


def _nvidia_gpu(compute_capability):
  return mock.Mock(
      platform="gpu",
      device_vendor="NVIDIA Corporation",
      compute_capability=compute_capability,
  )


def _amd_gpu(arch):
  # On ROCm, `compute_capability` is the architecture name, not a number.
  return mock.Mock(
      platform="gpu",
      device_vendor="Advanced Micro Devices, Inc",
      compute_capability=arch,
  )


class GpuUtilsTest(absltest.TestCase):

  def test_is_nvidia(self):
    self.assertTrue(gpu_utils.is_nvidia(_nvidia_gpu("9.0")))
    self.assertFalse(gpu_utils.is_nvidia(_amd_gpu("gfx942")))
    self.assertFalse(gpu_utils.is_nvidia(mock.Mock(platform="cpu")))
    # Devices that do not report a vendor keep being treated as NVIDIA.
    no_vendor_dev = mock.Mock(
        spec=["platform", "compute_capability"],
        platform="gpu",
        compute_capability="9.0",
    )
    self.assertTrue(gpu_utils.is_nvidia(no_vendor_dev))
    self.assertTrue(gpu_utils.is_sm90(no_vendor_dev))

  def test_triton_support(self):
    is_nvidia_gpu = (
        jax.default_backend() == "gpu"
        and "NVIDIA" in jax.devices()[0].device_kind
    )
    self.assertEqual(gpu_utils.has_triton_support(), is_nvidia_gpu)
    sm75_dev = _nvidia_gpu("7.5")
    sm90_dev = _nvidia_gpu("9.0")
    self.assertFalse(gpu_utils.has_triton_support(sm75_dev))
    self.assertTrue(gpu_utils.has_triton_support(sm90_dev))

  def test_triton_support_without_compute_capability(self):
    # Pathways devices have no `compute_capability` attribute.
    pathways_dev = mock.Mock(spec=["platform"], platform="gpu")
    self.assertTrue(gpu_utils.has_triton_support(pathways_dev))

  def test_amd_gpu_is_not_an_nvidia_generation(self):
    for arch in ("gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1201"):
      with self.subTest(arch):
        dev = _amd_gpu(arch)
        self.assertFalse(gpu_utils.is_sm80(dev))
        self.assertFalse(gpu_utils.is_sm90(dev))
        self.assertFalse(gpu_utils.is_sm100(dev))
        self.assertFalse(gpu_utils.is_sm103(dev))
        self.assertFalse(gpu_utils.is_sm107(dev))
        self.assertFalse(gpu_utils.has_tcgen05_int8_support(dev))
        self.assertFalse(gpu_utils.has_triton_support(dev))
        self.assertFalse(gpu_utils.has_mosaic_gpu_support(dev))

  def test_mosaic_gpu_support(self):
    if jax.default_backend() != "gpu":
      self.assertFalse(gpu_utils.has_mosaic_gpu_support())

    device_kind = jax.devices()[0].device_kind
    expected = any(k in device_kind for k in ("H100", "B200", "GB300"))
    self.assertEqual(gpu_utils.has_mosaic_gpu_support(), expected)
    sm75_dev = _nvidia_gpu("7.5")
    sm90_dev = _nvidia_gpu("9.0")
    self.assertFalse(gpu_utils.has_mosaic_gpu_support(sm75_dev))
    self.assertTrue(gpu_utils.has_mosaic_gpu_support(sm90_dev))

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
    sm100_dev = _nvidia_gpu("10.0")
    sm103_dev = _nvidia_gpu("10.3")
    self.assertFalse(gpu_utils.is_sm103(sm100_dev))
    self.assertTrue(gpu_utils.is_sm103(sm103_dev))
    self.assertTrue(gpu_utils.is_sm100(sm103_dev))

  def test_is_sm107(self):
    sm100_dev = _nvidia_gpu("10.0")
    sm107_dev = _nvidia_gpu("10.7")
    self.assertFalse(gpu_utils.is_sm107(sm100_dev))
    self.assertTrue(gpu_utils.is_sm107(sm107_dev))
    self.assertTrue(gpu_utils.is_sm100(sm107_dev))

  def test_has_tcgen05_int8_support(self):
    self.assertEqual(
        gpu_utils.has_tcgen05_int8_support(),
        "B200" in jax.devices()[0].device_kind,
    )
    sm100_dev = _nvidia_gpu("10.0")
    sm103_dev = _nvidia_gpu("10.3")
    sm107_dev = _nvidia_gpu("10.7")
    self.assertTrue(gpu_utils.has_tcgen05_int8_support(sm100_dev))
    self.assertFalse(gpu_utils.has_tcgen05_int8_support(sm103_dev))
    self.assertFalse(gpu_utils.has_tcgen05_int8_support(sm107_dev))


if __name__ == "__main__":
  absltest.main()
