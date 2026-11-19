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
from absl.testing import parameterized
import jax
from jax.extend import backend
from tokamax._src import gpu_utils


class GpuUtilsTest(parameterized.TestCase):

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

  def test_virtual_and_abstract_gpu_devices(self):
    h100_abs = jax.sharding.AbstractDevice(
        device_kind="NVIDIA H100 80GB HBM3", num_cores=None, platform="gpu"
    )
    t4_abs = jax.sharding.AbstractDevice(
        device_kind="Tesla T4", num_cores=None, platform="gpu"
    )
    tpu_abs = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    none_cc_gpu = mock.Mock(
        platform="gpu", compute_capability=None, device_kind="NVIDIA B200"
    )
    sm75_gpu = mock.Mock(
        platform="gpu", compute_capability="7.5", device_kind="Tesla T4"
    )
    unrecognized_fp4_gpu = jax.sharding.AbstractDevice(
        device_kind="NVIDIA NextGen FP4", num_cores=None, platform="gpu"
    )

    self.assertTrue(gpu_utils.has_mosaic_gpu_support(h100_abs))
    self.assertTrue(gpu_utils.has_triton_support(h100_abs))
    self.assertTrue(gpu_utils.has_mosaic_gpu_support(none_cc_gpu))
    self.assertTrue(gpu_utils.has_triton_support(none_cc_gpu))
    self.assertTrue(gpu_utils.has_mosaic_gpu_support(unrecognized_fp4_gpu))
    self.assertTrue(gpu_utils.has_triton_support(unrecognized_fp4_gpu))
    self.assertFalse(gpu_utils.has_mosaic_gpu_support(t4_abs))
    self.assertFalse(gpu_utils.has_triton_support(t4_abs))
    self.assertFalse(gpu_utils.has_mosaic_gpu_support(sm75_gpu))
    self.assertFalse(gpu_utils.has_triton_support(sm75_gpu))
    self.assertFalse(gpu_utils.has_mosaic_gpu_support(tpu_abs))
    self.assertFalse(gpu_utils.has_triton_support(tpu_abs))

  @parameterized.named_parameters(
      dict(
          testcase_name="b200",
          device_kind="NVIDIA B200",
          sm80=False,
          sm90=False,
          sm100=True,
          sm103=False,
          sm107=False,
          tcgen05_int8=True,
      ),
      dict(
          testcase_name="gb200",
          device_kind="NVIDIA GB200",
          sm80=False,
          sm90=False,
          sm100=True,
          sm103=False,
          sm107=False,
          tcgen05_int8=True,
      ),
      dict(
          testcase_name="gb300",
          device_kind="NVIDIA GB300",
          sm80=False,
          sm90=False,
          sm100=True,
          sm103=True,
          sm107=False,
          tcgen05_int8=False,
      ),
      dict(
          testcase_name="vr200",
          device_kind="NVIDIA VR200",
          sm80=False,
          sm90=False,
          sm100=True,
          sm103=False,
          sm107=True,
          tcgen05_int8=False,
      ),
      dict(
          testcase_name="h100",
          device_kind="NVIDIA H100 80GB HBM3",
          sm80=False,
          sm90=True,
          sm100=False,
          sm103=False,
          sm107=False,
          tcgen05_int8=False,
      ),
      dict(
          testcase_name="a100",
          device_kind="NVIDIA A100-SXM4-80GB",
          sm80=True,
          sm90=False,
          sm100=False,
          sm103=False,
          sm107=False,
          tcgen05_int8=False,
      ),
      dict(
          testcase_name="t4",
          device_kind="Tesla T4",
          sm80=False,
          sm90=False,
          sm100=False,
          sm103=False,
          sm107=False,
          tcgen05_int8=False,
      ),
      dict(
          testcase_name="unknown",
          device_kind="NVIDIA Unknown",
          sm80=False,
          sm90=False,
          sm100=False,
          sm103=False,
          sm107=False,
          tcgen05_int8=False,
      ),
  )
  def test_abstract_device_compute_capability(
      self,
      device_kind,
      sm80,
      sm90,
      sm100,
      sm103,
      sm107,
      tcgen05_int8,
  ):
    abs_dev = jax.sharding.AbstractDevice(
        device_kind=device_kind, num_cores=None, platform="gpu"
    )
    self.assertEqual(gpu_utils.is_sm80(abs_dev), sm80)
    self.assertEqual(gpu_utils.is_sm90(abs_dev), sm90)
    self.assertEqual(gpu_utils.is_sm100(abs_dev), sm100)
    self.assertEqual(gpu_utils.is_sm103(abs_dev), sm103)
    self.assertEqual(gpu_utils.is_sm107(abs_dev), sm107)
    self.assertEqual(gpu_utils.has_tcgen05_int8_support(abs_dev), tcgen05_int8)

  def test_abstract_device_inherits_matching_default_device_cc(self):
    abs_dev = jax.sharding.AbstractDevice(
        device_kind="NVIDIA CustomGPU", num_cores=None, platform="gpu"
    )
    default_dev = mock.Mock(
        platform="gpu",
        device_kind="NVIDIA CustomGPU",
        compute_capability="10.0",
    )
    with mock.patch.object(
        backend, "get_default_device", return_value=default_dev
    ):
      self.assertTrue(gpu_utils.is_sm100(abs_dev))
      self.assertTrue(gpu_utils.has_tcgen05_int8_support(abs_dev))


if __name__ == "__main__":
  absltest.main()
