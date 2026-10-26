# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
import jax.experimental.pallas.tpu as pltpu
from tokamax._src import mosaic_tpu

_MIB = 1024 * 1024
# The maximum scoped VMEM that XLA reports on each chip.
_MAX_SCOPED_VMEM_BYTES = {
    pltpu.ChipVersion.TPU_7X: 67043328,
    pltpu.ChipVersion.TPU_8I: 199229440,
}


class DefaultTcVmemLimitBytesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("tpu_7x", pltpu.ChipVersion.TPU_7X, int(64 * _MIB * 0.95)),
      ("tpu_8i", pltpu.ChipVersion.TPU_8I, int(192 * _MIB * 0.95)),
  )
  def test_default_fraction(self, chip_version, expected):
    tpu_info = pltpu.get_tpu_info_for_chip(chip_version, 1)
    limit = mosaic_tpu.default_tc_vmem_limit_bytes(tpu_info=tpu_info)
    self.assertEqual(limit, expected)
    self.assertLess(limit, _MAX_SCOPED_VMEM_BYTES[chip_version])

  def test_custom_fraction(self):
    tpu_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    self.assertEqual(
        mosaic_tpu.default_tc_vmem_limit_bytes(0.5, tpu_info=tpu_info),
        32 * _MIB,
    )
    self.assertEqual(
        mosaic_tpu.default_tc_vmem_limit_bytes(1.0, tpu_info=tpu_info),
        64 * _MIB,
    )

  @parameterized.parameters(0.0, -0.5, 1.5)
  def test_invalid_fraction(self, fraction):
    tpu_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    with self.assertRaisesRegex(ValueError, "fraction"):
      mosaic_tpu.default_tc_vmem_limit_bytes(fraction, tpu_info=tpu_info)

  def test_defaults_to_current_device(self):
    tpu_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_8I, 1)
    with mock.patch.object(
        pltpu,
        "get_tpu_info",
        autospec=True,
        spec_set=True,
        return_value=tpu_info,
    ):
      self.assertEqual(
          mosaic_tpu.default_tc_vmem_limit_bytes(), int(192 * _MIB * 0.95)
      )


if __name__ == "__main__":
  absltest.main()
