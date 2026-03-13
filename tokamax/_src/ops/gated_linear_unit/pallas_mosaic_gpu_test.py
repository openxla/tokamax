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
"""Tests for Pallas Mosaic GPU Gated Linear Unit."""

from absl.testing import absltest
from tokamax._src import gpu_utils
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu as pl_glu
from tokamax._src.ops.gated_linear_unit import test_base


class PallasMosaicGpuGatedLinearUnitTest(test_base.GatedLinearUnitTestBase):

  def __init__(self, *args):
    super().__init__(*args, glu_fn=pl_glu.PallasMosaicGpuGatedLinearUnit())

  def setUp(self):
    if not gpu_utils.has_mosaic_gpu_support():
      self.skipTest("Not supported on TPUs.")
    super().setUp()


if __name__ == "__main__":
  absltest.main()
