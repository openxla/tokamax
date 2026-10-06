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
"""Tests for Pallas TPU TopK operator wrapper."""

from absl.testing import absltest
import jax
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.experimental.tpu.topk import pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.topk import test_base


# TODO: Add more tests.
class PallasTpuTopKTest(test_base.TopKTestBase):

  def __init__(self, *args):
    super().__init__(*args, topk_fn=pallas_mosaic_tpu.PallasTpuTopK())

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if not pltpu.get_tpu_info().generation >= 7:
      self.skipTest(
          "SparseCore TopK Pallas TPU kernel requires TPU v7 or newer."
      )


if __name__ == "__main__":
  absltest.main()
