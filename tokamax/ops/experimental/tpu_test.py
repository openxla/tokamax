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
"""Tests for tokamax.ops.experimental.tpu exports."""

from absl.testing import absltest
from tokamax._src.ops.experimental.routing_moe_op import pallas_mosaic_tpu as routing_moe_tpu
from tokamax.ops.experimental import tpu as experimental_tpu


class ExperimentalTpuExportsTest(absltest.TestCase):

  def test_all_symbols_present(self):
    for name in experimental_tpu.__all__:
      self.assertTrue(hasattr(experimental_tpu, name), msg=name)

  def test_routing_moe_exports(self):
    self.assertIs(
        experimental_tpu.RoutingMoe,
        routing_moe_tpu.PallasMosaicTpuRoutingMoe,
    )
    self.assertIs(
        experimental_tpu.RoutingMoeConfig,
        routing_moe_tpu.Config,
    )


if __name__ == "__main__":
  absltest.main()
