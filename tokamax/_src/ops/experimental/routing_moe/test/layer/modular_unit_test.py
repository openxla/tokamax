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
"""Unit tests for `modular.routing_moe` against the reference MoE."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tokamax._src.ops.experimental.routing_moe.test.common import test_shapes
from tokamax._src.ops.experimental.routing_moe.test.layer import moe_test_base
from tokamax._src.ops.experimental.routing_moe.test.layer import moe_test_lib

jax.config.parse_flags_with_absl()


class ModularMoEUnitTest(moe_test_base.MoEUnitTest):
  mode = "modular"

  @parameterized.product(use_sp=[True, False])
  def test_folded_shared_expert(self, use_sp: bool):
    """Routes to the shared expert as an extra expert via external top-k."""
    shape = dataclasses.replace(test_shapes.BASIC, has_shared_expert=True)
    layout = moe_test_lib.WeightLayout(use_sp=use_sp, fold_shared_expert=True)
    self.check_matches_reference(shape, layout, reduce_output=False)


if __name__ == "__main__":
  absltest.main()
