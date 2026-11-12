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
"""Unit tests for the routing MoE layer planner."""

from absl.testing import absltest
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.layer import plan as moe_plan


class MoEPlanTest(absltest.TestCase):

  def test_resolved_moe_plan_populates_formats(self):
    resolved = moe_plan.ResolvedMoEPlan.create(
        moe_plan.RoutingMoEPlan(
            gather_format=out_format_lib.ActivationFormat(),
            scatter_format=out_format_lib.ActivationFormat(
                dtype=jnp.float8_e4m3fn, quant=out_format_lib.auto_quant
            ),
        ),
        x_dtype=jnp.bfloat16,
    )
    self.assertEqual(
        resolved.gather_format,
        out_format_lib.ResolvedActivationFormat(
            dtype=jnp.dtype(jnp.bfloat16), quant=None
        ),
    )
    self.assertEqual(
        resolved.scatter_format,
        out_format_lib.ResolvedActivationFormat(
            dtype=jnp.dtype(jnp.float8_e4m3fn),
            quant=out_format_lib.ResolvedLaneStrided(
                block_col_size=out_format_lib.tile_size
            ),
        ),
    )


if __name__ == "__main__":
  absltest.main()
