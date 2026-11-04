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
"""Unit tests that every metadata walk lays out the same tiles as NumPy."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.test.common import test_inputs
from tokamax._src.ops.experimental.routing_moe.test.common import test_util
from tokamax._src.ops.experimental.routing_moe.test.gmm import metadata_test_lib

jax.config.parse_flags_with_absl()


class MetadataUnitTest(parameterized.TestCase):

  def setUp(self):
    test_util.skip_unless_supported(self)
    super().setUp()

  @parameterized.product(
      routing_case=metadata_test_lib.ROUTINGS, plan=metadata_test_lib.PLANS
  )
  def test_walk_matches_numpy(
      self,
      routing_case: metadata_test_lib.RoutingCase,
      plan: gmm_plan.RoutingGmmPlan,
  ):
    case = routing_case.case
    test_util.check_gmm_supported(
        case, live_group_walk=bool(plan.live_group_walk)
    )
    shape = case.shape
    gating_output = test_inputs.random_gating(
        jax.random.key(0),
        case.num_tokens,
        shape.num_experts,
        num_skipped=routing_case.num_skipped,
    )
    _, topk_indices = test_inputs.top_k(gating_output, shape.top_k)
    routing = test_inputs.Routing.make_routing(
        topk_indices,
        shape.num_experts,
        group_offset=routing_case.group_offset,
        num_local_experts=case.num_local_experts,
    )
    cfg = metadata_test_lib.make_gmm_config(
        num_rows=case.num_rows,
        num_global_groups=shape.num_experts,
        num_local_groups=case.num_local_experts,
        size_k=shape.hidden_size,
        size_n=2 * shape.intermediate_size,
        custom_plan=plan,
    )
    actual = metadata_test_lib.fill_metadata(routing, cfg)
    expected = metadata_test_lib.expected_metadata(routing, cfg.plan.tile_m)
    self.assertLess(len(actual), cfg.max_num_gm)
    np.testing.assert_array_equal(actual, expected)

  @parameterized.named_parameters(
      # 512 global groups and 10 routed rows per token; a shard gets on average
      # rows * local / 512 of them, fewer than its groups iff rows < 512.
      dict(testcase_name="tp_decode", num_tokens=1, num_local_experts=512),
      dict(testcase_name="tp_last", num_tokens=51, num_local_experts=512),
      dict(testcase_name="tp_full", num_tokens=52, num_local_experts=512),
      dict(testcase_name="ep_decode", num_tokens=1, num_local_experts=64),
      dict(testcase_name="ep_mid", num_tokens=16, num_local_experts=64),
      dict(testcase_name="ep_last", num_tokens=51, num_local_experts=64),
      dict(testcase_name="ep_full", num_tokens=52, num_local_experts=64),
  )
  def test_live_group_walk_default(self, num_tokens, num_local_experts):
    # The live walk is chosen while most of the shard's groups are empty, which
    # depends on the global group count, not the shard's.
    num_experts, topk, size_k, size_n = 512, 10, 512, 1024
    rows = num_tokens * topk
    cfg = metadata_test_lib.make_gmm_config(
        num_rows=rows,
        num_global_groups=num_experts,
        num_local_groups=num_local_experts,
        size_k=size_k,
        size_n=size_n,
    )
    self.assertEqual(cfg.plan.live_group_walk, rows < num_experts)


if __name__ == "__main__":
  absltest.main()
