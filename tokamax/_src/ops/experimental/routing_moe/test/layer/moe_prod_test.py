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
"""Production-shape numerics and XProf traces for the MoE layer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import out_format
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan_lib
from tokamax._src.ops.experimental.routing_moe.layer import plan as moe_plan_lib
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_cases
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_shapes
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_util
from tokamax._src.ops.experimental.routing_moe.test.layer import moe_test_base
from tokamax._src.ops.experimental.routing_moe.test.layer import moe_test_lib

jax.config.parse_flags_with_absl()

_NON_SHARED_EXPERT_CASES = tuple(
    c for c in helper_cases.PROD_CASES if not c.shape.has_shared_expert
)
_SHARED_EXPERT_CASES = tuple(
    c for c in helper_cases.PROD_CASES if c.shape.has_shared_expert
)


class MoEProdTest(moe_test_base.MoETestCase):

  @parameterized.product(
      case=_NON_SHARED_EXPERT_CASES,
      use_sp=[True, False],
  )
  def test_non_shared_expert_model(
      self,
      case: helper_cases.Case,
      use_sp: bool,
  ):
    layout = moe_test_lib.WeightLayout(case.sharding, use_sp=use_sp)
    self.check_case(
        case, layout, capture_xprof=helper_util.ROUTING_MOE_PROFILE.value
    )

  @parameterized.product(
      case=_SHARED_EXPERT_CASES,
      use_sp=[True, False],
      remote_scatter=[True, False],
      fold_shared_expert=[True, False],
  )
  def test_shared_expert_model(
      self,
      case: helper_cases.Case,
      use_sp: bool,
      remote_scatter: bool,
      fold_shared_expert: bool,
  ):
    if remote_scatter and case.sharding.ep == 1:
      self.skipTest("remote_scatter is only valid when sharding.ep > 1.")
    if fold_shared_expert and case.sharding.ep > 1:
      self.skipTest("fold_shared_expert is only valid under pure TP.")
    layout = moe_test_lib.WeightLayout(
        case.sharding,
        use_sp=use_sp,
        fold_shared_expert=fold_shared_expert,
    )
    self.check_case(
        case,
        layout,
        reduce_output=remote_scatter,
        capture_xprof=helper_util.ROUTING_MOE_PROFILE.value,
        moe_plan=moe_plan_lib.RoutingMoEPlan(remote_scatter=remote_scatter),
    )

  @parameterized.product(
      num_tokens=[1280, 10240],
      use_sp=[True, False],
      remote_scatter=[False, True],
      scatter_format=[
          None,
          out_format.ActivationFormat(
              dtype=jnp.float8_e4m3fn,
              quant=out_format.auto_quant,
          ),
      ],
  )
  def test_remote_scatter(
      self,
      num_tokens: int,
      use_sp: bool,
      remote_scatter: bool,
      scatter_format: out_format.ActivationFormat | None,
  ):
    case = helper_cases.Case(
        "remote_scatter",
        helper_shapes.REMOTE_SCATTER,
        num_tokens,
        helper_cases.MoESharding(ep=4),
    )
    layout = moe_test_lib.WeightLayout(case.sharding, use_sp=use_sp)
    moe_plan = moe_plan_lib.RoutingMoEPlan(
        remote_scatter=remote_scatter, scatter_format=scatter_format
    )
    self.check_case(
        case,
        layout,
        capture_xprof=helper_util.ROUTING_MOE_PROFILE.value,
        moe_plan=moe_plan,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="_unfused",
          gmm_plan=gmm_plan_lib.RoutingGmmPlan(fuse_up_and_down=False),
      ),
      dict(
          testcase_name="_tm32_b3",
          gmm_plan=gmm_plan_lib.RoutingGmmPlan(tile_m=32, num_w_buffers=3),
      ),
  )
  def test_gmm_plan(self, gmm_plan: gmm_plan_lib.RoutingGmmPlan):
    """Runs a production case with gmm plans other than the default one."""
    self.check_case(
        helper_cases.QWEN_38_FLASH_NEXT_8,
        reduce_output=False,
        capture_xprof=helper_util.ROUTING_MOE_PROFILE.value,
        gmm_plan=gmm_plan,
    )


if __name__ == "__main__":
  absltest.main()
