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
"""Block-scaled FP8 numerical tests for the MoE layer on TPU_8T."""

import dataclasses

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
from tokamax._src.ops.experimental.routing_moe.test.layer import moe_test_lib

jax.config.parse_flags_with_absl()

_BLOCK_SCALED_BASIC = helper_shapes.MoEShape(
    "block_scaled_basic",
    num_experts=8,
    top_k=2,
    hidden_size=512,
    intermediate_size=512,
    w_dtype=jnp.dtype(jnp.float8_e4m3fn),
    w_block_scaled=True,
)
_BLOCK_SCALED_QWEN_38 = dataclasses.replace(
    helper_shapes.QWEN_38_FLASH_NEXT,
    name="qwen_38_flash_next_block_scaled",
    num_experts=4,
    top_k=2,
    w_block_scaled=True,
)
_FP8_SCATTER_FORMAT = out_format.ActivationFormat(
    dtype=jnp.float8_e4m3fn,
    quant=out_format.auto_quant,
)


class MoEBlockScaledTest(parameterized.TestCase):
  """Tests `modular` and `monolithic` MoE layers with block-scaled FP8."""

  def setUp(self):
    super().setUp()
    helper_util.skip_unless_block_scaled_supported(self)

  def _check_layer(
      self,
      shape: helper_shapes.MoEShape,
      *,
      mode: moe_test_lib.Mode,
      layout: moe_test_lib.WeightLayout = moe_test_lib.WeightLayout(),
      num_tokens: int = 32,
      reduce_output: bool = True,
      moe_plan: moe_plan_lib.RoutingMoEPlan | None = None,
      gmm_plan: gmm_plan_lib.RoutingGmmPlan | None = None,
  ) -> None:
    helper_util.skip_unless_block_scaled_supported(
        self, min_devices=layout.sharding.num_cores
    )
    case = helper_cases.Case(shape.name, shape, num_tokens, layout.sharding)
    moe_case = moe_test_lib.MoECase.make_case(case, layout, moe_plan=moe_plan)
    actual = moe_case.run(
        mode,
        reduce_output=reduce_output,
        moe_plan=moe_plan,
        gmm_plan=gmm_plan,
    )
    moe_case.assert_close(actual, moe_plan=moe_plan)

  @parameterized.product(
      mode=["modular", "monolithic"],
      w_dtype=[jnp.float8_e4m3fn, jnp.float8_e5m2],
      renormalize=[True, False],
  )
  def test_single_device(
      self,
      mode: moe_test_lib.Mode,
      w_dtype: jax.typing.DTypeLike,
      renormalize: bool,
  ):
    shape = dataclasses.replace(
        _BLOCK_SCALED_BASIC,
        w_dtype=jnp.dtype(w_dtype),
        renormalize=renormalize,
    )
    self._check_layer(shape, mode=mode)

  @parameterized.named_parameters(
      dict(
          testcase_name="fp8_scatter_modular",
          mode="modular",
          shape=_BLOCK_SCALED_BASIC,
          moe_plan=moe_plan_lib.RoutingMoEPlan(
              scatter_format=_FP8_SCATTER_FORMAT
          ),
      ),
      dict(
          testcase_name="fp8_scatter_monolithic",
          mode="monolithic",
          shape=_BLOCK_SCALED_BASIC,
          moe_plan=moe_plan_lib.RoutingMoEPlan(
              scatter_format=_FP8_SCATTER_FORMAT
          ),
      ),
      dict(
          testcase_name="swigluoai",
          mode="modular",
          shape=dataclasses.replace(
              _BLOCK_SCALED_BASIC, activation="swigluoai"
          ),
      ),
      dict(
          testcase_name="shared_expert_modular",
          mode="modular",
          shape=dataclasses.replace(
              _BLOCK_SCALED_BASIC, has_shared_expert=True
          ),
      ),
      dict(
          testcase_name="shared_expert_monolithic",
          mode="monolithic",
          shape=dataclasses.replace(
              _BLOCK_SCALED_BASIC, has_shared_expert=True
          ),
      ),
      dict(
          testcase_name="folded_shared_expert",
          mode="modular",
          shape=dataclasses.replace(
              _BLOCK_SCALED_BASIC, has_shared_expert=True
          ),
          layout=moe_test_lib.WeightLayout(fold_shared_expert=True),
      ),
      dict(
          testcase_name="unfused_gmm_plan",
          mode="monolithic",
          shape=dataclasses.replace(_BLOCK_SCALED_BASIC, hidden_size=1024),
          gmm_plan=gmm_plan_lib.RoutingGmmPlan(fuse_up_and_down=False),
      ),
      dict(
          testcase_name="qwen_38_flash_next_decode",
          mode="monolithic",
          shape=_BLOCK_SCALED_QWEN_38,
          num_tokens=8,
      ),
  )
  def test_single_device_features(
      self,
      *,
      mode: moe_test_lib.Mode,
      shape: helper_shapes.MoEShape,
      layout: moe_test_lib.WeightLayout = moe_test_lib.WeightLayout(),
      num_tokens: int = 32,
      moe_plan: moe_plan_lib.RoutingMoEPlan | None = None,
      gmm_plan: gmm_plan_lib.RoutingGmmPlan | None = None,
  ):
    self._check_layer(
        shape,
        mode=mode,
        layout=layout,
        num_tokens=num_tokens,
        moe_plan=moe_plan,
        gmm_plan=gmm_plan,
    )

  @parameterized.product(
      mode=["modular", "monolithic"],
      sharding=[
          helper_cases.MoESharding(ep=2),
          helper_cases.MoESharding(tp=2),
      ],
      use_sp=[True, False],
  )
  def test_sharded_ep_and_tp(
      self,
      mode: moe_test_lib.Mode,
      sharding: helper_cases.MoESharding,
      use_sp: bool,
  ):
    layout = moe_test_lib.WeightLayout(sharding, use_sp=use_sp)
    self._check_layer(_BLOCK_SCALED_BASIC, mode=mode, layout=layout)

  @parameterized.product(
      mode=["modular", "monolithic"],
      use_sp=[True, False],
      scatter_format=[None, _FP8_SCATTER_FORMAT],
  )
  def test_sharded_remote_scatter(
      self,
      mode: moe_test_lib.Mode,
      use_sp: bool,
      scatter_format: out_format.ActivationFormat | None,
  ):
    layout = moe_test_lib.WeightLayout(
        helper_cases.MoESharding(ep=2), use_sp=use_sp
    )
    moe_plan = moe_plan_lib.RoutingMoEPlan(
        remote_scatter=True, scatter_format=scatter_format
    )
    self._check_layer(
        _BLOCK_SCALED_BASIC, mode=mode, layout=layout, moe_plan=moe_plan
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="ep2_qwen38_shared_expert_sp",
          mode="monolithic",
          shape=_BLOCK_SCALED_QWEN_38,
          layout=moe_test_lib.WeightLayout(
              helper_cases.MoESharding(ep=2), use_sp=True
          ),
          num_tokens=16,
      ),
      dict(
          testcase_name="tp2_shared_expert_nosp",
          mode="monolithic",
          shape=dataclasses.replace(
              _BLOCK_SCALED_BASIC, has_shared_expert=True
          ),
          layout=moe_test_lib.WeightLayout(helper_cases.MoESharding(tp=2)),
      ),
      dict(
          testcase_name="tp2_folded_shared_expert_sp",
          mode="modular",
          shape=dataclasses.replace(
              _BLOCK_SCALED_BASIC, has_shared_expert=True
          ),
          layout=moe_test_lib.WeightLayout(
              helper_cases.MoESharding(tp=2),
              use_sp=True,
              fold_shared_expert=True,
          ),
      ),
  )
  def test_sharded_shared_expert(
      self,
      *,
      mode: moe_test_lib.Mode,
      shape: helper_shapes.MoEShape,
      layout: moe_test_lib.WeightLayout,
      num_tokens: int = 32,
  ):
    self._check_layer(
        shape,
        mode=mode,
        layout=layout,
        num_tokens=num_tokens,
        reduce_output=False,
    )


if __name__ == "__main__":
  absltest.main()
