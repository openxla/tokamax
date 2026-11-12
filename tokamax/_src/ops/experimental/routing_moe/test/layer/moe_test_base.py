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
"""Shared test base classes for the MoE layer entries."""

import dataclasses

from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import out_format
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan_lib
from tokamax._src.ops.experimental.routing_moe.layer import plan as moe_plan_lib
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_cases
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_inputs
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_reference
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_shapes
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_util
from tokamax._src.ops.experimental.routing_moe.test.layer import moe_test_lib

_UNIT_SHARDINGS = (
    helper_cases.MoESharding(ep=2),
    helper_cases.MoESharding(tp=2),
    helper_cases.MoESharding(ep=2, tp=2),
)
_FP8_SCATTER_FORMAT = out_format.ActivationFormat(
    dtype=jnp.float8_e4m3fn,
    quant=out_format.auto_quant,
)


class MoETestCase(parameterized.TestCase):
  """Base test case for verifying MoE layer execution against XLA reference."""

  mode: moe_test_lib.Mode = "modular"

  def setUp(self):
    helper_util.skip_unless_supported(self)
    super().setUp()

  def check_case(
      self,
      case: helper_cases.Case,
      layout: moe_test_lib.WeightLayout | None = None,
      *,
      reduce_output: bool = True,
      capture_xprof: bool = False,
      moe_plan: moe_plan_lib.RoutingMoEPlan | None = None,
      gmm_plan: gmm_plan_lib.RoutingGmmPlan | None = None,
  ) -> None:
    """Checks that the execution mode matches numerics on `case`."""
    sharding = case.sharding if layout is None else layout.sharding
    helper_util.skip_unless_supported(self, min_devices=sharding.num_cores)
    moe_case = moe_test_lib.MoECase.make_case(case, layout, moe_plan=moe_plan)
    actual = moe_case.run(
        self.mode,
        reduce_output=reduce_output,
        moe_plan=moe_plan,
        gmm_plan=gmm_plan,
    )
    moe_case.assert_close(actual, moe_plan=moe_plan)
    if capture_xprof:
      _, topk_indices = helper_reference.compute_topk(moe_case.raw, case.shape)
      routing = helper_inputs.Routing.make_routing(
          topk_indices, case.shape.num_experts
      )
      helper_util.log_expert_load(
          case.name, routing.group_sizes, ep=sharding.ep
      )
      with helper_util.XprofTrace():
        moe_case.run(
            self.mode,
            reduce_output=reduce_output,
            moe_plan=moe_plan,
            gmm_plan=gmm_plan,
        )

  def check_matches_reference(
      self,
      shape: helper_shapes.MoEShape,
      layout: moe_test_lib.WeightLayout = moe_test_lib.WeightLayout(),
      *,
      num_tokens: int = 256,
      reduce_output: bool = True,
      moe_plan: moe_plan_lib.RoutingMoEPlan | None = None,
  ) -> None:
    """Checks that the execution mode matches numerics on `shape`."""
    self.check_case(
        helper_cases.Case(shape.name, shape, num_tokens),
        layout,
        reduce_output=reduce_output,
        moe_plan=moe_plan,
    )


class MoEUnitTest(MoETestCase):
  """Unit tests for one MoE layer execution mode against the XLA reference."""

  @parameterized.product(case=helper_cases.UNIT_CASES)
  def test_unit_cases(self, case: helper_cases.Case):
    self.check_case(case)

  @parameterized.product(renormalize=[True, False])
  def test_basic(self, renormalize: bool):
    shape = dataclasses.replace(helper_shapes.BASIC, renormalize=renormalize)
    self.check_matches_reference(shape)

  @parameterized.product(
      use_sp=[True, False],
      remote_scatter=[True, False],
      w_dtype=[None, jnp.float8_e4m3fn],
  )
  def test_hybrid(
      self,
      use_sp: bool,
      remote_scatter: bool,
      w_dtype: jax.typing.DTypeLike | None,
  ):
    """Checks hybrid EP+TP execution across SP, scatter, and quantization."""
    dtype = None if w_dtype is None else jnp.dtype(w_dtype)
    shape = dataclasses.replace(helper_shapes.BASIC, w_dtype=dtype)
    layout = moe_test_lib.WeightLayout(
        helper_cases.MoESharding(ep=2, tp=2), use_sp=use_sp
    )
    moe_plan = moe_plan_lib.RoutingMoEPlan(remote_scatter=remote_scatter)
    self.check_matches_reference(shape, layout, moe_plan=moe_plan)

  @parameterized.product(
      use_sp=[True, False],
      scatter_format=[None, _FP8_SCATTER_FORMAT],
  )
  def test_remote_scatter(
      self, use_sp: bool, scatter_format: out_format.ActivationFormat | None
  ):
    """Checks EP remote scatter across SP and scatter quantization."""
    layout = moe_test_lib.WeightLayout(
        helper_cases.MoESharding(ep=2), use_sp=use_sp
    )
    moe_plan = moe_plan_lib.RoutingMoEPlan(
        remote_scatter=True, scatter_format=scatter_format
    )
    self.check_matches_reference(helper_shapes.BASIC, layout, moe_plan=moe_plan)

  @parameterized.product(sharding=_UNIT_SHARDINGS, use_sp=[True, False])
  def test_fp8_scatter(self, sharding: helper_cases.MoESharding, use_sp: bool):
    layout = moe_test_lib.WeightLayout(sharding, use_sp=use_sp)
    moe_plan = moe_plan_lib.RoutingMoEPlan(scatter_format=_FP8_SCATTER_FORMAT)
    self.check_matches_reference(helper_shapes.BASIC, layout, moe_plan=moe_plan)

  @parameterized.product(sharding=_UNIT_SHARDINGS, use_sp=[True, False])
  def test_unreduced_output(
      self, sharding: helper_cases.MoESharding, use_sp: bool
  ):
    layout = moe_test_lib.WeightLayout(sharding, use_sp=use_sp)
    self.check_matches_reference(
        helper_shapes.BASIC, layout, reduce_output=False
    )

  @parameterized.product(sharding=_UNIT_SHARDINGS, use_sp=[True, False])
  def test_shared_expert(
      self, sharding: helper_cases.MoESharding, use_sp: bool
  ):
    shape = dataclasses.replace(helper_shapes.BASIC, has_shared_expert=True)
    layout = moe_test_lib.WeightLayout(sharding, use_sp=use_sp)
    self.check_matches_reference(shape, layout, reduce_output=False)

  def test_unreduced_output_with_remote_scatter_raises(self):
    with self.assertRaisesRegex(
        ValueError, "remote_scatter cannot be True when reduce_output is False"
    ):
      self.check_matches_reference(
          helper_shapes.BASIC,
          moe_test_lib.WeightLayout(helper_cases.MoESharding(ep=2)),
          reduce_output=False,
          moe_plan=moe_plan_lib.RoutingMoEPlan(remote_scatter=True),
      )

  def test_remote_scatter_without_ep_raises(self):
    with self.assertRaisesRegex(ValueError, "remote_scatter requires ep_axis"):
      self.check_matches_reference(
          helper_shapes.BASIC,
          moe_plan=moe_plan_lib.RoutingMoEPlan(remote_scatter=True),
      )

  @parameterized.product(activation=["silu", "swigluoai"])
  def test_activation(self, activation: str):
    shape = dataclasses.replace(helper_shapes.BASIC, activation=activation)
    self.check_matches_reference(shape)

  @parameterized.product(
      w_dtype=[jnp.int8, jnp.float8_e5m2, jnp.float4_e2m1fn],
  )
  def test_sub_channel_quantization(self, w_dtype: jax.typing.DTypeLike):
    """Test weight sub-channel quantization."""
    shape = dataclasses.replace(
        helper_shapes.BASIC,
        w_dtype=jnp.dtype(w_dtype),
        w_quant_block_size=256,
    )
    self.check_matches_reference(shape)

  def test_bias(self):
    shape = dataclasses.replace(
        helper_shapes.BASIC, renormalize=False, has_bias=True
    )
    self.check_matches_reference(shape)
