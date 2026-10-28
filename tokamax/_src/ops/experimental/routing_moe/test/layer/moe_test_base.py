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
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases
from tokamax._src.ops.experimental.routing_moe.test.common import test_shapes
from tokamax._src.ops.experimental.routing_moe.test.common import test_util
from tokamax._src.ops.experimental.routing_moe.test.layer import moe_test_lib


class MoETestCase(parameterized.TestCase):
  """Base test case for verifying MoE layer execution against XLA reference."""

  mode: moe_test_lib.Mode = "modular"

  def setUp(self):
    test_util.skip_unless_supported(self)
    super().setUp()

  def check_case(
      self,
      case: test_cases.Case,
      layout: moe_test_lib.WeightLayout = moe_test_lib.WeightLayout(),
      *,
      remote_scatter: bool = False,
      reduce_output: bool = True,
      capture_xprof: bool = False,
  ) -> None:
    """Checks that the execution mode matches numerics on `case`."""
    moe_case = moe_test_lib.MoECase.make_case(case, layout)
    actual = moe_case.run(
        self.mode, remote_scatter=remote_scatter, reduce_output=reduce_output
    )
    moe_case.assert_close(actual)
    if capture_xprof:
      with test_util.XprofTrace():
        moe_case.run(
            self.mode,
            remote_scatter=remote_scatter,
            reduce_output=reduce_output,
        )

  def check_matches_reference(
      self,
      shape: test_shapes.MoEShape,
      layout: moe_test_lib.WeightLayout = moe_test_lib.WeightLayout(),
      *,
      num_tokens: int = 256,
      remote_scatter: bool = False,
      reduce_output: bool = True,
  ) -> None:
    """Checks that the execution mode matches numerics on `shape`."""
    self.check_case(
        test_cases.Case(shape.name, shape, num_tokens),
        layout,
        remote_scatter=remote_scatter,
        reduce_output=reduce_output,
    )


class MoEUnitTest(MoETestCase):
  """Unit tests for one MoE layer execution mode against the XLA reference."""

  @parameterized.product(case=test_cases.UNIT_CASES)
  def test_unit_cases(self, case: test_cases.Case):
    layout = moe_test_lib.WeightLayout(use_ep=case.ep_size > 1)
    self.check_case(case, layout)

  @parameterized.product(renormalize=[True, False])
  def test_basic(self, renormalize: bool):
    shape = dataclasses.replace(test_shapes.BASIC, renormalize=renormalize)
    self.check_matches_reference(shape)

  @parameterized.product(use_sp=[True, False])
  def test_remote_scatter(self, use_sp: bool):
    layout = moe_test_lib.WeightLayout(use_ep=True, use_sp=use_sp)
    self.check_matches_reference(test_shapes.BASIC, layout, remote_scatter=True)

  @parameterized.product(use_ep=[True, False], use_sp=[True, False])
  def test_unreduced_output(self, use_ep: bool, use_sp: bool):
    layout = moe_test_lib.WeightLayout(use_ep=use_ep, use_sp=use_sp)
    self.check_matches_reference(test_shapes.BASIC, layout, reduce_output=False)

  @parameterized.product(use_ep=[True, False], use_sp=[True, False])
  def test_shared_expert(self, use_ep: bool, use_sp: bool):
    shape = dataclasses.replace(test_shapes.BASIC, has_shared_expert=True)
    layout = moe_test_lib.WeightLayout(use_ep=use_ep, use_sp=use_sp)
    self.check_matches_reference(shape, layout, reduce_output=False)

  def test_unreduced_output_with_remote_scatter_raises(self):
    with self.assertRaisesRegex(
        ValueError, "remote_scatter cannot be True when reduce_output is False"
    ):
      self.check_matches_reference(
          test_shapes.BASIC,
          moe_test_lib.WeightLayout(use_ep=True),
          remote_scatter=True,
          reduce_output=False,
      )

  @parameterized.product(activation=["silu", "swigluoai"])
  def test_activation(self, activation: str):
    shape = dataclasses.replace(test_shapes.BASIC, activation=activation)
    self.check_matches_reference(shape)

  @parameterized.product(
      w_dtype=[jnp.int8, jnp.float8_e5m2, jnp.float4_e2m1fn],
  )
  def test_sub_channel_quantization(self, w_dtype: jax.typing.DTypeLike):
    shape = dataclasses.replace(
        test_shapes.BASIC, w_dtype=jnp.dtype(w_dtype), quant_block_size=256
    )
    self.check_matches_reference(shape, moe_test_lib.WeightLayout(use_ep=True))

  def test_bias(self):
    shape = dataclasses.replace(
        test_shapes.BASIC, renormalize=False, has_bias=True
    )
    self.check_matches_reference(shape)
