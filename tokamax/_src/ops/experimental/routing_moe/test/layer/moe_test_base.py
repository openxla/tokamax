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

_UNIT_SHARDINGS = (
    test_cases.MoESharding(ep=2),
    test_cases.MoESharding(tp=2),
    test_cases.MoESharding(ep=2, tp=2),
)


class MoETestCase(parameterized.TestCase):
  """Base test case for verifying MoE layer execution against XLA reference."""

  mode: moe_test_lib.Mode = "modular"

  def setUp(self):
    test_util.skip_unless_supported(self)
    super().setUp()

  def check_case(
      self,
      case: test_cases.Case,
      layout: moe_test_lib.WeightLayout | None = None,
      *,
      remote_scatter: bool = False,
      reduce_output: bool = True,
      capture_xprof: bool = False,
  ) -> None:
    """Checks that the execution mode matches numerics on `case`."""
    sharding = case.sharding if layout is None else layout.sharding
    test_util.skip_unless_supported(self, min_devices=sharding.num_cores)
    moe_case = moe_test_lib.MoECase.make_case(
        case, layout, remote_scatter=remote_scatter
    )
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
    self.check_case(case)

  @parameterized.product(renormalize=[True, False])
  def test_basic(self, renormalize: bool):
    shape = dataclasses.replace(test_shapes.BASIC, renormalize=renormalize)
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
    shape = dataclasses.replace(test_shapes.BASIC, w_dtype=dtype)
    layout = moe_test_lib.WeightLayout(
        test_cases.MoESharding(ep=2, tp=2), use_sp=use_sp
    )
    self.check_matches_reference(shape, layout, remote_scatter=remote_scatter)

  @parameterized.product(use_sp=[True, False])
  def test_remote_scatter(self, use_sp: bool):
    layout = moe_test_lib.WeightLayout(
        test_cases.MoESharding(ep=2), use_sp=use_sp
    )
    self.check_matches_reference(test_shapes.BASIC, layout, remote_scatter=True)

  @parameterized.product(sharding=_UNIT_SHARDINGS, use_sp=[True, False])
  def test_unreduced_output(
      self, sharding: test_cases.MoESharding, use_sp: bool
  ):
    layout = moe_test_lib.WeightLayout(sharding, use_sp=use_sp)
    self.check_matches_reference(test_shapes.BASIC, layout, reduce_output=False)

  @parameterized.product(sharding=_UNIT_SHARDINGS, use_sp=[True, False])
  def test_shared_expert(self, sharding: test_cases.MoESharding, use_sp: bool):
    shape = dataclasses.replace(test_shapes.BASIC, has_shared_expert=True)
    layout = moe_test_lib.WeightLayout(sharding, use_sp=use_sp)
    self.check_matches_reference(shape, layout, reduce_output=False)

  def test_unreduced_output_with_remote_scatter_raises(self):
    with self.assertRaisesRegex(
        ValueError, "remote_scatter cannot be True when reduce_output is False"
    ):
      self.check_matches_reference(
          test_shapes.BASIC,
          moe_test_lib.WeightLayout(test_cases.MoESharding(ep=2)),
          remote_scatter=True,
          reduce_output=False,
      )

  def test_remote_scatter_without_ep_raises(self):
    with self.assertRaisesRegex(ValueError, "remote_scatter requires ep_axis"):
      self.check_matches_reference(test_shapes.BASIC, remote_scatter=True)

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
    self.check_matches_reference(shape)

  def test_bias(self):
    shape = dataclasses.replace(
        test_shapes.BASIC, renormalize=False, has_bias=True
    )
    self.check_matches_reference(shape)
