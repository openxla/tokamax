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
"""Unit tests for routing_gmm against the grouped matmul reference."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.gmm import kernel
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases
from tokamax._src.ops.experimental.routing_moe.test.common import test_inputs
from tokamax._src.ops.experimental.routing_moe.test.common import test_reference
from tokamax._src.ops.experimental.routing_moe.test.common import test_shapes
from tokamax._src.ops.experimental.routing_moe.test.common import test_util
from tokamax._src.ops.experimental.routing_moe.test.gmm import gmm_test_lib

jax.config.parse_flags_with_absl()


class RoutingGmmUnitTest(parameterized.TestCase):

  def setUp(self):
    test_util.skip_unless_supported(self)
    super().setUp()

  @parameterized.product(
      case_spec=test_cases.UNIT_CASES,
      tile_n=[None, 512],
  )
  def test_gather_gmm(self, case_spec: test_cases.Case, tile_n: int | None):
    case = gmm_test_lib.GmmCase.make_case(case_spec, quant_w1=True)
    routing = case.routing
    actual = kernel.routing_gmm(
        case.x_compact,
        case.w1,
        gather=routing.gather,
        fuse_act="silu",
        custom_plan=gmm_plan.RoutingGmmPlan(tile_n=tile_n),
        **routing.kernel_kwargs,
    )
    expected = test_reference.reference_gmm(
        case.x_sorted[routing.row_start : routing.row_end],
        case.w1,
        routing.local_group_sizes,
        fuse_act="silu",
    )
    np.testing.assert_allclose(
        gmm_test_lib.to_large_layout(actual)[routing.sorted_rows],
        expected,
        rtol=5e-2,
        atol=5e-2,
    )

  @parameterized.product(
      case_spec=test_cases.UNIT_CASES,
      tile_n=[None, 512],
  )
  def test_gmm_scatter(self, case_spec: test_cases.Case, tile_n: int | None):
    case = gmm_test_lib.GmmCase.make_case(case_spec)
    routing = case.routing
    actual = kernel.routing_gmm(
        case.x_sorted_compact,
        case.w1,
        scatter=routing.scatter,
        custom_plan=gmm_plan.RoutingGmmPlan(tile_n=tile_n),
        **routing.kernel_kwargs,
    )
    expected = test_reference.reference_gmm(
        case.x_sorted[routing.row_start : routing.row_end],
        case.w1,
        routing.local_group_sizes,
    )
    np.testing.assert_allclose(
        gmm_test_lib.to_large_layout(actual)[routing.scattered_rows],
        expected,
        rtol=2e-2,
        atol=1e-2,
    )

  @parameterized.product(
      case_spec=test_cases.PREFILL_CASES,
      quant=[(True, True), (False, False), (True, False)],
      fuse_up_and_down=[None, True, False],
  )
  def test_gather_gmm_gmm_scatter(
      self,
      case_spec: test_cases.Case,
      quant: tuple[bool, bool],
      fuse_up_and_down: bool | None,
  ):
    quant_w1, quant_w2 = quant
    if quant == (True, False) and fuse_up_and_down is not None:
      self.skipTest("Mixed precision is only checked with the default plan.")
    case = gmm_test_lib.GmmCase.make_case(
        case_spec, quant_w1=quant_w1, quant_w2=quant_w2
    )
    routing = case.routing
    actual = kernel.routing_gmm(
        case.x_compact,
        case.w1,
        w2=case.w2,
        gather=routing.gather,
        scatter=routing.scatter,
        fuse_act="silu",
        custom_plan=gmm_plan.RoutingGmmPlan(fuse_up_and_down=fuse_up_and_down),
        **routing.kernel_kwargs,
    )
    expected = test_reference.reference_gmm(
        case.x_sorted[routing.row_start : routing.row_end],
        case.w1,
        routing.local_group_sizes,
        w2=case.w2,
        fuse_act="silu",
    )
    np.testing.assert_allclose(
        gmm_test_lib.to_large_layout(actual)[routing.scattered_rows],
        expected,
        rtol=6e-2,
        atol=6e-2,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="w_lt_x",
          size_k=4096,
          size_n=2048,
          w_q_block_size=256,
          x_q_block_size=512,
      ),
      dict(
          testcase_name="w_eq_x",
          size_k=4096,
          size_n=2048,
          w_q_block_size=512,
          x_q_block_size=512,
      ),
      dict(
          testcase_name="w_gt_x",
          size_k=4096,
          size_n=2048,
          w_q_block_size=1024,
          x_q_block_size=512,
      ),
      # DeepSeek-V4 gate/up and down weights: hidden=7168, intermediate=2048.
      dict(
          testcase_name="dsv4_w1",
          size_k=7168,
          size_n=2 * 2048,
          w_q_block_size=512,
          x_q_block_size=512,
      ),
      dict(
          testcase_name="dsv4_w2",
          size_k=2048,
          size_n=7168,
          w_q_block_size=512,
          x_q_block_size=512,
      ),
      # Per-channel weights whose K is not a multiple of the x block size.
      dict(
          testcase_name="per_channel_k_not_multiple_of_x",
          size_k=768,
          size_n=2048,
          w_q_block_size=None,
          x_q_block_size=512,
      ),
  )
  def test_sub_channel_numerics(
      self,
      size_k: int,
      size_n: int,
      w_q_block_size: int | None,
      x_q_block_size: int,
  ):
    # size_n is the fused gate/up width, i.e. 2 * intermediate_size.
    shape = test_shapes.MoEShape(
        "sub_channel",
        num_experts=8,
        top_k=2,
        hidden_size=size_k,
        intermediate_size=size_n // 2,
        w_quant_block_size=w_q_block_size,
    )
    case = gmm_test_lib.GmmCase.make_case(
        test_cases.Case("sub_channel_256", shape, 256), quant_w1=True
    )
    routing = case.routing
    actual = kernel.routing_gmm(
        case.x_compact,
        case.w1,
        gather=routing.gather,
        fuse_act="silu",
        custom_plan=gmm_plan.RoutingGmmPlan(x_quant_block_size=x_q_block_size),
        **routing.kernel_kwargs,
    )
    expected = test_reference.reference_gmm(
        case.x_sorted[routing.row_start : routing.row_end],
        case.w1,
        routing.local_group_sizes,
        fuse_act="silu",
    )
    np.testing.assert_allclose(
        gmm_test_lib.to_large_layout(actual)[routing.sorted_rows],
        expected,
        rtol=5e-2,
        atol=5e-2,
    )

  @parameterized.product(
      w_q_block_size=[256, 1024],
      fuse_up_and_down=[None, True, False],
  )
  def test_sub_channel_gather_gmm_gmm_scatter(
      self, w_q_block_size: int, fuse_up_and_down: bool | None
  ):
    # Covers sub-channel scales on both w1 (K=hidden) and w2 (K=intermediate).
    case_spec = test_cases.PREFILL_8K
    shape = dataclasses.replace(
        case_spec.shape, w_quant_block_size=w_q_block_size
    )
    case = gmm_test_lib.GmmCase.make_case(
        dataclasses.replace(case_spec, shape=shape),
        quant_w1=True,
        quant_w2=True,
    )
    routing = case.routing
    actual = kernel.routing_gmm(
        case.x_compact,
        case.w1,
        w2=case.w2,
        gather=routing.gather,
        scatter=routing.scatter,
        fuse_act="silu",
        custom_plan=gmm_plan.RoutingGmmPlan(
            x_quant_block_size=512, fuse_up_and_down=fuse_up_and_down
        ),
        **routing.kernel_kwargs,
    )
    expected = test_reference.reference_gmm(
        case.x_sorted[routing.row_start : routing.row_end],
        case.w1,
        routing.local_group_sizes,
        w2=case.w2,
        fuse_act="silu",
    )
    np.testing.assert_allclose(
        gmm_test_lib.to_large_layout(actual)[routing.scattered_rows],
        expected,
        rtol=6e-2,
        atol=6e-2,
    )

  def test_fuse_up_and_down_without_w2_raises(self):
    case = gmm_test_lib.GmmCase.make_case(test_cases.DECODE_16)
    with self.assertRaisesRegex(ValueError, "fuse_up_and_down"):
      kernel.routing_gmm(
          case.x_compact,
          case.w1,
          gather=case.routing.gather,
          fuse_act="silu",
          custom_plan=gmm_plan.RoutingGmmPlan(fuse_up_and_down=True),
          **case.routing.kernel_kwargs,
      )

  def test_tile_k2_without_fusion_raises(self):
    case = gmm_test_lib.GmmCase.make_case(test_cases.DECODE_16)
    with self.assertRaisesRegex(ValueError, "tile_k2"):
      kernel.routing_gmm(
          case.x_compact,
          case.w1,
          gather=case.routing.gather,
          fuse_act="silu",
          custom_plan=gmm_plan.RoutingGmmPlan(tile_k2=512),
          **case.routing.kernel_kwargs,
      )
    with self.assertRaisesRegex(ValueError, "tile_k2"):
      kernel.routing_gmm(
          case.x_compact,
          case.w1,
          w2=case.w2,
          gather=case.routing.gather,
          scatter=case.routing.scatter,
          fuse_act="silu",
          custom_plan=gmm_plan.RoutingGmmPlan(
              tile_k2=512, fuse_up_and_down=False
          ),
          **case.routing.kernel_kwargs,
      )

  @parameterized.named_parameters(
      ("skip", gmm_plan.RoutingGmmPlan(skip_metadata_size=2)),
      ("live_group_walk", gmm_plan.RoutingGmmPlan(live_group_walk=True)),
      ("sorted_walk", gmm_plan.RoutingGmmPlan(live_group_walk=False)),
  )
  def test_metadata_plan_matches_default(
      self, custom_plan: gmm_plan.RoutingGmmPlan
  ):
    # The walks themselves are checked in metadata_unit_test. This checks that
    # the kernel plumbs row_group and the plan through to the same output.
    case = gmm_test_lib.GmmCase.make_case(
        test_cases.DECODE_64_EP8, group_offset=128
    )
    calls = (
        (case.x_compact, dict(gather=case.routing.gather, fuse_act="silu")),
        (case.x_sorted_compact, dict(scatter=case.routing.scatter)),
    )
    rows = (case.routing.sorted_rows, case.routing.scattered_rows)
    for (x, kwargs), written in zip(calls, rows, strict=True):
      kwargs = kwargs | case.routing.kernel_kwargs
      expected = kernel.routing_gmm(x, case.w1, **kwargs)
      actual = kernel.routing_gmm(x, case.w1, custom_plan=custom_plan, **kwargs)
      np.testing.assert_array_equal(
          gmm_test_lib.to_large_layout(actual)[written],
          gmm_test_lib.to_large_layout(expected)[written],
      )

  def test_negative_skip_metadata_size(self):
    case = gmm_test_lib.GmmCase.make_case(test_cases.DECODE_16)
    with self.assertRaisesRegex(ValueError, "skip_metadata_size"):
      kernel.routing_gmm(
          case.x_compact,
          case.w1,
          gather=case.routing.gather,
          fuse_act="silu",
          custom_plan=gmm_plan.RoutingGmmPlan(skip_metadata_size=-1),
          **case.routing.kernel_kwargs,
      )


class RemoteRoutingGmmUnitTest(parameterized.TestCase):

  def setUp(self):
    test_util.skip_unless_supported(self, min_devices=4)
    super().setUp()

  @parameterized.product(
      sharding=[
          test_cases.MoESharding(ep=4),
          test_cases.MoESharding(dp=2, ep=2),
      ],
      num_tokens_topk=[(256, 2), (1024, 4)],
      skip_last_shard=[False, True],
  )
  def test_remote_scatter_matches_single_device(
      self,
      sharding: test_cases.MoESharding,
      num_tokens_topk: tuple[int, int],
      skip_last_shard: bool,
  ):
    num_tokens, topk = num_tokens_topk
    shape = test_shapes.MoEShape("remote", 16, topk, 1024, 256)
    mesh = test_inputs.make_mesh(sharding)
    num_local = shape.num_experts // sharding.ep
    case = gmm_test_lib.GmmCase.make_case(
        test_cases.Case("remote", shape, num_tokens),
        num_skipped=num_local if skip_last_shard else 0,
    )
    w = case.w1.weight
    expected = kernel.routing_gmm(
        case.x_sorted_compact,
        arg.make_weight_wrapper(w),
        case.routing.group_sizes,
        scatter=case.routing.scatter,
    ).data
    actual = gmm_test_lib.sharded_gmm_unpermute(
        case.x_sorted_compact,
        w,
        case.routing.group_sizes,
        case.routing.sorted_to_flat,
        mesh=mesh,
        remote_scatter=True,
    )
    np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
  absltest.main()
