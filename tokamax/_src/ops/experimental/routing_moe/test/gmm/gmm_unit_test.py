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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.gmm import kernel
from tokamax._src.ops.experimental.routing_moe.gmm import memory_ref
from tokamax._src.ops.experimental.routing_moe.gmm import metadata
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases
from tokamax._src.ops.experimental.routing_moe.test.common import test_inputs
from tokamax._src.ops.experimental.routing_moe.test.common import test_reference
from tokamax._src.ops.experimental.routing_moe.test.common import test_shapes
from tokamax._src.ops.experimental.routing_moe.test.common import test_util
from tokamax._src.ops.experimental.routing_moe.test.gmm import gmm_test_lib
from tokamax._src.ops.experimental.routing_moe.test.gmm import metadata_test_lib

jax.config.parse_flags_with_absl()

FP8 = gmm_test_lib.FP8


def _rel_diff(a: np.ndarray, b: np.ndarray) -> float:
  """Returns the maximum elementwise error normalized by `max(|b|)`."""
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6))


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

  @parameterized.product(case_spec=test_cases.UNIT_CASES)
  def test_gmm_scatter_fp8(self, case_spec: test_cases.Case):
    case = gmm_test_lib.GmmCase.make_case(case_spec)
    routing = case.routing
    actual = kernel.routing_gmm(
        case.x_sorted_compact,
        case.w1,
        scatter=routing.scatter,
        out_dtype=FP8,
        **routing.kernel_kwargs,
    ).data
    self.assertEqual(actual.dtype, FP8)
    expected = test_reference.reference_gmm(
        case.x_sorted[routing.row_start : routing.row_end],
        case.w1,
        routing.local_group_sizes,
    )
    rows = np.asarray(routing.scattered_rows)
    deq = np.asarray(gmm_test_lib.dequantize_rows(actual))[rows]
    expected = np.asarray(expected, np.float32)
    self.assertLessEqual(_rel_diff(deq, expected), 2**-4)
    # Each lane's largest value lands on the fp8 max.
    payload, _ = gmm_test_lib.split_scaled_rows(actual)
    payload = np.asarray(payload, np.float32)[rows]
    lane_max = np.abs(payload.reshape(len(rows), -1, 128)).max(axis=1)
    np.testing.assert_array_equal(lane_max, float(jnp.finfo(FP8).max))

  def test_gmm_scatter_fp8_zero_row_has_unit_scale(self):
    # A zero row has a zero absmax, so its scale should be 1 instead of 0.
    case = gmm_test_lib.GmmCase.make_case(test_cases.TOP1_200)
    routing = case.routing
    zeros = 4
    actual = kernel.routing_gmm(
        input_pack.input_pack(case.x_sorted.at[:zeros].set(0)),
        case.w1,
        scatter=routing.scatter,
        out_dtype=FP8,
        **routing.kernel_kwargs,
    ).data
    rows = np.asarray(routing.scattered_rows[:zeros])
    _, scales = gmm_test_lib.split_scaled_rows(actual)
    np.testing.assert_array_equal(np.asarray(scales)[rows], 1.0)
    np.testing.assert_array_equal(
        np.asarray(gmm_test_lib.dequantize_rows(actual))[rows], 0
    )

  def test_gmm_scatter_fp8_large_values(self):
    # Outputs far above the fp8 max need scales above 1, so `1 / scale` is
    # below 1 and any error in it is a large relative error.
    case = gmm_test_lib.GmmCase.make_case(test_cases.TOP1_200)
    routing = case.routing
    x = case.x_sorted * 2**13
    actual = kernel.routing_gmm(
        input_pack.input_pack(x),
        case.w1,
        scatter=routing.scatter,
        out_dtype=FP8,
        **routing.kernel_kwargs,
    ).data
    expected = test_reference.reference_gmm(
        x[routing.row_start : routing.row_end],
        case.w1,
        routing.local_group_sizes,
    )
    expected = np.asarray(expected, np.float32)
    rows = np.asarray(routing.scattered_rows)
    _, scales = gmm_test_lib.split_scaled_rows(actual)
    self.assertGreater(float(np.asarray(scales)[rows].min()), 1.0)
    deq = np.asarray(gmm_test_lib.dequantize_rows(actual))[rows]
    self.assertLessEqual(_rel_diff(deq, expected), 2**-4)

  def test_gmm_scatter_fp8_topk_sum(self):
    case_spec = test_cases.PREFILL_16E
    case = gmm_test_lib.GmmCase.make_case(case_spec)
    routing = case.routing
    num_tokens, topk = case_spec.num_tokens, routing.topk
    self.assertEqual(routing.row_end - routing.row_start, num_tokens * topk)
    routed = test_reference.reference_gmm(
        case.x_sorted, case.w1, routing.local_group_sizes
    )
    # Output row `t * topk + k` is token t's k-th expert.
    want = np.empty(routed.shape, np.float32)
    want[np.asarray(routing.scattered_rows)] = np.asarray(routed, np.float32)
    actual = kernel.routing_gmm(
        case.x_sorted_compact,
        case.w1,
        scatter=routing.scatter,
        out_dtype=FP8,
        **routing.kernel_kwargs,
    ).data
    got = np.asarray(gmm_test_lib.dequantize_rows(actual))
    want = want.reshape(num_tokens, topk, -1).sum(axis=1)
    got = got.reshape(num_tokens, topk, -1).sum(axis=1)
    err = np.abs(got - want)
    max_rel = float(err.max() / np.abs(want).max())
    mean_rel = float(err.mean() / np.abs(want).mean())
    logging.info("fp8 topk sum: max rel %.4f, mean rel %.4f", max_rel, mean_rel)
    self.assertLessEqual(max_rel, 2**-4)

  def test_quantized_buffered_ref_input_raises(self):
    cfg = metadata_test_lib.make_gmm_config(
        num_rows=128,
        num_global_groups=8,
        num_local_groups=8,
        size_k=512,
        size_n=512,
    )
    meta_ref = metadata.SortedMetadataRef.create_scratch_types(cfg)
    bref = memory_ref.QuantizedBufferedRef.input(
        spec=pl.BlockSpec((16, 4, 4, 128), lambda i, j, k: (0, 0, 0, 0)),
        dtype_or_type=FP8,
        cfg=cfg,
        metadata_ref=meta_ref,
    )
    with self.assertRaisesRegex(AssertionError, "branchless_copy_in"):
      bref.branchless_copy_in(meta_ref.gm_id_to_num_rows, (0, 0, 0))
    with self.assertRaisesRegex(AssertionError, "read_vmem"):
      bref.read_vmem(meta_ref.gm_id_to_num_rows, (0, 0, 0))

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

  @parameterized.product(
      case_offset=[
          (test_cases.DECODE_1, 0),
          (test_cases.DECODE_16, 0),
          (test_cases.DECODE_1_EP8, 128),
          (test_cases.DECODE_64_EP8, 128),
      ],
      tile_n=[None, 512],
  )
  def test_resident_idx(
      self, case_offset: tuple[test_cases.Case, int], tile_n: int | None
  ):
    # Resident index tables only change where each tile reads its row indices
    # from, so outputs must match the streamed ones.
    case_spec, group_offset = case_offset
    case = gmm_test_lib.GmmCase.make_case(case_spec, group_offset=group_offset)
    stages = (
        (case.x_compact, dict(gather=case.routing.gather, fuse_act="silu")),
        (case.x_sorted_compact, dict(scatter=case.routing.scatter)),
    )
    rows = (case.routing.sorted_rows, case.routing.scattered_rows)
    for (x, kwargs), written in zip(stages, rows, strict=True):
      kwargs = kwargs | case.routing.kernel_kwargs
      actual, expected = (
          kernel.routing_gmm(
              x,
              case.w1,
              custom_plan=gmm_plan.RoutingGmmPlan(
                  tile_n=tile_n, resident_idx=resident_idx
              ),
              **kwargs,
          )
          for resident_idx in (True, False)
      )
      np.testing.assert_array_equal(
          gmm_test_lib.to_large_layout(actual)[written],
          gmm_test_lib.to_large_layout(expected)[written],
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
      out_dtype=[None, FP8],
  )
  def test_remote_scatter_matches_single_device(
      self,
      sharding: test_cases.MoESharding,
      num_tokens_topk: tuple[int, int],
      skip_last_shard: bool,
      out_dtype: jnp.dtype | None,
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
        out_dtype=out_dtype,
    ).data
    actual = gmm_test_lib.sharded_gmm_unpermute(
        case.x_sorted_compact,
        w,
        case.routing.group_sizes,
        case.routing.sorted_to_flat,
        mesh=mesh,
        remote_scatter=True,
        out_dtype=out_dtype,
    )
    self.assertEqual(actual.dtype, expected.dtype)
    # Bytes, not values: an fp8 scale chunk is f32 bits, some of them fp8 NaNs.
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint8), np.asarray(expected).view(np.uint8)
    )


if __name__ == "__main__":
  absltest.main()
