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
"""Production-shape numerics and benchmarks for routing_gmm."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import sharding
import numpy as np
from tokamax._src.ops.experimental.routing_moe.gmm import kernel
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases
from tokamax._src.ops.experimental.routing_moe.test.common import test_inputs
from tokamax._src.ops.experimental.routing_moe.test.common import test_reference
from tokamax._src.ops.experimental.routing_moe.test.common import test_shapes
from tokamax._src.ops.experimental.routing_moe.test.common import test_util
from tokamax._src.ops.experimental.routing_moe.test.gmm import gmm_test_lib

P = sharding.PartitionSpec
NamedSharding = sharding.NamedSharding

jax.config.parse_flags_with_absl()


class RoutingGmmProdTest(parameterized.TestCase):

  def setUp(self):
    test_util.skip_unless_supported(self)
    super().setUp()

  def _check_and_log(
      self,
      fused: test_util.Timed,
      gmm_only: test_util.Timed,
      unfused: test_util.Timed,
      *,
      actual_rows: jax.Array | np.ndarray,
      expected_rows: jax.Array | np.ndarray,
  ) -> None:
    """Checks `fused` numerics against `gmm_only` and logs latencies."""
    actual = gmm_test_lib.to_large_layout(fused.fn(fused.args))[actual_rows]
    expected = gmm_only.fn(gmm_only.args)[expected_rows]
    np.testing.assert_allclose(actual, expected, rtol=5e-2, atol=8e-2)
    logging.info("gmm_only: %.1f us", gmm_only.latency_us())
    logging.info("unfused: %.1f us", unfused.latency_us())
    logging.info("fused: %.1f us %s", fused.latency_us(), fused.trace())

  @parameterized.product(case_spec=test_cases.PROD_CASES)
  def test_gather_gmm(self, case_spec: test_cases.Case):
    case = gmm_test_lib.GmmCase.make_case(case_spec)
    routing = case.routing
    self._check_and_log(
        fused=test_util.Timed.create(
            kernel.routing_gmm,
            case.x_compact,
            (case.w1,),
            gather=routing.gather,
            fuse_act="silu",
            **routing.kernel_kwargs,
        ),
        gmm_only=test_util.Timed.create(
            test_reference.reference_gmm,
            case.x_sorted,
            (case.w1,),
            routing.local_group_sizes,
            fuse_act="silu",
        ),
        unfused=test_util.Timed.create(
            test_reference.unfused_gather_gmm,
            case.x,
            case.w1,
            routing.local_group_sizes,
            routing.gather_indices,
        ),
        actual_rows=routing.sorted_rows,
        expected_rows=routing.sorted_rows,
    )

  @parameterized.product(case_spec=test_cases.PROD_CASES)
  def test_gmm_scatter(self, case_spec: test_cases.Case):
    case = gmm_test_lib.GmmCase.make_case(case_spec)
    routing = case.routing
    self._check_and_log(
        fused=test_util.Timed.create(
            kernel.routing_gmm,
            case.x_sorted_compact,
            (case.w1,),
            scatter=routing.scatter,
            **routing.kernel_kwargs,
        ),
        gmm_only=test_util.Timed.create(
            test_reference.reference_gmm,
            case.x_sorted,
            (case.w1,),
            routing.local_group_sizes,
        ),
        unfused=test_util.Timed.create(
            test_reference.unfused_gmm_scatter,
            case.x_sorted,
            case.w1,
            routing.local_group_sizes,
            routing.sorted_to_flat,
        ),
        actual_rows=routing.scattered_rows,
        expected_rows=routing.sorted_rows,
    )

  @parameterized.product(
      case_offset=[
          (test_cases.DECODE_1, 0),
          (test_cases.DECODE_16, 0),
          (test_cases.DECODE_1_EP8, 128),
          (test_cases.DECODE_64_EP8, 128),
      ],
  )
  def test_live_group_walk(self, case_offset: tuple[test_cases.Case, int]):
    """Times the metadata walk over all groups against only the live ones.

    Run after changing the metadata to check for regressions at decode.
    """
    case_spec, group_offset = case_offset
    case = gmm_test_lib.GmmCase.make_case(case_spec, group_offset=group_offset)
    routing = case.routing
    stages = {
        "gather": (
            case.x_compact,
            routing.sorted_rows,
            dict(gather=routing.gather, fuse_act="silu"),
        ),
        "scatter": (
            case.x_sorted_compact,
            routing.scattered_rows,
            dict(scatter=routing.scatter),
        ),
    }
    for name, (x, written, kwargs) in stages.items():
      outs = {}
      report = {}
      for live_group_walk in (False, True):
        timed = test_util.Timed.create(
            kernel.routing_gmm,
            x,
            (case.w1,),
            custom_plan=gmm_plan.RoutingGmmPlan(
                live_group_walk=live_group_walk
            ),
            **kwargs,
            **routing.kernel_kwargs,
        )
        outs[live_group_walk] = gmm_test_lib.to_large_layout(
            timed.fn(timed.args)
        )[written]
        report[live_group_walk] = (timed.latency_us(), timed.trace())
      np.testing.assert_array_equal(outs[True], outs[False])
      logging.info(
          "%s %s: all groups %.1f us %s, live groups %.1f us %s",
          case_spec.name,
          name,
          *report[False],
          *report[True],
      )
      # At one token the walk over 512 local groups is a large share of the
      # kernel (about 3 us of 20-25 us). With more tokens the matmul dominates,
      # and with 64 groups the gap is within noise, so those only log.
      if case_spec.num_local_experts >= 512 and case_spec.num_tokens == 1:
        self.assertLess(report[True][0], report[False][0])


class RemoteRoutingGmmProdTest(parameterized.TestCase):

  def setUp(self):
    test_util.skip_unless_supported(self, min_devices=4)
    super().setUp()

  @parameterized.product(num_tokens=[1280, 2560, 5120, 10240])
  def test_remote_vs_local_scatter(self, num_tokens: int):
    shape = test_shapes.REMOTE_SCATTER
    topk = shape.top_k
    moe_sharding = test_cases.MoESharding(ep=4)
    mesh = test_inputs.make_mesh(moe_sharding)
    topk_per_shard = topk // moe_sharding.ep
    case = gmm_test_lib.GmmCase.make_case(
        test_cases.Case("remote", shape, num_tokens),
        balanced_ep=moe_sharding.ep,
    )
    token = case.routing.sorted_to_flat // topk
    k = case.routing.sorted_to_flat % topk
    local_scatter_indices = token * topk_per_shard + k % topk_per_shard
    # Shard w up front so the timed functions do not reshard it.
    w = jax.device_put(
        case.w1.weight, NamedSharding(mesh, P(test_cases.EP_AXIS))
    )
    args = (case.x_sorted_compact, w, case.routing.group_sizes)
    local = test_util.Timed.create(
        gmm_test_lib.sharded_gmm_unpermute,
        *args,
        local_scatter_indices,
        mesh=mesh,
        topk=topk,
        remote_scatter=False,
        all_to_all=True,
    )
    local_kernel = test_util.Timed.create(
        gmm_test_lib.sharded_gmm_unpermute,
        *args,
        local_scatter_indices,
        mesh=mesh,
        topk=topk,
        remote_scatter=False,
        all_to_all=False,
    )
    remote = test_util.Timed.create(
        gmm_test_lib.sharded_gmm_unpermute,
        *args,
        case.routing.sorted_to_flat,
        mesh=mesh,
        remote_scatter=True,
    )
    np.testing.assert_array_equal(remote.fn(remote.args), local.fn(local.args))
    local_us = local.latency_us()
    kernel_us = local_kernel.latency_us()
    remote_us = remote.latency_us()
    with test_util.XprofTrace() as xprof:
      for timed in (local, remote):
        jax.block_until_ready(jax.jit(timed.fn)(timed.args))
    logging.info(
        "tokens=%d: local_total=%.0f us, local_kernel=%.0f us,"
        " local_a2a=%.0f us, remote=%.0f us (%.2fx) %s",
        num_tokens,
        local_us,
        kernel_us,
        local_us - kernel_us,
        remote_us,
        remote_us / local_us,
        xprof.links,
    )


if __name__ == "__main__":
  absltest.main()
