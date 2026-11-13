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
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe.gmm import kernel
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.test.gmm import gmm_test_lib
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_cases
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_inputs
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_reference
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_shapes
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_util

P = sharding.PartitionSpec
NamedSharding = sharding.NamedSharding

jax.config.parse_flags_with_absl()


class RoutingGmmProdTest(parameterized.TestCase):

  def setUp(self):
    helper_util.skip_unless_supported(self)
    super().setUp()

  def _check_and_log(
      self,
      fused: helper_util.Timed,
      gmm_only: helper_util.Timed,
      unfused: helper_util.Timed,
      *,
      actual_rows: jax.Array | np.ndarray,
      expected_rows: jax.Array | np.ndarray,
  ) -> None:
    """Checks `fused` numerics against `gmm_only` and logs latencies."""
    actual = gmm_test_lib.to_large_layout(fused.fn(fused.args))[actual_rows]
    expected = gmm_only.fn(gmm_only.args)[expected_rows]
    rtol = 5e-2
    atol = max(8e-2, rtol * float(np.abs(expected).max()))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    if helper_util.ROUTING_MOE_PROFILE.value:
      logging.info("gmm_only: %.1f us", gmm_only.latency_us())
      logging.info("unfused: %.1f us", unfused.latency_us())
      logging.info("fused: %.1f us %s", fused.latency_us(), fused.trace())

  @parameterized.product(case_spec=helper_cases.PROD_CASES)
  def test_gather_gmm(self, case_spec: helper_cases.Case):
    case = gmm_test_lib.GmmCase.make_case(case_spec)
    routing = case.routing
    # The kernel runs shard 0.
    helper_util.log_expert_load(
        case_spec.name, routing.group_sizes, ep=case_spec.sharding.ep
    )
    self._check_and_log(
        fused=helper_util.Timed.create(
            kernel.routing_gmm,
            case.x_compact,
            case.w1,
            gather=routing.gather,
            fuse_act="silu",
            **routing.kernel_kwargs,
        ),
        gmm_only=helper_util.Timed.create(
            helper_reference.reference_gmm,
            case.x_sorted,
            case.w1,
            routing.local_group_sizes,
            fuse_act="silu",
        ),
        unfused=helper_util.Timed.create(
            helper_reference.unfused_gather_gmm,
            case.x,
            case.w1,
            routing.local_group_sizes,
            routing.gather_indices,
        ),
        actual_rows=routing.sorted_rows,
        expected_rows=routing.sorted_rows,
    )

  @parameterized.product(case_spec=helper_cases.PROD_CASES)
  def test_gmm_scatter(self, case_spec: helper_cases.Case):
    case = gmm_test_lib.GmmCase.make_case(case_spec)
    routing = case.routing
    # The kernel runs shard 0.
    helper_util.log_expert_load(
        case_spec.name, routing.group_sizes, ep=case_spec.sharding.ep
    )
    self._check_and_log(
        fused=helper_util.Timed.create(
            kernel.routing_gmm,
            case.x_sorted_compact,
            case.w1,
            scatter=routing.scatter,
            **routing.kernel_kwargs,
        ),
        gmm_only=helper_util.Timed.create(
            helper_reference.reference_gmm,
            case.x_sorted,
            case.w1,
            routing.local_group_sizes,
        ),
        unfused=helper_util.Timed.create(
            helper_reference.unfused_gmm_scatter,
            case.x_sorted,
            case.w1,
            routing.local_group_sizes,
            routing.sorted_to_flat,
        ),
        actual_rows=routing.scattered_rows,
        expected_rows=routing.sorted_rows,
    )

  @parameterized.product(case_spec=helper_cases.PROD_CASES)
  def test_gather_gmm_gmm_scatter(self, case_spec: helper_cases.Case):
    """Times gather, gate/up, down, and scatter in one call."""
    case = gmm_test_lib.GmmCase.make_case(case_spec)
    routing = case.routing
    helper_util.log_expert_load(
        case_spec.name, routing.group_sizes, ep=case_spec.sharding.ep
    )
    self._check_and_log(
        fused=helper_util.Timed.create(
            kernel.routing_gmm,
            case.x_compact,
            case.w1,
            w2=case.w2,
            gather=routing.gather,
            scatter=routing.scatter,
            fuse_act="silu",
            **routing.kernel_kwargs,
        ),
        gmm_only=helper_util.Timed.create(
            helper_reference.reference_gmm,
            case.x_sorted,
            case.w1,
            routing.local_group_sizes,
            w2=case.w2,
            fuse_act="silu",
        ),
        unfused=helper_util.Timed.create(
            helper_reference.unfused_gather_gmm_gmm_scatter,
            case.x,
            case.w1,
            case.w2,
            routing.local_group_sizes,
            routing.gather_indices,
            routing.sorted_to_flat,
        ),
        actual_rows=routing.scattered_rows,
        expected_rows=routing.sorted_rows,
    )

  @parameterized.product(
      case_offset=[
          (helper_cases.DECODE_1, 0),
          (helper_cases.DECODE_16, 0),
          (helper_cases.DECODE_1_EP8, 128),
          (helper_cases.DECODE_64_EP8, 128),
      ],
  )
  def test_live_group_walk(self, case_offset: tuple[helper_cases.Case, int]):
    """Times the metadata walk over all groups against only the live ones.

    Run after changing the metadata to check for regressions at decode.
    """
    case_spec, group_offset = case_offset
    case = gmm_test_lib.GmmCase.make_case(case_spec, group_offset=group_offset)
    routing = case.routing
    calls = {
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
    check_latency = (
        case_spec.num_local_experts >= 512 and case_spec.num_tokens == 1
    )
    for name, (x, written, kwargs) in calls.items():
      timed = {
          walk: helper_util.Timed.create(
              kernel.routing_gmm,
              x,
              case.w1,
              custom_plan=gmm_plan.RoutingGmmPlan(live_group_walk=walk),
              **kwargs,
              **routing.kernel_kwargs,
          )
          for walk in (False, True)
      }
      outs = {
          walk: gmm_test_lib.to_large_layout(t.fn(t.args))[written]
          for walk, t in timed.items()
      }
      np.testing.assert_array_equal(outs[True], outs[False])
      # At one token the walk over 512 local groups is a large share of the
      # kernel (about 3 us of 20-25 us). With more tokens the matmul dominates,
      # and with 64 groups the gap is within noise, so those only log.
      if check_latency:
        self.assertLess(timed[True].latency_us(), timed[False].latency_us())
      if helper_util.ROUTING_MOE_PROFILE.value:
        logging.info(
            "%s %s: all groups %.1f us %s, live groups %.1f us %s",
            case_spec.name,
            name,
            timed[False].latency_us(),
            timed[False].trace(),
            timed[True].latency_us(),
            timed[True].trace(),
        )


class RemoteRoutingGmmProdTest(parameterized.TestCase):

  def setUp(self):
    helper_util.skip_unless_supported(self, min_devices=4)
    super().setUp()

  @parameterized.product(
      num_tokens=[1280, 2560, 5120, 10240],
      out_dtype=[None, gmm_test_lib.FP8],
  )
  def test_remote_vs_local_scatter(
      self, num_tokens: int, out_dtype: jax.typing.DTypeLike | None
  ):
    shape = helper_shapes.REMOTE_SCATTER
    topk = shape.top_k
    moe_sharding = helper_cases.MoESharding(ep=4)
    mesh = helper_inputs.make_mesh(moe_sharding)
    topk_per_shard = topk // moe_sharding.ep
    case = gmm_test_lib.GmmCase.make_case(
        helper_cases.Case("remote", shape, num_tokens),
        balanced_ep=moe_sharding.ep,
    )
    token = case.routing.sorted_to_flat // topk
    k = case.routing.sorted_to_flat % topk
    local_scatter_indices = token * topk_per_shard + k % topk_per_shard
    # Shard w up front so the timed functions do not reshard it.
    w = jax.device_put(
        case.w1.weight, NamedSharding(mesh, P(helper_cases.EP_AXIS))
    )
    args = (case.x_sorted_compact, w, case.routing.group_sizes)
    local = helper_util.Timed.create(
        gmm_test_lib.sharded_gmm_unpermute,
        *args,
        local_scatter_indices,
        mesh=mesh,
        topk=topk,
        remote_scatter=False,
        all_to_all=True,
        out_dtype=out_dtype,
    )
    local_kernel = helper_util.Timed.create(
        gmm_test_lib.sharded_gmm_unpermute,
        *args,
        local_scatter_indices,
        mesh=mesh,
        topk=topk,
        remote_scatter=False,
        all_to_all=False,
        out_dtype=out_dtype,
    )
    remote = helper_util.Timed.create(
        gmm_test_lib.sharded_gmm_unpermute,
        *args,
        case.routing.sorted_to_flat,
        mesh=mesh,
        remote_scatter=True,
        out_dtype=out_dtype,
    )
    # Bytes, not values: an fp8 scale chunk is f32 bits, some of them fp8 NaNs.
    np.testing.assert_array_equal(
        np.asarray(remote.fn(remote.args)).view(np.uint8),
        np.asarray(local.fn(local.args)).view(np.uint8),
    )
    if helper_util.ROUTING_MOE_PROFILE.value:
      local_us = local.latency_us()
      kernel_us = local_kernel.latency_us()
      remote_us = remote.latency_us()
      with helper_util.XprofTrace() as xprof:
        for timed in (local, remote):
          jax.block_until_ready(jax.jit(timed.fn)(timed.args))
      logging.info(
          "tokens=%d out=%s: local_total=%.0f us, local_kernel=%.0f us,"
          " local_a2a=%.0f us, remote=%.0f us (%.2fx) %s",
          num_tokens,
          jnp.dtype(out_dtype or jnp.bfloat16).name,
          local_us,
          kernel_us,
          local_us - kernel_us,
          remote_us,
          remote_us / local_us,
          xprof.links,
      )


if __name__ == "__main__":
  absltest.main()
