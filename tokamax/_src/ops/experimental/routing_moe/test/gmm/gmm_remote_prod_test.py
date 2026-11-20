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

import dataclasses

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import sharding
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe.common import arg
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

_TEST_CASES = tuple(
    helper_cases.Case(
        f"glm_5_3_{num_tokens}_ep4",
        helper_shapes.GLM_5_3,
        num_tokens,
        helper_cases.MoESharding(ep=4),
    )
    for num_tokens in (8, 256, 8192)
)


def _shard_weight(
    w: arg.SingleWeightWrapper[jax.Array], mesh: sharding.Mesh
) -> arg.SingleWeightWrapper[jax.Array]:
  """Places `w` (and its scale) sharded by expert along the EP axis."""
  ep_sharding = NamedSharding(mesh, P(helper_cases.EP_AXIS))
  scale = None if w.scale is None else jax.device_put(w.scale, ep_sharding)
  return arg.SingleWeightWrapper.create(
      jax.device_put(w.weight, ep_sharding),
      scale,
      fused_gate_up=w.spec.fused_gate_up,
  )


@jax.jit(static_argnames=("mesh", "custom_plan"))
def _remote_unpermute_gmm(
    x: arg.ActivationWrapper[jax.Array],
    w1: arg.SingleWeightWrapper[jax.Array],
    w2: arg.SingleWeightWrapper[jax.Array],
    group_sizes: jax.Array,
    gather_indices: jax.Array,
    scatter_indices: jax.Array,
    *,
    mesh: sharding.Mesh,
    custom_plan: gmm_plan.RoutingGmmPlan | None = None,
) -> jax.Array:
  """Runs gather -> gate/up -> down -> remote scatter on every EP shard.

  Each shard holds `num_experts // ep` experts and DMAs every output row it
  computes straight to the device that owns the row's flat slot. `routing_gmm`
  needs the EP axis to exist, which is why this runs under `shard_map`.

  Args:
    x: Unsharded token activations in the compact layout.
    w1: Gate/up weights of every expert, sharded along the expert dim.
    w2: Down weights of every expert, sharded along the expert dim.
    group_sizes: Row count per global expert.
    gather_indices: Token row of each sorted row.
    scatter_indices: Flat output slot of each sorted row.
    mesh: Mesh with the EP axis.
    custom_plan: Optional plan overriding the heuristics.

  Returns:
    The scattered output rows in the compact layout, sharded along the EP axis.
  """
  axis = helper_cases.EP_AXIS
  ep_size = mesh.shape[axis]
  num_experts = w1.weight.shape[0]
  group_offset = jnp.arange(0, num_experts, num_experts // ep_size)
  w1_scale_spec = None if w1.scale is None else P(axis)
  w2_scale_spec = None if w2.scale is None else P(axis)

  def run_shard(
      x: arg.ActivationWrapper[jax.Array],
      w1_weight: jax.Array,
      w1_scale: jax.Array | None,
      w2_weight: jax.Array,
      w2_scale: jax.Array | None,
      group_sizes: jax.Array,
      offset: jax.Array,
      gather_idx: jax.Array,
      scatter_idx: jax.Array,
  ) -> jax.Array:
    return kernel.routing_gmm(
        x,
        arg.SingleWeightWrapper.create(w1_weight, w1_scale, fused_gate_up=True),
        group_sizes,
        w2=arg.SingleWeightWrapper.create(w2_weight, w2_scale),
        group_offset=offset,
        gather=arg.RoutingWrapper.create(gather_idx),
        scatter=arg.RoutingWrapper.create(scatter_idx, remote_axis=axis),
        fuse_act="silu",
        custom_plan=custom_plan,
    ).data

  return jax.shard_map(
      run_shard,
      mesh=mesh,
      in_specs=(
          P(),
          P(axis),
          w1_scale_spec,
          P(axis),
          w2_scale_spec,
          P(),
          P(axis),
          P(),
          P(),
      ),
      out_specs=P(axis),
      check_vma=False,
  )(
      x,
      w1.weight,
      w1.scale,
      w2.weight,
      w2.scale,
      group_sizes,
      group_offset,
      gather_indices,
      scatter_indices,
  )


class RemoteRoutingGmmProdTest(parameterized.TestCase):

  def setUp(self):
    helper_util.skip_unless_supported(self, min_devices=4)
    super().setUp()

  @parameterized.product(case_spec=_TEST_CASES)
  def test_remote_unpermute_fused_vs_unfused(
      self, case_spec: helper_cases.Case
  ):
    """Times remote unpermute with the fused and the unfused kernels."""
    ep = case_spec.sharding.ep
    if jax.device_count() < ep:
      self.skipTest(f"Requires {ep} devices for EP={ep}.")
    mesh = helper_inputs.make_mesh(case_spec.sharding)
    # The weights of every expert; the runner shards them along the EP axis.
    case = gmm_test_lib.GmmCase.make_case(
        dataclasses.replace(case_spec, sharding=helper_cases.MoESharding())
    )
    routing = case.routing
    helper_util.log_expert_load(case_spec.name, routing.group_sizes, ep=ep)
    # Shard the weights up front so the timed functions do not reshard them.
    args = (
        case.x_compact,
        _shard_weight(case.w1, mesh),
        _shard_weight(case.w2, mesh),
        routing.group_sizes,
        routing.gather_indices,
        routing.sorted_to_flat,
    )
    expected = helper_reference.reference_gmm(
        case.x_sorted,
        case.w1,
        routing.local_group_sizes,
        w2=case.w2,
        fuse_act="silu",
    )
    # The kernel quantizes x and the intermediate to fp8 while the reference
    # keeps them in bf16, so as in the layer tests the tolerance scales with
    # the output range.
    rtol = 5e-2
    atol = rtol * float(jnp.abs(expected).max())

    plans = {
        "fused": gmm_plan.RoutingGmmPlan(fuse_up_and_down=True),
        "unfused": gmm_plan.RoutingGmmPlan(fuse_up_and_down=False),
    }
    report = []
    for name, custom_plan in plans.items():
      timed = helper_util.Timed.create(
          _remote_unpermute_gmm, *args, mesh=mesh, custom_plan=custom_plan
      )
      out = timed.fn(timed.args)
      actual = out.reshape(routing.sorted_to_flat.size, -1)
      np.testing.assert_allclose(
          actual[routing.scattered_rows], expected, rtol=rtol, atol=atol
      )
      report.append(f"{name}: {timed.latency_us():.1f} us {timed.trace()}")
    logging.info("%s ep=%d: %s", case_spec.name, ep, "; ".join(report))


if __name__ == "__main__":
  absltest.main()
