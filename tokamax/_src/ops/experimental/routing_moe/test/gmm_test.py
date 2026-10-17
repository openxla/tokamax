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

import math

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import sharding
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src import benchmarking
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.gmm import kernel as gmm
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.io_relayout import input_pack
from tokamax._src.ops.gmm_v2 import gmm_v2
from tokamax._src.ops.gmm_v2 import util as gmm_util
from tokamax._src.ops.ragged_gather.pallas_mosaic_tpu_kernel import ragged_gather_pallas

P = sharding.PartitionSpec
NamedSharding = sharding.NamedSharding
Mesh = sharding.Mesh

jax.config.parse_flags_with_absl()

SHAPES = [
    dict(
        testcase_name="prefill_64k",
        num_tokens=65536,
        size_k=4096,
        size_n=2048,
        num_experts=512,
        ep_size=8,
        topk=10,
    ),
    dict(
        testcase_name="prefill_8k",
        num_tokens=8192,
        size_k=4096,
        size_n=2048,
        num_experts=512,
        ep_size=8,
        topk=10,
    ),
    dict(
        testcase_name="top1_200",
        num_tokens=200,
        size_k=4096,
        size_n=2048,
        num_experts=1,
        ep_size=1,
        topk=1,
    ),
]


def ragged_permute_gmm(lhs, rhs, rhs_scale, group_sizes, gather_indices):
  lhs = ragged_gather_pallas(
      lhs, gather_indices, jnp.array([0], jnp.int32), group_sizes.sum()[None]
  )
  return gmm_v2.gmm_v2(
      lhs,
      rhs,
      group_sizes,
      rhs_scale=rhs_scale,
      zero_initialize=False,
      fuse_act="silu",
  )


def ragged_gmm_unpermute(lhs, rhs, group_sizes, topk_argsort_indices):
  out = gmm_v2.gmm_v2(lhs, rhs, group_sizes, zero_initialize=False)
  mask = jnp.arange(out.shape[0]) < group_sizes.sum()
  out = jnp.where(mask[:, None], out, 0)
  return jnp.zeros_like(out).at[topk_argsort_indices].set(out)


EP_AXIS = "model"


def make_mesh(mesh_shape):
  """Mesh over the first `math.prod(mesh_shape)` devices; last axis is EP."""
  devices = jax.devices()[: math.prod(mesh_shape)]
  return Mesh(
      np.array(devices).reshape(mesh_shape),
      axis_names=("data", EP_AXIS)[-len(mesh_shape) :],
  )


def random_topk_indices(key, num_tokens, topk, num_experts, num_skipped):
  """Top-k experts per token, never picking the last `num_skipped` experts."""
  gating_output = jax.random.uniform(key, (num_tokens, num_experts))
  gating_output = gating_output.at[:, num_experts - num_skipped :].set(-1)
  _, topk_indices = jax.lax.top_k(gating_output, topk)
  return topk_indices


def balanced_topk_indices(key, num_tokens, topk, num_experts, ep_size):
  """Top-k experts per token; the k-th is on EP shard `k // (topk / ep_size)`."""
  num_local_experts = num_experts // ep_size
  gating_output = jax.random.uniform(
      key, (num_tokens, ep_size, num_local_experts)
  )
  _, local_indices = jax.lax.top_k(gating_output, topk // ep_size)
  shard_offsets = jnp.arange(ep_size)[:, None] * num_local_experts
  return (local_indices + shard_offsets).reshape(num_tokens, topk)


def _routing_gmm_shard(lhs, rhs, group_sizes, group_offset, scatter_indices):
  """Per-shard body of `remote_gmm_unpermute`."""
  out = gmm.routing_gmm(
      lhs,
      arg.make_weight_wrapper(rhs),
      group_sizes,
      group_offset=group_offset,
      scatter=arg.make_routing_wrapper(scatter_indices, remote_axis=EP_AXIS),
  )
  return out.data


@jax.jit(static_argnames=("mesh",))
def remote_gmm_unpermute(lhs, rhs, group_sizes, topk_argsort_indices, *, mesh):
  """Each EP shard DMAs its rows straight to the shard that owns the token."""
  num_experts = rhs.shape[0]
  num_local_experts = num_experts // mesh.shape[EP_AXIS]
  group_offset = jnp.arange(0, num_experts, num_local_experts)

  return jax.shard_map(
      _routing_gmm_shard,
      mesh=mesh,
      in_specs=(P(), P(EP_AXIS), P(), P(EP_AXIS), P()),
      out_specs=P(EP_AXIS),
      check_vma=False,
  )(lhs, rhs, group_sizes, group_offset, topk_argsort_indices)


@jax.jit(static_argnames=("mesh", "topk", "all_to_all"))
def local_gmm_unpermute(
    lhs, rhs, group_sizes, local_scatter_indices, *, mesh, topk, all_to_all=True
):
  """Each EP shard unpermutes locally, then an all_to_all reshards by token.

  Each shard writes its `topk / ep_size` rows per token to a dense
  `[num_tokens, topk / ep_size]` block. The all_to_all splits the block by
  token and stacks it by k. `all_to_all=False` skips it to time the kernel.
  """
  ep_size = mesh.shape[EP_AXIS]
  num_experts = rhs.shape[0]
  group_offset = jnp.arange(0, num_experts, num_experts // ep_size)

  def unpermute(lhs, rhs, group_sizes, group_offset, scatter_indices):
    out = gmm.routing_gmm(
        lhs,
        arg.make_weight_wrapper(rhs),
        group_sizes,
        group_offset=group_offset,
        scatter=arg.make_routing_wrapper(scatter_indices),
    ).data
    if not all_to_all:
      return out
    num_tokens, topk_per_shard = lhs.spec.row_size // topk, topk // ep_size
    row_shape = out.shape[1:]
    out = out[: num_tokens * topk_per_shard]
    out = out.reshape((num_tokens, topk_per_shard) + row_shape)
    out = jax.lax.all_to_all(
        out, EP_AXIS, split_axis=0, concat_axis=1, tiled=True
    )
    return out.reshape((-1,) + row_shape)

  return jax.shard_map(
      unpermute,
      mesh=mesh,
      in_specs=(P(), P(EP_AXIS), P(), P(EP_AXIS), P()),
      out_specs=P(EP_AXIS),
      check_vma=False,
  )(lhs, rhs, group_sizes, group_offset, local_scatter_indices)


class RoutingGmmTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 7:
      self.skipTest("Only tested on TPUv7x for now.")
    if jax.__version_info__ < (0, 11, 2):
      self.skipTest("routing_gmm requires JAX >= 0.11.2.")
    super().setUp()

  def _setup_inputs(
      self, num_tokens, size_k, size_n, num_experts, ep_size, topk, **_
  ):
    num_local_experts = num_experts // ep_size

    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    self.lhs = jax.random.normal(k0, (num_tokens, size_k), jnp.bfloat16) / 10
    self.rhs = (
        jax.random.normal(k1, (num_local_experts, size_k, size_n), jnp.bfloat16)
        / 10
    )
    self.rhs_q, rhs_scale = gmm_util.quantize_tensor(
        self.rhs, jnp.float8_e4m3fn, axis=1, block_size=size_k
    )
    self.rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    gating_output = jax.random.uniform(k2, (num_tokens, num_experts))
    _, topk_indices = jax.lax.top_k(gating_output, topk)
    topk_indices_flat = topk_indices.flatten()
    self.topk_argsort_indices = jnp.argsort(topk_indices_flat)
    self.gather_indices = self.topk_argsort_indices // topk
    self.group_sizes = jnp.bincount(topk_indices_flat, length=num_experts)[
        :num_local_experts
    ]
    self.num_valid_rows = int(self.group_sizes.sum())
    self.lhs_gathered = self.lhs[self.gather_indices]

    # routing_gmm reads lhs in the compact layout.
    self.lhs_compact = input_pack.input_pack(self.lhs)
    self.lhs_gathered_compact = input_pack.input_pack(self.lhs_gathered)

    # routing_gmm takes weights and indices with their specs attached.
    self.rhs_wrapped = arg.make_weight_wrapper(self.rhs)
    self.rhs_q_wrapped = arg.make_weight_wrapper(
        self.rhs_q, self.rhs_scale, fused_gate_up=True
    )
    self.gather = arg.make_routing_wrapper(self.gather_indices)
    self.scatter = arg.make_routing_wrapper(self.topk_argsort_indices)

  def _benchmark_and_profile(self, plain, ragged, fused):
    latency_ms = {}
    for name, (fn, args) in [
        ("plain", plain),
        ("ragged", ragged),
        ("fused", fused),
    ]:
      latency_ms[name] = benchmarking.benchmark(
          fn, args, method="hermetic_xprof"
      ).median_evaluation_time_ms
      logging.info("%s: %.3f ms", name, latency_ms[name])
    self.assertLess(latency_ms["fused"], latency_ms["ragged"])

  @parameterized.product(
      SHAPES,
      tile_n=[None, 512],
  )
  def test_permute_gmm_numerics(self, tile_n, **shape):
    self._setup_inputs(**shape)
    expected = gmm_v2.gmm_v2(
        self.lhs_gathered,
        self.rhs_q,
        self.group_sizes,
        rhs_scale=self.rhs_scale,
        zero_initialize=False,
        fuse_act="silu",
    )
    actual = gmm.routing_gmm(
        self.lhs_compact,
        self.rhs_q_wrapped,
        self.group_sizes,
        gather=self.gather,
        fuse_act="silu",
        custom_plan=gmm_plan.RoutingGmmPlan(tile_n=tile_n),
    )
    actual = actual.data.reshape(actual.spec.large_shape)
    np.testing.assert_allclose(
        actual[: self.num_valid_rows],
        expected[: self.num_valid_rows],
        rtol=5e-2,
        atol=5e-2,
    )

  @parameterized.product(
      SHAPES,
      tile_n=[None, 512],
  )
  def test_gmm_unpermute_numerics(self, tile_n, **shape):
    self._setup_inputs(**shape)
    expected = gmm_v2.gmm_v2(
        self.lhs_gathered, self.rhs, self.group_sizes, zero_initialize=False
    )
    actual = gmm.routing_gmm(
        self.lhs_gathered_compact,
        self.rhs_wrapped,
        self.group_sizes,
        scatter=self.scatter,
        custom_plan=gmm_plan.RoutingGmmPlan(tile_n=tile_n),
    )
    actual = actual.data.reshape(actual.spec.large_shape)
    np.testing.assert_allclose(
        actual[self.topk_argsort_indices[: self.num_valid_rows]],
        expected[: self.num_valid_rows],
        rtol=2e-2,
        atol=1e-2,
    )

  @parameterized.named_parameters(*SHAPES)
  def test_permute_gmm_benchmark(self, **shape):
    self._setup_inputs(**shape)
    plain = benchmarking.standardize_function(
        gmm_v2.gmm_v2,
        self.lhs_gathered,
        self.rhs_q,
        self.group_sizes,
        kwargs=dict(
            rhs_scale=self.rhs_scale, zero_initialize=False, fuse_act="silu"
        ),
    )
    ragged = benchmarking.standardize_function(
        ragged_permute_gmm,
        self.lhs,
        self.rhs_q,
        self.rhs_scale,
        self.group_sizes,
        self.gather_indices,
    )
    fused = benchmarking.standardize_function(
        gmm.routing_gmm,
        self.lhs_compact,
        self.rhs_q_wrapped,
        self.group_sizes,
        kwargs=dict(gather=self.gather, fuse_act="silu"),
    )
    self._benchmark_and_profile(plain, ragged, fused)

  @parameterized.named_parameters(*SHAPES)
  def test_gmm_unpermute_benchmark(self, **shape):
    self._setup_inputs(**shape)
    plain = benchmarking.standardize_function(
        gmm_v2.gmm_v2,
        self.lhs_gathered,
        self.rhs,
        self.group_sizes,
        kwargs=dict(zero_initialize=False),
    )
    ragged = benchmarking.standardize_function(
        ragged_gmm_unpermute,
        self.lhs_gathered,
        self.rhs,
        self.group_sizes,
        self.topk_argsort_indices,
    )
    fused = benchmarking.standardize_function(
        gmm.routing_gmm,
        self.lhs_gathered_compact,
        self.rhs_wrapped,
        self.group_sizes,
        kwargs=dict(scatter=self.scatter),
    )
    self._benchmark_and_profile(plain, ragged, fused)


class RemoteRoutingGmmTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices.")

  def _setup_inputs(
      self,
      mesh_shape,
      num_tokens,
      topk,
      num_experts,
      size_k,
      size_n,
      *,
      balanced=False,
      skip_last_shard=False,
  ):
    self.mesh = make_mesh(mesh_shape)
    ep_size = self.mesh.shape[EP_AXIS]
    num_local_experts = num_experts // ep_size

    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    lhs = jax.random.normal(k0, (num_tokens, size_k), jnp.bfloat16) / 10
    self.rhs = (
        jax.random.normal(k1, (num_experts, size_k, size_n), jnp.bfloat16) / 10
    )

    if balanced:
      topk_indices = balanced_topk_indices(
          k2, num_tokens, topk, num_experts, ep_size
      )
    else:
      num_skipped = num_local_experts if skip_last_shard else 0
      topk_indices = random_topk_indices(
          k2, num_tokens, topk, num_experts, num_skipped
      )
    topk_indices_flat = topk_indices.flatten()
    self.topk_argsort_indices = jnp.argsort(topk_indices_flat)
    self.gather_indices = self.topk_argsort_indices // topk
    self.group_sizes = jnp.bincount(topk_indices_flat, length=num_experts)

    # The local baseline writes each row to its shard's dense
    # [num_tokens, topk / ep_size] block instead of to its final slot.
    token = self.topk_argsort_indices // topk
    k = self.topk_argsort_indices % topk
    topk_per_shard = topk // ep_size
    self.local_scatter_indices = token * topk_per_shard + k % topk_per_shard

    # routing_gmm reads lhs in the compact layout.
    self.lhs_gathered_compact = input_pack.input_pack(lhs[self.gather_indices])

  def _benchmark_and_profile(self, local, local_kernel, remote):
    """Returns median latencies in us and trace links of local, remote."""
    latency_us = {}
    for name, (fn, args) in [
        ("local", local),
        ("local_kernel", local_kernel),
        ("remote", remote),
    ]:
      latency_us[name] = 1e3 * (
          benchmarking.benchmark(
              fn, args, method="hermetic_xprof"
          ).median_evaluation_time_ms
      )

    trace = ""
    return latency_us, trace

  @parameterized.product(
      mesh_shape=((4,), (2, 2)),
      num_tokens_topk=((256, 2), (1024, 4)),
      skip_last_shard=(False, True),
  )
  def test_remote_gmm_unpermute_numerics(
      self, mesh_shape, num_tokens_topk, skip_last_shard
  ):
    num_tokens, topk = num_tokens_topk
    self._setup_inputs(
        mesh_shape,
        num_tokens,
        topk,
        num_experts=16,
        size_k=1024,
        size_n=512,
        skip_last_shard=skip_last_shard,
    )
    # One device holding all experts.
    expected = gmm.routing_gmm(
        self.lhs_gathered_compact,
        arg.make_weight_wrapper(self.rhs),
        self.group_sizes,
        scatter=arg.make_routing_wrapper(self.topk_argsort_indices),
    ).data
    actual = remote_gmm_unpermute(
        self.lhs_gathered_compact,
        self.rhs,
        self.group_sizes,
        self.topk_argsort_indices,
        mesh=self.mesh,
    )
    np.testing.assert_array_equal(actual, expected)

  def test_remote_gmm_unpermute_benchmark(self):
    """Prints remote vs local + all_to_all for the Qwen 3.5 combine at EP=4."""
    topk, num_experts, size_k, size_n = 8, 256, 1024, 4096
    rows = []
    for num_tokens in (1280, 2560, 5120, 10240):
      self._setup_inputs(
          (4,), num_tokens, topk, num_experts, size_k, size_n, balanced=True
      )
      # Shard rhs up front so the timed functions do not reshard it.
      rhs = jax.device_put(self.rhs, NamedSharding(self.mesh, P(EP_AXIS)))
      args = (self.lhs_gathered_compact, rhs, self.group_sizes)
      local = benchmarking.standardize_function(
          local_gmm_unpermute,
          *args,
          self.local_scatter_indices,
          kwargs=dict(mesh=self.mesh, topk=topk),
      )
      local_kernel = benchmarking.standardize_function(
          local_gmm_unpermute,
          *args,
          self.local_scatter_indices,
          kwargs=dict(mesh=self.mesh, topk=topk, all_to_all=False),
      )
      remote = benchmarking.standardize_function(
          remote_gmm_unpermute,
          *args,
          self.topk_argsort_indices,
          kwargs=dict(mesh=self.mesh),
      )
      (local_fn, local_args), (remote_fn, remote_args) = local, remote
      np.testing.assert_array_equal(
          remote_fn(remote_args), local_fn(local_args)
      )

      latency_us, trace = self._benchmark_and_profile(
          local, local_kernel, remote
      )
      local_us, kernel_us = latency_us["local"], latency_us["local_kernel"]
      remote_us = latency_us["remote"]
      rows.append(
          f"| {num_tokens} | {local_us:.0f} | {kernel_us:.0f} | "
          f"{local_us - kernel_us:.0f} | {remote_us:.0f} | "
          f"{remote_us / local_us:.2f}x | {trace} |"
      )

    print(
        f"=== combine: topk={topk} ep=4 K={size_k} N={size_n} E={num_experts};"
        " a2a = local total - local kernel ==="
    )
    print(
        "| tokens | local total (us) | local kernel (us) | local a2a (us) | "
        "remote (us) | remote / local | trace |"
    )
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows), flush=True)


if __name__ == "__main__":
  absltest.main()
