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

# Add test cases with less rows than experts, for utilization of our code.
DECODE_SHAPES = [
    dict(
        testcase_name="decode_16",
        num_tokens=16,
        size_k=512,
        size_n=1024,
        num_experts=512,
        ep_size=1,
        topk=10,
    ),
    dict(
        testcase_name="decode_64_ep8",
        num_tokens=64,
        size_k=512,
        size_n=1024,
        num_experts=512,
        ep_size=8,
        topk=10,
    ),
]

# Decode shapes with an explicit skip_metadata_size: 0 (the plain loop), small
# and large skips, and one past the number of groups.
SKIP_METADATA_SHAPES = [
    dict(
        shape,
        testcase_name=f"{shape['testcase_name']}_skip{skip}",
        skip_metadata_size=skip,
    )
    for shape in DECODE_SHAPES
    for skip in (0, 1, 2, 63, 1024)
]

LIVE_GROUP_WALK_SHAPES = (
    dict(
        testcase_name="single_token_offset0",
        num_tokens=1,
        num_experts=512,
        num_local_experts=512,
        group_offset=0,
        topk=10,
        size_k=512,
        size_n=1024,
    ),
    dict(
        testcase_name="multi_token_offset0",
        num_tokens=16,
        num_experts=512,
        num_local_experts=512,
        group_offset=0,
        topk=10,
        size_k=512,
        size_n=1024,
    ),
    dict(
        testcase_name="single_token_offset128",
        num_tokens=1,
        num_experts=512,
        num_local_experts=64,
        group_offset=128,
        topk=10,
        size_k=512,
        size_n=1024,
    ),
    dict(
        testcase_name="multi_token_offset128",
        num_tokens=16,
        num_experts=512,
        num_local_experts=64,
        group_offset=128,
        topk=10,
        size_k=512,
        size_n=1024,
    ),
)


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
      (arg.make_weight_wrapper(rhs),),
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
        (arg.make_weight_wrapper(rhs),),
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

    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    self.lhs = jax.random.normal(k0, (num_tokens, size_k), jnp.bfloat16) / 10
    self.rhs = (
        jax.random.normal(k1, (num_local_experts, size_k, size_n), jnp.bfloat16)
        / 10
    )
    self.rhs_q, rhs_scale = gmm_util.quantize_tensor(
        self.rhs, jnp.float8_e4m3fn, axis=1, block_size=size_k
    )
    self.rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    # Down projection from the activated gate/up output back to size_k.
    self.rhs2 = (
        jax.random.normal(
            k3, (num_local_experts, size_n // 2, size_k), jnp.bfloat16
        )
        / 10
    )
    self.rhs2_q, rhs2_scale = gmm_util.quantize_tensor(
        self.rhs2, jnp.float8_e4m3fn, axis=1, block_size=size_n // 2
    )
    self.rhs2_scale = jnp.expand_dims(rhs2_scale, axis=2)

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
    self.rhs_wrapped = arg.make_weight_wrapper(self.rhs, fused_gate_up=True)
    self.rhs_q_wrapped = arg.make_weight_wrapper(
        self.rhs_q, self.rhs_scale, fused_gate_up=True
    )
    self.rhs2_wrapped = arg.make_weight_wrapper(self.rhs2)
    self.rhs2_q_wrapped = arg.make_weight_wrapper(self.rhs2_q, self.rhs2_scale)
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
      [*SHAPES, *DECODE_SHAPES],
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
        (self.rhs_q_wrapped,),
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
      (
          dict(quant_w1=True, quant_w2=True, fuse_up_and_down=None),
          dict(quant_w1=True, quant_w2=True, fuse_up_and_down=True),
          dict(quant_w1=True, quant_w2=True, fuse_up_and_down=False),
          dict(quant_w1=False, quant_w2=False, fuse_up_and_down=None),
          dict(quant_w1=False, quant_w2=False, fuse_up_and_down=True),
          dict(quant_w1=False, quant_w2=False, fuse_up_and_down=False),
          dict(quant_w1=True, quant_w2=False, fuse_up_and_down=None),
      ),
  )
  def test_permute_gmm_gmm_unpermute_numerics(
      self, quant_w1, quant_w2, fuse_up_and_down, **shape
  ):
    self._setup_inputs(**shape)
    rhs1, scale1, w1 = (
        (self.rhs_q, self.rhs_scale, self.rhs_q_wrapped)
        if quant_w1
        else (self.rhs, None, self.rhs_wrapped)
    )
    rhs2, scale2, w2 = (
        (self.rhs2_q, self.rhs2_scale, self.rhs2_q_wrapped)
        if quant_w2
        else (self.rhs2, None, self.rhs2_wrapped)
    )
    hidden = gmm_v2.gmm_v2(
        self.lhs_gathered,
        rhs1,
        self.group_sizes,
        rhs_scale=scale1,
        zero_initialize=False,
        fuse_act="silu",
    )
    expected = gmm_v2.gmm_v2(
        hidden,
        rhs2,
        self.group_sizes,
        rhs_scale=scale2,
        zero_initialize=False,
    )
    actual = gmm.routing_gmm(
        self.lhs_compact,
        (w1, w2),
        self.group_sizes,
        gather=self.gather,
        scatter=self.scatter,
        fuse_act="silu",
        custom_plan=gmm_plan.RoutingGmmPlan(fuse_up_and_down=fuse_up_and_down),
    )
    actual = actual.data.reshape(actual.spec.large_shape)
    np.testing.assert_allclose(
        actual[self.topk_argsort_indices[: self.num_valid_rows]],
        expected[: self.num_valid_rows],
        rtol=5e-2,
        atol=5e-2,
    )

  @parameterized.product(
      [*SHAPES, *DECODE_SHAPES],
      tile_n=[None, 512],
  )
  def test_gmm_unpermute_numerics(self, tile_n, **shape):
    self._setup_inputs(**shape)
    expected = gmm_v2.gmm_v2(
        self.lhs_gathered, self.rhs, self.group_sizes, zero_initialize=False
    )
    actual = gmm.routing_gmm(
        self.lhs_gathered_compact,
        (self.rhs_wrapped,),
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

  @parameterized.named_parameters(*SKIP_METADATA_SHAPES)
  def test_skip_metadata_size(self, skip_metadata_size, **shape):
    # Skipping only changes how the metadata walks the groups, not the tiles it
    # writes, so any skip must give the same output as the default plan.
    self._setup_inputs(**shape)
    custom_plan = gmm_plan.RoutingGmmPlan(skip_metadata_size=skip_metadata_size)
    # Only rows of this shard's groups are written.
    gathered_rows = np.arange(self.num_valid_rows)
    scattered_rows = self.topk_argsort_indices[: self.num_valid_rows]
    for rows, kwargs in (
        (
            gathered_rows,
            dict(
                x=self.lhs_compact,
                w=(self.rhs_q_wrapped,),
                gather=self.gather,
                fuse_act="silu",
            ),
        ),
        (
            scattered_rows,
            dict(
                x=self.lhs_gathered_compact,
                w=(self.rhs_wrapped,),
                scatter=self.scatter,
            ),
        ),
    ):
      expected = gmm.routing_gmm(group_sizes=self.group_sizes, **kwargs)
      actual = gmm.routing_gmm(
          group_sizes=self.group_sizes, custom_plan=custom_plan, **kwargs
      )
      expected = expected.data.reshape(expected.spec.large_shape)
      actual = actual.data.reshape(actual.spec.large_shape)
      np.testing.assert_array_equal(actual[rows], expected[rows])

  def test_negative_skip_metadata_size(self):
    self._setup_inputs(**DECODE_SHAPES[0])
    with self.assertRaisesRegex(ValueError, "skip_metadata_size"):
      gmm.routing_gmm(
          self.lhs_compact,
          (self.rhs_q_wrapped,),
          self.group_sizes,
          gather=self.gather,
          fuse_act="silu",
          custom_plan=gmm_plan.RoutingGmmPlan(skip_metadata_size=-1),
      )

  @parameterized.product(
      num_tokens=[1, 4, 16],
      shard=[(512, 0), (64, 0), (64, 128)],
      tile_n=[None, 512],
  )
  def test_lowtok_matches_sorted(self, num_tokens, shard, tile_n):
    num_local_experts, group_offset_val = shard
    num_experts = 512
    topk = 10
    size_k = 1024
    size_n = 512

    key0, key1, key2, key3 = jax.random.split(jax.random.key(0), 4)
    lhs = jax.random.normal(key0, (num_tokens, size_k), jnp.bfloat16) / 10
    rhs1 = (
        jax.random.normal(
            key1, (num_local_experts, size_k, 2 * size_n), jnp.bfloat16
        )
        / 10
    )
    rhs1_q, rhs1_scale = gmm_util.quantize_tensor(
        rhs1, jnp.float8_e4m3fn, axis=1, block_size=size_k
    )
    rhs1_scale = jnp.expand_dims(rhs1_scale, axis=2)
    rhs2 = (
        jax.random.normal(
            key2, (num_local_experts, size_n, size_k), jnp.bfloat16
        )
        / 10
    )

    gating_output = jax.random.uniform(key3, (num_tokens, num_experts))
    _, topk_indices = jax.lax.top_k(gating_output, topk)
    topk_indices_flat = topk_indices.flatten()
    topk_argsort_indices = jnp.argsort(topk_indices_flat)
    group_sizes = jnp.bincount(topk_indices_flat, length=num_experts)
    group_offset = jnp.array([group_offset_val], dtype=jnp.int32)

    lhs_compact = input_pack.input_pack(lhs)
    w1_wrapped = arg.make_weight_wrapper(
        rhs1_q, rhs1_scale, fused_gate_up=True
    )
    w2_wrapped = arg.make_weight_wrapper(rhs2)
    custom_plan = gmm_plan.RoutingGmmPlan(tile_n=tile_n)

    sorted_mid = gmm.routing_gmm(
        lhs_compact,
        (w1_wrapped,),
        group_sizes,
        group_offset=group_offset,
        gather=arg.make_routing_wrapper(topk_argsort_indices // topk),
        fuse_act="silu",
        custom_plan=custom_plan,
    )
    sorted_out = gmm.routing_gmm(
        sorted_mid,
        (w2_wrapped,),
        group_sizes,
        group_offset=group_offset,
        scatter=arg.make_routing_wrapper(topk_argsort_indices),
        custom_plan=custom_plan,
    )

    slots_wrapped = arg.make_routing_wrapper(topk_indices_flat)
    lowtok_mid = gmm.routing_gmm(
        lhs_compact,
        (w1_wrapped,),
        None,
        group_offset=group_offset,
        gather=slots_wrapped,
        fuse_act="silu",
        custom_plan=custom_plan,
        topk=topk,
        num_global_groups=num_experts,
    )
    lowtok_out = gmm.routing_gmm(
        lowtok_mid,
        (w2_wrapped,),
        None,
        group_offset=group_offset,
        scatter=slots_wrapped,
        custom_plan=custom_plan,
        topk=topk,
        num_global_groups=num_experts,
    )

    written = (topk_indices_flat >= group_offset_val) & (
        topk_indices_flat < group_offset_val + num_local_experts
    )
    sorted_rows = sorted_out.data.reshape(sorted_out.spec.large_shape)
    lowtok_rows = lowtok_out.data.reshape(lowtok_out.spec.large_shape)
    np.testing.assert_array_equal(lowtok_rows[written], sorted_rows[written])

    if tile_n is None:
      w1_bf16 = arg.make_weight_wrapper(rhs1, fused_gate_up=True)
      fused_plan = gmm_plan.RoutingGmmPlan(fuse_up_and_down=True)
      sorted_fused = gmm.routing_gmm(
          lhs_compact,
          (w1_bf16, w2_wrapped),
          group_sizes,
          group_offset=group_offset,
          gather=arg.make_routing_wrapper(topk_argsort_indices // topk),
          scatter=arg.make_routing_wrapper(topk_argsort_indices),
          fuse_act="silu",
          custom_plan=fused_plan,
      )
      lowtok_fused = gmm.routing_gmm(
          lhs_compact,
          (w1_bf16, w2_wrapped),
          None,
          group_offset=group_offset,
          gather=slots_wrapped,
          scatter=slots_wrapped,
          fuse_act="silu",
          custom_plan=fused_plan,
          topk=topk,
          num_global_groups=num_experts,
      )
      sorted_fused_rows = sorted_fused.data.reshape(
          sorted_fused.spec.large_shape
      )
      lowtok_fused_rows = lowtok_fused.data.reshape(
          lowtok_fused.spec.large_shape
      )
      np.testing.assert_array_equal(
          lowtok_fused_rows[written], sorted_fused_rows[written]
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
        (self.rhs_q_wrapped,),
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
        (self.rhs_wrapped,),
        self.group_sizes,
        kwargs=dict(scatter=self.scatter),
    )
    self._benchmark_and_profile(plain, ragged, fused)

  def _live_group_walk_stages(
      self,
      *,
      num_tokens: int,
      num_experts: int,
      num_local_experts: int,
      group_offset: int,
      topk: int,
      size_k: int,
      size_n: int,
  ):
    """Returns `(x, w, kwargs, written rows)` of the gather and scatter stages."""
    key0, key1, key2, key3 = jax.random.split(jax.random.key(0), 4)
    lhs = jax.random.normal(key0, (num_tokens, size_k), jnp.bfloat16) / 10
    rhs1 = (
        jax.random.normal(
            key1, (num_local_experts, size_k, size_n), jnp.bfloat16
        )
        / 10
    )
    rhs2 = (
        jax.random.normal(
            key2, (num_local_experts, size_k, size_n), jnp.bfloat16
        )
        / 10
    )

    gating_output = jax.random.uniform(key3, (num_tokens, num_experts))
    # Ensure at least one token routes to this shard's experts.
    gating_output = gating_output.at[0, group_offset].set(2.0)
    _, topk_indices = jax.lax.top_k(gating_output, topk)
    topk_indices_flat = topk_indices.flatten()
    # The same sort as the MoE wrapper, whose sorted keys are `row_group`.
    row_group, topk_argsort_indices = jax.lax.sort_key_val(
        topk_indices_flat,
        jnp.arange(topk_indices_flat.size, dtype=jnp.int32),
        is_stable=True,
    )
    group_sizes = jnp.bincount(topk_indices_flat, length=num_experts)

    group_cum_sum = jnp.cumulative_sum(group_sizes, include_initial=True)
    start_row = int(group_cum_sum[group_offset])
    end_row = int(group_cum_sum[group_offset + num_local_experts])
    self.assertGreater(end_row - start_row, 0)

    common = dict(
        group_sizes=group_sizes,
        group_offset=jnp.array([group_offset], dtype=jnp.int32),
        row_group=row_group,
    )
    # 1. Gather (gmm1) with fuse_act="silu": writes the shard's sorted rows.
    gather_stage = (
        input_pack.input_pack(lhs),
        (arg.make_weight_wrapper(rhs1, fused_gate_up=True),),
        dict(
            common,
            gather=arg.make_routing_wrapper(topk_argsort_indices // topk),
            fuse_act="silu",
        ),
        np.arange(start_row, end_row),
    )
    # 2. Scatter (gmm2): writes the shard's rows back to their routed slots.
    scatter_stage = (
        input_pack.input_pack(lhs[topk_argsort_indices // topk]),
        (arg.make_weight_wrapper(rhs2),),
        dict(common, scatter=arg.make_routing_wrapper(topk_argsort_indices)),
        topk_argsort_indices[start_row:end_row],
    )
    return {"gather": gather_stage, "scatter": scatter_stage}

  @parameterized.product(
      LIVE_GROUP_WALK_SHAPES,
      tile_n=[None, 512],
  )
  def test_live_group_walk(
      self, tile_n: int | None, testcase_name: str = "", **shape
  ):
    del testcase_name
    stages = self._live_group_walk_stages(**shape)
    for x, w, kwargs, rows in stages.values():
      actual, expected = (
          gmm.routing_gmm(
              x,
              w,
              custom_plan=gmm_plan.RoutingGmmPlan(
                  tile_n=tile_n, live_group_walk=live_group_walk
              ),
              **kwargs,
          )
          for live_group_walk in (True, False)
      )
      actual = actual.data.reshape(actual.spec.large_shape)
      expected = expected.data.reshape(expected.spec.large_shape)
      np.testing.assert_array_equal(actual[rows], expected[rows])

  @parameterized.named_parameters(*LIVE_GROUP_WALK_SHAPES)
  def test_live_group_walk_benchmark(self, **shape):
    """Logs the time of each metadata walk at decode and checks the live wins.

    Run with --test_filter=test_live_group_walk_benchmark after changing the
    metadata to check for regressions at decode.
    """
    num_local_experts = shape["num_local_experts"]
    stages = self._live_group_walk_stages(**shape)
    for name, (x, w, kwargs, _) in stages.items():
      latency_us, trace = {}, {}
      for live_group_walk in (False, True):
        fn, args = benchmarking.standardize_function(
            gmm.routing_gmm,
            x,
            w,
            kwargs=dict(
                kwargs,
                custom_plan=gmm_plan.RoutingGmmPlan(
                    live_group_walk=live_group_walk
                ),
            ),
        )
        latency_us[live_group_walk] = 1e3 * (
            benchmarking.benchmark(
                fn, args, method="hermetic_xprof"
            ).median_evaluation_time_ms
        )
        trace[live_group_walk] = ""
      logging.info(
          "%s, %d tokens, %d local groups: all groups %.1f us%s, groups with"
          " rows %.1f us%s",
          name,
          shape["num_tokens"],
          num_local_experts,
          latency_us[False],
          trace[False],
          latency_us[True],
          trace[True],
      )
      # At one token the walk over 512 local groups is a large share of the
      # kernel (about 3 us of 20-25 us). With more tokens the matmul dominates,
      # and with 64 groups the gap is within noise, so those only log.
      if num_local_experts >= 512 and shape["num_tokens"] == 1:
        self.assertLess(latency_us[True], latency_us[False])

  @parameterized.product(
      LIVE_GROUP_WALK_SHAPES,
      tile_n=[None, 512],
      mode=["sorted", "live_group", "lowtok"],
  )
  def test_vmem_routing(
      self,
      tile_n: int | None,
      mode: str,
      num_tokens: int,
      num_experts: int,
      num_local_experts: int,
      group_offset: int,
      topk: int,
      size_k: int,
      size_n: int,
      testcase_name: str = "",
  ):
    del testcase_name
    key0, key1, key2, key3 = jax.random.split(jax.random.key(0), 4)
    lhs = jax.random.normal(key0, (num_tokens, size_k), jnp.bfloat16) / 10
    rhs1 = (
        jax.random.normal(
            key1, (num_local_experts, size_k, size_n), jnp.bfloat16
        )
        / 10
    )
    rhs2 = (
        jax.random.normal(
            key2, (num_local_experts, size_k, size_n), jnp.bfloat16
        )
        / 10
    )

    gating_output = jax.random.uniform(key3, (num_tokens, num_experts))
    gating_output = gating_output.at[0, group_offset].set(2.0)
    _, topk_indices = jax.lax.top_k(gating_output, topk)
    topk_indices_flat = topk_indices.flatten()
    topk_argsort_indices = jnp.argsort(topk_indices_flat)
    row_group = topk_indices_flat[topk_argsort_indices]
    group_sizes = jnp.bincount(topk_indices_flat, length=num_experts)

    group_offset_arr = jnp.array([group_offset], dtype=jnp.int32)
    group_start = group_offset
    group_end = group_start + num_local_experts
    group_cum_sum = jnp.cumulative_sum(group_sizes, include_initial=True)
    start_row = int(group_cum_sum[group_start])
    end_row = int(group_cum_sum[group_end])
    num_valid_rows = end_row - start_row
    self.assertGreater(num_valid_rows, 0)

    lhs_compact = input_pack.input_pack(lhs)
    rhs1_wrapped = arg.make_weight_wrapper(rhs1, fused_gate_up=True)
    lhs_gathered = lhs[topk_argsort_indices // topk]
    lhs_gathered_compact = input_pack.input_pack(lhs_gathered)
    rhs2_wrapped = arg.make_weight_wrapper(rhs2)

    if mode == "sorted":
      gather = arg.make_routing_wrapper(topk_argsort_indices // topk)
      scatter = arg.make_routing_wrapper(topk_argsort_indices)
      plan_vmem = gmm_plan.RoutingGmmPlan(
          tile_n=tile_n, live_group_walk=False, vmem_routing=True
      )
      plan_hbm = gmm_plan.RoutingGmmPlan(
          tile_n=tile_n, live_group_walk=False, vmem_routing=False
      )
      vmem_gmm1 = gmm.routing_gmm(
          lhs_compact,
          (rhs1_wrapped,),
          group_sizes,
          group_offset=group_offset_arr,
          gather=gather,
          fuse_act="silu",
          row_group=row_group,
          custom_plan=plan_vmem,
      )
      hbm_gmm1 = gmm.routing_gmm(
          lhs_compact,
          (rhs1_wrapped,),
          group_sizes,
          group_offset=group_offset_arr,
          gather=gather,
          fuse_act="silu",
          row_group=row_group,
          custom_plan=plan_hbm,
      )
      vmem_gmm2 = gmm.routing_gmm(
          lhs_gathered_compact,
          (rhs2_wrapped,),
          group_sizes,
          group_offset=group_offset_arr,
          scatter=scatter,
          row_group=row_group,
          custom_plan=plan_vmem,
      )
      hbm_gmm2 = gmm.routing_gmm(
          lhs_gathered_compact,
          (rhs2_wrapped,),
          group_sizes,
          group_offset=group_offset_arr,
          scatter=scatter,
          row_group=row_group,
          custom_plan=plan_hbm,
      )
      vmem_g1 = vmem_gmm1.data.reshape(vmem_gmm1.spec.large_shape)
      hbm_g1 = hbm_gmm1.data.reshape(hbm_gmm1.spec.large_shape)
      np.testing.assert_array_equal(
          vmem_g1[start_row:end_row],
          hbm_g1[start_row:end_row],
      )
      vmem_g2 = vmem_gmm2.data.reshape(vmem_gmm2.spec.large_shape)
      hbm_g2 = hbm_gmm2.data.reshape(hbm_gmm2.spec.large_shape)
      written_indices = topk_argsort_indices[start_row:end_row]
      np.testing.assert_array_equal(
          vmem_g2[written_indices],
          hbm_g2[written_indices],
      )
    elif mode == "live_group":
      gather = arg.make_routing_wrapper(topk_argsort_indices // topk)
      scatter = arg.make_routing_wrapper(topk_argsort_indices)
      plan_vmem = gmm_plan.RoutingGmmPlan(
          tile_n=tile_n, live_group_walk=True, vmem_routing=True
      )
      plan_hbm = gmm_plan.RoutingGmmPlan(
          tile_n=tile_n, live_group_walk=True, vmem_routing=False
      )
      vmem_gmm1 = gmm.routing_gmm(
          lhs_compact,
          (rhs1_wrapped,),
          group_sizes,
          group_offset=group_offset_arr,
          gather=gather,
          fuse_act="silu",
          row_group=row_group,
          custom_plan=plan_vmem,
      )
      hbm_gmm1 = gmm.routing_gmm(
          lhs_compact,
          (rhs1_wrapped,),
          group_sizes,
          group_offset=group_offset_arr,
          gather=gather,
          fuse_act="silu",
          row_group=row_group,
          custom_plan=plan_hbm,
      )
      vmem_gmm2 = gmm.routing_gmm(
          lhs_gathered_compact,
          (rhs2_wrapped,),
          group_sizes,
          group_offset=group_offset_arr,
          scatter=scatter,
          row_group=row_group,
          custom_plan=plan_vmem,
      )
      hbm_gmm2 = gmm.routing_gmm(
          lhs_gathered_compact,
          (rhs2_wrapped,),
          group_sizes,
          group_offset=group_offset_arr,
          scatter=scatter,
          row_group=row_group,
          custom_plan=plan_hbm,
      )
      vmem_g1 = vmem_gmm1.data.reshape(vmem_gmm1.spec.large_shape)
      hbm_g1 = hbm_gmm1.data.reshape(hbm_gmm1.spec.large_shape)
      np.testing.assert_array_equal(
          vmem_g1[start_row:end_row],
          hbm_g1[start_row:end_row],
      )
      vmem_g2 = vmem_gmm2.data.reshape(vmem_gmm2.spec.large_shape)
      hbm_g2 = hbm_gmm2.data.reshape(hbm_gmm2.spec.large_shape)
      written_indices = topk_argsort_indices[start_row:end_row]
      np.testing.assert_array_equal(
          vmem_g2[written_indices],
          hbm_g2[written_indices],
      )
    elif mode == "lowtok":
      slots_wrapped = arg.make_routing_wrapper(topk_indices_flat)
      plan_vmem = gmm_plan.RoutingGmmPlan(tile_n=tile_n, vmem_routing=True)
      plan_hbm = gmm_plan.RoutingGmmPlan(tile_n=tile_n, vmem_routing=False)
      vmem_gmm1 = gmm.routing_gmm(
          lhs_compact,
          (rhs1_wrapped,),
          None,
          group_offset=group_offset_arr,
          gather=slots_wrapped,
          fuse_act="silu",
          custom_plan=plan_vmem,
          topk=topk,
          num_global_groups=num_experts,
      )
      hbm_gmm1 = gmm.routing_gmm(
          lhs_compact,
          (rhs1_wrapped,),
          None,
          group_offset=group_offset_arr,
          gather=slots_wrapped,
          fuse_act="silu",
          custom_plan=plan_hbm,
          topk=topk,
          num_global_groups=num_experts,
      )
      vmem_gmm2 = gmm.routing_gmm(
          lhs_gathered_compact,
          (rhs2_wrapped,),
          None,
          group_offset=group_offset_arr,
          scatter=slots_wrapped,
          custom_plan=plan_vmem,
          topk=topk,
          num_global_groups=num_experts,
      )
      hbm_gmm2 = gmm.routing_gmm(
          lhs_gathered_compact,
          (rhs2_wrapped,),
          None,
          group_offset=group_offset_arr,
          scatter=slots_wrapped,
          custom_plan=plan_hbm,
          topk=topk,
          num_global_groups=num_experts,
      )
      vmem_g1 = vmem_gmm1.data.reshape(vmem_gmm1.spec.large_shape)
      hbm_g1 = hbm_gmm1.data.reshape(hbm_gmm1.spec.large_shape)
      np.testing.assert_array_equal(
          vmem_g1[:num_valid_rows],
          hbm_g1[:num_valid_rows],
      )
      vmem_g2 = vmem_gmm2.data.reshape(vmem_gmm2.spec.large_shape)
      hbm_g2 = hbm_gmm2.data.reshape(hbm_gmm2.spec.large_shape)
      written = (topk_indices_flat >= group_offset) & (
          topk_indices_flat < group_offset + num_local_experts
      )
      np.testing.assert_array_equal(
          vmem_g2[written],
          hbm_g2[written],
      )
    else:
      raise ValueError(f"Unknown mode: {mode}")


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
        (arg.make_weight_wrapper(self.rhs),),
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
