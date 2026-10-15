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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src import benchmarking
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.gmm import kernel as gmm
from tokamax._src.ops.experimental.routing_moe.io_relayout import input_pack
from tokamax._src.ops.gmm_v2 import gmm_v2
from tokamax._src.ops.gmm_v2 import util as gmm_util
from tokamax._src.ops.ragged_gather.pallas_mosaic_tpu_kernel import ragged_gather_pallas
from tokamax._src.ops.ragged_scatter.pallas_mosaic_tpu_kernel import ragged_scatter_pallas

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


def ragged_gmm_unpermute(lhs, rhs, group_sizes, topk_argsort_revert_indices):
  out = gmm_v2.gmm_v2(lhs, rhs, group_sizes, zero_initialize=False)
  return ragged_scatter_pallas(
      out,
      topk_argsort_revert_indices,
      jnp.array(0, jnp.int32),
      group_sizes.sum(),
  )


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
      self, num_tokens, size_k, size_n, num_experts, ep_size, topk
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

  @parameterized.named_parameters(*SHAPES)
  def test_permute_gmm_numerics(self, **shape):
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
    )
    actual = actual.data.reshape(actual.spec.large_shape)
    np.testing.assert_allclose(
        actual[: self.num_valid_rows],
        expected[: self.num_valid_rows],
        rtol=5e-2,
        atol=5e-2,
    )

  @parameterized.named_parameters(*SHAPES)
  def test_gmm_unpermute_numerics(self, **shape):
    self._setup_inputs(**shape)
    expected = gmm_v2.gmm_v2(
        self.lhs_gathered, self.rhs, self.group_sizes, zero_initialize=False
    )
    actual = gmm.routing_gmm(
        self.lhs_gathered_compact,
        self.rhs_wrapped,
        self.group_sizes,
        scatter=self.scatter,
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
    topk_argsort_revert_indices = jnp.argsort(self.topk_argsort_indices)
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
        topk_argsort_revert_indices,
    )
    fused = benchmarking.standardize_function(
        gmm.routing_gmm,
        self.lhs_gathered_compact,
        self.rhs_wrapped,
        self.group_sizes,
        kwargs=dict(scatter=self.scatter),
    )
    self._benchmark_and_profile(plain, ragged, fused)


if __name__ == "__main__":
  absltest.main()
