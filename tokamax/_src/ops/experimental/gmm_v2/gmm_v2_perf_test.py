# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
from tokamax._src import benchmarking
from tokamax._src.ops.experimental.gmm_v2 import tgmm_v2 as tgmm_backend
from tokamax._src.ops.experimental.gmm_v2 import util as gmm_util
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2

jax.config.parse_flags_with_absl()


class GmmPerfTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  def test_gmm_perf_regression_maxtext(self):
    m, k, n, num_groups = 262144, 7168, 1024, 256
    block_size = 256
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, k, n), jnp.bfloat16)
    group_sizes = gmm_util.get_group_sizes(m, num_groups)

    rhs_q, rhs_scale = gmm_util.quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    lhs_scale = jnp.full((1, 1), 224.0 / 448.0, dtype=jnp.float32)

    gmm_op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot()
    benchmark_config = dict(
        lhs=lhs,
        rhs=rhs_q,
        group_sizes=group_sizes,
        rhs_scale=rhs_scale,
        maybe_quantize_lhs=True,
        lhs_scale=lhs_scale,
        preferred_element_type=jnp.bfloat16,
    )
    fn, args = benchmarking.standardize_function(
        gmm_op,
        kwargs=benchmark_config,
        mode="forward",
    )
    fn = jax.jit(fn)
    res = benchmarking.benchmark(fn, args, method="hermetic_xprof")
    logging.info("Benchmark time (ms): %s", res.median_evaluation_time_ms)

    tpu_gen = pltpu.get_tpu_info().generation
    if tpu_gen == 7:
      threshold = 3.40  # 110% of measured median latency in ms
      self.assertLessEqual(res.median_evaluation_time_ms, threshold)
    else:
      self.skipTest(f"Unsupported TPU generation: {tpu_gen}")

  def test_tgmm_perf_regression_maxtext(self):
    m, k, n, num_groups = 262144, 7168, 1024, 256
    k0, k2 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(
        k0, (m, k), dtype=jnp.bfloat16
    ).astype(jnp.float8_e4m3fn)
    grad = jax.random.normal(k2, (m, n), dtype=jnp.float32)
    group_sizes = gmm_util.get_group_sizes(m, num_groups)

    grad_q, grad_scale = gmm_util.quantize_tensor(
        grad, jnp.float8_e5m2, axis=0, block_size=m
    )
    grad_scale = jnp.expand_dims(grad_scale, axis=1)

    tgmm_backend.validate_tgmm_inputs(group_sizes, num_groups)

    drhs_op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot(
        num_actual_groups=num_groups
    )
    benchmark_config = dict(
        lhs=lhs,
        rhs=grad_q,
        group_sizes=group_sizes,
        rhs_scale=grad_scale,
        ragged_dot_dimension_numbers=pallas_mosaic_tpu_v2.DRHS_RAGGED_DOT_DIM_NUMS,
        preferred_element_type=jnp.bfloat16,
    )
    fn, args = benchmarking.standardize_function(
        drhs_op,
        kwargs=benchmark_config,
        mode="forward",
    )
    fn = jax.jit(fn)
    res = benchmarking.benchmark(fn, args, method="hermetic_xprof")
    logging.info("Benchmark time (ms): %s", res.median_evaluation_time_ms)

    tpu_gen = pltpu.get_tpu_info().generation
    if tpu_gen == 7:
      threshold = 5.27  # 110% of measured median latency in ms
      self.assertLessEqual(res.median_evaluation_time_ms, threshold)
    else:
      self.skipTest(f"Unsupported TPU generation: {tpu_gen}")

  # The ULLM MoE layer runs two gmm calls per token batch: the fused gate + up
  # projection, followed by the down projection. Both are covered here at a
  # prefill and at a decode batch size.
  @parameterized.named_parameters(
      dict(
          testcase_name="ullm_prefill_gate_up",
          m=81920,  # 8192 tokens * topk 10.
          k=4096,  # hidden_size.
          n=2 * 1024,  # gate and up, each moe_intermediate_size wide.
          fuse_act="silu",
          threshold=0.2453,  # 110% of measured median latency in ms
      ),
      dict(
          testcase_name="ullm_prefill_down",
          m=81920,
          k=1024,  # moe_intermediate_size.
          n=4096,  # hidden_size.
          fuse_act=None,
          threshold=0.1617,
      ),
      dict(
          testcase_name="ullm_decode_gate_up",
          m=1280,  # 128 tokens * topk 10.
          k=4096,
          n=2 * 1024,
          fuse_act="silu",
          threshold=0.1936,
      ),
      dict(
          testcase_name="ullm_decode_down",
          m=1280,
          k=1024,
          n=4096,
          fuse_act=None,
          threshold=0.1221,
      ),
  )
  def test_gmm_perf_regression_ullm(self, m, k, n, fuse_act, threshold):
    num_groups = 512  # Global number of experts.
    num_local_groups = 64  # Experts per EP shard (512 / 8).
    group_offset = 256  # First expert of EP shard 4 (a middle shard).
    # The weights reach the kernel with a single per-output-channel scale
    # covering the whole k axis, i.e. `rhs_scale` is [64, 1, 1, n].
    block_size = k
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_local_groups, k, n), jnp.bfloat16)
    # Evenly routed. At decode sizes the experts outnumber the rows, so this
    # leaves the tail of `lhs` unrouted, much like padding tokens in a
    # partially filled decode batch.
    group_sizes = jnp.full((num_groups,), m // num_groups, jnp.int32)

    rhs_q, rhs_scale = gmm_util.quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    gmm_op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot()
    benchmark_config = dict(
        lhs=lhs,
        rhs=rhs_q,
        group_sizes=group_sizes,
        group_offset=jnp.array([group_offset], jnp.int32),
        rhs_scale=rhs_scale,
        maybe_quantize_lhs=True,
        # The MoE layer writes only the rows owned by this shard.
        zero_initialize=False,
        fuse_gateup_activation=fuse_act,
        preferred_element_type=jnp.bfloat16,
    )
    fn, args = benchmarking.standardize_function(
        gmm_op,
        kwargs=benchmark_config,
        mode="forward",
    )
    fn = jax.jit(fn)
    res = benchmarking.benchmark(fn, args, method="hermetic_xprof")
    logging.info("Benchmark time (ms): %s", res.median_evaluation_time_ms)

    tpu_gen = pltpu.get_tpu_info().generation
    if tpu_gen == 7:
      self.assertLessEqual(res.median_evaluation_time_ms, threshold)
    else:
      self.skipTest(f"Unsupported TPU generation: {tpu_gen}")


if __name__ == "__main__":
  absltest.main()
