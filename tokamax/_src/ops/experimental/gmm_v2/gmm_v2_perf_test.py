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
import time

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
    tpu_gen = pltpu.get_tpu_info().generation
    if tpu_gen < 7:
      self.skipTest(f"Unsupported TPU generation: {tpu_gen}")
    m, k, n, num_groups = 262144, 7168, 1024, 256
    block_size = 256
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, k, n), jnp.bfloat16)
    group_sizes = gmm_util.get_group_sizes(m, num_groups)

    rhs_q, rhs_scale = gmm_util.quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=block_size  # pyrefly: ignore[bad-argument-type]
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

    threshold = 3.40  # 110% of measured median latency in ms
    self.assertLessEqual(res.median_evaluation_time_ms, threshold)

  def test_tgmm_perf_regression_maxtext(self):
    tpu_gen = pltpu.get_tpu_info().generation
    if tpu_gen < 7:
      self.skipTest(f"Unsupported TPU generation: {tpu_gen}")
    m, k, n, num_groups = 262144, 7168, 1024, 256
    k0, k2 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(
        k0, (m, k), dtype=jnp.bfloat16
    ).astype(jnp.float8_e4m3fn)
    grad = jax.random.normal(k2, (m, n), dtype=jnp.float32)
    group_sizes = gmm_util.get_group_sizes(m, num_groups)

    grad_q, grad_scale = gmm_util.quantize_tensor(
        grad, jnp.float8_e5m2, axis=0, block_size=m  # pyrefly: ignore[bad-argument-type]
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

    threshold = 5.27  # 110% of measured median latency in ms
    self.assertLessEqual(res.median_evaluation_time_ms, threshold)

  # The ULLM MoE layer runs two gmm calls per token batch: the fused gate + up
  # projection, followed by the down projection. Both are covered here at a
  # prefill and at a decode batch size.
  @parameterized.named_parameters(
      dict(
          testcase_name="ullm_fp8_prefill_gate_up",
          m=81920,  # 8192 tokens * topk 10.
          k=4096,  # hidden_size.
          n=2 * 1024,  # gate and up, each moe_intermediate_size wide.
          fuse_act="silu",
          threshold=0.2453,  # 110% of measured median latency in ms
          weight_dtype=jnp.float8_e4m3fn,
          block_size=4096,
      ),
      dict(
          testcase_name="ullm_fp8_prefill_down",
          m=81920,
          k=1024,  # moe_intermediate_size.
          n=4096,  # hidden_size.
          fuse_act=None,
          threshold=0.1617,
          weight_dtype=jnp.float8_e4m3fn,
          block_size=1024,
      ),
      dict(
          testcase_name="ullm_fp8_decode_gate_up",
          m=1280,  # 128 tokens * topk 10.
          k=4096,
          n=2 * 1024,
          fuse_act="silu",
          threshold=0.1936,
          weight_dtype=jnp.float8_e4m3fn,
          block_size=4096,
      ),
      dict(
          testcase_name="ullm_fp8_decode_down",
          m=1280,
          k=1024,
          n=4096,
          fuse_act=None,
          threshold=0.1221,
          weight_dtype=jnp.float8_e4m3fn,
          block_size=1024,
      ),
      dict(
          testcase_name="ullm_fp4_prefill_gate_up",
          m=81920,
          k=4096,
          n=2 * 1024,
          fuse_act="silu",
          threshold=0.2805,  # 110% of measured median latency in ms
          weight_dtype=jnp.float4_e2m1fn,
          block_size=64,
      ),
      dict(
          testcase_name="ullm_fp4_prefill_down",
          m=81920,
          k=1024,
          n=4096,
          fuse_act=None,
          threshold=0.1672,
          weight_dtype=jnp.float4_e2m1fn,
          block_size=64,
      ),
      dict(
          testcase_name="ullm_fp4_decode_gate_up",
          m=1280,
          k=4096,
          n=2 * 1024,
          fuse_act="silu",
          threshold=0.2442,
          weight_dtype=jnp.float4_e2m1fn,
          block_size=64,
      ),
      dict(
          testcase_name="ullm_fp4_decode_down",
          m=1280,
          k=1024,
          n=4096,
          fuse_act=None,
          threshold=0.1397,
          weight_dtype=jnp.float4_e2m1fn,
          block_size=64,
      ),
  )
  def test_gmm_perf_regression_ullm(
      self, m, k, n, fuse_act, threshold, weight_dtype, block_size
  ):
    tpu_gen = pltpu.get_tpu_info().generation
    if tpu_gen < 7:
      self.skipTest(f"Unsupported TPU generation: {tpu_gen}")
    num_groups = 512  # Global number of experts.
    num_local_groups = 64  # Experts per EP shard (512 / 8).
    group_offset = 256  # First expert of EP shard 4 (a middle shard).
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_local_groups, k, n), jnp.bfloat16)
    # Evenly routed. At decode sizes the experts outnumber the rows, so this
    # leaves the tail of `lhs` unrouted, much like padding tokens in a
    # partially filled decode batch.
    group_sizes = jnp.full((num_groups,), m // num_groups, jnp.int32)

    rhs_q, rhs_scale = gmm_util.quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size  # pyrefly: ignore[bad-argument-type]
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
    self.assertLessEqual(res.median_evaluation_time_ms, threshold)


class GmmCompilePerfTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  def test_tgmm_trace_and_lowering_time_independent_of_num_groups(self):
    # Compares against a baseline with fewer groups, rather than an absolute
    # threshold, so the test doesn't depend on machine speed.
    tpu_gen = pltpu.get_tpu_info().generation
    if tpu_gen < 7:
      self.skipTest(f"Unsupported TPU generation: {tpu_gen}")
    m, k, n = 262144, 7168, 1024
    lhs = jax.ShapeDtypeStruct((m, k), jnp.float8_e4m3fn)
    grad_q = jax.ShapeDtypeStruct((m, n), jnp.float8_e5m2)
    grad_scale = jax.ShapeDtypeStruct((1, 1, n), jnp.float32)

    def measure(num_groups):
      group_sizes = jax.ShapeDtypeStruct((num_groups,), jnp.int32)
      trace_times, lowering_times = [], []
      for _ in range(3):
        jax.clear_caches()
        start = time.perf_counter()
        traced = tgmm_backend.tgmm_v2.trace(
            lhs,
            grad_q,
            group_sizes,
            num_groups,
            rhs_scale=grad_scale,
            preferred_element_type=jnp.bfloat16,
        )
        trace_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        traced.lower()
        lowering_times.append(time.perf_counter() - start)
      return min(trace_times), min(lowering_times)

    num_groups = 16
    trace_time, lowering_time = measure(num_groups)
    trace_time_3x, lowering_time_3x = measure(3 * num_groups)
    logging.info(
        "num_groups=%d: trace %.3fs, lowering %.3fs; num_groups=%d: trace"
        " %.3fs, lowering %.3fs",
        num_groups,
        trace_time,
        lowering_time,
        3 * num_groups,
        trace_time_3x,
        lowering_time_3x,
    )
    # If the kernel is unrolled over the groups, the trace/lower time grows
    # linearly with `num_groups`.
    max_ratio = 1.5
    self.assertLess(trace_time_3x / trace_time, max_ratio)
    self.assertLess(lowering_time_3x / lowering_time, max_ratio)


if __name__ == "__main__":
  absltest.main()
