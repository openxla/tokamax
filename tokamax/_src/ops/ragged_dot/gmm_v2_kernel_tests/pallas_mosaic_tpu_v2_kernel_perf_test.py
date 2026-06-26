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
import tokamax
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_backend
from tokamax._src.ops.ragged_dot.gmm_v2_kernel_tests import pallas_mosaic_tpu_v2_kernel_test as kernel_test

jax.config.parse_flags_with_absl()


class GmmPerfTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  def test_gmm_perf_regression(self):
    m, k, n, num_groups = 262144, 7168, 1024, 256
    block_size = 256
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, k, n), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(m, num_groups)

    rhs_q, rhs_scale = kernel_test.quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    gmm_op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot()
    benchmark_config = dict(
        lhs=lhs,
        rhs=rhs_q,
        group_sizes=group_sizes,
        rhs_scale=rhs_scale,
        maybe_quantize_lhs=True,
        preferred_element_type=jnp.bfloat16,
    )
    fn, args = tokamax.standardize_function(
        gmm_op,
        kwargs=benchmark_config,
        mode="forward",  # pytype: disable=wrong-arg-types
    )
    fn = jax.jit(fn)
    res = tokamax.benchmark(fn, args, method="hermetic_xprof")
    logging.info("Benchmark time (ms): %s", res.median_evaluation_time_ms)

    tpu_gen = pltpu.get_tpu_info().generation
    if tpu_gen == 7:
      threshold = 4.1162  # 110% of measured median latency in ms
      self.assertLessEqual(res.median_evaluation_time_ms, threshold)

    elif tpu_gen == 6:
      threshold = 13.2198  # 110% of measured median latency in ms
      self.assertLessEqual(res.median_evaluation_time_ms, threshold)
    else:
      self.skipTest(f"Unsupported TPU generation: {tpu_gen}")

  def test_tgmm_perf_regression(self):
    m, k, n, num_groups = 262144, 7168, 1024, 256
    k0, k2 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    grad = jax.random.normal(k2, (m, n), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(m, num_groups)

    tgmm_backend.validate_tgmm_inputs(group_sizes, num_groups)

    drhs_op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot(
        num_actual_groups=num_groups
    )
    benchmark_config = dict(
        lhs=lhs,
        rhs=grad,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=pallas_mosaic_tpu_v2.DRHS_RAGGED_DOT_DIM_NUMS,
        preferred_element_type=jnp.bfloat16,
    )
    fn, args = tokamax.standardize_function(
        drhs_op,
        kwargs=benchmark_config,
        mode="forward",  # pytype: disable=wrong-arg-types
    )
    fn = jax.jit(fn)
    res = tokamax.benchmark(fn, args, method="hermetic_xprof")
    logging.info("Benchmark time (ms): %s", res.median_evaluation_time_ms)

    tpu_gen = pltpu.get_tpu_info().generation
    if tpu_gen == 7:
      threshold = 7.1995  # 110% of measured median latency in ms
      self.assertLessEqual(res.median_evaluation_time_ms, threshold)

    elif tpu_gen == 6:
      threshold = 8.2753  # 110% of measured median latency in ms
      self.assertLessEqual(res.median_evaluation_time_ms, threshold)
    else:
      self.skipTest(f"Unsupported TPU generation: {tpu_gen}")


if __name__ == "__main__":
  absltest.main()
