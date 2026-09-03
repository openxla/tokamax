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

"""Benchmarks for ragged dot."""

import functools
import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tensorboardX import writer
import tokamax

from tokamax._src.ops.experimental.gmm_v2 import tgmm_v2 as tgmm_backend
from tokamax._src.ops.experimental.gmm_v2 import util as gmm_util
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2

SummaryWriter = writer.SummaryWriter
_TENSORBOARD_OUTPUT_ENV_VAR = flags.DEFINE_string(
    'tensorboard_output_env_var',
    'TENSORBOARD_OUTPUT_DIR',
    'Environment variable to use to retrieve TensorBoard output directory.',
)
_SKIP_IMPLEMENTATIONS = flags.DEFINE_list(
    'skip_implementations',
    [],
    'A comma-separated list of implementations to skip.',
)


# MaxText DeepSeek-v3 shapes.
EXAMPLE = {
    'lhs': jax.ShapeDtypeStruct((262144, 7168), jnp.bfloat16),
    'rhs': jax.ShapeDtypeStruct((256, 7168, 2048), jnp.bfloat16),
    'group_sizes': tokamax.RaggedDotGroupSizes(
        jax.ShapeDtypeStruct((256,), dtype=jnp.int32), 262144),
}


class RaggedDotBenchmark(parameterized.TestCase):
  """Benchmarks for ragged dot."""

  def _write_benchmark_res(
      self, res: tokamax.BenchmarkData, metric_tag: str
  ) -> None:
    tblog_dir = os.environ.get(_TENSORBOARD_OUTPUT_ENV_VAR.value)
    if tblog_dir:
      try:
        tb_writer = SummaryWriter(log_dir=tblog_dir)
        for i, value in enumerate(res.evaluation_times_ms):
          tb_writer.add_scalar(metric_tag, value, global_step=i)
        tb_writer.close()
      except (OSError, IOError):
        logging.exception('Error writing TensorBoard logs')
    else:
      logging.info(
          'metric_tag=%s, median benchmark time (ms): %s',
          metric_tag,
          res.median_evaluation_time_ms,
      )

  @parameterized.product(
      implementation=(None, 'xla', 'triton', 'mosaic'),
      benchmark_mode=('forward', 'forward_and_vjp'),
  )
  def test_ragged_dot(self, implementation, benchmark_mode):
    """Benchmarks the ragged dot operation."""
    if str(implementation) in _SKIP_IMPLEMENTATIONS.value:
      self.skipTest(f'Skipping implementation {implementation}')

    ragged_dot_fn = functools.partial(  # pylint: disable=g-long-ternary
        tokamax.ragged_dot,
        implementation=implementation,
    )

    fn, args = tokamax.standardize_function(
        ragged_dot_fn,
        kwargs=EXAMPLE,
        mode=benchmark_mode,
    )
    fn = jax.jit(fn)
    res = tokamax.benchmark(fn, args)
    metric_tag = f"ragged_dot/{implementation or 'default'}/{benchmark_mode}"
    self._write_benchmark_res(res, metric_tag)

  def test_gmm_v2_maxtext(self):
    device = jax.devices()[0]
    if not (device.platform == 'tpu' and pltpu.get_tpu_info().generation >= 5):
      self.skipTest('GMM v2 benchmark requires TPU v5+.')

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
        mode='forward',
    )
    fn = jax.jit(fn)
    res = tokamax.benchmark(fn, args, method='hermetic_xprof')
    self._write_benchmark_res(res, 'ragged_dot/gmm_v2_maxtext/forward')

  def test_tgmm_v2_maxtext(self):
    device = jax.devices()[0]
    if not (device.platform == 'tpu' and pltpu.get_tpu_info().generation >= 5):
      self.skipTest('TGMM v2 benchmark requires TPU v5+.')

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
    fn, args = tokamax.standardize_function(
        drhs_op,
        kwargs=benchmark_config,
        mode='forward',
    )
    fn = jax.jit(fn)
    res = tokamax.benchmark(fn, args, method='hermetic_xprof')
    self._write_benchmark_res(res, 'ragged_dot/tgmm_v2_maxtext/forward')

  @parameterized.named_parameters(
      dict(
          testcase_name='prefill_gate_up',
          m=81920,  # 8192 tokens * topk 10.
          k=4096,  # hidden_size.
          n=2 * 1024,  # gate and up, each moe_intermediate_size wide.
          fuse_act='silu',
      ),
      dict(
          testcase_name='prefill_down',
          m=81920,
          k=1024,  # moe_intermediate_size.
          n=4096,  # hidden_size.
          fuse_act=None,
      ),
      dict(
          testcase_name='decode_gate_up',
          m=1280,  # 128 tokens * topk 10.
          k=4096,
          n=2 * 1024,
          fuse_act='silu',
      ),
      dict(
          testcase_name='decode_down',
          m=1280,
          k=1024,
          n=4096,
          fuse_act=None,
      ),
  )
  def test_gmm_v2_ullm(self, m, k, n, fuse_act):
    device = jax.devices()[0]
    if not (device.platform == 'tpu' and pltpu.get_tpu_info().generation >= 5):
      self.skipTest('ULLM MoE GMM v2 benchmark requires TPU v5+.')

    num_groups = 512  # Global number of experts.
    num_local_groups = 64  # Experts per EP shard (512 / 8).
    group_offset = 256  # First expert of EP shard 4 (a middle shard).
    block_size = k
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_local_groups, k, n), jnp.bfloat16)
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
        zero_initialize=False,
        fuse_gateup_activation=fuse_act,
        preferred_element_type=jnp.bfloat16,
    )
    fn, args = tokamax.standardize_function(
        gmm_op,
        kwargs=benchmark_config,
        mode='forward',
    )
    fn = jax.jit(fn)
    res = tokamax.benchmark(fn, args, method='hermetic_xprof')
    # Use self._testMethodName to get e.g.
    # test_gmm_v2_ullmprefill_gate_up or extract the suffix
    tag_suffix = self._testMethodName.split('test_gmm_v2_ullm')[-1]
    self._write_benchmark_res(res, f'ragged_dot/gmm_v2_ullm/{tag_suffix}')


if __name__ == '__main__':
  absltest.main()
