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
"""Benchmarks for triangle_multiplication."""

import functools
import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tensorboardX import writer

from tokamax._src import benchmarking
from tokamax._src import numerics
from tokamax._src.ops.triangle_multiplication import api

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

triangle_multiplication = api.triangle_multiplication
dtype = jnp.bfloat16


try:
  import cuequivariance_jax  # pylint: disable=g-import-not-at-top,import-error
except ImportError:
  cuequivariance_jax = None


def get_example(n, c=128, h=32, d=128):
  """Generates example inputs for triangle_multiplication."""
  return {
      'x': jax.ShapeDtypeStruct((n, n, c), dtype=dtype),
      'mask': jax.ShapeDtypeStruct((n, n), dtype=jnp.bool_),
      'projection_in_weights': jax.ShapeDtypeStruct((c, 2, h), dtype=dtype),
      'gate_in_weights': jax.ShapeDtypeStruct((c, 2, h), dtype=dtype),
      'projection_out_weights': jax.ShapeDtypeStruct((h, d), dtype=dtype),
      'gate_out_weights': jax.ShapeDtypeStruct((c, d), dtype=dtype),
      'layernorm_in_scale': jax.ShapeDtypeStruct((c,), dtype=dtype),
      'layernorm_in_offset': jax.ShapeDtypeStruct((c,), dtype=dtype),
      'layernorm_out_scale': jax.ShapeDtypeStruct((h,), dtype=dtype),
      'layernorm_out_offset': jax.ShapeDtypeStruct((h,), dtype=dtype),
      'triangle_type': 'incoming',
  }


def map_to_cuequivariance_weights(tokamax_weights):
  """Maps Tokamax weights to cuEquivariance weights."""
  p_in = tokamax_weights['projection_in_weights']
  g_in = tokamax_weights['gate_in_weights']
  h_dim = p_in.shape[-1]

  # Map Tokamax (C, 2, H) -> cuEq (2*H, C)
  p_in_cueq = jnp.concatenate([p_in[:, 0, :], p_in[:, 1, :]], axis=-1)
  g_in_cueq = jnp.concatenate([g_in[:, 0, :], g_in[:, 1, :]], axis=-1)

  return {
      'norm_in_weight': tokamax_weights['layernorm_in_scale'],
      'norm_in_bias': tokamax_weights['layernorm_in_offset'],
      'p_in_weight': p_in_cueq,
      'g_in_weight': g_in_cueq,
      'norm_out_weight': tokamax_weights['layernorm_out_scale'],
      'norm_out_bias': tokamax_weights['layernorm_out_offset'],
      'p_out_weight': jnp.transpose(tokamax_weights['projection_out_weights']),
      'g_out_weight': jnp.transpose(tokamax_weights['gate_out_weights']),
      # cuEq requires biases, providing zeros as Tokamax doesn't have them.
      'p_in_bias': jnp.zeros(2 * h_dim, dtype=dtype),
      'g_in_bias': jnp.zeros(2 * h_dim, dtype=dtype),
      'p_out_bias': jnp.zeros(
          tokamax_weights['projection_out_weights'].shape[-1], dtype=dtype
      ),
      'g_out_bias': jnp.zeros(
          tokamax_weights['gate_out_weights'].shape[-1], dtype=dtype
      ),
  }


class TriangleMultiplicationBenchmark(parameterized.TestCase):
  """Benchmarks for different triangle_multiplication implementations."""

  @parameterized.product(
      implementation=(None, 'xla', 'cuequivariance'),
      benchmark_mode=('forward', 'forward_and_vjp'),
      n=(384, 768),
  )
  def test_triangle_multiplication(self, implementation, benchmark_mode, n):
    """Test triangle_multiplication."""

    if (implementation or 'None') in _SKIP_IMPLEMENTATIONS.value:
      self.skipTest(
          f"Skipping implementation '{implementation}' as per"
          ' --skip_implementations flag.'
      )

    if implementation == 'cuequivariance' and cuequivariance_jax is None:
      self.skipTest('cuequivariance is not installed.')

    input_dim, hidden_dim, output_dim = 128, 32, 128

    # Initialize all inputs once using Tokamax schema.
    example_schema = get_example(n, input_dim, hidden_dim, output_dim)
    all_inputs = numerics.random_initialize(example_schema, seed=0)

    if implementation == 'cuequivariance':
      cueq = cuequivariance_jax
      if cueq is None:
        self.skipTest('cuequivariance is not installed.')
        return

      cueq_weights = map_to_cuequivariance_weights(all_inputs)
      fn_partial = functools.partial(
          cueq.triangle_multiplicative_update,
          direction=all_inputs['triangle_type'],
      )
      dynamic_args = {
          'x': all_inputs['x'],
          'mask': all_inputs['mask'],
      } | cueq_weights

      # Numerical verification against Tokamax's XLA implementation.
      xla_out = jax.jit(
          functools.partial(triangle_multiplication, implementation='xla'),
          static_argnames=['triangle_type']
      )(**all_inputs)
      cueq_out = jax.jit(fn_partial)(**dynamic_args)
      diff = numerics.array_diff_summary(xla_out, cueq_out)
      # TODO(b/481381116): Log this to the proto.
      logging.info('Numerical diff (xla vs cuequivariance): %s', diff)
    else:  # Tokamax implementations
      fn_partial = functools.partial(
          triangle_multiplication,
          implementation=implementation,
      )
      dynamic_args = all_inputs

    fn, actual_args = benchmarking.standardize_function(
        fn_partial,
        kwargs=dynamic_args,
        mode=benchmark_mode,
        seed=None,
    )

    bench = benchmarking.compile_benchmark(fn, actual_args)
    res = bench(actual_args)

    metric_tag = (
        f"triangle_multiplication/{implementation or 'default'}/{benchmark_mode}"
    )
    tblog_dir = os.environ.get(_TENSORBOARD_OUTPUT_ENV_VAR.value)

    if tblog_dir:
      try:
        tb_writer = SummaryWriter(log_dir=tblog_dir)
        tb_writer.add_scalar(
            metric_tag,
            res.median_evaluation_time_ms,
            global_step=0,
        )
        # Also log individual evaluation times for more detail in TensorBoard
        for i, value in enumerate(res.evaluation_times_ms):
          tb_writer.add_scalar(
              f'{metric_tag}/all_iterations', value, global_step=i
          )

        tb_writer.close()
      except (OSError, IOError):
        logging.warning(
            'Failed to write to TensorBoard output directory: %s',
            tblog_dir,
        )
    else:
      logging.info(
          'n=%d, implementation=%s, benchmark_mode=%s, median benchmark time'
          ' (ms): %s',
          n,
          implementation,
          benchmark_mode,
          res.median_evaluation_time_ms,
      )


if __name__ == '__main__':
  absltest.main()
