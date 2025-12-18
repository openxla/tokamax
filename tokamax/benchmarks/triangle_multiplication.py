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
from typing import Any

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tensorboardX import writer
import tokamax
from tokamax._src import numerics


try:
  import cuequivariance_jax  # pylint: disable=g-import-not-at-top,import-error # pytype: disable=import-error
except ImportError:
  cuequivariance_jax = None

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


def get_example(n, c=64, h=64, d=64, dtype=jnp.float32) -> Any:
  """Generates example inputs for triangle_multiplication."""
  return {
      'x': jax.ShapeDtypeStruct((n, n, c), dtype),
      'mask': jnp.ones((n, n), dtype=bool),
      'projection_in_weights': jax.ShapeDtypeStruct((c, 2, h), dtype),
      'gate_in_weights': jax.ShapeDtypeStruct((c, 2, h), dtype),
      'projection_out_weights': jax.ShapeDtypeStruct((h, d), dtype),
      'gate_out_weights': jax.ShapeDtypeStruct((c, d), dtype),
      'layernorm_in_scale': jax.ShapeDtypeStruct((c,), dtype),
      'layernorm_in_offset': jax.ShapeDtypeStruct((c,), dtype),
      'layernorm_out_scale': jax.ShapeDtypeStruct((h,), dtype),
      'layernorm_out_offset': jax.ShapeDtypeStruct((h,), dtype),
      'triangle_type': 'incoming',
  }


def convert_tokamax_weights_to_cuequivariance(tokamax_weights):
  """Converts Tokamax weights to cuEquivariance format."""
  # Tokamax Input Proj: [C, 2, H] -> [2*H, C] for cuEquivariance.
  # Since C=H, we flatten (C, 2, H) -> (C, 2*H) then transpose -> (2*H, C)
  def transform_in(w):
    return w.reshape(w.shape[0], -1).T

  # Tokamax Output Proj: [H, D] -> [D, H] for cuEquivariance.
  def transform_out(w):
    return w.T

  cueq_weights = {}

  # Input Layers
  cueq_weights['p_in_weight'] = transform_in(
      tokamax_weights['projection_in_weights']
  )
  cueq_weights['g_in_weight'] = transform_in(tokamax_weights['gate_in_weights'])

  # Output Layers
  cueq_weights['p_out_weight'] = transform_out(
      tokamax_weights['projection_out_weights']
  )
  cueq_weights['g_out_weight'] = transform_out(
      tokamax_weights['gate_out_weights']
  )

  # Norms (Direct mapping)
  cueq_weights['norm_in_weight'] = tokamax_weights['layernorm_in_scale']
  cueq_weights['norm_in_bias'] = tokamax_weights['layernorm_in_offset']
  cueq_weights['norm_out_weight'] = tokamax_weights['layernorm_out_scale']
  cueq_weights['norm_out_bias'] = tokamax_weights['layernorm_out_offset']

  # Tokamax doesn't use linear biases, cuEquivariance does.
  # They must be explicitly zeroed out.
  cueq_weights['p_in_bias'] = None
  cueq_weights['g_in_bias'] = None
  cueq_weights['p_out_bias'] = None
  cueq_weights['g_out_bias'] = None

  return cueq_weights


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

    abstract_inputs = get_example(n)
    all_inputs = numerics.random_initialize(abstract_inputs, seed=0)

    if implementation == 'cuequivariance':
      cueq = cuequivariance_jax
      if cueq is None:
        self.skipTest('cuEquivariance is not installed.')

      cueq_weights = convert_tokamax_weights_to_cuequivariance(all_inputs)

      fn_partial = functools.partial(
          cueq.triangle_multiplicative_update,
          direction=all_inputs['triangle_type'],
          mask=all_inputs['mask'].astype(all_inputs['x'].dtype),
          eps=1e-6,
          fallback=True,
          **cueq_weights,
      )
      dynamic_args = {
          'x': all_inputs['x'],
      }

      # Calculate the numeric difference vs default version of Tokamax.
      out_cueq = fn_partial(**dynamic_args)
      out_tokamax = tokamax.triangle_multiplication(
          implementation=None, **all_inputs
      )

      diff = jnp.mean(jnp.abs(out_cueq - out_tokamax))
      logging.info(
          'Numeric Diff (cuEquivariance vs Tokamax for n=%d): %.8f', n, diff
      )
    else:  # Tokamax implementation.
      fn_partial = functools.partial(
          tokamax.triangle_multiplication,
          implementation=implementation,
      )
      dynamic_args = all_inputs

    fn, actual_args = tokamax.benchmarking.standardize_function(
        fn_partial,
        kwargs=dynamic_args,
        mode=benchmark_mode,
        seed=None,
    )

    bench = tokamax.benchmarking.compile_benchmark(fn, actual_args)
    res = bench(actual_args)

    metric_tag = (
        f"triangle_multiplication/n_{n}/{implementation or 'default'}/{benchmark_mode}"
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

        tb_writer.flush()
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
