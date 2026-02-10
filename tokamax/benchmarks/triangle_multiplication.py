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

try:
  import cuequivariance_jax  # pylint: disable=g-import-not-at-top,import-error
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

triangle_multiplication = api.triangle_multiplication
dtype = jnp.bfloat16


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

      # Let cuequivariance initialize its own weights by not providing them.
      # A key is needed for initialization.
      key = jax.random.PRNGKey(42)
      fn_partial = functools.partial(
          cueq.triangle_multiplicative_update,
          direction=all_inputs['triangle_type'],
          key=key,
          mask=all_inputs['mask'].astype(dtype),
      )
      dynamic_args = {
          'x': all_inputs['x'],
      }
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
