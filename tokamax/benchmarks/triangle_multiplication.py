# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
import tensorflow as tf
from tokamax._src import benchmarking
from tokamax._src.ops.triangle_multiplication import api


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
      implementation=(None, 'xla'),
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

    example = get_example(n)
    fn, args = benchmarking.standardize_function(
        functools.partial(
            triangle_multiplication,
            implementation=implementation,
        ),
        kwargs=example,
        mode=benchmark_mode,  # pytype: disable=wrong-arg-types
    )
    fn = jax.jit(fn)
    bench = benchmarking.compile_benchmark(fn, args)
    res = bench(args)
    metric_tag = (
        f"triangle_multiplication/{implementation or 'default'}/{benchmark_mode}"
    )
    tblog_dir = os.environ.get(_TENSORBOARD_OUTPUT_ENV_VAR.value)

    if tblog_dir:
      try:
        writer = tf.summary.create_file_writer(tblog_dir)
        with writer.as_default():
          for i, value in enumerate(res.evaluation_times_ms):
            tf.summary.scalar(metric_tag, value, step=i)
        writer.flush()
        writer.close()
      except (IOError, tf.errors.OpError) as e:
        logging.exception('Error writing TensorBoard logs: %s', e)
    else:
      logging.info(
          'implementation=%s, benchmark_mode=%s, benchmark time (ms): %s',
          implementation,
          benchmark_mode,
          res.median_evaluation_time_ms,
      )


if __name__ == '__main__':
  absltest.main()
