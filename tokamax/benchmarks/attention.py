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

"""Benchmarks for attention."""

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
from tokamax._src.ops.attention import api


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

dot_product_attention = api.dot_product_attention
EXAMPLE = {
    'query': jax.ShapeDtypeStruct((32, 4096, 32, 128), jnp.bfloat16),
    'key': jax.ShapeDtypeStruct((32, 4096, 8, 128), jnp.bfloat16),
    'value': jax.ShapeDtypeStruct((32, 4096, 8, 128), jnp.bfloat16),
    'is_causal': True,
}


class AttentionBenchmark(parameterized.TestCase):
  """Benchmarks for different attention implementations."""

  @parameterized.product(
      implementation=(None, 'triton', 'mosaic', 'cudnn'),
      benchmark_mode=('forward', 'forward_and_vjp'),
  )
  def test_attention(self, implementation, benchmark_mode):
    """Test attention."""

    if (implementation or 'None') in _SKIP_IMPLEMENTATIONS.value:
      self.skipTest(
          f"Skipping implementation '{implementation}' as per"
          ' --skip_implementations flag.'
      )

    fn, args = benchmarking.standardize_function(
        functools.partial(
            dot_product_attention,
            implementation=implementation,
            is_causal=True,
        ),
        kwargs=EXAMPLE,
        mode=benchmark_mode,  # pytype: disable=wrong-arg-types
    )
    fn = jax.jit(fn)
    bench = benchmarking.compile_benchmark(fn, args)
    res = bench(args)
    metric_tag = f"attention/{implementation or 'default'}/{benchmark_mode}"
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
