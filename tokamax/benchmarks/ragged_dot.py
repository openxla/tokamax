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
import jax.numpy as jnp
from tensorboardX import writer
from tokamax._src import benchmarking
from tokamax._src.ops.ragged_dot import api


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

ragged_dot = api.ragged_dot
# Maxtext deepseek-v3 shapes.
EXAMPLE = {
    'lhs': jax.ShapeDtypeStruct((262144, 7168), jnp.bfloat16),
    'rhs': jax.ShapeDtypeStruct((256, 7168, 2048), jnp.bfloat16),
    'group_sizes': jax.ShapeDtypeStruct((256,), dtype=jnp.int32),
}


class RaggedDotBenchmark(parameterized.TestCase):

  @parameterized.product(
      implementation=(
          None,
          'triton',
          'mosaic',
      ),
      benchmark_mode=('forward', 'forward_and_vjp'),
  )
  def test_ragged_dot(self, implementation, benchmark_mode):
    """Benchmarks the ragged dot operation."""
    if implementation in _SKIP_IMPLEMENTATIONS.value:
      self.skipTest(f'Skipping implementation {implementation}')

    fn, args = benchmarking.standardize_function(
        functools.partial(
            ragged_dot,
            implementation=implementation,
        ),
        kwargs=EXAMPLE,
        mode=benchmark_mode,  # pytype: disable=wrong-arg-types
    )
    fn = jax.jit(fn)
    bench = benchmarking.compile_benchmark(fn, args)
    res = bench(args)
    metric_tag = f"ragged_dot/{implementation or 'default'}/{benchmark_mode}"
    tblog_dir = os.environ.get(_TENSORBOARD_OUTPUT_ENV_VAR.value)

    if tblog_dir:
      try:
        tb_writer = SummaryWriter(log_dir=tblog_dir)
        for i, value in enumerate(res.evaluation_times_ms):
          tb_writer.add_scalar(metric_tag, value, global_step=i)

        tb_writer.close()
      except (OSError, IOError) as e:
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
