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

"""Benchmarks for linear softmax cross-entropy loss."""

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import tokamax
from tokamax.benchmarks import common


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

EXAMPLES = {
    'medium': {
        'x': jax.ShapeDtypeStruct((4096, 1024), jnp.bfloat16),
        'labels': jax.ShapeDtypeStruct((4096,), jnp.int32),
        'weights': jax.ShapeDtypeStruct((1024, 4096), jnp.bfloat16),
    }
}


class LinearSoftmaxCrossEntropyLossBenchmark(parameterized.TestCase):
  """Benchmarks for different linear softmax cross-entropy loss implementations."""

  @parameterized.product(
      implementation=(None, 'xla', 'mosaic_tpu', 'chunked_xla'),
      benchmark_mode=('forward', 'forward_and_vjp'),
      args_spec_name=tuple(EXAMPLES.keys()),
  )
  def test_linear_softmax_cross_entropy_loss(
      self, implementation, benchmark_mode, args_spec_name
  ):
    """Test linear_softmax_cross_entropy_loss."""

    logging.info('device_kind=%s', jax.devices()[0].device_kind)

    if implementation == 'mosaic_tpu' and jax.default_backend() != 'tpu':
      self.skipTest('Pallas Mosaic TPU is only supported on TPUs.')

    if str(implementation) in _SKIP_IMPLEMENTATIONS.value:
      self.skipTest(
          f"Skipping implementation '{implementation}' as per"
          ' --skip_implementations flag.'
      )

    example = EXAMPLES[args_spec_name] | {
        'reduction': 'sum',
        'implementation': implementation,
    }
    fn, args = tokamax.standardize_function(
        tokamax.linear_softmax_cross_entropy_loss,
        kwargs=example,
        mode=benchmark_mode,  # pytype: disable=wrong-keyword-args
    )
    fn = jax.jit(fn)
    res = tokamax.benchmark(fn, args)
    res_wallclock = tokamax.benchmark(fn, args, method='wallclock')

    logging.info(
        'wallclock_median_time_ms: %s', res_wallclock.median_evaluation_time_ms
    )

    common.write_tensorboard_logs(
        tensorboard_output=_TENSORBOARD_OUTPUT_ENV_VAR.value,
        value=res.evaluation_times_ms,
        metric_tag=(
            f"linear_softmax_cross_entropy_loss/{args_spec_name}/{implementation or 'default'}/{benchmark_mode}"
        ),
    )


if __name__ == '__main__':
  absltest.main()
