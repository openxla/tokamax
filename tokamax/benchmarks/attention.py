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

"""Benchmarks for attention."""

import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tensorboardX import writer
import tokamax
from tokamax import benchmarking


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

EXAMPLES = {
    'basic': {
        'query': jax.ShapeDtypeStruct((2, 8192, 8, 256), jnp.bfloat16),
        'key': jax.ShapeDtypeStruct((2, 8192, 8, 256), jnp.bfloat16),
        'value': jax.ShapeDtypeStruct((2, 8192, 8, 256), jnp.bfloat16),
        'is_causal': True,
    },
    'alphafold': {
        'query': jax.ShapeDtypeStruct((768, 768, 4, 64), jnp.bfloat16),
        'key': jax.ShapeDtypeStruct((768, 768, 4, 64), jnp.bfloat16),
        'value': jax.ShapeDtypeStruct((768, 768, 4, 64), jnp.bfloat16),
        'bias': jax.ShapeDtypeStruct((1, 4, 768, 768), jnp.bfloat16),
        'mask': jax.ShapeDtypeStruct((768, 1, 1, 768), bool),
    },
}


class AttentionBenchmark(parameterized.TestCase):
  """Benchmarks for different attention implementations."""

  @parameterized.product(
      implementation=(None, 'triton', 'mosaic', 'cudnn', 'xla', 'xla_chunked'),
      benchmark_mode=('forward', 'forward_and_vjp'),
      args_spec_name=tuple(EXAMPLES.keys()),
  )
  def test_attention(self, implementation, benchmark_mode, args_spec_name):
    """Test attention."""

    logging.info('device_kind=%s', jax.devices()[0].device_kind)

    # TODO: Re-enable once cuDNN bug is fixed.
    if (
        implementation == 'cudnn'
        and benchmark_mode == 'forward_and_vjp'
        and 'B200' in jax.devices()[0].device_kind
    ):
      self.skipTest('Skipping cudnn forward_and_vjp on B200.')

    # TODO: Re-enable once Mosaic GPU supports VJP on B200.
    if (
        implementation in ('mosaic', None)
        and benchmark_mode == 'forward_and_vjp'
        and 'B200' in jax.devices()[0].device_kind
    ):
      self.skipTest('Skipping Mosaic forward_and_vjp on B200.')

    if args_spec_name == 'alphafold':
      # TODO: Re-enable once Mosaic TPU supports learnable biases.
      if jax.default_backend() == 'tpu' and implementation == 'mosaic':
        self.skipTest('Skipping AlphaFold on TPU.')
      # TODO: Re-enable once Mosaic GPU supports learnable biases
      # on B200.
      if 'B200' in jax.devices()[0].device_kind and implementation == 'mosaic':
        self.skipTest('Skipping AlphaFold shape on B200.')

    if str(implementation) in _SKIP_IMPLEMENTATIONS.value:
      self.skipTest(
          f"Skipping implementation '{implementation}' as per"
          ' --skip_implementations flag.'
      )

    example = EXAMPLES[args_spec_name] | {'implementation': implementation}
    fn, args = benchmarking.standardize_function(
        tokamax.dot_product_attention,
        kwargs=example,
        mode=benchmark_mode,  # pytype: disable=wrong-arg-types
    )
    fn = jax.jit(fn)
    bench = benchmarking.compile_benchmark(fn, args)
    res = bench(args)

    res_wallclock = bench(args, method='wallclock')
    logging.info(
        'wallclock_median_time_ms: %s', res_wallclock.median_evaluation_time_ms
    )

    metric_tag = (
        f"attention/{args_spec_name}/{implementation or 'default'}/{benchmark_mode}"
    )
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

    # TODO: Add this to the proto once generic metadata is
    # supported.
    if implementation == 'cudnn':
      logging.info(
          'cudnn_version=%s',
          jax._src.lib.cuda_versions.cudnn_get_version(),  # pylint: disable=protected-access # pytype: disable=attribute-error
      )


if __name__ == '__main__':
  absltest.main()
