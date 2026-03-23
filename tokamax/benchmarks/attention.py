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

import time

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
        and args_spec_name == 'basic'
    ):
      self.skipTest('Skipping cudnn forward_and_vjp on B200.')

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
    fn, args = tokamax.standardize_function(
        tokamax.dot_product_attention,
        kwargs=example,
        mode=benchmark_mode,  # pytype: disable=wrong-arg-types
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
            f"attention/{args_spec_name}/{implementation or 'default'}/{benchmark_mode}"
        ),
    )

    # Benchmark autotuning for Mosaic.
    if implementation == 'mosaic' and benchmark_mode == 'forward_and_vjp':
      t1 = time.time()
      autotune_res = tokamax.autotune(fn, args)
      time_autotune = time.time() - t1

      common.write_tensorboard_logs(
          tensorboard_output=_TENSORBOARD_OUTPUT_ENV_VAR.value,
          value=time_autotune,
          metric_tag=(
              f'attention/{args_spec_name}/mosaic/forward_and_vjp/autotuning_time'
          ),
      )

      @jax.jit
      def fn_autotuned(args):
        with autotune_res:
          return fn(args)

      res_autotuned = tokamax.benchmark(fn_autotuned, args)

      common.write_tensorboard_logs(
          tensorboard_output=_TENSORBOARD_OUTPUT_ENV_VAR.value,
          value=res_autotuned.evaluation_times_ms,
          metric_tag=(
              f'attention/{args_spec_name}/mosaic/forward_and_vjp/autotuned'
          ),
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
