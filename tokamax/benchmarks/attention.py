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

import functools
import json
import os
import time
from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import tokamax
from tokamax._src import numerics
from tokamax.benchmarks import common

try:
  import cuequivariance_jax  # pylint: disable=g-import-not-at-top,import-error # pytype: disable=import-error
except Exception:  # pylint: disable=broad-except
  cuequivariance_jax = None


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
        'scale': 1.25,
    },
}


def _to_cuequivariance(args):
  """Converts args to cuEquivariance format."""
  transpose = lambda x: jnp.transpose(x, (0, 2, 1, 3))
  out = {
      'q': transpose(args['query']),
      'k': transpose(args['key']),
      'v': transpose(args['value']),
      'bias': args['bias'],
      'mask': args['mask'],
  }
  out = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), out)
  out['scale'] = args['scale']
  return out


def setUpModule():  # pylint: disable=invalid-name
  """Runs once before any tests in this module start."""
  metadata_dir = os.environ.get('WORKLOAD_METADATA_DIR')
  if not metadata_dir:
    return

  metadata: dict[str, str] = {}
  if jax.default_backend() == 'gpu':
    try:
      cudnn_version = jax._src.lib.cuda_versions.cudnn_get_version()  # pylint: disable=protected-access
      metadata['cudnn_version'] = str(cudnn_version)
    except AttributeError:
      pass

    try:
      metadata['cuequivariance_version'] = cuequivariance_jax.__version__
    except AttributeError:
      pass

  if metadata:
    with open(os.path.join(metadata_dir, 'workload_info.json'), 'w') as f:
      json.dump(metadata, f)


class AttentionBenchmark(parameterized.TestCase):
  """Benchmarks for different attention implementations."""

  @parameterized.product(
      implementation=(
          None,
          'triton',
          'mosaic',
          'cudnn',
          'xla',
          'xla_chunked',
          'cuequivariance',
      ),
      benchmark_mode=('forward', 'forward_and_vjp'),
      args_spec_name=tuple(EXAMPLES.keys()),
  )
  def test_attention(self, implementation, benchmark_mode, args_spec_name):
    """Test attention."""

    if str(implementation) in _SKIP_IMPLEMENTATIONS.value:
      self.skipTest(
          f"Skipping implementation '{implementation}' as per"
          ' --skip_implementations flag.'
      )

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

    logging.info('device_kind=%s', jax.devices()[0].device_kind)

    example_ref = numerics.random_initialize(EXAMPLES[args_spec_name])
    example = example_ref

    fn = functools.partial(
        tokamax.dot_product_attention, implementation=implementation
    )

    if implementation == 'cuequivariance':
      if cuequivariance_jax is None:
        self.skipTest('cuEquivariance is not installed.')

      if args_spec_name == 'basic':
        self.skipTest(
            'Skipping cuequivariance for basic shape is not supported.'
        )

      example = _to_cuequivariance(example)
      fn = cuequivariance_jax.triangle_attention

    fn, args = tokamax.standardize_function(
        fn,
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
    if (
        implementation == 'mosaic'
        and benchmark_mode == 'forward_and_vjp'
        and args_spec_name == 'basic'
    ):
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

    # Numerics test.
    if benchmark_mode == 'forward':
      fn_ref, args_ref = tokamax.standardize_function(
          functools.partial(
              tokamax.dot_product_attention, implementation='xla_chunked'
          ),
          kwargs=example_ref,
          mode=benchmark_mode,  # pytype: disable=wrong-arg-types
      )
      out_ref = jax.jit(fn_ref)(args_ref)

      out_actual = fn(args)
      if implementation == 'cuequivariance':
        # cuEquivariance returns (output, log-sum-exp, maximum value).
        out_actual = out_actual[0].squeeze(axis=0)
        out_actual = jnp.transpose(out_actual, (0, 2, 1, 3))

      diff = numerics.array_diff_summary(
          expected=out_ref,
          actual=out_actual,
      )
      logging.info(
          'max_absolute_diff: %s, max_absolute_diff_values: %s l2_diff: %s',
          str(diff.max_absolute_diff),
          str(diff.max_absolute_diff_values),
          str(diff.l2_diff),
      )


if __name__ == '__main__':
  absltest.main()
