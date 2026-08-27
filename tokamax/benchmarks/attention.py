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
from tokamax._src import gpu_utils
from tokamax._src import numerics
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
}


def setUpModule():  # pylint: disable=invalid-name
  """Runs once before any tests in this module start."""
  metadata_dir = os.environ.get('WORKLOAD_METADATA_DIR')
  if not metadata_dir:
    return

  metadata: dict[str, str] = {}
  if jax.default_backend() == 'gpu':
    if (cuda_versions := jax._src.lib.cuda_versions) is not None:  # pylint: disable=protected-access
      metadata['cudnn_version'] = str(cuda_versions.cudnn_get_version())

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

    logging.info('device_kind=%s', jax.devices()[0].device_kind)

    example_ref = numerics.random_initialize(EXAMPLES[args_spec_name])
    example = example_ref

    fn = functools.partial(
        tokamax.dot_product_attention, implementation=implementation
    )
    # TODO: Mosaic GPU B200 VJP does not support head dim of 256.
    if (
        gpu_utils.is_sm100()
        and args_spec_name == 'basic'
        and implementation in ('mosaic', 'cudnn', None)
        and benchmark_mode == 'forward_and_vjp'
    ):
      self.skipTest('Mosaic GPU VJP does not support head dim of 256.')

    fn, args = tokamax.standardize_function(
        fn,
        kwargs=example,
        mode=benchmark_mode,
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

    # --------------------------------------------------------------------------
    # Autotuning Benchmark
    # --------------------------------------------------------------------------
    if (
        implementation == 'mosaic'
        and benchmark_mode == 'forward_and_vjp'
        and args_spec_name == 'basic'
    ):
      t1 = time.time()
      autotune_res = tokamax.autotune(fn, args, ignore_cache=True)
      time_autotune = time.time() - t1
      time_autotune_ms = time_autotune * 1000

      common.write_tensorboard_logs(
          tensorboard_output=_TENSORBOARD_OUTPUT_ENV_VAR.value,
          value=time_autotune_ms,
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

    # --------------------------------------------------------------------------
    # Numerics
    # --------------------------------------------------------------------------
    # Checking for for numerical equivalence with the xla_chunked implementation
    # is a basic check that lik-for-like operaions are being benchmarked.
    fn_ref, args_ref = tokamax.standardize_function(
        functools.partial(
            tokamax.dot_product_attention, implementation='xla_chunked'
        ),
        kwargs=example_ref,
        mode=benchmark_mode,
    )
    out_ref = jax.jit(fn_ref)(args_ref)
    out_actual = fn(args)

    if benchmark_mode == 'forward_and_vjp':
      out_ref, _ = out_ref
      out_actual, _ = out_actual

    diff = numerics.array_diff_summary(
        expected=out_ref,
        actual=out_actual,
    )
    common.write_tensorboard_logs(
        tensorboard_output=_TENSORBOARD_OUTPUT_ENV_VAR.value,
        value=diff.max_absolute_diff,
        metric_tag=(
            f"attention_numerics/{args_spec_name}/{implementation or 'default'}/{benchmark_mode}/max_absolute_diff"
        ),
    )
    common.write_tensorboard_logs(
        tensorboard_output=_TENSORBOARD_OUTPUT_ENV_VAR.value,
        value=diff.l2_diff,
        metric_tag=(
            f"attention_numerics/{args_spec_name}/{implementation or 'default'}/{benchmark_mode}/l2_diff"
        ),
    )


if __name__ == '__main__':
  absltest.main()
