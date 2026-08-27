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

"""Benchmarks for triangle attention.

Triangle attention is a special case of dot_product_attention used in AlphaFold
2. Specialized implementations exist, such as in cuEquivariance
https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_jax.triangle_attention.html
"""

import functools
import json
import os
from typing import Any
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
  import cuequivariance_jax  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]
except ImportError:
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


def get_example(
    n: int, num_heads: int = 4, head_dim: int = 64, dtype=jnp.bfloat16
) -> Any:
  """Generates example inputs for triangle_attention."""
  return {
      'query': jax.ShapeDtypeStruct((n, n, num_heads, head_dim), dtype),
      'key': jax.ShapeDtypeStruct((n, n, num_heads, head_dim), dtype),
      'value': jax.ShapeDtypeStruct((n, n, num_heads, head_dim), dtype),
      'bias': jax.ShapeDtypeStruct((1, num_heads, n, n), dtype),
      'mask': jax.ShapeDtypeStruct((n, 1, 1, n), bool),
      'scale': 1.25,
  }


# Its important to convert the inputs to cuEquivariance format outside of the
# function being benchmarked as otherwise cuEquivariance performance will
# include the overhead of this conversion.
def _args_to_cuequivariance(args):
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


def _out_from_cuequivariance(out):
  """Converts cuEquivariance output to standard format."""
  out = out[0].squeeze(axis=0)
  return jnp.transpose(out, (0, 2, 1, 3))


def setUpModule():  # pylint: disable=invalid-name
  """Runs once before any tests in this module start."""
  metadata_dir = os.environ.get('WORKLOAD_METADATA_DIR')
  if not metadata_dir:
    return

  metadata: dict[str, str] = {}
  if jax.default_backend() == 'gpu':
    if (cuda_versions := jax._src.lib.cuda_versions) is not None:  # pylint: disable=protected-access
      metadata['cudnn_version'] = str(cuda_versions.cudnn_get_version())

    if cuequivariance_jax is not None:
      metadata['cuequivariance_version'] = cuequivariance_jax.__version__

  if metadata:
    with open(os.path.join(metadata_dir, 'workload_info.json'), 'w') as f:
      json.dump(metadata, f)


class TriangleAttentionBenchmark(parameterized.TestCase):
  """Benchmarks for different triangle_attention implementations."""

  @parameterized.product(
      implementation=(
          'mosaic',
          'cudnn',
          'xla',
          'cuequivariance',
      ),
      benchmark_mode=('forward', 'forward_and_vjp'),
      n=(384, 768),
  )
  def test_triangle_attention(self, implementation, benchmark_mode, n):
    """Test triangle_attention."""

    if str(implementation) in _SKIP_IMPLEMENTATIONS.value:
      self.skipTest(
          f"Skipping implementation '{implementation}' as per"
          ' --skip_implementations flag.'
      )

    abstract_inputs = get_example(n)
    example_ref = numerics.random_initialize(abstract_inputs, seed=0)
    example = example_ref

    fn = functools.partial(
        tokamax.dot_product_attention, implementation=implementation
    )

    if implementation == 'cuequivariance':
      if cuequivariance_jax is None:
        self.skipTest('cuEquivariance is not installed.')

      example = _args_to_cuequivariance(example)
      fn = cuequivariance_jax.triangle_attention

    fn, args = tokamax.standardize_function(
        fn,
        kwargs=example,
        mode=benchmark_mode,
    )
    fn = jax.jit(fn)
    res = tokamax.benchmark(fn, args)

    common.write_tensorboard_logs(
        tensorboard_output=_TENSORBOARD_OUTPUT_ENV_VAR.value,
        value=res.evaluation_times_ms,
        metric_tag=(
            f'triangle_attention/n_{n}/{implementation}/{benchmark_mode}'
        ),
    )

    # --------------------------------------------------------------------------
    # Numerics
    # --------------------------------------------------------------------------
    # Verify that the outputs are the same for all implementations.
    if benchmark_mode == 'forward':
      fn_ref, args_ref = tokamax.standardize_function(
          functools.partial(
              tokamax.dot_product_attention, implementation='xla'
          ),
          kwargs=example_ref,
          mode=benchmark_mode,
      )
      out_ref = jax.jit(fn_ref)(args_ref)

      out_actual = (
          _out_from_cuequivariance(fn(args))
          if implementation == 'cuequivariance'
          else fn(args)
      )

      diff = numerics.array_diff_summary(
          expected=out_ref,
          actual=out_actual,
      )
      logging.info(
          'Numeric Diff (implementation=%s, n=%d, mode=%s): max_abs_diff=%.8f,'
          ' l2_diff=%.8f',
          implementation,
          n,
          benchmark_mode,
          diff.max_absolute_diff,
          diff.l2_diff,
      )
      common.write_tensorboard_logs(
          tensorboard_output=_TENSORBOARD_OUTPUT_ENV_VAR.value,
          value=diff.max_absolute_diff,
          metric_tag=(
              f'triangle_attention_numerics/n_{n}/{implementation}/{benchmark_mode}/max_absolute_diff'
          ),
      )
      common.write_tensorboard_logs(
          tensorboard_output=_TENSORBOARD_OUTPUT_ENV_VAR.value,
          value=diff.l2_diff,
          metric_tag=(
              f'triangle_attention_numerics/n_{n}/{implementation}/{benchmark_mode}/l2_diff'
          ),
      )


if __name__ == '__main__':
  absltest.main()
