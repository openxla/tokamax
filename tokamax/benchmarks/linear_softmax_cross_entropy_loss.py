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


# Representative shapes from real LLM vocabularies.
EXAMPLES = {
    'qwen3-8b': {
        'x': jax.ShapeDtypeStruct((4096, 4096), jnp.bfloat16),
        'labels': jax.ShapeDtypeStruct((4096,), jnp.int32),
        'w': jax.ShapeDtypeStruct((4096, 151936), jnp.bfloat16),
        'reduction': 'mean',
    },
    'gemma3-4b': {
        'x': jax.ShapeDtypeStruct((4096, 2560), jnp.bfloat16),
        'labels': jax.ShapeDtypeStruct((4096,), jnp.int32),
        'w': jax.ShapeDtypeStruct((2560, 262144), jnp.bfloat16),
        'reduction': 'mean',
    },
    'gemma3-7b': {
        'x': jax.ShapeDtypeStruct((4096, 3840), jnp.bfloat16),
        'labels': jax.ShapeDtypeStruct((4096,), jnp.int32),
        'w': jax.ShapeDtypeStruct((3840, 262144), jnp.bfloat16),
        'reduction': 'mean',
    },
    'llama3.1-8b': {
        'x': jax.ShapeDtypeStruct((4096, 4096), jnp.bfloat16),
        'labels': jax.ShapeDtypeStruct((4096,), jnp.int32),
        'w': jax.ShapeDtypeStruct((4096, 128256), jnp.bfloat16),
        'reduction': 'mean',
    },
    'deepseek-v3-671b': {
        'x': jax.ShapeDtypeStruct((8192, 7168), jnp.bfloat16),
        'labels': jax.ShapeDtypeStruct((8192,), jnp.int32),
        'w': jax.ShapeDtypeStruct((7168, 128256), jnp.bfloat16),
        'reduction': 'mean',
    },
    'gpt-oss-120b': {
        'x': jax.ShapeDtypeStruct((4096, 2880), jnp.bfloat16),
        'labels': jax.ShapeDtypeStruct((4096,), jnp.int32),
        'w': jax.ShapeDtypeStruct((2880, 201088), jnp.bfloat16),
        'reduction': 'mean',
    },
}


class LinearSoftmaxCrossEntropyLossBenchmark(parameterized.TestCase):
  """Benchmarks for linear softmax cross-entropy loss."""

  @parameterized.product(
      implementation=(None, 'xla', 'triton', 'mosaic_gpu'),
      benchmark_mode=('forward', 'forward_and_vjp'),
      args_spec_name=tuple(EXAMPLES.keys()),
  )
  def test_linear_softmax_cross_entropy_loss(
      self, implementation, benchmark_mode, args_spec_name
  ):
    """Benchmarks the linear softmax cross-entropy loss operation."""
    if str(implementation) in _SKIP_IMPLEMENTATIONS.value:
      self.skipTest(f'Skipping implementation {implementation}')

    if implementation in ('triton', 'mosaic_gpu') and jax.default_backend() != 'gpu':
      self.skipTest(f'{implementation} implementation is GPU-only.')

    example = EXAMPLES[args_spec_name] | {'implementation': implementation}
    fn, args = tokamax.standardize_function(
        tokamax.linear_softmax_cross_entropy_loss,
        kwargs=example,
        mode=benchmark_mode,  # pytype: disable=wrong-arg-types
    )
    fn = jax.jit(fn)
    res = tokamax.benchmark(fn, args)

    common.write_tensorboard_logs(
        tensorboard_output=_TENSORBOARD_OUTPUT_ENV_VAR.value,
        value=res.evaluation_times_ms,
        metric_tag=(
            f'linear_softmax_cross_entropy_loss/{args_spec_name}'
            f'/{implementation or "default"}/{benchmark_mode}'
        ),
    )


if __name__ == '__main__':
  absltest.main()
