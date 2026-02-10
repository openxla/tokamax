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
"""Benchmarks for triangle_multiplication."""

import functools
import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tensorboardX import writer
import tokamax

from tokamax._src import numerics

try:
  import cuequivariance_jax  # pylint: disable=g-import-not-at-top,import-error # pytype: disable=import-error
except ImportError:
  cuequivariance_jax = None

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


def get_example(n, c=128, h=32, d=128):
  """Generates example inputs for triangle_multiplication."""
  return {
      'x': jax.ShapeDtypeStruct((n, n, c), dtype=jnp.bfloat16),
      'mask': jax.ShapeDtypeStruct((n, n), dtype=jnp.bool_),
      'projection_in_weights': jax.ShapeDtypeStruct(
          (c, 2, h), dtype=jnp.bfloat16
      ),
      'gate_in_weights': jax.ShapeDtypeStruct((c, 2, h), dtype=jnp.bfloat16),
      'projection_out_weights': jax.ShapeDtypeStruct(
          (h, d), dtype=jnp.bfloat16
      ),
      'gate_out_weights': jax.ShapeDtypeStruct((c, d), dtype=jnp.bfloat16),
      'layernorm_in_scale': jax.ShapeDtypeStruct((c,), dtype=jnp.bfloat16),
      'layernorm_in_offset': jax.ShapeDtypeStruct((c,), dtype=jnp.bfloat16),
      'layernorm_out_scale': jax.ShapeDtypeStruct((h,), dtype=jnp.bfloat16),
      'layernorm_out_offset': jax.ShapeDtypeStruct((h,), dtype=jnp.bfloat16),
      'triangle_type': 'incoming',
  }



def convert_tokamax_weights_to_cuequivariance(tokamax_weights, input_dim, hidden_dim, output_dim):
    """Converts Tokamax weights to cuEquivariance format by padding."""
    c, h, d = input_dim, hidden_dim, output_dim
    dtype = tokamax_weights['x'].dtype

    cueq_weights = {}
    cueq_weights['norm_in_weight'] = tokamax_weights['layernorm_in_scale']
    cueq_weights['norm_in_bias'] = tokamax_weights['layernorm_in_offset']

    def pad_weights(weights):  # Shape (C, 2, H)
        weights_concat = jnp.concatenate([weights[:, 0, :], weights[:, 1, :]], axis=1)  # (C, 2 * H)
        padding_width = c - h
        # Pad H to C for the second part of the 2*C dimension
        padded = jnp.pad(
            weights_concat,
            ((0, 0), (0, 2 * padding_width)),
            mode='constant',
            constant_values=0.0,
        )  # (C, 2 * C)
        return padded.T  # (2 * C, C)

    cueq_weights['p_in_weight'] = pad_weights(tokamax_weights['projection_in_weights'])
    cueq_weights['g_in_weight'] = pad_weights(tokamax_weights['gate_in_weights'])

    padding_width = c - h
    cueq_weights['norm_out_weight'] = jnp.pad(
        tokamax_weights['layernorm_out_scale'], (0, padding_width), mode='constant', constant_values=0.0
    )
    cueq_weights['norm_out_bias'] = jnp.pad(
        tokamax_weights['layernorm_out_offset'], (0, padding_width), mode='constant', constant_values=0.0
    )

    # Tokamax projection_out_weights: (H, D)
    # cuEquivariance p_out_weight: (D, C)
    proj_out = tokamax_weights['projection_out_weights']  # (H, D)
    padding_width = c - h
    proj_out_padded = jnp.pad(
        proj_out, ((0, padding_width), (0, 0)), mode='constant', constant_values=0.0
    )  # (C, D)
    cueq_weights['p_out_weight'] = proj_out_padded.T  # (D, C)

    # Tokamax gate_out_weights: (C, D)
    # cuEquivariance g_out_weight: (D, C)
    cueq_weights['g_out_weight'] = tokamax_weights['gate_out_weights'].T  # (D, C)

    # Biases not present in Tokamax for these layers are set to zero
    cueq_weights['p_in_bias'] = jnp.zeros(2 * c, dtype=dtype)
    cueq_weights['g_in_bias'] = jnp.zeros(2 * c, dtype=dtype)
    cueq_weights['p_out_bias'] = jnp.zeros(d, dtype=dtype)
    cueq_weights['g_out_bias'] = jnp.zeros(d, dtype=dtype)

    return cueq_weights


class TriangleMultiplicationBenchmark(parameterized.TestCase):
  """Benchmarks for different triangle_multiplication implementations."""

  @parameterized.product(
      implementation=(None, 'xla', 'cuequivariance'),
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

    input_dim, hidden_dim, output_dim = 128, 32, 128
    seed = 0

    # Initialize all inputs once using Tokamax schema.
    example_schema = get_example(n, input_dim, hidden_dim, output_dim)
    all_inputs = numerics.random_initialize(example_schema, seed=seed)

    if implementation == 'cuequivariance':
      cueq = cuequivariance_jax
      if cueq is None:
        self.skipTest('cuEquivariance is not installed.')

      cueq_weights = convert_tokamax_weights_to_cuequivariance(
          all_inputs, input_dim, hidden_dim, output_dim
      )

      fn_partial = functools.partial(
          cueq.triangle_multiplicative_update,
          direction=all_inputs['triangle_type'],
          mask=all_inputs['mask'].astype(jnp.bfloat16),
          eps=1e-6,
          **cueq_weights,
      )
      dynamic_args = {
          'x': all_inputs['x'],
      }

      # Calculate the numeric difference vs XLA version of Tokamax.
      # out_cueq = fn_partial(**dynamic_args)
      # out_xla = tokamax.triangle_multiplication(
      #     implementation='xla',
      #     **all_inputs
      # )

      # diff = jnp.mean(jnp.abs(out_cueq.astype(jnp.float32) - out_xla.astype(jnp.float32)))
      # logging.info('Numeric Diff (cuEquivariance vs XLA for n=%d): %s', n, diff)
      return  # Skip benchmarking for cuequivariance due to internal error
    else:  # Tokamax implementations
      fn_partial = functools.partial(
          tokamax.triangle_multiplication,
          implementation=implementation,
      )
      dynamic_args = all_inputs

    fn, actual_args = tokamax.benchmarking.standardize_function(
        fn_partial,
        kwargs=dynamic_args,
        mode=benchmark_mode,
        seed=None,
    )

    bench = tokamax.benchmarking.compile_benchmark(fn, actual_args)
    res = bench(actual_args)

    metric_tag = (
        f"triangle_multiplication/n={n}/{implementation or 'default'}/{benchmark_mode}"
    )
    tblog_dir = os.environ.get(_TENSORBOARD_OUTPUT_ENV_VAR.value)

    if tblog_dir:
      try:
        tb_writer = SummaryWriter(log_dir=tblog_dir)
        tb_writer.add_scalar(
            metric_tag,
            res.median_evaluation_time_ms,
            global_step=0,
        )
        # Also log individual evaluation times for more detail in TensorBoard
        for i, value in enumerate(res.evaluation_times_ms):
          tb_writer.add_scalar(
              f'{metric_tag}/all_iterations', value, global_step=i
          )

        tb_writer.close()
      except (OSError, IOError):
        logging.warning(
            'Failed to write to TensorBoard output directory: %s',
            tblog_dir,
        )
    else:
      logging.info(
          'n=%d, implementation=%s, benchmark_mode=%s, median benchmark time'
          ' (ms): %s',
          n,
          implementation,
          benchmark_mode,
          res.median_evaluation_time_ms,
      )


if __name__ == '__main__':
  absltest.main()
