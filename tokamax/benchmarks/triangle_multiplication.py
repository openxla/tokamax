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
import inspect
import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tensorboardX import writer

from tokamax._src import benchmarking
from tokamax._src.ops.triangle_multiplication import api

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

triangle_multiplication = api.triangle_multiplication
dtype = jnp.bfloat16


try:
  from cuequivariance_jax import triangle_multiplicative_update as cueq_tmu  # pylint: disable=g-import-not-at-top,import-error # pytype: disable=import-error
except ImportError:
  cueq_tmu = None


def get_tokamax_weights(key, c, h, d):
  """Generates random weights for Tokamax triangle_multiplication."""
  keys = jax.random.split(key, 8)
  return {
      'projection_in_weights': jax.random.normal(
          keys[0], (c, 2, h), dtype=dtype
      ),
      'gate_in_weights': jax.random.normal(keys[1], (c, 2, h), dtype=dtype),
      'projection_out_weights': jax.random.normal(keys[2], (h, d), dtype=dtype),
      'gate_out_weights': jax.random.normal(keys[3], (c, d), dtype=dtype),
      'layernorm_in_scale': jax.random.normal(keys[4], (c,), dtype=dtype),
      'layernorm_in_offset': jax.random.normal(keys[5], (c,), dtype=dtype),
      'layernorm_out_scale': jax.random.normal(keys[6], (h,), dtype=dtype),
      'layernorm_out_offset': jax.random.normal(keys[7], (h,), dtype=dtype),
  }


def get_cuequivariance_weights(key, c, d):
  """Generates random weights for cuEquivariance triangle_multiplicative_update."""
  # D_in = c, D_out = d
  # The intermediate dimension in cuEquivariance's triangle part is also c.
  keys = jax.random.split(key, 8)
  return {
      'norm_in_weight': jnp.ones(
          c, dtype=dtype
      ),  # Typically learned, but start with ones
      'norm_in_bias': jnp.zeros(c, dtype=dtype),
      'p_in_weight': jax.random.normal(keys[0], (2 * c, c), dtype=dtype),
      'g_in_weight': jax.random.normal(keys[1], (2 * c, c), dtype=dtype),
      'norm_out_weight': jnp.ones(
          c, dtype=dtype
      ),  # Acts on the 'c' dimension
      'norm_out_bias': jnp.zeros(c, dtype=dtype),
      'p_out_weight': jax.random.normal(keys[4], (d, c), dtype=dtype),
      'g_out_weight': jax.random.normal(keys[5], (d, c), dtype=dtype),
      # Biases for projections (optional in cueq, can be None)
      'p_in_bias': jax.random.normal(keys[2], (2 * c,), dtype=dtype),
      'g_in_bias': jax.random.normal(keys[3], (2 * c,), dtype=dtype),
      'p_out_bias': jax.random.normal(keys[6], (d,), dtype=dtype),
      'g_out_bias': jax.random.normal(keys[7], (d,), dtype=dtype),
  }


class TriangleMultiplicationBenchmark(parameterized.TestCase):
  """Benchmarks for different triangle_multiplication implementations."""

  @parameterized.product(
      implementation=(None, 'xla', 'cuequivariance'),  # Added 'cuequivariance'
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

    if implementation == 'cuequivariance' and cueq_tmu is None:
      self.skipTest('cuequivariance is not installed.')

    c, h, d = 128, 32, 128  # c: Input, h: Tokamax Hidden, d: Output
    key = jax.random.PRNGKey(0)
    k1, k2, kw = jax.random.split(key, 3)

    x = jax.random.normal(k1, (n, n, c), dtype=dtype)
    mask = jax.random.randint(k2, (n, n), 0, 2).astype(jnp.bool_)
    common_inputs = {'x': x, 'mask': mask}

    if implementation == 'cuequivariance':
      weights = get_cuequivariance_weights(kw, c, d)
      fn_partial = functools.partial(cueq_tmu, direction='incoming')
      dynamic_args = common_inputs | weights
    else:  # Tokamax implementations
      weights = get_tokamax_weights(kw, c, h, d)
      fn_partial = functools.partial(
          triangle_multiplication,
          implementation=implementation,
          triangle_type='incoming',
      )
      dynamic_args = common_inputs | weights

    dynamic_args_shapes = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), dynamic_args
    )

    fn, standardized_args_shapes = benchmarking.standardize_function(
        fn_partial,
        kwargs=dynamic_args_shapes,
        mode=benchmark_mode,  # pytype: disable=wrong-arg-types
    )
    fn = jax.jit(fn)

    # Convert dynamic_args to the same order as expected by fn
    # Standardize function returns the args list it created.
    # We should match the values in dynamic_args to the shapes in standardized_args_shapes.

    def get_actual_args(template, data):
      # Flatten template and data in a consistent way.
      # However, standardized_args_shapes is already flat (list of ShapeDtypeStruct)
      # We need to map dynamic_args to that list.
      # Standardize function does:
      # ba = inspect.signature(f).bind(*args, **({} if kwargs is None else kwargs))
      # ba.apply_defaults()
      # args_flat, args_tree = jax.tree.flatten((ba.args, ba.kwargs), ...)
      # arrays, other, merge = utils.split_merge(is_array, args_flat)
      # return func, arrays

      ba = inspect.signature(fn_partial).bind(**dynamic_args)
      ba.apply_defaults()
      args_flat, _ = jax.tree.flatten((ba.args, ba.kwargs))
      arrays = [x for x in args_flat if isinstance(x, (jax.Array, jax.ShapeDtypeStruct))]
      return arrays

    actual_args = get_actual_args(standardized_args_shapes, dynamic_args)

    bench = benchmarking.compile_benchmark(fn, standardized_args_shapes)
    res = bench(actual_args)

    metric_tag = (
        f"triangle_multiplication/{implementation or 'default'}/{benchmark_mode}"
    )
    tblog_dir = os.environ.get(_TENSORBOARD_OUTPUT_ENV_VAR.value)

    if tblog_dir:
      try:
        tb_writer = SummaryWriter(log_dir=tblog_dir)
        # Log median evaluation time as required by BAP using the tag from registry
        tb_writer.add_scalar(
            metric_tag,
            res.median_evaluation_time_ms,
            global_step=0,
        )

        # Also log individual evaluation times for more detail in TensorBoard
        for i, value in enumerate(res.evaluation_times_ms):
          tb_writer.add_scalar(f"{metric_tag}/all_iterations", value, global_step=i)

        tb_writer.close()
      except (OSError, IOError) as e:
        logging.exception('Error writing TensorBoard logs: %s', e)
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
