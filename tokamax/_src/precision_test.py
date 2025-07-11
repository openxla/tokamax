# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import precision


class PrecisionTest(parameterized.TestCase):

  @parameterized.product(
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32),
      precision_old=(
          None,
          jax.lax.Precision.DEFAULT,
          jax.lax.Precision.HIGH,
          jax.lax.Precision.HIGHEST,
          'highest',
      ),
  )
  def test_precision_equivalence(self, dtype, precision_old):
    # TODO
    if jax.default_backend() == 'cpu':
      self.skipTest('New precision API broken on CPU.')

    precision_new = precision.to_dot_algorithm_preset(
        a_dtype=dtype, b_dtype=dtype, precision=precision_old
    )

    x = jax.random.normal(jax.random.PRNGKey(0), (32, 32), dtype=dtype)
    y = jax.random.normal(jax.random.PRNGKey(1), (32, 32), dtype=dtype)

    out_old = jax.jit(functools.partial(jnp.dot, precision=precision_old))(x, y)
    out_new = jax.jit(functools.partial(jnp.dot, precision=precision_new))(x, y)

    chex.assert_trees_all_equal(out_new, out_old)

  def test_precision_input_dtype(self):
    types = {
        'F16_F16_F16': jnp.float16,
        'F16_F16_F32': jnp.float16,
        'BF16_BF16_BF16': jnp.bfloat16,
        'BF16_BF16_F32': jnp.bfloat16,
        'BF16_BF16_F32_X3': jnp.float32,
        'BF16_BF16_F32_X6': jnp.float32,
        'BF16_BF16_F32_X9': jnp.float32,
        'F32_F32_F32': jnp.float32,
        'TF32_TF32_F32': jnp.float32,
        'TF32_TF32_F32_X3': jnp.float32,
        'F64_F64_F64': jnp.float64,
    }

    for preset, dtype in types.items():
      preset = jax.lax.DotAlgorithmPreset[preset]
      self.assertEqual(precision.precision_input_dtype(preset), dtype)

  @parameterized.parameters(None, jax.lax.Precision.DEFAULT, 'BF16_BF16_F32_X6')
  def test_precision_jax_default(self, default_precision):

    # TODO
    if jax.default_backend() == 'cpu':
      self.skipTest('New precision API broken on CPU.')

    x = jax.random.normal(jax.random.PRNGKey(0), (32, 32), dtype=jnp.float32)
    y = jax.random.normal(jax.random.PRNGKey(1), (32, 32), dtype=jnp.float32)

    with jax.default_matmul_precision('BF16_BF16_F32_X3'):
      dot_gt = jax.jit(functools.partial(jnp.dot, precision=default_precision))
      expected = dot_gt(x, y)
      explicit_precision = precision.to_dot_algorithm_preset(
          jnp.float32, jnp.float32, default_precision
      )

    dot = jax.jit(functools.partial(jnp.dot, precision=explicit_precision))
    chex.assert_trees_all_equal(dot(x, y), expected)

  def test_to_dot_algorithm_preset_invalid_name(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported enum value'):
      precision.to_dot_algorithm_preset(
          jnp.float32, jnp.float32, 'invalid_precision'
      )


if __name__ == '__main__':
  absltest.main()
