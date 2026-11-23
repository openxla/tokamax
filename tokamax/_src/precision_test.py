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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import numerics
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

    x = jax.ShapeDtypeStruct((32, 32), dtype=dtype)
    y = jax.ShapeDtypeStruct((32, 32), dtype=dtype)
    x, y = numerics.random_initialize((x, y))

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

    x = jax.ShapeDtypeStruct((32, 32), dtype=jnp.float32)
    y = jax.ShapeDtypeStruct((32, 32), dtype=jnp.float32)
    x, y = numerics.random_initialize((x, y))

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

  @parameterized.parameters(
      ('NVIDIA Corporation', '7.5', jax.lax.DotAlgorithmPreset.F32_F32_F32),
      ('NVIDIA Corporation', '9.0', jax.lax.DotAlgorithmPreset.BF16_BF16_F32),
      # On ROCm, `compute_capability` is the architecture name, not a number.
      (
          'Advanced Micro Devices, Inc',
          'gfx942',
          jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
      ),
      (
          'Advanced Micro Devices, Inc',
          'gfx1151',
          jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
      ),
  )
  def test_to_dot_algorithm_preset_gpu_bf16(
      self, device_vendor, compute_capability, expected
  ):
    device = mock.Mock(
        platform='gpu',
        device_vendor=device_vendor,
        compute_capability=compute_capability,
    )
    with (
        mock.patch.object(jax, 'default_backend', return_value='gpu'),
        mock.patch.object(jax, 'devices', return_value=[device]),
    ):
      preset = precision.to_dot_algorithm_preset(
          jnp.bfloat16, jnp.bfloat16, jax.lax.Precision.DEFAULT
      )
    self.assertEqual(preset, expected)


if __name__ == '__main__':
  absltest.main()
