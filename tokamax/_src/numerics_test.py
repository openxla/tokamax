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
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import layout
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics

if jax.__version_info__ >= (0, 6, 3):
  DLL = layout.Layout
else:
  DLL = layout.DeviceLocalLayout  # type: ignore

jax.config.update('jax_threefry_partitionable', False)


class NumericsTest(parameterized.TestCase):

  def test_initializer_consistency(self):
    kwargs = {
        'u': 'test',
        'x': jax.ShapeDtypeStruct((3, 4), jnp.float32),
        'y': 3.4,
        'z': jax.ShapeDtypeStruct((4,), jnp.bool_),
    }
    kwargs_random = numerics.random_initialize(kwargs)
    chex.assert_trees_all_equal_structs(kwargs, kwargs_random)

  @parameterized.parameters([True, False])
  def test_numeric_summary(self, equal_nan):
    x = jnp.array([[1.2, -2.3], [jnp.nan, 3.4]], dtype=jnp.float32)
    y = np.array([[1.2, -2.3], [np.nan, -20.0]], dtype=np.float32)

    summary_diff = numerics.array_diff_summary(x, y, equal_nan=equal_nan)
    expected = (3.4, -20.0) if equal_nan else (np.nan, np.nan)

    chex.assert_trees_all_close(
        summary_diff.max_absolute_diff_values,
        expected,
    )

    expected = 3.4 + 20.0 if equal_nan else np.nan
    chex.assert_trees_all_close(summary_diff.max_absolute_diff, expected)
    self.assertFalse(summary_diff.allclose)

    summary = numerics.array_numeric_summary(x)

    self.assertEqual(summary.has_inf, False)
    self.assertEqual(summary.has_nan, True)
    self.assertLess(abs(summary.max - 3.4), 0.0001)

  def test_random_initialize_consistency(self):
    # To allow numerics comparisons over time, the random initializer should
    # always produce the same results.

    dtypes = (
        jnp.bool_,
        jnp.bfloat16,
        jnp.float16,
        jnp.float32,
        jnp.int4,
        jnp.int8,
        jnp.int32,
        jnp.int64,
        jnp.uint4,
        jnp.uint8,
        jnp.uint32,
        jnp.uint64,
    )
    shape = (50,)
    kwargs = {d.dtype.name: jax.ShapeDtypeStruct(shape, d) for d in dtypes}
    kwargs = numerics.random_initialize(kwargs)
    kwargs = jax.tree.map(
        lambda x: np.sum(np.array(x).astype(np.float64)), kwargs
    )

    kwargs_expected = {
        'bfloat16': np.float64(10.931640625),
        'bool': np.float64(28.0),
        'float16': np.float64(3.2483978271484375),
        'float32': np.float64(-5.48963075876236),
        'int32': np.float64(415.0),
        'int4': np.float64(7.0),
        'int64': np.float64(-419.0),
        'int8': np.float64(987.0),
        'uint32': np.float64(3460.0),
        'uint4': np.float64(371.0),
        'uint64': np.float64(3657.0),
        'uint8': np.float64(3021.0),
    }

    chex.assert_trees_all_close(kwargs, kwargs_expected)

  @parameterized.parameters(jnp.bool_, jnp.float32, jnp.int32, jnp.uint8)
  def test_random_initialize_layout(self, dtype):
    shape = (2, 3, 4)
    dll = DLL((1, 2, 0), ())
    no_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    dll_layout = layout.Format(dll, no_sharding)
    spec_with_layout = jax.ShapeDtypeStruct(shape, dtype, sharding=dll_layout)
    actual = numerics.random_initialize(spec_with_layout)
    expected = numerics.random_initialize(jax.ShapeDtypeStruct(shape, dtype))
    chex.assert_trees_all_close(actual, expected)
    self.assertEqual(actual.format, dll_layout)

  def test_initializable_array(self):
    x = numerics.InitializableArray(
        value=jnp.zeros((2, 3)),
        initializer=lambda _, shape, dtype: jnp.ones(shape, dtype),
    )
    expected = jnp.zeros((2, 3))
    chex.assert_trees_all_equal(numerics.random_initialize(x), expected)

    x = numerics.InitializableArray(
        value=jax.ShapeDtypeStruct((2, 3), jnp.float32),
        initializer=lambda _, shape, dtype: jnp.ones(shape, dtype),
    )
    chex.assert_trees_all_equal(numerics.random_initialize(x), jnp.ones((2, 3)))

    x = numerics.InitializableArray(
        value=jax.ShapeDtypeStruct((2, 3), jnp.float32),
        initializer=jax.random.normal,
    )
    expected = numerics.random_initialize(x.value)
    chex.assert_trees_all_equal(numerics.random_initialize(x), expected)


if __name__ == '__main__':
  absltest.main()
