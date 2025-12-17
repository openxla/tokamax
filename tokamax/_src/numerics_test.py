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
import qwix
from tokamax._src import batching
from tokamax._src import numerics


if jax.__version_info__ >= (0, 6, 3):
  Layout = layout.Layout
else:
  Layout = layout.DeviceLocalLayout  # type: ignore

jax.config.update('jax_threefry_partitionable', False)


class NumericsTest(parameterized.TestCase):

  def test_initializer_consistency(self):
    kwargs = {
        'u': 'test',
        'x': batching.BatchedShapeDtype((4,), jnp.float32, vmap_axes=((0, 3),)),
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
    self.assertTrue(np.isnan(summary_diff.l2_diff))

    summary = numerics.array_numeric_summary(x)

    self.assertEqual(summary.has_inf, False)
    self.assertEqual(summary.has_nan, True)
    self.assertLess(abs(summary.max - 3.4), 0.0001)

  def test_array_diff_summary_no_nan(self):
    x = jnp.array([2.0, 2.0], dtype=jnp.float32)
    y = np.array([-1.0, -2.0], dtype=np.float32)

    summary_diff = numerics.array_diff_summary(x, y)
    self.assertEqual(summary_diff.max_absolute_diff_values, (2.0, -2.0))
    self.assertEqual(summary_diff.max_absolute_diff, 4.0)
    self.assertEqual(summary_diff.l2_diff, 5.0)

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
        'bfloat16': np.float64(-2.4604949951171875),
        'bool': np.float64(23.0),
        'float16': np.float64(-1.642547607421875),
        'float32': np.float64(8.68816682882607),
        'int32': np.float64(50.0),
        'int4': np.float64(-22.0),
        'int64': np.float64(966.0),
        'int8': np.float64(204.0),
        'uint32': np.float64(3145.0),
        'uint4': np.float64(380.0),
        'uint64': np.float64(3317.0),
        'uint8': np.float64(3216.0),
    }

    chex.assert_trees_all_close(kwargs, kwargs_expected)

  @parameterized.parameters(jnp.bool_, jnp.float32, jnp.int32, jnp.uint8)
  def test_random_initialize_layout(self, dtype):
    if jax.default_backend() == 'tpu':
      self.skipTest('Test broken on TPU')

    shape = (2, 3, 4)
    no_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    format_ = layout.Format(Layout((1, 2, 0), ()), no_sharding)
    spec_with_layout = jax.ShapeDtypeStruct(shape, dtype, sharding=format_)
    actual = numerics.random_initialize(spec_with_layout)
    expected = numerics.random_initialize(jax.ShapeDtypeStruct(shape, dtype))
    chex.assert_trees_all_close(actual, expected)
    self.assertEqual(actual.format, format_)

  @parameterized.product(
      qtype=(jnp.int8, jnp.int4),
      scale=(
          jax.ShapeDtypeStruct((1, 128), jnp.bfloat16),
          jax.ShapeDtypeStruct((128, 1), jnp.bfloat16),
          jax.ShapeDtypeStruct((32, 32), jnp.float32),
      ),
  )
  def test_random_initialize_qarray(self, qtype, scale):
    qvalue = jax.ShapeDtypeStruct((256, 256), qtype)
    q = qwix.QArray(qvalue, scale)  # pytype: disable=wrong-arg-types
    q = numerics.random_initialize(q)
    self.assertEqual(q.qvalue.shape, qvalue.shape)
    self.assertEqual(q.scale.shape, scale.shape)
    self.assertEqual(q.qvalue.dtype, qtype)
    self.assertEqual(q.scale.dtype, scale.dtype)
    q_rms = jnp.sqrt(jnp.mean(qwix.dequantize(q) ** 2))
    self.assertGreater(q_rms, 0.8)
    self.assertLess(q_rms, 1.2)

  def test_ranged_array_initializer(self):
    x = numerics.RangedArrayInitializer((128, 128), jnp.int32, 3, 7)
    x = numerics.random_initialize(x)
    self.assertEqual(jnp.min(x), 3)
    self.assertEqual(jnp.max(x), 6)

  def test_seed(self):
    a = batching.BatchedShapeDtype((16,), jnp.int4, vmap_axes=((0, 16),))
    b = jax.ShapeDtypeStruct((16, 16), jnp.int4)

    a_random = numerics.random_initialize(a)
    b_random = numerics.random_initialize(b)
    chex.assert_trees_all_equal(a_random, b_random)

    a_random_2 = numerics.random_initialize(a, seed=8)
    b_random_2 = numerics.random_initialize(b, seed=8)
    chex.assert_trees_all_equal(a_random_2, b_random_2)

    self.assertFalse(jnp.array_equal(a_random, a_random_2))


if __name__ == '__main__':
  absltest.main()
