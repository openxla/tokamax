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
"""Tests for scaled arrays."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import quantization


class QuantizedArrayTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(shape=(42, 43), tile_shape=(1, -1), scales_shape=(42, 1)),
      dict(shape=(42, 43), tile_shape=(-1, 1), scales_shape=(1, 43)),
      dict(shape=(42, 43), tile_shape=(-1, -1), scales_shape=(1, 1)),
      dict(shape=(7, 11, 13), tile_shape=(-1, 1, 1), scales_shape=(1, 11, 13)),
      dict(shape=(7, 11, 13), tile_shape=(-1, -1, -1), scales_shape=(1, 1, 1)),
      dict(shape=(7, 11, 13), tile_shape=None, scales_shape=(1, 1, 1)),
      dict(shape=(4, 9), tile_shape=(1, 3), scales_shape=(4, 3)),
      dict(shape=(4, 9), tile_shape=(2, 1), scales_shape=(2, 9)),
      dict(shape=(4, 9), tile_shape=(2, 3), scales_shape=(2, 3)),
  )
  def test_shape_dtype(self, shape, tile_shape, scales_shape):
    x = jax.random.uniform(jax.random.PRNGKey(42), shape)
    q = quantization.quantize_as(jnp.int8, tile_shape=tile_shape)(x)

    self.assertEqual(q.shape, x.shape)
    self.assertEqual(q.dtype, x.dtype)
    self.assertEqual(q.values.shape, x.shape)
    self.assertEqual(q.values.dtype, jnp.int8)
    self.assertEqual(q.scales.shape, scales_shape)
    self.assertEqual(q.scales.dtype, x.dtype)

  @parameterized.parameters(
      dict(shape=(7, 11, 13), tile_shape=(-1, 1, 1)),
      dict(shape=(7, 11, 13), tile_shape=(1, -1, 1)),
      dict(shape=(7, 11, 13), tile_shape=(1, 1, -1)),
      dict(shape=(7, 11, 13), tile_shape=None),
      dict(shape=(4, 9), tile_shape=(2, 3)),
      dict(shape=(16, 32), tile_shape=(-1, 1), dtype=jnp.int4),
      dict(shape=(16, 32), tile_shape=(1, -1), dtype=jnp.int4),
      dict(shape=(16, 32), tile_shape=(4, 2), dtype=jnp.int4),
  )
  def test_roundtrip(self, shape, tile_shape, dtype=jnp.int8):
    x = jax.random.uniform(jax.random.PRNGKey(42), shape)
    q = quantization.quantize_as(dtype, tile_shape=tile_shape)(x)
    atol = 1.0 / jnp.iinfo(dtype).max
    chex.assert_trees_all_close(q.recompose(), x, atol=atol)

  def test_can_jit_quantize_fn(self):
    x = jax.random.uniform(jax.random.PRNGKey(42), (7, 11, 13))
    q = jax.jit(quantization.quantize_as(jnp.int8, tile_shape=(1, 1, -1)))(x)
    chex.assert_trees_all_close(q.recompose(), x, atol=(1.0 / 127))


if __name__ == '__main__':
  absltest.main()
