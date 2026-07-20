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
"""Tests for jax-triton normalization op."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from tokamax._src.ops.normalization import jax_triton
from tokamax._src.ops.normalization import pallas_triton
from tokamax._src.ops.normalization import test_base


class JaxTritonNormalizationTest(test_base.NormalizationTestBase):
  """Runs the standard normalization test suite against the jax-triton impl."""

  def __init__(self, *args):
    super().__init__(*args, norm_fn=jax_triton.JaxTritonNormalization())

  def setUp(self):
    if jax.default_backend() == 'tpu':
      self.skipTest('Not supported on TPUs.')
    super().setUp()


class JaxTritonVsPallasTritonTest(parameterized.TestCase):
  """Cross-checks that jax-triton and pallas-triton produce the same results."""

  def setUp(self):
    if jax.default_backend() == 'tpu':
      self.skipTest('Not supported on TPUs.')
    super().setUp()
    self._jt = jax_triton.JaxTritonNormalization(input_output_alias=False)
    self._pt = pallas_triton.PallasTritonNormalization(input_output_alias=False)

  @parameterized.parameters(
      dict(shape=(128, 64), axis=-1, subtract_mean=True),
      dict(shape=(128, 64), axis=-1, subtract_mean=False),
      dict(shape=(8, 128, 32), axis=-1, subtract_mean=True),
      dict(shape=(24, 32, 40), axis=1, subtract_mean=True),
      dict(shape=(256, 42), axis=-1, subtract_mean=True),
  )
  def test_fwd_matches(self, shape, axis, subtract_mean):
    rngs = list(jax.random.split(jax.random.PRNGKey(42), 3))
    x = jax.random.normal(rngs.pop(), shape)
    scale = jax.random.uniform(rngs.pop(), (shape[axis],))
    offset = jax.random.uniform(rngs.pop(), (shape[axis],))

    kwargs = dict(axis=axis, epsilon=1e-6, subtract_mean=subtract_mean)
    y_pt = self._pt(x, scale, offset, **kwargs)
    y_jt = self._jt(x, scale, offset, **kwargs)
    chex.assert_trees_all_close(y_jt, y_pt, atol=1e-5)

  @parameterized.parameters(
      dict(shape=(128, 64), axis=-1, subtract_mean=True),
      dict(shape=(128, 64), axis=-1, subtract_mean=False),
      dict(shape=(24, 32, 40), axis=1, subtract_mean=True),
  )
  def test_vjp_matches(self, shape, axis, subtract_mean):
    rngs = list(jax.random.split(jax.random.PRNGKey(42), 4))
    x = jax.random.normal(rngs.pop(), shape)
    scale = jax.random.uniform(rngs.pop(), (shape[axis],))
    offset = jax.random.uniform(rngs.pop(), (shape[axis],))
    dy = jax.random.normal(rngs.pop(), shape)

    kwargs = dict(axis=axis, epsilon=1e-6, subtract_mean=subtract_mean)
    f_pt = functools.partial(self._pt, **kwargs)
    f_jt = functools.partial(self._jt, **kwargs)

    _, vjp_pt = jax.vjp(f_pt, x, scale, offset)
    _, vjp_jt = jax.vjp(f_jt, x, scale, offset)

    grads_pt = vjp_pt(dy)
    grads_jt = vjp_jt(dy)
    chex.assert_trees_all_close(grads_jt, grads_pt, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
