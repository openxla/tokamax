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
"""Tests for auto-differentiation utilities."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import ad


class AutodiffTest(parameterized.TestCase):

  def test_vjp_taking_no_residuals(self):
    fn = lambda x: (x + x)

    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(rng0, (3, 4, 5))
    dout = jax.random.normal(rng1, x.shape)

    vjp_fn_ref = jax.vjp(fn, x)[1]
    vjp_fn = ad.get_vjp_taking_residuals(lambda x: (fn(x), ()), x)
    chex.assert_trees_all_equal(vjp_fn((), dout), vjp_fn_ref(dout))

  @parameterized.parameters(jax.nn.relu, jax.nn.sigmoid, jax.nn.swish)
  def test_vjp_taking_residuals_from_inputs(self, fn):
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(rng0, (3, 4, 5))
    dout = jax.random.normal(rng1, x.shape)

    # This is not a sensible use-case for `get_vjp_taking_residuals`, but we
    # want to test it still produces a valid VJP function.
    vjp_fn = ad.get_vjp_taking_residuals(lambda x: (fn(x), x), x)
    chex.assert_trees_all_equal(vjp_fn(x, dout), jax.vjp(fn, x)[1](dout))

  @parameterized.parameters(jax.nn.sigmoid, lambda x: (x + x))
  def test_vjp_taking_residuals_from_outputs(self, fn):
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(rng0, (3, 4, 5))
    dout = jax.random.normal(rng1, x.shape)

    y, vjp_fn_ref = jax.vjp(fn, x)
    vjp_fn = ad.get_vjp_taking_residuals(lambda x: (fn(x),) * 2, x)
    chex.assert_trees_all_equal(vjp_fn(y, dout), vjp_fn_ref(dout))

  @parameterized.parameters(jax.nn.swish, lambda x: (x * x))
  def test_vjp_taking_residuals_from_invalid_outputs(self, fn):
    x_shape = jax.ShapeDtypeStruct((3, 4, 5), jnp.float32)
    self.assertIsNone(
        ad.get_vjp_taking_residuals(lambda x: (fn(x),) * 2, x_shape)
    )

  def test_vjp_taking_residuals_from_intermediates(self):
    def fn_with_res(x):
      y = jax.nn.sigmoid(x)
      return y * y, y

    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(rng0, (3, 4, 5))
    dout = jax.random.normal(rng1, x.shape)

    _, vjp_fn_ref, res = jax.vjp(fn_with_res, x, has_aux=True)
    vjp_fn = ad.get_vjp_taking_residuals(fn_with_res, x)
    chex.assert_trees_all_equal(vjp_fn(res, dout), vjp_fn_ref(dout))

  def test_vjp_taking_residuals_with_multiple_args_and_results(self):
    def fn_with_res(x, y):
      z = jax.nn.sigmoid(x)
      return (y * y, y * z), (y, z)

    rng0, rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(0), 4)
    x = jax.random.normal(rng0, (3, 4, 5))
    y = jax.random.normal(rng1, x.shape)
    dout = (jax.random.normal(rng2, x.shape), jax.random.normal(rng3, x.shape))

    _, vjp_fn_ref, res = jax.vjp(fn_with_res, x, y, has_aux=True)
    vjp_fn = ad.get_vjp_taking_residuals(fn_with_res, x, y)
    chex.assert_trees_all_equal(vjp_fn(res, dout), vjp_fn_ref(dout))


if __name__ == "__main__":
  absltest.main()
