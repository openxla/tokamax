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
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.ops.triangle_multiplication import base


def _get_params(n, c, h, d, dtype):
  params = dict(
      x=jax.ShapeDtypeStruct((n, n, c), dtype=dtype),
      mask=jax.ShapeDtypeStruct((n, n), dtype=jnp.bool),
      projection_in_weights=jax.ShapeDtypeStruct((c, 2, h), dtype=dtype),
      gate_in_weights=jax.ShapeDtypeStruct((c, 2, h), dtype=dtype),
      projection_out_weights=jax.ShapeDtypeStruct((h, d), dtype=dtype),
      gate_out_weights=jax.ShapeDtypeStruct((c, d), dtype=dtype),
      layernorm_in_scale=jax.ShapeDtypeStruct((c,), dtype=dtype),
      layernorm_in_offset=jax.ShapeDtypeStruct((c,), dtype=dtype),
      layernorm_out_scale=jax.ShapeDtypeStruct((h,), dtype=dtype),
      layernorm_out_offset=jax.ShapeDtypeStruct((h,), dtype=dtype),
  )
  return numerics.random_initialize(params)


class TriangleMultiplicationTest(parameterized.TestCase):

  @parameterized.product(
      triangle_type=['incoming', 'outgoing'],
      dtype=[jnp.bfloat16, jnp.float32],
      precision=[jax.lax.Precision.HIGH],
  )
  def test_triangle_multiplication(self, triangle_type, dtype, precision):
    n = 8
    c = 16
    h = 32
    d = 64
    params = _get_params(n=n, c=c, h=h, d=d, dtype=dtype)

    @jax.jit
    def f(params):
      return base.TriangleMultiplication()(
          **params,
          triangle_type=triangle_type,
          precision=precision,
          return_residuals=True,
      )

    out, residuals = f(params)
    self.assertEqual(out.shape, (n, n, d))
    self.assertEqual(out.dtype, dtype)
    self.assertLen(residuals, 3)
    for res in residuals[:2]:
      self.assertEqual(res.shape, (h, n, n))
      self.assertEqual(res.dtype, dtype)
    self.assertEqual(residuals[2].shape, (n, n, h))
    self.assertEqual(residuals[2].dtype, dtype)

  @parameterized.product(
      triangle_type=['incoming', 'outgoing'],
      dtype=[jnp.bfloat16, jnp.float32],
      precision=[jax.lax.Precision.HIGHEST],
  )
  def test_triangle_multiplication_grad(self, triangle_type, dtype, precision):
    n = 8
    d = 64
    params = _get_params(n=n, c=16, h=32, d=d, dtype=dtype)
    mask = params.pop('mask')
    op_default = base.TriangleMultiplication()
    op_vjp = base.TriangleMultiplication(vjp=base.TriangleMultiplicationVjp())

    def loss(p, fn):
      y = fn(
          **p,
          mask=mask,
          triangle_type=triangle_type,
          precision=precision,
      )
      return jnp.sum(y)

    grads_default = jax.jit(jax.grad(lambda p: loss(p, op_default)))(params)
    grads_vjp = jax.jit(jax.grad(lambda p: loss(p, op_vjp)))(params)
    for k in params:
      self.assertFalse(jnp.isnan(grads_default[k]).any())
    chex.assert_trees_all_close(grads_vjp, grads_default, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
