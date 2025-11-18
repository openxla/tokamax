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
    d = 64
    params = _get_params(n=n, c=16, h=32, d=d, dtype=dtype)

    @jax.jit
    def f(params):
      y = base.TriangleMultiplication()(
          **params, triangle_type=triangle_type, precision=precision
      )
      return y

    out = f(params)
    self.assertEqual(out.shape, (n, n, d))
    self.assertEqual(out.dtype, dtype)

  @parameterized.product(
      triangle_type=['incoming', 'outgoing'],
      dtype=[jnp.bfloat16],
      precision=[jax.lax.Precision.HIGH],
  )
  def test_triangle_multiplication_grad(self, triangle_type, dtype, precision):
    n = 8
    d = 64
    params = _get_params(n=n, c=16, h=32, d=d, dtype=dtype)

    def f(x):
      y = base.TriangleMultiplication()(
          x=x,
          mask=params['mask'],
          projection_in_weights=params['projection_in_weights'],
          gate_in_weights=params['gate_in_weights'],
          projection_out_weights=params['projection_out_weights'],
          gate_out_weights=params['gate_out_weights'],
          layernorm_in_scale=params['layernorm_in_scale'],
          layernorm_in_offset=params['layernorm_in_offset'],
          layernorm_out_scale=params['layernorm_out_scale'],
          layernorm_out_offset=params['layernorm_out_offset'],
          triangle_type=triangle_type,
          precision=precision,
      )
      return jnp.sum(y)

    grad_fn = jax.grad(f)
    grads = grad_fn(params['x'])
    self.assertFalse(jnp.isnan(grads).any())


if __name__ == '__main__':
  absltest.main()
