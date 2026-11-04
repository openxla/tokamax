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
from tokamax._src import gpu_utils
from tokamax._src import hlo_utils
from tokamax._src import numerics
from tokamax._src.ops.gated_linear_unit import api as glu_api
from tokamax._src.ops.normalization import api as norm_api
from tokamax._src.ops.triangle_multiplication import base


def _has_triton_sub_ops() -> bool:
  return (
      'triton' in glu_api.IMPLEMENTATIONS
      and 'triton' in norm_api.IMPLEMENTATIONS
      and gpu_utils.has_triton_support()
  )


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
      implementation=[None, 'xla', 'triton', ('triton', 'xla')],
  )
  def test_triangle_multiplication(
      self, triangle_type, dtype, precision, implementation
  ):
    uses_triton = implementation in ('triton', ('triton', 'xla'))
    if uses_triton and not _has_triton_sub_ops():
      self.skipTest('Triton not supported on this platform.')
    n = 8
    c = 16
    h = 32
    d = 64
    params = _get_params(n=n, c=c, h=h, d=d, dtype=dtype)

    @jax.jit
    def f_no_res(params):
      return base.TriangleMultiplication()(
          **params,
          triangle_type=triangle_type,
          precision=precision,
          return_residuals=False,
          implementation=implementation,
      )

    @jax.jit
    def f(params):
      return base.TriangleMultiplication()(
          **params,
          triangle_type=triangle_type,
          precision=precision,
          return_residuals=True,
          implementation=implementation,
      )

    out_no_res = f_no_res(params)
    out, residuals = f(params)
    non_xla_kernels = hlo_utils.get_kernel_info(
        f_no_res.lower(params), include_xla_kernels=False
    )
    if uses_triton:
      self.assertNotEmpty(non_xla_kernels)
      tol = 1e-2 if dtype == jnp.bfloat16 else 1e-5
      chex.assert_trees_all_close(out, out_no_res, atol=tol, rtol=tol)
    else:
      self.assertEmpty(non_xla_kernels)
      chex.assert_trees_all_equal(out, out_no_res)
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
      implementation=[None, 'xla', 'triton', ('triton', 'xla')],
  )
  def test_triangle_multiplication_grad(
      self, triangle_type, dtype, precision, implementation
  ):
    uses_triton = implementation in ('triton', ('triton', 'xla'))
    if uses_triton and not _has_triton_sub_ops():
      self.skipTest('Triton not supported on this platform.')
    n = 8
    d = 64
    params = _get_params(n=n, c=16, h=32, d=d, dtype=dtype)
    mask = params.pop('mask')
    op_default = base.TriangleMultiplication()
    op_vjp = base.TriangleMultiplication(vjp=base.TriangleMultiplicationVjp())

    def apply(p, fn):
      return fn(
          **p,
          mask=mask,
          triangle_type=triangle_type,
          precision=precision,
          implementation=implementation,
      )

    grads_default = jax.jit(jax.grad(lambda p: jnp.sum(apply(p, op_default))))(
        params
    )
    grads_vjp = jax.jit(jax.grad(lambda p: jnp.sum(apply(p, op_vjp))))(params)
    _, vjp_fn = jax.vjp(lambda p: apply(p, op_vjp), params)
    dout = jnp.ones((n, n, d), dtype=dtype)
    self.assertEmpty(
        hlo_utils.get_kernel_info(
            jax.jit(vjp_fn).lower(dout), include_xla_kernels=False
        )
    )
    for k in params:
      self.assertFalse(jnp.isnan(grads_default[k]).any())
      self.assertFalse(jnp.isnan(grads_vjp[k]).any())
    if uses_triton:
      atol = 0.5 if dtype == jnp.bfloat16 else 5e-5
      rtol = 0.05 if dtype == jnp.bfloat16 else 5e-5
      chex.assert_trees_all_close(
          grads_vjp, grads_default, atol=atol, rtol=rtol
      )
    else:
      chex.assert_trees_all_close(
          grads_vjp, grads_default, atol=1e-5, rtol=1e-5
      )


if __name__ == '__main__':
  absltest.main()
