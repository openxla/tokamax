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


def _reference_triangle_multiplication(
    x,
    mask,
    projection_in_weights,
    gate_in_weights,
    projection_out_weights,
    gate_out_weights,
    layernorm_in_scale,
    layernorm_in_offset,
    layernorm_out_scale,
    layernorm_out_offset,
    triangle_type,
    epsilon=1e-6,
):
  """Pure textbook AlphaFold reference in FP32 for numerical verification."""
  prec = jax.lax.Precision.HIGHEST
  x_f32 = x.astype(jnp.float32)
  mean_in = jnp.mean(x_f32, axis=-1, keepdims=True)
  var_in = jnp.mean(jnp.square(x_f32 - mean_in), axis=-1, keepdims=True)
  left_act = (
      (x_f32 - mean_in) * jax.lax.rsqrt(var_in + epsilon) * layernorm_in_scale
      + layernorm_in_offset
  )
  masked = jnp.where(mask[:, :, None], left_act, 0.0)
  proj = jnp.einsum(
      'ijc,ckh->kijh', masked, projection_in_weights, precision=prec
  )
  gate = jax.nn.sigmoid(
      jnp.einsum('ijc,ckh->kijh', masked, gate_in_weights, precision=prec)
  )
  ab = proj * gate
  if triangle_type == 'outgoing':
    act = jnp.einsum('ikc,jkc->ijc', ab[0], ab[1], precision=prec)
  else:
    act = jnp.einsum('kjc,kic->ijc', ab[0], ab[1], precision=prec)
  mean_out = jnp.mean(act, axis=-1, keepdims=True)
  var_out = jnp.mean(jnp.square(act - mean_out), axis=-1, keepdims=True)
  act_norm = (
      (act - mean_out)
      * jax.lax.rsqrt(var_out + epsilon)
      * layernorm_out_scale
      + layernorm_out_offset
  )
  out_proj = jnp.einsum(
      'ijh,hd->ijd', act_norm, projection_out_weights, precision=prec
  )
  out_gate = jax.nn.sigmoid(
      jnp.einsum('ijc,cd->ijd', left_act, gate_out_weights, precision=prec)
  )
  return out_proj * out_gate


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
    for res in residuals:
      self.assertEqual(res.shape, (h, n, n))
      self.assertEqual(res.dtype, dtype)

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

  @parameterized.parameters('incoming', 'outgoing')
  def test_large_mean_shift_stability(self, triangle_type):
    params = _get_params(n=16, c=16, h=32, d=64, dtype=jnp.float32)
    params['layernorm_in_offset'] = params['layernorm_in_offset'] + 50.0
    params['projection_in_weights'] = (
        jnp.abs(params['projection_in_weights']) + 1.0
    )
    mask = params.pop('mask')
    dout = jax.random.normal(jax.random.key(7), (16, 16, 64), dtype=jnp.float32)

    op_default = base.TriangleMultiplication()
    op_vjp = base.TriangleMultiplication(vjp=base.TriangleMultiplicationVjp())

    def loss_op(p, fn):
      out = fn(
          **p,
          mask=mask,
          triangle_type=triangle_type,
          precision=jax.lax.Precision.HIGHEST,
      )
      return jnp.sum(out * dout)

    def loss_ref(p):
      out = _reference_triangle_multiplication(
          **p, mask=mask, triangle_type=triangle_type
      )
      return jnp.sum(out * dout)

    out_op = jax.jit(
        lambda p: op_default(
            **p,
            mask=mask,
            triangle_type=triangle_type,
            precision=jax.lax.Precision.HIGHEST,
        )
    )(params)
    out_ref = jax.jit(
        lambda p: _reference_triangle_multiplication(
            **p, mask=mask, triangle_type=triangle_type
        )
    )(params)
    chex.assert_trees_all_close(out_op, out_ref, atol=1e-3, rtol=1e-4)

    grads_default = jax.jit(jax.grad(lambda p: loss_op(p, op_default)))(params)
    grads_vjp = jax.jit(jax.grad(lambda p: loss_op(p, op_vjp)))(params)
    grads_ref = jax.jit(jax.grad(loss_ref))(params)
    chex.assert_trees_all_close(grads_default, grads_ref, atol=1e-2, rtol=1e-3)
    chex.assert_trees_all_close(grads_vjp, grads_ref, atol=1e-2, rtol=1e-3)

  @parameterized.parameters(jnp.nan, jnp.inf, -jnp.inf)
  def test_nonfinite_in_masked_tokens_does_not_propagate(self, bad_val):
    n = 16
    valid = jnp.arange(n) < 12
    mask = valid[:, None] & valid[None, :]
    params = _get_params(n=n, c=16, h=32, d=64, dtype=jnp.bfloat16)
    params.pop('mask')
    params_bad = dict(params)
    params_bad['x'] = jnp.where(mask[:, :, None], params['x'], bad_val)
    dout = jnp.where(
        mask[:, :, None],
        jax.random.normal(jax.random.key(3), (n, n, 64), dtype=jnp.bfloat16),
        0.0,
    )
    for fn in (
        base.TriangleMultiplication(),
        base.TriangleMultiplication(vjp=base.TriangleMultiplicationVjp()),
    ):
      out_clean, vjp_clean = jax.jit(
          lambda p, f=fn: jax.vjp(
              lambda q: f(**q, mask=mask, triangle_type='outgoing'), p
          )
      )(params)
      out_bad, vjp_bad = jax.jit(
          lambda p, f=fn: jax.vjp(
              lambda q: f(**q, mask=mask, triangle_type='outgoing'), p
          )
      )(params_bad)
      chex.assert_trees_all_close(out_bad[:12, :12], out_clean[:12, :12])
      (grads_clean,) = jax.jit(vjp_clean)(dout)
      (grads_bad,) = jax.jit(vjp_bad)(dout)
      for k in grads_bad:
        self.assertTrue(bool(jnp.all(jnp.isfinite(grads_bad[k]))))
        chex.assert_trees_all_close(grads_bad[k], grads_clean[k])

  def test_zero_epsilon_no_nan(self):
    n = 12
    params = _get_params(n=n, c=16, h=32, d=64, dtype=jnp.bfloat16)
    mask = params.pop('mask').at[8:, :].set(False)
    params['x'] = params['x'].at[8:, :, :].set(0.0)
    dout = jax.random.normal(jax.random.key(0), (n, n, 64), dtype=jnp.bfloat16)
    for fn in (
        base.TriangleMultiplication(),
        base.TriangleMultiplication(vjp=base.TriangleMultiplicationVjp()),
    ):
      out, vjp_fn = jax.jit(
          lambda p, f=fn: jax.vjp(
              lambda q: f(
                  **q, mask=mask, triangle_type='outgoing', epsilon=0.0
              ),
              p,
          )
      )(params)
      (grads,) = jax.jit(vjp_fn)(dout)
      self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
      for g in jax.tree.leaves(grads):
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))


if __name__ == '__main__':
  absltest.main()
