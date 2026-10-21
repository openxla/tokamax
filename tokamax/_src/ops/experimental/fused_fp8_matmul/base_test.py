# Copyright 2026 Google LLC
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
"""Tests for the XLA reference implementation of the fused FP8 matmul."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.fused_fp8_matmul import base
from tokamax._src.ops.experimental.fused_fp8_matmul import test_base

jax.config.parse_flags_with_absl()


class BaseFusedFp8MatmulTest(test_base.FusedFp8MatmulTestBase):

  def __init__(self, *args):
    super().__init__(*args, matmul_fn=base.FusedFp8Matmul())


class ReferenceVjpTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    m, k, n = 512, 1024, 2048
    self.lhs, self.rhs = test_base.random_inputs(m, k, n, jnp.bfloat16)
    self.dout = jax.random.normal(jax.random.PRNGKey(3), (m, n), jnp.bfloat16)
    _, self.residuals = base.fused_fp8_matmul_reference(self.lhs, self.rhs)

  def test_custom_vjp_is_kept(self):
    vjp = base.FusedFp8MatmulVjp()
    self.assertIs(base.FusedFp8Matmul(vjp=vjp).vjp, vjp)

  def test_dlhs_reference_close_to_f32_matmul(self):
    _, _, rhs_q, rhs_scale = self.residuals
    dlhs = jax.jit(base.fused_fp8_dlhs_reference)(self.dout, rhs_q, rhs_scale)
    self.assertEqual(
        (dlhs.shape, dlhs.dtype), (self.lhs.shape, self.dout.dtype)
    )
    rhs = rhs_q.astype(jnp.float32) * rhs_scale
    expected = jnp.matmul(
        self.dout.astype(jnp.float32),
        rhs.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    # Only the cotangent is quantized here (the weight is already FP8 in both
    # products), so the gap is one operand's worth of FP8 rounding.
    self.assertLess(
        test_base.relative_error(dlhs, expected), test_base.FP8_NOISE_TOL
    )

  def test_drhs_reference_is_f32_product_of_dequantized_lhs(self):
    lhs_q, lhs_scale, _, _ = self.residuals
    drhs = jax.jit(base.fused_fp8_drhs_reference)(lhs_q, lhs_scale, self.dout)
    self.assertEqual(
        (drhs.shape, drhs.dtype), (self.rhs.shape, self.dout.dtype)
    )
    expected = jnp.matmul(
        (lhs_q.astype(jnp.float32) * lhs_scale).T,
        self.dout.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    # Nothing is quantized in this product; only bf16 rounding differs.
    self.assertLess(
        test_base.relative_error(drhs, expected), test_base.BF16_ROUNDING_TOL
    )

  def test_op_vjp_matches_the_references(self):
    matmul = base.FusedFp8Matmul()

    def op_grads(lhs, rhs, dout):
      _, vjp_fn = jax.vjp(matmul, lhs, rhs)
      return vjp_fn(dout)

    def reference_grads(lhs, rhs, dout):
      _, residuals = base.fused_fp8_matmul_reference(lhs, rhs)
      lhs_q, lhs_scale, rhs_q, rhs_scale = residuals
      return (
          base.fused_fp8_dlhs_reference(dout, rhs_q, rhs_scale),
          base.fused_fp8_drhs_reference(lhs_q, lhs_scale, dout),
      )

    dlhs, drhs = jax.jit(op_grads)(self.lhs, self.rhs, self.dout)
    dlhs_ref, drhs_ref = jax.jit(reference_grads)(self.lhs, self.rhs, self.dout)
    self.assertEqual((dlhs.shape, dlhs.dtype), (self.lhs.shape, self.lhs.dtype))
    self.assertEqual((drhs.shape, drhs.dtype), (self.rhs.shape, self.rhs.dtype))
    # Same quantization on both sides; both jitted so only XLA fusion choices
    # can differ (see `test_base.SAME_QUANTIZATION_TOL`).
    self.assertLess(
        test_base.relative_error(dlhs, dlhs_ref),
        test_base.SAME_QUANTIZATION_TOL,
    )
    self.assertLess(
        test_base.relative_error(drhs, drhs_ref),
        test_base.SAME_QUANTIZATION_TOL,
    )


class QuantizationTest(absltest.TestCase):

  def test_quantize_rows_fp8(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 256), jnp.bfloat16)
    xq, scale = base.quantize_rows_fp8(x)
    self.assertEqual((xq.shape, xq.dtype), (x.shape, jnp.float8_e4m3fn))
    self.assertEqual((scale.shape, scale.dtype), ((64, 1), jnp.float32))
    # Every row's absmax lands on the FP8 max (up to bf16 rounding of the
    # scale, i.e. within one FP8 ulp), and nothing overflows to NaN.
    xq_f32 = np.asarray(xq.astype(jnp.float32))
    self.assertTrue(np.all(np.isfinite(xq_f32)))
    np.testing.assert_allclose(
        np.abs(xq_f32).max(axis=1), base.FP8_MAX, rtol=0.08
    )
    np.testing.assert_allclose(
        xq_f32 * np.asarray(scale),
        np.asarray(x, np.float32),
        rtol=test_base.FP8_RECONSTRUCTION_RTOL,
    )

  def test_quantize_cols_fp8(self):
    w = jax.random.normal(jax.random.PRNGKey(0), (256, 64), jnp.bfloat16)
    wq, scale = base.quantize_cols_fp8(w)
    self.assertEqual((wq.shape, wq.dtype), (w.shape, jnp.float8_e4m3fn))
    self.assertEqual((scale.shape, scale.dtype), ((64,), jnp.float32))
    wq_f32 = np.asarray(wq.astype(jnp.float32))
    self.assertTrue(np.all(np.isfinite(wq_f32)))
    np.testing.assert_allclose(
        np.abs(wq_f32).max(axis=0), base.FP8_MAX, rtol=0.08
    )
    np.testing.assert_allclose(
        wq_f32 * np.asarray(scale),
        np.asarray(w, np.float32),
        rtol=test_base.FP8_RECONSTRUCTION_RTOL,
    )

  def test_zero_row_does_not_divide_by_zero(self):
    x = jnp.zeros((8, 128), jnp.bfloat16)
    xq, scale = base.quantize_rows_fp8(x)
    self.assertTrue(bool(jnp.all(jnp.isfinite(scale))))
    np.testing.assert_array_equal(np.asarray(xq.astype(jnp.float32)), 0.0)


if __name__ == "__main__":
  absltest.main()
