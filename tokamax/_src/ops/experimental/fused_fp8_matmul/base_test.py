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
