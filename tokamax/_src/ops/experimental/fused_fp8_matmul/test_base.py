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
"""Shared correctness tests for fused FP8 matmul implementations."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import qwix
from tokamax._src.ops.experimental.fused_fp8_matmul import base

# (m, k, n) shapes. The first four are the projection shapes of the training
# workload this kernel was built for, at reduced `m`; the rest exercise large
# `k` (where a kernel may split the contraction), a block size that is not a
# power of two, and small shapes.
SHAPES = (
    (2048, 3072, 4096),
    (2048, 3072, 12288),
    (2048, 12288, 3072),
    (2048, 3072, 3072),
    (512, 256, 256),
    (1024, 1024, 384),
    (256, 6144, 512),
)

# Above this `k`, implementations are allowed to quantize `lhs` per k block
# rather than per full row, so they only agree with the reference up to FP8
# quantization noise rather than up to rounding of the final cast.
_SINGLE_ROW_SCALE_MAX_K = 4096

# Relative error (in norm) between two results computed with the *same* FP8
# quantization algorithm by different compilers. XLA on TPU may keep the bf16
# scaling multiply in f32 and round once to FP8, while an eager computation or
# a Pallas kernel rounds to bf16 first; elements that straddle an FP8 rounding
# boundary then differ by one FP8 ulp. Measured at ~1.3% on TPU7x but 2.4% on
# TPU v6e (no FP8 MXU, so XLA fuses the dot differently), which is why the
# suite compares jitted results with jitted results. 2% is tight enough to
# catch a wrong scale or tile index (tens of percent) without flaking.
SAME_QUANTIZATION_TOL = 2e-2

# Relative error (in norm) between two results whose operands were quantized
# to FP8 by different but equivalent methods, e.g. a different scale
# granularity. An e4m3 value has a 3-bit mantissa, so rounding gives each
# quantized operand ~2.5% RMS relative error; two independent quantizations of
# the same operand therefore differ by ~3.5%. 5% leaves margin without hiding
# a real bug (a wrong scale shows up as tens of percent).
FP8_NOISE_TOL = 5e-2

# Elementwise tolerance for reconstructing an operand from its FP8 value and
# scale: half an e4m3 ulp is 6.25% at the bottom of a binade, plus a little
# from rounding the scale to bf16 inside the quantizer.
FP8_RECONSTRUCTION_RTOL = 0.1


def random_inputs(m, k, n, dtype, seed=0):
  k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
  lhs = jax.random.normal(k1, (m, k), jnp.float32).astype(dtype)
  rhs = (jax.random.normal(k2, (k, n), jnp.float32) * 0.02).astype(dtype)
  return lhs, rhs


def _f32_matmul(lhs, rhs):
  return jnp.matmul(
      lhs.astype(jnp.float32),
      rhs.astype(jnp.float32),
      precision=jax.lax.Precision.HIGHEST,
  )


def relative_error(actual, expected) -> float:
  actual = np.asarray(actual, np.float32)
  expected = np.asarray(expected, np.float32)
  return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


class FusedFp8MatmulTestBase(parameterized.TestCase):
  """Correctness suite shared by all fused FP8 matmul implementations.

  Subclasses pass the op under test as `matmul_fn`. The op is compared against
  the XLA reference (`base.FusedFp8Matmul`) tightly, and against an exact f32
  matmul loosely (FP8 quantization error).
  """

  def __init__(self, *args, matmul_fn):
    super().__init__(*args)
    self._matmul_fn = matmul_fn

  def _matmul(self, lhs, rhs, **kwargs):
    return jax.jit(lambda a, b: self._matmul_fn(a, b, **kwargs))(lhs, rhs)

  def assert_close_to_reference(self, actual, expected, k):
    # With one scale per row both sides quantize the same way; above
    # `_SINGLE_ROW_SCALE_MAX_K` a kernel may quantize per k block instead.
    tol = (
        SAME_QUANTIZATION_TOL if k <= _SINGLE_ROW_SCALE_MAX_K else FP8_NOISE_TOL
    )
    self.assertLess(relative_error(actual, expected), tol)

  @parameterized.product(shape=SHAPES, dtype=(jnp.bfloat16, jnp.float32))
  def test_matches_reference(self, shape, dtype):
    m, k, n = shape
    lhs, rhs = random_inputs(m, k, n, dtype)
    actual = self._matmul(lhs, rhs)
    expected = jax.jit(base.FusedFp8Matmul())(lhs, rhs)
    self.assertEqual(actual.shape, expected.shape)
    self.assertEqual(actual.dtype, dtype)
    self.assert_close_to_reference(actual, expected, k=k)

  @parameterized.parameters(*SHAPES)
  def test_close_to_f32_matmul(self, m, k, n):
    lhs, rhs = random_inputs(m, k, n, jnp.bfloat16)
    actual = self._matmul(lhs, rhs)
    # Both operands are quantized (see `FP8_NOISE_TOL`), so the result lands
    # ~3.5% from the exact matmul.
    self.assertLess(relative_error(actual, _f32_matmul(lhs, rhs)), 0.05)

  def test_qarray_rhs_matches_plain_rhs(self):
    """A qwix-quantized rhs gives the same result as quantizing in-op."""
    m, k, n = 1024, 3072, 4096
    lhs, rhs = random_inputs(m, k, n, jnp.bfloat16)
    rhs_q = qwix.quantize(
        rhs, jnp.float8_e4m3fn, channelwise_axes=(1,), scale_dtype=jnp.float32
    )
    self.assertEqual(rhs_q.scale.shape, (1, n))
    actual = self._matmul(lhs, rhs_q)
    # qwix and the op both use absmax / 448 per column, but qwix divides by
    # the scale in f32 while the op scales in bf16, so some elements round to
    # a neighbouring FP8 value (measured: ~1.2%).
    expected = self._matmul(lhs, rhs)
    self.assertLess(relative_error(actual, expected), FP8_NOISE_TOL)

  def test_residuals(self):
    """Residuals have the documented layout and reproduce the operands."""
    m, k, n = 1024, 3072, 4096
    lhs, rhs = random_inputs(m, k, n, jnp.bfloat16)
    out, residuals = self._matmul(lhs, rhs, return_residuals=True)
    lhs_q, lhs_scale, rhs_q, rhs_scale = residuals
    self.assertEqual(out.shape, (m, n))
    self.assertEqual((lhs_q.shape, lhs_q.dtype), ((m, k), jnp.float8_e4m3fn))
    self.assertEqual((lhs_scale.shape, lhs_scale.dtype), ((m, 1), jnp.float32))
    self.assertEqual((rhs_q.shape, rhs_q.dtype), ((k, n), jnp.float8_e4m3fn))
    self.assertEqual((rhs_scale.shape, rhs_scale.dtype), ((n,), jnp.float32))
    # The residuals must reproduce the operands up to FP8 rounding, and the
    # output.
    np.testing.assert_allclose(
        np.asarray(lhs_q.astype(jnp.float32) * lhs_scale),
        np.asarray(lhs, np.float32),
        rtol=FP8_RECONSTRUCTION_RTOL,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(rhs_q.astype(jnp.float32) * rhs_scale),
        np.asarray(rhs, np.float32),
        rtol=FP8_RECONSTRUCTION_RTOL,
        atol=1e-5,
    )
    # Returning residuals must not change the output. Both calls go through
    # the same compiler, so this holds on every chip (measured bit-identical on
    # CPU, TPU v6e and TPU7x); agreement with the reference is covered by
    # `test_matches_reference`.
    self.assert_close_to_reference(out, self._matmul(lhs, rhs), k=k)

  def test_grad_matches_reference(self):
    """Gradients w.r.t. both operands match the reference VJP."""
    m, k, n = 512, 1024, 2048
    lhs, rhs = random_inputs(m, k, n, jnp.bfloat16)
    dout = jax.random.normal(jax.random.PRNGKey(3), (m, n), jnp.bfloat16)

    def loss(fn, a, b):
      return jnp.sum(fn(a, b).astype(jnp.float32) * dout.astype(jnp.float32))

    actual = jax.jit(jax.grad(lambda a, b: loss(self._matmul_fn, a, b), (0, 1)))
    expected = jax.jit(
        jax.grad(lambda a, b: loss(base.FusedFp8Matmul(), a, b), (0, 1))
    )
    dlhs, drhs = actual(lhs, rhs)
    dlhs_ref, drhs_ref = expected(lhs, rhs)
    self.assertEqual((dlhs.shape, dlhs.dtype), (lhs.shape, lhs.dtype))
    self.assertEqual((drhs.shape, drhs.dtype), (rhs.shape, rhs.dtype))
    self.assert_close_to_reference(dlhs, dlhs_ref, k=k)
    self.assert_close_to_reference(drhs, drhs_ref, k=k)

  def test_grad_with_qarray_rhs(self):
    """Differentiating w.r.t. lhs works when rhs is a pre-quantized QArray."""
    m, k, n = 512, 1024, 1024
    lhs, rhs = random_inputs(m, k, n, jnp.bfloat16)
    rhs_q = qwix.quantize(
        rhs, jnp.float8_e4m3fn, channelwise_axes=(1,), scale_dtype=jnp.float32
    )

    def loss(a, b):
      return jnp.sum(self._matmul_fn(a, b).astype(jnp.float32))

    dlhs = jax.jit(jax.grad(loss))(lhs, rhs_q)
    dlhs_ref = jax.jit(jax.grad(loss))(lhs, rhs)
    self.assertEqual((dlhs.shape, dlhs.dtype), (lhs.shape, lhs.dtype))
    # `rhs` was quantized by qwix in one case and by the op in the other.
    self.assertLess(relative_error(dlhs, dlhs_ref), FP8_NOISE_TOL)

  def test_rejects_bad_inputs(self):
    """Shape and dtype errors are raised at bind time with clear messages."""
    lhs = jnp.zeros((256, 512), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "Contracting dims differ"):
      self._matmul_fn(lhs, jnp.zeros((256, 512), jnp.bfloat16))
    with self.assertRaisesRegex(ValueError, "2-D"):
      self._matmul_fn(lhs[None], jnp.zeros((512, 256), jnp.bfloat16))
    with self.assertRaisesRegex(ValueError, "qwix.QArray"):
      self._matmul_fn(lhs, jnp.zeros((512, 256), jnp.float8_e4m3fn))
    with self.assertRaisesRegex(ValueError, "one scale per output column"):
      per_row = qwix.quantize(
          jnp.ones((512, 256), jnp.bfloat16),
          jnp.float8_e4m3fn,
          channelwise_axes=(0,),
      )
      self._matmul_fn(lhs, per_row)
