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
"""Tests for the Pallas/Mosaic TPU fused FP8 dLHS kernel."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_fp8_matmul import base
from tokamax._src.ops.experimental.fused_fp8_matmul import pallas_mosaic_tpu_dlhs_kernel as kernel
from tokamax._src.ops.experimental.fused_fp8_matmul import test_base

jax.config.parse_flags_with_absl()


def _inputs(m, k, n, dtype=jnp.bfloat16):
  """Returns `(dout, rhs_q, rhs_scale)` as the backward pass receives them."""
  _, rhs = test_base.random_inputs(m, k, n, dtype)
  dout = jax.random.normal(jax.random.PRNGKey(3), (m, n), jnp.float32)
  rhs_q, rhs_scale = base.rhs_as_fp8(rhs)
  return dout.astype(dtype), rhs_q, rhs_scale


class DlhsKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    fp8 = jnp.float8_e4m3fn
    if jax.default_backend() != "tpu" or not (
        pltpu.get_tpu_info().is_matmul_supported(fp8, fp8)
    ):
      self.skipTest("FP8 matmul is not supported on this device.")

  def _dlhs(self, dout, rhs_q, rhs_scale, **tiles):
    fn = functools.partial(kernel.fused_fp8_dlhs_pallas, **tiles)
    return jax.jit(fn)(dout, rhs_q, rhs_scale)

  @parameterized.named_parameters(
      dict(
          testcase_name="full_k_full_n_no_scratch",
          shape=(1024, 3072, 4096),
          tiles=dict(bm=512, bk=3072, bn=4096),
      ),
      dict(
          testcase_name="single_m_block",
          shape=(1024, 3072, 4096),
          tiles=dict(bm=1024, bk=3072, bn=4096),
      ),
      dict(
          testcase_name="partial_k_cotangent_cache",
          shape=(1024, 3072, 4096),
          tiles=dict(bm=256, bk=1024, bn=4096),
      ),
      dict(
          testcase_name="partial_n_accumulator",
          shape=(1024, 3072, 4096),
          tiles=dict(bm=256, bk=3072, bn=1024),
      ),
      dict(
          testcase_name="partial_k_and_n",
          shape=(512, 2048, 2048),
          tiles=dict(bm=256, bk=512, bn=512),
      ),
      dict(
          testcase_name="smallest_aligned_blocks",
          shape=(256, 384, 512),
          tiles=dict(bm=128, bk=128, bn=128),
      ),
  )
  def test_matches_reference(self, shape, tiles):
    m, k, n = shape
    dout, rhs_q, rhs_scale = _inputs(m, k, n)
    actual = self._dlhs(dout, rhs_q, rhs_scale, **tiles)
    expected = base.fused_fp8_dlhs_reference(dout, rhs_q, rhs_scale)
    self.assertEqual((actual.shape, actual.dtype), ((m, k), dout.dtype))
    # With `bn == n` the kernel quantizes each cotangent row against one scale,
    # exactly like the reference; with `bn < n` each n block gets its own
    # scale, which only agrees up to FP8 quantization noise.
    if tiles["bn"] == n:
      tol = test_base.SAME_QUANTIZATION_TOL
    else:
      tol = test_base.FP8_NOISE_TOL
    self.assertLess(test_base.relative_error(actual, expected), tol)

  def test_float32_cotangent_and_out_dtype(self):
    m, k, n = 512, 1024, 1024
    dout, rhs_q, rhs_scale = _inputs(m, k, n, jnp.float32)
    tiles = dict(bm=512, bk=1024, bn=1024)
    actual = self._dlhs(dout, rhs_q, rhs_scale, **tiles)
    expected = base.fused_fp8_dlhs_reference(dout, rhs_q, rhs_scale)
    self.assertEqual(actual.dtype, jnp.float32)
    self.assertLess(
        test_base.relative_error(actual, expected),
        test_base.SAME_QUANTIZATION_TOL,
    )
    narrow = self._dlhs(dout, rhs_q, rhs_scale, out_dtype=jnp.bfloat16, **tiles)
    self.assertEqual(narrow.dtype, jnp.bfloat16)
    self.assertLess(
        test_base.relative_error(narrow, expected),
        test_base.SAME_QUANTIZATION_TOL,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="contracting_dims_differ",
          corrupt=lambda dout, rhs_q, rhs_scale: (
              dout,
              rhs_q[:, :256],
              rhs_scale,
          ),
          error="Contracting dims differ",
      ),
      dict(
          testcase_name="rhs_scale_wrong_rank",
          corrupt=lambda dout, rhs_q, rhs_scale: (dout, rhs_q, rhs_scale[None]),
          error="rhs_scale must have shape",
      ),
      dict(
          testcase_name="rhs_q_not_fp8",
          corrupt=lambda dout, rhs_q, rhs_scale: (
              dout,
              rhs_q.astype(jnp.bfloat16),
              rhs_scale,
          ),
          error="rhs_q must be",
      ),
      dict(
          testcase_name="bm_does_not_divide_m",
          tiles=dict(bm=100),
          error="m .* must be divisible",
      ),
      dict(
          testcase_name="bk_does_not_divide_k",
          tiles=dict(bk=100),
          error="k .* must be divisible",
      ),
      dict(
          testcase_name="bn_does_not_divide_n",
          tiles=dict(bn=100),
          error="n .* must be divisible",
      ),
  )
  def test_rejects_bad_inputs(
      self, error, corrupt=lambda *args: args, tiles=()
  ):
    args = corrupt(*_inputs(256, 512, 384))
    with self.assertRaisesRegex(ValueError, error):
      self._dlhs(*args, **(dict(bm=128, bk=128, bn=128) | dict(tiles)))


if __name__ == "__main__":
  absltest.main()
