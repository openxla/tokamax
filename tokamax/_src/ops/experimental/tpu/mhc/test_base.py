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
"""Shared correctness tests for mHC implementations.

Inputs, tolerances and shapes follow the DeepSeek-V4 mHC kernel tests the
kernels were developed against.
"""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.mhc import reference

HC_MULT = 4
# Values from the official DeepSeek-V4 config.json: rms_norm_eps=1e-6,
# hc_eps=1e-6 (feeds both pre_eps and sinkhorn_eps), hc_sinkhorn_iters=20;
# hc_post_alpha=2.0 is hardcoded in the model.
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0
SINKHORN_REPEAT = 20

# f32 outputs: absorb summation-order differences of the (M * H)-long f32
# accumulations.
F32_RTOL, F32_ATOL = 1e-4, 1e-5
# bf16 outputs: ~1-2 ulp at magnitude O(1).
BF16_RTOL, BF16_ATOL = 2e-2, 2e-2
# Fused-path f32 gate outputs: the fused recombine and the reference's einsum
# can round near-boundary stream values to different adjacent bf16 values
# before the mix GEMM, so the gates carry ~1 bf16 ulp of extra input noise
# (worst at small H, where each element weighs more).
FUSED_F32_RTOL, FUSED_F32_ATOL = 1e-3, 1e-5
FUSED_SMALL_H_RTOL = 5e-3


def gate_constants(
    sinkhorn_repeat: int = SINKHORN_REPEAT,
) -> tuple[float, float, float, float, int]:
  """Returns the DeepSeek-V4 gate constants, in argument order."""
  return (
      RMS_EPS,
      HC_PRE_EPS,
      HC_SINKHORN_EPS,
      HC_POST_MULT_VALUE,
      sinkhorn_repeat,
  )


def make_pre_inputs(
    seed: int,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int = HC_MULT,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Returns random `(residual, fn, hc_scale, hc_base)`, `residual` flat."""
  rng = np.random.default_rng(seed)
  hc_mult3 = hc_mult * (hc_mult + 2)
  residual = rng.standard_normal((num_tokens, hc_mult, hidden_size), np.float32)
  fn = rng.standard_normal((hc_mult3, hc_mult * hidden_size), np.float32)
  hc_scale = (1.0 + 0.1 * rng.standard_normal(3)).astype(np.float32)
  hc_base = (0.5 * rng.standard_normal(hc_mult3)).astype(np.float32)
  return (
      jnp.asarray(residual.reshape(num_tokens, -1), jnp.bfloat16),
      jnp.asarray(fn * 0.02),
      jnp.asarray(hc_scale),
      jnp.asarray(hc_base),
  )


def make_post_inputs(
    seed: int,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int = HC_MULT,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Returns random `(x, residual, post_layer_mix, comb_res_mix)`.

  `residual` is flat `(T, M * H)`. The gates mimic `mhc_pre` outputs: a
  sigmoid-scaled `(T, M)` post mix and a positive, row-normalized `(T, M, M)`
  comb mix, both f32.

  Args:
    seed: The random seed.
    num_tokens: The number of tokens `T`.
    hidden_size: The hidden size `H`.
    hc_mult: The number of streams `M`.
  """
  rng = np.random.default_rng(seed)
  t, m = num_tokens, hc_mult
  x = rng.standard_normal((t, hidden_size), np.float32)
  residual = rng.standard_normal((t, m, hidden_size), np.float32)
  post_mix = 2.0 / (1.0 + np.exp(-rng.standard_normal((t, m, 1))))
  comb_raw = np.abs(rng.standard_normal((t, m, m))) + 0.01
  comb_mix = comb_raw / comb_raw.sum(axis=-1, keepdims=True)
  return (
      jnp.asarray(x, jnp.bfloat16),
      jnp.asarray(residual.reshape(t, -1), jnp.bfloat16),
      jnp.asarray(post_mix.reshape(t, m), jnp.float32),
      jnp.asarray(comb_mix, jnp.float32),
  )


def make_fused_inputs(
    seed: int,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int = HC_MULT,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
  """Returns `make_post_inputs(...) + make_pre_inputs(...)[1:]`."""
  post_args = make_post_inputs(seed, num_tokens, hidden_size, hc_mult)
  _, fn, hc_scale, hc_base = make_pre_inputs(
      seed + 1, num_tokens, hidden_size, hc_mult
  )
  return (*post_args, fn, hc_scale, hc_base)


def assert_allclose(actual, desired, *, rtol, atol, err_msg=""):
  np.testing.assert_allclose(
      np.asarray(jnp.asarray(actual, jnp.float32)),
      np.asarray(jnp.asarray(desired, jnp.float32)),
      rtol=rtol,
      atol=atol,
      err_msg=err_msg,
  )


def assert_pre_close(test, got, want, *, f32_rtol=F32_RTOL):
  """Checks `(post_mix, comb_mix, layer_input)` against the reference."""
  got_post, got_comb, got_layer = got
  want_post, want_comb, want_layer = want
  for g, w in zip(got, want):
    test.assertEqual(g.shape, w.shape)
    test.assertEqual(g.dtype, w.dtype)
  assert_allclose(got_post, want_post, rtol=f32_rtol, atol=F32_ATOL)
  assert_allclose(got_comb, want_comb, rtol=f32_rtol, atol=F32_ATOL)
  assert_allclose(got_layer, want_layer, rtol=BF16_RTOL, atol=BF16_ATOL)


class MhcTestBase(parameterized.TestCase):
  """Correctness suite shared by all mHC implementations.

  Subclasses pass the implementations under test as `pre_fn`, `post_fn` and
  `fused_fn`. Results are checked against `reference`.
  """

  def __init__(self, *args, pre_fn, post_fn, fused_fn):
    super().__init__(*args)
    self._pre_fn = pre_fn
    self._post_fn = post_fn
    self._fused_fn = fused_fn

  @parameterized.product(
      num_tokens=[1, 17, 128],
      hidden_size=[256, 7168],
      sinkhorn_repeat=[1, 2, 20],
  )
  def test_pre(self, num_tokens, hidden_size, sinkhorn_repeat):
    """Checks `mhc_pre` output shapes and values against the reference."""
    args = make_pre_inputs(0, num_tokens, hidden_size)
    consts = gate_constants(sinkhorn_repeat)
    want = reference.mhc_pre(*args, *consts)
    got = self._pre_fn(*args, *consts)
    self.assertEqual(got[0].shape, (num_tokens, HC_MULT))
    self.assertEqual(got[1].shape, (num_tokens, HC_MULT, HC_MULT))
    self.assertEqual(got[2].shape, (num_tokens, hidden_size))
    assert_pre_close(self, got, want)

  def test_pre_hc_mult_2(self):
    """Checks that `M` is inferred from `fn` rather than assumed to be 4."""
    args = make_pre_inputs(1, 17, 256, hc_mult=2)
    want = reference.mhc_pre(*args, *gate_constants())
    got = self._pre_fn(*args, *gate_constants())
    self.assertEqual(got[0].shape, (17, 2))
    assert_pre_close(self, got, want)

  def test_pre_padded_zero_rows_are_finite(self):
    """vLLM pads token buckets with all-zero rows; they must not NaN."""
    residual, fn, hc_scale, hc_base = make_pre_inputs(2, 8, 256)
    residual = residual.at[4:].set(0)
    got = self._pre_fn(residual, fn, hc_scale, hc_base, *gate_constants(2))
    for out in got:
      self.assertTrue(bool(jnp.isfinite(out.astype(jnp.float32)).all()))

  @parameterized.product(num_tokens=[1, 17, 128], hidden_size=[256, 7168])
  def test_post(self, num_tokens, hidden_size):
    args = make_post_inputs(3, num_tokens, hidden_size)
    want = reference.mhc_post(*args)
    got = self._post_fn(*args)
    self.assertEqual(got.shape, (num_tokens, HC_MULT * hidden_size))
    self.assertEqual(got.dtype, jnp.bfloat16)
    assert_allclose(got, want, rtol=BF16_RTOL, atol=BF16_ATOL)

  def test_post_hc_mult_2(self):
    """Checks that `M` is inferred from `comb_res_mix`."""
    args = make_post_inputs(4, 17, 256, hc_mult=2)
    want = reference.mhc_post(*args)
    got = self._post_fn(*args)
    self.assertEqual(got.shape, (17, 2 * 256))
    assert_allclose(got, want, rtol=BF16_RTOL, atol=BF16_ATOL)

  @parameterized.parameters(
      (1, 256, FUSED_SMALL_H_RTOL),
      (17, 256, FUSED_SMALL_H_RTOL),
      (64, 256, FUSED_SMALL_H_RTOL),
      (200, 256, FUSED_SMALL_H_RTOL),
      (16, 7168, FUSED_F32_RTOL),
      (128, 7168, FUSED_F32_RTOL),
  )
  def test_fused_post_pre(self, num_tokens, hidden_size, f32_rtol):
    """Checks the fused seam op against reference post-then-pre."""
    args = make_fused_inputs(5, num_tokens, hidden_size)
    want = reference.mhc_fused_post_pre(*args, *gate_constants())
    got = self._fused_fn(*args, *gate_constants())
    self.assertEqual(got[0].shape, (num_tokens, HC_MULT * hidden_size))
    self.assertEqual(got[0].dtype, jnp.bfloat16)
    assert_allclose(got[0], want[0], rtol=BF16_RTOL, atol=BF16_ATOL)
    assert_pre_close(self, got[1:], want[1:], f32_rtol=f32_rtol)

  def test_fused_post_pre_hc_mult_2(self):
    args = make_fused_inputs(6, 17, 256, hc_mult=2)
    want = reference.mhc_fused_post_pre(*args, *gate_constants())
    got = self._fused_fn(*args, *gate_constants())
    assert_allclose(got[0], want[0], rtol=BF16_RTOL, atol=BF16_ATOL)
    assert_pre_close(self, got[1:], want[1:], f32_rtol=FUSED_SMALL_H_RTOL)
