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
"""Tests for the baseline JAX implementation of mHC."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.mhc import base
from tokamax._src.ops.experimental.tpu.mhc import test_base

jax.config.parse_flags_with_absl()


def _np(x) -> np.ndarray:
  return np.asarray(jnp.asarray(x, jnp.float32), np.float64)


class BaseMhcTest(test_base.MhcTestBase):

  def __init__(self, *args):
    super().__init__(
        *args,
        pre_fn=base.MhcPre(),
        post_fn=base.MhcPost(),
        fused_fn=base.MhcFusedPostPre(),
    )

  def test_post_matches_numpy(self):
    """Checks the post formula against an independent float64 computation."""
    x, residual, post_mix, comb_mix = test_base.make_post_inputs(0, 9, 256)
    got = base.MhcPost()(x, residual, post_mix, comb_mix)
    x, residual, post_mix, comb_mix = map(
        _np, (x, residual, post_mix, comb_mix)
    )
    residual = residual.reshape(9, test_base.HC_MULT, 256)
    # new[j] = post[j] * x + sum_i comb[i, j] * residual[i]
    want = post_mix[:, :, None] * x[:, None, :] + np.einsum(
        "tij,tih->tjh", comb_mix, residual
    )
    test_base.assert_allclose(
        got,
        want.reshape(9, -1),
        rtol=test_base.BF16_RTOL,
        atol=test_base.BF16_ATOL,
    )

  def test_pre_matches_numpy(self):
    """Checks the pre gates and collapse against a float64 computation."""
    m = test_base.HC_MULT
    rms_eps, pre_eps, sk_eps, post_mult, repeat = test_base.gate_constants()
    residual, fn, hc_scale, hc_base = test_base.make_pre_inputs(0, 9, 256)
    got_post, got_comb, got_layer = base.MhcPre()(
        residual, fn, hc_scale, hc_base, *test_base.gate_constants()
    )

    x = _np(residual).reshape(9, -1)
    fn, hc_scale, hc_base = _np(fn), _np(hc_scale), _np(hc_base)
    mixes = x @ fn.T
    mixes /= np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + rms_eps)
    sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
    pre = sigmoid(mixes[:, :m] * hc_scale[0] + hc_base[:m]) + pre_eps
    post = sigmoid(mixes[:, m : 2 * m] * hc_scale[1] + hc_base[m : 2 * m])
    post *= post_mult
    comb = mixes[:, 2 * m :].reshape(-1, m, m) * hc_scale[2]
    comb += hc_base[2 * m :].reshape(m, m)
    comb = np.exp(comb - comb.max(axis=-1, keepdims=True))
    comb = comb / comb.sum(axis=-1, keepdims=True) + sk_eps
    comb /= comb.sum(axis=-2, keepdims=True) + sk_eps
    for _ in range(repeat - 1):
      comb /= comb.sum(axis=-1, keepdims=True) + sk_eps
      comb /= comb.sum(axis=-2, keepdims=True) + sk_eps
    layer = np.einsum("ti,tih->th", pre, x.reshape(9, m, -1))

    test_base.assert_allclose(
        got_post, post, rtol=test_base.F32_RTOL, atol=test_base.F32_ATOL
    )
    test_base.assert_allclose(
        got_comb, comb, rtol=test_base.F32_RTOL, atol=test_base.F32_ATOL
    )
    test_base.assert_allclose(
        got_layer, layer, rtol=test_base.BF16_RTOL, atol=test_base.BF16_ATOL
    )
    # After 20 Sinkhorn rounds the comb mix is (nearly) doubly stochastic.
    np.testing.assert_allclose(_np(got_comb).sum(axis=-2), 1.0, atol=1e-5)
    np.testing.assert_allclose(_np(got_comb).sum(axis=-1), 1.0, atol=1e-3)

  def test_rejects_non_bf16_residual(self):
    residual, fn, hc_scale, hc_base = test_base.make_pre_inputs(0, 4, 128)
    with self.assertRaisesRegex(ValueError, "residual must be bfloat16"):
      base.MhcPre()(
          residual.astype(jnp.float32),
          fn,
          hc_scale,
          hc_base,
          *test_base.gate_constants(),
      )

  def test_rejects_non_f32_fn(self):
    residual, fn, hc_scale, hc_base = test_base.make_pre_inputs(0, 4, 128)
    with self.assertRaisesRegex(ValueError, "fn must be float32"):
      base.MhcPre()(
          residual,
          fn.astype(jnp.bfloat16),
          hc_scale,
          hc_base,
          *test_base.gate_constants(),
      )

  @parameterized.parameters(23, 25)
  def test_rejects_fn_rows_not_m_times_m_plus_2(self, rows):
    residual, _, hc_scale, _ = test_base.make_pre_inputs(0, 4, 128)
    with self.assertRaisesRegex(ValueError, r"M \* \(M \+ 2\) rows"):
      base.MhcPre()(
          residual,
          jnp.zeros((rows, residual.shape[1]), jnp.float32),
          hc_scale,
          jnp.zeros((rows,), jnp.float32),
          *test_base.gate_constants(),
      )

  def test_rejects_residual_width_not_multiple_of_hc_mult(self):
    _, fn, hc_scale, hc_base = test_base.make_pre_inputs(0, 4, 128)
    width = 4 * 128 + 2
    with self.assertRaisesRegex(ValueError, "multiple of hc_mult=4"):
      base.MhcPre()(
          jnp.zeros((4, width), jnp.bfloat16),
          jnp.zeros((fn.shape[0], width), jnp.float32),
          hc_scale,
          hc_base,
          *test_base.gate_constants(),
      )

  def test_rejects_zero_sinkhorn_repeat(self):
    args = test_base.make_pre_inputs(0, 4, 128)
    with self.assertRaisesRegex(ValueError, "sinkhorn_repeat"):
      base.MhcPre()(*args, *test_base.gate_constants(0))

  def test_post_rejects_non_f32_gates(self):
    x, residual, post_mix, comb_mix = test_base.make_post_inputs(0, 4, 128)
    with self.assertRaisesRegex(ValueError, "comb_res_mix must be float32"):
      base.MhcPost()(x, residual, post_mix, comb_mix.astype(jnp.bfloat16))

  def test_post_rejects_residual_width_mismatch(self):
    x, residual, post_mix, comb_mix = test_base.make_post_inputs(0, 4, 128)
    with self.assertRaisesRegex(ValueError, "residual must have width"):
      base.MhcPost()(x[:, :64], residual, post_mix, comb_mix)

  def test_fused_rejects_non_bf16_x(self):
    args = list(test_base.make_fused_inputs(0, 4, 128))
    args[0] = args[0].astype(jnp.float32)
    with self.assertRaisesRegex(ValueError, "x must be bfloat16"):
      base.MhcFusedPostPre()(*args, *test_base.gate_constants())

  def test_fused_rejects_hc_mult_mismatch(self):
    # M = 2 for the post gates; fn is built for M = 4 (24 rows) at the same
    # residual width, so H differs between the two halves.
    x, residual, post_mix, comb_mix = test_base.make_post_inputs(
        0, 4, 256, hc_mult=2
    )
    _, fn, hc_scale, hc_base = test_base.make_pre_inputs(0, 4, 128)
    with self.assertRaisesRegex(ValueError, "hc_mult=2 but fn"):
      base.MhcFusedPostPre()(
          x,
          residual,
          post_mix,
          comb_mix,
          fn,
          hc_scale,
          hc_base,
          *test_base.gate_constants(),
      )


if __name__ == "__main__":
  absltest.main()
