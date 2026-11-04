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
"""Base tests for GDN attention."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import compute_gdn
from tokamax._src.ops.causal_conv1d_gated_delta_rule import reference
from tokamax._src.ops.causal_conv1d_gated_delta_rule import test_base
from tokamax._src.ops.causal_conv1d_gated_delta_rule import wrapper


def _fused_conv1d_gdn_tuple(*, conv_state, recurrent_state, **kwargs):
  return wrapper.fused_conv1d_gdn(
      states=(conv_state, recurrent_state),
      **kwargs,
  )


class GDNAttentionTest(test_base.CausalConv1dGatedDeltaRuleTestBase):

  def __init__(self, *args):
    # `wrapper.fused_conv1d_gdn` disables buffer donation.
    super().__init__(*args, gdn_fn=_fused_conv1d_gdn_tuple)


class GDNSecurityTest(test_base.CausalConv1dGatedDeltaRuleSecurityTestBase):

  def __init__(self, *args):
    super().__init__(*args, gdn_fn=_fused_conv1d_gdn_tuple)


class GDNReferenceL2NormTest(parameterized.TestCase):
  """Tests for the GDN l2norm helpers.

  These are implementation-independent, so they live here rather than in the
  shared suite in `test_base.py`.
  """

  def setUp(self):
    super().setUp()
    test_base.skip_if_unsupported(self)

  @parameterized.named_parameters(
      dict(testcase_name="chunked", l2norm_fn=reference.l2norm_chunked),
      dict(testcase_name="ref", l2norm_fn=reference.l2_normalize_ref),
      dict(testcase_name="compute_gdn", l2norm_fn=compute_gdn.l2_norm),
  )
  def test_l2norm_fp32_internal_more_precise_than_bf16(self, l2norm_fn):
    """The l2norm helper must accumulate in fp32 even for bf16 input."""
    rng = jax.random.key(13)
    x_f32 = jax.random.normal(rng, (32, 256)) * 5.0
    x_bf16 = x_f32.astype(jnp.bfloat16)

    x_fp64 = x_bf16.astype(jnp.float64)
    sq_sum_fp64 = (x_fp64 * x_fp64).sum(axis=-1, keepdims=True)
    ref = (x_fp64 / jnp.sqrt(sq_sum_fp64 + 1e-6)).astype(jnp.float32)

    if l2norm_fn is reference.l2norm_chunked:
      test_out = l2norm_fn(x_bf16, dim=-1, eps=1e-6)
    else:
      test_out = l2norm_fn(x_bf16, eps=1e-6)
    bf16_only = x_bf16 * jax.lax.rsqrt(
        (x_bf16 * x_bf16).sum(axis=-1, keepdims=True)
        + jnp.array(1e-6, dtype=jnp.bfloat16)
    )

    test_err = float(jnp.max(jnp.abs(test_out.astype(jnp.float32) - ref)))
    bf16_err = float(jnp.max(jnp.abs(bf16_only.astype(jnp.float32) - ref)))

    self.assertLess(
        test_err,
        bf16_err / 2.0,
        f"fp32 l2norm err {test_err:.4g} is not at "
        f"least 2× tighter than bf16 err {bf16_err:.4g}",
    )

  @parameterized.named_parameters(
      dict(testcase_name="chunked", l2norm_fn=reference.l2norm_chunked),
      dict(testcase_name="ref", l2norm_fn=reference.l2_normalize_ref),
      dict(testcase_name="compute_gdn", l2norm_fn=compute_gdn.l2_norm),
  )
  def test_l2norm_returns_input_dtype(self, l2norm_fn):
    """The l2norm helpers must return the input dtype unchanged."""
    x = jax.random.normal(jax.random.key(17), (4, 128)).astype(jnp.bfloat16)
    if l2norm_fn is reference.l2norm_chunked:
      out = l2norm_fn(x, dim=-1)
    else:
      out = l2norm_fn(x)
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_recurrent_step_fp32_internal_more_precise_than_bf16(self):
    """Recurrent step contractions must use fp32 precision rather than bf16."""
    k0, k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(29), 6)
    shape_qk = (2, 4, 1, 128)
    shape_v = (2, 4, 1, 128)
    shape_bg = (2, 4, 1)
    shape_state = (2, 4, 128, 128)

    q = jax.random.normal(k0, shape_qk, dtype=jnp.float32)
    k = jax.random.normal(k1, shape_qk, dtype=jnp.float32)
    v = jax.random.normal(k2, shape_v, dtype=jnp.float32)
    g = -jnp.abs(jax.random.normal(k3, shape_bg, dtype=jnp.float32)) * 0.1
    beta = jax.nn.sigmoid(jax.random.normal(k4, shape_bg, dtype=jnp.float32))
    state = jax.random.normal(k5, shape_state, dtype=jnp.float32)

    test_out, test_state = reference._recurrent_gated_delta_rule_step(
        q, k, v, g, beta, state=state
    )

    with jax.enable_x64():
      q64 = q.astype(jnp.float64)[:, :, 0] * (128**-0.5)
      k64 = k.astype(jnp.float64)[:, :, 0]
      v64 = v.astype(jnp.float64)[:, :, 0]
      g64 = g.astype(jnp.float64)[:, :, 0]
      beta64 = beta.astype(jnp.float64)[:, :, 0]
      s64 = state.astype(jnp.float64)
      k_state64 = jnp.einsum(
          "bhd, bhdm -> bhm", k64, s64, precision=jax.lax.Precision.HIGHEST
      )
      v_new64 = beta64[..., None] * (v64 - jnp.exp(g64)[..., None] * k_state64)
      q_state64 = jnp.einsum(
          "bhd, bhdm -> bhm", q64, s64, precision=jax.lax.Precision.HIGHEST
      )
      q_k64 = jnp.sum(q64 * k64, axis=-1, keepdims=True)
      ref_out = (jnp.exp(g64)[..., None] * q_state64 + q_k64 * v_new64)[
          :, :, None, :
      ].astype(jnp.float32)
      ref_state = (
          s64 * jnp.exp(g64)[..., None, None]
          + jnp.einsum(
              "bhd, bhm -> bhdm",
              k64,
              v_new64,
              precision=jax.lax.Precision.HIGHEST,
          )
      ).astype(jnp.float32)

    q_bf16 = (q[:, :, 0] * (128**-0.5)).astype(jnp.bfloat16)
    k_bf16 = k[:, :, 0].astype(jnp.bfloat16)
    s_bf16 = state.astype(jnp.bfloat16)
    k_state_bf16 = jnp.einsum(
        "bhd, bhdm -> bhm",
        k_bf16,
        s_bf16,
        preferred_element_type=jnp.float32,
    )
    v_new_bf16 = beta[:, :, 0, None] * (
        v[:, :, 0] - jnp.exp(g[:, :, 0])[..., None] * k_state_bf16
    )
    q_state_bf16 = jnp.einsum(
        "bhd, bhdm -> bhm",
        q_bf16,
        s_bf16,
        preferred_element_type=jnp.float32,
    )
    q_k_bf16 = jnp.sum(
        (q[:, :, 0] * (128**-0.5)) * k[:, :, 0], axis=-1, keepdims=True
    )
    bf16_out = (
        jnp.exp(g[:, :, 0])[..., None] * q_state_bf16 + q_k_bf16 * v_new_bf16
    )[:, :, None, :]

    test_out_err = float(jnp.max(jnp.abs(test_out - ref_out)))
    bf16_out_err = float(jnp.max(jnp.abs(bf16_out - ref_out)))
    test_state_err = float(jnp.max(jnp.abs(test_state - ref_state)))

    self.assertLess(
        test_out_err,
        bf16_out_err / 10.0,
        f"fp32 recurrent step out err {test_out_err:.4g} is not at "
        f"least 10× tighter than bf16 MXU err {bf16_out_err:.4g}",
    )
    self.assertLess(test_state_err, 1e-4)


if __name__ == "__main__":
  absltest.main()
