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
from tokamax._src.ops.causal_conv1d_gated_delta_rule import reference
from tokamax._src.ops.causal_conv1d_gated_delta_rule import test_base
from tokamax._src.ops.causal_conv1d_gated_delta_rule import wrapper


class GDNAttentionTest(test_base.CausalConv1dGatedDeltaRuleTestBase):

  def __init__(self, *args):
    # `wrapper.fused_conv1d_gdn` disables buffer donation.
    super().__init__(*args, gdn_fn=wrapper.fused_conv1d_gdn)


class GDNSecurityTest(test_base.CausalConv1dGatedDeltaRuleSecurityTestBase):

  def __init__(self, *args):
    super().__init__(*args, gdn_fn=wrapper.fused_conv1d_gdn)


class GDNReferenceL2NormTest(parameterized.TestCase):
  """Tests for the `reference.py` l2norm helpers.

  These are implementation-independent, so they live here rather than in the
  shared suite in `test_base.py`.
  """

  def setUp(self):
    super().setUp()
    test_base.skip_if_unsupported(self)

  @parameterized.named_parameters(
      dict(testcase_name="chunked", l2norm_fn=reference.l2norm_chunked),
      dict(testcase_name="ref", l2norm_fn=reference.l2_normalize_ref),
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
  )
  def test_l2norm_returns_input_dtype(self, l2norm_fn):
    """The l2norm helpers must return the input dtype unchanged."""
    x = jax.random.normal(jax.random.key(17), (4, 128)).astype(jnp.bfloat16)
    if l2norm_fn is reference.l2norm_chunked:
      out = l2norm_fn(x, dim=-1)
    else:
      out = l2norm_fn(x)
    self.assertEqual(out.dtype, jnp.bfloat16)


if __name__ == "__main__":
  absltest.main()
