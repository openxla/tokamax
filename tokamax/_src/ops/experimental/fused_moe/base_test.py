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
"""Correctness unit tests for Fused MoE base operator."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.fused_moe import base
from tokamax._src.ops.experimental.fused_moe import reference


class FusedMoeBaseTest(parameterized.TestCase):

  @parameterized.product(
      act_fn=["silu"],
      renormalize=[True, False],
      topk=[2],
  )
  def test_base_op_matches_reference(self, act_fn, renormalize, topk):
    tokens = 16
    hidden = 64
    inter = 32
    experts = 4

    key = jax.random.PRNGKey(0)
    k_x, k_w1, k_w2, k_g = jax.random.split(key, 4)

    x = jax.random.normal(k_x, (tokens, hidden), dtype=jnp.bfloat16)
    w1 = jax.random.normal(
        k_w1, (experts, hidden, 2 * inter), dtype=jnp.bfloat16
    )
    w2 = jax.random.normal(k_w2, (experts, inter, hidden), dtype=jnp.bfloat16)
    gating = jax.random.normal(k_g, (tokens, experts), dtype=jnp.float32)

    op_out, _ = base.FusedMoe()._fwd(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
    )

    ref_out = reference.fused_moe_reference(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
    )

    np_op = np.asarray(op_out, dtype=np.float32)
    np_ref = np.asarray(ref_out, dtype=np.float32)
    np.testing.assert_allclose(np_op, np_ref, rtol=1e-3, atol=1e-3)

  @parameterized.product(
      renormalize=(True, False),
      scale_layout=(
          None,
          "per_channel",
          "per_channel_4d",
          "per_block",
          "per_block_4d",
      ),
  )
  def test_matches_numpy_oracle(self, renormalize, scale_layout):
    tokens, hidden, inter, experts, topk, num_blocks = 8, 32, 16, 4, 2, 4
    rng = np.random.default_rng(0)
    x = rng.standard_normal((tokens, hidden)).astype(np.float32)
    w1 = rng.standard_normal((experts, hidden, 2 * inter)).astype(np.float32)
    w2 = rng.standard_normal((experts, inter, hidden)).astype(np.float32)
    gating = rng.standard_normal((tokens, experts)).astype(np.float32)
    x, w1, w2 = x / 4, w1 / 4, w2 / 4

    w1_scale = w2_scale = None
    w1_dq, w2_dq = w1, w2
    if scale_layout is not None:
      blocks = (
          1 if scale_layout in ("per_channel", "per_channel_4d") else num_blocks
      )
      s1 = rng.uniform(0.5, 2.0, (experts, blocks, 2 * inter))
      s2 = rng.uniform(0.5, 2.0, (experts, blocks, hidden))
      s1, s2 = s1.astype(np.float32), s2.astype(np.float32)
      # Blocks split the contraction dimension.
      w1_dq = w1 * np.repeat(s1, hidden // blocks, axis=1)
      w2_dq = w2 * np.repeat(s2, inter // blocks, axis=1)
      if scale_layout == "per_channel":
        s1, s2 = s1[:, 0], s2[:, 0]
      elif scale_layout in ("per_channel_4d", "per_block_4d"):
        s1, s2 = s1[:, :, None], s2[:, :, None]
      w1_scale, w2_scale = jnp.asarray(s1), jnp.asarray(s2)

    with jax.default_matmul_precision("float32"):
      out, _ = base.FusedMoe()._fwd(
          x=jnp.asarray(x),
          w1=jnp.asarray(w1),
          w2=jnp.asarray(w2),
          gating=jnp.asarray(gating),
          w1_scale=w1_scale,
          w2_scale=w2_scale,
          topk=topk,
          renormalize=renormalize,
      )

    gating_f64 = gating.astype(np.float64)
    scores = np.exp(gating_f64 - gating_f64.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)
    chosen = np.argsort(-scores, axis=-1)[:, :topk]
    weights = np.take_along_axis(scores, chosen, axis=-1)
    if renormalize:
      weights /= weights.sum(-1, keepdims=True)
    expected = np.zeros((tokens, hidden))
    for tok in range(tokens):
      for e, wt in zip(chosen[tok], weights[tok]):
        hidden_act = x[tok].astype(np.float64) @ w1_dq[e]
        gate, up = hidden_act[:inter], hidden_act[inter:]
        act = gate / (1.0 + np.exp(-gate)) * up
        expected[tok] += wt * (act @ w2_dq[e])

    np.testing.assert_allclose(
        np.asarray(out, np.float64), expected, rtol=1e-4, atol=1e-4
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="fp8_per_channel",
          w_dtype=jnp.float8_e4m3fn,
          num_blocks=1,
      ),
      dict(
          testcase_name="fp4_per_block",
          w_dtype=jnp.float4_e2m1fn,
          num_blocks=4,
      ),
  )
  def test_quantized_weights_match_dequantized_reference(
      self, w_dtype, num_blocks
  ):
    tokens, hidden, inter, experts, topk = 8, 32, 16, 4, 2
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((tokens, hidden)) / 4, jnp.float32)
    w1_q = jnp.asarray(
        rng.standard_normal((experts, hidden, 2 * inter)), w_dtype
    )
    w2_q = jnp.asarray(rng.standard_normal((experts, inter, hidden)), w_dtype)
    gating = jnp.asarray(rng.standard_normal((tokens, experts)), jnp.float32)
    s1 = jnp.asarray(
        rng.uniform(0.5, 2.0, (experts, num_blocks, 2 * inter)), jnp.float32
    )
    s2 = jnp.asarray(
        rng.uniform(0.5, 2.0, (experts, num_blocks, hidden)), jnp.float32
    )
    w1_dq = w1_q.astype(jnp.float32) * jnp.repeat(
        s1, hidden // num_blocks, axis=1
    )
    w2_dq = w2_q.astype(jnp.float32) * jnp.repeat(
        s2, inter // num_blocks, axis=1
    )
    if num_blocks == 1:
      s1, s2 = s1[:, 0], s2[:, 0]

    with jax.default_matmul_precision("float32"):
      got, _ = base.FusedMoe()._fwd(
          x=x,
          w1=w1_q,
          w2=w2_q,
          gating=gating,
          w1_scale=s1,
          w2_scale=s2,
          topk=topk,
      )
      want, _ = base.FusedMoe()._fwd(
          x=x, w1=w1_dq, w2=w2_dq, gating=gating, topk=topk
      )
    np.testing.assert_allclose(
        np.asarray(got, np.float32),
        np.asarray(want, np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="unsupported_rank",
          w1_scale_shape=(4, 32),
          w2_scale_shape=(4, 1, 1, 1, 32),
          error_regex="Unsupported weight scale shape",
      ),
      dict(
          testcase_name="non_unit_axis2_4d",
          w1_scale_shape=(4, 1, 4, 32),
          w2_scale_shape=(4, 1, 1, 32),
          error_regex="size 1 on axis 2",
      ),
      dict(
          testcase_name="non_divisible_blocks",
          w1_scale_shape=(4, 3, 32),
          w2_scale_shape=(4, 2, 32),
          error_regex="do not divide contraction dimension",
      ),
      dict(
          testcase_name="zero_blocks",
          w1_scale_shape=(4, 0, 32),
          w2_scale_shape=(4, 2, 32),
          error_regex="do not divide contraction dimension",
      ),
  )
  def test_invalid_weight_scale_raises(
      self, w1_scale_shape, w2_scale_shape, error_regex
  ):
    tokens, hidden, inter, experts = 4, 32, 16, 4
    x = jnp.zeros((tokens, hidden), jnp.float32)
    w1 = jnp.zeros((experts, hidden, 2 * inter), jnp.float32)
    w2 = jnp.zeros((experts, inter, hidden), jnp.float32)
    gating = jnp.zeros((tokens, experts), jnp.float32)
    with self.assertRaisesRegex(ValueError, error_regex):
      base.FusedMoe()._fwd(
          x=x,
          w1=w1,
          w2=w2,
          gating=gating,
          w1_scale=jnp.ones(w1_scale_shape, jnp.float32),
          w2_scale=jnp.ones(w2_scale_shape, jnp.float32),
          topk=2,
      )


if __name__ == "__main__":
  absltest.main()
