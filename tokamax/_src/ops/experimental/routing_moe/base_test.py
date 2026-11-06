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
"""Correctness unit tests for Routing MoE base operator."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe import base
from tokamax._src.ops.experimental.routing_moe import reference


class RoutingMoeBaseTest(parameterized.TestCase):

  @parameterized.product(
      act_fn=("silu",),
      scoring_fn=("softmax", "sigmoid"),
      renormalize=(True, False),
      topk=(2,),
  )
  def test_base_op_matches_reference(
      self, act_fn, scoring_fn, renormalize, topk
  ):
    tokens = 16
    hidden = 64
    inter = 32
    experts = 4

    key = jax.random.PRNGKey(0)
    k_x, k_w1, k_w2, k_g = jax.random.split(key, 4)

    inputs = jax.random.normal(k_x, (tokens, hidden), dtype=jnp.bfloat16)
    w1 = jax.random.normal(
        k_w1, (experts, hidden, 2 * inter), dtype=jnp.bfloat16
    )
    w2 = jax.random.normal(k_w2, (experts, inter, hidden), dtype=jnp.bfloat16)
    gating = jax.random.normal(k_g, (tokens, experts), dtype=jnp.float32)

    op_out, _ = base.RoutingMoe()._fwd(
        x=inputs,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
        scoring_fn=scoring_fn,
    )

    ref_out = reference.routing_moe_reference(
        x=inputs,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
        scoring_fn=scoring_fn,
    )

    np_op = np.asarray(op_out, dtype=np.float32)
    np_ref = np.asarray(ref_out, dtype=np.float32)
    np.testing.assert_allclose(np_op, np_ref, rtol=1e-3, atol=1e-3)

  @parameterized.product(
      scoring_fn=("softmax", "sigmoid"),
      renormalize=(True, False),
      scale_layout=(None, "per_channel", "per_block", "per_block_4d"),
  )
  def test_matches_numpy_oracle(self, scoring_fn, renormalize, scale_layout):
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
      blocks = 1 if scale_layout == "per_channel" else num_blocks
      s1 = rng.uniform(0.5, 2.0, (experts, blocks, 2 * inter))
      s2 = rng.uniform(0.5, 2.0, (experts, blocks, hidden))
      s1, s2 = s1.astype(np.float32), s2.astype(np.float32)
      # Blocks split the contraction dimension.
      w1_dq = w1 * np.repeat(s1, hidden // blocks, axis=1)
      w2_dq = w2 * np.repeat(s2, inter // blocks, axis=1)
      if scale_layout == "per_channel":
        s1, s2 = s1[:, 0], s2[:, 0]
      elif scale_layout == "per_block_4d":
        s1, s2 = s1[:, :, None], s2[:, :, None]
      w1_scale, w2_scale = jnp.asarray(s1), jnp.asarray(s2)

    # Full float32 matmuls, so the TPU run can use the same tolerance.
    with jax.default_matmul_precision("float32"):
      out, _ = base.RoutingMoe()._fwd(
          x=jnp.asarray(x),
          w1=jnp.asarray(w1),
          w2=jnp.asarray(w2),
          gating=jnp.asarray(gating),
          w1_scale=w1_scale,
          w2_scale=w2_scale,
          topk=topk,
          renormalize=renormalize,
          scoring_fn=scoring_fn,
      )

    # Dense NumPy oracle, written independently of `reference.py`.
    g = gating.astype(np.float64)
    if scoring_fn == "softmax":
      scores = np.exp(g - g.max(-1, keepdims=True))
      scores /= scores.sum(-1, keepdims=True)
    else:
      scores = 1.0 / (1.0 + np.exp(-g))
    chosen = np.argsort(-scores, axis=-1)[:, :topk]
    weights = np.take_along_axis(scores, chosen, axis=-1)
    if renormalize:
      weights /= weights.sum(-1, keepdims=True)
    expected = np.zeros((tokens, hidden))
    for t in range(tokens):
      for e, wt in zip(chosen[t], weights[t]):
        h = x[t].astype(np.float64) @ w1_dq[e]
        gate, up = h[:inter], h[inter:]
        act = gate / (1.0 + np.exp(-gate)) * up
        expected[t] += wt * (act @ w2_dq[e])

    np.testing.assert_allclose(
        np.asarray(out, np.float64), expected, rtol=1e-4, atol=1e-4
    )


if __name__ == "__main__":
  absltest.main()
