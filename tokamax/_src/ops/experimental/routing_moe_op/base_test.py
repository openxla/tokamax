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
import jaxtyping
import numpy as np
from tokamax._src.ops.experimental.routing_moe_op import arg_specs
from tokamax._src.ops.experimental.routing_moe_op import base
from tokamax._src.ops.experimental.routing_moe_op import reference


def _numpy_activation(
    gate: np.ndarray, up: np.ndarray, activation: str
) -> np.ndarray:
  """Computes gated activation in float64 using NumPy."""
  match activation:
    case "silu":
      return gate / (1.0 + np.exp(-gate)) * up
    case "gelu" | "gelu_tanh":
      inner = np.sqrt(2.0 / np.pi) * (gate + 0.044715 * gate**3)
      return 0.5 * gate * (1.0 + np.tanh(inner)) * up
    case "swigluoai":
      gate_clipped = np.clip(gate, None, 7.0)
      up_clipped = np.clip(up, -7.0, 7.0)
      return (
          (up_clipped + 1.0)
          * gate_clipped
          / (1.0 + np.exp(-1.702 * gate_clipped))
      )
    case "silu_and_mul_with_clamp":
      gate_clipped = np.clip(gate, None, 10.0)
      up_clipped = np.clip(up, -10.0, 10.0)
      return gate_clipped / (1.0 + np.exp(-gate_clipped)) * up_clipped
    case _:
      raise ValueError(f"Unsupported activation: {activation}")


def _numpy_moe_oracle(
    hidden_states: np.ndarray,
    topk_weights: np.ndarray,
    topk_indices: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    *,
    w1_scale: np.ndarray | None = None,
    w2_scale: np.ndarray | None = None,
    w1_bias: np.ndarray | None = None,
    w2_bias: np.ndarray | None = None,
    activation: str = "silu",
) -> np.ndarray:
  """Independent float64 NumPy oracle for routing MoE."""
  tokens, hidden = hidden_states.shape
  inter = w2.shape[1]
  expected = np.zeros((tokens, hidden), dtype=np.float64)
  w1_f64 = w1.astype(np.float64)
  w2_f64 = w2.astype(np.float64)
  if w1_scale is not None:
    blocks = w1_scale.shape[1]
    w1_f64 = (
        w1_f64.reshape(w1.shape[0], blocks, -1, w1.shape[2])
        * w1_scale.astype(np.float64)
    ).reshape(w1.shape)
  if w2_scale is not None:
    blocks = w2_scale.shape[1]
    w2_f64 = (
        w2_f64.reshape(w2.shape[0], blocks, -1, w2.shape[2])
        * w2_scale.astype(np.float64)
    ).reshape(w2.shape)
  for token_idx in range(tokens):
    for expert_idx, wt in zip(topk_indices[token_idx], topk_weights[token_idx]):
      acc1 = hidden_states[token_idx].astype(np.float64) @ w1_f64[expert_idx]
      if w1_bias is not None:
        acc1 = acc1 + w1_bias[expert_idx, 0].astype(np.float64)
      act = _numpy_activation(acc1[:inter], acc1[inter:], activation)
      acc2 = act @ w2_f64[expert_idx]
      if w2_bias is not None:
        acc2 = acc2 + w2_bias[expert_idx, 0].astype(np.float64)
      expected[token_idx] += float(wt) * acc2
  return expected


class RoutingMoeBaseTest(parameterized.TestCase):

  @parameterized.product(
      activation=(
          "silu",
          "gelu",
          "gelu_tanh",
          "swigluoai",
          "silu_and_mul_with_clamp",
      ),
      with_bias=(False, True),
      topk=(2, 4),
  )
  def test_base_op_matches_reference(self, activation, with_bias, topk):
    tokens = 12
    hidden = 32
    inter = 16
    experts = 4

    rng = np.random.default_rng(7)
    hidden_states_np = (rng.standard_normal((tokens, hidden)) / 4).astype(
        np.float32
    )
    w1_np = (rng.standard_normal((experts, hidden, 2 * inter)) / 4).astype(
        np.float32
    )
    w2_np = (rng.standard_normal((experts, inter, hidden)) / 4).astype(
        np.float32
    )
    logits = rng.standard_normal((tokens, experts)).astype(np.float64)
    scores = np.exp(logits - logits.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)
    chosen = np.argsort(-scores, axis=-1)[:, :topk].astype(np.int32)
    weights = np.take_along_axis(scores, chosen, axis=-1)
    weights = (weights / weights.sum(-1, keepdims=True)).astype(np.float32)

    w1_bias_np = w2_bias_np = None
    w1_bias_jnp = w2_bias_jnp = None
    if with_bias:
      w1_bias_np = (rng.standard_normal((experts, 1, 2 * inter)) / 4).astype(
          np.float32
      )
      w2_bias_np = (rng.standard_normal((experts, 1, hidden)) / 4).astype(
          np.float32
      )
      w1_bias_jnp = jnp.asarray(w1_bias_np)
      w2_bias_jnp = jnp.asarray(w2_bias_np)

    with jax.default_matmul_precision("highest"):
      op_out = base.RoutingMoe()(
          hidden_states=jnp.asarray(hidden_states_np),
          topk_weights=jnp.asarray(weights),
          topk_indices=jnp.asarray(chosen),
          w1=jnp.asarray(w1_np),
          w2=jnp.asarray(w2_np),
          w1_bias=w1_bias_jnp,
          w2_bias=w2_bias_jnp,
          activation=activation,
      )

    expected = _numpy_moe_oracle(
        hidden_states_np,
        weights,
        chosen,
        w1_np,
        w2_np,
        w1_bias=w1_bias_np,
        w2_bias=w2_bias_np,
        activation=activation,
    )
    tol = 1e-4
    np.testing.assert_allclose(
        np.asarray(op_out, dtype=np.float64), expected, rtol=tol, atol=tol
    )

  @parameterized.product(
      scale_layout=(None, "per_channel", "per_block"),
  )
  def test_matches_numpy_oracle(self, scale_layout):
    tokens, hidden, inter, experts, topk, num_blocks = 8, 32, 16, 4, 2, 4
    rng = np.random.default_rng(0)
    hidden_states = rng.standard_normal((tokens, hidden)).astype(np.float32)
    w1 = rng.standard_normal((experts, hidden, 2 * inter)).astype(np.float32)
    w2 = rng.standard_normal((experts, inter, hidden)).astype(np.float32)
    router_logits = rng.standard_normal((tokens, experts)).astype(np.float32)
    hidden_states, w1, w2 = hidden_states / 4, w1 / 4, w2 / 4

    w1_scale = w2_scale = None
    s1, s2 = None, None
    if scale_layout is not None:
      blocks = 1 if scale_layout == "per_channel" else num_blocks
      s1 = rng.uniform(0.5, 2.0, (experts, blocks, 1, 2 * inter)).astype(
          np.float32
      )
      s2 = rng.uniform(0.5, 2.0, (experts, blocks, 1, hidden)).astype(
          np.float32
      )
      w1_scale, w2_scale = jnp.asarray(s1), jnp.asarray(s2)

    logits_f64 = router_logits.astype(np.float64)
    scores = np.exp(logits_f64 - logits_f64.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)
    chosen = np.argsort(-scores, axis=-1)[:, :topk].astype(np.int32)
    weights = np.take_along_axis(scores, chosen, axis=-1)
    weights /= weights.sum(-1, keepdims=True)

    with jax.default_matmul_precision("highest"):
      out, _ = base.RoutingMoe()._fwd(
          hidden_states=jnp.asarray(hidden_states),
          topk_weights=jnp.asarray(weights, dtype=jnp.float32),
          topk_indices=jnp.asarray(chosen, dtype=jnp.int32),
          w1=jnp.asarray(w1),
          w2=jnp.asarray(w2),
          w1_scale=w1_scale,
          w2_scale=w2_scale,
      )

    expected = _numpy_moe_oracle(
        hidden_states,
        weights,
        chosen,
        w1,
        w2,
        w1_scale=s1,
        w2_scale=s2,
    )

    tol = 1e-4
    np.testing.assert_allclose(
        np.asarray(out, np.float64), expected, rtol=tol, atol=tol
    )

  def test_rejects_mismatched_topk_shapes(self):
    tokens, hidden, inter, experts = 8, 32, 16, 4
    hidden_states = jnp.zeros((tokens, hidden), jnp.bfloat16)
    w1 = jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16)
    w2 = jnp.zeros((experts, inter, hidden), jnp.bfloat16)
    topk_weights = jnp.zeros((tokens, 2), jnp.float32)
    topk_indices = jnp.zeros((tokens, 3), jnp.int32)

    with self.assertRaises((ValueError, TypeError, jaxtyping.TypeCheckError)):
      base.RoutingMoe().bind(
          hidden_states=hidden_states,
          topk_weights=topk_weights,
          topk_indices=topk_indices,
          w1=w1,
          w2=w2,
      )

  def test_rejects_non_integer_topk_indices(self):
    tokens, hidden, inter, experts = 8, 32, 16, 4
    hidden_states = jnp.zeros((tokens, hidden), jnp.bfloat16)
    w1 = jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16)
    w2 = jnp.zeros((experts, inter, hidden), jnp.bfloat16)
    topk_weights = jnp.zeros((tokens, 2), jnp.float32)
    topk_indices = jnp.zeros((tokens, 2), jnp.float32)

    with self.assertRaises((TypeError, ValueError, jaxtyping.TypeCheckError)):
      base.RoutingMoe().bind(
          hidden_states=hidden_states,
          topk_weights=topk_weights,
          topk_indices=topk_indices,
          w1=w1,
          w2=w2,
      )

  def test_rejects_mismatched_tokens(self):
    tokens, hidden, inter, experts = 8, 32, 16, 4
    hidden_states = jnp.zeros((tokens, hidden), jnp.bfloat16)
    w1 = jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16)
    w2 = jnp.zeros((experts, inter, hidden), jnp.bfloat16)
    topk_weights = jnp.zeros((tokens + 1, 2), jnp.float32)
    topk_indices = jnp.zeros((tokens + 1, 2), jnp.int32)

    with self.assertRaises((ValueError, TypeError, jaxtyping.TypeCheckError)):
      base.RoutingMoe().bind(
          hidden_states=hidden_states,
          topk_weights=topk_weights,
          topk_indices=topk_indices,
          w1=w1,
          w2=w2,
      )

  @parameterized.product(
      weight_dtype=("float32", "bfloat16", "float8_e4m3fn"),
      scale_layout=(None, "per_channel", "per_block"),
      with_bias=(False, True),
      activation=("silu", "gelu"),
  )
  def test_reference_matches_numpy_oracle(
      self, weight_dtype, scale_layout, with_bias, activation
  ):
    tokens, hidden, inter, experts, topk, num_blocks = 10, 16, 8, 6, 2, 4
    rng = np.random.default_rng(13)
    in_dtype = jnp.float32 if weight_dtype == "float32" else jnp.bfloat16
    hidden_states = jnp.asarray(
        (rng.standard_normal((tokens, hidden)) / 4).astype(np.float32),
        dtype=in_dtype,
    )
    w1_f32 = (rng.standard_normal((experts, hidden, 2 * inter)) / 4).astype(
        np.float32
    )
    w2_f32 = (rng.standard_normal((experts, inter, hidden)) / 4).astype(
        np.float32
    )
    if weight_dtype == "float8_e4m3fn":
      w1 = jnp.asarray(w1_f32).astype(jnp.float8_e4m3fn)
      w2 = jnp.asarray(w2_f32).astype(jnp.float8_e4m3fn)
    elif weight_dtype == "bfloat16":
      w1 = jnp.asarray(w1_f32, dtype=jnp.bfloat16)
      w2 = jnp.asarray(w2_f32, dtype=jnp.bfloat16)
    else:
      w1 = jnp.asarray(w1_f32, dtype=jnp.float32)
      w2 = jnp.asarray(w2_f32, dtype=jnp.float32)

    w1_scale = w2_scale = None
    if scale_layout is not None:
      blocks = 1 if scale_layout == "per_channel" else num_blocks
      w1_scale = jnp.asarray(
          rng.uniform(0.5, 2.0, (experts, blocks, 1, 2 * inter)).astype(
              np.float32
          )
      )
      w2_scale = jnp.asarray(
          rng.uniform(0.5, 2.0, (experts, blocks, 1, hidden)).astype(np.float32)
      )

    w1_bias = w2_bias = None
    if with_bias:
      w1_bias = jnp.asarray(
          (rng.standard_normal((experts, 1, 2 * inter)) / 4).astype(np.float32),
          dtype=in_dtype,
      )
      w2_bias = jnp.asarray(
          (rng.standard_normal((experts, 1, hidden)) / 4).astype(np.float32),
          dtype=in_dtype,
      )

    logits = rng.standard_normal((tokens, experts)).astype(np.float64)
    scores = np.exp(logits - logits.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)
    chosen = np.argsort(-scores, axis=-1)[:, :topk].astype(np.int32)
    weights = np.take_along_axis(scores, chosen, axis=-1)
    weights = (weights / weights.sum(-1, keepdims=True)).astype(np.float32)
    topk_weights = jnp.asarray(weights)
    topk_indices = jnp.asarray(chosen)

    with jax.default_matmul_precision("highest"):
      actual = reference.routing_moe_reference(
          hidden_states=hidden_states,
          topk_weights=topk_weights,
          topk_indices=topk_indices,
          w1=w1,
          w2=w2,
          w1_scale=w1_scale,
          w2_scale=w2_scale,
          w1_bias=w1_bias,
          w2_bias=w2_bias,
          activation=activation,
      )

    desired = _numpy_moe_oracle(
        np.asarray(hidden_states, dtype=np.float64),
        weights,
        chosen,
        w1_f32,
        w2_f32,
        w1_scale=None if w1_scale is None else np.asarray(w1_scale),
        w2_scale=None if w2_scale is None else np.asarray(w2_scale),
        w1_bias=None if w1_bias is None else np.asarray(w1_bias),
        w2_bias=None if w2_bias is None else np.asarray(w2_bias),
        activation=activation,
    )

    tol = 1e-4 if weight_dtype == "float32" else 2e-2
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(desired, dtype=np.float32),
        rtol=tol,
        atol=tol,
    )

  def test_rejects_tp_axis(self):
    tokens, hidden, inter, experts = 8, 32, 16, 4
    with self.assertRaisesRegex(NotImplementedError, "tp_axis"):
      base.RoutingMoe()(
          hidden_states=jnp.zeros((tokens, hidden), jnp.bfloat16),
          topk_weights=jnp.zeros((tokens, 2), jnp.float32),
          topk_indices=jnp.zeros((tokens, 2), jnp.int32),
          w1=jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16),
          w2=jnp.zeros((experts, inter, hidden), jnp.bfloat16),
          tp_axis="tp",
      )

  def test_rejects_reduce_output_false(self):
    tokens, hidden, inter, experts = 8, 32, 16, 4
    with self.assertRaisesRegex(NotImplementedError, "reduce_output=True"):
      reference.routing_moe_reference(
          hidden_states=jnp.zeros((tokens, hidden), jnp.bfloat16),
          topk_weights=jnp.zeros((tokens, 2), jnp.float32),
          topk_indices=jnp.zeros((tokens, 2), jnp.int32),
          w1=jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16),
          w2=jnp.zeros((experts, inter, hidden), jnp.bfloat16),
          reduce_output=False,
      )

  def test_topk_routing_array_initializer(self):
    num_tokens = 16
    num_experts = 32
    k = 4
    seed = 12345
    init_indices = arg_specs._TopkRoutingArrayInitializer(
        num_tokens=num_tokens,
        num_experts=num_experts,
        k=k,
        field="indices",
        seed=seed,
    )
    init_weights = arg_specs._TopkRoutingArrayInitializer(
        num_tokens=num_tokens,
        num_experts=num_experts,
        k=k,
        field="weights",
        seed=seed,
    )
    dummy_rng = np.random.default_rng(0)
    indices1 = init_indices(dummy_rng)
    weights1 = init_weights(dummy_rng)
    indices2 = init_indices(dummy_rng)
    weights2 = init_weights(dummy_rng)

    self.assertEqual(indices1.shape, (num_tokens, k))
    self.assertEqual(indices1.dtype, np.int32)
    self.assertEqual(weights1.shape, (num_tokens, k))
    self.assertEqual(weights1.dtype, np.float32)

    np.testing.assert_array_equal(indices1, indices2)
    np.testing.assert_array_equal(weights1, weights2)

    self.assertTrue(np.all(indices1 >= 0))
    self.assertTrue(np.all(indices1 < num_experts))

    for row in indices1:
      self.assertLen(set(row), k)

    self.assertTrue(np.all(weights1 > 0))
    np.testing.assert_allclose(weights1.sum(axis=-1), 1.0, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
