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
"""Tests for Routing MoE API."""

from typing import Any, cast
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe_op import api
from tokamax._src.ops.experimental.routing_moe_op import base

jax.config.parse_flags_with_absl()


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
    w1_bias: np.ndarray | None = None,
    w2_bias: np.ndarray | None = None,
    activation: str = "silu",
) -> np.ndarray:
  """Computes expected MoE output independently in NumPy float64."""
  tokens, hidden = hidden_states.shape
  inter = w2.shape[1]
  expected = np.zeros((tokens, hidden), dtype=np.float64)
  w1_f64 = w1.astype(np.float64)
  w2_f64 = w2.astype(np.float64)
  for token_idx in range(tokens):
    for expert_idx, weight in zip(
        topk_indices[token_idx], topk_weights[token_idx]
    ):
      acc1 = hidden_states[token_idx].astype(np.float64) @ w1_f64[expert_idx]
      if w1_bias is not None:
        acc1 = acc1 + w1_bias[expert_idx, 0].astype(np.float64)
      act = _numpy_activation(acc1[:inter], acc1[inter:], activation)
      acc2 = act @ w2_f64[expert_idx]
      if w2_bias is not None:
        acc2 = acc2 + w2_bias[expert_idx, 0].astype(np.float64)
      expected[token_idx] += float(weight) * acc2
  return expected


def _mock_callable_implementation(**kwargs):
  return kwargs["hidden_states"]


class ApiTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    api._reset_routing_moe_from_logits_warning()

  @parameterized.product(
      activation=(
          "silu",
          "gelu",
          "gelu_tanh",
          "swigluoai",
          "silu_and_mul_with_clamp",
      ),
      with_bias=(False, True),
      topk=(2,),
  )
  def test_basic_api(self, activation, with_bias, topk):
    tokens = 16
    hidden = 64
    inter = 32
    experts = 4

    rng = np.random.default_rng(0)
    inputs_np = (rng.standard_normal((tokens, hidden)) / 4).astype(np.float32)
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

    with jax.default_matmul_precision("float32"):
      actual = api.routing_moe(
          hidden_states=jnp.asarray(inputs_np),
          topk_weights=jnp.asarray(weights),
          topk_indices=jnp.asarray(chosen),
          w1=jnp.asarray(w1_np),
          w2=jnp.asarray(w2_np),
          w1_bias=w1_bias_jnp,
          w2_bias=w2_bias_jnp,
          activation=activation,
          implementation="xla",
      )

    desired = _numpy_moe_oracle(
        inputs_np,
        weights,
        chosen,
        w1_np,
        w2_np,
        w1_bias=w1_bias_np,
        w2_bias=w2_bias_np,
        activation=activation,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64), desired, rtol=1e-4, atol=1e-4
    )

  def test_tp_axis_xla_raises_not_implemented_error(self):
    args = self._zeros_args()
    with self.assertRaisesRegex(NotImplementedError, "tp_axis"):
      api.routing_moe(**args, tp_axis="tp", implementation="xla")

  def test_default_implementation_cpu_fallback(self):
    tokens = 16
    hidden = 64
    inter = 32
    experts = 4
    topk = 2

    key = jax.random.PRNGKey(0)
    key_x, key_w1, key_w2, key_logits = jax.random.split(key, 4)

    inputs = jax.random.normal(key_x, (tokens, hidden), dtype=jnp.bfloat16)
    w1 = jax.random.normal(
        key_w1, (experts, hidden, 2 * inter), dtype=jnp.bfloat16
    )
    w2 = jax.random.normal(key_w2, (experts, inter, hidden), dtype=jnp.bfloat16)
    router_logits = jax.random.normal(
        key_logits, (tokens, experts), dtype=jnp.float32
    )

    scores = jax.nn.softmax(router_logits, axis=-1)
    topk_weights, topk_indices = lax.top_k(scores, topk)
    topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    topk_indices = topk_indices.astype(jnp.int32)

    if "mosaic_tpu" in api.IMPLEMENTATIONS:
      mosaic_op = api.IMPLEMENTATIONS["mosaic_tpu"]
      self.assertFalse(mosaic_op.supported_on(jax.devices("cpu")[0]))
      with self.assertRaises(NotImplementedError):
        mosaic_op(
            hidden_states=inputs,
            topk_weights=topk_weights,
            topk_indices=topk_indices,
            w1=w1,
            w2=w2,
        )

    actual = api.routing_moe(
        hidden_states=inputs,
        topk_weights=topk_weights,
        topk_indices=topk_indices,
        w1=w1,
        w2=w2,
        implementation=None,
    )

    desired = base.reference.routing_moe_reference(
        hidden_states=inputs,
        topk_weights=topk_weights,
        topk_indices=topk_indices,
        w1=w1,
        w2=w2,
    )

    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        desired.astype(jnp.float32),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_unknown_implementation(self):
    args = self._zeros_args()
    with self.assertRaisesRegex(ValueError, "Unknown implementation"):
      api.routing_moe(
          **args,
          implementation=cast(Any, "unsupported"),
      )

  def _zeros_args(self):
    tokens, hidden, inter, experts = 4, 16, 8, 2
    return dict(
        hidden_states=jnp.zeros((tokens, hidden), jnp.bfloat16),
        topk_weights=jnp.zeros((tokens, 2), jnp.float32),
        topk_indices=jnp.zeros((tokens, 2), jnp.int32),
        w1=jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16),
        w2=jnp.zeros((experts, inter, hidden), jnp.bfloat16),
    )

  def test_empty_implementation_sequence(self):
    with self.assertRaisesRegex(ValueError, "must not be an empty sequence"):
      api.routing_moe(**self._zeros_args(), implementation=())

  def test_all_implementations_fail(self):
    def unsupported(**kwargs):
      del kwargs
      raise NotImplementedError("unsupported")

    with self.assertRaises(ExceptionGroup) as cm:
      api.routing_moe(
          **self._zeros_args(), implementation=(unsupported, unsupported)
      )
    self.assertLen(cm.exception.exceptions, 2)

  def test_falls_back_to_next_implementation(self):
    def unsupported(**kwargs):
      del kwargs
      raise NotImplementedError("unsupported")

    args = self._zeros_args()
    out = api.routing_moe(**args, implementation=(unsupported, "xla"))
    self.assertEqual(out.shape, args["hidden_states"].shape)

  def test_mesh_is_forwarded_to_op_instances_only(self):
    seen = {}

    class RecordingRoutingMoe(base.RoutingMoe):

      def bind(self, *args, **kwargs):
        ba = super().bind(*args, **kwargs)
        seen["op"] = ba.arguments.get("mesh")
        return ba

    def plain_callable(**kwargs):
      seen["callable"] = kwargs.get("mesh")
      return kwargs["hidden_states"]

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("d",))
    args = self._zeros_args()
    api.routing_moe(**args, mesh=mesh, implementation=[RecordingRoutingMoe()])
    api.routing_moe(**args, mesh=mesh, implementation=[plain_callable])
    self.assertEqual(seen["op"], mesh)
    self.assertIsNone(seen["callable"])

  @parameterized.product(
      scoring_fn=("softmax", "sigmoid"),
      renormalize=(True, False),
  )
  def test_routing_topk(self, scoring_fn, renormalize):
    tokens = 16
    experts = 8
    topk = 2
    key = jax.random.PRNGKey(0)
    router_logits = jax.random.normal(
        key, (tokens, experts), dtype=jnp.float32
    )

    actual_weights, actual_indices = api.routing_topk(
        gating_output=router_logits,
        topk=topk,
        renormalize=renormalize,
        scoring_fn=scoring_fn,
    )

    scores = (
        jax.nn.softmax(router_logits, axis=-1)
        if scoring_fn == "softmax"
        else jax.nn.sigmoid(router_logits)
    )
    expected_weights, expected_indices = lax.top_k(scores, topk)
    if renormalize:
      expected_weights = expected_weights / expected_weights.sum(
          axis=-1, keepdims=True
      )
    expected_indices = expected_indices.astype(jnp.int32)

    np.testing.assert_allclose(
        actual_weights, expected_weights, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(actual_indices, expected_indices)
    self.assertEqual(actual_indices.dtype, jnp.int32)

  @parameterized.product(
      scoring_fn=("softmax", "sigmoid"),
      renormalize=(True, False),
  )
  def test_routing_moe_from_logits_matches_routing_moe(
      self, scoring_fn, renormalize
  ):
    tokens = 16
    hidden = 64
    inter = 32
    experts = 4
    topk = 2

    key = jax.random.PRNGKey(0)
    key_inputs, key_weights1, key_weights2, key_logits = jax.random.split(
        key, 4
    )

    inputs = jax.random.normal(key_inputs, (tokens, hidden), dtype=jnp.bfloat16)
    weights1 = jax.random.normal(
        key_weights1, (experts, hidden, 2 * inter), dtype=jnp.bfloat16
    )
    weights2 = jax.random.normal(
        key_weights2, (experts, inter, hidden), dtype=jnp.bfloat16
    )
    router_logits = jax.random.normal(
        key_logits, (tokens, experts), dtype=jnp.float32
    )

    hand_weights, hand_indices = api.routing_topk(
        gating_output=router_logits,
        topk=topk,
        renormalize=renormalize,
        scoring_fn=scoring_fn,
    )

    actual = api.routing_moe_from_logits(
        hidden_states=inputs,
        gating_output=router_logits,
        w1=weights1,
        w2=weights2,
        topk=topk,
        renormalize=renormalize,
        scoring_fn=scoring_fn,
        implementation="xla",
    )

    expected = api.routing_moe(
        hidden_states=inputs,
        topk_weights=hand_weights,
        topk_indices=hand_indices,
        w1=weights1,
        w2=weights2,
        implementation="xla",
    )

    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        expected.astype(jnp.float32),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_routing_moe_from_logits_forwards_implementation(self):
    tokens, hidden, inter, experts = 4, 16, 8, 2
    inputs = jnp.zeros((tokens, hidden), jnp.bfloat16)
    router_logits = jnp.zeros((tokens, experts), jnp.float32)
    weights1 = jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16)
    weights2 = jnp.zeros((experts, inter, hidden), jnp.bfloat16)

    out = api.routing_moe_from_logits(
        hidden_states=inputs,
        gating_output=router_logits,
        w1=weights1,
        w2=weights2,
        topk=2,
        renormalize=True,
        implementation=[_mock_callable_implementation],
    )
    self.assertEqual(out.shape, inputs.shape)

  def test_routing_topk_and_from_logits_require_topk_and_renormalize(self):
    tokens, hidden, inter, experts = 4, 16, 8, 2
    inputs = jnp.zeros((tokens, hidden), jnp.bfloat16)
    router_logits = jnp.zeros((tokens, experts), jnp.float32)
    weights1 = jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16)
    weights2 = jnp.zeros((experts, inter, hidden), jnp.bfloat16)

    routing_topk_any: Any = api.routing_topk
    routing_from_logits_any: Any = api.routing_moe_from_logits
    with self.assertRaises(TypeError):
      routing_topk_any(router_logits, 2)
    with self.assertRaises(TypeError):
      routing_topk_any(router_logits, renormalize=True)
    with self.assertRaises(TypeError):
      routing_from_logits_any(inputs, router_logits, weights1, weights2)
    with self.assertRaises(TypeError):
      routing_from_logits_any(
          inputs, router_logits, weights1, weights2, topk=2
      )
    with self.assertRaises(TypeError):
      routing_from_logits_any(
          inputs, router_logits, weights1, weights2, renormalize=True
      )
    with self.assertRaisesRegex(NotImplementedError, "routing_topk"):
      api.routing_topk(
          router_logits,
          topk=2,
          renormalize=True,
          scoring_fn="unknown",
      )

  def test_warning_emitted_once(self):
    api._reset_routing_moe_from_logits_warning()
    tokens, hidden, inter, experts = 4, 16, 8, 2
    inputs = jnp.zeros((tokens, hidden), jnp.bfloat16)
    router_logits = jnp.zeros((tokens, experts), jnp.float32)
    weights1 = jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16)
    weights2 = jnp.zeros((experts, inter, hidden), jnp.bfloat16)

    with warnings.catch_warnings(record=True) as caught_warnings:
      warnings.simplefilter("always")
      api.routing_moe_from_logits(
          hidden_states=inputs,
          gating_output=router_logits,
          w1=weights1,
          w2=weights2,
          topk=2,
          renormalize=True,
          implementation="xla",
      )
      self.assertLen(caught_warnings, 1)
      self.assertTrue(
          issubclass(
              caught_warnings[0].category, api.RoutingMoeFromLogitsWarning
          )
      )

      # Second call must not emit the warning again.
      api.routing_moe_from_logits(
          hidden_states=inputs,
          gating_output=router_logits,
          w1=weights1,
          w2=weights2,
          topk=2,
          renormalize=True,
          implementation="xla",
      )
      self.assertLen(caught_warnings, 1)

  def test_warning_can_be_silenced(self):
    api._reset_routing_moe_from_logits_warning()
    tokens, hidden, inter, experts = 4, 16, 8, 2
    inputs = jnp.zeros((tokens, hidden), jnp.bfloat16)
    router_logits = jnp.zeros((tokens, experts), jnp.float32)
    weights1 = jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16)
    weights2 = jnp.zeros((experts, inter, hidden), jnp.bfloat16)

    with warnings.catch_warnings(record=True) as caught_warnings:
      warnings.filterwarnings(
          "ignore", category=api.RoutingMoeFromLogitsWarning
      )
      api.routing_moe_from_logits(
          hidden_states=inputs,
          gating_output=router_logits,
          w1=weights1,
          w2=weights2,
          topk=2,
          renormalize=True,
          implementation="xla",
      )
      self.assertEmpty(caught_warnings)

  def test_routing_moe_never_warns(self):
    api._reset_routing_moe_from_logits_warning()
    args = self._zeros_args()
    with warnings.catch_warnings(record=True) as caught_warnings:
      warnings.simplefilter("always")
      api.routing_moe(**args, implementation="xla")
      self.assertEmpty(caught_warnings)


if __name__ == "__main__":
  absltest.main()
