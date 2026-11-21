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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe_op import api
from tokamax._src.ops.experimental.routing_moe_op import base
from tokamax._src.ops.experimental.routing_moe_op import reference

jax.config.parse_flags_with_absl()


def _unsupported_impl(**kwargs) -> jax.Array:
  del kwargs
  raise NotImplementedError("unsupported")


class _RecordingRoutingMoe(base.RoutingMoe):
  """Test op that records the bound mesh."""

  recorder: dict[str, Any] | None = None

  def bind(self, *args, **kwargs):
    bound_args = super().bind(*args, **kwargs)
    if _RecordingRoutingMoe.recorder is not None:
      _RecordingRoutingMoe.recorder["op"] = bound_args.arguments.get("mesh")
    return bound_args


class _PlainCallable:
  """Test callable that records the passed mesh."""

  def __init__(self, recorder: dict[str, Any]):
    self._recorder = recorder

  def __call__(self, **kwargs) -> jax.Array:
    self._recorder["callable"] = kwargs.get("mesh")
    return kwargs["hidden_states"]


class ApiTest(parameterized.TestCase):

  def test_basic_api(self):
    args = self._non_zero_args()
    actual = api.routing_moe(**args, implementation="xla")
    desired = reference.routing_moe_reference(**args)
    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        desired.astype(jnp.float32),
        rtol=1e-3,
        atol=1e-3,
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
      cpu_device = mock.create_autospec(jax.Device, instance=True)
      cpu_device.platform = "cpu"
      self.assertFalse(mosaic_op.supported_on(cpu_device))
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

    desired = reference.routing_moe_reference(
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

  def _non_zero_args(self, seed: int = 0):
    rng = np.random.default_rng(seed)
    tokens, hidden, inter, experts = 4, 16, 8, 2
    weights = np.full((tokens, 2), 0.5, dtype=np.float32)
    indices = np.array([[0, 1]] * tokens, dtype=np.int32)
    return dict(
        hidden_states=jnp.asarray(
            rng.standard_normal((tokens, hidden), dtype=np.float32),
            dtype=jnp.bfloat16,
        ),
        topk_weights=jnp.asarray(weights, dtype=jnp.float32),
        topk_indices=jnp.asarray(indices, dtype=jnp.int32),
        w1=jnp.asarray(
            rng.standard_normal((experts, hidden, 2 * inter), dtype=np.float32),
            dtype=jnp.bfloat16,
        ),
        w2=jnp.asarray(
            rng.standard_normal((experts, inter, hidden), dtype=np.float32),
            dtype=jnp.bfloat16,
        ),
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
    with self.assertRaises(ExceptionGroup) as cm:
      api.routing_moe(
          **self._zeros_args(),
          implementation=(_unsupported_impl, _unsupported_impl),
      )
    self.assertLen(cm.exception.exceptions, 2)

  def test_falls_back_to_next_implementation(self):
    args = self._non_zero_args()
    out = api.routing_moe(**args, implementation=(_unsupported_impl, "xla"))
    desired = reference.routing_moe_reference(**args)
    np.testing.assert_allclose(
        out.astype(jnp.float32),
        desired.astype(jnp.float32),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_mesh_is_forwarded_to_op_instances_only(self):
    seen = {}
    _RecordingRoutingMoe.recorder = seen
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("d",))
    args = self._zeros_args()
    api.routing_moe(**args, mesh=mesh, implementation=[_RecordingRoutingMoe()])
    api.routing_moe(**args, mesh=mesh, implementation=[_PlainCallable(seen)])
    self.assertEqual(seen["op"], mesh)
    self.assertIsNone(seen["callable"])


if __name__ == "__main__":
  absltest.main()
