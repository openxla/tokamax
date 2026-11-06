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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe import api
from tokamax._src.ops.experimental.routing_moe import base

jax.config.parse_flags_with_absl()


class ApiTest(parameterized.TestCase):

  @parameterized.product(
      act_fn=("silu",),
      scoring_fn=("softmax", "sigmoid"),
      renormalize=(True, False),
      topk=(2,),
      impl=("xla",),
  )
  def test_basic_api(self, act_fn, scoring_fn, renormalize, topk, impl):
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

    actual = api.routing_moe(
        x=inputs,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
        scoring_fn=scoring_fn,
        implementation=impl,
    )

    desired, _ = base.RoutingMoe()._fwd(
        x=inputs,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
        scoring_fn=scoring_fn,
    )

    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        desired.astype(jnp.float32),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_default_implementation_cpu_fallback(self):
    tokens = 16
    hidden = 64
    inter = 32
    experts = 4
    topk = 2

    key = jax.random.PRNGKey(0)
    key_x, key_w1, key_w2, key_g = jax.random.split(key, 4)

    inputs = jax.random.normal(key_x, (tokens, hidden), dtype=jnp.bfloat16)
    w1 = jax.random.normal(
        key_w1, (experts, hidden, 2 * inter), dtype=jnp.bfloat16
    )
    w2 = jax.random.normal(key_w2, (experts, inter, hidden), dtype=jnp.bfloat16)
    gating = jax.random.normal(key_g, (tokens, experts), dtype=jnp.float32)

    if "mosaic_tpu" in api.IMPLEMENTATIONS:
      mosaic_op = api.IMPLEMENTATIONS["mosaic_tpu"]
      self.assertFalse(mosaic_op.supported_on(jax.devices()[0]))
      with self.assertRaises(NotImplementedError):
        mosaic_op(
            x=inputs,
            w1=w1,
            w2=w2,
            gating=gating,
            topk=topk,
        )

    actual = api.routing_moe(
        x=inputs,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
        implementation=None,
    )

    desired = base.reference.routing_moe_reference(
        x=inputs,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
    )

    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        desired.astype(jnp.float32),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_unknown_implementation(self):
    tokens, hidden, inter, experts = 4, 16, 8, 2
    inputs = jnp.zeros((tokens, hidden), jnp.bfloat16)
    w1 = jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16)
    w2 = jnp.zeros((experts, inter, hidden), jnp.bfloat16)
    gating = jnp.zeros((tokens, experts), jnp.float32)

    with self.assertRaisesRegex(ValueError, "Unknown implementation"):
      api.routing_moe(
          x=inputs,
          w1=w1,
          w2=w2,
          gating=gating,
          implementation=cast(Any, "unsupported"),
      )

  def _zeros_args(self):
    tokens, hidden, inter, experts = 4, 16, 8, 2
    return dict(
        x=jnp.zeros((tokens, hidden), jnp.bfloat16),
        w1=jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16),
        w2=jnp.zeros((experts, inter, hidden), jnp.bfloat16),
        gating=jnp.zeros((tokens, experts), jnp.float32),
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
    self.assertEqual(out.shape, args["x"].shape)

  def test_mesh_is_forwarded_to_op_instances_only(self):
    seen = {}

    class RecordingRoutingMoe(base.RoutingMoe):

      def bind(self, *args, **kwargs):
        seen["op"] = kwargs.get("mesh")
        return super().bind(*args, **kwargs)

    def plain_callable(**kwargs):
      seen["callable"] = kwargs.get("mesh")
      return kwargs["x"]

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("d",))
    args = self._zeros_args()
    api.routing_moe(**args, mesh=mesh, implementation=[RecordingRoutingMoe()])
    api.routing_moe(**args, mesh=mesh, implementation=[plain_callable])
    self.assertEqual(seen["op"], mesh)
    self.assertIsNone(seen["callable"])


if __name__ == "__main__":
  absltest.main()
