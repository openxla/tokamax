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
"""Tests for Fused MoE API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.fused_moe import api
from tokamax._src.ops.experimental.fused_moe import base

jax.config.parse_flags_with_absl()


class ApiTest(parameterized.TestCase):

  @parameterized.product(
      act_fn=["silu"],
      renormalize=[True, False],
      topk=[2],
      impl=["xla"],
  )
  def test_basic_api(self, act_fn, renormalize, topk, impl):
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

    actual = api.fused_moe(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
        implementation=impl,
    )

    desired, _ = base.FusedMoe()._fwd(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
    )

    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        desired.astype(jnp.float32),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_unknown_implementation(self):
    tokens, hidden, inter, experts = 4, 16, 8, 2
    x = jnp.zeros((tokens, hidden), jnp.bfloat16)
    w1 = jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16)
    w2 = jnp.zeros((experts, inter, hidden), jnp.bfloat16)
    gating = jnp.zeros((tokens, experts), jnp.float32)

    with self.assertRaisesRegex(ValueError, "Unknown implementation"):
      api.fused_moe(
          x=x,
          w1=w1,
          w2=w2,
          gating=gating,
          implementation="unsupported",  # pyrefly: ignore[bad-argument-type]
      )


if __name__ == "__main__":
  absltest.main()
