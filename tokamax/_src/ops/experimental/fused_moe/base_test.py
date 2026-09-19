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


if __name__ == "__main__":
  absltest.main()
