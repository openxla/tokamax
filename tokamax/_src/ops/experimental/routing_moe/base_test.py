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
"""Tests for RoutingMoe base operator and reference implementation."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe import base


class RoutingMoeBaseTest(parameterized.TestCase):

  def test_ep_sharded_sum_matches_full_experts(self) -> None:
    key = jax.random.key(0)
    k_x, k_w1, k_w2, k_g = jax.random.split(key, 4)
    tokens, hidden, inter, experts, topk = 16, 256, 128, 8, 2
    x = jax.random.normal(k_x, (tokens, hidden), dtype=jnp.bfloat16) * 0.2
    w1 = (
        jax.random.normal(
            k_w1, (experts, hidden, 2 * inter), dtype=jnp.bfloat16
        )
        * 0.1
    )
    w2 = (
        jax.random.normal(k_w2, (experts, inter, hidden), dtype=jnp.bfloat16)
        * 0.1
    )
    logits = jax.random.normal(k_g, (tokens, experts), dtype=jnp.float32)
    topk_vals, topk_ids = jax.lax.top_k(logits, topk)
    topk_weights = jax.nn.softmax(topk_vals, axis=-1)

    op = base.RoutingMoe()
    full_out = op(x, topk_weights, topk_ids, w1, w2)

    shard0 = op(
        x,
        topk_weights,
        topk_ids,
        w1[:4],
        w2[:4],
        experts_start=jnp.array(0, dtype=jnp.int32),
        global_num_experts=experts,
    )
    shard1 = op(
        x,
        topk_weights,
        topk_ids,
        w1[4:],
        w2[4:],
        experts_start=jnp.array(4, dtype=jnp.int32),
        global_num_experts=experts,
    )
    np.testing.assert_allclose(
        np.asarray(shard0 + shard1, dtype=np.float32),
        np.asarray(full_out, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )

  def test_skip_padded_tokens_zeros_padded_rows(self) -> None:
    key = jax.random.key(1)
    k_x, k_w1, k_w2 = jax.random.split(key, 3)
    x = jax.random.normal(k_x, (8, 256), dtype=jnp.bfloat16)
    w1 = jax.random.normal(k_w1, (4, 256, 256), dtype=jnp.bfloat16)
    w2 = jax.random.normal(k_w2, (4, 128, 256), dtype=jnp.bfloat16)
    topk_weights = jnp.array(
        [[0.6, 0.4]] * 3 + [[0.0, 0.0]] * 5, dtype=jnp.float32
    )
    topk_ids = jnp.tile(jnp.array([[0, 1]], dtype=jnp.int32), (8, 1))
    op = base.RoutingMoe()
    out = op(x, topk_weights, topk_ids, w1, w2, skip_padded_tokens=True)
    np.testing.assert_array_equal(
        np.asarray(out[3:], dtype=np.float32),
        np.zeros((5, 256), dtype=np.float32),
    )


if __name__ == "__main__":
  absltest.main()
