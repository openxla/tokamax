# Copyright 2026 Google LLC. All Rights Reserved.
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

"""Unit tests for Raven GSA base operator and API."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.raven import api
from tokamax._src.ops.experimental.raven import base


def _recurrent_two_stage_ref(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    v: jax.Array,
    s1_0: jax.Array,
    s2_0: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Step-by-step recurrent reference for two-stage Raven GSA."""
  q = q * float(q.shape[-1] ** -0.5)

  def step(carry, inputs):
    s1, s2 = carry
    q_t, k_t, s_t, g_t, v_t = inputs
    s1_next = (
        s1 * jnp.exp(g_t)[:, :, None, :]
        + k_t[:, :, :, None] * s_t[:, :, None, :]
    )
    ok_t = jnp.einsum(
        "bhk,bhkm->bhm", q_t, s1_next, precision=jax.lax.Precision.HIGHEST
    )
    qv_t = jax.nn.softmax(ok_t, axis=-1)
    s2_next = (
        s2 * jnp.exp(g_t)[:, :, :, None]
        + s_t[:, :, :, None] * v_t[:, :, None, :]
    )
    out_t = jnp.einsum(
        "bhm,bhmv->bhv", qv_t, s2_next, precision=jax.lax.Precision.HIGHEST
    )
    return (s1_next, s2_next), out_t

  inps = tuple(jnp.swapaxes(x, 0, 1) for x in (q, k, s, g, v))
  (s1_final, s2_final), outs = jax.lax.scan(step, (s1_0, s2_0), inps)
  return jnp.swapaxes(outs, 0, 1), (s1_final, s2_final)


class RavenGSABaseTest(absltest.TestCase):

  def test_base_stage1_matches_reference(self):
    key = jax.random.PRNGKey(42)
    batch_size, seq_len, num_heads, key_dim, num_slots = 2, 128, 4, 16, 8
    chunk_size = 64

    key_q, key_k, key_s, key_g = jax.random.split(key, 4)
    q = jax.random.normal(
        key_q, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    k = jax.random.normal(
        key_k, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    s = jax.random.normal(
        key_s, (batch_size, seq_len, num_heads, num_slots), dtype=jnp.float32
    )
    g = -jax.nn.sigmoid(
        jax.random.normal(
            key_g,
            (batch_size, seq_len, num_heads, num_slots),
            dtype=jnp.float32,
        )
    )

    op_instance = base.RavenGsa()
    (actual_out, _), _ = op_instance._fwd(q, k, s, g, chunk_size=chunk_size)
    expected_out, _ = base.raven_stage1_reference(
        q, k, s, g, chunk_size=chunk_size
    )

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-5, atol=1e-5)

  def test_base_two_stage_matches_recurrent_reference(self):
    key = jax.random.PRNGKey(101)
    batch_size, seq_len, num_heads, key_dim, num_slots, val_dim = (
        2,
        128,
        4,
        16,
        8,
        16,
    )
    chunk_size = 64

    keys = jax.random.split(key, 7)
    q = jax.random.normal(
        keys[0], (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    k = jax.random.normal(
        keys[1], (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    s = jax.nn.sigmoid(
        jax.random.normal(
            keys[2],
            (batch_size, seq_len, num_heads, num_slots),
            dtype=jnp.float32,
        )
    )
    g = (
        -jnp.abs(
            jax.random.normal(
                keys[3],
                (batch_size, seq_len, num_heads, num_slots),
                dtype=jnp.float32,
            )
        )
        * 0.1
    )
    v = jax.random.normal(
        keys[4], (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    s1_0 = jax.random.normal(
        keys[5], (batch_size, num_heads, key_dim, num_slots), dtype=jnp.float32
    )
    s2_0 = jax.random.normal(
        keys[6], (batch_size, num_heads, num_slots, val_dim), dtype=jnp.float32
    )

    actual_out, (actual_s1, actual_s2) = api.raven_gsa(
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        initial_state=(s1_0, s2_0),
        output_final_state=True,
        implementation="xla",
    )
    expected_out, (expected_s1, expected_s2) = _recurrent_two_stage_ref(
        q, k, s, g, v, s1_0, s2_0
    )

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_s1, expected_s1, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_s2, expected_s2, rtol=1e-4, atol=1e-4)

  def test_unaligned_seq_len_raises_value_error(self):
    q = jnp.ones((1, 100, 2, 16), dtype=jnp.float32)
    k = jnp.ones((1, 100, 2, 16), dtype=jnp.float32)
    s = jnp.ones((1, 100, 2, 8), dtype=jnp.float32)
    g = -0.1 * jnp.ones((1, 100, 2, 8), dtype=jnp.float32)
    with self.assertRaises(ValueError):
      base.RavenGsa().bind(q, k, s, g, chunk_size=64)


if __name__ == "__main__":
  absltest.main()
