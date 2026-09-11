# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Tests for the pure-JAX/XLA chunked KDA implementation."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp

from tokamax._src.ops.experimental.kda import api
from tokamax._src.ops.experimental.kda import xla_chunked


def _make_inputs(
    *,
    heads: int = 2,
    batch: int = 1,
    seq_len: int = 64,
    key_dim: int = 8,
    value_dim: int = 6,
):
  keys = jax.random.split(jax.random.PRNGKey(0), 6)
  q = 0.1 * jax.random.normal(keys[0], (heads, batch, seq_len, key_dim))
  k = 0.1 * jax.random.normal(keys[1], q.shape)
  v = 0.1 * jax.random.normal(keys[2], (heads, batch, seq_len, value_dim))
  raw_gate = 0.1 * jax.random.normal(keys[3], q.shape)
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], (heads, batch, seq_len)))
  initial_state = 0.01 * jax.random.normal(
      keys[5], (batch, 1, heads, key_dim, value_dim)
  )
  return q, k, v, raw_gate, beta, initial_state


def _call(
    implementation,
    q,
    k,
    v,
    gate,
    beta,
    initial_state,
    *,
    use_gate_in_kernel,
    segment_ids=None,
    max_num_segments=None,
):
  a_log = None
  delta_time_bias = None
  if use_gate_in_kernel:
    a_log = jnp.log(jnp.linspace(1.0, 1.5, q.shape[0]))
    delta_time_bias = jnp.linspace(-0.2, 0.2, q.shape[0] * q.shape[-1])
  else:
    gate = -0.05 * jax.nn.softplus(gate)
  return api.kimi_delta_attention(
      q,
      k,
      v,
      gate,
      beta,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      initial_state=initial_state,
      output_final_state=True,
      use_gate_in_kernel=use_gate_in_kernel,
      segment_ids=segment_ids,
      lower_bound=-5.0 if use_gate_in_kernel else None,
      max_num_segments=max_num_segments,
      implementation=implementation,
  )


class XlaChunkedKimiDeltaAttentionTest(parameterized.TestCase):

  def test_default_execution_config(self):
    implementation = xla_chunked.XlaChunkedKimiDeltaAttention()
    vjp = xla_chunked.XlaChunkedKimiDeltaAttentionVjp()
    expected = xla_chunked.Config(chunk_size=64)

    self.assertEqual(implementation._get_heuristics_config(None), expected)
    self.assertEqual(implementation._get_autotuning_configs(None), {expected})
    self.assertEqual(vjp._get_heuristics_config(None), expected)
    self.assertEqual(vjp._get_autotuning_configs(None), {expected})

  @parameterized.parameters(False, True)
  def test_fixed_forward_matches_recurrence(self, use_gate_in_kernel):
    args = _make_inputs()
    expected = _call("xla", *args, use_gate_in_kernel=use_gate_in_kernel)
    actual = _call("xla_chunked", *args, use_gate_in_kernel=use_gate_in_kernel)

    chex.assert_trees_all_close(actual, expected, atol=2e-4, rtol=2e-4)

  def test_packed_forward_matches_recurrence(self):
    q, k, v, gate, beta, _ = _make_inputs()
    segment_ids = jnp.concatenate(
        [
            jnp.ones((1, 17), dtype=jnp.int32),
            jnp.full((1, 40), 2, dtype=jnp.int32),
            jnp.zeros((1, 7), dtype=jnp.int32),
        ],
        axis=1,
    )
    initial_state = jnp.zeros(
        (1, 2, q.shape[0], q.shape[-1], v.shape[-1]), dtype=jnp.float32
    )

    expected = _call(
        "xla",
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        use_gate_in_kernel=True,
        segment_ids=segment_ids,
        max_num_segments=2,
    )
    actual = _call(
        "xla_chunked",
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        use_gate_in_kernel=True,
        segment_ids=segment_ids,
        max_num_segments=2,
    )

    chex.assert_trees_all_close(actual, expected, atol=3e-4, rtol=3e-4)

  def test_vjp_matches_recurrence(self):
    q, k, v, gate, beta, initial_state = _make_inputs(heads=1, key_dim=4, value_dim=3)
    a_log = jnp.zeros((1,), dtype=jnp.float32)
    delta_time_bias = jnp.zeros((4,), dtype=jnp.float32)

    def loss(implementation, q, k, v, gate, beta, a_log, bias, state):
      output, final_state = api.kimi_delta_attention(
          q,
          k,
          v,
          gate,
          beta,
          a_log=a_log,
          delta_time_bias=bias,
          initial_state=state,
          output_final_state=True,
          use_gate_in_kernel=True,
          lower_bound=-5.0,
          implementation=implementation,
      )
      assert final_state is not None
      return jnp.sum(output.astype(jnp.float32)) + 0.1 * jnp.sum(final_state)

    argnums = tuple(range(1, 9))
    expected = jax.grad(loss, argnums=argnums)(
        "xla", q, k, v, gate, beta, a_log, delta_time_bias, initial_state
    )
    actual = jax.grad(loss, argnums=argnums)(
        "xla_chunked",
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        delta_time_bias,
        initial_state,
    )

    chex.assert_trees_all_close(actual, expected, atol=1e-3, rtol=1e-3)

  def test_packed_vjp_without_final_state_matches_recurrence(self):
    q, k, v, gate, beta, _ = _make_inputs(heads=1, key_dim=4, value_dim=3)
    segment_ids = jnp.concatenate(
        [
            jnp.ones((1, 17), dtype=jnp.int32),
            jnp.full((1, 40), 2, dtype=jnp.int32),
            jnp.zeros((1, 7), dtype=jnp.int32),
        ],
        axis=1,
    )
    a_log = jnp.zeros((1,), dtype=jnp.float32)
    delta_time_bias = jnp.zeros((4,), dtype=jnp.float32)

    def loss(implementation, q, k, v, gate, beta, a_log, bias):
      output, final_state = api.kimi_delta_attention(
          q,
          k,
          v,
          gate,
          beta,
          a_log=a_log,
          delta_time_bias=bias,
          output_final_state=False,
          use_gate_in_kernel=True,
          segment_ids=segment_ids,
          lower_bound=-5.0,
          max_num_segments=2,
          implementation=implementation,
      )
      assert final_state is None
      return jnp.sum(output.astype(jnp.float32))

    argnums = tuple(range(1, 8))
    expected = jax.grad(loss, argnums=argnums)(
        "xla", q, k, v, gate, beta, a_log, delta_time_bias
    )
    actual = jax.grad(loss, argnums=argnums)(
        "xla_chunked", q, k, v, gate, beta, a_log, delta_time_bias
    )

    chex.assert_trees_all_close(actual, expected, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
  absltest.main()
