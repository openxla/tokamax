# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import jaxtyping
from tokamax._src import numerics
from tokamax._src.ops.kda import api


def _accumulator_dtype(dtype):
  return jnp.float32


def _reference_kda(
    q,
    k,
    v,
    g,
    beta,
    *,
    scale=None,
    initial_state=None,
    output_final_state=False,
):
  acc_dtype = _accumulator_dtype(q.dtype)
  batch, seq_len, heads, key_dim = q.shape
  value_dim = v.shape[-1]
  if scale is None:
    scale = key_dim**-0.5

  q_h = jnp.swapaxes(q, 1, 2).astype(acc_dtype) * scale
  k_h = jnp.swapaxes(k, 1, 2).astype(acc_dtype)
  v_h = jnp.swapaxes(v, 1, 2).astype(acc_dtype)
  g_h = jnp.swapaxes(g, 1, 2).astype(acc_dtype)
  beta_h = jnp.swapaxes(beta, 1, 2).astype(acc_dtype)
  state = jnp.zeros((batch, heads, key_dim, value_dim), dtype=acc_dtype)
  if initial_state is not None:
    state += initial_state.astype(acc_dtype)
  outputs = []

  for t in range(seq_len):
    state = state * jnp.exp(g_h[:, :, t])[..., None]
    prediction = jnp.einsum("bhk,bhkv->bhv", k_h[:, :, t], state)
    residual = v_h[:, :, t] - prediction
    state = state + jnp.einsum(
        "bhk,bhv->bhkv",
        beta_h[:, :, t, None] * k_h[:, :, t],
        residual,
    )
    outputs.append(jnp.einsum("bhk,bhkv->bhv", q_h[:, :, t], state))

  output = jnp.stack(outputs, axis=2)
  output = jnp.swapaxes(output, 1, 2).astype(q.dtype)
  return output, state if output_final_state else None


def _make_inputs(dtype):
  q = jax.ShapeDtypeStruct((2, 7, 3, 8), dtype)
  k = jax.ShapeDtypeStruct((2, 7, 3, 8), dtype)
  v = jax.ShapeDtypeStruct((2, 7, 3, 5), dtype)
  g = jax.ShapeDtypeStruct((2, 7, 3, 8), dtype)
  beta = jax.ShapeDtypeStruct((2, 7, 3), dtype)
  initial_state = jax.ShapeDtypeStruct((2, 3, 8, 5), jnp.float32)
  q, k, v, g, beta, initial_state = numerics.random_initialize(
      (q, k, v, g, beta, initial_state)
  )
  q = jax.nn.silu(q)
  k = jax.nn.silu(k)
  g = -0.1 * jax.nn.softplus(g)
  beta = jax.nn.sigmoid(beta)
  return q, k, v, g, beta, initial_state


class KimiDeltaAttentionTest(parameterized.TestCase):

  @parameterized.parameters(jnp.bfloat16, jnp.float32)
  def test_kimi_delta_attention_matches_reference(self, dtype):
    q, k, v, g, beta, initial_state = _make_inputs(dtype)

    @jax.jit
    def f(q, k, v, g, beta, initial_state):
      return api.kimi_delta_attention(
          q,
          k,
          v,
          g,
          beta,
          initial_state=initial_state,
          output_final_state=True,
      )

    output, final_state = f(q, k, v, g, beta, initial_state)
    ref_output, ref_final_state = _reference_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )

    self.assertEqual(output.shape, v.shape)
    self.assertEqual(output.dtype, q.dtype)
    self.assertEqual(final_state.shape, initial_state.shape)
    self.assertEqual(final_state.dtype, jnp.float32)
    chex.assert_trees_all_close(output, ref_output, atol=0.01, rtol=0.01)
    chex.assert_trees_all_close(
        final_state, ref_final_state, atol=0.01, rtol=0.01
    )

  def test_kimi_delta_attention_gradients_match_reference(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)

    def loss(fn, q, k, v, g, beta):
      output, _ = fn(q, k, v, g, beta)
      return jnp.sum(output * output)

    grad = jax.grad(
        lambda q, k, v, g, beta: loss(api.kimi_delta_attention, q, k, v, g, beta),
        argnums=(0, 1, 2, 3, 4),
    )(q, k, v, g, beta)
    ref_grad = jax.grad(
        lambda q, k, v, g, beta: loss(_reference_kda, q, k, v, g, beta),
        argnums=(0, 1, 2, 3, 4),
    )(q, k, v, g, beta)

    chex.assert_trees_all_close(grad, ref_grad, atol=0.01, rtol=0.01)

  def test_no_final_state_by_default(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    output, final_state = api.kimi_delta_attention(q, k, v, g, beta)
    self.assertEqual(output.shape, v.shape)
    self.assertIsNone(final_state)

  def test_invalid_shape(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    with self.assertRaisesRegex(ValueError, "`k` shape"):
      with jaxtyping.disable_jaxtyping():
        api.kimi_delta_attention(q, k[:, :-1], v, g, beta)

  def test_unsupported_implementation(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    with self.assertRaisesRegex(NotImplementedError, "Only XLA"):
      with jaxtyping.disable_jaxtyping():
        api.kimi_delta_attention(
            q, k, v, g, beta, implementation="mosaic_tpu"
        )


if __name__ == "__main__":
  absltest.main()
