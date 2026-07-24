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

import types

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import jaxtyping
from tokamax._src import numerics
from tokamax._src.ops.experimental.kda import api, CPContext  # pylint: disable=g-multiple-import
from tokamax._src.ops.experimental.kda import pallas_tpu


def _accumulator_dtype(dtype):
  del dtype
  return jnp.float32


def _reference_kda(
    q,
    k,
    v,
    g,
    beta,
    *,
  A_log=None,
  dt_bias=None,
  scale=None,
  initial_state=None,
  output_final_state=False,
  use_qk_l2norm_in_kernel=False,
  use_gate_in_kernel=False,
  segment_ids=None,
  lower_bound=None,
):
  acc_dtype = _accumulator_dtype(q.dtype)
  heads, batch, seq_len, key_dim = q.shape
  value_dim = v.shape[-1]
  if scale is None:
    scale = key_dim**-0.5

  if use_gate_in_kernel:
    g_h = g.astype(jnp.float32)
    if dt_bias is not None:
      g_h = g_h + dt_bias.reshape(heads, 1, 1, key_dim)
    A = jnp.exp(A_log.astype(jnp.float32)).reshape(heads, 1, 1, 1)
    if lower_bound is None:
      g = -A * jax.nn.softplus(g_h)
    else:
      g = lower_bound * jax.nn.sigmoid(A * g_h)

  if use_qk_l2norm_in_kernel:
    q_h = q.astype(acc_dtype)
    q_h = q_h * jax.lax.rsqrt(jnp.sum(q_h * q_h, axis=-1, keepdims=True) + 1e-6)
    k_h = k.astype(acc_dtype)
    k_h = k_h * jax.lax.rsqrt(jnp.sum(k_h * k_h, axis=-1, keepdims=True) + 1e-6)
  else:
    q_h = q.astype(acc_dtype)
    k_h = k.astype(acc_dtype)
  q_h = q_h * scale
  v_h = v.astype(acc_dtype)
  g_h = g.astype(acc_dtype)
  beta_h = beta.astype(acc_dtype)
  if initial_state is not None:
    num_states = initial_state.shape[1]
  elif segment_ids is not None:
    num_states = 3
  else:
    num_states = 1
  states = jnp.zeros(
      (batch, num_states, heads, key_dim, value_dim), dtype=acc_dtype
  )
  if initial_state is not None:
    states += initial_state.astype(acc_dtype)
  output = jnp.zeros((heads, batch, seq_len, value_dim), dtype=acc_dtype)

  for h in range(heads):
    for b in range(batch):
      for t in range(seq_len):
        if segment_ids is None:
          idx = jnp.array(0, jnp.int32)
          valid = jnp.array(True)
        else:
          seg = segment_ids[b, t].astype(jnp.int32)
          idx = jnp.clip(seg - 1, 0, num_states - 1)
          valid = (seg > 0) & (seg <= num_states)
        previous_state = states[b, idx, h]
        state = previous_state * jnp.exp(g_h[h, b, t])[:, None]
        prediction = k_h[h, b, t] @ state
        residual = v_h[h, b, t] - prediction
        new_state = state + (
            beta_h[h, b, t] * k_h[h, b, t]
        )[:, None] * residual[None, :]
        out = q_h[h, b, t] @ new_state
        output = output.at[h, b, t].set(
            jnp.where(valid, out, jnp.zeros_like(out))
        )
        states = states.at[b, idx, h].set(
            jnp.where(valid, new_state, previous_state)
        )

  return output.astype(q.dtype), states if output_final_state else None


def _make_inputs(dtype):
  q = jax.ShapeDtypeStruct((3, 2, 7, 8), dtype)
  k = jax.ShapeDtypeStruct((3, 2, 7, 8), dtype)
  v = jax.ShapeDtypeStruct((3, 2, 7, 5), dtype)
  g = jax.ShapeDtypeStruct((3, 2, 7, 8), dtype)
  beta = jax.ShapeDtypeStruct((3, 2, 7), dtype)
  initial_state = jax.ShapeDtypeStruct((2, 1, 3, 8, 5), jnp.float32)
  q, k, v, g, beta, initial_state = numerics.random_initialize(
      (q, k, v, g, beta, initial_state)
  )
  q = jax.nn.silu(q)
  k = jax.nn.silu(k)
  g = -0.1 * jax.nn.softplus(g)
  beta = jax.nn.sigmoid(beta)
  return q, k, v, g, beta, initial_state


def _check_pallas_inputs_support(q, v, **overrides):
  kwargs = dict(
      initial_state=None,
      output_final_state=False,
      segment_ids=None,
      cp_context=None,
      chunk_size=64,
      N_max=None,
  )
  kwargs.update(overrides)
  return pallas_tpu.check_inputs_support(q, v, **kwargs)


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

  def test_varlen_gate_l2norm_matches_reference(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    segment_ids = jnp.array(
        [
            [1, 1, 2, 2, 2, 0, 0],
            [1, 2, 2, 3, 3, 3, 0],
        ],
        dtype=jnp.int32,
    )
    initial_state = jnp.zeros((2, 3, 3, 8, 5), dtype=jnp.float32)
    A_log = jnp.log(jnp.array([1.0, 1.5, 2.0], dtype=jnp.float32))
    dt_bias = jnp.linspace(-0.2, 0.2, 3 * 8, dtype=jnp.float32)

    output, final_state = api.kimi_delta_attention(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        segment_ids=segment_ids,
        safe_gate=False,
        N_max=3,
        implementation="xla",
    )
    ref_output, ref_final_state = _reference_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        segment_ids=segment_ids,
    )

    self.assertEqual(final_state.shape, initial_state.shape)
    chex.assert_trees_all_close(output, ref_output, atol=0.01, rtol=0.01)
    chex.assert_trees_all_close(
        final_state, ref_final_state, atol=0.01, rtol=0.01
    )

  def test_pallas_tpu_registered_and_default_falls_back_to_xla(self):
    q, k, v, g, beta, initial_state = _make_inputs(jnp.float32)
    self.assertIn("pallas_tpu", api.IMPLEMENTATIONS)
    self.assertIsNone(api.IMPLEMENTATIONS["pallas_tpu"].vjp)
    self.assertEqual(api._DEFAULT_IMPLEMENTATIONS, ("pallas_tpu", "xla"))

    output, final_state = api.kimi_delta_attention(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    ref_output, ref_final_state = api.kimi_delta_attention(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        implementation="xla",
    )

    chex.assert_trees_all_close(output, ref_output, atol=0.01, rtol=0.01)
    chex.assert_trees_all_close(
        final_state, ref_final_state, atol=0.01, rtol=0.01
    )

  def test_pallas_tpu_alone_is_not_supported_in_cpu_test_shape(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    with self.assertRaisesRegex(
        NotImplementedError, "Not supported|requires the sequence length"
    ):
      api.kimi_delta_attention(q, k, v, g, beta, implementation="pallas_tpu")

  def test_pallas_tpu_rejects_large_key_dimension_before_kernel(self):
    q = jnp.ones((1, 1, 64, 257), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, 1), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "up to 256"):
      _check_pallas_inputs_support(q, v)

  @parameterized.parameters((0, 1), (1, 0))
  def test_pallas_tpu_rejects_empty_kv_dimension_before_kernel(
      self, key_dim, value_dim
  ):
    q = jnp.ones((1, 1, 64, key_dim), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, value_dim), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "positive key and value"):
      _check_pallas_inputs_support(q, v)

  @parameterized.parameters((0, 1, 64), (1, 0, 64), (1, 1, 0))
  def test_pallas_tpu_rejects_empty_grid_dimension_before_kernel(
      self, heads, batch, seq_len
  ):
    q = jnp.ones((heads, batch, seq_len, 1), dtype=jnp.float32)
    v = jnp.ones((heads, batch, seq_len, 1), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "positive head, batch"):
      _check_pallas_inputs_support(q, v)

  def test_pallas_tpu_rejects_multiple_fixed_states_before_kernel(self):
    q = jnp.ones((1, 1, 64, 1), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, 1), dtype=jnp.float32)
    initial_state = jnp.zeros((1, 2, 1, 1, 1), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "exactly one"):
      _check_pallas_inputs_support(q, v, initial_state=initial_state)

  @parameterized.parameters((64, 128), (128, 64))
  def test_pallas_tpu_rejects_unaligned_cp_dimensions_before_kernel(
      self, key_dim, value_dim
  ):
    q = jnp.ones((1, 1, 64, key_dim), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, value_dim), dtype=jnp.float32)
    segment_ids = jnp.ones((1, 64), dtype=jnp.int32)
    cp_context = CPContext(mesh=types.SimpleNamespace(shape={"context": 2}))

    with self.assertRaisesRegex(NotImplementedError, "multiples of 128"):
      _check_pallas_inputs_support(
          q,
          v,
          segment_ids=segment_ids,
          cp_context=cp_context,
          N_max=1,
      )

  def test_pallas_tpu_rejects_cp_contract_gaps_before_kernel(self):
    q = jnp.ones((1, 1, 64, 128), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, 128), dtype=jnp.float32)
    segment_ids = jnp.ones((1, 64), dtype=jnp.int32)
    initial_state = jnp.zeros((1, 1, 1, 128, 128), dtype=jnp.float32)
    cp_context = CPContext(mesh=types.SimpleNamespace(shape={"context": 2}))
    cases = (
        (
            "initial_state",
            dict(
                initial_state=initial_state,
                segment_ids=segment_ids,
                N_max=1,
            ),
        ),
        (
            "output_final_state",
            dict(
                output_final_state=True,
                segment_ids=segment_ids,
                N_max=1,
            ),
        ),
        ("segment_ids", dict(N_max=1)),
        ("N_max", dict(segment_ids=segment_ids)),
    )

    for error_fragment, overrides in cases:
      with self.subTest(error_fragment=error_fragment):
        with self.assertRaisesRegex(NotImplementedError, error_fragment):
          _check_pallas_inputs_support(
              q,
              v,
              cp_context=cp_context,
              **overrides,
          )

  def test_no_final_state_by_default(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    output, final_state = api.kimi_delta_attention(q, k, v, g, beta)
    self.assertEqual(output.shape, v.shape)
    self.assertIsNone(final_state)

  def test_varlen_requires_n_max_without_initial_state(self):
    shape = (1, 1, 65, 1)
    q = k = v = beta_4d = jnp.ones(shape, dtype=jnp.float32)
    g = jnp.zeros_like(q)
    beta = beta_4d[..., 0]
    segment_ids = jnp.concatenate([
        jnp.ones((1, 20), dtype=jnp.int32),
        jnp.full((1, 20), 2, dtype=jnp.int32),
        jnp.full((1, 25), 3, dtype=jnp.int32),
    ], axis=1)

    with self.assertRaisesRegex(ValueError, "`N_max` is required"):
      api.kimi_delta_attention(
          q,
          k,
          v,
          g,
          beta,
          segment_ids=segment_ids,
          implementation="xla",
      )

  def test_n_max_must_match_initial_state_segment_dimension(self):
    q, k, v, g, beta, initial_state = _make_inputs(jnp.float32)

    with self.assertRaisesRegex(ValueError, "must match"):
      api.kimi_delta_attention(
          q,
          k,
          v,
          g,
          beta,
          initial_state=initial_state,
          N_max=2,
          implementation="xla",
      )

  def test_padding_preserves_final_state(self):
    q = k = v = jnp.ones((1, 1, 3, 1), dtype=jnp.float32)
    g = jnp.full_like(q, jnp.log(0.5))
    beta = jnp.ones((1, 1, 3), dtype=jnp.float32)
    segment_ids = jnp.array([[1, 0, 0]], dtype=jnp.int32)

    output, final_state = api.kimi_delta_attention(
        q,
        k,
        v,
        g,
        beta,
        segment_ids=segment_ids,
        output_final_state=True,
        N_max=1,
        implementation="xla",
    )
    _, unpadded_final_state = api.kimi_delta_attention(
        q[:, :, :1],
        k[:, :, :1],
        v[:, :, :1],
        g[:, :, :1],
        beta[:, :, :1],
        segment_ids=segment_ids[:, :1],
        output_final_state=True,
        N_max=1,
        implementation="xla",
    )

    chex.assert_trees_all_close(final_state, unpadded_final_state)
    chex.assert_trees_all_close(
        output[:, :, 1:], jnp.zeros_like(output[:, :, 1:])
    )

  def test_cp_context_does_not_break_public_op_metadata(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("context",))

    output, _ = api.kimi_delta_attention(
        q,
        k,
        v,
        g,
        beta,
        cp_context=CPContext(mesh=mesh),
        implementation="xla",
    )

    self.assertEqual(output.shape, v.shape)

  def test_invalid_shape(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    with self.assertRaisesRegex(ValueError, "`k` shape"):
      with jaxtyping.disable_jaxtyping():
        api.kimi_delta_attention(q, k[:, :, :-1], v, g, beta)

  def test_unsupported_implementation(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    with self.assertRaisesRegex(ValueError, "Unknown implementation"):
      with jaxtyping.disable_jaxtyping():
        api.kimi_delta_attention(
            q, k, v, g, beta, implementation="xla_chunked"
        )


if __name__ == "__main__":
  absltest.main()
