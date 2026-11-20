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
"""Tests for the KDA base contract and API dispatch."""

import inspect

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from tokamax._src import jaxtyping
from tokamax._src import numerics
from tokamax._src.ops.kda import api
from tokamax._src.ops.kda import reference
from tokamax._src.ops.kda.cp_utils import ContextParallelMetadata


def _l2_normalize(x: jax.Array) -> jax.Array:
  x_f32 = x.astype(jnp.float32)
  rstd = jax.lax.rsqrt(jnp.sum(jnp.square(x_f32), axis=-1) + 1e-6)
  return (x_f32 * rstd[..., None]).astype(x.dtype)


def _make_inputs(
    dtype,
    *,
    heads=3,
    batch=2,
    seq_len=7,
    key_dim=8,
    value_dim=5,
):
  q = jax.ShapeDtypeStruct((heads, batch, seq_len, key_dim), dtype)
  k = jax.ShapeDtypeStruct((heads, batch, seq_len, key_dim), dtype)
  v = jax.ShapeDtypeStruct((heads, batch, seq_len, value_dim), dtype)
  g = jax.ShapeDtypeStruct((heads, batch, seq_len, key_dim), dtype)
  beta = jax.ShapeDtypeStruct((heads, batch, seq_len), dtype)
  initial_state = jax.ShapeDtypeStruct(
      (batch, 1, heads, key_dim, value_dim), jnp.float32
  )
  q, k, v, g, beta, initial_state = numerics.random_initialize(
      (q, k, v, g, beta, initial_state)
  )
  q = jax.nn.silu(q)
  k = jax.nn.silu(k)
  g = -0.1 * jax.nn.softplus(g)
  beta = jax.nn.sigmoid(beta)
  return q, k, v, g, beta, initial_state


class KimiDeltaAttentionTest(parameterized.TestCase):

  def test_public_gdn_parameters(self):
    parameters = inspect.signature(api.kimi_delta_attention).parameters
    self.assertIn("chunk_size", parameters)
    self.assertIn("use_conv1d_in_kernel", parameters)

  def test_public_api_uses_descriptive_names(self):
    parameters = inspect.signature(api.kimi_delta_attention).parameters
    for name in (
        "delta_time_bias",
        "use_qk_l2norm",
        "context_parallel_metadata",
    ):
      self.assertIn(name, parameters)
    for name in (
        "dt_bias",
        "use_qk_l2norm_in_kernel",
        "cp_context",
        "safe_gate",
        "disable_recompute",
    ):
      self.assertNotIn(name, parameters)

  @parameterized.parameters(
      (jnp.bfloat16, jnp.float32),
      (jnp.float16, jnp.float32),
      (jnp.float32, jnp.float32),
      (jnp.float64, jnp.float64),
  )
  def test_accumulator_dtype_uses_float32_as_floor(self, dtype, expected):
    self.assertEqual(reference._accumulator_dtype(dtype), jnp.dtype(expected))

  @parameterized.named_parameters(
      ("bfloat16", jnp.bfloat16, False, False, False),
      ("float32", jnp.float32, False, False, False),
      ("qk_l2norm", jnp.float32, True, False, False),
      ("raw_gate", jnp.float32, False, True, False),
      ("varlen_gate_l2norm", jnp.float32, True, True, True),
  )
  def test_default_implementation_matches_xla(
      self,
      dtype,
      use_qk_l2norm,
      use_gate_in_kernel,
      variable_length,
  ):
    if variable_length:
      q, k, v, g, beta, _ = _make_inputs(
          dtype,
          heads=2,
          batch=2,
          key_dim=64,
          value_dim=64,
      )
      segment_ids = jnp.array(
          [
              [1, 1, 2, 2, 2, 0, 0],
              [1, 2, 2, 3, 3, 3, 0],
          ],
          dtype=jnp.int32,
      )
      initial_state = jnp.zeros((2, 3, 2, 64, 64), dtype=jnp.float32)
      max_num_segments = 3
    else:
      q, k, v, g, beta, initial_state = _make_inputs(
          dtype,
          heads=2,
          batch=1,
          seq_len=64,
          key_dim=64,
          value_dim=64,
      )
      segment_ids = None
      max_num_segments = None

    if not use_qk_l2norm:
      q = _l2_normalize(q)
      k = _l2_normalize(k)
    initial_state = 0.1 * initial_state

    heads, _, _, key_dim = q.shape
    if use_gate_in_kernel:
      a_log = jnp.log(jnp.linspace(1.0, 2.0, heads, dtype=jnp.float32))
      delta_time_bias = jnp.linspace(
          -0.2, 0.2, heads * key_dim, dtype=jnp.float32
      )
    else:
      a_log = delta_time_bias = None

    def call(implementation):
      @jax.jit
      def f(q, k, v, g, beta, initial_state, a_log, delta_time_bias, segment_ids):
        return api.kimi_delta_attention(
            q,
            k,
            v,
            g,
            beta,
            a_log=a_log,
            delta_time_bias=delta_time_bias,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm=use_qk_l2norm,
            use_gate_in_kernel=use_gate_in_kernel,
            segment_ids=segment_ids,
            max_num_segments=max_num_segments,
            implementation=implementation,
        )

      return f(q, k, v, g, beta, initial_state, a_log, delta_time_bias, segment_ids)

    actual = call(None)
    expected = call("xla")

    chex.assert_trees_all_close(actual, expected, atol=0.05, rtol=0.05)

  def test_mosaic_registered(self):
    if "mosaic_tpu" not in api.IMPLEMENTATIONS:
      self.skipTest("mosaic_tpu implementation is not registered.")
    self.assertIn("mosaic_tpu", api.IMPLEMENTATIONS)
    self.assertIsNotNone(api.IMPLEMENTATIONS["mosaic_tpu"].vjp)
    self.assertEqual(api._DEFAULT_IMPLEMENTATIONS, ("mosaic", "xla"))

  def test_no_final_state_by_default(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    output, final_state = api.kimi_delta_attention(
        query=q,
        key=k,
        value=v,
        gate=g,
        beta=beta,
    )
    self.assertEqual(output.shape, v.shape)
    self.assertIsNone(final_state)

  @parameterized.parameters("xla", "mosaic")
  def test_varlen_requires_max_num_segments_without_initial_state(
      self, implementation
  ):
    if implementation == "mosaic" and "mosaic_tpu" not in api.IMPLEMENTATIONS:
      self.skipTest("mosaic_tpu implementation is not registered.")
    shape = (1, 1, 65, 1)
    q = k = v = beta_4d = jnp.ones(shape, dtype=jnp.float32)
    g = jnp.zeros_like(q)
    beta = beta_4d[..., 0]
    segment_ids = jnp.concatenate([
        jnp.ones((1, 20), dtype=jnp.int32),
        jnp.full((1, 20), 2, dtype=jnp.int32),
        jnp.full((1, 25), 3, dtype=jnp.int32),
    ], axis=1)

    with self.assertRaisesRegex(ValueError, "`max_num_segments` is required"):
      api.kimi_delta_attention(
          q,
          k,
          v,
          g,
          beta,
          segment_ids=segment_ids,
          implementation=implementation,
      )

  def test_max_num_segments_must_match_initial_state_segment_dimension(
      self,
  ):
    q, k, v, g, beta, initial_state = _make_inputs(jnp.float32)

    with self.assertRaisesRegex(ValueError, "must match"):
      api.kimi_delta_attention(
          q,
          k,
          v,
          g,
          beta,
          initial_state=initial_state,
          max_num_segments=2,
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
        max_num_segments=1,
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
        max_num_segments=1,
        implementation="xla",
    )

    chex.assert_trees_all_close(final_state, unpadded_final_state)
    chex.assert_trees_all_close(
        output[:, :, 1:], jnp.zeros_like(output[:, :, 1:])
    )

  def test_context_parallel_metadata_does_not_break_public_op_metadata(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("context",))

    output, _ = api.kimi_delta_attention(
        q,
        k,
        v,
        g,
        beta,
        context_parallel_metadata=ContextParallelMetadata(mesh=mesh),
        implementation="xla",
    )

    self.assertEqual(output.shape, v.shape)

  def test_bind_uses_jaxtyping_validation(self):
    q, k, v, g, beta, initial_state = _make_inputs(jnp.float32)
    implementation = api.IMPLEMENTATIONS["xla"]

    with self.assertRaises(jt.TypeCheckError):
      implementation.bind(q[:, :, 0], k, v, g, beta)

    mismatched_inputs = (
        ("key", (q, k[..., :-1], v, g, beta)),
        ("value", (q, k, v[:, :-1], g, beta)),
        ("gate", (q, k, v, g[:, :, :-1], beta)),
        ("beta", (q, k, v, g, beta[..., :-1])),
    )
    for name, inputs in mismatched_inputs:
      with self.subTest(name=name), self.assertRaises(jt.TypeCheckError):
        implementation.bind(*inputs)

    with self.subTest(name="initial_state"), self.assertRaises(
        jt.TypeCheckError
    ):
      implementation.bind(
          q,
          k,
          v,
          g,
          beta,
          initial_state=initial_state[:, :, :, :-1],
      )

    _, batch, seq_len, _ = q.shape
    with self.subTest(name="segment_ids"), self.assertRaises(
        jt.TypeCheckError
    ):
      implementation.bind(
          q,
          k,
          v,
          g,
          beta,
          segment_ids=jnp.ones((batch, seq_len + 1), jnp.int32),
          max_num_segments=1,
      )

  def test_single_token_decode_with_fused_conv1d(self):
    heads, batch, seq_len, dim, conv_width = 3, 1, 1, 128, 4
    dtype = jnp.bfloat16
    q = jnp.ones((heads, batch, seq_len, dim), dtype=dtype) * 0.1
    k = jnp.ones((heads, batch, seq_len, dim), dtype=dtype) * 0.2
    v = jnp.ones((heads, batch, seq_len, dim), dtype=dtype) * 0.3
    gate = jnp.zeros((heads, batch, seq_len, dim), dtype=dtype)
    beta = jnp.full((heads, batch, seq_len), 0.5, dtype=dtype)
    a_log = jnp.zeros((heads,), dtype=jnp.float32)
    dt_bias = jnp.zeros((heads * dim,), dtype=jnp.float32)
    h0 = jnp.ones((batch, 1, heads, dim, dim), dtype=jnp.float32) * 0.05
    cw = jnp.ones((heads, conv_width, dim), dtype=jnp.float32) * 0.25
    segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)

    out, final_state = api.kimi_delta_attention(
        q,
        k,
        v,
        gate,
        beta,
        a_log=a_log,
        delta_time_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        segment_ids=segment_ids,
        max_num_segments=1,
        conv_weight_q=cw,
        conv_weight_k=cw,
        conv_weight_v=cw,
        use_conv1d_in_kernel=True,
        implementation="xla",
    )
    self.assertEqual(out.shape, (heads, batch, seq_len, dim))
    self.assertIsNotNone(final_state)
    assert final_state is not None
    self.assertEqual(final_state.shape, (batch, 1, heads, dim, dim))

  def test_unsupported_implementation(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    with self.assertRaisesRegex(ValueError, "Unknown implementation"):
      with jaxtyping.disable_jaxtyping():
        api.kimi_delta_attention(
            q, k, v, g, beta, implementation="unknown"
        )


class GatedDeltaNetTest(parameterized.TestCase):

  def test_public_gdn_signature(self):
    params = inspect.signature(api.gated_delta_net).parameters
    for name in (
        "query",
        "key",
        "value",
        "gate",
        "beta",
        "qkv",
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
        "conv_weight",
        "conv_bias",
        "a_log",
        "delta_time_bias",
        "initial_state",
        "output_final_state",
        "use_qk_l2norm",
        "use_gate_in_kernel",
        "use_beta_sigmoid_in_kernel",
        "chunk_size",
        "segment_ids",
        "context_parallel_metadata",
        "residual_checkpoint_name",
    ):
      self.assertIn(name, params)

  def test_xla_gdn_gqa_and_fused_qkv_parity(self):
    b, t, h_qk, h_v, k_dim, v_dim = 2, 16, 2, 8, 16, 16
    c_qkv = 2 * h_qk * k_dim + h_v * v_dim
    keys = jax.random.split(jax.random.PRNGKey(0), 7)
    qkv = jax.random.normal(keys[0], (b, t, c_qkv), jnp.float32)
    conv_w = jax.random.normal(keys[1], (4, c_qkv), jnp.float32) * 0.2
    conv_b = jax.random.normal(keys[2], (c_qkv,), jnp.float32) * 0.1
    gate_raw = jax.random.normal(keys[3], (b, t, h_v), jnp.float32)
    beta_raw = jax.random.normal(keys[4], (b, t, h_v), jnp.float32)
    a_log = jnp.log(jax.random.uniform(keys[5], (h_v,), minval=0.1, maxval=1.0))
    dt_bias = jax.random.normal(keys[6], (h_v,), jnp.float32) * 0.1
    h0 = jnp.ones((b, h_v, k_dim, v_dim), jnp.float32) * 0.05

    out_fused, ht_fused = api.gated_delta_net(
        qkv=qkv,
        gate=gate_raw,
        beta=beta_raw,
        num_k_heads=h_qk,
        num_v_heads=h_v,
        head_k_dim=k_dim,
        head_v_dim=v_dim,
        conv_weight=conv_w,
        conv_bias=conv_b,
        a_log=a_log,
        delta_time_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        implementation="xla",
    )
    q_s = qkv[:, :, : h_qk * k_dim].reshape(b, t, h_qk, k_dim)
    k_s = qkv[:, :, h_qk * k_dim : 2 * h_qk * k_dim].reshape(b, t, h_qk, k_dim)
    v_s = qkv[:, :, 2 * h_qk * k_dim :].reshape(b, t, h_v, v_dim)
    out_split, ht_split = api.gated_delta_net(
        q_s,
        k_s,
        v_s,
        gate_raw,
        beta_raw,
        conv_weight=conv_w,
        conv_bias=conv_b,
        a_log=a_log,
        delta_time_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        implementation="xla",
    )
    chex.assert_trees_all_close(
        (out_fused, ht_fused), (out_split, ht_split), atol=1e-6, rtol=1e-6
    )

  def test_zero_norm_l2norm_zero_gradient(self):
    b, t, h_qk, h_v, d = 1, 8, 2, 4, 16
    keys = jax.random.split(jax.random.PRNGKey(99), 5)
    q_nz = jax.random.normal(keys[0], (b, t, h_qk, d), jnp.float32) * 0.5
    k_nz = jax.random.normal(keys[1], (b, t, h_qk, d), jnp.float32) * 0.5
    # Zero out disjoint token halves so upstream cotangents are non-zero at
    # the zero-norm rows (t < 4 for q, t >= 4 for k).
    q = jnp.concatenate([jnp.zeros_like(q_nz[:, :4]), q_nz[:, 4:]], axis=1)
    k = jnp.concatenate([k_nz[:, :4], jnp.zeros_like(k_nz[:, 4:])], axis=1)
    v = jax.random.normal(keys[2], (b, t, h_v, d), jnp.float32)
    h0 = jax.random.normal(keys[3], (b, h_v, d, d), jnp.float32) * 0.2
    do = jax.random.normal(keys[4], (b, t, h_v, d), jnp.float32)
    gate = jnp.full((b, t, h_v), -0.1, jnp.float32)
    beta = jnp.full((b, t, h_v), 0.5, jnp.float32)

    def loss_fn(q_in, k_in):
      out, _ = api.gated_delta_net(
          q_in,
          k_in,
          v,
          gate,
          beta,
          initial_state=h0,
          use_qk_l2norm=True,
          implementation="xla",
      )
      return jnp.sum(out * do)

    dq, dk = jax.grad(loss_fn, argnums=(0, 1))(q, k)
    np.testing.assert_array_equal(np.asarray(dq[:, :4]), np.zeros_like(dq[:, :4]))
    np.testing.assert_array_equal(np.asarray(dk[:, 4:]), np.zeros_like(dk[:, 4:]))
    self.assertTrue(bool(jnp.all(jnp.isfinite(dq))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(dk))))
    self.assertGreater(float(jnp.max(jnp.abs(dq[:, 4:]))), 0.0)
    self.assertGreater(float(jnp.max(jnp.abs(dk[:, :4]))), 0.0)

  def test_gated_delta_net_bind_validates_split_qkv_and_segment_shapes(self):
    b, t, h_qk, h_v, d = 2, 16, 2, 4, 16
    q = jnp.ones((b, t, h_qk, d), jnp.float32)
    k_bad = jnp.ones((b, t, h_qk + 1, d), jnp.float32)
    v = jnp.ones((b, t, h_v, d), jnp.float32)
    v_bad = jnp.ones((b, t + 1, h_v, d), jnp.float32)
    gate = jnp.full((b, t, h_v), -0.1, jnp.float32)
    beta = jnp.full((b, t, h_v), 0.5, jnp.float32)

    with self.assertRaises(jt.TypeCheckError):
      api.gated_delta_net(q, k_bad, v, gate, beta, implementation="xla")
    with jaxtyping.disable_jaxtyping():
      with self.assertRaisesRegex(ValueError, "key.shape.*must match"):
        api.gated_delta_net(q, k_bad, v, gate, beta, implementation="xla")
      with self.assertRaisesRegex(ValueError, "value.shape.*leading"):
        api.gated_delta_net(q, q, v_bad, gate, beta, implementation="xla")
      with self.assertRaisesRegex(ValueError, "segment_ids.*must have shape"):
        api.gated_delta_net(
            q,
            q,
            v,
            gate,
            beta,
            segment_ids=jnp.ones((b, t + 1), jnp.int32),
            implementation="xla",
        )
      c_qkv = 2 * h_qk * d + h_v * d
      qkv = jnp.ones((b, t, c_qkv), jnp.float32)
      conv_w = jnp.ones((4, c_qkv), jnp.float32)
      with self.assertRaisesRegex(ValueError, "qkv.*must be rank 3"):
        api.gated_delta_net(
            qkv=jnp.ones((b, c_qkv), jnp.float32),
            gate=gate,
            beta=beta,
            num_k_heads=h_qk,
            head_k_dim=d,
            implementation="xla",
        )
      with self.assertRaisesRegex(ValueError, "gate.*must be rank 3"):
        api.gated_delta_net(
            qkv=qkv,
            gate=jnp.ones((b, t), jnp.float32),
            beta=beta,
            num_k_heads=h_qk,
            head_k_dim=d,
            implementation="xla",
        )
      with self.assertRaisesRegex(ValueError, "conv_weight.*must have shape"):
        api.gated_delta_net(
            qkv=qkv,
            gate=gate,
            beta=beta,
            num_k_heads=h_qk,
            head_k_dim=d,
            conv_weight=jnp.ones((0, c_qkv), jnp.float32),
            implementation="xla",
        )
      with self.assertRaisesRegex(ValueError, "conv_bias.*must have shape"):
        api.gated_delta_net(
            qkv=qkv,
            gate=gate,
            beta=beta,
            num_k_heads=h_qk,
            head_k_dim=d,
            conv_weight=conv_w,
            conv_bias=jnp.ones((1,), jnp.float32),
            implementation="xla",
        )

  @parameterized.parameters((1, False), (1, True), (2, False), (2, True))
  def test_short_seq_smaller_than_conv_width_fwd_and_bwd(
      self, seq_len: int, with_segments: bool
  ):
    b, h_qk, h_v, d, w_len = 2, 2, 4, 8, 4
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(10 + seq_len), 5)
    qkv = jax.random.normal(keys[0], (b, seq_len, c_qkv), jnp.float32)
    conv_w = jax.random.normal(keys[1], (w_len, c_qkv), jnp.float32) * 0.2
    conv_b = jax.random.normal(keys[2], (c_qkv,), jnp.float32) * 0.1
    gate = jax.random.normal(keys[3], (b, seq_len, h_v), jnp.float32) - 1.0
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], (b, seq_len, h_v), jnp.float32)
    )
    segment_ids = (
        jnp.ones((b, seq_len), dtype=jnp.int32) if with_segments else None
    )

    def loss_fn(qkv_, conv_w_, conv_b_):
      out, _ = api.gated_delta_net(
          qkv=qkv_,
          gate=gate,
          beta=beta,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          conv_weight=conv_w_,
          conv_bias=conv_b_,
          segment_ids=segment_ids,
          implementation="xla",
      )
      return jnp.sum(out**2)

    loss, (dqkv, dconv_w, dconv_b) = jax.value_and_grad(
        loss_fn, argnums=(0, 1, 2)
    )(qkv, conv_w, conv_b)
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertEqual(dqkv.shape, qkv.shape)
    self.assertEqual(dconv_w.shape, (w_len, c_qkv))
    self.assertEqual(dconv_b.shape, (c_qkv,))
    np.testing.assert_allclose(
        np.asarray(dconv_w[: w_len - seq_len]),
        0.0,
        atol=1e-7,
    )

  def test_initial_state_feeds_first_active_non_unit_segment(self):
    b, h_qk, h_v, d = 1, 2, 4, 8
    keys = jax.random.split(jax.random.PRNGKey(77), 6)
    q = jax.random.normal(keys[0], (b, 4, h_qk, d), jnp.float32)
    k = jax.random.normal(keys[1], (b, 4, h_qk, d), jnp.float32)
    v = jax.random.normal(keys[2], (b, 4, h_v, d), jnp.float32)
    gate = -jax.random.uniform(keys[3], (b, 4, h_v), minval=0.05, maxval=0.3)
    beta = jax.random.uniform(keys[4], (b, 4, h_v), minval=0.2, maxval=0.8)
    h0 = jax.random.normal(keys[5], (b, h_v, d, d), jnp.float32) * 0.5
    segment_ids = jnp.array([[0, 2, 2, 3]], dtype=jnp.int32)

    out, _ = api.gated_delta_net(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=h0,
        segment_ids=segment_ids,
        implementation="xla",
    )
    out_seg2, _ = api.gated_delta_net(
        q[:, 1:3],
        k[:, 1:3],
        v[:, 1:3],
        gate[:, 1:3],
        beta[:, 1:3],
        initial_state=h0,
        implementation="xla",
    )
    out_seg3, _ = api.gated_delta_net(
        q[:, 3:4],
        k[:, 3:4],
        v[:, 3:4],
        gate[:, 3:4],
        beta[:, 3:4],
        initial_state=None,
        implementation="xla",
    )
    np.testing.assert_array_equal(np.asarray(out[:, :1]), 0.0)
    chex.assert_trees_all_close(out[:, 1:3], out_seg2, atol=1e-6, rtol=1e-6)
    chex.assert_trees_all_close(out[:, 3:4], out_seg3, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
