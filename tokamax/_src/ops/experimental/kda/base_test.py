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
from tokamax._src.ops.experimental.kda import api
from tokamax._src.ops.experimental.kda import reference
from tokamax._src.ops.experimental.kda.cp_utils import ContextParallelMetadata


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

  def test_chunk_size_is_not_public(self):
    self.assertNotIn(
        "chunk_size", inspect.signature(api.kimi_delta_attention).parameters
    )

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

    with self.subTest(name="segment_ids"), self.assertRaises(
        jt.TypeCheckError
    ):
      implementation.bind(
          q,
          k,
          v,
          g,
          beta,
          segment_ids=jnp.ones((q.shape[1], q.shape[2] + 1), jnp.int32),
          max_num_segments=1,
      )

  def test_unsupported_implementation(self):
    q, k, v, g, beta, _ = _make_inputs(jnp.float32)
    with self.assertRaisesRegex(ValueError, "Unknown implementation"):
      with jaxtyping.disable_jaxtyping():
        api.kimi_delta_attention(
            q, k, v, g, beta, implementation="xla_chunked"
        )


if __name__ == "__main__":
  absltest.main()
