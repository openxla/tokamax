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

"""Unit tests for Raven GSA Pallas TPU operator."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu
from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu_bwd_kernel
from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu_kernel
from tokamax._src.ops.experimental.raven import reference


class PallasTpuRavenGSATest(absltest.TestCase):

  def test_pallas_fwd_matches_reference(self):
    key = jax.random.PRNGKey(123)
    batch_size, seq_len, num_heads, key_dim, num_slots = 2, 128, 4, 16, 8
    chunk_size = 64

    key_q, key_k, key_s, key_g = jax.random.split(key, 4)
    q = jax.random.normal(key_q, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32)
    k = jax.random.normal(key_k, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32)
    s = jax.random.normal(key_s, (batch_size, seq_len, num_heads, num_slots), dtype=jnp.float32)
    g = -jax.nn.sigmoid(jax.random.normal(key_g, (batch_size, seq_len, num_heads, num_slots), dtype=jnp.float32))

    op_instance = pallas_mosaic_tpu.PallasTpuRavenGSA()
    (actual_out, _), _ = op_instance._fwd(
        q, k, s, g, chunk_size=chunk_size
    )
    expected_out, _ = reference.raven_stage1_reference(
        q, k, s, g, chunk_size=chunk_size
    )

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-4, atol=1e-4)

  def test_pallas_bwd_matches_reference(self):
    key = jax.random.PRNGKey(456)
    batch_size, seq_len, num_heads, key_dim, num_slots = 2, 128, 4, 16, 8
    chunk_size = 64

    key_q, key_k, key_s, key_g, key_do = jax.random.split(key, 5)
    q = jax.random.normal(key_q, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32)
    k = jax.random.normal(key_k, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32)
    s = jax.random.normal(key_s, (batch_size, seq_len, num_heads, num_slots), dtype=jnp.float32)
    g = -jax.nn.sigmoid(jax.random.normal(key_g, (batch_size, seq_len, num_heads, num_slots), dtype=jnp.float32))
    do = jax.random.normal(key_do, (batch_size, seq_len, num_heads, num_slots), dtype=jnp.float32)

    def ref_loss(q_, k_, s_, g_):
      out, _ = reference.raven_stage1_reference(q_, k_, s_, g_, chunk_size=chunk_size)
      return jnp.sum(out * do)

    dq_ref, dk_ref, ds_ref, dg_ref = jax.grad(ref_loss, argnums=(0, 1, 2, 3))(q, k, s, g)
    dq, dk, ds, dg, _ = pallas_mosaic_tpu_bwd_kernel.raven_pallas_stage1_bwd(
        q, k, s, g, do, chunk_size=chunk_size
    )

    np.testing.assert_allclose(dq, dq_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dk, dk_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ds, ds_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dg, dg_ref, rtol=1e-4, atol=1e-4)

  def test_pallas_stage2_fwd_and_bwd_matches_reference(self):
    key = jax.random.PRNGKey(789)
    batch_size, seq_len, num_heads, num_slots, val_dim = 2, 128, 4, 8, 16
    chunk_size = 64

    key_qv, key_s, key_v, key_g, key_s0, key_do, key_dsf = jax.random.split(
        key, 7
    )
    qv = jax.nn.softmax(
        jax.random.normal(
            key_qv,
            (batch_size, seq_len, num_heads, num_slots),
            dtype=jnp.float32,
        ),
        axis=-1,
    )
    s = jax.nn.sigmoid(
        jax.random.normal(
            key_s,
            (batch_size, seq_len, num_heads, num_slots),
            dtype=jnp.float32,
        )
    )
    v = jax.random.normal(
        key_v, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    g = (
        -jnp.abs(
            jax.random.normal(
                key_g,
                (batch_size, seq_len, num_heads, num_slots),
                dtype=jnp.float32,
            )
        )
        * 0.1
    )
    s0 = jax.random.normal(
        key_s0, (batch_size, num_heads, num_slots, val_dim), dtype=jnp.float32
    )
    do = jax.random.normal(
        key_do, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    d_s2_final = jax.random.normal(
        key_dsf, (batch_size, num_heads, num_slots, val_dim), dtype=jnp.float32
    )

    def stage2_recurrent_ref(qv_, s_, v_, g_, s0_):
      def step(state, inputs):
        qv_t, s_t, v_t, g_t = inputs
        state_next = (
            state * jnp.exp(g_t)[:, :, :, None]
            + s_t[:, :, :, None] * v_t[:, :, None, :]
        )
        out_t = jnp.einsum(
            "bhm,bhmv->bhv",
            qv_t,
            state_next,
            precision=jax.lax.Precision.HIGHEST,
        )
        return state_next, out_t

      inps = (
          jnp.swapaxes(qv_, 0, 1),
          jnp.swapaxes(s_, 0, 1),
          jnp.swapaxes(v_, 0, 1),
          jnp.swapaxes(g_, 0, 1),
      )
      s_final, outs = jax.lax.scan(step, s0_, inps)
      return jnp.swapaxes(outs, 0, 1), s_final

    expected_out, expected_s2 = stage2_recurrent_ref(qv, s, v, g, s0)
    (actual_out, actual_s2), residuals = (
        pallas_mosaic_tpu_kernel.raven_pallas_stage2_fwd(
            qv,
            s,
            v,
            g,
            chunk_size=chunk_size,
            initial_state=s0,
            output_final_state=True,
            return_residuals=True,
        )
    )
    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_s2, expected_s2, rtol=1e-4, atol=1e-4)

    def ref_loss(qv_, s_, v_, g_, s0_):
      out, s_final = stage2_recurrent_ref(qv_, s_, v_, g_, s0_)
      return jnp.sum(out * do) + jnp.sum(s_final * d_s2_final)

    dqv_ref, ds_ref, dv_ref, dg_ref, ds0_ref = jax.grad(
        ref_loss, argnums=(0, 1, 2, 3, 4)
    )(qv, s, v, g, s0)
    dqv, ds, dv, dg, ds0 = pallas_mosaic_tpu_bwd_kernel.raven_pallas_stage2_bwd(
        qv,
        s,
        v,
        g,
        do,
        grad_final_state=d_s2_final,
        chunk_size=chunk_size,
        initial_state=s0,
        s2_in_c=residuals,
    )
    np.testing.assert_allclose(dqv, dqv_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ds, ds_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dv, dv_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dg, dg_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ds0, ds0_ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
