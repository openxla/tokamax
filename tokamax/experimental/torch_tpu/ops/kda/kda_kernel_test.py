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

"""Unit tests for Kalman Delta Attention (KDA) Pallas kernel."""

import unittest

from absl import logging
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import torch
from tokamax.experimental.torch_tpu.ops.kda import kda_kernel as kda_pallas


class KDAPallasKernelTest(unittest.TestCase):
  """Tests for KDA Pallas kernel on TPU and fallback devices."""

  def _get_device(self) -> str:
    if hasattr(torch, "tpu") and torch.tpu.is_available():
      return "tpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

  def test_forward_shape_and_parity(self):
    device = self._get_device()
    chunk_size = 64

    for seq_len in (64, 128, 256):
      batch_size, num_heads, head_dim = 2, 4, 16
      torch.manual_seed(42)
      q = (
          torch.randn(
              batch_size,
              seq_len,
              num_heads,
              head_dim,
              device=device,
              dtype=torch.float32,
          )
          * (head_dim**-0.5)
      )
      k = (
          torch.randn(
              batch_size,
              seq_len,
              num_heads,
              head_dim,
              device=device,
              dtype=torch.float32,
          )
          * (head_dim**-0.5)
      )
      v = torch.randn(
          batch_size,
          seq_len,
          num_heads,
          head_dim,
          device=device,
          dtype=torch.float32,
      )
      g = -torch.sigmoid(
          torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
      )
      beta = torch.sigmoid(
          torch.randn(batch_size, seq_len, num_heads, 1, device=device)
      )

      actual = kda_pallas.kda_pallas_delta_rule(
          q, k, v, g, beta, chunk_size=chunk_size
      )
      expected = kda_pallas._execute_cpu_fallback(
          q.cpu(),
          k.cpu(),
          v.cpu(),
          g.cpu(),
          beta.cpu(),
          chunk_size=chunk_size,
      )

      self.assertEqual(actual.shape, (batch_size, seq_len, num_heads, head_dim))
      torch.testing.assert_close(
          actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3
      )
      logging.info(
          "KDA forward shape and parity test passed for seq_len=%d.", seq_len
      )

  def test_chunk_size_validation(self):
    device = self._get_device()
    q = torch.randn(2, 100, 4, 16, device=device)
    k = torch.randn(2, 100, 4, 16, device=device)
    v = torch.randn(2, 100, 4, 16, device=device)
    g = torch.randn(2, 100, 4, 16, device=device)
    beta = torch.randn(2, 100, 4, 1, device=device)

    with self.assertRaises(ValueError):
      kda_pallas.kda_pallas_delta_rule(q, k, v, g, beta, chunk_size=64)

    q_128 = torch.randn(2, 128, 4, 16, device=device)
    k_128 = torch.randn(2, 128, 4, 16, device=device)
    v_128 = torch.randn(2, 128, 4, 16, device=device)
    g_128 = torch.randn(2, 128, 4, 16, device=device)
    beta_128 = torch.randn(2, 128, 4, 1, device=device)
    with self.assertRaises(NotImplementedError):
      kda_pallas.kda_pallas_delta_rule(
          q_128, k_128, v_128, g_128, beta_128, chunk_size=128
      )

  def test_backward_autograd(self):
    device = self._get_device()
    batch_size, seq_len, num_heads, head_dim = 2, 128, 4, 16
    chunk_size = 64

    torch.manual_seed(42)
    q = (
        (
            torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            * (head_dim**-0.5)
        )
        .detach()
        .requires_grad_(True)
    )
    k = (
        (
            torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            * (head_dim**-0.5)
        )
        .detach()
        .requires_grad_(True)
    )
    v = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        requires_grad=True,
    )
    g = (
        -torch.sigmoid(
            torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        )
    ).detach().requires_grad_(True)
    beta = (
        torch.sigmoid(
            torch.randn(batch_size, seq_len, num_heads, 1, device=device)
        )
    ).detach().requires_grad_(True)

    out = kda_pallas.kda_pallas_delta_rule(
        q, k, v, g, beta, chunk_size=chunk_size
    )
    loss = out.sum()
    loss.backward()

    for tensor, name in [
        (q, "q"),
        (k, "k"),
        (v, "v"),
        (g, "g"),
        (beta, "beta"),
    ]:
      self.assertIsNotNone(tensor.grad, f"{name}.grad should not be None")
      self.assertTrue(
          torch.isfinite(tensor.grad).all(),
          f"{name}.grad contains non-finite values",
      )
    logging.info("KDA backward autograd test passed.")

  def test_jax_pallas_fwd_matches_cpu_fallback(self):
    batch_size, seq_len, num_heads, head_dim = 2, 128, 4, 16
    torch.manual_seed(101)
    q_t = torch.randn(batch_size, seq_len, num_heads, head_dim) * (
        head_dim**-0.5
    )
    k_t = torch.randn(batch_size, seq_len, num_heads, head_dim) * (
        head_dim**-0.5
    )
    v_t = torch.randn(batch_size, seq_len, num_heads, head_dim)
    g_t = -torch.sigmoid(torch.randn(batch_size, seq_len, num_heads, head_dim))
    beta_t = torch.sigmoid(torch.randn(batch_size, seq_len, num_heads))

    expected = kda_pallas._execute_cpu_fallback(
        q_t, k_t, v_t, g_t, beta_t, chunk_size=64
    ).numpy()
    actual = np.asarray(
        kda_pallas.kda_scan_pure_jax(
            jnp.asarray(q_t.numpy()),
            jnp.asarray(k_t.numpy()),
            jnp.asarray(v_t.numpy()),
            jnp.asarray(g_t.numpy()),
            jnp.asarray(beta_t.numpy()),
            chunk_size=64,
            scale=1.0,
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-4)

  def test_jax_pallas_bwd_matches_autodiff_reference(self):
    key = jax.random.PRNGKey(2026)
    batch_size, seq_len, num_heads, key_dim, val_dim = 2, 128, 4, 16, 16
    chunk_size = 64

    k0, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 8)
    q = jax.random.normal(
        k0, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    k = jax.random.normal(
        k1, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    v = jax.random.normal(
        k2, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    g = -0.1 * jax.nn.sigmoid(
        jax.random.normal(
            k3, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
        )
    )
    beta = jax.nn.sigmoid(
        jax.random.normal(
            k4, (batch_size, seq_len, num_heads), dtype=jnp.float32
        )
    )
    s0 = jax.random.normal(
        k5, (batch_size, num_heads, key_dim, val_dim), dtype=jnp.float32
    )
    grad_out = jax.random.normal(
        k6, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    grad_final_state = jax.random.normal(
        k7, (batch_size, num_heads, key_dim, val_dim), dtype=jnp.float32
    )

    (_, _), (a_mat_c, merged_out_c, s_in_c) = kda_pallas.kda_pallas_fwd(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        initial_state=s0,
        output_final_state=True,
        return_residuals=True,
    )
    dq, dk, dv, dg, dbeta, ds0 = kda_pallas.kda_pallas_bwd(
        q,
        k,
        v,
        g,
        beta,
        grad_out,
        grad_final_state=grad_final_state,
        chunk_size=chunk_size,
        initial_state=s0,
        a_mat_c=a_mat_c,
        merged_out_c=merged_out_c,
        s_in_c=s_in_c,
    )

    def ref_loss(q_, k_, v_, g_, beta_, s0_):
      num_chunks = seq_len // chunk_size
      scale = key_dim**-0.5
      q_c = kda_pallas._chunkify(q_ * scale, batch_size, num_chunks, chunk_size)
      k_c = kda_pallas._chunkify(k_, batch_size, num_chunks, chunk_size)
      v_c = kda_pallas._chunkify(v_, batch_size, num_chunks, chunk_size)
      g_c = jnp.cumsum(
          kda_pallas._chunkify(g_, batch_size, num_chunks, chunk_size), axis=-2
      )
      beta_c = kda_pallas._chunkify(
          beta_[..., None], batch_size, num_chunks, chunk_size
      )
      row_idx = jnp.arange(chunk_size)[:, None]
      col_idx = jnp.arange(chunk_size)[None, :]
      strictly_lower = (col_idx < row_idx).astype(jnp.float32)
      causal_keep = (col_idx <= row_idx).astype(jnp.float32)

      bk = beta_c * k_c
      t_lower = -kda_pallas._decayed_gram(bk, k_c, g_c, g_c) * strictly_lower
      a_mat = kda_pallas._unit_lower_inverse(t_lower)
      merged_rhs = jnp.concatenate([jnp.exp(g_c) * bk, beta_c * v_c], axis=-1)
      merged_out = jnp.matmul(a_mat, merged_rhs)
      wwy_c = merged_out[..., :key_dim]
      u_c = merged_out[..., key_dim:]
      aqk_c = kda_pallas._decayed_gram(q_c, k_c, g_c, g_c) * causal_keep

      s_curr = s0_
      outs = []
      for c_idx in range(num_chunks):
        q_i = q_c[:, :, c_idx]
        k_i = k_c[:, :, c_idx]
        g_i = g_c[:, :, c_idx]
        wwy_i = wwy_c[:, :, c_idx]
        u_i = u_c[:, :, c_idx]
        aqk_i = aqk_c[:, :, c_idx]
        v_new = u_i - jnp.matmul(wwy_i, s_curr)
        o_i = jnp.matmul(q_i * jnp.exp(g_i), s_curr) + jnp.matmul(aqk_i, v_new)
        outs.append(o_i)
        g_last = g_i[:, :, -1, :]
        decayed_k = jnp.exp(g_last[:, :, None, :] - g_i) * k_i
        s_curr = s_curr * jnp.exp(g_last)[:, :, :, None] + jnp.matmul(
            jnp.swapaxes(decayed_k, -1, -2), v_new
        )
      o_c = jnp.stack(outs, axis=2)
      out = kda_pallas._unchunk(
          o_c, val_dim, batch_size, num_chunks, chunk_size, num_heads
      )
      return jnp.sum(out * grad_out) + jnp.sum(s_curr * grad_final_state)

    ref_dq, ref_dk, ref_dv, ref_dg, ref_dbeta, ref_ds0 = jax.grad(
        ref_loss, argnums=(0, 1, 2, 3, 4, 5)
    )(q, k, v, g, beta, s0)

    np.testing.assert_allclose(dq, ref_dq, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(dk, ref_dk, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(dv, ref_dv, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(dg, ref_dg, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(dbeta, ref_dbeta, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(ds0, ref_ds0, rtol=5e-4, atol=5e-4)

  def test_jax_pallas_ring_two_rank_matches_single_device(self):
    key = jax.random.PRNGKey(99)
    batch_size, seq_len, num_heads, key_dim, val_dim = 2, 256, 4, 16, 16
    chunk_size = 64
    t_local = seq_len // 2
    nt_local = t_local // chunk_size
    scale = key_dim**-0.5

    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(
        k0, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    k = jax.random.normal(
        k1, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    v = jax.random.normal(
        k2, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    g = -0.05 * jax.nn.sigmoid(
        jax.random.normal(
            k3, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
        )
    )
    beta = jax.nn.sigmoid(
        jax.random.normal(
            k4, (batch_size, seq_len, num_heads), dtype=jnp.float32
        )
    )
    do = jax.random.normal(
        k5, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    zero_s = jnp.zeros(
        (batch_size, num_heads, key_dim, val_dim), dtype=jnp.float32
    )

    (out_single, _), (a_mat_s, merged_s, s_in_s) = kda_pallas.kda_pallas_fwd(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        initial_state=zero_s,
        output_final_state=True,
        return_residuals=True,
    )
    dq_s, dk_s, dv_s, dg_s, dbeta_s, _ = kda_pallas.kda_pallas_bwd(
        q,
        k,
        v,
        g,
        beta,
        do,
        chunk_size=chunk_size,
        initial_state=zero_s,
        a_mat_c=a_mat_s,
        merged_out_c=merged_s,
        s_in_c=s_in_s,
    )

    q0, q1 = q[:, :t_local], q[:, t_local:]
    k0_r, k1_r = k[:, :t_local], k[:, t_local:]
    v0, v1 = v[:, :t_local], v[:, t_local:]
    g0, g1 = g[:, :t_local], g[:, t_local:]
    b0, b1 = beta[:, :t_local], beta[:, t_local:]
    do0, do1 = do[:, :t_local], do[:, t_local:]

    def prep_rank(q_r, k_r, v_r, g_r, b_r, do_r):
      return (
          kda_pallas._chunkify(q_r * scale, batch_size, nt_local, chunk_size),
          kda_pallas._chunkify(k_r, batch_size, nt_local, chunk_size),
          kda_pallas._chunkify(v_r, batch_size, nt_local, chunk_size),
          jnp.cumsum(
              kda_pallas._chunkify(g_r, batch_size, nt_local, chunk_size),
              axis=-2,
          ),
          kda_pallas._chunkify(
              b_r[..., None], batch_size, nt_local, chunk_size
          ),
          kda_pallas._chunkify(do_r, batch_size, nt_local, chunk_size),
      )

    q0_c, k0_c, v0_c, g0_c, b0_c, do0_c = prep_rank(
        q0, k0_r, v0, g0, b0, do0
    )
    q1_c, k1_c, v1_c, g1_c, b1_c, do1_c = prep_rank(
        q1, k1_r, v1, g1, b1, do1
    )

    o0_c, s_final_0, _a_rank_0, a_mat_0, merged_0, s_in_0 = (
        kda_pallas._kda_ring_phase1_fwd_pallas(
            q0_c, k0_c, v0_c, g0_c, b0_c, zero_s, jnp.float32, jnp.float32
        )
    )
    _, _s_loc_1, _a_rank_1, a_mat_1, merged_1, _ = (
        kda_pallas._kda_ring_phase1_fwd_pallas(
            q1_c, k1_c, v1_c, g1_c, b1_c, zero_s, jnp.float32, jnp.float32
        )
    )
    o1_c, _s_final_1, s_in_1 = kda_pallas._kda_state_fwd_pallas(
        q1_c, k1_c, g1_c, merged_1, s_final_0, jnp.float32, jnp.float32
    )

    out_ring = jnp.concatenate(
        [
            kda_pallas._unchunk(
                o0_c, val_dim, batch_size, nt_local, chunk_size, num_heads
            ),
            kda_pallas._unchunk(
                o1_c, val_dim, batch_size, nt_local, chunk_size, num_heads
            ),
        ],
        axis=1,
    )
    np.testing.assert_allclose(out_ring, out_single, rtol=5e-4, atol=5e-4)

    ds0_1 = kda_pallas._kda_state_bwd_ds0_pallas(
        q1_c, k1_c, g1_c, merged_1, do1_c, zero_s
    )
    dq0_c, dk0_c, dv0_c, dg0_c, db0_c, _ = kda_pallas._kda_bwd_pallas(
        q0_c, k0_c, v0_c, g0_c, b0_c, a_mat_0, merged_0, s_in_0, do0_c, ds0_1
    )
    dq1_c, dk1_c, dv1_c, dg1_c, db1_c, ds0_1_full = kda_pallas._kda_bwd_pallas(
        q1_c, k1_c, v1_c, g1_c, b1_c, a_mat_1, merged_1, s_in_1, do1_c, zero_s
    )
    np.testing.assert_allclose(ds0_1, ds0_1_full, rtol=5e-4, atol=5e-4)

    def join_grad(g0_chunk, g1_chunk, dim):
      return jnp.concatenate(
          [
              kda_pallas._unchunk(
                  g0_chunk, dim, batch_size, nt_local, chunk_size, num_heads
              ),
              kda_pallas._unchunk(
                  g1_chunk, dim, batch_size, nt_local, chunk_size, num_heads
              ),
          ],
          axis=1,
      )

    np.testing.assert_allclose(
        join_grad(dq0_c, dq1_c, key_dim) * scale, dq_s, rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        join_grad(dk0_c, dk1_c, key_dim), dk_s, rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        join_grad(dv0_c, dv1_c, val_dim), dv_s, rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        join_grad(dg0_c, dg1_c, key_dim), dg_s, rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        join_grad(db0_c, db1_c, 1)[..., 0], dbeta_s, rtol=5e-4, atol=5e-4
    )


if __name__ == "__main__":
  absltest.main()
