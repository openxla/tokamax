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

"""Unit tests for Gated Delta Net 2 (GDN-2) Pallas kernel."""

import unittest
from absl import logging
from absl.testing import absltest
import torch

from tokamax.experimental.torch_tpu.ops.gdn2 import gdn2 as gdn2_pallas


def naive_chunk_gdn2_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
  """Canonical mathematical reference implementation of chunked GDN-2."""
  b, t, h, k_dim = q.shape
  v_dim = v.shape[-1]
  bt = chunk_size
  nt = t // bt

  q_c = q.reshape(b, nt, bt, h, k_dim).permute(0, 3, 1, 2, 4)
  k_c = k.reshape(b, nt, bt, h, k_dim).permute(0, 3, 1, 2, 4)
  v_c = v.reshape(b, nt, bt, h, v_dim).permute(0, 3, 1, 2, 4)
  g_c = g.reshape(b, nt, bt, h, k_dim).permute(0, 3, 1, 2, 4).cumsum(dim=-2)
  b_c = b_gate.reshape(b, nt, bt, h, k_dim).permute(0, 3, 1, 2, 4)
  w_c = w_gate.reshape(b, nt, bt, h, v_dim).permute(0, 3, 1, 2, 4)

  s = torch.zeros(b, h, k_dim, v_dim, dtype=torch.float32, device=q.device)
  strictly_lower = torch.tril(torch.ones(bt, bt, device=q.device), diagonal=-1)
  causal_mask = torch.tril(torch.ones(bt, bt, device=q.device), diagonal=0)
  eye = torch.eye(bt, device=q.device)

  o_chunks = []
  for i in range(nt):
    qi = q_c[:, :, i]
    ki = k_c[:, :, i]
    vi = v_c[:, :, i]
    gi = g_c[:, :, i]
    bi = b_c[:, :, i]
    wi = w_c[:, :, i]

    g_diff = torch.clamp(gi[:, :, :, None, :] - gi[:, :, None, :, :], max=0.0)
    bk = bi * ki
    decay = torch.exp(g_diff)
    t_lower = (
        -torch.sum(bk[:, :, :, None, :] * ki[:, :, None, :, :] * decay, dim=-1)
        * strictly_lower
    )

    a_inv = torch.linalg.inv(eye - t_lower)
    k_g_b = torch.exp(gi) * ki * bi
    wv = wi * vi
    wwy_i = a_inv @ k_g_b
    u_i = a_inv @ wv

    qg = qi[:, :, :, None, :] * torch.exp(g_diff)
    aqk = torch.sum(qg * ki[:, :, None, :, :], dim=-1) * causal_mask

    v_curr = u_i - torch.matmul(wwy_i, s)
    o_i = torch.matmul(qi * torch.exp(gi), s) + torch.matmul(aqk, v_curr)

    g_last = gi[:, :, -1]
    decayed_k = torch.exp(g_last[:, :, None, :] - gi) * ki
    s_update = torch.matmul(decayed_k.transpose(-1, -2), v_curr)
    s = s * torch.exp(g_last)[:, :, :, None] + s_update
    o_chunks.append(o_i)

  return (
      torch.stack(o_chunks, dim=2)
      .permute(0, 2, 3, 1, 4)
      .reshape(b, t, h, v_dim)
      .to(q.dtype)
  )


class GDN2PallasKernelTest(unittest.TestCase):
  """Tests for GDN-2 Pallas kernel on TPU and fallback devices."""

  def test_forward_shape_and_parity_multi_chunk(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunk_size = 64

    for seq_len in (64, 128, 256):
      b, nheads, head_dim = 2, 4, 16
      torch.manual_seed(42)
      q = (
          torch.randn(
              b, seq_len, nheads, head_dim, device=device, dtype=torch.float32
          )
          * (head_dim**-0.5)
      )
      k = (
          torch.randn(
              b, seq_len, nheads, head_dim, device=device, dtype=torch.float32
          )
          * (head_dim**-0.5)
      )
      v = torch.randn(
          b, seq_len, nheads, head_dim, device=device, dtype=torch.float32
      )
      g = -torch.sigmoid(
          torch.randn(
              b, seq_len, nheads, head_dim, device=device, dtype=torch.float32
          )
      )
      b_mat = torch.sigmoid(
          torch.randn(
              b, seq_len, nheads, head_dim, device=device, dtype=torch.float32
          )
      )
      w_mat = torch.sigmoid(
          torch.randn(
              b, seq_len, nheads, head_dim, device=device, dtype=torch.float32
          )
      )

      actual = gdn2_pallas.gdn2_pallas_delta_rule(
          q, k, v, g, b_mat, w_mat, chunk_size=chunk_size
      )
      expected = naive_chunk_gdn2_ref(
          q, k, v, g, b_mat, w_mat, chunk_size=chunk_size
      )

      self.assertEqual(actual.shape, (b, seq_len, nheads, head_dim))
      torch.testing.assert_close(
          actual.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4
      )
      logging.info(
          f"GDN-2 forward parity test passed for seq_len={seq_len}."
      )

  def test_chunk_size_validation(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(2, 100, 4, 16, device=device)
    k = torch.randn(2, 100, 4, 16, device=device)
    v = torch.randn(2, 100, 4, 16, device=device)
    g = torch.randn(2, 100, 4, 16, device=device)
    b_mat = torch.randn(2, 100, 4, 16, device=device)
    w_mat = torch.randn(2, 100, 4, 16, device=device)

    with self.assertRaises(ValueError):
      gdn2_pallas.gdn2_pallas_delta_rule(
          q, k, v, g, b_mat, w_mat, chunk_size=64
      )

  def test_backward_autograd(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, seq_len, nheads, head_dim = 2, 128, 4, 16
    chunk_size = 64

    torch.manual_seed(42)
    q = (
        torch.randn(b, seq_len, nheads, head_dim, device=device)
        * (head_dim**-0.5)
    ).requires_grad_(True)
    k = (
        torch.randn(b, seq_len, nheads, head_dim, device=device)
        * (head_dim**-0.5)
    ).requires_grad_(True)
    v = torch.randn(
        b, seq_len, nheads, head_dim, device=device, requires_grad=True
    )
    g = (-torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device))).requires_grad_(True)
    b_mat = (torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device))).requires_grad_(True)
    w_mat = (torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device))).requires_grad_(True)

    out = gdn2_pallas.gdn2_pallas_delta_rule(
        q, k, v, g, b_mat, w_mat, chunk_size=chunk_size
    )
    loss = out.sum()
    loss.backward()

    for tensor, name in [
        (q, "q"),
        (k, "k"),
        (v, "v"),
        (g, "g"),
        (b_mat, "b"),
        (w_mat, "w"),
    ]:
      self.assertIsNotNone(tensor.grad, f"{name}.grad should not be None")
      self.assertTrue(
          torch.isfinite(tensor.grad).all(),
          f"{name}.grad contains non-finite values",
      )
    logging.info("GDN-2 multi-chunk backward autograd test passed.")


if __name__ == "__main__":
  absltest.main()