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

"""Unit tests for Gated Delta Net 2 (GDN-2) PyTorch Pallas operator."""

import unittest

from absl import logging
from absl.testing import absltest
import torch
from tokamax.experimental.torch_tpu.ops.gdn2 import gdn2 as gdn2_pallas
from tokamax.experimental.torch_tpu.ops.gdn2 import torch_base


class GDN2PallasKernelTest(unittest.TestCase):
  """Tests for GDN-2 Pallas kernel against base reference."""

  def _get_device(self) -> str:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
      accel = torch.accelerator.current_accelerator()
      if accel is not None and accel.type in ("tpu", "xla"):
        return accel.type
    return "cpu"

  def test_forward_shape_and_parity_multi_chunk(self):
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
          torch.randn(
              batch_size,
              seq_len,
              num_heads,
              head_dim,
              device=device,
              dtype=torch.float32,
          )
      )
      b_mat = torch.sigmoid(
          torch.randn(
              batch_size,
              seq_len,
              num_heads,
              head_dim,
              device=device,
              dtype=torch.float32,
          )
      )
      w_mat = torch.sigmoid(
          torch.randn(
              batch_size,
              seq_len,
              num_heads,
              head_dim,
              device=device,
              dtype=torch.float32,
          )
      )

      actual = gdn2_pallas(q, k, v, g, b_mat, w_mat, chunk_size=chunk_size)
      expected = torch_base.gated_delta_net_2(
          q, k, v, g, b_mat, w_mat, chunk_size=chunk_size
      )

      self.assertEqual(actual.shape, (batch_size, seq_len, num_heads, head_dim))
      torch.testing.assert_close(
          actual.cpu(), expected.cpu(), rtol=5e-4, atol=5e-4
      )
      logging.info("GDN-2 forward parity test passed for seq_len=%d.", seq_len)

  def test_chunk_size_validation(self):
    device = self._get_device()
    q = torch.randn(2, 100, 4, 16, device=device)
    k = torch.randn(2, 100, 4, 16, device=device)
    v = torch.randn(2, 100, 4, 16, device=device)
    g = torch.randn(2, 100, 4, 16, device=device)
    b_mat = torch.randn(2, 100, 4, 16, device=device)
    w_mat = torch.randn(2, 100, 4, 16, device=device)

    with self.assertRaises(ValueError):
      gdn2_pallas(q, k, v, g, b_mat, w_mat, chunk_size=64)

  def test_backward_autograd_matches_reference(self):
    device = self._get_device()
    batch_size, seq_len, num_heads, head_dim = 2, 128, 4, 16
    chunk_size = 64

    torch.manual_seed(42)
    q0 = (
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        * (head_dim**-0.5)
    )
    k0 = (
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        * (head_dim**-0.5)
    )
    v0 = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    g0 = -torch.sigmoid(
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    )
    b0 = torch.sigmoid(
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    )
    w0 = torch.sigmoid(
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    )
    do = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    q_pal = q0.clone().detach().requires_grad_(True)
    k_pal = k0.clone().detach().requires_grad_(True)
    v_pal = v0.clone().detach().requires_grad_(True)
    g_pal = g0.clone().detach().requires_grad_(True)
    b_pal = b0.clone().detach().requires_grad_(True)
    w_pal = w0.clone().detach().requires_grad_(True)

    out_pal = gdn2_pallas(
        q_pal, k_pal, v_pal, g_pal, b_pal, w_pal, chunk_size=chunk_size
    )
    (out_pal * do).sum().backward()

    q_ref = q0.clone().detach().requires_grad_(True)
    k_ref = k0.clone().detach().requires_grad_(True)
    v_ref = v0.clone().detach().requires_grad_(True)
    g_ref = g0.clone().detach().requires_grad_(True)
    b_ref = b0.clone().detach().requires_grad_(True)
    w_ref = w0.clone().detach().requires_grad_(True)

    out_ref = torch_base.gated_delta_net_2(
        q_ref, k_ref, v_ref, g_ref, b_ref, w_ref, chunk_size=chunk_size
    )
    (out_ref * do).sum().backward()

    for pal_t, ref_t, name in [
        (q_pal, q_ref, "q"),
        (k_pal, k_ref, "k"),
        (v_pal, v_ref, "v"),
        (g_pal, g_ref, "g"),
        (b_pal, b_ref, "b"),
        (w_pal, w_ref, "w"),
    ]:
      self.assertIsNotNone(pal_t.grad, f"{name}.grad should not be None")
      torch.testing.assert_close(
          pal_t.grad.cpu(),
          ref_t.grad.cpu(),
          rtol=1e-3,
          atol=1e-3,
          msg=f"Gradient mismatch for {name}",
      )
    logging.info("GDN-2 backward autograd parity test passed.")


if __name__ == "__main__":
  absltest.main()
