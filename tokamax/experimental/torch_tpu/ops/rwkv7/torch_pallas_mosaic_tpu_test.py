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

"""Unit tests for RWKV-7 PyTorch operator."""

import unittest
from absl import logging
from absl.testing import absltest
import torch
import torch.nn.functional as F
from tokamax.experimental.torch_tpu.ops.rwkv7 import rwkv7 as rwkv7_pallas


class RWKV7PallasKernelTest(unittest.TestCase):
  """Tests for RWKV-7 operator on TPU and fallback devices."""

  def test_forward_shape_and_parity(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, seq_len, nheads, head_dim = 2, 128, 4, 16
    chunk_size = 64

    torch.manual_seed(42)
    q = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32) * (head_dim ** -0.5)
    k = F.normalize(torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32), p=2, dim=-1)
    v = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32)
    alpha = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32) * (head_dim ** -0.5)
    beta = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32) * (head_dim ** -0.5)
    gk = -torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device))

    actual = rwkv7_pallas.rwkv7_pallas_delta_rule(q, k, v, alpha, beta, gk, chunk_size=chunk_size)
    expected = rwkv7_pallas._cpu_fallback(q, k, v, alpha, beta, gk, chunk_size=chunk_size)

    self.assertEqual(actual.shape, (b, seq_len, nheads, head_dim))
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)
    logging.info("RWKV-7 forward shape and parity test passed.")

  def test_chunk_size_validation(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(2, 100, 4, 16, device=device)
    k = torch.randn(2, 100, 4, 16, device=device)
    v = torch.randn(2, 100, 4, 16, device=device)
    alpha = torch.randn(2, 100, 4, 16, device=device)
    beta = torch.randn(2, 100, 4, 16, device=device)
    gk = torch.randn(2, 100, 4, 16, device=device)

    with self.assertRaises(ValueError):
      rwkv7_pallas.rwkv7_pallas_delta_rule(q, k, v, alpha, beta, gk, chunk_size=64)

  def test_backward_autograd(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, seq_len, nheads, head_dim = 2, 64, 4, 16
    chunk_size = 64

    torch.manual_seed(42)
    q = (torch.randn(b, seq_len, nheads, head_dim, device=device) * (head_dim ** -0.5)).detach().requires_grad_(True)
    k = (F.normalize(torch.randn(b, seq_len, nheads, head_dim, device=device), p=2, dim=-1)).detach().requires_grad_(True)
    v = torch.randn(b, seq_len, nheads, head_dim, device=device, requires_grad=True)
    alpha = (torch.randn(b, seq_len, nheads, head_dim, device=device) * (head_dim ** -0.5)).detach().requires_grad_(True)
    beta = (torch.randn(b, seq_len, nheads, head_dim, device=device) * (head_dim ** -0.5)).detach().requires_grad_(True)
    gk = (-torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device))).detach().requires_grad_(True)

    out = rwkv7_pallas.rwkv7_pallas_delta_rule(q, k, v, alpha, beta, gk, chunk_size=chunk_size)
    loss = out.sum()
    loss.backward()

    for tensor, name in [(q, "q"), (k, "k"), (v, "v"), (alpha, "alpha"), (beta, "beta"), (gk, "gk")]:
      self.assertIsNotNone(tensor.grad, f"{name}.grad should not be None")
      self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad contains non-finite values")
    logging.info("RWKV-7 backward autograd test passed.")


if __name__ == "__main__":
  absltest.main()
