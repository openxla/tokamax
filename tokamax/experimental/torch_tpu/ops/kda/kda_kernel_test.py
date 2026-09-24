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

from absl import logging
from absl.testing import absltest
import torch
import unittest
from tokamax.experimental.torch_tpu.ops.kda import kda_kernel as kda_pallas


class KDAPallasKernelTest(unittest.TestCase):
  """Tests for KDA Pallas kernel on TPU and fallback devices."""

  def test_forward_shape_and_parity(self):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    b, seq_len, nheads, head_dim = 2, 128, 4, 16
    chunk_size = 64

    torch.manual_seed(42)
    q = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32) * (head_dim ** -0.5)
    k = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32) * (head_dim ** -0.5)
    v = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32)
    g = -torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device))
    beta = torch.sigmoid(torch.randn(b, seq_len, nheads, 1, device=device))

    actual = kda_pallas.kda_pallas_delta_rule(q, k, v, g, beta, chunk_size=chunk_size)
    expected = kda_pallas._execute_cpu_fallback(q, k, v, g, beta, chunk_size=chunk_size)

    self.assertEqual(actual.shape, (b, seq_len, nheads, head_dim))
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)
    logging.info("KDA forward shape and parity test passed.")

  def test_chunk_size_validation(self):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    q = torch.randn(2, 100, 4, 16, device=device)
    k = torch.randn(2, 100, 4, 16, device=device)
    v = torch.randn(2, 100, 4, 16, device=device)
    g = torch.randn(2, 100, 4, 16, device=device)
    beta = torch.randn(2, 100, 4, 1, device=device)

    with self.assertRaises(ValueError):
      kda_pallas.kda_pallas_delta_rule(q, k, v, g, beta, chunk_size=64)

  def test_backward_autograd(self):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    b, seq_len, nheads, head_dim = 2, 64, 4, 16
    chunk_size = 64

    torch.manual_seed(42)
    q = (torch.randn(b, seq_len, nheads, head_dim, device=device) * (head_dim ** -0.5)).detach().requires_grad_(True)
    k = (torch.randn(b, seq_len, nheads, head_dim, device=device) * (head_dim ** -0.5)).detach().requires_grad_(True)
    v = torch.randn(b, seq_len, nheads, head_dim, device=device, requires_grad=True)
    g = (-torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device))).detach().requires_grad_(True)
    beta = (torch.sigmoid(torch.randn(b, seq_len, nheads, 1, device=device))).detach().requires_grad_(True)

    out = kda_pallas.kda_pallas_delta_rule(q, k, v, g, beta, chunk_size=chunk_size)
    loss = out.sum()
    loss.backward()

    for tensor, name in [(q, "q"), (k, "k"), (v, "v"), (g, "g"), (beta, "beta")]:
      self.assertIsNotNone(tensor.grad, f"{name}.grad should not be None")
      self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad contains non-finite values")
    logging.info("KDA backward autograd test passed.")


if __name__ == "__main__":
  absltest.main()