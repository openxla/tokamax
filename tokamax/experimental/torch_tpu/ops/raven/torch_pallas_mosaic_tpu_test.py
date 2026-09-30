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

"""Unit tests for Raven Gated Slot Attention (GSA) PyTorch operator."""

import unittest
from absl import logging
from absl.testing import absltest
import torch
from tokamax.experimental.torch_tpu.ops.raven import raven_gsa as raven_pallas


class RavenPallasKernelTest(unittest.TestCase):
  """Tests for Raven Pallas kernel on TPU and fallback devices."""

  def test_forward_shape_and_parity(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, seq_len, nheads, head_dim = 2, 128, 4, 16
    num_slots, chunk_size = 8, 64

    torch.manual_seed(42)
    q = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32)
    s = torch.randn(b, seq_len, nheads, num_slots, device=device, dtype=torch.float32)
    g = -torch.sigmoid(torch.randn(b, seq_len, nheads, num_slots, device=device))

    actual = raven_pallas.raven_pallas_gsa(q, k, s, g, chunk_size=chunk_size)
    expected = raven_pallas._cpu_fallback(q, k, s, g, chunk_size=chunk_size)

    self.assertEqual(actual.shape, (b, seq_len, nheads, num_slots))
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)
    logging.info("Raven forward shape and parity test passed.")

  def test_chunk_size_validation(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(2, 100, 4, 16, device=device)
    k = torch.randn(2, 100, 4, 16, device=device)
    s = torch.randn(2, 100, 4, 8, device=device)
    g = torch.randn(2, 100, 4, 8, device=device)

    with self.assertRaises(ValueError):
      raven_pallas.raven_pallas_gsa(q, k, s, g, chunk_size=64)

  def test_backward_autograd(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, seq_len, nheads, head_dim = 2, 64, 4, 16
    num_slots, chunk_size = 8, 64

    torch.manual_seed(42)
    q = torch.randn(b, seq_len, nheads, head_dim, device=device, requires_grad=True)
    k = torch.randn(b, seq_len, nheads, head_dim, device=device, requires_grad=True)
    s = torch.randn(b, seq_len, nheads, num_slots, device=device, requires_grad=True)
    g = (-torch.sigmoid(torch.randn(b, seq_len, nheads, num_slots, device=device))).detach().requires_grad_(True)

    out = raven_pallas.raven_pallas_gsa(q, k, s, g, chunk_size=chunk_size)
    loss = out.sum()
    loss.backward()

    for tensor, name in [(q, "q"), (k, "k"), (s, "s"), (g, "g")]:
      self.assertIsNotNone(tensor.grad, f"{name}.grad should not be None")
      self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad contains non-finite values")
    logging.info("Raven backward autograd test passed.")


if __name__ == "__main__":
  absltest.main()
