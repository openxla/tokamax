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

"""Unit tests for Mixture of Block Attention (MoBA) PyTorch operator."""

import unittest
from absl import logging
from absl.testing import absltest
import torch
from tokamax.experimental.torch_tpu.ops.moba import moba as moba_pallas


class MoBAPallasKernelTest(unittest.TestCase):
  """Tests for MoBA PyTorch operator on TPU and fallback devices."""

  def test_forward_shape_and_parity(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, seq_len, nheads, head_dim = 2, 128, 4, 16
    chunk_size, topk = 64, 2

    torch.manual_seed(42)
    q = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32)

    actual_out = moba_pallas(
        q, k, v, chunk_size=chunk_size, topk=topk
    )
    expected_out = moba_pallas._cpu_fallback(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), chunk_size=chunk_size, topk=topk
    ).transpose(1, 2)

    self.assertEqual(actual_out.shape, (b, seq_len, nheads, head_dim))
    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=1e-3, atol=1e-3
    )
    logging.info("MoBA forward shape and parity test passed.")

  def test_chunk_size_validation(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(2, 100, 4, 16, device=device)
    k = torch.randn(2, 100, 4, 16, device=device)
    v = torch.randn(2, 100, 4, 16, device=device)

    with self.assertRaises(ValueError):
      moba_pallas(q, k, v, chunk_size=64)

  def test_backward_autograd(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, seq_len, nheads, head_dim = 2, 64, 4, 16
    chunk_size, topk = 64, 1

    torch.manual_seed(42)
    q = torch.randn(b, seq_len, nheads, head_dim, device=device, requires_grad=True)
    k = torch.randn(b, seq_len, nheads, head_dim, device=device, requires_grad=True)
    v = torch.randn(b, seq_len, nheads, head_dim, device=device, requires_grad=True)

    out = moba_pallas(
        q, k, v, chunk_size=chunk_size, topk=topk
    )
    loss = out.sum()
    loss.backward()

    for tensor, name in [(q, "q"), (k, "k"), (v, "v")]:
      self.assertIsNotNone(tensor.grad, f"{name}.grad should not be None")
      self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad contains non-finite values")
    logging.info("MoBA backward autograd test passed.")


if __name__ == "__main__":
  absltest.main()
