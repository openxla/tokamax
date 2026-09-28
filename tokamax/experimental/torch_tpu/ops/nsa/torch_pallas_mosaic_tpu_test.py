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

"""Unit tests for Native Sparse Attention (NSA) PyTorch operator."""

import unittest
from absl import logging
from absl.testing import absltest
import torch
from tokamax.experimental.torch_tpu.ops.nsa import nsa as nsa_pallas


class NSAPallasKernelTest(unittest.TestCase):
  """Tests for NSA PyTorch operator on TPU and fallback devices."""

  def test_forward_shape_and_parity(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, seq_len, nheads, head_dim = 2, 128, 4, 16
    chunk_size, topk, window, cmp_size = 64, 2, 32, 16

    torch.manual_seed(42)
    q = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(b, seq_len, nheads, head_dim, device=device, dtype=torch.float32)
    gc = torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device))
    gs = torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device))
    gw = torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device))

    actual = nsa_pallas(
        q,
        k,
        v,
        gc,
        gs,
        gw,
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_size,
    )
    expected = nsa_pallas._cpu_fallback(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        gc.transpose(1, 2),
        gs.transpose(1, 2),
        gw.transpose(1, 2),
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_size,
        scale=None,
    ).transpose(1, 2)

    self.assertEqual(actual.shape, (b, seq_len, nheads, head_dim))
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)
    logging.info("NSA forward shape and parity test passed.")

  def test_chunk_size_validation(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(2, 100, 4, 16, device=device)
    k = torch.randn(2, 100, 4, 16, device=device)
    v = torch.randn(2, 100, 4, 16, device=device)

    with self.assertRaises(ValueError):
      nsa_pallas(q, k, v, chunk_size=64)

  def test_backward_autograd(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, seq_len, nheads, head_dim = 2, 64, 4, 16
    chunk_size, topk, window, cmp_size = 64, 1, 32, 16

    torch.manual_seed(42)
    q = torch.randn(b, seq_len, nheads, head_dim, device=device, requires_grad=True)
    k = torch.randn(b, seq_len, nheads, head_dim, device=device, requires_grad=True)
    v = torch.randn(b, seq_len, nheads, head_dim, device=device, requires_grad=True)
    gc = torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device)).detach().requires_grad_(True)
    gs = torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device)).detach().requires_grad_(True)
    gw = torch.sigmoid(torch.randn(b, seq_len, nheads, head_dim, device=device)).detach().requires_grad_(True)

    out = nsa_pallas(
        q,
        k,
        v,
        gc,
        gs,
        gw,
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_size,
    )
    loss = out.sum()
    loss.backward()

    for tensor, name in [(q, "q"), (k, "k"), (v, "v"), (gc, "gc"), (gs, "gs"), (gw, "gw")]:
      self.assertIsNotNone(tensor.grad, f"{name}.grad should not be None")
      self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}.grad contains non-finite values")
    logging.info("NSA backward autograd test passed.")


if __name__ == "__main__":
  absltest.main()
