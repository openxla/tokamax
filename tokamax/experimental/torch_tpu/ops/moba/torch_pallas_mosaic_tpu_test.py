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
import jax
from tokamax._src.ops.experimental.moba import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.moba import torch_base
from tokamax.experimental.torch_tpu.ops.moba import torch_pallas_mosaic_tpu
import torch
import torch_tpu  # pylint: disable=unused-import

moba_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuMixtureOfBlockAttention


class MoBAPallasKernelTest(unittest.TestCase):
  """Tests for MoBA PyTorch operator on TPU."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  def test_forward_shape_and_parity(self):
    device = "tpu"
    batch_size, seq_len, num_heads, head_dim = 2, 256, 4, 128
    chunk_size, topk = 128, 2

    torch.manual_seed(42)
    q = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    )
    k = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    )
    v = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    )

    configs = torch_utils.get_configs(
        moba_pallas,
        q,
        k,
        v,
        chunk_size=chunk_size,
        topk=topk,
        from_autotuning_cache=False,
    )
    self.assertEqual(
        configs,
        (
            jax_pallas_mosaic_tpu.Config(chunk_size=chunk_size, topk=topk),
            jax_pallas_mosaic_tpu.Config(chunk_size=chunk_size, topk=topk),
        ),
    )

    actual_out = moba_pallas(
        q, k, v, chunk_size=chunk_size, topk=topk, configs=configs
    )
    expected_out = torch_base.MixtureOfBlockAttention(
        q, k, v, chunk_size=chunk_size, topk=topk
    )

    self.assertEqual(
        actual_out.shape, (batch_size, seq_len, num_heads, head_dim)
    )
    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=1e-3, atol=1e-3
    )
    logging.info("MoBA forward shape and parity test passed.")

  def test_chunk_size_validation(self):
    device = "tpu"
    q = torch.randn(2, 100, 4, 128, device=device)
    k = torch.randn(2, 100, 4, 128, device=device)
    v = torch.randn(2, 100, 4, 128, device=device)

    with self.assertRaises(ValueError):
      moba_pallas(q, k, v, chunk_size=128)

  def test_backward_autograd(self):
    device = "tpu"
    batch_size, seq_len, num_heads, head_dim = 2, 256, 4, 128
    chunk_size, topk = 128, 1

    torch.manual_seed(42)
    q = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        requires_grad=True,
    )

    out = moba_pallas(
        q, k, v, chunk_size=chunk_size, topk=topk
    )
    loss = out.sum()
    loss.backward()

    for tensor, name in [(q, "q"), (k, "k"), (v, "v")]:
      self.assertIsNotNone(tensor.grad, f"{name}.grad should not be None")
      self.assertTrue(
          torch.isfinite(tensor.grad).all(),
          f"{name}.grad contains non-finite values",
      )
    logging.info("MoBA backward autograd test passed.")


if __name__ == "__main__":
  absltest.main()
