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

"""Unit tests for RWKV-7 PyTorch Pallas operator."""

import unittest

from absl import logging
from absl.testing import absltest
import jax
import torch
from torch._subclasses import fake_tensor
import torch.nn.functional as F
import torch_tpu  # pylint: disable=unused-import
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.rwkv7 import torch_base
from tokamax.experimental.torch_tpu.ops.rwkv7 import torch_pallas_mosaic_tpu

Config = torch_pallas_mosaic_tpu.Config


class PallasMosaicTpuRwkv7Test(unittest.TestCase):
  """Tests for RWKV-7 Pallas operator against base reference."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  def test_forward_shape_and_parity(self):
    device = "tpu"
    chunk_size = 64

    for seq_len in (64, 128, 256):
      batch_size, num_heads, head_dim = 2, 4, 128
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
      k = F.normalize(
          torch.randn(
              batch_size,
              seq_len,
              num_heads,
              head_dim,
              device=device,
              dtype=torch.float32,
          ),
          p=2,
          dim=-1,
      )
      v = torch.randn(
          batch_size,
          seq_len,
          num_heads,
          head_dim,
          device=device,
          dtype=torch.float32,
      )
      alpha = (
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
      beta = (
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
      gk = -torch.sigmoid(
          torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
      )

      configs = torch_utils.get_configs(
          torch_pallas_mosaic_tpu.PallasMosaicTpuRwkv7,
          q,
          k,
          v,
          alpha,
          beta,
          gk,
          chunk_size=chunk_size,
          from_autotuning_cache=False,
      )
      actual = torch_pallas_mosaic_tpu.PallasMosaicTpuRwkv7(
          q,
          k,
          v,
          alpha,
          beta,
          gk,
          chunk_size=chunk_size,
          configs=configs,
      )
      expected = torch_base.Rwkv7(
          q, k, v, alpha, beta, gk, chunk_size=chunk_size
      )

      self.assertEqual(actual.shape, (batch_size, seq_len, num_heads, head_dim))
      torch.testing.assert_close(
          actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3
      )
      logging.info(
          "RWKV-7 forward shape and parity test passed for seq_len=%d.", seq_len
      )

  def test_chunk_size_validation(self):
    device = "tpu"
    q = torch.randn(2, 100, 4, 128, device=device)
    k = torch.randn(2, 100, 4, 128, device=device)
    v = torch.randn(2, 100, 4, 128, device=device)
    alpha = torch.randn(2, 100, 4, 128, device=device)
    beta = torch.randn(2, 100, 4, 128, device=device)
    gk = torch.randn(2, 100, 4, 128, device=device)

    with self.assertRaises(ValueError):
      torch_pallas_mosaic_tpu.PallasMosaicTpuRwkv7(
          q, k, v, alpha, beta, gk, chunk_size=64
      )

  def test_backward_autograd_matches_reference(self):
    device = "tpu"
    batch_size, seq_len, num_heads, head_dim = 2, 128, 4, 128
    chunk_size = 64

    torch.manual_seed(42)
    q0 = (
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        * (head_dim**-0.5)
    )
    k0 = F.normalize(
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device),
        p=2,
        dim=-1,
    )
    v0 = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    alpha0 = (
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        * (head_dim**-0.5)
    )
    beta0 = (
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        * (head_dim**-0.5)
    )
    gk0 = -torch.sigmoid(
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    )
    do = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    q_pal = q0.clone().detach().requires_grad_(True)
    k_pal = k0.clone().detach().requires_grad_(True)
    v_pal = v0.clone().detach().requires_grad_(True)
    alpha_pal = alpha0.clone().detach().requires_grad_(True)
    beta_pal = beta0.clone().detach().requires_grad_(True)
    gk_pal = gk0.clone().detach().requires_grad_(True)

    out_pal = torch_pallas_mosaic_tpu.PallasMosaicTpuRwkv7(
        q_pal, k_pal, v_pal, alpha_pal, beta_pal, gk_pal, chunk_size=chunk_size
    )
    (out_pal * do).sum().backward()

    q_ref = q0.clone().detach().requires_grad_(True)
    k_ref = k0.clone().detach().requires_grad_(True)
    v_ref = v0.clone().detach().requires_grad_(True)
    alpha_ref = alpha0.clone().detach().requires_grad_(True)
    beta_ref = beta0.clone().detach().requires_grad_(True)
    gk_ref = gk0.clone().detach().requires_grad_(True)

    out_ref = torch_base.Rwkv7(
        q_ref, k_ref, v_ref, alpha_ref, beta_ref, gk_ref, chunk_size=chunk_size
    )
    (out_ref * do).sum().backward()

    for pal_t, ref_t, name in [
        (q_pal, q_ref, "q"),
        (k_pal, k_ref, "k"),
        (v_pal, v_ref, "v"),
        (alpha_pal, alpha_ref, "alpha"),
        (beta_pal, beta_ref, "beta"),
        (gk_pal, gk_ref, "gk"),
    ]:
      self.assertIsNotNone(pal_t.grad, f"{name}.grad should not be None")
      torch.testing.assert_close(
          pal_t.grad.cpu(),
          ref_t.grad.cpu(),
          rtol=1e-3,
          atol=1e-3,
          msg=f"Gradient mismatch for {name}",
      )
    logging.info("RWKV-7 backward autograd parity test passed.")

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      q = torch.empty((2, 128, 4, 128), dtype=torch.float32)
      k = torch.empty((2, 128, 4, 128), dtype=torch.float32)
      v = torch.empty((2, 128, 4, 128), dtype=torch.float32)
      alpha = torch.empty((2, 128, 4, 128), dtype=torch.float32)
      beta = torch.empty((2, 128, 4, 128), dtype=torch.float32)
      gk = torch.empty((2, 128, 4, 128), dtype=torch.float32)
      out = torch_pallas_mosaic_tpu.PallasMosaicTpuRwkv7(
          q, k, v, alpha, beta, gk, chunk_size=64
      )
      self.assertEqual(out.shape, (2, 128, 4, 128))
      self.assertEqual(out.dtype, torch.float32)

  def test_deconstruct_and_reconstruct_config_roundtrip(self):
    fwd_op = torch_pallas_mosaic_tpu.PallasMosaicTpuRwkv7
    bwd_op = fwd_op.backward_op_torch
    self.assertIsNotNone(bwd_op)
    for op_instance in (fwd_op, bwd_op):
      cfg = Config(chunk_size=64)
      deconstructed = op_instance.deconstruct_config(cfg)
      self.assertEqual(deconstructed, (64,))
      reconstructed = op_instance.reconstruct_config(deconstructed)
      self.assertEqual(reconstructed, cfg)


if __name__ == "__main__":
  absltest.main()
