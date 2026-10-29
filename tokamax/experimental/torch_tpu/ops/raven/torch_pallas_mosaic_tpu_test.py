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
import jax
import torch
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.raven import torch_base
from tokamax.experimental.torch_tpu.ops.raven import torch_pallas_mosaic_tpu


class RavenPallasKernelTest(unittest.TestCase):
  """Tests for Raven Pallas kernel against the PyTorch reference op."""

  def test_no_disallowed_self_attributes(self):
    torch_utils.inspect_for_no_self_attributes(type(torch_base.RavenGsa))
    torch_utils.inspect_for_no_self_attributes(
        type(torch_base.RavenGsa.backward_op_torch)
    )
    torch_utils.inspect_for_no_self_attributes(
        torch_pallas_mosaic_tpu.PallasMosaicTpuRavenGsa
    )
    torch_utils.inspect_for_no_self_attributes(
        torch_pallas_mosaic_tpu.PallasMosaicTpuRavenGsa.backward_op_torch
    )

  def test_forward_stage1_and_two_stage_parity(self):
    device = "tpu" if jax.default_backend() == "tpu" else "cpu"
    batch_size, seq_len, num_heads, head_dim = 2, 128, 4, 16
    num_slots, val_dim, chunk_size = 8, 16, 64

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
    s = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        num_slots,
        device=device,
        dtype=torch.float32,
    )
    g = -torch.sigmoid(
        torch.randn(batch_size, seq_len, num_heads, num_slots, device=device)
    )
    v = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        val_dim,
        device=device,
        dtype=torch.float32,
    )

    fwd_cfg, bwd_cfg = torch_utils.get_configs(
        torch_pallas_mosaic_tpu.PallasMosaicTpuRavenGsa,
        q,
        k,
        s,
        g,
        chunk_size=chunk_size,
        from_autotuning_cache=False,
    )
    actual_s1 = torch_pallas_mosaic_tpu.PallasMosaicTpuRavenGsa(
        q, k, s, g, chunk_size=chunk_size, configs=(fwd_cfg, bwd_cfg)
    )
    expected_s1 = torch_base.RavenGsa(q, k, s, g, chunk_size=chunk_size)
    self.assertEqual(
        actual_s1.shape, (batch_size, seq_len, num_heads, num_slots)
    )
    torch.testing.assert_close(
        actual_s1.cpu(), expected_s1.cpu(), rtol=1e-4, atol=1e-4
    )

    actual_s2 = torch_pallas_mosaic_tpu.PallasMosaicTpuRavenGsa(
        q, k, s, g, v, chunk_size=chunk_size
    )
    expected_s2 = torch_base.RavenGsa(q, k, s, g, v, chunk_size=chunk_size)
    self.assertEqual(actual_s2.shape, (batch_size, seq_len, num_heads, val_dim))
    torch.testing.assert_close(
        actual_s2.cpu(), expected_s2.cpu(), rtol=1e-4, atol=1e-4
    )
    logging.info("Raven forward Stage 1 and 2-stage parity test passed.")

  def test_chunk_size_validation(self):
    device = "tpu" if jax.default_backend() == "tpu" else "cpu"
    batch_size, unaligned_seq_len, num_heads, head_dim = 2, 100, 4, 16
    num_slots = 8
    q = torch.randn(
        batch_size, unaligned_seq_len, num_heads, head_dim, device=device
    )
    k = torch.randn(
        batch_size, unaligned_seq_len, num_heads, head_dim, device=device
    )
    s = torch.randn(
        batch_size, unaligned_seq_len, num_heads, num_slots, device=device
    )
    g = torch.randn(
        batch_size, unaligned_seq_len, num_heads, num_slots, device=device
    )

    with self.assertRaises(ValueError):
      torch_pallas_mosaic_tpu.PallasMosaicTpuRavenGsa(
          q, k, s, g, chunk_size=64
      )

  def test_backward_autograd_matches_reference(self):
    device = "tpu" if jax.default_backend() == "tpu" else "cpu"
    batch_size, seq_len, num_heads, head_dim = 2, 128, 4, 16
    num_slots, val_dim, chunk_size = 8, 16, 64

    torch.manual_seed(42)
    q_init = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device
    )
    k_init = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device
    )
    s_init = torch.sigmoid(
        torch.randn(batch_size, seq_len, num_heads, num_slots, device=device)
    )
    g_init = (
        -torch.abs(
            torch.randn(
                batch_size, seq_len, num_heads, num_slots, device=device
            )
        )
        * 0.1
    )
    v_init = torch.randn(
        batch_size, seq_len, num_heads, val_dim, device=device
    )
    grad_out = torch.randn(
        batch_size, seq_len, num_heads, val_dim, device=device
    )

    q_pal = q_init.clone().detach().requires_grad_(True)
    k_pal = k_init.clone().detach().requires_grad_(True)
    s_pal = s_init.clone().detach().requires_grad_(True)
    g_pal = g_init.clone().detach().requires_grad_(True)
    v_pal = v_init.clone().detach().requires_grad_(True)

    q_ref = q_init.clone().detach().requires_grad_(True)
    k_ref = k_init.clone().detach().requires_grad_(True)
    s_ref = s_init.clone().detach().requires_grad_(True)
    g_ref = g_init.clone().detach().requires_grad_(True)
    v_ref = v_init.clone().detach().requires_grad_(True)

    out_pal = torch_pallas_mosaic_tpu.PallasMosaicTpuRavenGsa(
        q_pal, k_pal, s_pal, g_pal, v_pal, chunk_size=chunk_size
    )
    (out_pal * grad_out).sum().backward()

    out_ref = torch_base.RavenGsa(
        q_ref, k_ref, s_ref, g_ref, v_ref, chunk_size=chunk_size
    )
    (out_ref * grad_out).sum().backward()

    torch.testing.assert_close(
        out_pal.cpu(), out_ref.cpu(), rtol=1e-4, atol=1e-4
    )
    for t_pal, t_ref, name in [
        (q_pal, q_ref, "q"),
        (k_pal, k_ref, "k"),
        (s_pal, s_ref, "s"),
        (g_pal, g_ref, "g"),
        (v_pal, v_ref, "v"),
    ]:
      self.assertIsNotNone(t_pal.grad, f"{name}.grad should not be None")
      self.assertIsNotNone(t_ref.grad, f"{name}_ref.grad should not be None")
      torch.testing.assert_close(
          t_pal.grad.cpu(),
          t_ref.grad.cpu(),
          rtol=1e-4,
          atol=1e-4,
          msg=f"Gradient mismatch for {name}",
      )
    logging.info("Raven backward autograd parity test passed.")


if __name__ == "__main__":
  absltest.main()
