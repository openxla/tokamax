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

from typing import Any
import unittest
from absl import logging
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.nsa import torch_base
from tokamax.experimental.torch_tpu.ops.nsa import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor

nsa_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuNativeSparseAttention


def _register_cpu_kernels_for_test(op: Any) -> None:
  """Registers CPU stand-in kernels on `op` and its VJP for CPU host tests."""

  def _to_jax(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
      return jnp.asarray(x.detach().cpu().numpy())
    if isinstance(x, list):
      return tuple(x)
    return x

  def _to_torch(x: jax.Array, ref: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(np.array(x, copy=True)).to(
        device=ref.device, dtype=ref.dtype
    )

  def _fwd_cpu_kernel(
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g_cmp: torch.Tensor,
      g_slc: torch.Tensor,
      g_swa: torch.Tensor,
      chunk_size: int = 256,
      topk: int = 4,
      window: int = 128,
      cmp_block_size: int = 64,
      scale: float | None = None,
      config: Any = None,
      bwd_config: Any = None,
  ) -> torch.Tensor:
    out_j = op.op_impl_call(
        _to_jax(q),
        _to_jax(k),
        _to_jax(v),
        _to_jax(g_cmp),
        _to_jax(g_slc),
        _to_jax(g_swa),
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_block_size,
        scale=scale,
        config=_to_jax(config),
        bwd_config=_to_jax(bwd_config),
    )
    return _to_torch(out_j, q)

  op._torch_tokamax_op.register_kernel("cpu")(_fwd_cpu_kernel)  # pylint: disable=protected-access

  bwd_op = op.backward_op_torch
  if bwd_op is not None:

    def _bwd_cpu_kernel(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_cmp: torch.Tensor,
        g_slc: torch.Tensor,
        g_swa: torch.Tensor,
        chunk_size: int = 256,
        topk: int = 4,
        window: int = 128,
        cmp_block_size: int = 64,
        scale: float | None = None,
        config: Any = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
      grads_j = bwd_op.op_impl_call(
          _to_jax(dout),
          _to_jax(q),
          _to_jax(k),
          _to_jax(v),
          _to_jax(g_cmp),
          _to_jax(g_slc),
          _to_jax(g_swa),
          chunk_size=chunk_size,
          topk=topk,
          window=window,
          cmp_block_size=cmp_block_size,
          scale=scale,
          config=_to_jax(config),
      )
      return tuple(
          _to_torch(g_j, ref)
          for g_j, ref in zip(grads_j, (q, k, v, g_cmp, g_slc, g_swa))
      )

    bwd_op._torch_tokamax_op.register_kernel("cpu")(_bwd_cpu_kernel)  # pylint: disable=protected-access


class NSAPallasKernelTest(unittest.TestCase):
  """Tests for NSA PyTorch operator on TPU and fallback devices."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if jax.default_backend() != "tpu":
      _register_cpu_kernels_for_test(torch_base.NativeSparseAttention)
      _register_cpu_kernels_for_test(
          torch_pallas_mosaic_tpu.PallasMosaicTpuNativeSparseAttention
      )

  def _device(self) -> str:
    return "tpu" if jax.default_backend() == "tpu" else "cpu"

  def test_forward_shape_and_parity(self):
    device = self._device()
    batch_size, seq_len, num_heads, head_dim = 2, 128, 4, 16
    chunk_size, topk, window, cmp_size = 64, 2, 32, 16

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
    gc = torch.sigmoid(
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    )
    gs = torch.sigmoid(
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    )
    gw = torch.sigmoid(
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    )

    configs = torch_utils.get_configs(
        nsa_pallas,
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
        from_autotuning_cache=False,
    )
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
        configs=configs,
    )
    expected = torch_base.NativeSparseAttention(
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

    self.assertEqual(actual.shape, (batch_size, seq_len, num_heads, head_dim))
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3
    )
    logging.info("NSA forward shape and parity test passed.")

  def test_chunk_size_validation(self):
    device = self._device()
    q = torch.randn(2, 100, 4, 16, device=device)
    k = torch.randn(2, 100, 4, 16, device=device)
    v = torch.randn(2, 100, 4, 16, device=device)

    with self.assertRaises(ValueError):
      nsa_pallas(q, k, v, chunk_size=64)

    q_cmp = torch.randn(2, 128, 4, 16, device=device)
    with self.assertRaises(ValueError):
      nsa_pallas(q_cmp, q_cmp, q_cmp, chunk_size=64, cmp_block_size=48)

  def test_backward_autograd(self):
    device = self._device()
    batch_size, seq_len, num_heads, head_dim = 2, 64, 4, 16
    chunk_size, topk, window, cmp_size = 64, 1, 32, 16

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
    gc = (
        torch.sigmoid(
            torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        )
        .detach()
        .requires_grad_(True)
    )
    gs = (
        torch.sigmoid(
            torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        )
        .detach()
        .requires_grad_(True)
    )
    gw = (
        torch.sigmoid(
            torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        )
        .detach()
        .requires_grad_(True)
    )

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

    for tensor, name in [
        (q, "q"),
        (k, "k"),
        (v, "v"),
        (gc, "gc"),
        (gs, "gs"),
        (gw, "gw"),
    ]:
      self.assertIsNotNone(tensor.grad, f"{name}.grad should not be None")
      self.assertTrue(
          torch.isfinite(tensor.grad).all(),
          f"{name}.grad contains non-finite values",
      )
    logging.info("NSA backward autograd test passed.")

  def test_deconstruct_and_reconstruct_config_roundtrip(self):
    cfg = torch_pallas_mosaic_tpu.Config(
        chunk_size=128, topk=2, window=64, cmp_block_size=32
    )
    for op_instance in (nsa_pallas, nsa_pallas.backward_op_torch):
      self.assertIsNotNone(op_instance)
      deconstructed = op_instance.deconstruct_config(cfg)
      self.assertEqual(deconstructed, (128, 2, 64, 32))
      reconstructed = op_instance.reconstruct_config(deconstructed)
      self.assertEqual(reconstructed, cfg)
      self.assertIsNone(op_instance.deconstruct_config(None))
      with self.assertRaises(ValueError):
        op_instance.reconstruct_config(None)

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      q = torch.empty((2, 128, 4, 16), dtype=torch.float32)
      k = torch.empty((2, 128, 4, 16), dtype=torch.float32)
      v = torch.empty((2, 128, 4, 16), dtype=torch.float32)
      out = nsa_pallas(q, k, v, chunk_size=64, cmp_block_size=16)
      self.assertEqual(out.shape, (2, 128, 4, 16))
      self.assertEqual(out.dtype, torch.float32)


if __name__ == "__main__":
  absltest.main()
