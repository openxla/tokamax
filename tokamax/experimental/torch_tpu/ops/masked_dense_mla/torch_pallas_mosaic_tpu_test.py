# Copyright 2026 Google LLC
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
"""Tests for Pallas Mosaic TPU implementation of MaskedDenseMLA."""

import math
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.masked_dense_mla import base as jax_base
from tokamax._src.ops.masked_dense_mla import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax._src.ops.masked_dense_mla import reference
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.masked_dense_mla import torch_base
from tokamax.experimental.torch_tpu.ops.masked_dense_mla import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import

Config = jax_pallas_mosaic_tpu.Config
ATOL = RTOL = 2e-2
ROPE_DIM = 64
HEAD_DIM = reference.NOPE_DIM + ROPE_DIM


def _make_torch_and_jax_inputs(
    *,
    batch_size: int = 4,
    num_heads: int = 16,
    page_size: int = 32,
    pages_per_seq: int = 8,
    topk: int = 64,
    pad_tokens: bool = False,
    seed: int = 0,
    device: str = "tpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any], int]:
  """Creates PyTorch inputs on `device` and converts them to JAX arrays."""
  torch.manual_seed(seed)
  rng = np.random.default_rng(seed)

  num_decode_seqs = batch_size // 2
  prefill_lens = rng.integers(8, 24, size=(batch_size - num_decode_seqs,))
  q_lens = np.concatenate(
      [np.ones((num_decode_seqs,), dtype=np.int32), prefill_lens]
  )
  cu_q_lens_np = np.concatenate(
      [np.array([0], dtype=np.int32), np.cumsum(q_lens, dtype=np.int32)]
  )
  kv_lens_np = (q_lens + rng.integers(20, 120, size=(batch_size,))).astype(
      np.int32
  )

  actual_tokens = int(np.sum(q_lens))
  total_tokens = (
      ((actual_tokens + 16 + 31) // 32) * 32 if pad_tokens else actual_tokens
  )

  q_torch = torch.randn(
      (total_tokens, num_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
  )

  max_kv_len = pages_per_seq * page_size
  analytic_mode = max_kv_len <= topk
  topk_rows = np.full((total_tokens, topk), -1, dtype=np.int32)
  tok_idx = 0
  for i in range(batch_size):
    kv_len_i = int(kv_lens_np[i])
    q_len_i = int(q_lens[i])
    base_pos = kv_len_i - q_len_i
    for local_q in range(q_len_i):
      q_pos = base_pos + local_q
      if analytic_mode:
        topk_rows[tok_idx, : q_pos + 1] = np.arange(q_pos + 1, dtype=np.int32)
      else:
        n_sel = min(max(1, (q_pos + 1) // 2), topk)
        sel = rng.choice(q_pos + 1, size=n_sel, replace=False)
        topk_rows[tok_idx, :n_sel] = np.sort(sel)
      tok_idx += 1

  total_pages = batch_size * pages_per_seq
  nope_fp8_torch = torch.randn(
      (total_pages, page_size, reference.NOPE_DIM),
      dtype=torch.float32,
      device=device,
  ).to(torch.float8_e4m3fn)
  rope_fp8_torch = torch.randn(
      (total_pages, page_size, ROPE_DIM), dtype=torch.float32, device=device
  ).to(torch.float8_e4m3fn)
  rope_padded_torch = torch.nn.functional.pad(
      rope_fp8_torch.view(torch.uint8),
      (0, reference.ROPE_STORAGE_DIM - ROPE_DIM),
      value=0,
  )

  cache_kv_nope_torch = nope_fp8_torch.view(torch.uint8).reshape(
      total_pages, page_size, 4, 128
  )
  cache_kv_rope_torch = rope_padded_torch.reshape(
      total_pages, page_size // 4, 4, 128
  )
  kv_lens_torch = torch.as_tensor(kv_lens_np, dtype=torch.int32, device=device)
  topk_indices_torch = torch.as_tensor(
      topk_rows, dtype=torch.int32, device=device
  )
  page_indices_torch = torch.arange(
      total_pages, dtype=torch.int32, device=device
  )
  cu_q_lens_torch = torch.as_tensor(
      cu_q_lens_np, dtype=torch.int32, device=device
  )
  distribution_torch = torch.tensor(
      [num_decode_seqs, num_decode_seqs, batch_size],
      dtype=torch.int32,
      device=device,
  )

  torch_inputs = dict(
      q=q_torch,
      cache_kv_nope=cache_kv_nope_torch,
      cache_kv_rope=cache_kv_rope_torch,
      kv_lens=kv_lens_torch,
      topk_indices=topk_indices_torch,
      page_indices=page_indices_torch,
      cu_q_lens=cu_q_lens_torch,
      distribution=distribution_torch,
  )
  jax_inputs = dict(
      q=jnp.asarray(
          q_torch.detach().cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16
      ),
      cache_kv_nope=jnp.asarray(
          cache_kv_nope_torch.detach().cpu().numpy(), dtype=jnp.uint8
      ),
      cache_kv_rope=jnp.asarray(
          cache_kv_rope_torch.detach().cpu().numpy(), dtype=jnp.uint8
      ),
      kv_lens=jnp.asarray(
          kv_lens_torch.detach().cpu().numpy(), dtype=jnp.int32
      ),
      topk_indices=jnp.asarray(
          topk_indices_torch.detach().cpu().numpy(), dtype=jnp.int32
      ),
      page_indices=jnp.asarray(
          page_indices_torch.detach().cpu().numpy(), dtype=jnp.int32
      ),
      cu_q_lens=jnp.asarray(
          cu_q_lens_torch.detach().cpu().numpy(), dtype=jnp.int32
      ),
      distribution=jnp.asarray(
          distribution_torch.detach().cpu().numpy(), dtype=jnp.int32
      ),
  )
  return torch_inputs, jax_inputs, actual_tokens


class PallasMosaicTpuMaskedDenseMlaTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if backend.get_default_device().device_kind != "TPU7x":
      self.skipTest("Only tested on TPU7x.")
    torch.manual_seed(0)

  @parameterized.named_parameters(
      dict(
          testcase_name="bitmap_unpadded",
          topk=64,
          pad_tokens=False,
          k_scale=1.0,
      ),
      dict(
          testcase_name="bitmap_padded",
          topk=64,
          pad_tokens=True,
          k_scale=1.0,
      ),
      dict(
          testcase_name="bitmap_k_scale_half",
          topk=64,
          pad_tokens=False,
          k_scale=0.5,
      ),
      dict(
          testcase_name="analytic_causal_mask",
          topk=256,
          pad_tokens=False,
          k_scale=1.0,
      ),
      dict(
          testcase_name="custom_mask_value",
          topk=64,
          pad_tokens=False,
          k_scale=1.0,
          mask_value=-1e30,
      ),
      dict(
          testcase_name="sequence_start",
          topk=64,
          pad_tokens=False,
          k_scale=1.0,
          sequence_start=2,
      ),
  )
  def test_pallas_mosaic_tpu_matches_jax_op(
      self,
      topk: int,
      pad_tokens: bool,
      k_scale: float,
      mask_value: float | None = None,
      sequence_start: int | None = None,
  ):
    device = "tpu"
    torch_inputs, jax_inputs, actual_tokens = _make_torch_and_jax_inputs(
        topk=topk, pad_tokens=pad_tokens, seed=0, device=device
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    seq_start_torch = (
        None
        if sequence_start is None
        else torch.tensor(sequence_start, dtype=torch.int32, device=device)
    )
    seq_start_jax = (
        None
        if sequence_start is None
        else jnp.asarray(sequence_start, dtype=jnp.int32)
    )

    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuMaskedDenseMla(
        **torch_inputs,
        sm_scale=sm_scale,
        k_scale=k_scale,
        mask_value=mask_value,
        sequence_start=seq_start_torch,
    )
    desired_pallas_jax = jax_pallas_mosaic_tpu.PallasTpuMaskedDenseMla()(
        **jax_inputs,
        sm_scale=sm_scale,
        k_scale=k_scale,
        mask_value=mask_value,
        sequence_start=seq_start_jax,
    )
    desired_base_jax = jax_base.MaskedDenseMla()(
        **jax_inputs,
        sm_scale=sm_scale,
        k_scale=k_scale,
        mask_value=mask_value,
        sequence_start=seq_start_jax,
    )

    desired_pallas = torch.as_tensor(
        np.asarray(desired_pallas_jax, dtype=np.float32),
        dtype=torch.bfloat16,
        device=device,
    )
    desired_base = torch.as_tensor(
        np.asarray(desired_base_jax, dtype=np.float32),
        dtype=torch.bfloat16,
        device=device,
    )

    self.assertEqual(actual.shape, desired_pallas.shape)
    torch.testing.assert_close(
        actual[:actual_tokens],
        desired_pallas[:actual_tokens],
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        actual[:actual_tokens],
        desired_base[:actual_tokens],
        rtol=RTOL,
        atol=ATOL,
    )

  def test_custom_config(self):
    device = "tpu"
    torch_inputs, jax_inputs, actual_tokens = _make_torch_and_jax_inputs(
        topk=64, pad_tokens=True, seed=13, device=device
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    cfg = Config(
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 8, 8),
    )

    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuMaskedDenseMla(
        **torch_inputs, sm_scale=sm_scale, configs=(cfg, None)
    )
    desired_jax = jax_pallas_mosaic_tpu.PallasTpuMaskedDenseMla(config=cfg)(
        **jax_inputs, sm_scale=sm_scale
    )
    desired = torch.as_tensor(
        np.asarray(desired_jax, dtype=np.float32),
        dtype=torch.bfloat16,
        device=device,
    )
    torch.testing.assert_close(
        actual[:actual_tokens],
        desired[:actual_tokens],
        rtol=RTOL,
        atol=ATOL,
    )

  def test_call_without_config_uses_heuristics_config(self):
    device = "tpu"
    torch_inputs, _, actual_tokens = _make_torch_and_jax_inputs(
        topk=64, pad_tokens=False, seed=19, device=device
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuMaskedDenseMla
    expected_configs = torch_utils.get_configs(
        op_pallas,
        torch_inputs["q"],
        torch_inputs["cache_kv_nope"],
        torch_inputs["cache_kv_rope"],
        torch_inputs["kv_lens"],
        torch_inputs["topk_indices"],
        torch_inputs["page_indices"],
        torch_inputs["cu_q_lens"],
        torch_inputs["distribution"],
        sm_scale=sm_scale,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(expected_configs[0])

    out = op_pallas(**torch_inputs, sm_scale=sm_scale)
    expected_out = op_pallas(
        **torch_inputs, sm_scale=sm_scale, configs=expected_configs
    )
    torch.testing.assert_close(
        out[:actual_tokens], expected_out[:actual_tokens]
    )

  def test_torch_compile_with_config(self):
    device = "tpu"
    torch_inputs, _, actual_tokens = _make_torch_and_jax_inputs(
        topk=64, pad_tokens=False, seed=23, device=device
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuMaskedDenseMla
    configs = torch_utils.get_configs(
        op_pallas,
        torch_inputs["q"],
        torch_inputs["cache_kv_nope"],
        torch_inputs["cache_kv_rope"],
        torch_inputs["kv_lens"],
        torch_inputs["topk_indices"],
        torch_inputs["page_indices"],
        torch_inputs["cu_q_lens"],
        torch_inputs["distribution"],
        sm_scale=sm_scale,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(configs[0])

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        q,
        cache_kv_nope,
        cache_kv_rope,
        kv_lens,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
    ):
      q = q + 0.1
      out = op_pallas(
          q,
          cache_kv_nope,
          cache_kv_rope,
          kv_lens,
          topk_indices,
          page_indices,
          cu_q_lens,
          distribution,
          sm_scale=sm_scale,
          configs=configs,
      )
      return out + 2.0

    actual = compiled_fn(**torch_inputs)
    expected = (
        torch_base.MaskedDenseMla(
            **{**torch_inputs, "q": torch_inputs["q"] + 0.1},
            sm_scale=sm_scale,
        )
        + 2.0
    )
    torch.testing.assert_close(
        actual[:actual_tokens],
        expected[:actual_tokens],
        rtol=RTOL,
        atol=ATOL,
    )

  def test_torch_compile_call_without_config_uses_heuristics_config(self):
    device = "tpu"
    torch_inputs, _, actual_tokens = _make_torch_and_jax_inputs(
        topk=64, pad_tokens=False, seed=29, device=device
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuMaskedDenseMla
    expected_configs = torch_utils.get_configs(
        op_pallas,
        torch_inputs["q"],
        torch_inputs["cache_kv_nope"],
        torch_inputs["cache_kv_rope"],
        torch_inputs["kv_lens"],
        torch_inputs["topk_indices"],
        torch_inputs["page_indices"],
        torch_inputs["cu_q_lens"],
        torch_inputs["distribution"],
        sm_scale=sm_scale,
        from_autotuning_cache=False,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        q,
        cache_kv_nope,
        cache_kv_rope,
        kv_lens,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
    ):
      return op_pallas(
          q,
          cache_kv_nope,
          cache_kv_rope,
          kv_lens,
          topk_indices,
          page_indices,
          cu_q_lens,
          distribution,
          sm_scale=sm_scale,
      )

    actual = compiled_fn(**torch_inputs)
    expected = op_pallas(
        **torch_inputs, sm_scale=sm_scale, configs=expected_configs
    )
    torch.testing.assert_close(actual[:actual_tokens], expected[:actual_tokens])

  def test_fake_impl_and_symbolic_shapes(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuMaskedDenseMla
    with fake_tensor.FakeTensorMode():
      q_fake = torch.empty((12, 16, HEAD_DIM), dtype=torch.bfloat16)
      cache_kv_nope_fake = torch.empty((8, 32, 4, 128), dtype=torch.uint8)
      cache_kv_rope_fake = torch.empty((8, 8, 4, 128), dtype=torch.uint8)
      kv_lens_fake = torch.empty((2,), dtype=torch.int32)
      topk_indices_fake = torch.empty((12, 64), dtype=torch.int32)
      page_indices_fake = torch.empty((8,), dtype=torch.int32)
      cu_q_lens_fake = torch.empty((3,), dtype=torch.int32)
      dist_fake = torch.empty((3,), dtype=torch.int32)
      out = op_pallas(
          q_fake,
          cache_kv_nope_fake,
          cache_kv_rope_fake,
          kv_lens_fake,
          topk_indices_fake,
          page_indices_fake,
          cu_q_lens_fake,
          dist_fake,
      )
    self.assertEqual(tuple(out.shape), (12, 16, HEAD_DIM))
    self.assertEqual(out.dtype, torch.bfloat16)

    class _Module(torch.nn.Module):

      def forward(
          self,
          q: torch.Tensor,
          cache_kv_nope: torch.Tensor,
          cache_kv_rope: torch.Tensor,
          kv_lens: torch.Tensor,
          topk_indices: torch.Tensor,
          page_indices: torch.Tensor,
          cu_q_lens: torch.Tensor,
          distribution: torch.Tensor,
      ) -> torch.Tensor:
        return op_pallas(
            q,
            cache_kv_nope,
            cache_kv_rope,
            kv_lens,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
        )

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    q = torch.ones((12, 16, HEAD_DIM), dtype=torch.bfloat16)
    cache_kv_nope = torch.zeros((8, 32, 4, 128), dtype=torch.uint8)
    cache_kv_rope = torch.zeros((8, 8, 4, 128), dtype=torch.uint8)
    kv_lens = torch.tensor([64, 64], dtype=torch.int32)
    topk_indices = torch.zeros((12, 64), dtype=torch.int32)
    page_indices = torch.arange(8, dtype=torch.int32)
    cu_q_lens = torch.tensor([0, 4, 12], dtype=torch.int32)
    distribution = torch.tensor([0, 0, 2], dtype=torch.int32)

    exported = torch.export.export(
        _Module(),
        (
            q,
            cache_kv_nope,
            cache_kv_rope,
            kv_lens,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
        ),
        dynamic_shapes=(
            {0: num_tokens},
            None,
            None,
            None,
            {0: num_tokens},
            None,
            None,
            None,
        ),
    )
    op_nodes = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function"
        and op_pallas.jax_op_name in str(node.target)
    ]
    self.assertLen(op_nodes, 1)
    meta = op_nodes[0].meta["val"]
    self.assertIsInstance(meta.shape[0], torch.SymInt)
    self.assertEqual(meta.shape[1], 16)
    self.assertEqual(meta.shape[2], HEAD_DIM)

  def test_deconstruct_and_reconstruct_config_roundtrip(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuMaskedDenseMla
    cfg = Config(
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 8, 8),
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    deconstructed = op_pallas.deconstruct_config(cfg)
    self.assertEqual(
        deconstructed,
        (2, 2, 2, 1, 8, 8, 64 * 1024 * 1024),
    )
    reconstructed = op_pallas.reconstruct_config(deconstructed)
    self.assertEqual(reconstructed, cfg)

    default_cfg = Config()
    self.assertEqual(
        op_pallas.reconstruct_config(op_pallas.deconstruct_config(default_cfg)),
        default_cfg,
    )
    self.assertIsNone(op_pallas.deconstruct_config(None))
    with self.assertRaises(ValueError):
      op_pallas.reconstruct_config(None)


if __name__ == "__main__":
  absltest.main()
