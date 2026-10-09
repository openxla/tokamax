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
"""Tests for Pallas Mosaic TPU implementation of Lightning Indexer."""

from collections.abc import Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.lightning_indexer import base as jax_base
from tokamax._src.ops.experimental.lightning_indexer import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax._src.ops.experimental.lightning_indexer import test_base as jax_test_base
from tokamax._src.ops.experimental.lightning_indexer.kernel import config as kernel_config
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.lightning_indexer import torch_base
from tokamax.experimental.torch_tpu.ops.lightning_indexer import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import

Config = jax_pallas_mosaic_tpu.Config
KVLayout = kernel_config.KVLayout


def _make_torch_and_jax_inputs(
    q_lens: Sequence[int],
    seq_lens: Sequence[int],
    *,
    page_size: int = 128,
    pages_per_seq: int = 4,
    num_q_heads: int = 4,
    head_dim: int = 128,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    seed: int = 42,
    device: str = "tpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  """Creates PyTorch inputs on `device` and converts them to JAX arrays."""
  torch.manual_seed(seed)
  rng = np.random.default_rng(seed)
  num_seqs = len(q_lens)
  num_tokens = int(sum(q_lens))
  num_pages = num_seqs * pages_per_seq

  q_torch = torch.randn(
      (num_tokens, num_q_heads, head_dim), dtype=torch.float32, device=device
  )
  indexer_weights_torch = torch.empty(
      (num_tokens, num_q_heads), dtype=torch.float32, device=device
  ).uniform_(0.25, 1.75)

  keys_torch = torch.randn(
      (num_pages, page_size, head_dim), dtype=torch.float32, device=device
  )
  keys_jax = jnp.asarray(keys_torch.detach().cpu().numpy(), dtype=jnp.float32)
  cache_kv_jax_packed = jax_test_base.quantize_and_pack_cache(
      keys_jax, kv_layout=kv_layout
  )
  cache_kv_torch = torch.as_tensor(
      np.asarray(cache_kv_jax_packed, dtype=np.uint8),
      dtype=torch.uint8,
      device=device,
  )

  block_table_np = rng.permutation(num_pages).astype(np.int32)
  seq_lens_torch = torch.tensor(seq_lens, dtype=torch.int32, device=device)
  page_indices_torch = torch.as_tensor(
      block_table_np, dtype=torch.int32, device=device
  )
  cu_q_lens_torch = torch.tensor(
      [0, *np.cumsum(q_lens).tolist()], dtype=torch.int32, device=device
  )
  num_decodes = 0
  while num_decodes < num_seqs and q_lens[num_decodes] == 1:
    num_decodes += 1
  distribution_torch = torch.tensor(
      [num_decodes, num_decodes, num_seqs], dtype=torch.int32, device=device
  )

  torch_inputs = dict(
      q=q_torch,
      indexer_weights=indexer_weights_torch,
      cache_kv=cache_kv_torch,
      seq_lens=seq_lens_torch,
      page_indices=page_indices_torch,
      cu_q_lens=cu_q_lens_torch,
      distribution=distribution_torch,
  )
  jax_inputs = {
      name: jnp.asarray(tensor.detach().cpu().numpy())
      for name, tensor in torch_inputs.items()
  }
  return torch_inputs, jax_inputs


class PallasMosaicTpuLightningIndexerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax_test_base.skip_if_unsupported(self)
    torch.manual_seed(0)

  @parameterized.named_parameters(
      dict(
          testcase_name="decode_only",
          q_lens=[1, 1, 1, 1],
          seq_lens=[256, 384, 128, 512],
          k=16,
          compression_ratio=1,
          kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dict(
          testcase_name="prefill_only",
          q_lens=[16, 8],
          seq_lens=[256, 384],
          k=16,
          compression_ratio=1,
          kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dict(
          testcase_name="mixed_decode_and_prefill",
          q_lens=[1, 1, 12],
          seq_lens=[256, 128, 384],
          k=16,
          compression_ratio=1,
          kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dict(
          testcase_name="compressed_kv_ratio_4",
          q_lens=[1, 8],
          seq_lens=[512, 1024],
          k=16,
          compression_ratio=4,
          kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dict(
          testcase_name="seq_along_lane_layout",
          q_lens=[1, 1, 8],
          seq_lens=[256, 384, 256],
          k=16,
          compression_ratio=1,
          kv_layout=KVLayout.SEQ_ALONG_LANE,
      ),
  )
  def test_pallas_mosaic_tpu_matches_jax_op(
      self, q_lens, seq_lens, k, compression_ratio, kv_layout
  ):
    device = "tpu"
    torch_inputs, jax_inputs = _make_torch_and_jax_inputs(
        q_lens, seq_lens, kv_layout=kv_layout, seed=42, device=device
    )

    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuLightningIndexer(
        **torch_inputs,
        k=k,
        compression_ratio=compression_ratio,
        kv_layout=kv_layout,
    )
    desired_pallas_jax = jax_pallas_mosaic_tpu.PallasTpuLightningIndexer()(
        **jax_inputs,
        k=k,
        compression_ratio=compression_ratio,
        kv_layout=kv_layout,
    )
    desired_base_jax = jax_base.LightningIndexer()(
        **jax_inputs,
        k=k,
        compression_ratio=compression_ratio,
        kv_layout=kv_layout,
    )

    desired_pallas = torch.as_tensor(
        np.asarray(desired_pallas_jax, dtype=np.int32),
        dtype=torch.int32,
        device=device,
    )
    desired_base = torch.as_tensor(
        np.asarray(desired_base_jax, dtype=np.int32),
        dtype=torch.int32,
        device=device,
    )

    self.assertEqual(actual.shape, desired_pallas.shape)
    torch.testing.assert_close(
        torch.sort(actual, dim=-1).values,
        torch.sort(desired_pallas, dim=-1).values,
    )
    torch.testing.assert_close(
        torch.sort(actual, dim=-1).values,
        torch.sort(desired_base, dim=-1).values,
    )

  def test_return_scores_matches_jax_op(self):
    device = "tpu"
    torch_inputs, jax_inputs = _make_torch_and_jax_inputs(
        [1, 4], [256, 256], seed=7, device=device
    )

    actual_idxs, actual_scores_bits = (
        torch_pallas_mosaic_tpu.PallasMosaicTpuLightningIndexer(
            **torch_inputs, k=16, compression_ratio=1, return_scores=True
        )
    )
    desired_idxs_jax, desired_scores_bits_jax = (
        jax_pallas_mosaic_tpu.PallasTpuLightningIndexer()(
            **jax_inputs, k=16, compression_ratio=1, return_scores=True
        )
    )

    desired_idxs = torch.as_tensor(
        np.asarray(desired_idxs_jax, dtype=np.int32),
        dtype=torch.int32,
        device=device,
    )
    torch.testing.assert_close(
        torch.sort(actual_idxs, dim=-1).values,
        torch.sort(desired_idxs, dim=-1).values,
    )

    actual_scores = np.sort(
        actual_scores_bits.detach().cpu().numpy().view(np.float32), axis=-1
    )
    desired_scores = np.sort(
        np.asarray(desired_scores_bits_jax, dtype=np.int32).view(np.float32),
        axis=-1,
    )
    np.testing.assert_allclose(
        actual_scores, desired_scores, rtol=1e-2, atol=1e-2
    )

  def test_custom_config(self):
    device = "tpu"
    torch_inputs, jax_inputs = _make_torch_and_jax_inputs(
        [1, 4], [256, 256], seed=13, device=device
    )
    cfg = Config(
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 8, 8),
        decode_req_batch_size=1,
    )

    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuLightningIndexer(
        **torch_inputs, k=16, config=cfg
    )
    desired_jax = jax_pallas_mosaic_tpu.PallasTpuLightningIndexer(config=cfg)(
        **jax_inputs, k=16
    )
    desired = torch.as_tensor(
        np.asarray(desired_jax, dtype=np.int32),
        dtype=torch.int32,
        device=device,
    )
    torch.testing.assert_close(
        torch.sort(actual, dim=-1).values,
        torch.sort(desired, dim=-1).values,
    )

  def test_call_without_config_uses_heuristics_config(self):
    device = "tpu"
    torch_inputs, _ = _make_torch_and_jax_inputs(
        [1, 4], [256, 256], seed=19, device=device
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuLightningIndexer
    expected_config, _ = torch_utils.get_configs(
        op_pallas,
        torch_inputs["q"],
        torch_inputs["indexer_weights"],
        torch_inputs["cache_kv"],
        torch_inputs["seq_lens"],
        torch_inputs["page_indices"],
        torch_inputs["cu_q_lens"],
        torch_inputs["distribution"],
        k=16,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(expected_config)

    out = op_pallas(**torch_inputs, k=16)
    expected_out = op_pallas(**torch_inputs, k=16, config=expected_config)
    torch.testing.assert_close(out, expected_out)

  def test_torch_compile_with_config(self):
    device = "tpu"
    torch_inputs, _ = _make_torch_and_jax_inputs(
        [1, 4], [256, 256], seed=23, device=device
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuLightningIndexer
    configs = torch_utils.get_configs(
        op_pallas,
        torch_inputs["q"],
        torch_inputs["indexer_weights"],
        torch_inputs["cache_kv"],
        torch_inputs["seq_lens"],
        torch_inputs["page_indices"],
        torch_inputs["cu_q_lens"],
        torch_inputs["distribution"],
        k=16,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(configs[0])

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ):
      q = q + 0.1
      out = op_pallas(
          q,
          indexer_weights,
          cache_kv,
          seq_lens,
          page_indices,
          cu_q_lens,
          distribution,
          k=16,
          configs=configs,
      )
      return out + 2

    actual = compiled_fn(**torch_inputs)
    expected = (
        torch_base.LightningIndexer(
            **{**torch_inputs, "q": torch_inputs["q"] + 0.1},
            k=16,
        )
        + 2
    )
    torch.testing.assert_close(
        torch.sort(actual, dim=-1).values,
        torch.sort(expected, dim=-1).values,
    )

  def test_torch_compile_call_without_config_uses_heuristics_config(self):
    device = "tpu"
    torch_inputs, _ = _make_torch_and_jax_inputs(
        [1, 4], [256, 256], seed=29, device=device
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuLightningIndexer
    expected_config, _ = torch_utils.get_configs(
        op_pallas,
        torch_inputs["q"],
        torch_inputs["indexer_weights"],
        torch_inputs["cache_kv"],
        torch_inputs["seq_lens"],
        torch_inputs["page_indices"],
        torch_inputs["cu_q_lens"],
        torch_inputs["distribution"],
        k=16,
        from_autotuning_cache=False,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ):
      return op_pallas(
          q,
          indexer_weights,
          cache_kv,
          seq_lens,
          page_indices,
          cu_q_lens,
          distribution,
          k=16,
      )

    actual = compiled_fn(**torch_inputs)
    expected = op_pallas(**torch_inputs, k=16, config=expected_config)
    torch.testing.assert_close(actual, expected)

  def test_fake_impl_and_symbolic_shapes(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuLightningIndexer
    with fake_tensor.FakeTensorMode():
      q_fake = torch.empty((12, 4, 128), dtype=torch.float32)
      weights_fake = torch.empty((12, 4), dtype=torch.float32)
      cache_kv_fake = torch.empty((8, 32, 4, 256), dtype=torch.uint8)
      seq_lens_fake = torch.empty((2,), dtype=torch.int32)
      page_indices_fake = torch.empty((8,), dtype=torch.int32)
      cu_q_lens_fake = torch.empty((3,), dtype=torch.int32)
      dist_fake = torch.empty((3,), dtype=torch.int32)
      out = op_pallas(
          q_fake,
          weights_fake,
          cache_kv_fake,
          seq_lens_fake,
          page_indices_fake,
          cu_q_lens_fake,
          dist_fake,
          k=16,
      )
    self.assertEqual(tuple(out.shape), (12, 16))
    self.assertEqual(out.dtype, torch.int32)

    class _Module(torch.nn.Module):

      def forward(
          self,
          q: torch.Tensor,
          indexer_weights: torch.Tensor,
          cache_kv: torch.Tensor,
          seq_lens: torch.Tensor,
          page_indices: torch.Tensor,
          cu_q_lens: torch.Tensor,
          distribution: torch.Tensor,
      ) -> torch.Tensor:
        return op_pallas(
            q,
            indexer_weights,
            cache_kv,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            k=16,
        )

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    q = torch.ones((12, 4, 128), dtype=torch.float32)
    weights = torch.ones((12, 4), dtype=torch.float32)
    cache_kv = torch.zeros((8, 32, 4, 256), dtype=torch.uint8)
    seq_lens = torch.tensor([128, 256], dtype=torch.int32)
    page_indices = torch.arange(8, dtype=torch.int32)
    cu_q_lens = torch.tensor([0, 4, 12], dtype=torch.int32)
    distribution = torch.tensor([0, 0, 2], dtype=torch.int32)

    exported = torch.export.export(
        _Module(),
        (
            q,
            weights,
            cache_kv,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
        ),
        dynamic_shapes=(
            {0: num_tokens},
            {0: num_tokens},
            None,
            None,
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
    meta = op_nodes[0].meta["val"][0]
    self.assertIsInstance(meta.shape[0], torch.SymInt)
    self.assertEqual(meta.shape[1], 16)

  def test_deconstruct_and_reconstruct_config_roundtrip(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuLightningIndexer
    cfg = Config(
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 8, 8),
        buffer_count=(4, 3, 3),
        vmem_limit_bytes=64 * 1024 * 1024,
        decode_req_batch_size=2,
        enable_early_exit=True,
        chunk_tokens=32,
    )
    deconstructed = op_pallas.deconstruct_config(cfg)
    self.assertEqual(
        deconstructed,
        (2, 2, 2, 1, 8, 8, 4, 3, 3, 64 * 1024 * 1024, 2, 1, 32),
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
