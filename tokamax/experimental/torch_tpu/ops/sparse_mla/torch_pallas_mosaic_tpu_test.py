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
"""Tests for Pallas Mosaic TPU implementation of SparseMLA."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.sparse_mla import base as jax_base
from tokamax._src.ops.sparse_mla import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax._src.ops.sparse_mla import reference as jax_reference
from tokamax._src.ops.sparse_mla import test_base as jax_test_base
from tokamax._src.ops.dsv4_compressor import csa_cache_layout
from tokamax.experimental.torch_tpu.ops import torch_test_utils
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.sparse_mla import torch_base
from tokamax.experimental.torch_tpu.ops.sparse_mla import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import

Config = jax_pallas_mosaic_tpu.Config


def _make_torch_and_jax_inputs(
    *,
    batch_size: int = 12,
    num_heads: int = 64,
    page_size: int = 16,
    topk: int = 1024,
    pad_tokens: bool = False,
    seed: int = 0,
    device: str = "tpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any], int]:
  """Creates PyTorch inputs on `device` and converts them to JAX arrays."""
  torch.manual_seed(seed)
  rng = np.random.default_rng(seed)
  head_dim = jax_reference.HEAD_DIM

  num_decode_seqs = batch_size // 2
  prefill_lens = rng.integers(8, 60, size=(batch_size - num_decode_seqs,))
  new_kv_lens = np.concatenate(
      [np.ones((num_decode_seqs,), dtype=np.int32), prefill_lens]
  )
  cu_q_lens_np = np.concatenate(
      [np.array([0], dtype=np.int32), np.cumsum(new_kv_lens, dtype=np.int32)]
  )
  kv_lens = new_kv_lens + rng.integers(30, 200, size=(batch_size,))
  actual_tokens = int(np.sum(new_kv_lens))
  total_tokens = (
      ((actual_tokens + 32 + 63) // 64) * 64 if pad_tokens else actual_tokens
  )

  q_torch = torch.rand(
      (total_tokens, num_heads, head_dim), dtype=torch.bfloat16, device=device
  )

  topk_indices_list = []
  for i in range(batch_size):
    kv_len_i = int(kv_lens[i])
    for _ in range(int(new_kv_lens[i])):
      perm = rng.permutation(kv_len_i)
      indices = list(perm[:topk])
      if len(indices) < topk:
        indices.extend([-1] * (topk - len(indices)))
      topk_indices_list.append(indices)
  for _ in range(total_tokens - actual_tokens):
    topk_indices_list.append([-1] * topk)
  topk_indices_torch = torch.tensor(
      topk_indices_list, dtype=torch.int32, device=device
  )

  pages_per_seq = (500 + page_size - 1) // page_size
  total_pages = batch_size * pages_per_seq
  page_indices_torch = torch.arange(
      total_pages, dtype=torch.int32, device=device
  )

  kv_c_torch = torch.rand(
      (total_pages, page_size, head_dim), dtype=torch.bfloat16, device=device
  )
  kv_c_jax = torch_test_utils.torch_to_jax(kv_c_torch)
  fp8_part = kv_c_jax[..., : jax_reference.NOPE_DIM]
  bf16_part = kv_c_jax[..., jax_reference.NOPE_DIM : head_dim]
  fp8_quant, scales_quant = jax_test_base._quantize_lane_periodic(  # pylint: disable=protected-access
      fp8_part
  )
  fp8_uint8 = jax.lax.bitcast_convert_type(
      fp8_quant.reshape(total_pages, page_size, jax_reference.NOPE_DIM),
      jnp.uint8,
  )
  scales_uint8 = jax.lax.bitcast_convert_type(scales_quant, jnp.uint8)
  nope_slabs = jnp.concatenate([fp8_uint8, scales_uint8], axis=-1).reshape(
      total_pages, page_size, 4, 128
  )
  cache_kv_nope_jax = csa_cache_layout.slabs_to_words(nope_slabs)
  cache_kv_rope_jax = csa_cache_layout.encode_rope(
      bf16_part.reshape(-1, jax_reference.ROPE_DIM)
  ).reshape(total_pages, page_size // 4, 128)

  cache_kv_nope_torch = torch.as_tensor(
      np.asarray(cache_kv_nope_jax, dtype=np.int32),
      dtype=torch.int32,
      device=device,
  )
  cache_kv_rope_torch = torch.as_tensor(
      np.asarray(cache_kv_rope_jax, dtype=np.int32),
      dtype=torch.int32,
      device=device,
  )

  cu_q_lens_torch = torch.as_tensor(
      cu_q_lens_np, dtype=torch.int32, device=device
  )
  distribution_torch = torch.tensor(
      [num_decode_seqs, num_decode_seqs, batch_size],
      dtype=torch.int32,
      device=device,
  )
  attention_sinks_torch = torch.rand(
      (num_heads,), dtype=torch.float32, device=device
  )
  swa_accumution_torch = torch.rand(
      (total_tokens, num_heads, head_dim), dtype=torch.bfloat16, device=device
  )
  swa_l_torch = torch.rand(
      (total_tokens, num_heads), dtype=torch.float32, device=device
  )
  swa_m_torch = torch.rand(
      (total_tokens, num_heads), dtype=torch.float32, device=device
  )

  torch_inputs = dict(
      q=q_torch,
      cache_kv_nope=cache_kv_nope_torch,
      cache_kv_rope=cache_kv_rope_torch,
      topk_indices=topk_indices_torch,
      page_indices=page_indices_torch,
      cu_q_lens=cu_q_lens_torch,
      distribution=distribution_torch,
      attention_sinks=attention_sinks_torch,
      swa_accumution=swa_accumution_torch,
      swa_l=swa_l_torch,
      swa_m=swa_m_torch,
  )
  jax_inputs = {
      name: torch_test_utils.torch_to_jax(tensor)
      for name, tensor in torch_inputs.items()
  }
  return torch_inputs, jax_inputs, actual_tokens


class PallasMosaicTpuSparseMlaTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    device = jax.devices()[0]
    if not jax_pallas_mosaic_tpu.PallasTpuSparseMla().supported_on(device):
      self.skipTest(f"Op not supported on {device.device_kind}.")
    torch.manual_seed(0)

  @parameterized.named_parameters(
      dict(
          testcase_name="unpadded_topk1024",
          topk=1024,
          pad_tokens=False,
      ),
      dict(
          testcase_name="padded_topk1024",
          topk=1024,
          pad_tokens=True,
      ),
      dict(
          testcase_name="unpadded_topk512",
          topk=512,
          pad_tokens=False,
      ),
      dict(
          testcase_name="unpadded_topk128",
          topk=128,
          pad_tokens=False,
      ),
  )
  def test_pallas_mosaic_tpu_matches_jax_op(
      self, topk: int, pad_tokens: bool
  ):
    device = "tpu"
    torch_inputs, jax_inputs, actual_tokens = _make_torch_and_jax_inputs(
        topk=topk, pad_tokens=pad_tokens, seed=42, device=device
    )

    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuSparseMla(
        **torch_inputs, sm_scale=1.0
    )
    desired_pallas_jax = jax_pallas_mosaic_tpu.PallasTpuSparseMla()(
        **jax_inputs, sm_scale=1.0
    )
    desired_base_jax = jax_base.SparseMla()(**jax_inputs, sm_scale=1.0)

    desired_pallas = torch_test_utils.jax_to_torch(
        desired_pallas_jax, device=device
    )
    desired_base = torch_test_utils.jax_to_torch(
        desired_base_jax, device=device
    )

    self.assertEqual(actual.shape, desired_pallas.shape)
    self.assertEqual(actual.dtype, desired_pallas.dtype)
    torch.testing.assert_close(
        actual[:actual_tokens],
        desired_pallas[:actual_tokens],
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        actual[:actual_tokens],
        desired_base[:actual_tokens],
        rtol=jax_test_base.RTOL,
        atol=jax_test_base.ATOL,
    )

  def test_custom_config(self):
    device = "tpu"
    torch_inputs, jax_inputs, actual_tokens = _make_torch_and_jax_inputs(
        batch_size=4, topk=512, pad_tokens=True, seed=13, device=device
    )
    cfg = Config(
        gather_and_attention_chunk_size=64,
        attention_kernel_batch_size=8,
    )

    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuSparseMla(
        **torch_inputs, sm_scale=1.0, configs=(cfg, None)
    )
    desired_jax = jax_pallas_mosaic_tpu.PallasTpuSparseMla(config=cfg)(
        **jax_inputs, sm_scale=1.0
    )
    desired = torch_test_utils.jax_to_torch(desired_jax, device=device)

    torch.testing.assert_close(
        actual[:actual_tokens],
        desired[:actual_tokens],
        rtol=1e-3,
        atol=1e-3,
    )

  def test_call_without_config_uses_heuristics_config(self):
    device = "tpu"
    torch_inputs, _, _ = _make_torch_and_jax_inputs(
        batch_size=4, topk=128, seed=19, device=device
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuSparseMla
    expected_configs = torch_utils.get_configs(
        op_pallas,
        *torch_inputs.values(),
        sm_scale=1.0,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(expected_configs[0])

    out = op_pallas(**torch_inputs, sm_scale=1.0)
    expected_out = op_pallas(
        **torch_inputs, sm_scale=1.0, configs=expected_configs
    )
    torch.testing.assert_close(out, expected_out)

  def test_torch_compile_with_config(self):
    device = "tpu"
    torch_inputs, _, actual_tokens = _make_torch_and_jax_inputs(
        batch_size=4, topk=128, seed=23, device=device
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuSparseMla
    configs = torch_utils.get_configs(
        op_pallas,
        *torch_inputs.values(),
        sm_scale=1.0,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(configs[0])

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        q,
        cache_kv_nope,
        cache_kv_rope,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m,
    ):
      q = q + 0.1
      out = op_pallas(
          q,
          cache_kv_nope,
          cache_kv_rope,
          topk_indices,
          page_indices,
          cu_q_lens,
          distribution,
          attention_sinks,
          swa_accumution,
          swa_l,
          swa_m,
          sm_scale=1.0,
          configs=configs,
      )
      return out + 2.0

    actual = compiled_fn(**torch_inputs)
    expected = (
        torch_base.SparseMla(
            **{**torch_inputs, "q": torch_inputs["q"] + 0.1},
            sm_scale=1.0,
        )
        + 2.0
    )
    torch.testing.assert_close(
        actual[:actual_tokens],
        expected[:actual_tokens],
        rtol=jax_test_base.RTOL,
        atol=jax_test_base.ATOL,
    )

  def test_torch_compile_call_without_config_uses_heuristics_config(self):
    device = "tpu"
    torch_inputs, _, _ = _make_torch_and_jax_inputs(
        batch_size=4, topk=128, seed=29, device=device
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuSparseMla
    expected_configs = torch_utils.get_configs(
        op_pallas,
        *torch_inputs.values(),
        sm_scale=1.0,
        from_autotuning_cache=False,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        q,
        cache_kv_nope,
        cache_kv_rope,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m,
    ):
      return op_pallas(
          q,
          cache_kv_nope,
          cache_kv_rope,
          topk_indices,
          page_indices,
          cu_q_lens,
          distribution,
          attention_sinks,
          swa_accumution,
          swa_l,
          swa_m,
          sm_scale=1.0,
      )

    actual = compiled_fn(**torch_inputs)
    expected = op_pallas(**torch_inputs, sm_scale=1.0, configs=expected_configs)
    torch.testing.assert_close(actual, expected)

  def test_fake_impl_and_symbolic_shapes(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuSparseMla
    with fake_tensor.FakeTensorMode():
      q_fake = torch.empty((16, 64, 512), dtype=torch.bfloat16)
      cache_nope_fake = torch.empty((8, 16, 128), dtype=torch.int32)
      cache_rope_fake = torch.empty((8, 4, 128), dtype=torch.int32)
      topk_fake = torch.empty((16, 128), dtype=torch.int32)
      page_indices_fake = torch.empty((8,), dtype=torch.int32)
      cu_q_lens_fake = torch.empty((3,), dtype=torch.int32)
      dist_fake = torch.empty((3,), dtype=torch.int32)
      sinks_fake = torch.empty((64,), dtype=torch.float32)
      swa_acc_fake = torch.empty((16, 64, 512), dtype=torch.bfloat16)
      swa_l_fake = torch.empty((16, 64), dtype=torch.float32)
      swa_m_fake = torch.empty((16, 64), dtype=torch.float32)
      out = op_pallas(
          q_fake,
          cache_nope_fake,
          cache_rope_fake,
          topk_fake,
          page_indices_fake,
          cu_q_lens_fake,
          dist_fake,
          sinks_fake,
          swa_acc_fake,
          swa_l_fake,
          swa_m_fake,
      )
    self.assertEqual(tuple(out.shape), (16, 64, 512))
    self.assertEqual(out.dtype, torch.bfloat16)

    class _Module(torch.nn.Module):

      def forward(
          self,
          q: torch.Tensor,
          cache_kv_nope: torch.Tensor,
          cache_kv_rope: torch.Tensor,
          topk_indices: torch.Tensor,
          page_indices: torch.Tensor,
          cu_q_lens: torch.Tensor,
          distribution: torch.Tensor,
          attention_sinks: torch.Tensor,
          swa_accumution: torch.Tensor,
          swa_l: torch.Tensor,
          swa_m: torch.Tensor,
      ) -> torch.Tensor:
        return op_pallas(
            q,
            cache_kv_nope,
            cache_kv_rope,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
        )

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    q = torch.ones((16, 64, 512), dtype=torch.bfloat16)
    cache_kv_nope = torch.zeros((8, 16, 128), dtype=torch.int32)
    cache_kv_rope = torch.zeros((8, 4, 128), dtype=torch.int32)
    topk_indices = torch.zeros((16, 128), dtype=torch.int32)
    page_indices = torch.arange(8, dtype=torch.int32)
    cu_q_lens = torch.tensor([0, 8, 16], dtype=torch.int32)
    distribution = torch.tensor([0, 0, 2], dtype=torch.int32)
    attention_sinks = torch.zeros((64,), dtype=torch.float32)
    swa_accumution = torch.ones((16, 64, 512), dtype=torch.bfloat16)
    swa_l = torch.ones((16, 64), dtype=torch.float32)
    swa_m = torch.zeros((16, 64), dtype=torch.float32)

    exported = torch.export.export(
        _Module(),
        (
            q,
            cache_kv_nope,
            cache_kv_rope,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
        ),
        dynamic_shapes=(
            {0: num_tokens},
            None,
            None,
            {0: num_tokens},
            None,
            None,
            None,
            None,
            {0: num_tokens},
            {0: num_tokens},
            {0: num_tokens},
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
    self.assertEqual(meta.shape[1], 64)
    self.assertEqual(meta.shape[2], 512)

  def test_deconstruct_and_reconstruct_config_roundtrip(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuSparseMla
    cfg = Config(
        gather_and_attention_chunk_size=64,
        attention_kernel_batch_size=8,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    deconstructed = op_pallas.deconstruct_config(cfg)
    self.assertEqual(deconstructed, (64, 8, 64 * 1024 * 1024))
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
