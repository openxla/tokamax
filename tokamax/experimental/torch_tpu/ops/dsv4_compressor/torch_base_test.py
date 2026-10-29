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
"""Tests for Torch Dsv4Compressor base op."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tokamax._src.ops.dsv4_compressor import base as jax_base
from tokamax._src.ops.dsv4_compressor import test_base as jax_test_base
from tokamax.experimental.torch_tpu.ops import torch_test_utils
from tokamax.experimental.torch_tpu.ops.dsv4_compressor import torch_base
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


def _make_torch_and_jax_inputs(
    case_name: str,
    device: str = "tpu",
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    jax_test_base.Dsv4CompressorInputs,
]:
  """Creates PyTorch inputs on `device` and converts them to JAX arrays."""
  case_inputs = jax_test_base.make_case_inputs(case_name)
  torch_tensors = dict(
      cache=torch_test_utils.jax_to_torch(case_inputs.cache, device=device),
      positions=torch_test_utils.jax_to_torch(
          case_inputs.positions, device=device
      ),
      block_table=torch_test_utils.jax_to_torch(
          case_inputs.block_table, device=device
      ),
      token_to_req_indices=torch_test_utils.jax_to_torch(
          case_inputs.token_to_req_indices, device=device
      ),
      kv_slot_mapping=torch_test_utils.jax_to_torch(
          case_inputs.kv_slot_mapping, device=device
      ),
      rms_weight=torch_test_utils.jax_to_torch(
          case_inputs.rms_weight, device=device
      ),
      cos_sin_cache=torch_test_utils.jax_to_torch(
          case_inputs.cos_sin_cache, device=device
      ),
  )
  state_cache_torch = (
      None
      if case_inputs.state_cache is None
      else torch_test_utils.jax_to_torch(case_inputs.state_cache, device=device)
  )
  rope_cache_torch = (
      None
      if case_inputs.rope_cache is None
      else torch_test_utils.jax_to_torch(case_inputs.rope_cache, device=device)
  )
  scalar_kwargs = dict(
      block_table_stride=case_inputs.block_table_stride,
      state_block_size=case_inputs.state_block_size,
      compress_ratio=case_inputs.compress_ratio,
      overlap=case_inputs.overlap,
      quant_block=case_inputs.quant_block,
      rms_eps=case_inputs.rms_eps,
  )
  torch_inputs = {
      **torch_tensors,
      **scalar_kwargs,
      "state_cache": state_cache_torch,
      "rope_cache": rope_cache_torch,
  }
  jax_inputs = {
      **{
          name: torch_test_utils.torch_to_jax(tensor)
          for name, tensor in torch_tensors.items()
      },
      **scalar_kwargs,
      "state_cache": (
          None
          if state_cache_torch is None
          else torch_test_utils.torch_to_jax(state_cache_torch)
      ),
      "rope_cache": (
          None
          if rope_cache_torch is None
          else torch_test_utils.torch_to_jax(rope_cache_torch)
      ),
  }
  return torch_inputs, jax_inputs, case_inputs


class Dsv4CompressorBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    torch.manual_seed(0)

  @parameterized.named_parameters(
      (name, name) for name in jax_test_base.CASES
  )
  def test_base_matches_jax_op(self, case_name: str):
    device = "tpu"
    torch_inputs, jax_inputs, case_inputs = _make_torch_and_jax_inputs(
        case_name, device=device
    )

    actual_cache, actual_rope = torch_base.Dsv4Compressor(**torch_inputs)
    desired_cache_jax, desired_rope_jax = jax_base.Dsv4Compressor()(
        **jax_inputs
    )

    desired_cache = torch_test_utils.jax_to_torch(
        desired_cache_jax, device=device
    )
    self.assertEqual(actual_cache.shape, desired_cache.shape)
    self.assertEqual(actual_cache.dtype, desired_cache.dtype)
    torch.testing.assert_close(actual_cache, desired_cache)

    if desired_rope_jax is None:
      self.assertIsNone(actual_rope)
    else:
      self.assertIsNotNone(actual_rope)
      desired_rope = torch_test_utils.jax_to_torch(
          desired_rope_jax, device=device
      )
      self.assertEqual(actual_rope.shape, desired_rope.shape)
      self.assertEqual(actual_rope.dtype, desired_rope.dtype)
      torch.testing.assert_close(actual_rope, desired_rope)

    jax_test_base.assert_matches_reference(
        case_inputs,
        (
            torch_test_utils.torch_to_jax(actual_cache),
            None
            if actual_rope is None
            else torch_test_utils.torch_to_jax(actual_rope),
        ),
    )

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      # CSA mode (head_dim=512, overlap=True): returns (cache, rope_cache).
      cache_csa = torch.empty((4, 256, 128), dtype=torch.int32)
      rope_csa = torch.empty((4, 64, 128), dtype=torch.int32)
      positions = torch.empty((8,), dtype=torch.int32)
      block_table = torch.empty((4,), dtype=torch.int32)
      token_to_req = torch.empty((8,), dtype=torch.int32)
      kv_slots = torch.empty((8,), dtype=torch.int32)
      rms_weight_512 = torch.empty((512,), dtype=torch.float32)
      cos_sin = torch.empty((64, 64), dtype=torch.float32)

      out_cache_csa, out_rope_csa = torch_base.Dsv4Compressor(
          cache_csa,
          positions,
          block_table,
          token_to_req,
          kv_slots,
          rms_weight_512,
          cos_sin,
          block_table_stride=4,
          state_block_size=16,
          compress_ratio=4,
          overlap=True,
          rope_cache=rope_csa,
      )
      self.assertEqual(tuple(out_cache_csa.shape), (4, 256, 128))
      self.assertEqual(out_cache_csa.dtype, torch.int32)
      self.assertIsNotNone(out_rope_csa)
      self.assertEqual(tuple(out_rope_csa.shape), (4, 64, 128))
      self.assertEqual(out_rope_csa.dtype, torch.int32)

      # HCA mode (head_dim=512, overlap=False): returns (cache, None).
      cache_hca = torch.empty((4, 16, 4, 128), dtype=torch.uint8)
      state_hca = torch.empty((4, 256, 4, 128), dtype=torch.uint8)
      out_cache_hca, out_rope_hca = torch_base.Dsv4Compressor(
          cache_hca,
          positions,
          block_table,
          token_to_req,
          kv_slots,
          rms_weight_512,
          cos_sin,
          block_table_stride=4,
          state_block_size=32,
          compress_ratio=128,
          overlap=False,
          state_cache=state_hca,
      )
      self.assertEqual(tuple(out_cache_hca.shape), (4, 16, 4, 128))
      self.assertEqual(out_cache_hca.dtype, torch.uint8)
      self.assertIsNone(out_rope_hca)

      # CSA Indexer mode (head_dim=128, overlap=True): returns (cache, None).
      cache_idx = torch.empty((4, 32, 4, 256), dtype=torch.uint8)
      rms_weight_128 = torch.empty((128,), dtype=torch.float32)
      out_cache_idx, out_rope_idx = torch_base.Dsv4Compressor(
          cache_idx,
          positions,
          block_table,
          token_to_req,
          kv_slots,
          rms_weight_128,
          cos_sin,
          block_table_stride=4,
          state_block_size=4,
          compress_ratio=4,
          overlap=True,
          quant_block=128,
      )
      self.assertEqual(tuple(out_cache_idx.shape), (4, 32, 4, 256))
      self.assertEqual(out_cache_idx.dtype, torch.uint8)
      self.assertIsNone(out_rope_idx)

  def test_fake_impl_symbolic_shapes(self):
    class _Module(torch.nn.Module):

      def forward(
          self,
          cache: torch.Tensor,
          positions: torch.Tensor,
          block_table: torch.Tensor,
          token_to_req_indices: torch.Tensor,
          kv_slot_mapping: torch.Tensor,
          rms_weight: torch.Tensor,
          cos_sin_cache: torch.Tensor,
          rope_cache: torch.Tensor,
      ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return torch_base.Dsv4Compressor(
            cache,
            positions,
            block_table,
            token_to_req_indices,
            kv_slot_mapping,
            rms_weight,
            cos_sin_cache,
            block_table_stride=4,
            state_block_size=16,
            compress_ratio=4,
            overlap=True,
            rope_cache=rope_cache,
        )

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    cache = torch.zeros((4, 256, 128), dtype=torch.int32)
    positions = torch.zeros((8,), dtype=torch.int32)
    block_table = torch.arange(4, dtype=torch.int32)
    token_to_req_indices = torch.zeros((8,), dtype=torch.int32)
    kv_slot_mapping = torch.full((8,), -1, dtype=torch.int32)
    rms_weight = torch.ones((512,), dtype=torch.float32)
    cos_sin_cache = torch.ones((64, 64), dtype=torch.float32)
    rope_cache = torch.zeros((4, 64, 128), dtype=torch.int32)

    exported = torch.export.export(
        _Module(),
        (
            cache,
            positions,
            block_table,
            token_to_req_indices,
            kv_slot_mapping,
            rms_weight,
            cos_sin_cache,
            rope_cache,
        ),
        dynamic_shapes=(
            None,
            {0: num_tokens},
            None,
            {0: num_tokens},
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
        and torch_base.Dsv4Compressor.jax_op_name in str(node.target)
    ]
    self.assertLen(op_nodes, 1)
    out_cache_meta, out_rope_meta = op_nodes[0].meta["val"]
    self.assertEqual(tuple(out_cache_meta.shape), (4, 256, 128))
    self.assertEqual(tuple(out_rope_meta.shape), (4, 64, 128))

  @parameterized.parameters(
      "csa_decode_batch_seq",
      "hca_decode_batch_seq",
      "csa_indexer_decode",
  )
  def test_torch_compile(self, case_name: str):
    device = "tpu"
    torch_inputs, _, _ = _make_torch_and_jax_inputs(case_name, device=device)

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        cache,
        positions,
        block_table,
        token_to_req_indices,
        kv_slot_mapping,
        rms_weight,
        cos_sin_cache,
        state_cache,
        rope_cache,
    ):
      rms_weight = rms_weight + 0.1
      return torch_base.Dsv4Compressor(
          cache,
          positions,
          block_table,
          token_to_req_indices,
          kv_slot_mapping,
          rms_weight,
          cos_sin_cache,
          block_table_stride=torch_inputs["block_table_stride"],
          state_block_size=torch_inputs["state_block_size"],
          compress_ratio=torch_inputs["compress_ratio"],
          overlap=torch_inputs["overlap"],
          state_cache=state_cache,
          rope_cache=rope_cache,
          quant_block=torch_inputs["quant_block"],
          rms_eps=torch_inputs["rms_eps"],
      )

    actual_cache, actual_rope = compiled_fn(
        torch_inputs["cache"].clone(),
        torch_inputs["positions"],
        torch_inputs["block_table"],
        torch_inputs["token_to_req_indices"],
        torch_inputs["kv_slot_mapping"],
        torch_inputs["rms_weight"],
        torch_inputs["cos_sin_cache"],
        torch_inputs["state_cache"],
        None
        if torch_inputs["rope_cache"] is None
        else torch_inputs["rope_cache"].clone(),
    )
    eager_inputs: dict[str, Any] = {
        **torch_inputs,
        "cache": torch_inputs["cache"].clone(),
        "rope_cache": (
            None
            if torch_inputs["rope_cache"] is None
            else torch_inputs["rope_cache"].clone()
        ),
        "rms_weight": torch_inputs["rms_weight"] + 0.1,
    }
    expected_cache, expected_rope = torch_base.Dsv4Compressor(**eager_inputs)

    torch.testing.assert_close(actual_cache, expected_cache)
    if expected_rope is None:
      self.assertIsNone(actual_rope)
    else:
      self.assertIsNotNone(actual_rope)
      torch.testing.assert_close(actual_rope, expected_rope)


if __name__ == "__main__":
  absltest.main()
