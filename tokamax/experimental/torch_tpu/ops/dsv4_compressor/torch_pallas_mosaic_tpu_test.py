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
"""Tests for Pallas Mosaic TPU implementation of Dsv4Compressor."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.dsv4_compressor import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax._src.ops.dsv4_compressor import test_base as jax_test_base
from tokamax.experimental.torch_tpu.ops import torch_test_utils
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.dsv4_compressor import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import

Config = jax_pallas_mosaic_tpu.Config


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


def _clone_torch_inputs(torch_inputs: dict[str, Any]) -> dict[str, Any]:
  """Returns `torch_inputs` with fresh copies of the donated cache tensors."""
  return {
      **torch_inputs,
      "cache": torch_inputs["cache"].clone(),
      "rope_cache": (
          None
          if torch_inputs["rope_cache"] is None
          else torch_inputs["rope_cache"].clone()
      ),
  }


def _clone_jax_inputs(jax_inputs: dict[str, Any]) -> dict[str, Any]:
  """Returns `jax_inputs` with fresh copies of the donated cache arrays."""
  return {
      **jax_inputs,
      "cache": jnp.copy(jax_inputs["cache"]),
      "rope_cache": (
          None
          if jax_inputs["rope_cache"] is None
          else jnp.copy(jax_inputs["rope_cache"])
      ),
  }


class PallasMosaicTpuDsv4CompressorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    device = jax.devices()[0]
    if not jax_pallas_mosaic_tpu.PallasTpuDsv4Compressor().supported_on(device):
      self.skipTest(f"Op not supported on {device.device_kind}.")
    torch.manual_seed(0)

  @parameterized.named_parameters(
      (name, name) for name in jax_test_base.CASES
  )
  def test_pallas_mosaic_tpu_matches_jax_op(self, case_name: str):
    device = "tpu"
    torch_inputs, jax_inputs, case_inputs = _make_torch_and_jax_inputs(
        case_name, device=device
    )

    actual_cache, actual_rope = (
        torch_pallas_mosaic_tpu.PallasMosaicTpuDsv4Compressor(
            **_clone_torch_inputs(torch_inputs)
        )
    )
    desired_pallas_cache_jax, desired_pallas_rope_jax = (
        jax_pallas_mosaic_tpu.PallasTpuDsv4Compressor()(
            **_clone_jax_inputs(jax_inputs)
        )
    )

    desired_pallas_cache = torch_test_utils.jax_to_torch(
        desired_pallas_cache_jax, device=device
    )
    self.assertEqual(actual_cache.shape, desired_pallas_cache.shape)
    self.assertEqual(actual_cache.dtype, desired_pallas_cache.dtype)
    torch.testing.assert_close(actual_cache, desired_pallas_cache)

    if desired_pallas_rope_jax is None:
      self.assertIsNone(actual_rope)
    else:
      self.assertIsNotNone(actual_rope)
      desired_pallas_rope = torch_test_utils.jax_to_torch(
          desired_pallas_rope_jax, device=device
      )
      self.assertEqual(actual_rope.shape, desired_pallas_rope.shape)
      self.assertEqual(actual_rope.dtype, desired_pallas_rope.dtype)
      torch.testing.assert_close(actual_rope, desired_pallas_rope)

    jax_test_base.assert_matches_reference(
        case_inputs,
        (
            torch_test_utils.torch_to_jax(actual_cache),
            None
            if actual_rope is None
            else torch_test_utils.torch_to_jax(actual_rope),
        ),
    )

  @parameterized.parameters(
      ("csa_prefill", 8),
      ("hca_prefill_small", 8),
      ("csa_indexer_prefill", 16),
  )
  def test_custom_config(self, case_name: str, tile_n: int):
    device = "tpu"
    torch_inputs, jax_inputs, case_inputs = _make_torch_and_jax_inputs(
        case_name, device=device
    )
    cfg = Config(tile_n=tile_n)

    actual_cache, actual_rope = (
        torch_pallas_mosaic_tpu.PallasMosaicTpuDsv4Compressor(
            **_clone_torch_inputs(torch_inputs), configs=(cfg, None)
        )
    )
    desired_cache_jax, desired_rope_jax = (
        jax_pallas_mosaic_tpu.PallasTpuDsv4Compressor(config=cfg)(
            **_clone_jax_inputs(jax_inputs)
        )
    )

    desired_cache = torch_test_utils.jax_to_torch(
        desired_cache_jax, device=device
    )
    torch.testing.assert_close(actual_cache, desired_cache)
    if desired_rope_jax is None:
      self.assertIsNone(actual_rope)
    else:
      self.assertIsNotNone(actual_rope)
      desired_rope = torch_test_utils.jax_to_torch(
          desired_rope_jax, device=device
      )
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

  def test_call_without_config_uses_heuristics_config(self):
    device = "tpu"
    torch_inputs, _, _ = _make_torch_and_jax_inputs(
        "csa_decode_batch_seq", device=device
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuDsv4Compressor
    expected_configs = torch_utils.get_configs(
        op_pallas,
        **torch_inputs,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(expected_configs[0])

    out_cache, out_rope = op_pallas(**_clone_torch_inputs(torch_inputs))
    expected_cache, expected_rope = op_pallas(
        **_clone_torch_inputs(torch_inputs), configs=expected_configs
    )
    torch.testing.assert_close(out_cache, expected_cache)
    self.assertIsNotNone(out_rope)
    self.assertIsNotNone(expected_rope)
    torch.testing.assert_close(out_rope, expected_rope)

  @parameterized.parameters(
      "csa_decode_batch_seq",
      "hca_decode_batch_seq",
      "csa_indexer_decode",
  )
  def test_torch_compile_with_config(self, case_name: str):
    device = "tpu"
    torch_inputs, _, case_inputs = _make_torch_and_jax_inputs(
        case_name, device=device
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuDsv4Compressor
    configs = torch_utils.get_configs(
        op_pallas,
        **torch_inputs,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(configs[0])

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
      return op_pallas(
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
          configs=configs,
      )

    cloned_for_compile = _clone_torch_inputs(torch_inputs)
    actual_cache, actual_rope = compiled_fn(
        cloned_for_compile["cache"],
        cloned_for_compile["positions"],
        cloned_for_compile["block_table"],
        cloned_for_compile["token_to_req_indices"],
        cloned_for_compile["kv_slot_mapping"],
        cloned_for_compile["rms_weight"],
        cloned_for_compile["cos_sin_cache"],
        cloned_for_compile["state_cache"],
        cloned_for_compile["rope_cache"],
    )
    expected_cache, expected_rope = op_pallas(
        **_clone_torch_inputs(torch_inputs), configs=configs
    )

    torch.testing.assert_close(actual_cache, expected_cache)
    if expected_rope is None:
      self.assertIsNone(actual_rope)
    else:
      self.assertIsNotNone(actual_rope)
      torch.testing.assert_close(actual_rope, expected_rope)

    jax_test_base.assert_matches_reference(
        case_inputs,
        (
            torch_test_utils.torch_to_jax(actual_cache),
            None
            if actual_rope is None
            else torch_test_utils.torch_to_jax(actual_rope),
        ),
    )

  def test_torch_compile_call_without_config_uses_heuristics_config(self):
    device = "tpu"
    torch_inputs, _, _ = _make_torch_and_jax_inputs(
        "csa_decode_batch_seq", device=device
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuDsv4Compressor
    expected_configs = torch_utils.get_configs(
        op_pallas,
        **torch_inputs,
        from_autotuning_cache=False,
    )

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
      return op_pallas(
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

    cloned_for_compile = _clone_torch_inputs(torch_inputs)
    actual_cache, actual_rope = compiled_fn(
        cloned_for_compile["cache"],
        cloned_for_compile["positions"],
        cloned_for_compile["block_table"],
        cloned_for_compile["token_to_req_indices"],
        cloned_for_compile["kv_slot_mapping"],
        cloned_for_compile["rms_weight"],
        cloned_for_compile["cos_sin_cache"],
        cloned_for_compile["state_cache"],
        cloned_for_compile["rope_cache"],
    )
    expected_cache, expected_rope = op_pallas(
        **_clone_torch_inputs(torch_inputs), configs=expected_configs
    )
    torch.testing.assert_close(actual_cache, expected_cache)
    self.assertIsNotNone(actual_rope)
    self.assertIsNotNone(expected_rope)
    torch.testing.assert_close(actual_rope, expected_rope)

  def test_fake_impl_and_symbolic_shapes(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuDsv4Compressor
    with fake_tensor.FakeTensorMode():
      cache_fake = torch.empty((4, 256, 128), dtype=torch.int32)
      rope_fake = torch.empty((4, 64, 128), dtype=torch.int32)
      positions_fake = torch.empty((8,), dtype=torch.int32)
      block_table_fake = torch.empty((4,), dtype=torch.int32)
      token_to_req_fake = torch.empty((8,), dtype=torch.int32)
      kv_slots_fake = torch.empty((8,), dtype=torch.int32)
      rms_weight_fake = torch.empty((512,), dtype=torch.float32)
      cos_sin_fake = torch.empty((64, 64), dtype=torch.float32)

      out_cache, out_rope = op_pallas(
          cache_fake,
          positions_fake,
          block_table_fake,
          token_to_req_fake,
          kv_slots_fake,
          rms_weight_fake,
          cos_sin_fake,
          block_table_stride=4,
          state_block_size=16,
          compress_ratio=4,
          overlap=True,
          rope_cache=rope_fake,
      )
    self.assertEqual(tuple(out_cache.shape), (4, 256, 128))
    self.assertEqual(out_cache.dtype, torch.int32)
    self.assertIsNotNone(out_rope)
    self.assertEqual(tuple(out_rope.shape), (4, 64, 128))
    self.assertEqual(out_rope.dtype, torch.int32)

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
        return op_pallas(
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
        and op_pallas.jax_op_name in str(node.target)
    ]
    self.assertLen(op_nodes, 1)
    out_cache_meta, out_rope_meta = op_nodes[0].meta["val"]
    self.assertEqual(tuple(out_cache_meta.shape), (4, 256, 128))
    self.assertEqual(tuple(out_rope_meta.shape), (4, 64, 128))

  def test_deconstruct_and_reconstruct_config_roundtrip(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuDsv4Compressor
    cfg = Config(tile_n=8)
    deconstructed = op_pallas.deconstruct_config(cfg)
    self.assertEqual(deconstructed, (8,))
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
