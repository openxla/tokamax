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
"""Tests for Pallas Mosaic TPU implementation of GroupedLowrankProj."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from tokamax._src.ops.grouped_lowrank_proj import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax._src.ops.grouped_lowrank_proj import test_base as jax_test_base
from tokamax.experimental.torch_tpu.ops import torch_test_utils
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.grouped_lowrank_proj import torch_base
from tokamax.experimental.torch_tpu.ops.grouped_lowrank_proj import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import

Config = jax_pallas_mosaic_tpu.Config


def _make_torch_and_jax_inputs(
    *,
    num_tokens: int = 256,
    num_groups: int = 8,
    lora_rank: int = 1024,
    head_dim: int = jax_test_base.HEAD_DIM,
    rotary_dim: int = jax_test_base.ROTARY_DIM,
    max_position: int = jax_test_base.MAX_POSITION,
    heads_per_group: int = jax_test_base.HEADS_PER_GROUP,
    seed: int = 0,
    device: str = "tpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Creates JAX inputs and matching PyTorch inputs on `device`."""
  x_jax, positions_jax, cos_sin_cache_jax, wo_a_jax, wo_a_scale_jax = (
      jax_test_base.make_inputs(
          num_tokens=num_tokens,
          num_groups=num_groups,
          lora_rank=lora_rank,
          head_dim=head_dim,
          rotary_dim=rotary_dim,
          max_position=max_position,
          heads_per_group=heads_per_group,
          seed=seed,
      )
  )
  jax_inputs = dict(
      x=x_jax,
      positions=positions_jax,
      cos_sin_cache=cos_sin_cache_jax,
      wo_a=wo_a_jax,
      wo_a_scale=wo_a_scale_jax,
  )
  torch_inputs = dict(
      x=torch_test_utils.jax_to_torch(x_jax, device=device),
      positions=torch_test_utils.jax_to_torch(positions_jax, device=device),
      cos_sin_cache=torch_test_utils.jax_to_torch(
          cos_sin_cache_jax, device=device
      ),
      wo_a=torch.as_tensor(
          np.asarray(wo_a_jax, dtype=np.float32),
          dtype=torch.float8_e4m3fn,
          device=device,
      ),
      wo_a_scale=torch_test_utils.jax_to_torch(wo_a_scale_jax, device=device),
  )
  return torch_inputs, jax_inputs


class PallasMosaicTpuGroupedLowrankProjTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    device = jax.devices()[0]
    if not jax_pallas_mosaic_tpu.PallasTpuGroupedLowrankProj().supported_on(
        device
    ):
      self.skipTest(f"Op not supported on {device.device_kind}.")
    torch.manual_seed(0)

  @parameterized.named_parameters(
      dict(
          testcase_name="t256_g8_r1024_unquant_inv",
          num_tokens=256,
          num_groups=8,
          lora_rank=1024,
          quantize_activations=False,
          inverse=True,
      ),
      dict(
          testcase_name="t256_g8_r1024_quant_inv",
          num_tokens=256,
          num_groups=8,
          lora_rank=1024,
          quantize_activations=True,
          inverse=True,
      ),
      dict(
          testcase_name="t1024_g16_r1024_unquant_inv",
          num_tokens=1024,
          num_groups=16,
          lora_rank=1024,
          quantize_activations=False,
          inverse=True,
      ),
      dict(
          testcase_name="t1024_g16_r1024_quant_inv",
          num_tokens=1024,
          num_groups=16,
          lora_rank=1024,
          quantize_activations=True,
          inverse=True,
      ),
      dict(
          testcase_name="t64_g2_r512_unquant_fwd",
          num_tokens=64,
          num_groups=2,
          lora_rank=512,
          quantize_activations=False,
          inverse=False,
      ),
      dict(
          testcase_name="t64_g2_r512_quant_fwd",
          num_tokens=64,
          num_groups=2,
          lora_rank=512,
          quantize_activations=True,
          inverse=False,
      ),
      dict(
          testcase_name="t1040_g1_r256_unquant_inv",
          num_tokens=1040,
          num_groups=1,
          lora_rank=256,
          quantize_activations=False,
          inverse=True,
      ),
  )
  def test_pallas_mosaic_tpu_matches_jax_op(
      self,
      num_tokens: int,
      num_groups: int,
      lora_rank: int,
      quantize_activations: bool,
      inverse: bool,
  ):
    device = "tpu"
    torch_inputs, jax_inputs = _make_torch_and_jax_inputs(
        num_tokens=num_tokens,
        num_groups=num_groups,
        lora_rank=lora_rank,
        seed=42,
        device=device,
    )

    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuGroupedLowrankProj(
        **torch_inputs,
        inverse=inverse,
        quantize_activations=quantize_activations,
    )
    desired_pallas_jax = jax_pallas_mosaic_tpu.PallasTpuGroupedLowrankProj()(
        **jax_inputs,
        inverse=inverse,
        quantize_activations=quantize_activations,
    )
    desired_pallas = torch_test_utils.jax_to_torch(
        desired_pallas_jax, device=device
    )

    self.assertEqual(actual.shape, (num_tokens, num_groups * lora_rank))
    self.assertEqual(actual.shape, desired_pallas.shape)
    self.assertEqual(actual.dtype, torch.bfloat16)
    self.assertEqual(actual.dtype, desired_pallas.dtype)
    torch.testing.assert_close(actual, desired_pallas, rtol=1e-3, atol=1e-3)
    desired_base = torch_base.GroupedLowrankProj(
        **torch_inputs,
        inverse=inverse,
        quantize_activations=False,
    )
    if quantize_activations:
      jax_test_base.assert_close_quantized(
          torch_test_utils.torch_to_jax(actual),
          torch_test_utils.torch_to_jax(desired_base),
      )
    else:
      jax_test_base.assert_close(
          torch_test_utils.torch_to_jax(actual),
          torch_test_utils.torch_to_jax(desired_base),
      )
    jax_test_base.check_output(
        torch_test_utils.torch_to_jax(actual),
        tuple(jax_inputs.values()),
        inverse=inverse,
        quantize_activations=quantize_activations,
    )

  def test_custom_config(self):
    device = "tpu"
    torch_inputs, jax_inputs = _make_torch_and_jax_inputs(
        num_tokens=256, num_groups=8, lora_rank=1024, seed=13, device=device
    )
    cfg = Config(tile_t=128, tile_r=512, sub_t=64)

    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuGroupedLowrankProj(
        **torch_inputs,
        quantize_activations=True,
        configs=(cfg, None),
    )
    desired_jax = jax_pallas_mosaic_tpu.PallasTpuGroupedLowrankProj(config=cfg)(
        **jax_inputs,
        quantize_activations=True,
    )
    desired = torch_test_utils.jax_to_torch(desired_jax, device=device)

    torch.testing.assert_close(actual, desired, rtol=1e-3, atol=1e-3)
    jax_test_base.check_output(
        torch_test_utils.torch_to_jax(actual),
        tuple(jax_inputs.values()),
        quantize_activations=True,
    )

  def test_autotuning_configs(self):
    device = "tpu"
    torch_inputs, jax_inputs = _make_torch_and_jax_inputs(
        num_tokens=96, num_groups=2, lora_rank=512, seed=17, device=device
    )
    jax_op = jax_pallas_mosaic_tpu.PallasTpuGroupedLowrankProj()
    ba = jax_op.bind(**jax_inputs)
    configs = ba.autotuning_configs
    self.assertGreater(len(configs), 1)

    for quantize_activations in (False, True):
      for cfg in configs:
        with self.subTest(f"{cfg}, {quantize_activations=}"):
          out = torch_pallas_mosaic_tpu.PallasMosaicTpuGroupedLowrankProj(
              **torch_inputs,
              quantize_activations=quantize_activations,
              configs=(cfg, None),
          )
          jax_test_base.check_output(
              torch_test_utils.torch_to_jax(out),
              tuple(jax_inputs.values()),
              quantize_activations=quantize_activations,
          )

  def test_call_without_config_uses_heuristics_config(self):
    device = "tpu"
    torch_inputs, _ = _make_torch_and_jax_inputs(
        num_tokens=96, num_groups=2, lora_rank=512, seed=19, device=device
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuGroupedLowrankProj
    expected_configs = torch_utils.get_configs(
        op_pallas,
        *torch_inputs.values(),
        inverse=True,
        quantize_activations=True,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(expected_configs[0])

    out = op_pallas(
        **torch_inputs, inverse=True, quantize_activations=True
    )
    expected_out = op_pallas(
        **torch_inputs,
        inverse=True,
        quantize_activations=True,
        configs=expected_configs,
    )
    torch.testing.assert_close(out, expected_out)

  def test_torch_compile_with_config(self):
    device = "tpu"
    torch_inputs, _ = _make_torch_and_jax_inputs(
        num_tokens=64, num_groups=2, lora_rank=512, seed=23, device=device
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuGroupedLowrankProj
    configs = torch_utils.get_configs(
        op_pallas,
        *torch_inputs.values(),
        inverse=True,
        quantize_activations=False,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(configs[0])

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, positions, cos_sin_cache, wo_a, wo_a_scale):
      x = x + 0.1
      out = op_pallas(
          x,
          positions,
          cos_sin_cache,
          wo_a,
          wo_a_scale,
          inverse=True,
          quantize_activations=False,
          configs=configs,
      )
      return out + 2.0

    actual = compiled_fn(**torch_inputs)
    expected_pallas = (
        op_pallas(
            **{**torch_inputs, "x": torch_inputs["x"] + 0.1},
            inverse=True,
            quantize_activations=False,
            configs=configs,
        )
        + 2.0
    )
    torch.testing.assert_close(actual, expected_pallas, rtol=1e-3, atol=1e-3)

  def test_torch_compile_call_without_config_uses_heuristics_config(self):
    device = "tpu"
    torch_inputs, _ = _make_torch_and_jax_inputs(
        num_tokens=64, num_groups=2, lora_rank=512, seed=29, device=device
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuGroupedLowrankProj
    expected_configs = torch_utils.get_configs(
        op_pallas,
        *torch_inputs.values(),
        inverse=True,
        quantize_activations=True,
        from_autotuning_cache=False,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, positions, cos_sin_cache, wo_a, wo_a_scale):
      return op_pallas(
          x,
          positions,
          cos_sin_cache,
          wo_a,
          wo_a_scale,
          inverse=True,
          quantize_activations=True,
      )

    actual = compiled_fn(**torch_inputs)
    expected = op_pallas(
        **torch_inputs,
        inverse=True,
        quantize_activations=True,
        configs=expected_configs,
    )
    torch.testing.assert_close(actual, expected)

  def test_fake_impl_and_symbolic_shapes(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuGroupedLowrankProj
    with fake_tensor.FakeTensorMode():
      x_fake = torch.empty((64, 16, 512), dtype=torch.bfloat16)
      positions_fake = torch.empty((64,), dtype=torch.int32)
      cos_sin_cache_fake = torch.empty((4096, 64), dtype=torch.float32)
      wo_a_fake = torch.empty((4096, 1024), dtype=torch.float8_e4m3fn)
      wo_a_scale_fake = torch.empty((1024,), dtype=torch.float32)
      out = op_pallas(
          x_fake,
          positions_fake,
          cos_sin_cache_fake,
          wo_a_fake,
          wo_a_scale_fake,
      )
    self.assertEqual(tuple(out.shape), (64, 1024))
    self.assertEqual(out.dtype, torch.bfloat16)

    class _Module(torch.nn.Module):

      def forward(
          self,
          x: torch.Tensor,
          positions: torch.Tensor,
          cos_sin_cache: torch.Tensor,
          wo_a: torch.Tensor,
          wo_a_scale: torch.Tensor,
      ) -> torch.Tensor:
        return op_pallas(
            x,
            positions,
            cos_sin_cache,
            wo_a,
            wo_a_scale,
        )

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    x = torch.ones((64, 16, 512), dtype=torch.bfloat16)
    positions = torch.zeros((64,), dtype=torch.int32)
    cos_sin_cache = torch.ones((4096, 64), dtype=torch.float32)
    wo_a = torch.ones((4096, 1024), dtype=torch.float32).to(torch.float8_e4m3fn)
    wo_a_scale = torch.ones((1024,), dtype=torch.float32)

    exported = torch.export.export(
        _Module(),
        (x, positions, cos_sin_cache, wo_a, wo_a_scale),
        dynamic_shapes=(
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
    meta = op_nodes[0].meta["val"]
    self.assertIsInstance(meta.shape[0], torch.SymInt)
    self.assertEqual(meta.shape[1], 1024)

  def test_deconstruct_and_reconstruct_config_roundtrip(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuGroupedLowrankProj
    cfg = Config(tile_t=256, tile_r=512, sub_t=128)
    deconstructed = op_pallas.deconstruct_config(cfg)
    self.assertEqual(deconstructed, (256, 512, 128))
    reconstructed = op_pallas.reconstruct_config(deconstructed)
    self.assertEqual(reconstructed, cfg)

    self.assertEqual(op_pallas.reconstruct_config(cfg), cfg)
    self.assertIsNone(op_pallas.deconstruct_config(None))
    with self.assertRaises(ValueError):
      op_pallas.reconstruct_config(None)


if __name__ == "__main__":
  absltest.main()
