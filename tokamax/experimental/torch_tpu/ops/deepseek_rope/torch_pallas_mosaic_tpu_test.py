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
"""Tests for Pallas Mosaic TPU implementation of DeepSeek-V4 RoPE."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
from tokamax._src.ops.deepseek_rope import base as jax_base
from tokamax._src.ops.deepseek_rope import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax._src.ops.deepseek_rope import test_base as jax_test_base
from tokamax.experimental.torch_tpu.ops import torch_test_utils
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.deepseek_rope import torch_base
from tokamax.experimental.torch_tpu.ops.deepseek_rope import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import

Config = jax_pallas_mosaic_tpu.Config


def _make_torch_and_jax_inputs(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    seed: int = 0,
    max_position: int = jax_test_base.MAX_POSITION,
    rotary_dim: int = jax_test_base.ROTARY_DIM,
    device: str = "tpu",
) -> tuple[dict[str, torch.Tensor], dict[str, jax.Array]]:
  """Creates PyTorch inputs on `device` and matching JAX inputs."""
  x_jax, positions_jax, cos_sin_cache_jax = jax_test_base.make_inputs(
      shape,
      dtype,
      seed=seed,
      max_position=max_position,
      rotary_dim=rotary_dim,
  )
  torch_inputs = dict(
      x=torch_test_utils.jax_to_torch(x_jax, device=device),
      positions=torch_test_utils.jax_to_torch(positions_jax, device=device),
      cos_sin_cache=torch_test_utils.jax_to_torch(
          cos_sin_cache_jax, device=device
      ),
  )
  jax_inputs = dict(
      x=x_jax,
      positions=positions_jax,
      cos_sin_cache=cos_sin_cache_jax,
  )
  return torch_inputs, jax_inputs


def _assert_output_matches_jax(
    testcase: parameterized.TestCase,
    mode: str,
    actual: Any,
    desired_jax: Any,
) -> None:
  """Compares PyTorch output against JAX output at the suite's tolerances."""
  if mode == "rope_quant":
    q_actual, scales_actual = actual
    q_desired_jax, scales_desired_jax = desired_jax
    scales_desired = torch_test_utils.jax_to_torch(scales_desired_jax)
    testcase.assertEqual(q_actual.dtype, torch.float8_e4m3fn)
    testcase.assertEqual(scales_actual.dtype, torch.float32)
    testcase.assertEqual(tuple(scales_actual.shape), scales_desired_jax.shape)
    torch.testing.assert_close(
        scales_actual, scales_desired, rtol=1e-6, atol=1e-6
    )
    jax_test_base.assert_bits_equal(
        q_actual.view(torch.uint8).detach().cpu().numpy(),
        q_desired_jax,
    )
    return

  desired = torch_test_utils.jax_to_torch(desired_jax)
  testcase.assertEqual(actual.shape, desired.shape)
  testcase.assertEqual(actual.dtype, desired.dtype)
  tol = 1e-6 if actual.dtype == torch.float32 else 1e-2
  torch.testing.assert_close(actual, desired, rtol=tol, atol=tol)


class PallasMosaicTpuDeepseekRopeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if (
        backend.get_default_device().platform != "tpu"
        or pltpu.get_tpu_info().generation < 7
    ):
      self.skipTest("Only tested on TPU7x and newer.")
    torch.manual_seed(0)

  @parameterized.named_parameters(
      dict(
          testcase_name="rope_multi_head_f32",
          mode="rope",
          shape=(64, jax_test_base.NUM_HEADS, jax_test_base.HEAD_DIM),
          dtype=jnp.float32,
          inverse=False,
      ),
      dict(
          testcase_name="rope_multi_head_f32_inverse",
          mode="rope",
          shape=(64, jax_test_base.NUM_HEADS, jax_test_base.HEAD_DIM),
          dtype=jnp.float32,
          inverse=True,
      ),
      dict(
          testcase_name="rope_single_head_f32",
          mode="rope",
          shape=(128, jax_test_base.HEAD_DIM),
          dtype=jnp.float32,
          inverse=False,
      ),
      dict(
          testcase_name="rope_multi_head_bf16",
          mode="rope",
          shape=(96, 8, 256),
          dtype=jnp.bfloat16,
          inverse=False,
      ),
      dict(
          testcase_name="qnorm_rope_bf16",
          mode="qnorm_rope",
          shape=(64, 128, jax_test_base.HEAD_DIM),
          dtype=jnp.bfloat16,
          inverse=False,
      ),
      dict(
          testcase_name="qnorm_rope_bf16_inverse",
          mode="qnorm_rope",
          shape=(64, 128, jax_test_base.HEAD_DIM),
          dtype=jnp.bfloat16,
          inverse=True,
      ),
      dict(
          testcase_name="rope_quant_3d_bf16",
          mode="rope_quant",
          shape=(64, 64, 128),
          dtype=jnp.bfloat16,
          inverse=False,
      ),
      dict(
          testcase_name="rope_quant_3d_bf16_inverse",
          mode="rope_quant",
          shape=(64, 64, 128),
          dtype=jnp.bfloat16,
          inverse=True,
      ),
      dict(
          testcase_name="rope_quant_2d_bf16",
          mode="rope_quant",
          shape=(256, 128),
          dtype=jnp.bfloat16,
          inverse=False,
      ),
  )
  def test_pallas_mosaic_tpu_matches_jax_op(
      self, mode, shape, dtype, inverse
  ):
    torch_inputs, jax_inputs = _make_torch_and_jax_inputs(
        shape, dtype, seed=42
    )
    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuDeepseekRope(
        **torch_inputs, mode=mode, inverse=inverse
    )
    desired_pallas_jax = jax_pallas_mosaic_tpu.PallasTpuDeepseekRope()(
        x=jnp.copy(jax_inputs["x"]),
        positions=jax_inputs["positions"],
        cos_sin_cache=jax_inputs["cos_sin_cache"],
        mode=mode,
        inverse=inverse,
    )
    desired_base_jax = jax_base.DeepseekRope()(
        **jax_inputs, mode=mode, inverse=inverse
    )

    _assert_output_matches_jax(self, mode, actual, desired_pallas_jax)
    _assert_output_matches_jax(self, mode, actual, desired_base_jax)

  @parameterized.parameters("rope", "qnorm_rope", "rope_quant")
  def test_custom_config(self, mode):
    head_dim = 128 if mode == "rope_quant" else 512
    torch_inputs, jax_inputs = _make_torch_and_jax_inputs(
        (64, 8, head_dim), jnp.bfloat16, seed=13
    )
    cfg = Config(tile_n=16)

    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuDeepseekRope(
        **torch_inputs, mode=mode, config=cfg
    )
    desired_jax = jax_pallas_mosaic_tpu.PallasTpuDeepseekRope(config=cfg)(
        **jax_inputs, mode=mode
    )
    _assert_output_matches_jax(self, mode, actual, desired_jax)

  @parameterized.parameters(
      ("rope", 256, 128),
      ("qnorm_rope", 256, 64),
      ("rope_quant", 256, 128),
  )
  def test_call_without_config_uses_heuristics_config(
      self, mode, num_tokens, expected_tile_n
  ):
    head_dim = 128 if mode == "rope_quant" else 512
    torch_inputs, _ = _make_torch_and_jax_inputs(
        (num_tokens, 8, head_dim), jnp.bfloat16, seed=19
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuDeepseekRope
    expected_config, vjp_config = torch_utils.get_configs(
        op_pallas,
        torch_inputs["x"],
        torch_inputs["positions"],
        torch_inputs["cos_sin_cache"],
        mode=mode,
        from_autotuning_cache=False,
    )
    self.assertEqual(expected_config, Config(tile_n=expected_tile_n))
    self.assertIsNone(vjp_config)

    out = op_pallas(**torch_inputs, mode=mode)
    expected_out = op_pallas(**torch_inputs, mode=mode, config=expected_config)
    if mode == "rope_quant":
      q_out, scales_out = out
      q_expected, scales_expected = expected_out
      torch.testing.assert_close(scales_out, scales_expected)
      jax_test_base.assert_bits_equal(
          q_out.view(torch.uint8).detach().cpu().numpy(),
          q_expected.view(torch.uint8).detach().cpu().numpy(),
      )
    else:
      torch.testing.assert_close(out, expected_out)

  @parameterized.parameters("rope", "qnorm_rope", "rope_quant")
  def test_torch_compile_with_config(self, mode):
    head_dim = 128 if mode == "rope_quant" else 512
    torch_inputs, _ = _make_torch_and_jax_inputs(
        (64, 8, head_dim), jnp.bfloat16, seed=23
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuDeepseekRope
    configs = torch_utils.get_configs(
        op_pallas,
        torch_inputs["x"],
        torch_inputs["positions"],
        torch_inputs["cos_sin_cache"],
        mode=mode,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(configs[0])

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, positions, cos_sin_cache):
      x = x + 0.1
      return op_pallas(
          x, positions, cos_sin_cache, mode=mode, configs=configs
      )

    actual = compiled_fn(**torch_inputs)
    expected = torch_base.DeepseekRope(
        **{**torch_inputs, "x": torch_inputs["x"] + 0.1},
        mode=mode,
    )
    if mode == "rope_quant":
      q_actual, scales_actual = actual
      q_expected, scales_expected = expected
      torch.testing.assert_close(
          scales_actual, scales_expected, rtol=1e-6, atol=1e-6
      )
      jax_test_base.assert_bits_equal(
          q_actual.view(torch.uint8).detach().cpu().numpy(),
          q_expected.view(torch.uint8).detach().cpu().numpy(),
      )
    else:
      torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

  @parameterized.parameters("rope", "qnorm_rope", "rope_quant")
  def test_torch_compile_call_without_config_uses_heuristics_config(self, mode):
    head_dim = 128 if mode == "rope_quant" else 512
    torch_inputs, _ = _make_torch_and_jax_inputs(
        (64, 8, head_dim), jnp.bfloat16, seed=29
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuDeepseekRope
    expected_config, _ = torch_utils.get_configs(
        op_pallas,
        torch_inputs["x"],
        torch_inputs["positions"],
        torch_inputs["cos_sin_cache"],
        mode=mode,
        from_autotuning_cache=False,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, positions, cos_sin_cache):
      return op_pallas(x, positions, cos_sin_cache, mode=mode)

    actual = compiled_fn(**torch_inputs)
    expected = op_pallas(**torch_inputs, mode=mode, config=expected_config)
    if mode == "rope_quant":
      q_actual, scales_actual = actual
      q_expected, scales_expected = expected
      torch.testing.assert_close(scales_actual, scales_expected)
      jax_test_base.assert_bits_equal(
          q_actual.view(torch.uint8).detach().cpu().numpy(),
          q_expected.view(torch.uint8).detach().cpu().numpy(),
      )
    else:
      torch.testing.assert_close(actual, expected)

  def test_fake_impl_and_symbolic_shapes(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuDeepseekRope
    with fake_tensor.FakeTensorMode():
      x_fake = torch.empty((16, 8, 512), dtype=torch.bfloat16)
      x_quant_fake = torch.empty((16, 8, 128), dtype=torch.bfloat16)
      positions_fake = torch.empty((16,), dtype=torch.int32)
      cos_sin_fake = torch.empty((4096, 64), dtype=torch.float32)

      out = op_pallas(x_fake, positions_fake, cos_sin_fake, mode="rope")
      q_out, scales_out = op_pallas(
          x_quant_fake, positions_fake, cos_sin_fake, mode="rope_quant"
      )

    self.assertEqual(tuple(out.shape), (16, 8, 512))
    self.assertEqual(out.dtype, torch.bfloat16)
    self.assertEqual(tuple(q_out.shape), (16, 8, 128))
    self.assertEqual(q_out.dtype, torch.float8_e4m3fn)
    self.assertEqual(tuple(scales_out.shape), (16, 8))
    self.assertEqual(scales_out.dtype, torch.float32)

    class _Module(torch.nn.Module):

      def forward(
          self,
          x: torch.Tensor,
          positions: torch.Tensor,
          cos_sin_cache: torch.Tensor,
      ) -> torch.Tensor:
        return op_pallas(x, positions, cos_sin_cache, mode="rope")

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    x = torch.ones((16, 8, 512), dtype=torch.bfloat16)
    positions = torch.zeros((16,), dtype=torch.int32)
    cos_sin_cache = torch.ones((4096, 64), dtype=torch.float32)

    exported = torch.export.export(
        _Module(),
        (x, positions, cos_sin_cache),
        dynamic_shapes=({0: num_tokens}, {0: num_tokens}, None),
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
    self.assertEqual(tuple(meta.shape[1:]), (8, 512))

  def test_deconstruct_and_reconstruct_config_roundtrip(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuDeepseekRope
    cfg = Config(tile_n=64)
    deconstructed = op_pallas.deconstruct_config(cfg)
    self.assertEqual(deconstructed, (64,))
    reconstructed = op_pallas.reconstruct_config(deconstructed)
    self.assertEqual(reconstructed, cfg)

    self.assertEqual(op_pallas.reconstruct_config(cfg), cfg)
    self.assertIsNone(op_pallas.deconstruct_config(None))
    with self.assertRaises(ValueError):
      op_pallas.reconstruct_config(None)


if __name__ == "__main__":
  absltest.main()
