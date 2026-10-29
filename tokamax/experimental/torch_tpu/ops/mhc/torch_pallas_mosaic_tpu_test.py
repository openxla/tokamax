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
"""Tests for Pallas Mosaic TPU implementations of mHC ops."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tokamax._src.ops.mhc import base as jax_base
from tokamax._src.ops.mhc import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax._src.ops.mhc import test_base as jax_test_base
from tokamax.experimental.torch_tpu.ops import torch_test_utils
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.mhc import torch_base
from tokamax.experimental.torch_tpu.ops.mhc import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import

Config = jax_pallas_mosaic_tpu.Config


def _make_pre_torch_and_jax_inputs(
    *,
    num_tokens: int = 16,
    hidden_size: int = 256,
    hc_mult: int = jax_test_base.HC_MULT,
    seed: int = 0,
    device: str = "tpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  """Creates PyTorch inputs for `mhc_pre` and converts them to JAX arrays."""
  torch.manual_seed(seed)
  hc_mult3 = hc_mult * (hc_mult + 2)
  residual_torch = torch.randn(
      (num_tokens, hc_mult * hidden_size),
      dtype=torch.float32,
      device=device,
  ).to(torch.bfloat16)
  fn_torch = (
      torch.randn(
          (hc_mult3, hc_mult * hidden_size),
          dtype=torch.float32,
          device=device,
      )
      * 0.02
  )
  hc_scale_torch = 1.0 + 0.1 * torch.randn(
      (3,), dtype=torch.float32, device=device
  )
  hc_base_torch = 0.5 * torch.randn(
      (hc_mult3,), dtype=torch.float32, device=device
  )

  torch_inputs = dict(
      residual=residual_torch,
      fn=fn_torch,
      hc_scale=hc_scale_torch,
      hc_base=hc_base_torch,
  )
  jax_inputs = {
      name: torch_test_utils.torch_to_jax(tensor)
      for name, tensor in torch_inputs.items()
  }
  return torch_inputs, jax_inputs


def _make_post_torch_and_jax_inputs(
    *,
    num_tokens: int = 16,
    hidden_size: int = 256,
    hc_mult: int = jax_test_base.HC_MULT,
    seed: int = 0,
    device: str = "tpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  """Creates PyTorch inputs for `mhc_post` and converts them to JAX arrays."""
  torch.manual_seed(seed)
  x_torch = torch.randn(
      (num_tokens, hidden_size), dtype=torch.float32, device=device
  ).to(torch.bfloat16)
  residual_torch = torch.randn(
      (num_tokens, hc_mult * hidden_size), dtype=torch.float32, device=device
  ).to(torch.bfloat16)
  post_layer_mix_torch = 2.0 * torch.sigmoid(
      torch.randn((num_tokens, hc_mult), dtype=torch.float32, device=device)
  )
  comb_raw = (
      torch.abs(
          torch.randn(
              (num_tokens, hc_mult, hc_mult),
              dtype=torch.float32,
              device=device,
          )
      )
      + 0.01
  )
  comb_res_mix_torch = comb_raw / comb_raw.sum(dim=-1, keepdim=True)

  torch_inputs = dict(
      x=x_torch,
      residual=residual_torch,
      post_layer_mix=post_layer_mix_torch,
      comb_res_mix=comb_res_mix_torch,
  )
  jax_inputs = {
      name: torch_test_utils.torch_to_jax(tensor)
      for name, tensor in torch_inputs.items()
  }
  return torch_inputs, jax_inputs


def _make_fused_torch_and_jax_inputs(
    *,
    num_tokens: int = 16,
    hidden_size: int = 256,
    hc_mult: int = jax_test_base.HC_MULT,
    seed: int = 0,
    device: str = "tpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  """Creates PyTorch inputs for `mhc_fused_post_pre` and converts to JAX."""
  post_torch, post_jax = _make_post_torch_and_jax_inputs(
      num_tokens=num_tokens,
      hidden_size=hidden_size,
      hc_mult=hc_mult,
      seed=seed,
      device=device,
  )
  pre_torch, pre_jax = _make_pre_torch_and_jax_inputs(
      num_tokens=num_tokens,
      hidden_size=hidden_size,
      hc_mult=hc_mult,
      seed=seed + 1,
      device=device,
  )
  torch_inputs = {
      **post_torch,
      "fn": pre_torch["fn"],
      "hc_scale": pre_torch["hc_scale"],
      "hc_base": pre_torch["hc_base"],
  }
  jax_inputs = {
      **post_jax,
      "fn": pre_jax["fn"],
      "hc_scale": pre_jax["hc_scale"],
      "hc_base": pre_jax["hc_base"],
  }
  return torch_inputs, jax_inputs


def _gate_kwargs(
    sinkhorn_repeat: int = jax_test_base.SINKHORN_REPEAT,
) -> dict[str, Any]:
  rms_eps, pre_eps, sk_eps, post_mult, repeat = jax_test_base.gate_constants(
      sinkhorn_repeat
  )
  return dict(
      rms_eps=rms_eps,
      hc_pre_eps=pre_eps,
      hc_sinkhorn_eps=sk_eps,
      hc_post_mult_value=post_mult,
      sinkhorn_repeat=repeat,
  )


class PallasMosaicTpuMhcTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    device = jax.devices()[0]
    if not jax_pallas_mosaic_tpu.PallasTpuMhcPre().supported_on(device):
      self.skipTest(f"Op not supported on {device.device_kind}.")
    torch.manual_seed(0)

  @parameterized.named_parameters(
      dict(
          testcase_name="single_token_h256",
          num_tokens=1,
          hidden_size=256,
          hc_mult=4,
          sinkhorn_repeat=2,
      ),
      dict(
          testcase_name="unaligned_tokens_h256",
          num_tokens=17,
          hidden_size=256,
          hc_mult=4,
          sinkhorn_repeat=20,
      ),
      dict(
          testcase_name="hc_mult_2",
          num_tokens=17,
          hidden_size=256,
          hc_mult=2,
          sinkhorn_repeat=20,
      ),
      dict(
          testcase_name="dsv4_hidden_size",
          num_tokens=16,
          hidden_size=7168,
          hc_mult=4,
          sinkhorn_repeat=20,
      ),
  )
  def test_pre_pallas_mosaic_tpu_matches_jax_op(
      self,
      num_tokens: int,
      hidden_size: int,
      hc_mult: int,
      sinkhorn_repeat: int,
  ):
    device = "tpu"
    torch_inputs, jax_inputs = _make_pre_torch_and_jax_inputs(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        seed=42,
        device=device,
    )
    consts = _gate_kwargs(sinkhorn_repeat)

    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPre(
        **torch_inputs, **consts
    )
    desired_pallas_jax = jax_pallas_mosaic_tpu.PallasTpuMhcPre()(
        **jax_inputs, **consts
    )
    desired_base_jax = jax_base.MhcPre()(**jax_inputs, **consts)

    for act, des_pallas_jax in zip(actual, desired_pallas_jax):
      des_pallas = torch_test_utils.jax_to_torch(des_pallas_jax, device=device)
      self.assertEqual(act.shape, des_pallas.shape)
      self.assertEqual(act.dtype, des_pallas.dtype)
      torch.testing.assert_close(act, des_pallas, rtol=1e-4, atol=1e-5)

    torch.testing.assert_close(
        actual[0],
        torch_test_utils.jax_to_torch(desired_base_jax[0], device=device),
        rtol=jax_test_base.F32_RTOL,
        atol=jax_test_base.F32_ATOL,
    )
    torch.testing.assert_close(
        actual[1],
        torch_test_utils.jax_to_torch(desired_base_jax[1], device=device),
        rtol=jax_test_base.F32_RTOL,
        atol=jax_test_base.F32_ATOL,
    )
    torch.testing.assert_close(
        actual[2],
        torch_test_utils.jax_to_torch(desired_base_jax[2], device=device),
        rtol=jax_test_base.BF16_RTOL,
        atol=jax_test_base.BF16_ATOL,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="single_token_h256",
          num_tokens=1,
          hidden_size=256,
          hc_mult=4,
      ),
      dict(
          testcase_name="unaligned_tokens_h256",
          num_tokens=17,
          hidden_size=256,
          hc_mult=4,
      ),
      dict(
          testcase_name="hc_mult_2",
          num_tokens=17,
          hidden_size=256,
          hc_mult=2,
      ),
      dict(
          testcase_name="dsv4_hidden_size",
          num_tokens=16,
          hidden_size=7168,
          hc_mult=4,
      ),
  )
  def test_post_pallas_mosaic_tpu_matches_jax_op(
      self, num_tokens: int, hidden_size: int, hc_mult: int
  ):
    device = "tpu"
    torch_inputs, jax_inputs = _make_post_torch_and_jax_inputs(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        seed=43,
        device=device,
    )

    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPost(**torch_inputs)
    desired_pallas_jax = jax_pallas_mosaic_tpu.PallasTpuMhcPost()(**jax_inputs)
    desired_base_jax = jax_base.MhcPost()(**jax_inputs)

    desired_pallas = torch_test_utils.jax_to_torch(
        desired_pallas_jax, device=device
    )
    desired_base = torch_test_utils.jax_to_torch(
        desired_base_jax, device=device
    )

    self.assertEqual(actual.shape, desired_pallas.shape)
    self.assertEqual(actual.dtype, desired_pallas.dtype)
    torch.testing.assert_close(actual, desired_pallas, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        actual,
        desired_base,
        rtol=jax_test_base.BF16_RTOL,
        atol=jax_test_base.BF16_ATOL,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="single_token_h256",
          num_tokens=1,
          hidden_size=256,
          hc_mult=4,
          f32_rtol=jax_test_base.FUSED_SMALL_H_RTOL,
      ),
      dict(
          testcase_name="unaligned_tokens_h256",
          num_tokens=17,
          hidden_size=256,
          hc_mult=4,
          f32_rtol=jax_test_base.FUSED_SMALL_H_RTOL,
      ),
      dict(
          testcase_name="hc_mult_2",
          num_tokens=17,
          hidden_size=256,
          hc_mult=2,
          f32_rtol=jax_test_base.FUSED_SMALL_H_RTOL,
      ),
      dict(
          testcase_name="dsv4_hidden_size",
          num_tokens=16,
          hidden_size=7168,
          hc_mult=4,
          f32_rtol=jax_test_base.FUSED_F32_RTOL,
      ),
  )
  def test_fused_post_pre_pallas_mosaic_tpu_matches_jax_op(
      self,
      num_tokens: int,
      hidden_size: int,
      hc_mult: int,
      f32_rtol: float,
  ):
    device = "tpu"
    torch_inputs, jax_inputs = _make_fused_torch_and_jax_inputs(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        seed=44,
        device=device,
    )
    consts = _gate_kwargs()

    actual = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcFusedPostPre(
        **torch_inputs, **consts
    )
    desired_pallas_jax = jax_pallas_mosaic_tpu.PallasTpuMhcFusedPostPre()(
        **jax_inputs, **consts
    )
    desired_base_jax = jax_base.MhcFusedPostPre()(**jax_inputs, **consts)

    for act, des_pallas_jax in zip(actual, desired_pallas_jax):
      des_pallas = torch_test_utils.jax_to_torch(des_pallas_jax, device=device)
      self.assertEqual(act.shape, des_pallas.shape)
      self.assertEqual(act.dtype, des_pallas.dtype)
      torch.testing.assert_close(act, des_pallas, rtol=1e-4, atol=1e-5)

    torch.testing.assert_close(
        actual[0],
        torch_test_utils.jax_to_torch(desired_base_jax[0], device=device),
        rtol=jax_test_base.BF16_RTOL,
        atol=jax_test_base.BF16_ATOL,
    )
    torch.testing.assert_close(
        actual[1],
        torch_test_utils.jax_to_torch(desired_base_jax[1], device=device),
        rtol=f32_rtol,
        atol=jax_test_base.F32_ATOL,
    )
    torch.testing.assert_close(
        actual[2],
        torch_test_utils.jax_to_torch(desired_base_jax[2], device=device),
        rtol=f32_rtol,
        atol=jax_test_base.F32_ATOL,
    )
    torch.testing.assert_close(
        actual[3],
        torch_test_utils.jax_to_torch(desired_base_jax[3], device=device),
        rtol=jax_test_base.BF16_RTOL,
        atol=jax_test_base.BF16_ATOL,
    )

  def test_custom_config(self):
    device = "tpu"
    consts = _gate_kwargs()
    cfg = Config(token_block_size=16)

    pre_torch, pre_jax = _make_pre_torch_and_jax_inputs(
        num_tokens=32, hidden_size=256, seed=13, device=device
    )
    actual_pre = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPre(
        **pre_torch, **consts, configs=(cfg.token_block_size, None)
    )
    desired_pre_jax = jax_pallas_mosaic_tpu.PallasTpuMhcPre(config=cfg)(
        **pre_jax, **consts
    )
    for act, des in zip(actual_pre, desired_pre_jax):
      torch.testing.assert_close(
          act,
          torch_test_utils.jax_to_torch(des, device=device),
          rtol=1e-4,
          atol=1e-5,
      )

    post_torch, post_jax = _make_post_torch_and_jax_inputs(
        num_tokens=32, hidden_size=256, seed=14, device=device
    )
    actual_post = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPost(
        **post_torch, configs=(cfg.token_block_size, None)
    )
    desired_post_jax = jax_pallas_mosaic_tpu.PallasTpuMhcPost(config=cfg)(
        **post_jax
    )
    torch.testing.assert_close(
        actual_post,
        torch_test_utils.jax_to_torch(desired_post_jax, device=device),
        rtol=1e-4,
        atol=1e-5,
    )

    fused_torch, fused_jax = _make_fused_torch_and_jax_inputs(
        num_tokens=32, hidden_size=256, seed=15, device=device
    )
    actual_fused = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcFusedPostPre(
        **fused_torch, **consts, configs=(cfg.token_block_size, None)
    )
    desired_fused_jax = jax_pallas_mosaic_tpu.PallasTpuMhcFusedPostPre(
        config=cfg
    )(**fused_jax, **consts)
    for act, des in zip(actual_fused, desired_fused_jax):
      torch.testing.assert_close(
          act,
          torch_test_utils.jax_to_torch(des, device=device),
          rtol=1e-4,
          atol=1e-5,
      )

  def test_call_without_config_uses_heuristics_config(self):
    device = "tpu"
    consts = _gate_kwargs()

    pre_torch, _ = _make_pre_torch_and_jax_inputs(
        num_tokens=32, hidden_size=256, seed=19, device=device
    )
    op_pre = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPre
    expected_pre_configs = torch_utils.get_configs(
        op_pre,
        *pre_torch.values(),
        **consts,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(expected_pre_configs[0])
    out_pre = op_pre(**pre_torch, **consts)
    expected_out_pre = op_pre(
        **pre_torch,
        **consts,
        configs=(expected_pre_configs[0].token_block_size, None),
    )
    for act, exp in zip(out_pre, expected_out_pre):
      torch.testing.assert_close(act, exp)

    post_torch, _ = _make_post_torch_and_jax_inputs(
        num_tokens=32, hidden_size=256, seed=20, device=device
    )
    op_post = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPost
    expected_post_configs = torch_utils.get_configs(
        op_post,
        *post_torch.values(),
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(expected_post_configs[0])
    out_post = op_post(**post_torch)
    expected_out_post = op_post(
        **post_torch,
        configs=(expected_post_configs[0].token_block_size, None),
    )
    torch.testing.assert_close(out_post, expected_out_post)

    fused_torch, _ = _make_fused_torch_and_jax_inputs(
        num_tokens=32, hidden_size=256, seed=21, device=device
    )
    op_fused = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcFusedPostPre
    expected_fused_configs = torch_utils.get_configs(
        op_fused,
        *fused_torch.values(),
        **consts,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(expected_fused_configs[0])
    out_fused = op_fused(**fused_torch, **consts)
    expected_out_fused = op_fused(
        **fused_torch,
        **consts,
        configs=(expected_fused_configs[0].token_block_size, None),
    )
    for act, exp in zip(out_fused, expected_out_fused):
      torch.testing.assert_close(act, exp)

  def test_torch_compile_with_config(self):
    device = "tpu"
    torch_inputs, _ = _make_fused_torch_and_jax_inputs(
        num_tokens=16, hidden_size=256, seed=23, device=device
    )
    consts = _gate_kwargs()

    op_pre = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPre
    op_post = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPost
    op_fused = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcFusedPostPre

    pre_configs = torch_utils.get_configs(
        op_pre,
        torch_inputs["residual"],
        torch_inputs["fn"],
        torch_inputs["hc_scale"],
        torch_inputs["hc_base"],
        **consts,
        from_autotuning_cache=False,
    )
    post_configs = torch_utils.get_configs(
        op_post,
        torch_inputs["x"],
        torch_inputs["residual"],
        torch_inputs["post_layer_mix"],
        torch_inputs["comb_res_mix"],
        from_autotuning_cache=False,
    )
    fused_configs = torch_utils.get_configs(
        op_fused,
        *torch_inputs.values(),
        **consts,
        from_autotuning_cache=False,
    )
    pre_cfg_tuple = (pre_configs[0].token_block_size, None)
    post_cfg_tuple = (post_configs[0].token_block_size, None)
    fused_cfg_tuple = (fused_configs[0].token_block_size, None)

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base
    ):
      post_mix_1, comb_mix_1, layer_1 = op_pre(
          residual, fn, hc_scale, hc_base, **consts, configs=pre_cfg_tuple
      )
      res_1 = op_post(
          x, residual, post_mix_1, comb_mix_1, configs=post_cfg_tuple
      )
      res_2, post_mix_2, comb_mix_2, layer_2 = op_fused(
          x + layer_1,
          res_1,
          post_layer_mix,
          comb_res_mix,
          fn,
          hc_scale,
          hc_base,
          **consts,
          configs=fused_cfg_tuple,
      )
      return res_2, post_mix_2, comb_mix_2, layer_2

    actual = compiled_fn(**torch_inputs)

    post_mix_1, comb_mix_1, layer_1 = torch_base.MhcPre(
        torch_inputs["residual"],
        torch_inputs["fn"],
        torch_inputs["hc_scale"],
        torch_inputs["hc_base"],
        **consts,
    )
    res_1 = torch_base.MhcPost(
        torch_inputs["x"], torch_inputs["residual"], post_mix_1, comb_mix_1
    )
    expected = torch_base.MhcFusedPostPre(
        torch_inputs["x"] + layer_1,
        res_1,
        torch_inputs["post_layer_mix"],
        torch_inputs["comb_res_mix"],
        torch_inputs["fn"],
        torch_inputs["hc_scale"],
        torch_inputs["hc_base"],
        **consts,
    )

    torch.testing.assert_close(
        actual[0],
        expected[0],
        rtol=jax_test_base.BF16_RTOL,
        atol=jax_test_base.BF16_ATOL,
    )
    torch.testing.assert_close(
        actual[1],
        expected[1],
        rtol=jax_test_base.FUSED_SMALL_H_RTOL,
        atol=jax_test_base.F32_ATOL,
    )
    torch.testing.assert_close(
        actual[2],
        expected[2],
        rtol=jax_test_base.FUSED_SMALL_H_RTOL,
        atol=jax_test_base.F32_ATOL,
    )
    torch.testing.assert_close(
        actual[3],
        expected[3],
        rtol=jax_test_base.BF16_RTOL,
        atol=jax_test_base.BF16_ATOL,
    )

  def test_torch_compile_call_without_config_uses_heuristics_config(self):
    device = "tpu"
    torch_inputs, _ = _make_fused_torch_and_jax_inputs(
        num_tokens=16, hidden_size=256, seed=29, device=device
    )
    consts = _gate_kwargs()

    op_pre = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPre
    op_post = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPost
    op_fused = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcFusedPostPre

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base
    ):
      post_mix_1, comb_mix_1, layer_1 = op_pre(
          residual, fn, hc_scale, hc_base, **consts
      )
      res_1 = op_post(x, residual, post_mix_1, comb_mix_1)
      return op_fused(
          x + layer_1,
          res_1,
          post_layer_mix,
          comb_res_mix,
          fn,
          hc_scale,
          hc_base,
          **consts,
      )

    actual = compiled_fn(**torch_inputs)

    post_mix_1, comb_mix_1, layer_1 = op_pre(
        torch_inputs["residual"],
        torch_inputs["fn"],
        torch_inputs["hc_scale"],
        torch_inputs["hc_base"],
        **consts,
    )
    res_1 = op_post(
        torch_inputs["x"], torch_inputs["residual"], post_mix_1, comb_mix_1
    )
    expected = op_fused(
        torch_inputs["x"] + layer_1,
        res_1,
        torch_inputs["post_layer_mix"],
        torch_inputs["comb_res_mix"],
        torch_inputs["fn"],
        torch_inputs["hc_scale"],
        torch_inputs["hc_base"],
        **consts,
    )
    for act, exp in zip(actual, expected):
      torch.testing.assert_close(act, exp)

  def test_fake_impl_and_symbolic_shapes(self):
    consts = _gate_kwargs()
    op_pre = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPre
    op_post = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPost
    op_fused = torch_pallas_mosaic_tpu.PallasMosaicTpuMhcFusedPostPre

    with fake_tensor.FakeTensorMode():
      x_fake = torch.empty((16, 256), dtype=torch.bfloat16)
      residual_fake = torch.empty((16, 4 * 256), dtype=torch.bfloat16)
      post_mix_fake = torch.empty((16, 4), dtype=torch.float32)
      comb_mix_fake = torch.empty((16, 4, 4), dtype=torch.float32)
      fn_fake = torch.empty((24, 4 * 256), dtype=torch.float32)
      hc_scale_fake = torch.empty((3,), dtype=torch.float32)
      hc_base_fake = torch.empty((24,), dtype=torch.float32)

      pre_out = op_pre(
          residual_fake, fn_fake, hc_scale_fake, hc_base_fake, **consts
      )
      post_out = op_post(x_fake, residual_fake, post_mix_fake, comb_mix_fake)
      fused_out = op_fused(
          x_fake,
          residual_fake,
          post_mix_fake,
          comb_mix_fake,
          fn_fake,
          hc_scale_fake,
          hc_base_fake,
          **consts,
      )

    self.assertEqual(tuple(pre_out[0].shape), (16, 4))
    self.assertEqual(tuple(pre_out[1].shape), (16, 4, 4))
    self.assertEqual(tuple(pre_out[2].shape), (16, 256))
    self.assertEqual(tuple(post_out.shape), (16, 4 * 256))
    self.assertEqual(tuple(fused_out[0].shape), (16, 4 * 256))
    self.assertEqual(tuple(fused_out[1].shape), (16, 4))
    self.assertEqual(tuple(fused_out[2].shape), (16, 4, 4))
    self.assertEqual(tuple(fused_out[3].shape), (16, 256))

    class _Module(torch.nn.Module):

      def forward(
          self,
          x: torch.Tensor,
          residual: torch.Tensor,
          post_layer_mix: torch.Tensor,
          comb_res_mix: torch.Tensor,
          fn: torch.Tensor,
          hc_scale: torch.Tensor,
          hc_base: torch.Tensor,
      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, pre_layer = op_pre(residual, fn, hc_scale, hc_base, **consts)
        post_res = op_post(x, residual, post_layer_mix, comb_res_mix)
        fused_res, _, _, _ = op_fused(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            **consts,
        )
        return pre_layer, post_res, fused_res

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    x = torch.ones((16, 256), dtype=torch.bfloat16)
    residual = torch.ones((16, 4 * 256), dtype=torch.bfloat16)
    post_mix = torch.ones((16, 4), dtype=torch.float32)
    comb_mix = torch.ones((16, 4, 4), dtype=torch.float32)
    fn = torch.ones((24, 4 * 256), dtype=torch.float32)
    hc_scale = torch.ones((3,), dtype=torch.float32)
    hc_base = torch.ones((24,), dtype=torch.float32)

    exported = torch.export.export(
        _Module(),
        (x, residual, post_mix, comb_mix, fn, hc_scale, hc_base),
        dynamic_shapes=(
            {0: num_tokens},
            {0: num_tokens},
            {0: num_tokens},
            {0: num_tokens},
            None,
            None,
            None,
        ),
    )
    for op_obj in (op_pre, op_post, op_fused):
      op_nodes = [
          node
          for node in exported.graph.nodes
          if node.op == "call_function"
          and op_obj.jax_op_name in str(node.target)
      ]
      self.assertLen(op_nodes, 1)
      meta = op_nodes[0].meta["val"]
      first_meta = meta[0] if isinstance(meta, (tuple, list)) else meta
      self.assertIsInstance(first_meta.shape[0], torch.SymInt)

  def test_reconstruct_config(self):
    for op_pallas in (
        torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPre,
        torch_pallas_mosaic_tpu.PallasMosaicTpuMhcPost,
        torch_pallas_mosaic_tpu.PallasMosaicTpuMhcFusedPostPre,
    ):
      with self.subTest(op_name=op_pallas.jax_op_name):
        cfg = Config(token_block_size=32)
        deconstructed = op_pallas.deconstruct_config(cfg)
        self.assertEqual(deconstructed, 32)
        reconstructed = op_pallas.reconstruct_config(deconstructed)
        self.assertEqual(reconstructed, cfg)

        default_cfg = Config()
        self.assertEqual(
            op_pallas.reconstruct_config(
                op_pallas.deconstruct_config(default_cfg)
            ),
            default_cfg,
        )
        self.assertIsNone(op_pallas.deconstruct_config(None))
        with self.assertRaises(AssertionError):
          op_pallas.reconstruct_config(None)


if __name__ == "__main__":
  absltest.main()
