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
"""Tests for Torch mHC base ops."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tokamax._src.ops.mhc import base as jax_base
from tokamax._src.ops.mhc import test_base as jax_test_base
from tokamax.experimental.torch_tpu.ops import torch_test_utils
from tokamax.experimental.torch_tpu.ops.mhc import torch_base
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


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


class MhcBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
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
          num_tokens=16,
          hidden_size=256,
          hc_mult=2,
          sinkhorn_repeat=20,
      ),
  )
  def test_pre_base_matches_jax_op(
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

    actual_post, actual_comb, actual_layer = torch_base.MhcPre(
        **torch_inputs, **consts
    )
    desired_post_jax, desired_comb_jax, desired_layer_jax = jax_base.MhcPre()(
        **jax_inputs, **consts
    )

    desired_post = torch_test_utils.jax_to_torch(
        desired_post_jax, device=device
    )
    desired_comb = torch_test_utils.jax_to_torch(
        desired_comb_jax, device=device
    )
    desired_layer = torch_test_utils.jax_to_torch(
        desired_layer_jax, device=device
    )

    torch.testing.assert_close(
        actual_post,
        desired_post,
        rtol=jax_test_base.F32_RTOL,
        atol=jax_test_base.F32_ATOL,
    )
    torch.testing.assert_close(
        actual_comb,
        desired_comb,
        rtol=jax_test_base.F32_RTOL,
        atol=jax_test_base.F32_ATOL,
    )
    torch.testing.assert_close(
        actual_layer,
        desired_layer,
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
          num_tokens=16,
          hidden_size=256,
          hc_mult=2,
      ),
  )
  def test_post_base_matches_jax_op(
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

    actual = torch_base.MhcPost(**torch_inputs)
    desired_jax = jax_base.MhcPost()(**jax_inputs)
    desired = torch_test_utils.jax_to_torch(desired_jax, device=device)

    torch.testing.assert_close(
        actual,
        desired,
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
          num_tokens=16,
          hidden_size=256,
          hc_mult=2,
      ),
  )
  def test_fused_post_pre_base_matches_jax_op(
      self, num_tokens: int, hidden_size: int, hc_mult: int
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

    actual_res, actual_post, actual_comb, actual_layer = (
        torch_base.MhcFusedPostPre(**torch_inputs, **consts)
    )
    desired_res_jax, desired_post_jax, desired_comb_jax, desired_layer_jax = (
        jax_base.MhcFusedPostPre()(**jax_inputs, **consts)
    )

    torch.testing.assert_close(
        actual_res,
        torch_test_utils.jax_to_torch(desired_res_jax, device=device),
        rtol=jax_test_base.BF16_RTOL,
        atol=jax_test_base.BF16_ATOL,
    )
    torch.testing.assert_close(
        actual_post,
        torch_test_utils.jax_to_torch(desired_post_jax, device=device),
        rtol=jax_test_base.F32_RTOL,
        atol=jax_test_base.F32_ATOL,
    )
    torch.testing.assert_close(
        actual_comb,
        torch_test_utils.jax_to_torch(desired_comb_jax, device=device),
        rtol=jax_test_base.F32_RTOL,
        atol=jax_test_base.F32_ATOL,
    )
    torch.testing.assert_close(
        actual_layer,
        torch_test_utils.jax_to_torch(desired_layer_jax, device=device),
        rtol=jax_test_base.BF16_RTOL,
        atol=jax_test_base.BF16_ATOL,
    )

  def test_fake_impl(self):
    consts = _gate_kwargs()
    with fake_tensor.FakeTensorMode():
      x_fake = torch.empty((16, 256), dtype=torch.bfloat16)
      residual_fake = torch.empty((16, 4 * 256), dtype=torch.bfloat16)
      post_mix_fake = torch.empty((16, 4), dtype=torch.float32)
      comb_mix_fake = torch.empty((16, 4, 4), dtype=torch.float32)
      fn_fake = torch.empty((24, 4 * 256), dtype=torch.float32)
      hc_scale_fake = torch.empty((3,), dtype=torch.float32)
      hc_base_fake = torch.empty((24,), dtype=torch.float32)

      pre_out = torch_base.MhcPre(
          residual_fake, fn_fake, hc_scale_fake, hc_base_fake, **consts
      )
      post_out = torch_base.MhcPost(
          x_fake, residual_fake, post_mix_fake, comb_mix_fake
      )
      fused_out = torch_base.MhcFusedPostPre(
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
    self.assertEqual(pre_out[0].dtype, torch.float32)
    self.assertEqual(tuple(pre_out[1].shape), (16, 4, 4))
    self.assertEqual(pre_out[1].dtype, torch.float32)
    self.assertEqual(tuple(pre_out[2].shape), (16, 256))
    self.assertEqual(pre_out[2].dtype, torch.bfloat16)

    self.assertEqual(tuple(post_out.shape), (16, 4 * 256))
    self.assertEqual(post_out.dtype, torch.bfloat16)

    self.assertEqual(tuple(fused_out[0].shape), (16, 4 * 256))
    self.assertEqual(fused_out[0].dtype, torch.bfloat16)
    self.assertEqual(tuple(fused_out[1].shape), (16, 4))
    self.assertEqual(fused_out[1].dtype, torch.float32)
    self.assertEqual(tuple(fused_out[2].shape), (16, 4, 4))
    self.assertEqual(fused_out[2].dtype, torch.float32)
    self.assertEqual(tuple(fused_out[3].shape), (16, 256))
    self.assertEqual(fused_out[3].dtype, torch.bfloat16)

  def test_fake_impl_symbolic_shapes(self):
    consts = _gate_kwargs()

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
        _, _, pre_layer = torch_base.MhcPre(
            residual, fn, hc_scale, hc_base, **consts
        )
        post_res = torch_base.MhcPost(
            x, residual, post_layer_mix, comb_res_mix
        )
        fused_res, _, _, _ = torch_base.MhcFusedPostPre(
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
    for op_obj in (
        torch_base.MhcPre,
        torch_base.MhcPost,
        torch_base.MhcFusedPostPre,
    ):
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

  def test_torch_compile(self):
    device = "tpu"
    torch_inputs, _ = _make_fused_torch_and_jax_inputs(
        num_tokens=16, hidden_size=256, seed=45, device=device
    )
    consts = _gate_kwargs()

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base
    ):
      post_mix_1, comb_mix_1, layer_1 = torch_base.MhcPre(
          residual, fn, hc_scale, hc_base, **consts
      )
      res_1 = torch_base.MhcPost(x, residual, post_mix_1, comb_mix_1)
      res_2, post_mix_2, comb_mix_2, layer_2 = torch_base.MhcFusedPostPre(
          x + layer_1,
          res_1,
          post_layer_mix,
          comb_res_mix,
          fn,
          hc_scale,
          hc_base,
          **consts,
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

    for act, exp in zip(actual, expected):
      torch.testing.assert_close(act, exp, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  absltest.main()
