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
"""Tests for Pallas Mosaic TPU implementation of Splash Attention."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import base as jax_base
from tokamax._src.ops.experimental.tpu.splash_attention import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.splash_attention import torch_base
from tokamax.experimental.torch_tpu.ops.splash_attention import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


class PallasMosaicTpuSplashAttentionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")
    torch.manual_seed(0)

  @parameterized.parameters(
      (4, 4, 256, 256, 64, False, None, True),
      (4, 2, 256, 256, 64, False, None, True),
      (8, 2, 128, 128, 64, False, None, True),
      (4, 1, 256, 256, 64, False, None, True),
      (4, 1, 256, 256, 64, True, None, True),
      (4, 1, 256, 256, 64, True, None, False),
      (2, 2, 128, 128, 64, False, 30.0, True),
      (4, 2, 128, 128, 64, False, 30.0, True),
      (4, 2, 128, 256, 64, False, None, True),
      (4, 2, 256, 128, 64, False, None, True),
  )
  def test_pallas_mosaic_tpu_matches_jax_op(
      self,
      num_q_heads,
      num_kv_heads,
      q_seq_len,
      kv_seq_len,
      head_dim,
      is_mqa,
      attn_logits_soft_cap,
      mqa_2d,
  ):
    device = "tpu"
    q_torch = torch.randn(
        (num_q_heads, q_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    if is_mqa and mqa_2d:
      kv_shape = (kv_seq_len, head_dim)
    else:
      kv_shape = (num_kv_heads, kv_seq_len, head_dim)
    k_torch = torch.randn(
        kv_shape,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    v_torch = torch.randn(
        kv_shape,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    do_torch = torch.randn(
        (num_q_heads, q_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
    )

    mask_torch = torch_base.CAUSAL_MASK
    mask_jax = jax_base.CAUSAL_MASK

    q_jax = jnp.array(q_torch.detach().cpu().numpy(), dtype=jnp.float32)
    k_jax = jnp.array(k_torch.detach().cpu().numpy(), dtype=jnp.float32)
    v_jax = jnp.array(v_torch.detach().cpu().numpy(), dtype=jnp.float32)
    do_jax = jnp.array(do_torch.detach().cpu().numpy(), dtype=jnp.float32)

    actual_torch = torch_pallas_mosaic_tpu.PallasMosaicTpuSplashAttention(
        q_torch,
        k_torch,
        v_torch,
        mask=mask_torch,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    actual_torch.backward(do_torch)

    jax_pallas_op = jax_pallas_mosaic_tpu.PallasMosaicTpuSplashAttention()
    desired_pallas_jax, f_vjp_pallas = jax.vjp(
        functools.partial(
            jax_pallas_op,
            mask=mask_jax,
            is_mqa=is_mqa,
            attn_logits_soft_cap=attn_logits_soft_cap,
        ),
        q_jax,
        k_jax,
        v_jax,
    )
    dq_pallas_jax, dk_pallas_jax, dv_pallas_jax = f_vjp_pallas(do_jax)

    desired_pallas_torch = torch.as_tensor(
        np.asarray(desired_pallas_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    dq_pallas_torch = torch.as_tensor(
        np.asarray(dq_pallas_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    dk_pallas_torch = torch.as_tensor(
        np.asarray(dk_pallas_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    dv_pallas_torch = torch.as_tensor(
        np.asarray(dv_pallas_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )

    self.assertEqual(actual_torch.shape, desired_pallas_torch.shape)
    torch.testing.assert_close(
        actual_torch, desired_pallas_torch, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        q_torch.grad, dq_pallas_torch, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        k_torch.grad, dk_pallas_torch, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        v_torch.grad, dv_pallas_torch, rtol=1e-3, atol=1e-3
    )

  @parameterized.parameters(1, 2, 4)
  def test_causal_mask_object(self, num_heads):
    device = "tpu"
    head_dim = 32
    q_seq_len = 128
    kv_seq_len = 128

    q_torch = torch.randn(
        (num_heads, q_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    k_torch = torch.randn(
        (num_heads, kv_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    v_torch = torch.randn(
        (num_heads, kv_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    do_torch = torch.randn(
        (num_heads, q_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
    )
    mask_torch = torch_base.Mask(is_causal=True)
    mask_jax = jax_base.Mask(is_causal=True)

    q_jax = jnp.array(q_torch.detach().cpu().numpy(), dtype=jnp.float32)
    k_jax = jnp.array(k_torch.detach().cpu().numpy(), dtype=jnp.float32)
    v_jax = jnp.array(v_torch.detach().cpu().numpy(), dtype=jnp.float32)
    do_jax = jnp.array(do_torch.detach().cpu().numpy(), dtype=jnp.float32)

    actual_torch = torch_pallas_mosaic_tpu.PallasMosaicTpuSplashAttention(
        q_torch, k_torch, v_torch, mask=mask_torch
    )
    actual_torch.backward(do_torch)

    jax_pallas_op = jax_pallas_mosaic_tpu.PallasMosaicTpuSplashAttention()
    desired_jax, f_vjp = jax.vjp(
        functools.partial(jax_pallas_op, mask=mask_jax),
        q_jax,
        k_jax,
        v_jax,
    )
    dq_jax, dk_jax, dv_jax = f_vjp(do_jax)

    desired_torch = torch.as_tensor(
        np.asarray(desired_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    dq_desired = torch.as_tensor(
        np.asarray(dq_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    dk_desired = torch.as_tensor(
        np.asarray(dk_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    dv_desired = torch.as_tensor(
        np.asarray(dv_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        actual_torch, desired_torch, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(q_torch.grad, dq_desired, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k_torch.grad, dk_desired, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_torch.grad, dv_desired, rtol=1e-3, atol=1e-3)

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      q = torch.empty((4, 128, 64), dtype=torch.float32)
      k = torch.empty((4, 128, 64), dtype=torch.float32)
      v = torch.empty((4, 128, 64), dtype=torch.float32)
      out = torch_pallas_mosaic_tpu.PallasMosaicTpuSplashAttention(
          q, k, v, mask=torch_base.CAUSAL_MASK
      )
      self.assertEqual(out.shape, (4, 128, 64))
      self.assertEqual(out.dtype, torch.float32)

  def test_torch_compile(self):
    device = "tpu"
    q_seq_len = 128
    kv_seq_len = 128
    num_q_heads = 4
    num_kv_heads = 4
    head_dim = 64

    q_torch = torch.randn(
        (num_q_heads, q_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    k_torch = torch.randn(
        (num_kv_heads, kv_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    v_torch = torch.randn(
        (num_kv_heads, kv_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    configs = torch_utils.get_configs(
        torch_pallas_mosaic_tpu.PallasMosaicTpuSplashAttention,
        q_torch,
        k_torch,
        v_torch,
        mask=torch_base.CAUSAL_MASK,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(configs[0])
    self.assertIsNotNone(configs[1])

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(q, k, v):
      q = q + 1.0
      out = torch_pallas_mosaic_tpu.PallasMosaicTpuSplashAttention(
          q, k, v, mask=torch_base.CAUSAL_MASK, configs=configs
      )
      return out + 2.0

    actual = compiled_fn(q_torch, k_torch, v_torch)
    actual.sum().backward()
    self.assertIsNotNone(q_torch.grad)
    self.assertIsNotNone(k_torch.grad)
    self.assertIsNotNone(v_torch.grad)

    expected = (
        torch_base.SplashAttention(
            q_torch.detach() + 1.0,
            k_torch.detach(),
            v_torch.detach(),
            mask=torch_base.CAUSAL_MASK,
        )
        + 2.0
    )
    torch.testing.assert_close(actual, expected, rtol=0.15, atol=0.15)

  def test_torch_compile_call_without_configs_uses_heuristics_config(self):
    device = "tpu"
    q_torch = torch.randn(
        (4, 128, 64), dtype=torch.float32, device=device, requires_grad=True
    )
    k_torch = torch.randn(
        (4, 128, 64), dtype=torch.float32, device=device, requires_grad=True
    )
    v_torch = torch.randn(
        (4, 128, 64), dtype=torch.float32, device=device, requires_grad=True
    )
    q_expected = q_torch.clone().detach().requires_grad_(True)
    k_expected = k_torch.clone().detach().requires_grad_(True)
    v_expected = v_torch.clone().detach().requires_grad_(True)

    op_splash = torch_pallas_mosaic_tpu.PallasMosaicTpuSplashAttention
    expected_configs = torch_utils.get_configs(
        op_splash,
        q_torch,
        k_torch,
        v_torch,
        mask=torch_base.CAUSAL_MASK,
        from_autotuning_cache=False,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(q, k, v):
      return op_splash(q, k, v, mask=torch_base.CAUSAL_MASK)

    out = compiled_fn(q_torch, k_torch, v_torch)
    out.sum().backward()

    expected_out = op_splash(
        q_expected,
        k_expected,
        v_expected,
        mask=torch_base.CAUSAL_MASK,
        configs=expected_configs,
    )
    expected_out.sum().backward()

    torch.testing.assert_close(out, expected_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        q_torch.grad, q_expected.grad, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        k_torch.grad, k_expected.grad, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        v_torch.grad, v_expected.grad, rtol=1e-4, atol=1e-4
    )


if __name__ == "__main__":
  absltest.main()
