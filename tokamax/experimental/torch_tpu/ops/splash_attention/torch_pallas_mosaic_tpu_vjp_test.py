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
"""Tests for Pallas Mosaic TPU Splash Attention VJP PyTorch Op."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import base as jax_base
from tokamax._src.ops.experimental.tpu.splash_attention import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.splash_attention import pallas_mosaic_tpu_vjp as jax_pallas_mosaic_tpu_vjp
from tokamax.experimental.torch_tpu.ops.splash_attention import torch_base
from tokamax.experimental.torch_tpu.ops.splash_attention import torch_pallas_mosaic_tpu_vjp
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


class PallasMosaicTpuSplashAttentionVjpTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")
    torch.manual_seed(0)

  @parameterized.parameters(
      (4, 4, False, True, None),
      (4, 2, False, True, None),
      (4, 1, True, True, None),
      (4, 4, False, False, 30.0),
      (4, 2, False, True, 30.0),
  )
  def test_vjp_matches_jax_vjp_op(
      self, num_q_heads, num_kv_heads, is_mqa, is_causal, attn_logits_soft_cap
  ):
    device = "tpu"
    q_seq_len = 128
    kv_seq_len = 128
    head_dim = 64

    q_torch = torch.randn(
        (num_q_heads, q_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
    )
    kv_shape = (
        (kv_seq_len, head_dim)
        if is_mqa
        else (num_kv_heads, kv_seq_len, head_dim)
    )
    k_torch = torch.randn(
        kv_shape,
        dtype=torch.float32,
        device=device,
    )
    v_torch = torch.randn(
        kv_shape,
        dtype=torch.float32,
        device=device,
    )
    do_torch = torch.randn(
        (num_q_heads, q_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
    )

    q_jax = jnp.array(q_torch.detach().cpu().numpy(), dtype=jnp.float32)
    k_jax = jnp.array(k_torch.detach().cpu().numpy(), dtype=jnp.float32)
    v_jax = jnp.array(v_torch.detach().cpu().numpy(), dtype=jnp.float32)
    do_jax = jnp.array(do_torch.detach().cpu().numpy(), dtype=jnp.float32)

    mask_jax = jax_base.CAUSAL_MASK if is_causal else jax_base.FULL_MASK
    mask_torch = torch_base.CAUSAL_MASK if is_causal else torch_base.FULL_MASK

    fwd_jax_op = jax_pallas_mosaic_tpu.PallasMosaicTpuSplashAttention()
    out_jax, (max_logits_jax, lse_jax) = fwd_jax_op(
        q_jax,
        k_jax,
        v_jax,
        mask=mask_jax,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
        return_residuals=True,
    )

    out_torch = torch.as_tensor(
        np.asarray(out_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    max_logits_torch = torch.as_tensor(
        np.asarray(max_logits_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    lse_torch = torch.as_tensor(
        np.asarray(lse_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )

    dq_torch, dk_torch, dv_torch, _ = (
        torch_pallas_mosaic_tpu_vjp.PallasMosaicTpuSplashAttentionVjp(
            max_logits_torch,
            lse_torch,
            out_torch,
            do_torch,
            q_torch,
            k_torch,
            v_torch,
            mask=mask_torch,
            is_mqa=is_mqa,
            attn_logits_soft_cap=attn_logits_soft_cap,
        )
    )

    vjp_jax_op = jax_pallas_mosaic_tpu_vjp.PallasMosaicTpuSplashAttentionVjp()
    grads_jax = vjp_jax_op(
        (max_logits_jax, lse_jax),
        out_jax,
        do_jax,
        q_jax,
        k_jax,
        v_jax,
        mask=mask_jax,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    dq_desired = torch.as_tensor(
        np.asarray(grads_jax["q"], dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    dk_desired = torch.as_tensor(
        np.asarray(grads_jax["k"], dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    dv_desired = torch.as_tensor(
        np.asarray(grads_jax["v"], dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )

    torch.testing.assert_close(dq_torch, dq_desired, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dk_torch, dk_desired, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dv_torch, dv_desired, rtol=1e-4, atol=1e-4)

  @parameterized.parameters(
      (4, 4, 128, 128, 64, 64, False, torch.float32, jnp.float32),
      (4, 2, 256, 128, 64, 128, False, torch.float32, jnp.float32),
      (4, 1, 128, 256, 64, 64, True, torch.bfloat16, jnp.bfloat16),
  )
  def test_derive_backward_shapes(
      self,
      num_q_heads,
      num_kv_heads,
      q_seq_len,
      kv_seq_len,
      head_dim_qk,
      head_dim_v,
      is_mqa,
      torch_dtype,
      jax_dtype,
  ):
    vjp_op = torch_pallas_mosaic_tpu_vjp.PallasMosaicTpuSplashAttentionVjp

    empty_shapes = vjp_op.derive_backward_shapes({})
    self.assertSameElements(empty_shapes.keys(), ("residuals", "out", "dout"))

    device = "tpu"
    q_torch = torch.randn(
        (num_q_heads, q_seq_len, head_dim_qk),
        dtype=torch_dtype,
        device=device,
    )
    kv_shape_k = (
        (kv_seq_len, head_dim_qk)
        if is_mqa
        else (num_kv_heads, kv_seq_len, head_dim_qk)
    )
    kv_shape_v = (
        (kv_seq_len, head_dim_v)
        if is_mqa
        else (num_kv_heads, kv_seq_len, head_dim_v)
    )
    k_torch = torch.randn(kv_shape_k, dtype=torch_dtype, device=device)
    v_torch = torch.randn(kv_shape_v, dtype=torch_dtype, device=device)

    q_jax = jnp.ones(q_torch.shape, dtype=jax_dtype)
    k_jax = jnp.ones(k_torch.shape, dtype=jax_dtype)
    v_jax = jnp.ones(v_torch.shape, dtype=jax_dtype)

    fwd_jax_op = jax_pallas_mosaic_tpu.PallasMosaicTpuSplashAttention()
    out_jax, (max_logits_jax, lse_jax) = fwd_jax_op(
        q_jax,
        k_jax,
        v_jax,
        mask=jax_base.CAUSAL_MASK,
        is_mqa=is_mqa,
        return_residuals=True,
    )

    abstract_args = {
        "q": jax.ShapeDtypeStruct(q_jax.shape, q_jax.dtype),
        "k": jax.ShapeDtypeStruct(k_jax.shape, k_jax.dtype),
        "v": jax.ShapeDtypeStruct(v_jax.shape, v_jax.dtype),
    }
    derived = vjp_op.derive_backward_shapes(abstract_args)

    self.assertLen(derived["residuals"], 2)
    self.assertEqual(derived["residuals"][0].shape, max_logits_jax.shape)
    self.assertEqual(derived["residuals"][0].dtype, max_logits_jax.dtype)
    self.assertEqual(derived["residuals"][1].shape, lse_jax.shape)
    self.assertEqual(derived["residuals"][1].dtype, lse_jax.dtype)
    self.assertEqual(derived["out"].shape, out_jax.shape)
    self.assertEqual(derived["out"].dtype, out_jax.dtype)
    self.assertEqual(derived["dout"].shape, out_jax.shape)
    self.assertEqual(derived["dout"].dtype, out_jax.dtype)

    ba = vjp_op.get_bound_args(q_torch, k_torch, v_torch, is_mqa=is_mqa)
    self.assertEqual(ba.arguments["residuals"], derived["residuals"])
    self.assertEqual(ba.arguments["out"], derived["out"])
    self.assertEqual(ba.arguments["dout"], derived["dout"])

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      max_logits = torch.empty((4, 128), dtype=torch.float32)
      lse = torch.empty((4, 128), dtype=torch.float32)
      out = torch.empty((4, 128, 64), dtype=torch.float32)
      dout = torch.empty((4, 128, 64), dtype=torch.float32)
      q = torch.empty((4, 128, 64), dtype=torch.float32)
      k = torch.empty((4, 128, 64), dtype=torch.float32)
      v = torch.empty((4, 128, 64), dtype=torch.float32)
      dq, dk, dv, dsinks = (
          torch_pallas_mosaic_tpu_vjp.PallasMosaicTpuSplashAttentionVjp(
              max_logits,
              lse,
              out,
              dout,
              q,
              k,
              v,
              mask=torch_base.CAUSAL_MASK,
          )
      )
      self.assertEqual(dq.shape, (4, 128, 64))
      self.assertEqual(dk.shape, (4, 128, 64))
      self.assertEqual(dv.shape, (4, 128, 64))
      self.assertIsNone(dsinks)

  def test_torch_compile(self):
    device = "tpu"
    max_logits = torch.randn((4, 128), dtype=torch.float32, device=device)
    lse = torch.randn((4, 128), dtype=torch.float32, device=device)
    out = torch.randn((4, 128, 64), dtype=torch.float32, device=device)
    dout = torch.randn((4, 128, 64), dtype=torch.float32, device=device)
    q = torch.randn((4, 128, 64), dtype=torch.float32, device=device)
    k = torch.randn((4, 128, 64), dtype=torch.float32, device=device)
    v = torch.randn((4, 128, 64), dtype=torch.float32, device=device)

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_vjp(max_logits, lse, out, dout, q, k, v):
      return torch_pallas_mosaic_tpu_vjp.PallasMosaicTpuSplashAttentionVjp(
          max_logits,
          lse,
          out,
          dout,
          q,
          k,
          v,
          mask=torch_base.CAUSAL_MASK,
      )

    dq, dk, dv, dsinks = compiled_vjp(max_logits, lse, out, dout, q, k, v)
    self.assertEqual(dq.shape, (4, 128, 64))
    self.assertEqual(dk.shape, (4, 128, 64))
    self.assertEqual(dv.shape, (4, 128, 64))
    self.assertIsNone(dsinks)


if __name__ == "__main__":
  absltest.main()
