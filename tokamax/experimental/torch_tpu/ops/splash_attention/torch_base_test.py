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
"""Tests for the base class of the Splash Attention PyTorch Op."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import base as jax_base
from tokamax.experimental.torch_tpu.ops.splash_attention import torch_base
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


class SplashAttentionBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")
    torch.manual_seed(0)

  @parameterized.parameters(
      (4, 4, 128, 128, 64, False, "causal", None),
      (2, 2, 128, 128, 64, False, "full", None),
      (4, 2, 128, 128, 64, False, "causal", 30.0),
      (4, 1, 128, 128, 64, True, "causal", None),
  )
  def test_forward_and_backward_equivalence_with_jax_op(
      self,
      num_q_heads,
      num_kv_heads,
      q_seq_len,
      kv_seq_len,
      head_dim,
      is_mqa,
      mask_type,
      attn_logits_soft_cap,
  ):
    device = "tpu"
    q_torch = torch.randn(
        (num_q_heads, q_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
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

    if mask_type == "causal":
      mask_torch = torch_base.CAUSAL_MASK
      mask_jax = jax_base.CAUSAL_MASK
    else:
      mask_torch = torch.ones(
          (q_seq_len, kv_seq_len), dtype=torch.bool, device=device
      )
      mask_jax = jnp.ones((q_seq_len, kv_seq_len), dtype=jnp.bool_)

    q_jax = jnp.array(q_torch.detach().cpu().numpy(), dtype=jnp.float32)
    k_jax = jnp.array(k_torch.detach().cpu().numpy(), dtype=jnp.float32)
    v_jax = jnp.array(v_torch.detach().cpu().numpy(), dtype=jnp.float32)
    do_jax = jnp.array(do_torch.detach().cpu().numpy(), dtype=jnp.float32)

    actual_torch = torch_base.SplashAttention(
        q_torch,
        k_torch,
        v_torch,
        mask=mask_torch,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    actual_torch.backward(do_torch)

    jax_op = jax_base.SplashAttention()
    desired_jax, f_vjp = jax.vjp(
        functools.partial(
            jax_op,
            mask=mask_jax,
            is_mqa=is_mqa,
            attn_logits_soft_cap=attn_logits_soft_cap,
        ),
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
    dq_desired_torch = torch.as_tensor(
        np.asarray(dq_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    dk_desired_torch = torch.as_tensor(
        np.asarray(dk_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    dv_desired_torch = torch.as_tensor(
        np.asarray(dv_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )

    self.assertEqual(actual_torch.shape, desired_torch.shape)
    torch.testing.assert_close(
        actual_torch, desired_torch, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        q_torch.grad, dq_desired_torch, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        k_torch.grad, dk_desired_torch, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        v_torch.grad, dv_desired_torch, rtol=1e-4, atol=1e-4
    )

  def test_with_segment_ids(self):
    device = "tpu"
    q_seq_len, head_dim = 128, 64
    num_q_heads = 2

    q_torch = torch.randn(
        (num_q_heads, q_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    k_torch = torch.randn(
        (num_q_heads, q_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    v_torch = torch.randn(
        (num_q_heads, q_seq_len, head_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    q_seg_torch = torch.tensor(
        [0] * 64 + [1] * 64, dtype=torch.int32, device=device
    )
    kv_seg_torch = torch.tensor(
        [0] * 64 + [1] * 64, dtype=torch.int32, device=device
    )
    segment_ids_torch = torch_base.SegmentIds(q=q_seg_torch, kv=kv_seg_torch)

    q_jax = jnp.array(q_torch.detach().cpu().numpy(), dtype=jnp.float32)
    k_jax = jnp.array(k_torch.detach().cpu().numpy(), dtype=jnp.float32)
    v_jax = jnp.array(v_torch.detach().cpu().numpy(), dtype=jnp.float32)
    segment_ids_jax = jax_base.SegmentIds(
        q=jnp.array([0] * 64 + [1] * 64, dtype=jnp.int32),
        kv=jnp.array([0] * 64 + [1] * 64, dtype=jnp.int32),
    )

    out_torch = torch_base.SplashAttention(
        q_torch,
        k_torch,
        v_torch,
        mask=torch_base.CAUSAL_MASK,
        segment_ids=segment_ids_torch,
    )
    out_jax = jax_base.SplashAttention()(
        q_jax,
        k_jax,
        v_jax,
        mask=jax_base.CAUSAL_MASK,
        segment_ids=segment_ids_jax,
    )
    out_desired = torch.as_tensor(
        np.asarray(out_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    torch.testing.assert_close(out_torch, out_desired, rtol=1e-4, atol=1e-4)

  def test_derive_backward_shapes(self):
    vjp_op = torch_base.SplashAttentionVjp
    empty_shapes = vjp_op.derive_backward_shapes({})
    self.assertSameElements(empty_shapes.keys(), ("residuals", "out", "dout"))

    q_jax = jnp.ones((4, 128, 64), dtype=jnp.float32)
    k_jax = jnp.ones((2, 128, 64), dtype=jnp.float32)
    v_jax = jnp.ones((2, 128, 64), dtype=jnp.float32)
    out_jax, (max_logits_jax, lse_jax) = jax_base.SplashAttention()(
        q_jax,
        k_jax,
        v_jax,
        mask=jax_base.CAUSAL_MASK,
        return_residuals=True,
    )
    abstract_args = {
        "q": jax.ShapeDtypeStruct(q_jax.shape, q_jax.dtype),
        "k": jax.ShapeDtypeStruct(k_jax.shape, k_jax.dtype),
        "v": jax.ShapeDtypeStruct(v_jax.shape, v_jax.dtype),
    }
    derived = vjp_op.derive_backward_shapes(abstract_args)
    self.assertEqual(derived["residuals"][0].shape, max_logits_jax.shape)
    self.assertEqual(derived["residuals"][1].shape, lse_jax.shape)
    self.assertEqual(derived["out"].shape, out_jax.shape)
    self.assertEqual(derived["dout"].shape, out_jax.shape)

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      q = torch.empty((4, 128, 64), dtype=torch.float32)
      k = torch.empty((4, 128, 64), dtype=torch.float32)
      v = torch.empty((4, 128, 64), dtype=torch.float32)
      out = torch_base.SplashAttention(q, k, v, mask=torch_base.CAUSAL_MASK)
      self.assertEqual(out.shape, (4, 128, 64))
      self.assertEqual(out.dtype, torch.float32)

  def test_torch_compile(self):
    device = "tpu"
    q_torch = torch.randn(
        (2, 128, 64), dtype=torch.float32, device=device, requires_grad=True
    )
    k_torch = torch.randn(
        (2, 128, 64), dtype=torch.float32, device=device, requires_grad=True
    )
    v_torch = torch.randn(
        (2, 128, 64), dtype=torch.float32, device=device, requires_grad=True
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(q, k, v):
      return torch_base.SplashAttention(q, k, v, mask=torch_base.CAUSAL_MASK)

    out = compiled_fn(q_torch, k_torch, v_torch)
    out.sum().backward()
    self.assertEqual(out.shape, (2, 128, 64))
    self.assertIsNotNone(q_torch.grad)
    self.assertIsNotNone(k_torch.grad)
    self.assertIsNotNone(v_torch.grad)


if __name__ == "__main__":
  absltest.main()
