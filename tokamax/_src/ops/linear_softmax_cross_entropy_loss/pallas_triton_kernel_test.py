# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for pallas_triton_kernel.py (forward and backward passes)."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_triton_kernel as kernel
from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference
from tokamax._src.ops.linear_softmax_cross_entropy_loss import test_utils


class PallasTritonLceFwdKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "gpu":
      self.skipTest("GPU-only test.")

  @parameterized.named_parameters(
      dict(
          testcase_name="small_sum",
          b_dim=64,
          h_dim=128,
          v_dim=256,
          reduction="sum",
          b_block_size=32,
          h_block_size=64,
          v_block_size=128,
      ),
      dict(
          testcase_name="small_mean",
          b_dim=64,
          h_dim=128,
          v_dim=256,
          reduction="mean",
          b_block_size=32,
          h_block_size=64,
          v_block_size=128,
      ),
      dict(
          testcase_name="medium_sum",
          b_dim=128,
          h_dim=256,
          v_dim=512,
          reduction="sum",
          b_block_size=32,
          h_block_size=64,
          v_block_size=128,
      ),
      dict(
          testcase_name="medium_mean",
          b_dim=128,
          h_dim=256,
          v_dim=512,
          reduction="mean",
          b_block_size=32,
          h_block_size=64,
          v_block_size=128,
      ),
      dict(
          testcase_name="bfloat16",
          b_dim=64,
          h_dim=128,
          v_dim=256,
          reduction="sum",
          b_block_size=32,
          h_block_size=64,
          v_block_size=128,
          dtype=jnp.bfloat16,
      ),
  )
  def test_forward_matches_reference(
      self,
      b_dim,
      h_dim,
      v_dim,
      reduction,
      b_block_size,
      h_block_size,
      v_block_size,
      dtype=jnp.float32,
  ):
    x, labels, w = test_utils.generate_random_data(
        jax.random.key(0), b_dim, h_dim, v_dim, dtype=dtype
    )

    ref_loss, ref_lse = reference.linear_softmax_cross_entropy_loss_fwd_reference(
        x, labels, w, reduction=reduction
    )
    kernel_loss, kernel_lse = kernel.linear_softmax_cross_entropy_loss_fwd_pallas_triton(
        x, labels, w,
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=v_block_size,
        reduction=reduction,
    )

    loss_atol = 5e-2 if dtype == jnp.bfloat16 else 1e-4
    loss_rtol = 5e-2 if dtype == jnp.bfloat16 else 1e-4
    # LSE tolerance: the conftest sets xla_gpu_enable_triton_gemm=False so the
    # reference x@w uses cuBLAS while the kernel uses Triton tiled matmul;
    # per-token LSE differs by ~1.2e-2 for float32 at medium dims (~4e-6 when
    # both use Triton GEMM).
    lse_atol = 5e-2 if dtype == jnp.bfloat16 else 2e-2
    lse_rtol = 5e-2 if dtype == jnp.bfloat16 else 2e-2

    self.assertTrue(
        jnp.allclose(ref_loss, kernel_loss, atol=loss_atol, rtol=loss_rtol),
        msg=f"loss mismatch: ref={ref_loss:.6f} kernel={kernel_loss:.6f}",
    )
    self.assertTrue(
        jnp.allclose(ref_lse, kernel_lse, atol=lse_atol, rtol=lse_rtol),
        msg=f"lse mismatch: max_diff={jnp.max(jnp.abs(ref_lse - kernel_lse)):.6f}",
    )


class PallasTritonLceBwdKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "gpu":
      self.skipTest("GPU-only test.")

  @parameterized.named_parameters(
      dict(
          testcase_name="small_sum",
          b_dim=64,
          h_dim=128,
          v_dim=256,
          reduction="sum",
          b_block_size=32,
          h_block_size=64,
          v_block_size=128,
      ),
      dict(
          testcase_name="small_mean",
          b_dim=64,
          h_dim=128,
          v_dim=256,
          reduction="mean",
          b_block_size=32,
          h_block_size=64,
          v_block_size=128,
      ),
      dict(
          testcase_name="medium_sum",
          b_dim=128,
          h_dim=256,
          v_dim=512,
          reduction="sum",
          b_block_size=32,
          h_block_size=64,
          v_block_size=128,
      ),
      dict(
          testcase_name="medium_mean",
          b_dim=128,
          h_dim=256,
          v_dim=512,
          reduction="mean",
          b_block_size=32,
          h_block_size=64,
          v_block_size=128,
      ),
      dict(
          testcase_name="bfloat16",
          b_dim=64,
          h_dim=128,
          v_dim=256,
          reduction="sum",
          b_block_size=32,
          h_block_size=64,
          v_block_size=128,
          dtype=jnp.bfloat16,
      ),
  )
  def test_backward_matches_reference(
      self,
      b_dim,
      h_dim,
      v_dim,
      reduction,
      b_block_size,
      h_block_size,
      v_block_size,
      dtype=jnp.float32,
  ):
    x, labels, w = test_utils.generate_random_data(
        jax.random.key(0), b_dim, h_dim, v_dim, dtype=dtype
    )
    dout = jnp.float32(1.0)

    # Reference: use jax.grad on the reference forward.
    # For bfloat16 inputs, our backward kernel computes in float32 internally
    # (inputs are upcast), so compare against a float32-upcast reference.
    x_ref = x.astype(jnp.float32) if dtype == jnp.bfloat16 else x
    w_ref = w.astype(jnp.float32) if dtype == jnp.bfloat16 else w

    def ref_fn(x, w):
      loss, _ = reference.linear_softmax_cross_entropy_loss_fwd_reference(
          x, labels, w, reduction=reduction
      )
      return loss

    ref_x_grad, ref_w_grad = jax.grad(ref_fn, argnums=(0, 1))(x_ref, w_ref)

    # Kernel: explicit backward call with lse residual from the forward.
    _, lse = kernel.linear_softmax_cross_entropy_loss_fwd_pallas_triton(
        x, labels, w,
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=v_block_size,
        reduction=reduction,
    )
    kernel_x_grad, kernel_w_grad = kernel.linear_softmax_cross_entropy_loss_bwd_pallas_triton(
        dout, lse, x, labels, w,
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=v_block_size,
        reduction=reduction,
    )

    # The conftest sets xla_gpu_enable_triton_gemm=False so the reference
    # uses cuBLAS for x@w while the kernel uses Triton tiled matmul; differences
    # of ~1e-2 are observed for float32 gradients at medium dims (~2e-3 when
    # both use Triton GEMM).
    atol = 2e-2
    rtol = 2e-2

    self.assertTrue(
        jnp.allclose(
            ref_x_grad.astype(jnp.float32),
            kernel_x_grad,
            atol=atol,
            rtol=rtol,
        ),
        msg=f"x_grad mismatch: max_diff={jnp.max(jnp.abs(ref_x_grad.astype(jnp.float32) - kernel_x_grad)):.6f}",
    )
    self.assertTrue(
        jnp.allclose(
            ref_w_grad.astype(jnp.float32),
            kernel_w_grad,
            atol=atol,
            rtol=rtol,
        ),
        msg=f"w_grad mismatch: max_diff={jnp.max(jnp.abs(ref_w_grad.astype(jnp.float32) - kernel_w_grad)):.6f}",
    )


if __name__ == "__main__":
  absltest.main()
