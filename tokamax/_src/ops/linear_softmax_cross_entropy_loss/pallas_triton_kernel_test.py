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
"""Tests for pallas_triton_kernel.py (forward pass only)."""

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
    # LSE tolerance is looser: the reference uses cuBLAS (xla_gpu_enable_triton_gemm=False
    # in conftest) while the kernel uses Triton tiled accumulation, so per-token lse
    # values can differ by ~O(1e-2) even for float32 at medium dimensions.
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


if __name__ == "__main__":
  absltest.main()
