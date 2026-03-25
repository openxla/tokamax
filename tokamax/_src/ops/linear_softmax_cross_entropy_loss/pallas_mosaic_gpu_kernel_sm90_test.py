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
"""Tests for the SM90 Pallas/Mosaic-GPU forward and backward kernel functions.

Covers a range of tile configurations representative of the autotuning search
space (tile_n in {64, 128, 256}, tile_k in {64, 128}, num_stages in {2, 4}).
This ensures that configurations beyond the default (128/128/64) are correct,
which is important for autotuning to produce meaningful results.

SMEM budget (H100: 227 KB):
  forward:  num_stages * (cta_tile_m*tile_k + tile_k*tile_n) * 2 bytes + ~1 KB lse
  backward: 2           * (cta_tile_m*tile_k + tile_k*tile_n) * 2 bytes + cta_tile_m*tile_n*2

For the backward the additional s_smem (cta_tile_m*tile_n*2 = 256*tile_n*2) is the
binding constraint. tile_n=128,tile_k=128 (256 KB) and tile_n=256 (256+ KB) exceed
the 227 KB limit and are not tested here. The forward has no s_smem and supports
tile_n=256 at num_stages=2 (129 KB).
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from tokamax._src import gpu_utils
from tokamax._src.ops.linear_softmax_cross_entropy_loss import (
    pallas_mosaic_gpu_kernel_sm90 as kernel_sm90,
)
from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference
from tokamax._src.ops.linear_softmax_cross_entropy_loss import test_utils


# B=512 is divisible by 2*tile_m=256 for all tile_m=128 configs.
# V=512 is divisible by tile_n in {64, 128, 256}.
# H=256 is divisible by tile_k in {64, 128}.
_B, _H, _V = 512, 256, 512


def _skip_if_not_sm90(test_case):
  if jax.default_backend() != "gpu":
    test_case.skipTest("GPU-only test.")
  if not gpu_utils.has_mosaic_gpu_support():
    test_case.skipTest("Mosaic GPU requires SM90+ (H100 or newer).")


class PallasMosaicGpuSm90FwdKernelTest(parameterized.TestCase):
  """Direct tests of the SM90 forward kernel with various tile configs.

  The forward kernel has no s_smem, so it supports tile_n=256 and
  tile_k=128 at num_stages=2 (193 KB and 129 KB respectively).
  """

  def setUp(self):
    super().setUp()
    _skip_if_not_sm90(self)

  @parameterized.named_parameters(
      dict(
          testcase_name="default",
          tile_m=128, tile_n=128, tile_k=64, num_stages=4,
      ),
      dict(
          testcase_name="small_tile_n",
          tile_m=128, tile_n=64, tile_k=64, num_stages=2,
      ),
      dict(
          testcase_name="large_tile_n",
          tile_m=128, tile_n=256, tile_k=64, num_stages=2,
      ),
      dict(
          testcase_name="large_tile_k",
          tile_m=128, tile_n=128, tile_k=128, num_stages=2,
      ),
  )
  def test_forward_matches_reference(
      self, tile_m, tile_n, tile_k, num_stages,
  ):
    x, labels, w = test_utils.generate_random_data(
        jax.random.key(0), _B, _H, _V
    )

    ref_loss, ref_lse = reference.linear_softmax_cross_entropy_loss_fwd_reference(
        x, labels, w, reduction="sum"
    )
    kernel_loss, kernel_lse = kernel_sm90.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_gpu_sm90(
        x, labels, w,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        num_stages=num_stages, reduction="sum",
    )

    # bf16 WGMMA precision: the forward loss and per-token LSE are insensitive
    # to the bf16 quantization (logsumexp is well-conditioned), so 2e-2 holds.
    self.assertTrue(
        jnp.allclose(ref_loss, kernel_loss.astype(jnp.float32), atol=2e-2, rtol=2e-2),
        msg=f"loss: ref={float(ref_loss):.6f} kernel={float(kernel_loss):.6f}",
    )
    self.assertTrue(
        jnp.allclose(ref_lse, kernel_lse.astype(jnp.float32), atol=2e-2, rtol=2e-2),
        msg=f"lse max_diff={float(jnp.max(jnp.abs(ref_lse - kernel_lse))):.6f}",
    )


class PallasMosaicGpuSm90BwdKernelTest(parameterized.TestCase):
  """Direct tests of the SM90 backward kernel with various tile configs.

  These cases form the autotuning test coverage for the backward pass: they
  verify that the same dimensions produce correct gradients across the range
  of tile sizes the autotuner searches over.

  Backward SMEM: 2*(cta_tile_m*tile_k + tile_k*tile_n)*2 + cta_tile_m*tile_n*2.
  Valid configs at tile_m=128 (cta_tile_m=256):
    tile_n=64,  tile_k=64:  112 KB  (covered: small_tile_n_sum)
    tile_n=64,  tile_k=128: 192 KB  (covered: large_tile_k_sum — note tile_n=64)
    tile_n=128, tile_k=64:  160 KB  (covered: default_*)

  Tolerance notes (see pallas_mosaic_gpu_test.py for full derivation):
    float32, sum: bf16 WGMMA introduces absolute noise up to ~0.2 per
      gradient element, uniform across magnitudes; atol=0.20, rtol=0.05.
    float32, mean: gradients are O(1/B), so element errors are ~B× smaller;
      atol=2e-2 suffices.
  """

  def setUp(self):
    super().setUp()
    _skip_if_not_sm90(self)

  @parameterized.named_parameters(
      dict(
          testcase_name="default_sum",
          tile_m=128, tile_n=128, tile_k=64, num_stages=4, reduction="sum",
      ),
      dict(
          testcase_name="default_mean",
          tile_m=128, tile_n=128, tile_k=64, num_stages=4, reduction="mean",
      ),
      dict(
          testcase_name="few_stages_sum",
          tile_m=128, tile_n=128, tile_k=64, num_stages=2, reduction="sum",
      ),
      dict(
          testcase_name="small_tile_n_sum",
          tile_m=128, tile_n=64, tile_k=64, num_stages=2, reduction="sum",
      ),
      # tile_n=64 is required to keep tile_k=128 within the 227 KB SMEM budget.
      # (tile_n=128, tile_k=128 would need 256 KB.)
      dict(
          testcase_name="large_tile_k_sum",
          tile_m=128, tile_n=64, tile_k=128, num_stages=2, reduction="sum",
      ),
  )
  def test_backward_matches_reference(
      self, tile_m, tile_n, tile_k, num_stages, reduction,
  ):
    x, labels, w = test_utils.generate_random_data(
        jax.random.key(0), _B, _H, _V
    )
    dout = jnp.float32(1.0)

    def ref_fn(x, w):
      loss, _ = reference.linear_softmax_cross_entropy_loss_fwd_reference(
          x, labels, w, reduction=reduction
      )
      return loss

    ref_x_grad, ref_w_grad = jax.grad(ref_fn, argnums=(0, 1))(x, w)

    _, lse = kernel_sm90.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_gpu_sm90(
        x, labels, w,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        num_stages=num_stages, reduction=reduction,
    )
    kernel_x_grad, kernel_w_grad = kernel_sm90.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_gpu_sm90(
        dout, lse, x, labels, w,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        num_stages=num_stages, reduction=reduction,
    )

    if reduction == "sum":
      atol_grad, rtol_grad = 0.20, 0.05
    else:  # mean
      atol_grad, rtol_grad = 2e-2, 2e-2

    self.assertTrue(
        jnp.allclose(
            ref_x_grad.astype(jnp.float32),
            kernel_x_grad.astype(jnp.float32),
            atol=atol_grad,
            rtol=rtol_grad,
        ),
        msg=f"x_grad max_diff={float(jnp.max(jnp.abs(ref_x_grad - kernel_x_grad))):.6f}",
    )
    self.assertTrue(
        jnp.allclose(
            ref_w_grad.astype(jnp.float32),
            kernel_w_grad.astype(jnp.float32),
            atol=atol_grad,
            rtol=rtol_grad,
        ),
        msg=f"w_grad max_diff={float(jnp.max(jnp.abs(ref_w_grad - kernel_w_grad))):.6f}",
    )


if __name__ == "__main__":
  absltest.main()
