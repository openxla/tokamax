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
"""End-to-end tests for the Pallas/Mosaic-GPU linear softmax cross-entropy loss Op."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from tokamax._src import gpu_utils
from tokamax._src.ops.linear_softmax_cross_entropy_loss.base import (
    LinearSoftmaxCrossEntropyLoss,
)
from tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_gpu import (
    PallasMosaicGpuLinearSoftmaxCrossEntropyLoss,
)
from tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_gpu_common import (
    Config,
)
from tokamax._src.ops.linear_softmax_cross_entropy_loss.test_utils import (
    generate_random_data,
)


class PallasMosaicGpuLceOpTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "gpu":
      self.skipTest("GPU-only test.")
    if not gpu_utils.has_mosaic_gpu_support():
      self.skipTest("Mosaic GPU requires SM90+ (H100 or newer).")

  @parameterized.named_parameters(
      dict(
          testcase_name="small_sum",
          b_dim=256,
          h_dim=128,
          v_dim=256,
          reduction="sum",
      ),
      dict(
          testcase_name="small_mean",
          b_dim=256,
          h_dim=128,
          v_dim=256,
          reduction="mean",
      ),
      dict(
          testcase_name="medium_sum",
          b_dim=256,
          h_dim=256,
          v_dim=512,
          reduction="sum",
      ),
      dict(
          testcase_name="medium_mean",
          b_dim=256,
          h_dim=256,
          v_dim=512,
          reduction="mean",
      ),
      dict(
          testcase_name="bfloat16",
          b_dim=256,
          h_dim=128,
          v_dim=256,
          reduction="sum",
          dtype=jnp.bfloat16,
      ),
  )
  def test_value_and_grad_matches_reference(
      self,
      b_dim,
      h_dim,
      v_dim,
      reduction,
      dtype=jnp.float32,
  ):
    x, labels, w = generate_random_data(
        jax.random.key(42), b_dim, h_dim, v_dim, dtype=dtype
    )
    # tile_m=128 so 2*tile_m=256 divides b_dim=256.
    config = Config(tile_m=128, tile_n=128, tile_k=64, num_stages=4)

    mosaic_op = PallasMosaicGpuLinearSoftmaxCrossEntropyLoss(config=config)
    ref_op = LinearSoftmaxCrossEntropyLoss()

    # For bfloat16 compare against float32-upcast reference (kernel accumulates
    # in float32 internally).
    x_ref = x.astype(jnp.float32) if dtype == jnp.bfloat16 else x
    w_ref = w.astype(jnp.float32) if dtype == jnp.bfloat16 else w

    mosaic_loss, (mosaic_x_grad, mosaic_w_grad) = jax.value_and_grad(
        mosaic_op, argnums=(0, 2)
    )(x, labels, w, reduction=reduction)

    ref_loss, (ref_x_grad, ref_w_grad) = jax.value_and_grad(
        ref_op, argnums=(0, 2)
    )(x_ref, labels, w_ref, reduction=reduction)

    # Tolerance notes:
    #
    # bfloat16 inputs: the kernel internally keeps bf16 inputs and the
    # reference is run on float32-upcast values, so errors are modest.
    #
    # float32 inputs with "mean" reduction: scale = dout/B is tiny, so
    # gradient magnitudes are O(1/B) and element-wise absolute errors
    # are proportionally small.
    #
    # float32 inputs with "sum" reduction: the SM90 forward kernel down-casts
    # float32 inputs to bf16 for WGMMA (hardware requirement), which makes the
    # stored lse slightly imprecise. The chunked-scan backward uses float32
    # arithmetic on this bf16-derived lse, which can produce errors up to ~0.35
    # per gradient element vs a fully float32 reference. We use atol=0.40 here
    # (with headroom above the empirical worst-case of ~0.35). The loss scalar
    # has much smaller absolute values and is checked at the tighter 2e-2 level.
    if dtype == jnp.bfloat16:
      atol_grad, rtol_grad = 5e-2, 5e-2
    elif reduction == "sum":
      atol_grad, rtol_grad = 0.40, 0.05
    else:  # float32, mean
      atol_grad, rtol_grad = 2e-2, 2e-2
    atol_loss = 2e-2
    rtol_loss = 2e-2

    self.assertTrue(
        jnp.allclose(
            ref_loss.astype(jnp.float32),
            mosaic_loss.astype(jnp.float32),
            atol=atol_loss,
            rtol=rtol_loss,
        ),
        msg=f"loss: ref={float(ref_loss):.6f} mosaic={float(mosaic_loss):.6f}",
    )
    self.assertTrue(
        jnp.allclose(
            ref_x_grad.astype(jnp.float32),
            mosaic_x_grad.astype(jnp.float32),
            atol=atol_grad,
            rtol=rtol_grad,
        ),
        msg=f"x_grad max_diff={float(jnp.max(jnp.abs(ref_x_grad.astype(jnp.float32) - mosaic_x_grad.astype(jnp.float32)))):.6f}",
    )
    self.assertTrue(
        jnp.allclose(
            ref_w_grad.astype(jnp.float32),
            mosaic_w_grad.astype(jnp.float32),
            atol=atol_grad,
            rtol=rtol_grad,
        ),
        msg=f"w_grad max_diff={float(jnp.max(jnp.abs(ref_w_grad.astype(jnp.float32) - mosaic_w_grad.astype(jnp.float32)))):.6f}",
    )


if __name__ == "__main__":
  absltest.main()
