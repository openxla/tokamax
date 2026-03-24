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
"""End-to-end tests for the Pallas/Triton linear softmax cross-entropy loss Op."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from tokamax._src.ops.linear_softmax_cross_entropy_loss.base import (
    LinearSoftmaxCrossEntropyLoss,
)
from tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_triton import (
    PallasTritonLinearSoftmaxCrossEntropyLoss,
)
from tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_triton_config import (
    Config,
)
from tokamax._src.ops.linear_softmax_cross_entropy_loss.test_utils import (
    generate_random_data,
)


class PallasTritonLceOpTest(parameterized.TestCase):

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
      ),
      dict(
          testcase_name="small_mean",
          b_dim=64,
          h_dim=128,
          v_dim=256,
          reduction="mean",
      ),
      dict(
          testcase_name="medium_sum",
          b_dim=128,
          h_dim=256,
          v_dim=512,
          reduction="sum",
      ),
      dict(
          testcase_name="medium_mean",
          b_dim=128,
          h_dim=256,
          v_dim=512,
          reduction="mean",
      ),
      dict(
          testcase_name="bfloat16",
          b_dim=64,
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
    config = Config(b_block_size=32, h_block_size=64, v_block_size=128)

    triton_op = PallasTritonLinearSoftmaxCrossEntropyLoss(config=config)
    ref_op = LinearSoftmaxCrossEntropyLoss()

    # For bfloat16 compare against a float32-upcast reference (our kernel
    # accumulates in float32 internally).
    x_ref = x.astype(jnp.float32) if dtype == jnp.bfloat16 else x
    w_ref = w.astype(jnp.float32) if dtype == jnp.bfloat16 else w

    kernel_loss, (kernel_x_grad, kernel_w_grad) = jax.value_and_grad(
        triton_op, argnums=(0, 2)
    )(x, labels, w, reduction=reduction)

    ref_loss, (ref_x_grad, ref_w_grad) = jax.value_and_grad(
        ref_op, argnums=(0, 2)
    )(x_ref, labels, w_ref, reduction=reduction)

    # The conftest sets xla_gpu_enable_triton_gemm=False so the reference op
    # uses cuBLAS for x@w while our kernel uses Triton tiled matmul; differences
    # of ~1e-2 are observed for float32 gradients at medium dims (~4e-6 when
    # both use Triton GEMM).
    atol = 2e-2
    rtol = 2e-2

    self.assertTrue(
        jnp.allclose(
            ref_loss.astype(jnp.float32),
            kernel_loss.astype(jnp.float32),
            atol=atol,
            rtol=rtol,
        ),
        msg=f"loss: ref={float(ref_loss):.6f} kernel={float(kernel_loss):.6f}",
    )
    self.assertTrue(
        jnp.allclose(
            ref_x_grad.astype(jnp.float32),
            kernel_x_grad.astype(jnp.float32),
            atol=atol,
            rtol=rtol,
        ),
        msg=f"x_grad max_diff={float(jnp.max(jnp.abs(ref_x_grad.astype(jnp.float32) - kernel_x_grad.astype(jnp.float32)))):.6f}",
    )
    self.assertTrue(
        jnp.allclose(
            ref_w_grad.astype(jnp.float32),
            kernel_w_grad.astype(jnp.float32),
            atol=atol,
            rtol=rtol,
        ),
        msg=f"w_grad max_diff={float(jnp.max(jnp.abs(ref_w_grad.astype(jnp.float32) - kernel_w_grad.astype(jnp.float32)))):.6f}",
    )


if __name__ == "__main__":
  absltest.main()
