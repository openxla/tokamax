# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.linear_softmax_cross_entropy_loss.base import (
    LinearSoftmaxCrossEntropyLoss,
)
from tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu import (
    PallasMosaicTpuLinearSoftmaxCrossEntropyLoss,
    get_tpu_specific_default_config,
)
from tokamax._src.ops.linear_softmax_cross_entropy_loss.test_utils import (
    generate_random_data,
)


class FlashLcePallasMosaicTpuTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @parameterized.named_parameters(
      dict(
          testcase_name="op_wrapper_fwd_bwd_small_size_sum_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="op_wrapper_fwd_bwd_medium_size_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="sum",
      ),
      dict(
          testcase_name="op_wrapper_fwd_bwd_large_size_sum_reduction_test",
          b_dim=16384,
          h_dim=4096,
          v_dim=16384,
          reduction="sum",
      ),
      dict(
          testcase_name="op_wrapper_fwd_bwd_small_size_mean_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="mean",
      ),
      dict(
          testcase_name="op_wrapper_fwd_bwd_medium_size_mean_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="mean",
      ),
      dict(
          testcase_name="op_wrapper_fwd_bwd_large_size_mean_reduction_test",
          b_dim=16384,
          h_dim=4096,
          v_dim=16384,
          reduction="mean",
      ),
  )
  def test_op_wrapper_fwd_bwd(self, b_dim, h_dim, v_dim, reduction):
    x, labels, w = generate_random_data(jax.random.key(42), b_dim, h_dim, v_dim)

    fn_pallas = PallasMosaicTpuLinearSoftmaxCrossEntropyLoss()
    fn_reference = LinearSoftmaxCrossEntropyLoss()

    kernel_loss, (kernel_grad_x, kernel_grad_w) = jax.value_and_grad(
        fn_pallas, argnums=(0, 2)
    )(x, labels, w, reduction=reduction)
    ref_loss, (ref_grad_x, ref_grad_w) = jax.value_and_grad(
        fn_reference, argnums=(0, 2)
    )(x, labels, w, reduction=reduction)

    self.assertTrue(jnp.allclose(ref_loss, kernel_loss, atol=3e-2, rtol=3e-2))
    self.assertTrue(
        jnp.allclose(ref_grad_x, kernel_grad_x, atol=3e-2, rtol=3e-2)
    )
    self.assertTrue(
        jnp.allclose(ref_grad_w, kernel_grad_w, atol=3e-2, rtol=3e-2)
    )


if __name__ == "__main__":
  absltest.main()
