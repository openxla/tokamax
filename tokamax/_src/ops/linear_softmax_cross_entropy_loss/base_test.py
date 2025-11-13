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

from functools import partial
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base
from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference
from tokamax._src.ops.linear_softmax_cross_entropy_loss import test_utils


class LinearSoftmaxCrossEntropyLossBaseTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="small_size_sum_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="medium_size_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="sum",
      ),
  )
  def test_reference_running_correctly(self, b_dim, h_dim, v_dim, reduction):
    x, labels, w = test_utils.generate_random_data(
        jax.random.key(42), b_dim, h_dim, v_dim
    )

    ref_loss, ref_lse = (
        reference.linear_softmax_cross_entropy_loss_fwd_reference(
            x, labels, w, reduction=reduction
        )
    )
    dout = 1.0
    ref_grad_x, ref_grad_w = (
        reference.linear_softmax_cross_entropy_loss_bwd_reference(
            dout, ref_lse, x, labels, w, reduction=reduction
        )
    )

    fn = base.LinearSoftmaxCrossEntropyLoss()

    # Forward and Backward
    op_loss, (op_grad_x, op_grad_w) = jax.value_and_grad(fn, argnums=(0, 2))(
        x, labels, w
    )

    self.assertTrue(jnp.allclose(ref_loss, op_loss, atol=1e-5, rtol=1e-5))
    self.assertTrue(jnp.allclose(ref_grad_x, op_grad_x, atol=1e-5, rtol=1e-5))
    self.assertTrue(jnp.allclose(ref_grad_w, op_grad_w, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
  absltest.main()
