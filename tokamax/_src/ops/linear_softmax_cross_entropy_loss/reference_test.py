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
from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference
from tokamax._src.ops.linear_softmax_cross_entropy_loss import test_utils


class FlashLceReferenceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="ref_fwd_small_size_sum_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=1024,
          reduction="sum",
      ),
      dict(
          testcase_name="ref_fwd_medium_size_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="sum",
      ),
      dict(
          testcase_name="ref_fwd_small_size_mean_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=1024,
          reduction="sum",
      ),
      dict(
          testcase_name="ref_fwd_medium_size_mean_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="sum",
      ),
  )
  def test_reference_fwd_running_correctly(
      self, b_dim, h_dim, v_dim, reduction
  ):
    x, labels, w = test_utils.generate_random_data(
        jax.random.key(42), b_dim, h_dim, v_dim
    )

    self.assertIsNotNone(
        reference.linear_softmax_cross_entropy_loss_fwd_reference(
            x, labels, w, reduction=reduction
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="ref_bwd_small_size_sum_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="ref_bwd_medium_size_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="sum",
      ),
      dict(
          testcase_name="ref_bwd_small_size_mean_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="mean",
      ),
      dict(
          testcase_name="ref_bwd_medium_size_mean_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="mean",
      ),
  )
  def test_reference_bwd_matches_jax_grad(self, b_dim, h_dim, v_dim, reduction):
    x, labels, w = test_utils.generate_random_data(
        jax.random.key(42), b_dim, h_dim, v_dim
    )

    (jax_grad_x, jax_grad_w), _ = jax.grad(
        reference.linear_softmax_cross_entropy_loss_fwd_reference,
        argnums=(0, 2),
        has_aux=True,
    )(x, labels, w, reduction=reduction)

    lse = jax.nn.logsumexp(x @ w, axis=-1)
    dout = 1.0
    ref_grad_x, ref_grad_w = (
        reference.linear_softmax_cross_entropy_loss_bwd_reference(
            dout, lse, x, labels, w, reduction=reduction
        )
    )

    self.assertTrue(jnp.allclose(ref_grad_x, jax_grad_x, atol=3e-2, rtol=3e-2))
    self.assertTrue(jnp.allclose(ref_grad_w, jax_grad_w, atol=3e-2, rtol=3e-2))


if __name__ == "__main__":
  absltest.main()
