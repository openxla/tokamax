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
from tokamax._src import hlo_utils
from tokamax._src.ops.linear_softmax_cross_entropy_loss import api
from tokamax._src.ops.linear_softmax_cross_entropy_loss import test_utils


class ApiTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="small_size_sum_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="sum",
          test_impl=None,
          reference_impl="xla",
      ),
      dict(
          testcase_name="medium_size_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="sum",
          test_impl=None,
          reference_impl="xla",
      ),
      dict(
          testcase_name="small_size_mean_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=1024,
          reduction="mean",
          test_impl=None,
          reference_impl="xla",
      ),
      dict(
          testcase_name="medium_size_mean_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="mean",
          test_impl=None,
          reference_impl="xla",
      ),
  )
  def test_api_fwd_bwd_matches_reference(
      self,
      b_dim,
      h_dim,
      v_dim,
      reduction,
      test_impl,
      reference_impl,
  ):
    if test_impl == "mosaic_tpu" and jax.default_backend() != "tpu":
      self.skipTest("Pallas Mosaic TPU is only supported on TPUs.")

    x, labels, w = test_utils.generate_random_data(
        jax.random.key(42), b_dim, h_dim, v_dim
    )

    test_loss, (test_x_grad, test_w_grad) = jax.value_and_grad(
        api.linear_softmax_cross_entropy_loss, argnums=(0, 2)
    )(x, labels, w, reduction=reduction, implementation=test_impl)

    ref_loss, (ref_x_grad, ref_w_grad) = jax.value_and_grad(
        api.linear_softmax_cross_entropy_loss, argnums=(0, 2)
    )(x, labels, w, reduction=reduction, implementation=reference_impl)

    self.assertTrue(jnp.allclose(ref_loss, test_loss, atol=1e-4, rtol=1e-4))
    self.assertTrue(jnp.allclose(ref_x_grad, test_x_grad, atol=3e-2, rtol=3e-2))
    self.assertTrue(jnp.allclose(ref_w_grad, test_w_grad, atol=3e-2, rtol=3e-2))

  @parameterized.named_parameters(
      dict(
          testcase_name="small_sum_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="sum",
          test_impl=None,
      ),
      dict(
          testcase_name="medium_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="sum",
          test_impl=None,
      ),
  )
  def test_correct_implementation_used(
      self,
      b_dim,
      h_dim,
      v_dim,
      reduction,
      test_impl,
  ):
    x, labels, w = test_utils.generate_random_data(
        jax.random.key(42), b_dim, h_dim, v_dim
    )

    opspecs = hlo_utils.get_opspecs(
        jax.jit(
            jax.value_and_grad(
                api.linear_softmax_cross_entropy_loss, argnums=(0, 2)
            ),
            static_argnames=[
                "reduction",
                "implementation",
            ],
        ).lower(x, labels, w, reduction=reduction, implementation=test_impl),
        include_xla_kernels=False,
    )

    mosaic_tpu_impl = (
        type(api.IMPLEMENTATIONS.get("mosaic_tpu"))
        if jax.default_backend() == "tpu"
        else None
    )

    if test_impl == "mosaic_tpu":
      self.assertIsInstance(opspecs[0].op, mosaic_tpu_impl)
    else:
      # CPU / GPU are using XLA implementation
      self.assertEmpty(opspecs)


if __name__ == "__main__":
  absltest.main()
