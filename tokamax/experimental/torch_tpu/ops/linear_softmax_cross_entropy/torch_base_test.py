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
"""Tests for the base class of the Linear Softmax Cross-Entropy Loss PyTorch Op."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.linear_softmax_cross_entropy import torch_base
import torch
import torch_tpu  # pylint: disable=unused-import


class BaseTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    seed = absltest.FLAGS.test_random_seed
    if seed is None or not isinstance(seed, int):
      raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)
    self.seed = seed
    torch.manual_seed(seed)

  def test_lsce_numerics_matches_ref_implementation(self):
    """Tests that the base class numerics matches the reference implementation."""
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")

    # Arrange: Data.
    batch_size = 2
    seq_len = 512
    hidden_dim = 1024
    vocab_size = 2048

    embed_flat_torch: torch.Tensor = torch.randn(
        batch_size * seq_len,
        hidden_dim,
        device="tpu",
        dtype=torch.float32,
        requires_grad=True,
    )
    labels_flat_torch: torch.Tensor = torch.randint(
        0,
        vocab_size,
        (batch_size * seq_len,),
        device="tpu",
        dtype=torch.int32,
    )
    weights_torch: torch.Tensor = torch.randn(
        hidden_dim,
        vocab_size,
        device="tpu",
        dtype=torch.float32,
        requires_grad=True,
    )

    jax_embed_flat: jax.Array = jnp.array(
        embed_flat_torch.detach().cpu().numpy()
    )
    jax_labels_flat: jax.Array = jnp.array(
        labels_flat_torch.detach().cpu().numpy()
    )
    jax_weights: jax.Array = jnp.array(weights_torch.detach().cpu().numpy())

    loss_torch: torch.Tensor
    forward_config, backward_config = torch_utils.get_configs(
        torch_base.LinearSoftmaxCrossEntropyLoss,
        embed_flat_torch,
        labels_flat_torch,
        weights_torch,
        reduction="mean",
        from_autotuning_cache=False,
    )

    loss_torch, _ = torch_base.LinearSoftmaxCrossEntropyLoss(
        embed_flat_torch,
        labels_flat_torch,
        weights_torch,
        reduction="mean",
        configs=(forward_config, backward_config),
    )

    loss_torch.backward()
    grad_x_torch: torch.Tensor = embed_flat_torch.grad.clone()
    grad_w_torch: torch.Tensor = weights_torch.grad.clone()

    fn_reference = jax_base.LinearSoftmaxCrossEntropyLoss()
    loss_ref, (grad_x_ref, grad_w_ref) = jax.value_and_grad(
        fn_reference, argnums=(0, 2)
    )(jax_embed_flat, jax_labels_flat, jax_weights, reduction="mean")

    # Assert
    loss_ref_as_torch = torch.as_tensor(np.asarray(loss_ref), device="tpu")
    grad_x_ref_as_torch = torch.as_tensor(np.asarray(grad_x_ref), device="tpu")
    grad_w_ref_as_torch = torch.as_tensor(np.asarray(grad_w_ref), device="tpu")
    torch.testing.assert_close(
        loss_torch, loss_ref_as_torch, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        grad_x_torch, grad_x_ref_as_torch, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        grad_w_torch, grad_w_ref_as_torch, rtol=1e-5, atol=1e-5
    )


class LinearSoftmaxCrossEntropyLossBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")

    seed = absltest.FLAGS.test_random_seed
    if seed is None or not isinstance(seed, int):
      raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)
    torch.manual_seed(seed)

  def _generate_random_data(self, b_dim: int, h_dim: int, v_dim: int):
    x = torch.randn(
        b_dim,
        h_dim,
        device="tpu",
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.randint(
        0,
        v_dim,
        (b_dim,),
        device="tpu",
        dtype=torch.int32,
    )
    w = torch.randn(
        h_dim,
        v_dim,
        device="tpu",
        dtype=torch.float32,
        requires_grad=True,
    )
    return x, labels, w

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
    x, labels, w = self._generate_random_data(b_dim, h_dim, v_dim)
    configs = torch_utils.get_configs(
        torch_base.LinearSoftmaxCrossEntropyLoss,
        x,
        labels,
        w,
        reduction=reduction,
        from_autotuning_cache=False,
    )
    loss, lse = torch_base.LinearSoftmaxCrossEntropyLoss(
        x, labels, w, reduction=reduction, configs=configs
    )
    loss.backward()
    grad_x = x.grad
    grad_w = w.grad
    self.assertIsNotNone(grad_x)
    self.assertIsNotNone(grad_w)
    self.assertIsNotNone(loss)
    self.assertIsNotNone(lse)


if __name__ == "__main__":
  absltest.main()
