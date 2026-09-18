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
"""Tests for Pallas Mosaic TPU implementation of Linear Softmax Cross Entropy Loss."""

from tokamax.experimental.torch_tpu.ops.linear_softmax_cross_entropy import torch_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops.linear_softmax_cross_entropy import torch_base
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax._src.ops import op
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import jax
import jax.numpy as jnp
import torch
import torch_tpu  # pylint: disable=unused-import

Config = jax_pallas_mosaic_tpu.Config


def _lsce_ref(
    x: torch.Tensor,
    labels: torch.Tensor,
    w: torch.Tensor,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
  """Reference implementation for LinearSoftmaxCrossEntropyLoss.

  This serves as a test fixture for comparing numerics.

  Args:
    x: Input tensor.
    labels: Ground truth labels.
    w: Weights tensor.
    reduction: Reduction method.

  Returns:
    A tuple of (loss, lse).
  """
  logits: torch.Tensor = x @ w
  loss: torch.Tensor = torch.nn.functional.cross_entropy(
      logits, labels.long(), reduction=reduction
  )
  lse: torch.Tensor = torch.logsumexp(logits, dim=-1)
  return loss, lse


class PallasMosaicTpuTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    seed = absltest.FLAGS.test_random_seed
    if seed is None or not isinstance(seed, int):
      raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)
    torch.manual_seed(seed)

  def test_lsce_numerics_matches_ref_implementation(self):
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

    fn_reference = (
        jax_pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss()
    )
    loss_ref, (grad_x_ref, grad_w_ref) = jax.value_and_grad(
        fn_reference, argnums=(0, 2)
    )(jax_embed_flat, jax_labels_flat, jax_weights, reduction="mean")

    # Assert
    loss_ref_as_torch = torch.as_tensor(np.asarray(loss_ref), device="tpu")
    grad_x_ref_as_torch = torch.as_tensor(np.asarray(grad_x_ref), device="tpu")
    grad_w_ref_as_torch = torch.as_tensor(np.asarray(grad_w_ref), device="tpu")

    # Act: Call tokamax.torch op (forward).
    loss_tokamax: torch.Tensor
    lse_tokamax: torch.Tensor
    forward_config, backward_config = torch_utils.get_configs(
        torch_pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss,
        embed_flat_torch,
        labels_flat_torch,
        weights_torch,
        reduction="mean",
        from_autotuning_cache=False,
    )

    loss_tokamax, lse_tokamax = (
        torch_pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss(
            embed_flat_torch,
            labels_flat_torch,
            weights_torch,
            reduction="mean",
            configs=(forward_config, backward_config),
        )
    )
    loss_tokamax.backward()
    grad_x_tokamax: torch.Tensor = embed_flat_torch.grad.clone()
    grad_w_tokamax: torch.Tensor = weights_torch.grad.clone()

    embed_flat_torch.grad.zero_()
    weights_torch.grad.zero_()
    loss_ref, lse_ref = _lsce_ref(
        embed_flat_torch, labels_flat_torch, weights_torch, reduction="mean"
    )
    loss_ref.backward()
    grad_x_ref: torch.Tensor = embed_flat_torch.grad
    grad_w_ref: torch.Tensor = weights_torch.grad

    # # Assert
    torch.testing.assert_close(loss_tokamax, loss_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lse_tokamax, lse_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad_x_tokamax, grad_x_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad_w_tokamax, grad_w_ref, rtol=1e-5, atol=1e-5)

    torch.testing.assert_close(
        loss_tokamax, loss_ref_as_torch, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        grad_x_tokamax, grad_x_ref_as_torch, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        grad_w_tokamax, grad_w_ref_as_torch, rtol=1e-5, atol=1e-5
    )

  def _generate_random_data(self, b_dim, h_dim, v_dim):
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
  def test_kernel_running_correctly(self, b_dim, h_dim, v_dim, reduction):

    x, labels, w = self._generate_random_data(b_dim, h_dim, v_dim)
    forward_config, backward_config = torch_utils.get_configs(
        torch_pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss,
        x,
        labels,
        w,
        reduction="mean",
        from_autotuning_cache=False,
    )
    (
        loss,
        lse,
    ) = torch_pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss(
        x,
        labels,
        w,
        reduction=reduction,
        configs=(forward_config, backward_config),
    )
    loss.backward()
    grad_x = x.grad
    grad_w = w.grad
    self.assertIsNotNone(grad_x)
    self.assertIsNotNone(grad_w)
    self.assertIsNotNone(loss)
    self.assertIsNotNone(lse)

    forward_config, backward_config = torch_utils.get_configs(
        torch_base.LinearSoftmaxCrossEntropyLoss,
        x,
        labels,
        w,
        reduction=reduction,
        from_autotuning_cache=False,
    )
    loss_ref, lse_ref = torch_base.LinearSoftmaxCrossEntropyLoss(
        x,
        labels,
        w,
        reduction=reduction,
        configs=(forward_config, backward_config),
    )
    loss_ref.backward()
    grad_x_ref = x.grad
    grad_w_ref = w.grad
    self.assertIsNotNone(grad_x_ref)
    self.assertIsNotNone(grad_w_ref)
    self.assertIsNotNone(loss_ref)
    self.assertIsNotNone(lse_ref)

    torch.testing.assert_close(loss, loss_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad_x, grad_x_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad_w, grad_w_ref, rtol=1e-5, atol=1e-5)

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
  def test_torch_compile(self, b_dim, h_dim, v_dim, reduction):
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")

    x, labels, w = self._generate_random_data(b_dim, h_dim, v_dim)
    forward_config, backward_config = torch_utils.get_configs(
        torch_pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss,
        x,
        labels,
        w,
        reduction="mean",
        from_autotuning_cache=False,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def foo(
        x: torch.Tensor, labels: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
      # pointwise addition before
      x = x + 1.0
      loss, lse = (
          torch_pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss(
              x,
              labels,
              w,
              reduction=reduction,
              configs=(forward_config, backward_config),
          )
      )
      # pointwise addition after
      loss = loss + 2.0
      lse = lse + 3.0
      return loss, lse

    loss_compiled, lse_compiled = foo(x, labels, w)
    loss_compiled.backward()
    grad_x_compiled = x.grad
    grad_w_compiled = w.grad
    self.assertIsNotNone(grad_x_compiled)
    self.assertIsNotNone(grad_w_compiled)
    self.assertIsNotNone(loss_compiled)
    self.assertIsNotNone(lse_compiled)


if __name__ == "__main__":
  absltest.main()
