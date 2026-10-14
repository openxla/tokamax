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
"""A test of the Tokamax PyTorch API."""

from absl.testing import absltest
import jax
import tokamax
import tokamax.torch
import torch
import torch_tpu  # pylint: disable=unused-import


class TokamaxTorchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    seed = absltest.FLAGS.test_random_seed
    if seed is None or not isinstance(seed, int):
      raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)
    torch.manual_seed(seed)

  def test_autotune_not_implemented_for_torch(self):
    with self.assertRaisesRegex(
        NotImplementedError, "Autotuning pytorch ops is not yet supported."
    ):
      tokamax.autotune(
          tokamax.torch.linear_softmax_cross_entropy_loss,
          torch.zeros(2, 2),
      )

  def test_call_op_numerics_matches_reference_implementation(self):
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")

    # Arrange: PyTorch reference implementation.
    def torch_linear_softmax_cross_entropy_loss_ref(
        x, labels, w, reduction="mean"
    ):
      logits = x @ w
      loss = torch.nn.functional.cross_entropy(
          logits, labels.long(), reduction=reduction
      )
      lse = torch.logsumexp(logits, dim=-1)
      return loss, lse

    # Arrange: Data.
    batch_size = 2
    seq_len = 16
    hidden_dim = 64
    vocab_size = 128

    embed_flat_torch = torch.randn(
        batch_size * seq_len, hidden_dim, device="tpu", dtype=torch.float32
    )
    labels_flat_torch = torch.randint(
        0,
        vocab_size,
        (batch_size * seq_len,),
        device="tpu",
        dtype=torch.int32,
    )
    weights_torch = torch.randn(
        hidden_dim, vocab_size, device="tpu", dtype=torch.float32
    )

    # Act: Call tokamax.torch op.
    loss_torch, lse_torch = tokamax.torch.linear_softmax_cross_entropy_loss(
        embed_flat_torch,
        labels_flat_torch,
        weights_torch,
        32,  # b_block_size
        64,  # h_block_size
        128,  # v_block_size
        "mean",
        "float32",
    )

    # Assert: Setup reference values.
    # Call Reference Op (PyTorch)
    loss_ref, lse_ref = torch_linear_softmax_cross_entropy_loss_ref(
        embed_flat_torch, labels_flat_torch, weights_torch, reduction="mean"
    )

    # Assert
    torch.testing.assert_close(loss_torch, loss_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lse_torch, lse_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
