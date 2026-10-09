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
"""Tests for the base class of the TopK PyTorch Op."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.topk import base as jax_base
from tokamax.experimental.torch_tpu.ops.topk import torch_base
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


class TopKBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")
    torch.manual_seed(0)

  @parameterized.parameters(
      ((2, 128), 16, False, False),
      ((4, 256), 16, False, True),
      ((4, 128), 16, True, False),
      ((4, 256), 16, True, True),
  )
  def test_equivalence_with_jax_op(
      self, shape, k, with_row_lengths, return_scores
  ):
    device = "tpu"
    b, n = shape
    rng = np.random.default_rng(42)
    scores_np = rng.standard_normal(shape, dtype=np.float32)

    scores_torch = torch.as_tensor(
        scores_np, dtype=torch.float32, device=device
    )
    scores_jax = jnp.asarray(scores_np, dtype=jnp.float32)

    if with_row_lengths:
      row_lengths_np = rng.integers(k // 2, n + 1, size=(b,), dtype=np.int32)
      row_lengths_torch = torch.as_tensor(
          row_lengths_np, dtype=torch.int32, device=device
      )
      row_lengths_jax = jnp.asarray(row_lengths_np, dtype=jnp.int32)
    else:
      row_lengths_torch = None
      row_lengths_jax = None

    actual_idx, actual_scores_bits = torch_base.TopK(
        scores_torch, k, row_lengths_torch, return_scores=return_scores
    )
    desired_jax = jax_base.TopK()(
        scores_jax, k, row_lengths_jax, return_scores=return_scores
    )

    if return_scores:
      desired_idx_jax, desired_scores_bits_jax = desired_jax
      desired_idx = torch.as_tensor(
          np.asarray(desired_idx_jax, dtype=np.int32),
          device=device,
          dtype=torch.int32,
      )
      desired_scores_bits = torch.as_tensor(
          np.asarray(desired_scores_bits_jax, dtype=np.int32),
          device=device,
          dtype=torch.int32,
      )
      torch.testing.assert_close(actual_idx, desired_idx)
      torch.testing.assert_close(actual_scores_bits, desired_scores_bits)
    else:
      self.assertIsNone(actual_scores_bits)
      desired_idx = torch.as_tensor(
          np.asarray(desired_jax, dtype=np.int32),
          device=device,
          dtype=torch.int32,
      )
      torch.testing.assert_close(actual_idx, desired_idx)

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      scores_fake = torch.empty((2, 128), dtype=torch.float32)
      out_idx, out_scores_none = torch_base.TopK(
          scores_fake, 16, return_scores=False
      )
      out_idx_2, out_scores = torch_base.TopK(
          scores_fake, 16, return_scores=True
      )
    self.assertEqual(tuple(out_idx.shape), (2, 16))
    self.assertEqual(out_idx.dtype, torch.int32)
    self.assertIsNone(out_scores_none)
    self.assertEqual(tuple(out_idx_2.shape), (2, 16))
    self.assertEqual(out_idx_2.dtype, torch.int32)
    self.assertIsNotNone(out_scores)
    self.assertEqual(tuple(out_scores.shape), (2, 16))
    self.assertEqual(out_scores.dtype, torch.int32)

  def test_torch_compile(self):
    device = "tpu"
    shape = (2, 128)
    k = 16
    rng = np.random.default_rng(0)
    scores_np = rng.standard_normal(shape, dtype=np.float32)
    scores_torch = torch.as_tensor(
        scores_np, dtype=torch.float32, device=device
    )
    scores_jax = jnp.asarray(scores_np + 1.0, dtype=jnp.float32)

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x):
      x = x + 1.0
      top_idx, _ = torch_base.TopK(x, k)
      return top_idx + 2

    actual_idx = compiled_fn(scores_torch)
    desired_idx_jax = jax_base.TopK()(scores_jax, k)
    expected_idx = (
        torch.as_tensor(
            np.asarray(desired_idx_jax, dtype=np.int32),
            device=device,
            dtype=torch.int32,
        )
        + 2
    )
    torch.testing.assert_close(actual_idx, expected_idx)


if __name__ == "__main__":
  absltest.main()
