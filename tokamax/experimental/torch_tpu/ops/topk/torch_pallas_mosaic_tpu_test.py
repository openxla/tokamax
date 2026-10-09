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
"""Tests for Pallas Mosaic TPU implementation of TopK."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.topk import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.topk import test_base
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.topk import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


class PallasMosaicTpuTopKTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if not pltpu.get_tpu_info().generation >= 7:
      self.skipTest(
          "SparseCore TopK Pallas TPU kernel requires TPU v7 or newer."
      )
    torch.manual_seed(0)

  @parameterized.parameters(
      ((32, 2048), 16, False, False),
      ((32, 2048), 64, True, False),
      ((4, 8192), 64, False, True),
      ((4, 8192), 64, True, True),
  )
  def test_pallas_mosaic_tpu_matches_jax_op(
      self, shape, k, with_row_lengths, return_scores
  ):
    device = "tpu"
    b, n = shape
    rng = np.random.default_rng(42 + b + n + k)
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
      row_lengths_np = None
      row_lengths_torch = None
      row_lengths_jax = None

    actual_idx, actual_scores_bits = (
        torch_pallas_mosaic_tpu.PallasMosaicTpuTopK(
            scores_torch, k, row_lengths_torch, return_scores=return_scores
        )
    )
    desired_pallas_jax = jax_pallas_mosaic_tpu.PallasTpuTopK()(
        scores_jax, k, row_lengths_jax, return_scores=return_scores
    )

    if return_scores:
      self.assertIsNotNone(actual_scores_bits)
      desired_idx_jax, desired_scores_bits_jax = desired_pallas_jax
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
      self.assertEqual(actual_idx.shape, desired_idx.shape)
      self.assertEqual(actual_scores_bits.shape, desired_scores_bits.shape)
      torch.testing.assert_close(actual_idx, desired_idx)
      torch.testing.assert_close(actual_scores_bits, desired_scores_bits)
      test_base.assert_topk_matches_reference(
          scores_np,
          actual_idx.cpu().numpy(),
          k,
          row_lengths=row_lengths_np,
          actual_scores_bits=actual_scores_bits.cpu().numpy(),
      )
    else:
      self.assertIsNone(actual_scores_bits)
      desired_idx = torch.as_tensor(
          np.asarray(desired_pallas_jax, dtype=np.int32),
          device=device,
          dtype=torch.int32,
      )
      self.assertEqual(actual_idx.shape, desired_idx.shape)
      torch.testing.assert_close(actual_idx, desired_idx)
      test_base.assert_topk_matches_reference(
          scores_np,
          actual_idx.cpu().numpy(),
          k,
          row_lengths=row_lengths_np,
      )

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      scores_fake = torch.empty((32, 2048), dtype=torch.float32)
      out_idx, out_scores_none = torch_pallas_mosaic_tpu.PallasMosaicTpuTopK(
          scores_fake, 16, return_scores=False
      )
      out_idx_2, out_scores = torch_pallas_mosaic_tpu.PallasMosaicTpuTopK(
          scores_fake, 16, return_scores=True
      )
    self.assertEqual(tuple(out_idx.shape), (32, 16))
    self.assertEqual(out_idx.dtype, torch.int32)
    self.assertIsNone(out_scores_none)
    self.assertEqual(tuple(out_idx_2.shape), (32, 16))
    self.assertEqual(out_idx_2.dtype, torch.int32)
    self.assertIsNotNone(out_scores)
    self.assertEqual(tuple(out_scores.shape), (32, 16))
    self.assertEqual(out_scores.dtype, torch.int32)

  def test_torch_compile(self):
    device = "tpu"
    shape = (32, 2048)
    k = 16
    rng = np.random.default_rng(0)
    scores_np = rng.standard_normal(shape, dtype=np.float32)
    scores_torch = torch.as_tensor(
        scores_np, dtype=torch.float32, device=device
    )

    configs = torch_utils.get_configs(
        torch_pallas_mosaic_tpu.PallasMosaicTpuTopK,
        scores_torch,
        k,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(configs[0])

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x):
      x = x + 1.0
      top_idx, _ = torch_pallas_mosaic_tpu.PallasMosaicTpuTopK(
          x, k, configs=configs
      )
      return top_idx + 2

    actual_idx = compiled_fn(scores_torch)
    expected_idx, _ = torch_pallas_mosaic_tpu.PallasMosaicTpuTopK(
        scores_torch + 1.0, k, configs=configs
    )
    expected_idx = expected_idx + 2
    torch.testing.assert_close(actual_idx, expected_idx)

  def test_torch_compile_call_without_configs_uses_heuristics_config(self):
    device = "tpu"
    shape = (32, 2048)
    k = 16
    rng = np.random.default_rng(0)
    scores_np = rng.standard_normal(shape, dtype=np.float32)
    scores_torch = torch.as_tensor(
        scores_np, dtype=torch.float32, device=device
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x):
      x = x + 1.0
      top_idx, _ = torch_pallas_mosaic_tpu.PallasMosaicTpuTopK(x, k)
      return top_idx + 2

    actual_idx = compiled_fn(scores_torch)
    expected_idx, _ = torch_pallas_mosaic_tpu.PallasMosaicTpuTopK(
        scores_torch + 1.0, k
    )
    expected_idx = expected_idx + 2
    torch.testing.assert_close(actual_idx, expected_idx)


if __name__ == "__main__":
  absltest.main()
