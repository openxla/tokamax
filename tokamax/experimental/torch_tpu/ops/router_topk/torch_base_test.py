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
"""Tests for the base class of the RouterTopK PyTorch Op."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.router_topk import base as jax_base
from tokamax.experimental.torch_tpu.ops.router_topk import torch_base
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


class RouterTopKBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")
    torch.manual_seed(0)

  @parameterized.parameters(
      ((8, 512), 10, torch.float32, jnp.float32),
      ((64, 512), 10, torch.float32, jnp.float32),
      ((250, 512), 10, torch.bfloat16, jnp.bfloat16),
      ((40, 256), 8, torch.float16, jnp.float16),
  )
  def test_equivalence_with_jax_op(self, shape, k, torch_dtype, jax_dtype):
    device = "tpu"
    rng = np.random.default_rng(42)
    scores_np = rng.random(shape, dtype=np.float32)

    scores_torch = torch.as_tensor(
        scores_np, dtype=torch.float32, device=device
    ).to(torch_dtype)
    scores_jax = jnp.asarray(
        scores_torch.detach().cpu().to(torch.float32).numpy(), dtype=jax_dtype
    )

    actual_weights, actual_indices = torch_base.RouterTopK(scores_torch, k)
    desired_weights_jax, desired_indices_jax = jax_base.RouterTopK()(
        scores_jax, k
    )

    desired_weights = torch.as_tensor(
        np.asarray(desired_weights_jax, dtype=np.float32),
        device=device,
        dtype=torch.float32,
    )
    desired_indices = torch.as_tensor(
        np.asarray(desired_indices_jax, dtype=np.int32),
        device=device,
        dtype=torch.int32,
    )

    self.assertEqual(actual_weights.shape, desired_weights.shape)
    self.assertEqual(actual_indices.shape, desired_indices.shape)
    self.assertEqual(actual_weights.dtype, torch.float32)
    self.assertEqual(actual_indices.dtype, torch.int32)
    torch.testing.assert_close(actual_weights, desired_weights)
    torch.testing.assert_close(actual_indices, desired_indices)

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      scores_fake = torch.empty((16, 512), dtype=torch.bfloat16)
      out_weights, out_indices = torch_base.RouterTopK(scores_fake, 10)
    self.assertEqual(tuple(out_weights.shape), (16, 10))
    self.assertEqual(out_weights.dtype, torch.float32)
    self.assertEqual(tuple(out_indices.shape), (16, 10))
    self.assertEqual(out_indices.dtype, torch.int32)

  def test_torch_compile(self):
    device = "tpu"
    shape = (16, 512)
    k = 10
    rng = np.random.default_rng(0)
    scores_np = rng.random(shape, dtype=np.float32)
    scores_torch = torch.as_tensor(
        scores_np, dtype=torch.float32, device=device
    )
    scores_jax = jnp.asarray(scores_np + 1.0, dtype=jnp.float32)

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x):
      x = x + 1.0
      weights, indices = torch_base.RouterTopK(x, k)
      return weights + 2.0, indices + 2

    actual_weights, actual_indices = compiled_fn(scores_torch)
    desired_weights_jax, desired_indices_jax = jax_base.RouterTopK()(
        scores_jax, k
    )
    expected_weights = (
        torch.as_tensor(
            np.asarray(desired_weights_jax, dtype=np.float32),
            device=device,
            dtype=torch.float32,
        )
        + 2.0
    )
    expected_indices = (
        torch.as_tensor(
            np.asarray(desired_indices_jax, dtype=np.int32),
            device=device,
            dtype=torch.int32,
        )
        + 2
    )
    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(actual_indices, expected_indices)


if __name__ == "__main__":
  absltest.main()
