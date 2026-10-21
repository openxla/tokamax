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
"""Tests for the base class of the Ragged Gather PyTorch Op."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_gather import base as jax_base
from tokamax.experimental.torch_tpu.ops.ragged_gather import torch_base
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


class BaseRaggedGatherTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")
    torch.manual_seed(0)

  @parameterized.product(
      in_out_size=[(512, 32), (512, 400)],
      start_end=[(3, 28), (3, 338)],
      hidden_size=[128, 512],
      dtype=[torch.bfloat16, torch.float32],
  )
  def test_base_matches_reference(
      self, in_out_size, hidden_size, start_end, dtype
  ):
    device = "tpu"
    in_size, out_size = in_out_size
    start, end = start_end
    start = min(start, out_size)
    end = min(end, out_size)

    x_torch = torch.randn(
        (in_size, hidden_size),
        dtype=dtype,
        device=device,
    )
    indices_torch = torch.randint(
        0,
        in_size,
        (out_size,),
        dtype=torch.int32,
        device=device,
    )
    start_torch = torch.tensor([start], dtype=torch.int32, device=device)
    end_torch = torch.tensor([end], dtype=torch.int32, device=device)

    jax_dtype = jnp.bfloat16 if dtype == torch.bfloat16 else jnp.float32
    x_jax = jnp.array(
        x_torch.detach().cpu().to(torch.float32).numpy(), dtype=jax_dtype
    )
    indices_jax = jnp.array(
        indices_torch.detach().cpu().numpy(), dtype=jnp.int32
    )
    start_jax = jnp.array(start_torch.detach().cpu().numpy(), dtype=jnp.int32)
    end_jax = jnp.array(end_torch.detach().cpu().numpy(), dtype=jnp.int32)

    actual_torch = torch_base.RaggedGather(
        x_torch, indices_torch, start_torch, end_torch
    )
    desired_jax = jax_base.RaggedGather()(
        x_jax, indices_jax, start_jax, end_jax
    )
    desired_torch = torch.as_tensor(
        np.asarray(desired_jax, dtype=np.float32),
        device=device,
        dtype=dtype,
    )

    self.assertEqual(actual_torch.shape, desired_jax.shape)
    torch.testing.assert_close(
        actual_torch, desired_torch, rtol=1e-5, atol=1e-5
    )

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      x_fake = torch.empty((512, 128), dtype=torch.float32)
      indices_fake = torch.empty((32,), dtype=torch.int32)
      start_fake = torch.empty((1,), dtype=torch.int32)
      end_fake = torch.empty((1,), dtype=torch.int32)
      out = torch_base.RaggedGather(x_fake, indices_fake, start_fake, end_fake)
    self.assertEqual(tuple(out.shape), (32, 128))
    self.assertEqual(out.dtype, torch.float32)

  def test_torch_compile(self):
    device = "tpu"
    in_size, out_size, hidden_size = 512, 400, 128
    start, end = 3, 338

    x_torch = torch.randn(
        (in_size, hidden_size), dtype=torch.float32, device=device
    )
    indices_torch = torch.randint(
        0, in_size, (out_size,), dtype=torch.int32, device=device
    )
    start_torch = torch.tensor([start], dtype=torch.int32, device=device)
    end_torch = torch.tensor([end], dtype=torch.int32, device=device)

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, indices, start, end):
      return torch_base.RaggedGather(x, indices, start, end)

    actual = compiled_fn(x_torch, indices_torch, start_torch, end_torch)
    expected = torch_base.RaggedGather(
        x_torch, indices_torch, start_torch, end_torch
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
