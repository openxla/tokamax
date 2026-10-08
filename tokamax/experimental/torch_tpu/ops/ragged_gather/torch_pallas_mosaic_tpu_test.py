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
"""Tests for Pallas Mosaic TPU implementation of Ragged Gather."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_gather import base as jax_base
from tokamax._src.ops.ragged_gather import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.ragged_gather import torch_base
from tokamax.experimental.torch_tpu.ops.ragged_gather import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


class PallasMosaicTpuRaggedGatherTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")
    torch.manual_seed(0)

  @parameterized.named_parameters(
      dict(
          testcase_name="small_float32",
          in_out_size=(512, 32),
          start_end=(3, 28),
          hidden_size=128,
          dtype=torch.float32,
      ),
      dict(
          testcase_name="medium_float32",
          in_out_size=(512, 400),
          start_end=(3, 338),
          hidden_size=512,
          dtype=torch.float32,
      ),
      dict(
          testcase_name="small_bfloat16",
          in_out_size=(512, 32),
          start_end=(3, 28),
          hidden_size=128,
          dtype=torch.bfloat16,
      ),
  )
  def test_pallas_matches_jax_op(
      self, in_out_size, hidden_size, start_end, dtype
  ):
    device = "tpu"
    in_size, out_size = in_out_size
    start, end = start_end
    start = min(start, out_size)
    end = min(end, out_size)

    x_torch = torch.randn((in_size, hidden_size), dtype=dtype, device=device)
    jax_dtype = jnp.bfloat16 if dtype == torch.bfloat16 else jnp.float32

    indices_torch = torch.randint(
        0,
        in_size,
        (out_size,),
        dtype=torch.int32,
        device=device,
    )
    start_torch = torch.tensor([start], dtype=torch.int32, device=device)
    end_torch = torch.tensor([end], dtype=torch.int32, device=device)

    x_jax = jnp.array(
        x_torch.detach().cpu().to(torch.float32).numpy(), dtype=jax_dtype
    )
    indices_jax = jnp.array(
        indices_torch.detach().cpu().numpy(), dtype=jnp.int32
    )
    start_jax = jnp.array(start_torch.detach().cpu().numpy(), dtype=jnp.int32)
    end_jax = jnp.array(end_torch.detach().cpu().numpy(), dtype=jnp.int32)

    actual_torch = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedGather(
        x_torch, indices_torch, start_torch, end_torch
    )
    desired_pallas_jax = jax_pallas_mosaic_tpu.PallasTpuRaggedGather()(
        x_jax, indices_jax, start_jax, end_jax
    )
    desired_base_jax = jax_base.RaggedGather()(
        x_jax, indices_jax, start_jax, end_jax
    )

    desired_pallas_torch = torch.as_tensor(
        np.asarray(desired_pallas_jax, dtype=np.float32),
        device=device,
        dtype=dtype,
    )
    desired_base_torch = torch.as_tensor(
        np.asarray(desired_base_jax, dtype=np.float32),
        device=device,
        dtype=dtype,
    )

    self.assertEqual(actual_torch.shape, desired_pallas_jax.shape)
    torch.testing.assert_close(
        actual_torch[start:end],
        desired_pallas_torch[start:end],
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        actual_torch[start:end],
        desired_base_torch[start:end],
        rtol=1e-2,
        atol=1e-2,
    )

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      x_fake = torch.empty((512, 128), dtype=torch.float32)
      indices_fake = torch.empty((32,), dtype=torch.int32)
      start_fake = torch.empty((1,), dtype=torch.int32)
      end_fake = torch.empty((1,), dtype=torch.int32)
      out = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedGather(
          x_fake, indices_fake, start_fake, end_fake
      )
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

    configs = torch_utils.get_configs(
        torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedGather,
        x_torch,
        indices_torch,
        start_torch,
        end_torch,
        from_autotuning_cache=False,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, indices, start, end):
      x = x + 1.0
      out = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedGather(
          x, indices, start, end, configs=configs
      )
      return out + 2.0

    actual = compiled_fn(x_torch, indices_torch, start_torch, end_torch)
    expected = (
        torch_base.RaggedGather(
            x_torch + 1.0, indices_torch, start_torch, end_torch
        )
        + 2.0
    )
    torch.testing.assert_close(
        actual[start:end], expected[start:end], rtol=1e-2, atol=1e-2
    )


if __name__ == "__main__":
  absltest.main()
