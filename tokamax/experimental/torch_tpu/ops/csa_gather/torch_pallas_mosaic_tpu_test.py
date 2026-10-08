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
"""Tests for Pallas Mosaic TPU implementation of CsaGather."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.csa_gather import base as jax_base
from tokamax._src.ops.experimental.tpu.csa_gather import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.csa_gather import torch_base
from tokamax.experimental.torch_tpu.ops.csa_gather import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


def _make_test_data(
    num_pages: int = 8,
    page_size: int = 128,
    top_k: int = 256,
    num_queries: int = 4,
    seed: int = 42,
):
  rng = np.random.default_rng(seed)
  n = num_queries * top_k
  nope_np = rng.integers(
      0, 2**16, size=(num_pages, page_size, 128), dtype=np.int32
  )
  rope_np = rng.integers(
      0, 2**16, size=(num_pages, page_size // 4, 128), dtype=np.int32
  )
  indices_np = rng.integers(0, num_pages * page_size, size=(n,), dtype=np.int32)
  return nope_np, rope_np, indices_np


class PallasMosaicTpuCsaGatherTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    device = jax.devices()[0]
    if not jax_pallas_mosaic_tpu.PallasTpuCsaGather().supported_on(device):
      self.skipTest(f"Op not supported on {device.device_kind}.")
    torch.manual_seed(0)

  @parameterized.parameters(
      (8, 128, 256, 4, None),
      (16, 64, 512, 2, None),
      (8, 128, 128, 8, None),
      (8, 128, 256, 4, 512),
  )
  def test_pallas_mosaic_tpu_matches_jax_op(
      self, num_pages, page_size, top_k, num_queries, num_valid
  ):
    device = "tpu"
    nope_np, rope_np, indices_np = _make_test_data(
        num_pages=num_pages,
        page_size=page_size,
        top_k=top_k,
        num_queries=num_queries,
    )
    nope_torch = torch.as_tensor(nope_np, dtype=torch.int32, device=device)
    rope_torch = torch.as_tensor(rope_np, dtype=torch.int32, device=device)
    indices_torch = torch.as_tensor(
        indices_np, dtype=torch.int32, device=device
    )

    nope_jax = jnp.asarray(nope_np, dtype=jnp.int32)
    rope_jax = jnp.asarray(rope_np, dtype=jnp.int32)
    indices_jax = jnp.asarray(indices_np, dtype=jnp.int32)

    if num_valid is not None:
      nvi_np = np.array([num_valid], dtype=np.int32)
      nvi_torch = torch.as_tensor(nvi_np, dtype=torch.int32, device=device)
      nvi_jax = jnp.asarray(nvi_np, dtype=jnp.int32)
    else:
      nvi_torch = None
      nvi_jax = None

    actual_nope, actual_rope = (
        torch_pallas_mosaic_tpu.PallasMosaicTpuCsaGather(
            nope_torch, rope_torch, indices_torch, nvi_torch, top_k=top_k
        )
    )
    desired_pallas_nope_jax, desired_pallas_rope_jax = (
        jax_pallas_mosaic_tpu.PallasTpuCsaGather()(
            nope_jax, rope_jax, indices_jax, nvi_jax, top_k=top_k
        )
    )
    desired_base_nope_jax, desired_base_rope_jax = jax_base.CsaGather()(
        nope_jax, rope_jax, indices_jax, nvi_jax, top_k=top_k
    )

    desired_pallas_nope = torch.as_tensor(
        np.asarray(desired_pallas_nope_jax, dtype=np.int32),
        device=device,
        dtype=torch.int32,
    )
    desired_pallas_rope = torch.as_tensor(
        np.asarray(desired_pallas_rope_jax, dtype=np.int32),
        device=device,
        dtype=torch.int32,
    )
    desired_base_nope = torch.as_tensor(
        np.asarray(desired_base_nope_jax, dtype=np.int32),
        device=device,
        dtype=torch.int32,
    )
    desired_base_rope = torch.as_tensor(
        np.asarray(desired_base_rope_jax, dtype=np.int32),
        device=device,
        dtype=torch.int32,
    )

    self.assertEqual(actual_nope.shape, desired_pallas_nope.shape)
    self.assertEqual(actual_rope.shape, desired_pallas_rope.shape)
    if num_valid is not None:
      torch.testing.assert_close(
          actual_nope[:num_valid], desired_pallas_nope[:num_valid]
      )
      torch.testing.assert_close(
          actual_rope[: num_valid // 4],
          desired_pallas_rope[: num_valid // 4],
      )
      torch.testing.assert_close(
          actual_nope[:num_valid], desired_base_nope[:num_valid]
      )
      torch.testing.assert_close(
          actual_rope[: num_valid // 4], desired_base_rope[: num_valid // 4]
      )
    else:
      torch.testing.assert_close(actual_nope, desired_pallas_nope)
      torch.testing.assert_close(actual_rope, desired_pallas_rope)
      torch.testing.assert_close(actual_nope, desired_base_nope)
      torch.testing.assert_close(actual_rope, desired_base_rope)

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      nope_fake = torch.empty((8, 128, 128), dtype=torch.int32)
      rope_fake = torch.empty((8, 32, 128), dtype=torch.int32)
      indices_fake = torch.empty((1024,), dtype=torch.int32)
      out_nope, out_rope = torch_pallas_mosaic_tpu.PallasMosaicTpuCsaGather(
          nope_fake, rope_fake, indices_fake, top_k=256
      )
    self.assertEqual(tuple(out_nope.shape), (1024, 128))
    self.assertEqual(out_nope.dtype, torch.int32)
    self.assertEqual(tuple(out_rope.shape), (256, 128))
    self.assertEqual(out_rope.dtype, torch.int32)

  def test_torch_compile(self):
    device = "tpu"
    top_k = 256
    nope_np, rope_np, indices_np = _make_test_data(
        num_pages=8, page_size=128, top_k=top_k, num_queries=4
    )
    nope_torch = torch.as_tensor(nope_np, dtype=torch.int32, device=device)
    rope_torch = torch.as_tensor(rope_np, dtype=torch.int32, device=device)
    indices_torch = torch.as_tensor(
        indices_np, dtype=torch.int32, device=device
    )

    configs = torch_utils.get_configs(
        torch_pallas_mosaic_tpu.PallasMosaicTpuCsaGather,
        nope_torch,
        rope_torch,
        indices_torch,
        top_k=top_k,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(configs[0])

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(nope, rope, indices):
      nope = nope + 1
      rope = rope + 1
      nope_out, rope_out = torch_pallas_mosaic_tpu.PallasMosaicTpuCsaGather(
          nope, rope, indices, top_k=top_k, configs=configs
      )
      return nope_out + 2, rope_out + 2

    actual_nope, actual_rope = compiled_fn(
        nope_torch, rope_torch, indices_torch
    )
    expected_nope, expected_rope = torch_base.CsaGather(
        nope_torch + 1, rope_torch + 1, indices_torch, top_k=top_k
    )
    expected_nope = expected_nope + 2
    expected_rope = expected_rope + 2

    torch.testing.assert_close(actual_nope, expected_nope)
    torch.testing.assert_close(actual_rope, expected_rope)

  def test_torch_compile_call_without_configs_uses_heuristics_config(self):
    device = "tpu"
    top_k = 256
    nope_np, rope_np, indices_np = _make_test_data(
        num_pages=8, page_size=128, top_k=top_k, num_queries=4
    )
    nope_torch = torch.as_tensor(nope_np, dtype=torch.int32, device=device)
    rope_torch = torch.as_tensor(rope_np, dtype=torch.int32, device=device)
    indices_torch = torch.as_tensor(
        indices_np, dtype=torch.int32, device=device
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(nope, rope, indices):
      nope = nope + 1
      rope = rope + 1
      nope_out, rope_out = torch_pallas_mosaic_tpu.PallasMosaicTpuCsaGather(
          nope, rope, indices, top_k=top_k
      )
      return nope_out + 2, rope_out + 2

    actual_nope, actual_rope = compiled_fn(
        nope_torch, rope_torch, indices_torch
    )
    expected_nope, expected_rope = torch_base.CsaGather(
        nope_torch + 1, rope_torch + 1, indices_torch, top_k=top_k
    )
    expected_nope = expected_nope + 2
    expected_rope = expected_rope + 2

    torch.testing.assert_close(actual_nope, expected_nope)
    torch.testing.assert_close(actual_rope, expected_rope)


if __name__ == "__main__":
  absltest.main()
