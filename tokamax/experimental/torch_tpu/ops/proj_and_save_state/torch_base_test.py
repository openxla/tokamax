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
"""Tests for Torch ProjAndSaveState base op."""

import math
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from tokamax._src.ops.proj_and_save_state import base as jax_base
from tokamax._src.ops.proj_and_save_state import test_base as jax_test_base
from tokamax.experimental.torch_tpu.ops import torch_test_utils
from tokamax.experimental.torch_tpu.ops.proj_and_save_state import torch_base
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


def _make_torch_and_jax_inputs(
    geometry: jax_test_base.Geometry,
    num_tokens: int,
    *,
    hidden_size: int = 4096,
    num_pages: int | None = None,
    num_padding: int = 0,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    device: str = "tpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  """Creates PyTorch inputs on `device` and converts them to JAX arrays."""
  torch.manual_seed(seed)
  rng = np.random.default_rng(seed)

  if num_pages is None:
    num_pages = -(-num_tokens // geometry.tokens_per_page) + 1
  state_dim = 2 * geometry.state_width

  hidden_states_torch = torch.randn(
      (num_tokens, hidden_size), dtype=dtype, device=device
  )
  wkv_wgate_torch = (
      torch.randn((hidden_size, state_dim), dtype=torch.float32, device=device)
      / math.sqrt(hidden_size)
  ).to(dtype)
  ape_torch = torch.randn(
      (geometry.compress_ratio, geometry.state_width),
      dtype=dtype,
      device=device,
  )
  positions_torch = torch.randint(
      0, 1 << 16, (num_tokens,), dtype=torch.int32, device=device
  )

  token_slots = rng.permutation(num_pages * geometry.tokens_per_page)[
      :num_tokens
  ]
  slot_mapping_np = (token_slots * geometry.rows_per_token).astype(np.int32)
  if num_padding:
    slot_mapping_np[num_tokens - num_padding :] = -1
  slot_mapping_torch = torch.as_tensor(
      slot_mapping_np, dtype=torch.int32, device=device
  )

  cache_shape = geometry.cache_shape(num_pages)
  if geometry.words:
    cache_np = rng.integers(
        -(1 << 31), 1 << 31, size=cache_shape, dtype=np.int32
    )
    cache_torch = torch.as_tensor(cache_np, dtype=torch.int32, device=device)
  else:
    cache_np = rng.integers(0, 256, size=cache_shape, dtype=np.uint8)
    cache_torch = torch.as_tensor(cache_np, dtype=torch.uint8, device=device)

  torch_inputs = dict(
      hidden_states=hidden_states_torch,
      wkv_wgate=wkv_wgate_torch,
      ape=ape_torch,
      positions=positions_torch,
      slot_mapping=slot_mapping_torch,
      cache=cache_torch,
  )
  jax_inputs = {
      name: torch_test_utils.torch_to_jax(tensor)
      for name, tensor in torch_inputs.items()
  }
  return torch_inputs, jax_inputs


class ProjAndSaveStateBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    torch.manual_seed(0)

  @parameterized.named_parameters(
      dict(
          testcase_name="csa_decode_t8",
          geometry_name="csa",
          num_tokens=8,
          num_padding=0,
          dtype=torch.float32,
      ),
      dict(
          testcase_name="csa_unaligned_t13_padded",
          geometry_name="csa",
          num_tokens=13,
          num_padding=3,
          dtype=torch.float32,
      ),
      dict(
          testcase_name="csa_u8_t32",
          geometry_name="csa_u8",
          num_tokens=32,
          num_padding=0,
          dtype=torch.float32,
      ),
      dict(
          testcase_name="hca_t32",
          geometry_name="hca",
          num_tokens=32,
          num_padding=4,
          dtype=torch.float32,
      ),
      dict(
          testcase_name="indexer_t32",
          geometry_name="indexer",
          num_tokens=32,
          num_padding=0,
          dtype=torch.float32,
      ),
      dict(
          testcase_name="csa_bf16_t40",
          geometry_name="csa",
          num_tokens=40,
          num_padding=0,
          dtype=torch.bfloat16,
      ),
  )
  def test_base_matches_jax_op(
      self,
      geometry_name: str,
      num_tokens: int,
      num_padding: int,
      dtype: torch.dtype,
  ):
    device = "tpu"
    geometry = jax_test_base.GEOMETRIES[geometry_name]
    torch_inputs, jax_inputs = _make_torch_and_jax_inputs(
        geometry,
        num_tokens,
        hidden_size=512,
        num_padding=num_padding,
        dtype=dtype,
        seed=42,
        device=device,
    )

    actual = torch_base.ProjAndSaveState(
        **torch_inputs, compress_ratio=geometry.compress_ratio
    )
    desired_jax = jax_base.ProjAndSaveState()(
        **jax_inputs, compress_ratio=geometry.compress_ratio
    )
    desired = torch_test_utils.jax_to_torch(desired_jax, device=device)

    self.assertEqual(actual.shape, desired.shape)
    self.assertEqual(actual.dtype, desired.dtype)
    torch.testing.assert_close(actual, desired)
    jax_test_base.assert_caches_match(
        torch_test_utils.torch_to_jax(actual),
        desired_jax,
        jax_inputs["slot_mapping"],
        geometry.rows_per_token,
    )

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      hidden_states_fake = torch.empty((16, 512), dtype=torch.bfloat16)
      wkv_wgate_fake = torch.empty((512, 2048), dtype=torch.bfloat16)
      ape_fake = torch.empty((4, 1024), dtype=torch.bfloat16)
      positions_fake = torch.empty((16,), dtype=torch.int32)
      slot_mapping_fake = torch.empty((16,), dtype=torch.int32)
      cache_fake = torch.empty((2, 256, 128), dtype=torch.int32)

      out = torch_base.ProjAndSaveState(
          hidden_states_fake,
          wkv_wgate_fake,
          ape_fake,
          positions_fake,
          slot_mapping_fake,
          cache_fake,
          compress_ratio=4,
      )
    self.assertEqual(tuple(out.shape), (2, 256, 128))
    self.assertEqual(out.dtype, torch.int32)

  def test_fake_impl_symbolic_shapes(self):
    class _Module(torch.nn.Module):

      def forward(
          self,
          hidden_states: torch.Tensor,
          wkv_wgate: torch.Tensor,
          ape: torch.Tensor,
          positions: torch.Tensor,
          slot_mapping: torch.Tensor,
          cache: torch.Tensor,
      ) -> torch.Tensor:
        return torch_base.ProjAndSaveState(
            hidden_states,
            wkv_wgate,
            ape,
            positions,
            slot_mapping,
            cache,
            compress_ratio=4,
        )

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    num_pages = torch.export.Dim("num_pages", min=1, max=64)
    hidden_states = torch.ones((16, 512), dtype=torch.bfloat16)
    wkv_wgate = torch.ones((512, 2048), dtype=torch.bfloat16)
    ape = torch.ones((4, 1024), dtype=torch.bfloat16)
    positions = torch.zeros((16,), dtype=torch.int32)
    slot_mapping = torch.zeros((16,), dtype=torch.int32)
    cache = torch.zeros((4, 256, 128), dtype=torch.int32)

    exported = torch.export.export(
        _Module(),
        (hidden_states, wkv_wgate, ape, positions, slot_mapping, cache),
        dynamic_shapes=(
            {0: num_tokens},
            None,
            None,
            {0: num_tokens},
            {0: num_tokens},
            {0: num_pages},
        ),
    )
    op_nodes = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function"
        and torch_base.ProjAndSaveState.jax_op_name in str(node.target)
    ]
    self.assertLen(op_nodes, 1)
    meta = op_nodes[0].meta["val"]
    self.assertIsInstance(meta.shape[0], torch.SymInt)
    self.assertEqual(meta.shape[1], 256)
    self.assertEqual(meta.shape[2], 128)

  def test_torch_compile(self):
    device = "tpu"
    geometry = jax_test_base.CSA
    torch_inputs, _ = _make_torch_and_jax_inputs(
        geometry, 16, hidden_size=512, seed=42, device=device
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        hidden_states, wkv_wgate, ape, positions, slot_mapping, cache
    ):
      hidden_states = hidden_states + 0.1
      out = torch_base.ProjAndSaveState(
          hidden_states,
          wkv_wgate,
          ape,
          positions,
          slot_mapping,
          cache,
          compress_ratio=geometry.compress_ratio,
      )
      return out

    actual = compiled_fn(**torch_inputs)
    expected = torch_base.ProjAndSaveState(
        **{
            **torch_inputs,
            "hidden_states": torch_inputs["hidden_states"] + 0.1,
        },
        compress_ratio=geometry.compress_ratio,
    )
    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
  absltest.main()
