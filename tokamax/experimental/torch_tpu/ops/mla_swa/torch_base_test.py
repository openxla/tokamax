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
"""Tests for Torch MLA-SWA base op."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.mla_swa import base as jax_base
from tokamax._src.ops.experimental.mla_swa import test_base as jax_test_base
from tokamax.experimental.torch_tpu.ops import torch_test_utils
from tokamax.experimental.torch_tpu.ops.mla_swa import torch_base
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import

HEAD_DIM = jax_test_base.HEAD_DIM
PAGE_SIZE = jax_test_base.PAGE_SIZE
SLIDING_WINDOW = jax_test_base.SLIDING_WINDOW

_INPUT_NAMES = (
    "q",
    "new_kv",
    "cache_kv",
    "kv_lens",
    "page_indices",
    "cu_q_lens",
    "distribution",
    "attention_sinks",
)


def _make_torch_and_jax_inputs(
    *,
    batch_size: int = 4,
    num_heads: int = 128,
    page_size: int = PAGE_SIZE,
    sliding_window: int = SLIDING_WINDOW,
    pad_tokens: bool = False,
    int32_cache: bool = False,
    non_causal_block: bool = False,
    seed: int = 0,
    device: str = "tpu",
) -> tuple[dict[str, torch.Tensor], dict[str, jax.Array], int]:
  """Creates PyTorch inputs on `device` and converts them to JAX arrays."""
  raw_jax_inputs, actual_tokens = jax_test_base.make_inputs(
      batch_size=batch_size,
      num_heads=num_heads,
      page_size=page_size,
      sliding_window=sliding_window,
      pad_tokens=pad_tokens,
      int32_cache=int32_cache,
      non_causal_block=non_causal_block,
      seed=seed,
  )
  torch_inputs = {
      name: torch_test_utils.jax_to_torch(arr, device=device)
      for name, arr in zip(_INPUT_NAMES, raw_jax_inputs, strict=True)
  }
  jax_inputs = {
      name: torch_test_utils.torch_to_jax(tensor)
      for name, tensor in torch_inputs.items()
  }
  return torch_inputs, jax_inputs, actual_tokens


def _clone_torch_inputs(
    torch_inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  """Returns `torch_inputs` with a fresh copy of `cache_kv`."""
  return {**torch_inputs, "cache_kv": torch_inputs["cache_kv"].clone()}


def _clone_jax_inputs(jax_inputs: dict[str, jax.Array]) -> dict[str, jax.Array]:
  """Returns `jax_inputs` with a fresh copy of `cache_kv`."""
  return {**jax_inputs, "cache_kv": jnp.copy(jax_inputs["cache_kv"])}


class MlaSwaBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    torch.manual_seed(0)

  @parameterized.named_parameters(
      dict(
          testcase_name="unpadded_u8",
          pad_tokens=False,
          int32_cache=False,
          unnormalized_output=False,
          non_causal_block=False,
      ),
      dict(
          testcase_name="padded_u8",
          pad_tokens=True,
          int32_cache=False,
          unnormalized_output=False,
          non_causal_block=False,
      ),
      dict(
          testcase_name="unpadded_int32",
          pad_tokens=False,
          int32_cache=True,
          unnormalized_output=False,
          non_causal_block=False,
      ),
      dict(
          testcase_name="unnormalized_output",
          pad_tokens=False,
          int32_cache=False,
          unnormalized_output=True,
          non_causal_block=False,
      ),
      dict(
          testcase_name="non_causal_block",
          pad_tokens=False,
          int32_cache=False,
          unnormalized_output=False,
          non_causal_block=True,
      ),
  )
  def test_base_matches_jax_op(
      self,
      pad_tokens: bool,
      int32_cache: bool,
      unnormalized_output: bool,
      non_causal_block: bool,
  ):
    device = "tpu"
    torch_inputs, jax_inputs, actual_tokens = _make_torch_and_jax_inputs(
        batch_size=4,
        pad_tokens=pad_tokens,
        int32_cache=int32_cache,
        non_causal_block=non_causal_block,
        seed=0,
        device=device,
    )

    actual_out, actual_cache, actual_l, actual_m = torch_base.MlaSwa(
        **_clone_torch_inputs(torch_inputs),
        sliding_window=SLIDING_WINDOW,
        logical_page_size=PAGE_SIZE,
        sm_scale=1.0,
        unnormalized_output=unnormalized_output,
        non_causal_block=non_causal_block,
    )
    (
        desired_out_jax,
        desired_cache_jax,
        desired_l_jax,
        desired_m_jax,
    ) = jax_base.MlaSwa()(
        **_clone_jax_inputs(jax_inputs),
        sliding_window=SLIDING_WINDOW,
        logical_page_size=PAGE_SIZE,
        sm_scale=1.0,
        unnormalized_output=unnormalized_output,
        non_causal_block=non_causal_block,
    )

    desired_out = torch_test_utils.jax_to_torch(desired_out_jax, device=device)
    desired_cache = torch_test_utils.jax_to_torch(
        desired_cache_jax, device=device
    )
    desired_l = torch_test_utils.jax_to_torch(desired_l_jax, device=device)
    desired_m = torch_test_utils.jax_to_torch(desired_m_jax, device=device)

    self.assertEqual(actual_out.shape, desired_out.shape)
    self.assertEqual(actual_out.dtype, desired_out.dtype)
    self.assertEqual(actual_cache.shape, desired_cache.shape)
    self.assertEqual(actual_cache.dtype, desired_cache.dtype)
    self.assertEqual(actual_l.shape, desired_l.shape)
    self.assertEqual(actual_l.dtype, desired_l.dtype)
    self.assertEqual(actual_m.shape, desired_m.shape)
    self.assertEqual(actual_m.dtype, desired_m.dtype)

    torch.testing.assert_close(
        actual_out[:actual_tokens],
        desired_out[:actual_tokens],
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        actual_l[:actual_tokens],
        desired_l[:actual_tokens],
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        actual_m[:actual_tokens],
        desired_m[:actual_tokens],
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(actual_cache, desired_cache, rtol=0, atol=0)

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      q_fake = torch.empty((12, 128, HEAD_DIM), dtype=torch.bfloat16)
      new_kv_fake = torch.empty((12, HEAD_DIM), dtype=torch.bfloat16)
      cache_kv_u8_fake = torch.empty((8, 40, 4, 128), dtype=torch.uint8)
      kv_lens_fake = torch.empty((2,), dtype=torch.int32)
      page_indices_fake = torch.empty((8,), dtype=torch.int32)
      cu_q_lens_fake = torch.empty((3,), dtype=torch.int32)
      dist_fake = torch.empty((3,), dtype=torch.int32)
      sinks_fake = torch.empty((128,), dtype=torch.float32)

      out, updated_cache, out_l, out_m = torch_base.MlaSwa(
          q_fake,
          new_kv_fake,
          cache_kv_u8_fake,
          kv_lens_fake,
          page_indices_fake,
          cu_q_lens_fake,
          dist_fake,
          sinks_fake,
          sliding_window=SLIDING_WINDOW,
          logical_page_size=PAGE_SIZE,
      )
      self.assertEqual(tuple(out.shape), (12, 128, HEAD_DIM))
      self.assertEqual(out.dtype, torch.bfloat16)
      self.assertEqual(tuple(updated_cache.shape), (8, 40, 4, 128))
      self.assertEqual(updated_cache.dtype, torch.uint8)
      self.assertEqual(tuple(out_l.shape), (12, 128))
      self.assertEqual(out_l.dtype, torch.float32)
      self.assertEqual(tuple(out_m.shape), (12, 128))
      self.assertEqual(out_m.dtype, torch.float32)

      cache_kv_i32_fake = torch.empty((8, 40, 128), dtype=torch.int32)
      _, updated_cache_i32, _, _ = torch_base.MlaSwa(
          q_fake,
          new_kv_fake,
          cache_kv_i32_fake,
          kv_lens_fake,
          page_indices_fake,
          cu_q_lens_fake,
          dist_fake,
          sinks_fake,
          sliding_window=SLIDING_WINDOW,
          logical_page_size=PAGE_SIZE,
      )
      self.assertEqual(tuple(updated_cache_i32.shape), (8, 40, 128))
      self.assertEqual(updated_cache_i32.dtype, torch.int32)

  def test_fake_impl_symbolic_shapes(self):
    class _Module(torch.nn.Module):

      def forward(
          self,
          q: torch.Tensor,
          new_kv: torch.Tensor,
          cache_kv: torch.Tensor,
          kv_lens: torch.Tensor,
          page_indices: torch.Tensor,
          cu_q_lens: torch.Tensor,
          distribution: torch.Tensor,
          attention_sinks: torch.Tensor,
      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch_base.MlaSwa(
            q,
            new_kv,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sinks,
            sliding_window=SLIDING_WINDOW,
            logical_page_size=PAGE_SIZE,
        )

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    q = torch.ones((12, 128, HEAD_DIM), dtype=torch.bfloat16)
    new_kv = torch.ones((12, HEAD_DIM), dtype=torch.bfloat16)
    cache_kv = torch.zeros((8, 40, 4, 128), dtype=torch.uint8)
    kv_lens = torch.tensor([16, 16], dtype=torch.int32)
    page_indices = torch.arange(8, dtype=torch.int32)
    cu_q_lens = torch.tensor([0, 4, 12], dtype=torch.int32)
    distribution = torch.tensor([0, 0, 2], dtype=torch.int32)
    attention_sinks = torch.zeros((128,), dtype=torch.float32)

    exported = torch.export.export(
        _Module(),
        (
            q,
            new_kv,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sinks,
        ),
        dynamic_shapes=(
            {0: num_tokens},
            {0: num_tokens},
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    op_nodes = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function"
        and torch_base.MlaSwa.jax_op_name in str(node.target)
    ]
    self.assertLen(op_nodes, 1)
    out_meta, cache_meta, l_meta, m_meta = op_nodes[0].meta["val"]
    self.assertIsInstance(out_meta.shape[0], torch.SymInt)
    self.assertEqual(out_meta.shape[1], 128)
    self.assertEqual(out_meta.shape[2], HEAD_DIM)
    self.assertEqual(tuple(cache_meta.shape), (8, 40, 4, 128))
    self.assertIsInstance(l_meta.shape[0], torch.SymInt)
    self.assertEqual(l_meta.shape[1], 128)
    self.assertIsInstance(m_meta.shape[0], torch.SymInt)
    self.assertEqual(m_meta.shape[1], 128)

  def test_torch_compile(self):
    device = "tpu"
    torch_inputs, _, actual_tokens = _make_torch_and_jax_inputs(
        batch_size=4, pad_tokens=False, seed=42, device=device
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        q,
        new_kv,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
    ):
      q = q + 0.1
      out, updated_cache, out_l, out_m = torch_base.MlaSwa(
          q,
          new_kv,
          cache_kv,
          kv_lens,
          page_indices,
          cu_q_lens,
          distribution,
          attention_sinks,
          sliding_window=SLIDING_WINDOW,
          logical_page_size=PAGE_SIZE,
          sm_scale=0.5,
      )
      return out + 1.0, updated_cache, out_l, out_m

    actual_out, actual_cache, actual_l, actual_m = compiled_fn(
        **_clone_torch_inputs(torch_inputs)
    )
    expected_out, expected_cache, expected_l, expected_m = torch_base.MlaSwa(
        **{**_clone_torch_inputs(torch_inputs), "q": torch_inputs["q"] + 0.1},
        sliding_window=SLIDING_WINDOW,
        logical_page_size=PAGE_SIZE,
        sm_scale=0.5,
    )
    expected_out = expected_out + 1.0

    torch.testing.assert_close(
        actual_out[:actual_tokens],
        expected_out[:actual_tokens],
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        actual_l[:actual_tokens],
        expected_l[:actual_tokens],
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        actual_m[:actual_tokens],
        expected_m[:actual_tokens],
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)


if __name__ == "__main__":
  absltest.main()
