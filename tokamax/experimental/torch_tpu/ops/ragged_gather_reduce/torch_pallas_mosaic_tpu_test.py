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
"""Tests for Pallas Mosaic TPU Ragged Gather Reduce."""

from absl.testing import absltest
from absl.testing import parameterized
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_gather_reduce import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.ragged_gather_reduce import torch_base
from tokamax.experimental.torch_tpu.ops.ragged_gather_reduce import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


def _make_torch_inputs(
    num_tokens: int,
    reduce_group_size: int,
    hidden_size: int,
    valid_mode: str = "random",
    num_rows: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "tpu",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Creates random `(x, indices, topk_weights, valid_rows_mask)` on `device`."""
  torch.manual_seed(seed)
  input_size = num_tokens * reduce_group_size
  if num_rows is None:
    num_rows = input_size
  x = torch.randn((num_rows, hidden_size), dtype=dtype).to(device)
  indices = torch.randint(0, num_rows, (input_size,), dtype=torch.int32).to(
      device
  )
  topk_weights = torch.rand((input_size,), dtype=dtype).to(device)
  shape = (num_tokens, reduce_group_size)
  match valid_mode:
    case "all":
      valid = torch.ones(shape, dtype=torch.bool)
    case "none":
      valid = torch.zeros(shape, dtype=torch.bool)
    case "random":
      valid = torch.rand(shape, dtype=torch.float32) < 0.7
    case "two_per_token" | "uneven_pairs":
      ranks = (
          torch.rand(shape, dtype=torch.float32)
          .argsort(dim=1)
          .argsort(dim=1)
      )
      if valid_mode == "two_per_token":
        valid = ranks < 2
      else:
        num_valid = torch.where(torch.arange(num_tokens) % 2 == 0, 3, 1)[
            :, None
        ]
        valid = ranks < num_valid
    case _:
      raise ValueError(f"Unknown valid_mode: {valid_mode}")
  valid_rows_mask = valid.reshape(-1).to(device)
  return x, indices, topk_weights, valid_rows_mask


class PallasMosaicTpuRaggedGatherReduceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    op_jax = jax_pallas_mosaic_tpu.PallasTpuRaggedGatherReduce()
    if not op_jax.supported_on(backend.get_default_device()):
      self.skipTest("PallasTpuRaggedGatherReduce not supported on this device.")

  @parameterized.parameters(
      # Small input (< 60% of VMEM): exercises the XLA reference shortcut.
      (256, 4, 512, "random"),
      # Every block has two adjacent routes per token: the fixed fast path.
      (8192, 2, 2048, "all"),
      (1024, 8, 4096, "all"),
      # DeepSeek's hidden size; not a power-of-two column split.
      (1024, 8, 7168, "random"),
      (2048, 8, 2048, "two_per_token"),
      # 128 routes per block but not two per token: the generic path.
      (4096, 4, 2048, "uneven_pairs"),
      (2048, 8, 2048, "none"),
      # Not a whole number of token blocks: exercises the token padding.
      (2100, 8, 2048, "random"),
      # An odd number of rows of `x`: exercises the row padding.
      (2048, 8, 2048, "random", 16383),
  )
  def test_pallas_matches_reference(
      self,
      num_tokens: int,
      reduce_group_size: int,
      hidden_size: int,
      valid_mode: str,
      num_rows: int | None = None,
  ):
    device = "tpu"
    x, indices, topk_weights, valid_rows_mask = _make_torch_inputs(
        num_tokens,
        reduce_group_size,
        hidden_size,
        valid_mode=valid_mode,
        num_rows=num_rows,
        dtype=torch.bfloat16,
        device=device,
    )

    jax_x = jnp.array(
        x.detach().cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16
    )
    jax_indices = jnp.array(indices.detach().cpu().numpy(), dtype=jnp.int32)
    jax_topk_weights = jnp.array(
        topk_weights.detach().cpu().to(torch.float32).numpy(),
        dtype=jnp.bfloat16,
    )
    jax_valid_rows_mask = jnp.array(
        valid_rows_mask.detach().cpu().numpy(), dtype=jnp.bool_
    )

    op_jax = jax_pallas_mosaic_tpu.PallasTpuRaggedGatherReduce()
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedGatherReduce

    ref_out = op_jax(
        jax_x,
        jax_indices,
        jax_topk_weights,
        jax_valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )
    pallas_out = op_pallas(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )
    base_out = torch_base.RaggedGatherReduce(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )

    ref_out_as_torch = torch.as_tensor(
        np.asarray(ref_out, dtype=np.float32),
        device=device,
        dtype=torch.bfloat16,
    )

    self.assertEqual(pallas_out.shape, (num_tokens, hidden_size))
    self.assertEqual(pallas_out.shape, ref_out.shape)
    torch.testing.assert_close(
        pallas_out, ref_out_as_torch, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        pallas_out.to(torch.float32),
        base_out.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )

  @parameterized.parameters(1024, 2048, 4096)
  def test_autotuning_configs(self, hidden_size: int):
    """Checks every autotuning config matches the reference."""
    reduce_group_size = 8
    num_tokens = (1 << 19) // hidden_size  # 64 MiB of `x` (num_tokens * 8 rows)
    x, indices, topk_weights, valid_rows_mask = _make_torch_inputs(
        num_tokens, reduce_group_size, hidden_size
    )
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedGatherReduce
    heuristics_config, _ = torch_utils.get_configs(
        op_pallas,
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
        from_autotuning_cache=False,
    )
    ba = op_pallas.get_bound_args(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )
    autotuning_configs = ba.autotuning_configs
    self.assertIn(heuristics_config, autotuning_configs)
    self.assertGreater(len(autotuning_configs), 1)

    expected = torch_base.RaggedGatherReduce(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )
    for config in autotuning_configs:
      with self.subTest(str(config)):
        out = op_pallas(
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size=reduce_group_size,
            config=config,
        )
        torch.testing.assert_close(
            out.to(torch.float32),
            expected.to(torch.float32),
            rtol=1e-2,
            atol=1e-2,
        )

  def test_call_without_config_uses_heuristics_config(self):
    num_tokens = 1024
    reduce_group_size = 8
    hidden_size = 4096
    x, indices, topk_weights, valid_rows_mask = _make_torch_inputs(
        num_tokens, reduce_group_size, hidden_size
    )

    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedGatherReduce
    expected_config, _ = torch_utils.get_configs(
        op_pallas,
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
        from_autotuning_cache=False,
    )

    out = op_pallas(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )
    expected_out = op_pallas(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
        config=expected_config,
    )
    torch.testing.assert_close(out, expected_out)

  @parameterized.named_parameters(
      dict(
          testcase_name="small_bf16",
          num_tokens=256,
          reduce_group_size=4,
          hidden_size=512,
          tol=1e-2,
      ),
      dict(
          testcase_name="large_sc_bf16_4096",
          num_tokens=1024,
          reduce_group_size=8,
          hidden_size=4096,
          tol=1e-5,
      ),
      dict(
          testcase_name="large_sc_bf16_7168",
          num_tokens=1024,
          reduce_group_size=8,
          hidden_size=7168,
          tol=1e-5,
      ),
  )
  def test_torch_compile(
      self,
      num_tokens: int,
      reduce_group_size: int,
      hidden_size: int,
      tol: float,
  ):
    x, indices, topk_weights, valid_rows_mask = _make_torch_inputs(
        num_tokens, reduce_group_size, hidden_size
    )

    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedGatherReduce
    expected_config, _ = torch_utils.get_configs(
        op_pallas,
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
        from_autotuning_cache=False,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, indices, topk_weights, valid_rows_mask):
      return op_pallas(
          x,
          indices,
          topk_weights,
          valid_rows_mask,
          reduce_group_size=reduce_group_size,
      )

    out = compiled_fn(x, indices, topk_weights, valid_rows_mask)
    expected_out = op_pallas(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
        config=expected_config,
    )
    torch.testing.assert_close(out, expected_out, rtol=tol, atol=tol)

  def test_deconstruct_and_reconstruct_config_roundtrip(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedGatherReduce
    for num_col_partitions, expected_tuple in ((None, (-1,)), (4, (4,))):
      cfg = torch_pallas_mosaic_tpu.Config(
          num_column_partitions=num_col_partitions
      )
      deconstructed = op_pallas.deconstruct_config(cfg)
      self.assertEqual(deconstructed, expected_tuple)
      reconstructed = op_pallas.reconstruct_config(deconstructed)
      self.assertEqual(reconstructed, cfg)
    self.assertEqual(op_pallas.deconstruct_config((8,)), (8,))
    self.assertEqual(
        op_pallas.reconstruct_config((8,)),
        torch_pallas_mosaic_tpu.Config(num_column_partitions=8),
    )
    self.assertIsNone(op_pallas.deconstruct_config(None))

  def test_op_rejects_float32(self):
    device = "tpu"
    x = torch.zeros((64, 128), dtype=torch.float32, device=device)
    indices = torch.zeros((64,), dtype=torch.int32, device=device)
    topk_weights = torch.zeros((64,), dtype=torch.float32, device=device)
    valid_rows_mask = torch.zeros((64,), dtype=torch.bool, device=device)
    with self.assertRaises(NotImplementedError):
      torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedGatherReduce(
          x,
          indices,
          topk_weights,
          valid_rows_mask,
          reduce_group_size=8,
      )

  def test_fake_impl_symbolic_shapes(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedGatherReduce
    reduce_group_size = 4
    hidden_size = 128

    class _Module(torch.nn.Module):

      def forward(
          self,
          x: torch.Tensor,
          indices: torch.Tensor,
          topk_weights: torch.Tensor,
          valid_rows_mask: torch.Tensor,
      ) -> torch.Tensor:
        return op_pallas(
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size=reduce_group_size,
        )

    with fake_tensor.FakeTensorMode():
      x_fake = torch.empty((300, hidden_size), dtype=torch.bfloat16)
      indices_fake = torch.empty((512,), dtype=torch.int32)
      topk_weights_fake = torch.empty((512,), dtype=torch.bfloat16)
      valid_rows_mask_fake = torch.empty((512,), dtype=torch.bool)
      out_fake = op_pallas(
          x_fake,
          indices_fake,
          topk_weights_fake,
          valid_rows_mask_fake,
          reduce_group_size=reduce_group_size,
      )
      self.assertEqual(
          tuple(out_fake.shape), (512 // reduce_group_size, hidden_size)
      )
      self.assertEqual(out_fake.dtype, torch.bfloat16)

    num_groups = torch.export.Dim("num_groups", min=1, max=256)
    input_dim = reduce_group_size * num_groups
    exported = torch.export.export(
        _Module(),
        (
            torch.ones((512, hidden_size), dtype=torch.bfloat16),
            torch.zeros((512,), dtype=torch.int32),
            torch.ones((512,), dtype=torch.bfloat16),
            torch.ones((512,), dtype=torch.bool),
        ),
        dynamic_shapes=(
            {0: input_dim},
            {0: input_dim},
            {0: input_dim},
            {0: input_dim},
        ),
    )
    op_nodes = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function"
        and op_pallas.jax_op_name in str(node.target)
    ]
    self.assertLen(op_nodes, 1)
    out_meta = op_nodes[0].meta["val"]
    self.assertIsInstance(out_meta.shape[0], torch.SymInt)
    self.assertEqual(out_meta.shape[1], hidden_size)


if __name__ == "__main__":
  absltest.main()
