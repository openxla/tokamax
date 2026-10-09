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
"""Tests for the base class of the Ragged Gather Reduce PyTorch Op."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_gather_reduce import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.ragged_gather_reduce import torch_base
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


class RaggedGatherReduceBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")
    torch.manual_seed(0)

  @parameterized.product(
      params=[
          (512, 512, 128),
          (512, 300, 128),
          (1024, 1024, 512),
      ],
      reduce_group_size=[1, 4, 8],
      dtype=[torch.bfloat16, torch.float32],
  )
  def test_base_matches_reference(
      self,
      params: tuple[int, int, int],
      reduce_group_size: int,
      dtype: torch.dtype,
  ):
    input_size, num_rows, hidden_size = params
    device = "tpu"

    x = torch.randn((num_rows, hidden_size), dtype=dtype, device=device)
    indices = torch.randint(
        0, num_rows, (input_size,), dtype=torch.int32, device=device
    )
    topk_weights = torch.randn((input_size,), dtype=dtype, device=device)
    valid_rows_mask = (
        torch.randint(0, 2, (input_size,), dtype=torch.int32, device=device)
        > 0
    )

    jax_dtype = jnp.bfloat16 if dtype == torch.bfloat16 else jnp.float32
    jax_x = jnp.array(
        x.detach().cpu().to(torch.float32).numpy(), dtype=jax_dtype
    )
    jax_indices = jnp.array(indices.detach().cpu().numpy(), dtype=jnp.int32)
    jax_topk_weights = jnp.array(
        topk_weights.detach().cpu().to(torch.float32).numpy(), dtype=jax_dtype
    )
    jax_valid_rows_mask = jnp.array(
        valid_rows_mask.detach().cpu().numpy(), dtype=jnp.bool_
    )

    forward_config, backward_config = torch_utils.get_configs(
        torch_base.RaggedGatherReduce,
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(forward_config)
    self.assertIsNone(backward_config)

    ref_out = jax_base.RaggedGatherReduce()(
        jax_x,
        jax_indices,
        jax_topk_weights,
        jax_valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )

    base_out = torch_base.RaggedGatherReduce(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
        config=forward_config,
    )

    self.assertEqual(base_out.shape, ref_out.shape)
    ref_out_as_torch = torch.as_tensor(
        np.asarray(ref_out, dtype=np.float32),
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(
        base_out, ref_out_as_torch, rtol=1e-5, atol=1e-5
    )

  def test_torch_compile(self):
    device = "tpu"
    input_size = 512
    num_rows = 300
    hidden_size = 128
    reduce_group_size = 4

    x = torch.randn(
        (num_rows, hidden_size), dtype=torch.float32, device=device
    )
    indices = torch.randint(
        0, num_rows, (input_size,), dtype=torch.int32, device=device
    )
    topk_weights = torch.randn(
        (input_size,), dtype=torch.float32, device=device
    )
    valid_rows_mask = (
        torch.randint(0, 2, (input_size,), dtype=torch.int32, device=device)
        > 0
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, indices, topk_weights, valid_rows_mask):
      return torch_base.RaggedGatherReduce(
          x,
          indices,
          topk_weights,
          valid_rows_mask,
          reduce_group_size=reduce_group_size,
      )

    actual = compiled_fn(x, indices, topk_weights, valid_rows_mask)
    expected = torch_base.RaggedGatherReduce(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

  def test_fake_impl_symbolic_shapes(self):
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
        return torch_base.RaggedGatherReduce(
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
      out_fake = torch_base.RaggedGatherReduce(
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
            torch.ones((300, hidden_size), dtype=torch.bfloat16),
            torch.zeros((512,), dtype=torch.int32),
            torch.ones((512,), dtype=torch.bfloat16),
            torch.ones((512,), dtype=torch.bool),
        ),
        dynamic_shapes=(
            None,
            {0: input_dim},
            {0: input_dim},
            {0: input_dim},
        ),
    )
    op_nodes = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function"
        and torch_base.RaggedGatherReduce.jax_op_name in str(node.target)
    ]
    self.assertLen(op_nodes, 1)
    out_meta = op_nodes[0].meta["val"]
    self.assertIsInstance(out_meta.shape[0], torch.SymInt)
    self.assertEqual(out_meta.shape[1], hidden_size)


if __name__ == "__main__":
  absltest.main()
