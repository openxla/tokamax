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
"""Tests for Fused MoE base operator."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.fused_moe import base as jax_base
from tokamax.experimental.torch_tpu.ops.fused_moe import torch_base
import torch
import torch_tpu  # pylint: disable=unused-import


class FusedMoeBaseTest(parameterized.TestCase):

  @parameterized.product(
      act_fn=["silu"],
      renormalize=[True, False],
      topk=[2],
  )
  def test_base_matches_reference(self, act_fn, renormalize, topk):
    tokens = 16
    hidden = 64
    inter = 32
    experts = 4

    torch.manual_seed(0)
    device = "tpu"

    x = torch.randn(
        (tokens, hidden),
        dtype=torch.bfloat16,
        device=device,
    )
    w1 = torch.randn(
        (experts, hidden, 2 * inter),
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.randn(
        (experts, inter, hidden),
        dtype=torch.bfloat16,
        device=device,
    )
    gating = torch.randn(
        (tokens, experts),
        dtype=torch.float32,
        device=device,
    )

    jax_x = jnp.array(
        x.detach().cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16
    )
    jax_w1 = jnp.array(
        w1.detach().cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16
    )
    jax_w2 = jnp.array(
        w2.detach().cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16
    )
    jax_gating = jnp.array(gating.detach().cpu().numpy(), dtype=jnp.float32)

    ref_out, _ = jax_base.FusedMoe()._fwd(
        x=jax_x,
        w1=jax_w1,
        w2=jax_w2,
        gating=jax_gating,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
    )

    base_out = torch_base.FusedMoe(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
    )

    self.assertEqual(ref_out.shape, base_out.shape)
    ref_out_as_torch = torch.as_tensor(
        np.asarray(ref_out, dtype=np.float32),
        device=device,
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(
        base_out, ref_out_as_torch, rtol=1e-3, atol=1e-3
    )

  def test_torch_compile(self):
    tokens = 16
    hidden = 64
    inter = 32
    experts = 4

    torch.manual_seed(0)
    device = "tpu"

    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device)
    w1 = torch.randn(
        (experts, hidden, 2 * inter), dtype=torch.bfloat16, device=device
    )
    w2 = torch.randn(
        (experts, inter, hidden), dtype=torch.bfloat16, device=device
    )
    gating = torch.randn((tokens, experts), dtype=torch.float32, device=device)

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, w1, w2, gating):
      return torch_base.FusedMoe(
          x=x,
          w1=w1,
          w2=w2,
          gating=gating,
          topk=2,
          renormalize=True,
          act_fn="silu",
      )

    out_compiled = compiled_fn(x, w1, w2, gating)
    out_eager = torch_base.FusedMoe(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=2,
        renormalize=True,
        act_fn="silu",
    )
    torch.testing.assert_close(
        out_compiled, out_eager, rtol=1e-3, atol=1e-3
    )

  def test_fake_impl_symbolic_shapes(self):
    class _Module(torch.nn.Module):

      def forward(
          self,
          x: torch.Tensor,
          w1: torch.Tensor,
          w2: torch.Tensor,
          gating: torch.Tensor,
      ) -> torch.Tensor:
        return torch_base.FusedMoe(x=x, w1=w1, w2=w2, gating=gating)

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    x = torch.ones((16, 64), dtype=torch.bfloat16)
    w1 = torch.ones((4, 64, 64), dtype=torch.bfloat16)
    w2 = torch.ones((4, 32, 64), dtype=torch.bfloat16)
    gating = torch.ones((16, 4), dtype=torch.float32)
    exported = torch.export.export(
        _Module(),
        (x, w1, w2, gating),
        dynamic_shapes=({0: num_tokens}, None, None, {0: num_tokens}),
    )
    op_nodes = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function"
        and torch_base.FusedMoe.jax_op_name in str(node.target)
    ]
    self.assertLen(op_nodes, 1)
    meta = op_nodes[0].meta["val"]
    self.assertIsInstance(meta.shape[0], torch.SymInt)
    self.assertEqual(meta.shape[1], 64)


if __name__ == "__main__":
  absltest.main()
