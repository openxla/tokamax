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
"""Tests for Pallas Mosaic TPU Routing MoE PyTorch bridge."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.routing_moe import torch_base
from tokamax.experimental.torch_tpu.ops.routing_moe import torch_pallas_mosaic_tpu
import torch
import torch_tpu  # pylint: disable=unused-import


def _quantize_fp4_kmajor(w: torch.Tensor, block: int):
  """Quantizes `[G, K, N]` float32 tensor on TPU to `float4_e2m1fn_x2`."""

  def _jax_quant(w_jax: jax.Array) -> tuple[jax.Array, jax.Array]:
    g, k, n = w_jax.shape
    blocks = w_jax.astype(jnp.float32).reshape(g, k // block, block, n)
    scale = jnp.max(jnp.abs(blocks), axis=2, keepdims=True) / 6.0
    inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    q = (blocks * inv).astype(jnp.float4_e2m1fn).reshape(g, k, n)
    return q, scale.astype(jnp.float32)

  op = torch.tpu.pallas.jax_op(f"tokamax_test::quant_fp4_b{block}", _jax_quant)
  return op(w)


class PallasMosaicTpuRoutingMoeTorchTest(parameterized.TestCase):

  def test_bf16_and_ep_shard_with_padded_tokens(self):
    if jax.default_backend() != "tpu":
      self.skipTest("TPU required.")
    device = "tpu"
    torch.manual_seed(42)
    tokens, hidden, inter, local_experts, global_experts, topk = (
        16,
        2560,
        640,
        8,
        32,
        10,
    )
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device) * 0.2
    w1 = (
        torch.randn(
            (local_experts, hidden, 2 * inter),
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.1
    )
    w2 = (
        torch.randn(
            (local_experts, inter, hidden), dtype=torch.bfloat16, device=device
        )
        * 0.1
    )
    logits = torch.randn(
        (tokens, global_experts), dtype=torch.float32, device=device
    )
    vals, ids = torch.topk(logits, k=topk, dim=-1)
    weights = torch.softmax(vals, dim=-1)
    weights[-4:] = 0.0
    ids = ids.to(torch.int32)
    ids[-4:] = torch.arange(topk, dtype=torch.int32, device=device)
    experts_start = torch.tensor(
        local_experts, dtype=torch.int32, device=device
    )

    pallas_out = torch_pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe(
        x,
        weights,
        ids,
        w1,
        w2,
        experts_start=experts_start,
        global_num_experts=global_experts,
        skip_padded_tokens=True,
    )
    ref_out = torch_base.RoutingMoe(
        x,
        weights,
        ids,
        w1,
        w2,
        experts_start=experts_start,
        global_num_experts=global_experts,
        skip_padded_tokens=True,
    )
    np_out = pallas_out.detach().cpu().to(torch.float64).numpy()
    np_ref = ref_out.detach().cpu().to(torch.float64).numpy()
    np.testing.assert_array_equal(np_out[-4:], np.zeros((4, hidden)))
    rel_l2 = float(
        np.linalg.norm(np_out - np_ref) / max(np.linalg.norm(np_ref), 1e-8)
    )
    self.assertLess(rel_l2, 0.02)

  def test_fp4_w4a8_qwen38_widths_and_torch_compile(self):
    if jax.default_backend() != "tpu":
      self.skipTest("TPU required.")
    if not hasattr(torch, "float4_e2m1fn_x2"):
      self.skipTest("torch.float4_e2m1fn_x2 required.")
    device = "tpu"
    torch.manual_seed(7)
    tokens, hidden, inter, local_experts, global_experts, topk = (
        16,
        2560,
        640,
        8,
        32,
        10,
    )
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device) * 0.2
    w1_f = (
        torch.randn(
            (local_experts, hidden, 2 * inter),
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    )
    w2_f = (
        torch.randn(
            (local_experts, inter, hidden), dtype=torch.float32, device=device
        )
        * 0.1
    )
    w1_fp4, w1_scale = _quantize_fp4_kmajor(w1_f, block=128)
    w2_fp4, w2_scale = _quantize_fp4_kmajor(w2_f, block=128)
    logits = torch.randn(
        (tokens, global_experts), dtype=torch.float32, device=device
    )
    vals, ids = torch.topk(logits, k=topk, dim=-1)
    weights = torch.softmax(vals, dim=-1)
    ids = ids.to(torch.int32)
    experts_start = torch.tensor(0, dtype=torch.int32, device=device)

    op = torch_pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe
    expected_cfg, _ = torch_utils.get_configs(
        op,
        x,
        weights,
        ids,
        w1_fp4,
        w2_fp4,
        w1_scale,
        w2_scale,
        None,
        None,
        experts_start,
        global_num_experts=global_experts,
        skip_padded_tokens=True,
    )
    self.assertEqual(expected_cfg.bucket_base, 8)
    self.assertEqual(expected_cfg.tile_m, 16)

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x_in, w_in, id_in, es_in):
      return op(
          x_in,
          w_in,
          id_in,
          w1_fp4,
          w2_fp4,
          w1_scale,
          w2_scale,
          experts_start=es_in,
          global_num_experts=global_experts,
          skip_padded_tokens=True,
      )

    out_compiled = compiled_fn(x, weights, ids, experts_start)
    out_eager = op(
        x,
        weights,
        ids,
        w1_fp4,
        w2_fp4,
        w1_scale,
        w2_scale,
        experts_start=experts_start,
        global_num_experts=global_experts,
        skip_padded_tokens=True,
    )
    torch.testing.assert_close(out_compiled, out_eager)

    fp4_ref_out = torch_base.RoutingMoe(
        x,
        weights,
        ids,
        w1_fp4,
        w2_fp4,
        w1_scale,
        w2_scale,
        experts_start=experts_start,
        global_num_experts=global_experts,
        skip_padded_tokens=True,
    )
    np_out = out_compiled.detach().cpu().to(torch.float64).numpy()
    np_fp4_ref = fp4_ref_out.detach().cpu().to(torch.float64).numpy()
    fp4_rel_l2 = float(
        np.linalg.norm(np_out - np_fp4_ref)
        / max(np.linalg.norm(np_fp4_ref), 1e-8)
    )
    self.assertLess(fp4_rel_l2, 0.06)

    bf16_ref_out = torch_base.RoutingMoe(
        x,
        weights,
        ids,
        w1_f.to(torch.bfloat16),
        w2_f.to(torch.bfloat16),
        experts_start=experts_start,
        global_num_experts=global_experts,
        skip_padded_tokens=True,
    )
    np_bf16_ref = bf16_ref_out.detach().cpu().to(torch.float64).numpy()
    bf16_rel_l2 = float(
        np.linalg.norm(np_out - np_bf16_ref)
        / max(np.linalg.norm(np_bf16_ref), 1e-8)
    )
    self.assertLess(bf16_rel_l2, 0.25)

  def test_fake_impl_symbolic_shapes(self):
    op = torch_pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe
    encoded_cfg = op.deconstruct_config(torch_pallas_mosaic_tpu.Config())
    self.assertEqual(op.deconstruct_config(encoded_cfg), encoded_cfg)
    self.assertEqual(
        op.reconstruct_config(encoded_cfg), torch_pallas_mosaic_tpu.Config()
    )

    class _Mod(torch.nn.Module):

      def forward(self, x, weights, ids, w1, w2):
        return op(x, weights, ids, w1, w2, skip_padded_tokens=True)

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    x = torch.ones((32, 512), dtype=torch.bfloat16)
    weights = torch.ones((32, 4), dtype=torch.float32)
    ids = torch.zeros((32, 4), dtype=torch.int32)
    w1 = torch.ones((8, 512, 1280), dtype=torch.bfloat16)
    w2 = torch.ones((8, 640, 512), dtype=torch.bfloat16)
    exported = torch.export.export(
        _Mod(),
        (x, weights, ids, w1, w2),
        dynamic_shapes=(
            {0: num_tokens},
            {0: num_tokens},
            {0: num_tokens},
            None,
            None,
        ),
    )
    nodes = [
        n
        for n in exported.graph.nodes
        if n.op == "call_function" and op.jax_op_name in str(n.target)
    ]
    self.assertLen(nodes, 1)
    meta = nodes[0].meta["val"]
    self.assertIsInstance(meta.shape[0], torch.SymInt)
    self.assertEqual(meta.shape[1], 512)


if __name__ == "__main__":
  absltest.main()
