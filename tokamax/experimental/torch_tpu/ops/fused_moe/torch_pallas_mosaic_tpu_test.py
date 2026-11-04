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
"""Tests for Pallas Mosaic TPU Fused MoE."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from tokamax._src.ops.experimental.fused_moe import kernel as jax_kernel
from tokamax._src.ops.experimental.fused_moe import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.fused_moe import torch_base
from tokamax.experimental.torch_tpu.ops.fused_moe import torch_pallas_mosaic_tpu
import torch
import torch_tpu  # pylint: disable=unused-import


class PallasMosaicTpuFusedMoeTest(parameterized.TestCase):

  def _generate_inputs(
      self,
      tokens: int = 256,
      hidden: int = 512,
      inter: int = 256,
      experts: int = 32,
  ):
    device = "tpu"
    torch.manual_seed(42)
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device)
    w1 = torch.randn(
        (experts, hidden, 2 * inter), dtype=torch.bfloat16, device=device
    )
    w2 = torch.randn(
        (experts, inter, hidden), dtype=torch.bfloat16, device=device
    )
    gating = torch.randn((tokens, experts), dtype=torch.float32, device=device)
    return x, w1, w2, gating

  @parameterized.named_parameters(
      dict(
          testcase_name="renormalize_true",
          topk=4,
          renormalize=True,
      ),
      dict(
          testcase_name="renormalize_false",
          topk=4,
          renormalize=False,
      ),
  )
  def test_pallas_matches_jax_op_and_reference(self, topk, renormalize):
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")

    device = "tpu"
    x, w1, w2, gating = self._generate_inputs()

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

    mesh = Mesh(np.asarray(jax.devices()[:1]), axis_names=(jax_kernel.AXIS,))
    op_jax_kernel = jax_pallas_mosaic_tpu.PallasMosaicTpuFusedMoe()
    jax_out, _ = op_jax_kernel._fwd(
        x=jax_x,
        w1=jax_w1,
        w2=jax_w2,
        gating=jax_gating,
        topk=topk,
        renormalize=renormalize,
        act_fn="silu",
        mesh=mesh,
    )

    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuFusedMoe
    pallas_out = op_pallas(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
        renormalize=renormalize,
        act_fn="silu",
    )

    self.assertEqual(pallas_out.shape, jax_out.shape)
    jax_out_as_torch = torch.as_tensor(
        np.asarray(jax_out, dtype=np.float32),
        device=device,
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(
        pallas_out, jax_out_as_torch, atol=1e-2, rtol=1e-2
    )

    base_out = torch_base.FusedMoe(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=topk,
        renormalize=renormalize,
        act_fn="silu",
    )
    np_out = pallas_out.detach().cpu().to(torch.float64).numpy()
    np_ref = base_out.detach().cpu().to(torch.float64).numpy()
    rel_l2 = float(np.linalg.norm(np_out - np_ref) / np.linalg.norm(np_ref))
    self.assertLess(rel_l2, 0.01)

  def test_call_without_config_uses_heuristics_config(self):
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")

    x, w1, w2, gating = self._generate_inputs()
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuFusedMoe
    expected_config, _ = torch_utils.get_configs(
        op_pallas,
        x,
        w1,
        w2,
        gating,
        topk=4,
        renormalize=True,
        act_fn="silu",
        from_autotuning_cache=False,
    )

    out = op_pallas(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=4,
        renormalize=True,
        act_fn="silu",
    )
    expected_out = op_pallas(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=4,
        renormalize=True,
        act_fn="silu",
        config=expected_config,
    )
    torch.testing.assert_close(out, expected_out)

  def test_torch_compile_call_without_config_uses_heuristics_config(self):
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")

    x, w1, w2, gating = self._generate_inputs()
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuFusedMoe
    expected_config, _ = torch_utils.get_configs(
        op_pallas,
        x,
        w1,
        w2,
        gating,
        topk=4,
        renormalize=True,
        act_fn="silu",
        from_autotuning_cache=False,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, w1, w2, gating):
      return op_pallas(
          x=x,
          w1=w1,
          w2=w2,
          gating=gating,
          topk=4,
          renormalize=True,
          act_fn="silu",
      )

    out = compiled_fn(x, w1, w2, gating)
    expected_out = op_pallas(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=4,
        renormalize=True,
        act_fn="silu",
        config=expected_config,
    )
    torch.testing.assert_close(out, expected_out)

  def test_torch_compile_with_config(self):
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")

    x, w1, w2, gating = self._generate_inputs()
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuFusedMoe
    config = torch_pallas_mosaic_tpu.Config(
        capacity=128,
        block=128,
        ragged_stride=1280,
        sharded_plan=True,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, w1, w2, gating):
      x = x + 1.0
      out = op_pallas(
          x=x,
          w1=w1,
          w2=w2,
          gating=gating,
          topk=4,
          renormalize=True,
          act_fn="silu",
          config=config,
      )
      return out + 2.0

    out_compiled = compiled_fn(x, w1, w2, gating)
    out_expected = (
        op_pallas(
            x=x + 1.0,
            w1=w1,
            w2=w2,
            gating=gating,
            topk=4,
            renormalize=True,
            act_fn="silu",
            config=config,
        )
        + 2.0
    )
    torch.testing.assert_close(out_compiled, out_expected)

  def test_fake_impl_symbolic_shapes(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuFusedMoe

    class _Module(torch.nn.Module):

      def forward(
          self,
          x: torch.Tensor,
          w1: torch.Tensor,
          w2: torch.Tensor,
          gating: torch.Tensor,
      ) -> torch.Tensor:
        return op_pallas(x=x, w1=w1, w2=w2, gating=gating)

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    x = torch.ones((256, 512), dtype=torch.bfloat16)
    w1 = torch.ones((32, 512, 512), dtype=torch.bfloat16)
    w2 = torch.ones((32, 256, 512), dtype=torch.bfloat16)
    gating = torch.ones((256, 32), dtype=torch.float32)
    exported = torch.export.export(
        _Module(),
        (x, w1, w2, gating),
        dynamic_shapes=({0: num_tokens}, None, None, {0: num_tokens}),
    )
    op_nodes = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function"
        and op_pallas.jax_op_name in str(node.target)
    ]
    self.assertLen(op_nodes, 1)
    meta = op_nodes[0].meta["val"]
    self.assertIsInstance(meta.shape[0], torch.SymInt)
    self.assertEqual(meta.shape[1], 512)

  def test_op_impl_call_respects_abstract_mesh(self):
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuFusedMoe
    config = op_pallas.deconstruct_config(
        torch_pallas_mosaic_tpu.Config(
            capacity=128,
            block=128,
            ragged_stride=1280,
            sharded_plan=True,
        )
    )
    abstract_device = jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    specs = (
        jax.ShapeDtypeStruct((256, 512), jnp.bfloat16),
        jax.ShapeDtypeStruct((32, 512, 512), jnp.bfloat16),
        jax.ShapeDtypeStruct((32, 256, 512), jnp.bfloat16),
        jax.ShapeDtypeStruct((256, 32), jnp.float32),
    )
    fn = jax.jit(
        lambda x, w1, w2, gating: op_pallas.op_impl_call(
            x,
            w1,
            w2,
            gating,
            topk=4,
            renormalize=True,
            act_fn="silu",
            config=config,
        )
    )

    for axis_sizes, axis_names, axis_types in (
        ((1,), ("x",), (jax.sharding.AxisType.Auto,)),
        ((2,), ("my_ep",), (jax.sharding.AxisType.Explicit,)),
    ):
      with self.subTest(axis_sizes=axis_sizes, axis_names=axis_names):
        abstract_mesh = jax.sharding.AbstractMesh(
            axis_sizes,
            axis_names,
            axis_types=axis_types,
            abstract_device=abstract_device,
        )
        with (
            jax.sharding.use_abstract_mesh(abstract_mesh),
            mock.patch.object(
                jax,
                "devices",
                side_effect=RuntimeError("jax.devices() should not be called"),
            ),
        ):
          exported = jax.export.export(fn)(*specs)
        self.assertEqual(exported.out_avals[0].shape, (256, 512))

    bad_mesh = jax.sharding.AbstractMesh(
        (1, 1), ("dp", "ep"), abstract_device=abstract_device
    )
    with jax.sharding.use_abstract_mesh(bad_mesh):
      with self.assertRaisesRegex(ValueError, "requires a 1D mesh"):
        jax.export.export(fn)(*specs)


if __name__ == "__main__":
  absltest.main()
