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
"""Tests for Pallas Mosaic TPU implementation of Ragged Dot."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_dot import base as jax_base
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2 as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.ragged_dot import torch_base
from tokamax.experimental.torch_tpu.ops.ragged_dot import torch_pallas_mosaic_tpu
import torch
import torch_tpu  # pylint: disable=unused-import

Config = jax_pallas_mosaic_tpu.Config


class PallasMosaicTpuRaggedDotTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")

    seed = absltest.FLAGS.test_random_seed
    if seed is None or not isinstance(seed, int):
      raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)
    self.seed = seed
    torch.manual_seed(seed)

  def _generate_random_data(
      self,
      m: int,
      k: int,
      n: int,
      num_groups: int,
      dtype: torch.dtype = torch.float32,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lhs = torch.randn(
        m,
        k,
        device="tpu",
        dtype=dtype,
        requires_grad=True,
    )
    rhs = torch.randn(
        num_groups,
        k,
        n,
        device="tpu",
        dtype=dtype,
        requires_grad=True,
    )
    gs_tuple = jax_base.generate_group_sizes(
        m=m, num_groups=num_groups, seed=self.seed
    )
    group_sizes = torch.tensor(gs_tuple, device="tpu", dtype=torch.int32)
    dout = torch.randn(m, n, device="tpu", dtype=dtype)
    return lhs, rhs, group_sizes, dout

  def test_ragged_dot_numerics_matches_jax_implementation(self):
    m = 512
    k = 256
    n = 512
    num_groups = 4

    lhs_torch, rhs_torch, group_sizes_torch, dout_torch = (
        self._generate_random_data(m, k, n, num_groups, dtype=torch.float32)
    )

    jax_lhs = jnp.array(lhs_torch.detach().cpu().numpy())
    jax_rhs = jnp.array(rhs_torch.detach().cpu().numpy())
    jax_group_sizes = jnp.array(group_sizes_torch.detach().cpu().numpy())
    jax_dout = jnp.array(dout_torch.detach().cpu().numpy())

    fn_jax = jax_pallas_mosaic_tpu.PallasMosaicTpuV2RaggedDot()
    out_jax, vjp_fn = jax.vjp(
        lambda l, r: fn_jax(l, r, group_sizes=jax_group_sizes),
        jax_lhs,
        jax_rhs,
    )
    dlhs_jax, drhs_jax = vjp_fn(jax_dout)

    out_jax_as_torch = torch.as_tensor(np.asarray(out_jax), device="tpu")
    dlhs_jax_as_torch = torch.as_tensor(np.asarray(dlhs_jax), device="tpu")
    drhs_jax_as_torch = torch.as_tensor(np.asarray(drhs_jax), device="tpu")

    forward_config, backward_config = torch_utils.get_configs(
        torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedDot,
        lhs_torch,
        rhs_torch,
        group_sizes=group_sizes_torch,
        from_autotuning_cache=False,
    )
    self.assertIsInstance(forward_config, Config)
    self.assertIsInstance(backward_config, Config)

    out_tokamax = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(
        lhs_torch,
        rhs_torch,
        group_sizes_torch,
        configs=(forward_config, backward_config),
    )
    out_tokamax.backward(dout_torch)
    dlhs_tokamax = lhs_torch.grad.clone()
    drhs_tokamax = rhs_torch.grad.clone()

    torch.testing.assert_close(
        out_tokamax, out_jax_as_torch, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        dlhs_tokamax, dlhs_jax_as_torch, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        drhs_tokamax, drhs_jax_as_torch, rtol=1e-4, atol=1e-4
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="small_size_test",
          m=256,
          k=128,
          n=256,
          num_groups=4,
      ),
      dict(
          testcase_name="medium_size_test",
          m=1024,
          k=512,
          n=1024,
          num_groups=8,
      ),
  )
  def test_kernel_running_correctly(self, m, k, n, num_groups):
    lhs, rhs, group_sizes, dout = self._generate_random_data(
        m, k, n, num_groups
    )
    forward_config, backward_config = torch_utils.get_configs(
        torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedDot,
        lhs,
        rhs,
        group_sizes,
        from_autotuning_cache=False,
    )
    out = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(
        lhs,
        rhs,
        group_sizes,
        configs=(forward_config, backward_config),
    )
    out.backward(dout)
    dlhs = lhs.grad.clone()
    drhs = rhs.grad.clone()
    self.assertIsNotNone(dlhs)
    self.assertIsNotNone(drhs)
    self.assertIsNotNone(out)

    lhs.grad.zero_()
    rhs.grad.zero_()
    forward_config_ref, backward_config_ref = torch_utils.get_configs(
        torch_base.RaggedDot,
        lhs,
        rhs,
        group_sizes,
        from_autotuning_cache=False,
    )
    out_ref = torch_base.RaggedDot(
        lhs,
        rhs,
        group_sizes,
        configs=(forward_config_ref, backward_config_ref),
    )
    out_ref.backward(dout)
    dlhs_ref = lhs.grad.clone()
    drhs_ref = rhs.grad.clone()

    torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dlhs, dlhs_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(drhs, drhs_ref, rtol=1e-3, atol=1e-3)

  def test_derive_backward_shapes_and_autotuning_cache_configs(self):
    lhs_meta = torch.empty((262144, 7168), dtype=torch.bfloat16, device="meta")
    rhs_meta = torch.empty((8, 7168, 2048), dtype=torch.bfloat16, device="meta")
    group_sizes_meta = torch.empty((8,), dtype=torch.int32, device="meta")

    forward_config, backward_config = torch_utils.get_configs(
        torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedDot,
        lhs_meta,
        rhs_meta,
        group_sizes=group_sizes_meta,
        from_autotuning_cache=True,
    )
    self.assertIsInstance(forward_config, Config)
    self.assertIsInstance(backward_config, Config)

  @parameterized.named_parameters(
      dict(
          testcase_name="small_size_compile_test",
          m=256,
          k=128,
          n=256,
          num_groups=4,
      ),
      dict(
          testcase_name="medium_size_compile_test",
          m=1024,
          k=512,
          n=1024,
          num_groups=8,
      ),
  )
  def test_torch_compile(self, m, k, n, num_groups):
    lhs, rhs, group_sizes, dout = self._generate_random_data(
        m, k, n, num_groups
    )
    forward_config, backward_config = torch_utils.get_configs(
        torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedDot,
        lhs,
        rhs,
        group_sizes=group_sizes,
        from_autotuning_cache=False,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def foo(
        lhs_in: torch.Tensor,
        rhs_in: torch.Tensor,
        gs_in: torch.Tensor,
    ) -> torch.Tensor:
      lhs_in = lhs_in + 1.0
      out = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(
          lhs_in,
          rhs_in,
          gs_in,
          configs=(forward_config, backward_config),
      )
      return out + 2.0

    out_compiled = foo(lhs, rhs, group_sizes)
    out_compiled.backward(dout)
    self.assertIsNotNone(lhs.grad)
    self.assertIsNotNone(rhs.grad)
    self.assertIsNotNone(out_compiled)

  def test_call_without_configs_uses_heuristics_config(self):
    lhs, rhs, group_sizes, dout = self._generate_random_data(
        m=256, k=128, n=256, num_groups=4
    )
    lhs_expected = lhs.clone().detach().requires_grad_(True)
    rhs_expected = rhs.clone().detach().requires_grad_(True)
    op_rd = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedDot
    expected_fwd_config, expected_bwd_config = torch_utils.get_configs(
        op_rd,
        lhs,
        rhs,
        group_sizes=group_sizes,
        from_autotuning_cache=False,
    )

    out = op_rd(lhs, rhs, group_sizes)
    out.backward(dout)

    expected_out = op_rd(
        lhs_expected,
        rhs_expected,
        group_sizes,
        configs=(expected_fwd_config, expected_bwd_config),
    )
    expected_out.backward(dout)

    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(lhs.grad, lhs_expected.grad)
    torch.testing.assert_close(rhs.grad, rhs_expected.grad)

  def test_torch_compile_call_without_configs_uses_heuristics_config(self):
    lhs, rhs, group_sizes, dout = self._generate_random_data(
        m=256, k=128, n=256, num_groups=4
    )
    lhs_expected = lhs.clone().detach().requires_grad_(True)
    rhs_expected = rhs.clone().detach().requires_grad_(True)
    op_rd = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedDot
    expected_fwd_config, expected_bwd_config = torch_utils.get_configs(
        op_rd,
        lhs,
        rhs,
        group_sizes=group_sizes,
        from_autotuning_cache=False,
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        lhs_in: torch.Tensor,
        rhs_in: torch.Tensor,
        gs_in: torch.Tensor,
    ) -> torch.Tensor:
      return op_rd(lhs_in, rhs_in, gs_in)

    out = compiled_fn(lhs, rhs, group_sizes)
    out.backward(dout)

    expected_out = op_rd(
        lhs_expected,
        rhs_expected,
        group_sizes,
        configs=(expected_fwd_config, expected_bwd_config),
    )
    expected_out.backward(dout)

    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(lhs.grad, lhs_expected.grad)
    torch.testing.assert_close(rhs.grad, rhs_expected.grad)

  def test_activation_raises_error(self):
    lhs, rhs, group_sizes, _ = self._generate_random_data(
        m=256, k=128, n=256, num_groups=4
    )
    op_rd = torch_pallas_mosaic_tpu.PallasMosaicTpuRaggedDot
    with self.assertRaisesRegex(
        NotImplementedError,
        "activations are not supported on Torch TPU Tokamax",
    ):
      op_rd(lhs, rhs, group_sizes, activation=jax.nn.relu)


if __name__ == "__main__":
  absltest.main()
