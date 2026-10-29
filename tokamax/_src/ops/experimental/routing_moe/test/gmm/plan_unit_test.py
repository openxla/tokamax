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
"""Unit tests for the routing_gmm planner.

The planner only does arithmetic on shapes and TPU info, so these tests run on
CPU against a fixed TPU generation.
"""

import math

from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan


def _tpu_info() -> pltpu.TpuInfo:
  return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _resolve(
    num_tokens: int,
    custom_plan: gmm_plan.RoutingGmmPlan | None = None,
    *,
    quantized: bool = True,
    with_down: bool = False,
) -> gmm_plan.ResolvedGmmPlan:
  """Resolves the plan of a 512-expert, top-10 MoE layer with EP=4."""
  num_experts, ep_size, top_k, size_k, size_n = 512, 4, 10, 2560, 1280
  if with_down:
    # The fused kernel needs the intermediate size to be a multiple of 256.
    size_n = 1536
  x = arg.ActivationSpec(
      row_size=num_tokens, col_size=size_k, dtype=jnp.dtype(jnp.bfloat16)
  )
  w_dtype = jnp.dtype(jnp.float8_e4m3fn if quantized else jnp.bfloat16)

  def weight_spec(in_size: int, out_size: int) -> arg.WeightSpec:
    return arg.WeightSpec(
        group_size=num_experts // ep_size,
        in_size=in_size,
        out_size=out_size,
        dtype=w_dtype,
        quant=arg.QuantSpec(block_size=in_size) if quantized else None,
    )

  # Gate/up projection, and optionally the down projection, which takes the
  # activation of the gate/up output (half its width).
  w = weight_spec(size_k, size_n)
  w2 = weight_spec(size_n // 2, size_k) if with_down else None
  return gmm_plan.ResolvedGmmPlan.create(
      custom_plan,
      x=x,
      w=w,
      w2=w2,
      is_fused_act=with_down,
      size_out_m=num_tokens * top_k,
      size_global_groups=num_experts,
      size_k=size_k,
      size_k2=None if w2 is None else w2.in_size,
      size_out_n=size_k if with_down else size_n,
      tpu_info=_tpu_info(),
  )


def _sublane_tiling(quantized: bool) -> int:
  matmul_dtype = jnp.dtype(jnp.float8_e4m3fn if quantized else jnp.bfloat16)
  return max(
      _tpu_info().get_sublane_tiling(jnp.dtype(jnp.bfloat16)),
      _tpu_info().get_sublane_tiling(matmul_dtype),
  )


class TileMTest(parameterized.TestCase):
  """Checks the tile_m the planner picks."""

  @parameterized.product(num_tokens=[1, 4, 8, 12], quantized=[True, False])
  def test_decode_tile_is_one_sublane_tile(
      self, num_tokens: int, quantized: bool
  ):
    # An expert gets at most one row per token, so one sublane tile suffices.
    resolved = _resolve(num_tokens, quantized=quantized)
    min_tile_m = _sublane_tiling(quantized)
    self.assertEqual(resolved.tile_m, min_tile_m)
    self.assertEqual(resolved.bucket_rows, [min_tile_m])

  @parameterized.parameters(64, 200)
  def test_tile_is_capped_at_tokens(self, num_tokens: int):
    min_tile_m = _sublane_tiling(quantized=True)
    self.assertEqual(
        _resolve(num_tokens).tile_m,
        min_tile_m * math.ceil(num_tokens / min_tile_m),
    )

  @parameterized.parameters(256, 4096)
  def test_tile_is_unchanged_from_256_tokens(self, num_tokens: int):
    self.assertEqual(_resolve(num_tokens).tile_m, 256)

  def test_explicit_tile_m_is_kept(self):
    custom_plan = gmm_plan.RoutingGmmPlan(tile_m=128)
    self.assertEqual(_resolve(8, custom_plan).tile_m, 128)


class NumWBuffersTest(parameterized.TestCase):
  """Checks the number of weight buffers the planner picks."""

  @parameterized.product(
      num_tokens=[1, 8, 128, 256, 4096], quantized=[True, False]
  )
  def test_spare_vmem_grows_weight_buffers_to_three(
      self, num_tokens: int, quantized: bool
  ):
    resolved = _resolve(num_tokens, quantized=quantized)
    self.assertEqual(resolved.num_w_buffers, 3)

  @parameterized.parameters(2, 4)
  def test_max_num_w_buffers_caps_growth(self, max_num_w_buffers: int):
    custom_plan = gmm_plan.RoutingGmmPlan(max_num_w_buffers=max_num_w_buffers)
    resolved = _resolve(8, custom_plan)
    self.assertEqual(resolved.num_w_buffers, max_num_w_buffers)

  @parameterized.parameters(2, 4)
  def test_explicit_num_w_buffers_is_kept(self, num_w_buffers: int):
    custom_plan = gmm_plan.RoutingGmmPlan(num_w_buffers=num_w_buffers)
    self.assertEqual(_resolve(8, custom_plan).num_w_buffers, num_w_buffers)

  def test_extra_weight_buffers_do_not_break_fusion(self):
    resolved = _resolve(8, with_down=True)
    self.assertTrue(resolved.fuse_up_and_down)
    self.assertEqual(resolved.num_w_buffers, 3)
    # This VMEM limit fits the fused kernel with 2 weight buffers but not with
    # 3, so the default 2 is kept and the projections stay fused.
    for fuse_up_and_down in (None, True):
      with self.subTest(fuse_up_and_down=fuse_up_and_down):
        custom_plan = gmm_plan.RoutingGmmPlan(
            vmem_limit_bytes=16 * 1024 * 1024,
            fuse_up_and_down=fuse_up_and_down,
        )
        resolved = _resolve(8, custom_plan, with_down=True)
        self.assertTrue(resolved.fuse_up_and_down)
        self.assertEqual(resolved.num_w_buffers, 2)


if __name__ == "__main__":
  absltest.main()
