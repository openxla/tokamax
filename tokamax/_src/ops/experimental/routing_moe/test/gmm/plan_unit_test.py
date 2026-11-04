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

import dataclasses
import math

from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.common import spec
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.gmm import spec as gmm_spec_lib


def _tpu_info() -> pltpu.TpuInfo:
  return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _make_spec(
    num_tokens: int,
    *,
    quantized: bool = True,
    with_down: bool = False,
    size_n: int = 1280,
    w_dtype: jnp.dtype | None = None,
) -> gmm_spec_lib.RoutingGmmSpec:
  """Builds a RoutingGmmSpec for a 512-expert, top-10 MoE layer."""
  num_experts, ep_size, top_k, size_k = 512, 4, 10, 2560
  if with_down:
    # The fused kernel needs the intermediate size to be a multiple of 256.
    size_n = 1536
  x = spec.ActivationSpec(
      row_size=num_tokens,
      col_size=size_k,
      dtype=jnp.dtype(jnp.bfloat16),
      quant=None,
  )
  w_dtype = (
      w_dtype
      if w_dtype is not None
      else jnp.dtype(jnp.float8_e4m3fn if quantized else jnp.bfloat16)
  )

  def weight_spec(
      in_size: int, out_size: int, *, fused_gate_up: bool = False
  ) -> spec.WeightSpec:
    return spec.WeightSpec(
        group_size=num_experts // ep_size,
        in_size=in_size,
        out_size=out_size,
        dtype=w_dtype,
        quant=spec.SubchannelQuantSpec(block_size=in_size)
        if quantized
        else None,
        has_bias=False,
        fused_gate_up=fused_gate_up,
    )

  # Gate/up projection, and optionally the down projection, which takes the
  # activation of the gate/up output (half its width).
  w = weight_spec(size_k, size_n, fused_gate_up=with_down)
  w2 = weight_spec(size_n // 2, size_k) if with_down else None
  routed_rows = num_tokens * top_k
  return gmm_spec_lib.RoutingGmmSpec(
      x=x,
      w=w,
      w2=w2,
      gather=spec.RoutingSpec(num_indices=routed_rows, remote_axis=None),
      scatter=spec.RoutingSpec(num_indices=routed_rows, remote_axis=None),
      num_global_groups=num_experts,
      fuse_act="silu" if with_down else None,
      size_ep=1,
  )


def _resolve(
    num_tokens: int,
    custom_plan: gmm_plan.RoutingGmmPlan | None = None,
    *,
    quantized: bool = True,
    with_down: bool = False,
    out_format: out_format_lib.ActivationFormat | None = None,
    size_n: int = 1280,
    tpu_info: pltpu.TpuInfo | None = None,
    w_dtype: jnp.dtype | None = None,
) -> gmm_plan.ResolvedGmmPlan:
  gmm_spec = _make_spec(
      num_tokens,
      quantized=quantized,
      with_down=with_down,
      size_n=size_n,
      w_dtype=w_dtype,
  )
  resolved_out_format = out_format_lib.ResolvedActivationFormat.create(
      out_format,
      default_dtype=gmm_spec.x.dtype,
      default_quant=out_format_lib.ResolvedLaneStrided(
          block_col_size=out_format_lib.tile_size
      ),
  )
  return gmm_plan.ResolvedGmmPlan.create(
      custom_plan,
      spec=gmm_spec,
      out_format=resolved_out_format,
      tpu_info=_tpu_info() if tpu_info is None else tpu_info,
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


class MatmulDtypeTest(parameterized.TestCase):
  """Checks the matmul dtype the planner picks."""

  def test_fp8_weights_without_supported_fp8_activation_raises(self):
    # TPU v6e has no matmul for float8_e4m3fn, and float8_e4m3b11fnuz cannot
    # hold every float8_e4m3fn value.
    tpu_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
    with self.assertRaisesRegex(
        NotImplementedError, "No supported fp8 activation dtype"
    ):
      _resolve(8, tpu_info=tpu_info, w_dtype=jnp.dtype(jnp.float8_e4m3fn))


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


class QuantStashPlanTest(parameterized.TestCase):

  def test_unquantized_out_resolves_stash_fields_to_none(self):
    resolved = _resolve(8)
    self.assertIsNone(resolved.num_stash_buffers)
    self.assertIsNone(resolved.stash_dtype)

  def test_quantized_out_resolves_default_stash_fields(self):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn, quant=out_format_lib.auto_quant
    )
    resolved = _resolve(8, out_format=fmt, size_n=1024)
    self.assertEqual(resolved.num_stash_buffers, 2)
    self.assertEqual(resolved.stash_dtype, jnp.dtype(jnp.bfloat16))

  def test_explicit_supported_stash_fields_are_preserved(self):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn, quant=out_format_lib.auto_quant
    )
    custom_plan = gmm_plan.RoutingGmmPlan(
        num_stash_buffers=2, stash_dtype=jnp.bfloat16
    )
    resolved = _resolve(8, custom_plan, out_format=fmt, size_n=1024)
    self.assertEqual(resolved.num_stash_buffers, 2)
    self.assertEqual(resolved.stash_dtype, jnp.dtype(jnp.bfloat16))

  @parameterized.named_parameters(
      dict(
          testcase_name="extra_stash_buffers",
          extra_plan=gmm_plan.RoutingGmmPlan(num_stash_buffers=3),
      ),
      dict(
          testcase_name="wider_stash_dtype",
          extra_plan=gmm_plan.RoutingGmmPlan(stash_dtype=jnp.float32),
      ),
  )
  def test_tiling_and_fusion_vmem_estimates_account_for_stash(
      self, extra_plan: gmm_plan.RoutingGmmPlan
  ):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn, quant=out_format_lib.auto_quant
    )
    # 13.5 MB fits tile_m=256 with 2 bf16 stash buffers (13,377,536 bytes), but
    # overflows to tile_m=128 with 3 buffers or f32 stash dtype.
    single_budget = dict(vmem_limit_bytes=13_500_000, vmem_usage_fraction=1.0)
    default_single = _resolve(
        256,
        gmm_plan.RoutingGmmPlan(**single_budget),
        out_format=fmt,
        size_n=1024,
    )
    larger_stash_single = _resolve(
        256,
        dataclasses.replace(extra_plan, **single_budget),
        out_format=fmt,
        size_n=1024,
    )
    self.assertEqual(default_single.tile_m, 256)
    self.assertEqual(larger_stash_single.tile_m, 128)

    # 13.55 MB fits fused up/down with 2 bf16 stash buffers (13,459,456 bytes),
    # but unfuses with 3 buffers or f32 stash dtype.
    fused_budget = dict(vmem_limit_bytes=13_550_000, vmem_usage_fraction=1.0)
    default_two_stage = _resolve(
        8,
        gmm_plan.RoutingGmmPlan(**fused_budget),
        with_down=True,
        out_format=fmt,
    )
    larger_stash_two_stage = _resolve(
        8,
        dataclasses.replace(extra_plan, **fused_budget),
        with_down=True,
        out_format=fmt,
    )
    self.assertTrue(default_two_stage.fuse_up_and_down)
    self.assertFalse(larger_stash_two_stage.fuse_up_and_down)


if __name__ == "__main__":
  absltest.main()
