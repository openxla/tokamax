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
"""Unit tests for RoutingGmmConfig."""

import dataclasses
from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src import block_scaled
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.common import spec
from tokamax._src.ops.experimental.routing_moe.gmm import config as gmm_config
from tokamax._src.ops.experimental.routing_moe.gmm import memory_ref
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.gmm import spec as gmm_spec_lib


def _tpu_info() -> pltpu.TpuInfo:
  return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _make_config(
    num_tokens: int,
    custom_plan: gmm_plan.RoutingGmmPlan | None = None,
    *,
    out_format: out_format_lib.ActivationFormat | None = None,
    size_ep: int = 1,
    size_n: int = 1280,
    quantize_w: bool = True,
    w_q_block_size: int | None = None,
    fuse_act: str | None = None,
    with_w2: bool = False,
    routed: bool = True,
    top_k: int = 10,
) -> gmm_config.RoutingGmmConfig:
  """Builds a RoutingGmmConfig for a 512-expert, top-`top_k` MoE layer."""
  num_experts, ep_size, size_k = 512, 4, 2560
  w_q_block_size = (
      w_q_block_size
      if w_q_block_size is not None
      else size_k
  )
  x = spec.ActivationSpec(
      row_size=num_tokens,
      col_size=size_k,
      dtype=jnp.dtype(jnp.bfloat16),
      quant=None,
  )
  w = spec.WeightSpec(
      group_size=num_experts // ep_size,
      in_size=size_k,
      out_size=size_n,
      dtype=jnp.dtype(jnp.float8_e4m3fn if quantize_w else jnp.bfloat16),
      quant=spec.SubchannelQuantSpec(block_size=w_q_block_size)
      if quantize_w
      else None,
      has_bias=False,
      fused_gate_up=fuse_act is not None,
  )
  w2 = None
  if with_w2:
    assert fuse_act is not None
    w2 = spec.WeightSpec(
        group_size=num_experts // ep_size,
        in_size=size_n // 2,
        out_size=size_k,
        dtype=jnp.dtype(jnp.float8_e4m3fn if quantize_w else jnp.bfloat16),
        quant=spec.SubchannelQuantSpec(block_size=w_q_block_size)
        if quantize_w
        else None,
        has_bias=False,
        fused_gate_up=False,
    )
  routed_rows = num_tokens * top_k
  gather = scatter = None
  if routed:
    gather = spec.RoutingSpec(num_indices=routed_rows, remote_axis=None)
    scatter = spec.RoutingSpec(
        num_indices=routed_rows,
        remote_axis=None if size_ep == 1 else "ep",
    )
  resolved_out_format = out_format_lib.ResolvedActivationFormat.create(
      out_format,
      default_dtype=x.dtype,
      default_quant=out_format_lib.ResolvedLaneStrided(
          block_col_size=out_format_lib.tile_size
      ),
  )
  gmm_spec = gmm_spec_lib.RoutingGmmSpec(
      x=x,
      w=w,
      w2=w2,
      gather=gather,
      scatter=scatter,
      num_global_groups=num_experts,
      fuse_act=fuse_act,
      size_ep=size_ep,
      size_row_group=routed_rows,
  )
  tpu_info = _tpu_info()
  resolved_plan = gmm_plan.ResolvedGmmPlan.create(
      custom_plan,
      spec=gmm_spec,
      out_format=resolved_out_format,
      tpu_info=tpu_info,
  )
  return gmm_config.RoutingGmmConfig.create(
      spec=gmm_spec,
      out_format=resolved_out_format,
      plan=resolved_plan,
      tpu_info=tpu_info,
  )


class OutSpecAndValidationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="unquantized",
          quantize_w=False,
          custom_plan=None,
          expected=gmm_config.MatmulMode.UNQUANTIZED,
      ),
      dict(
          testcase_name="quantized",
          quantize_w=True,
          custom_plan=None,
          expected=gmm_config.MatmulMode.QUANTIZED,
      ),
      dict(
          testcase_name="weight_only",
          quantize_w=True,
          custom_plan=gmm_plan.RoutingGmmPlan(matmul_dtype=jnp.bfloat16),
          expected=gmm_config.MatmulMode.WEIGHT_ONLY_SCALE_AFTER,
      ),
      dict(
          testcase_name="small_w_block",
          quantize_w=True,
          custom_plan=None,
          w_q_block_size=128,
          expected=gmm_config.MatmulMode.WEIGHT_ONLY_SCALE_BEFORE,
      ),
      dict(
          testcase_name="small_w_block_quantized_x",
          quantize_w=True,
          custom_plan=gmm_plan.RoutingGmmPlan(matmul_dtype=jnp.float8_e4m3fn),
          w_q_block_size=128,
          expected=gmm_config.MatmulMode.QUANTIZED,
      ),
  )
  def test_matmul_mode(
      self,
      quantize_w: bool,
      custom_plan: gmm_plan.RoutingGmmPlan | None,
      expected: gmm_config.MatmulMode,
      w_q_block_size: int | None = None,
  ):
    cfg = _make_config(
        8, custom_plan, quantize_w=quantize_w, w_q_block_size=w_q_block_size
    )
    self.assertEqual(cfg.compute.matmul_mode, expected)

  def test_default_out_format_uses_input_dtype_and_unquantized(self):
    out_spec = _make_config(8).out.spec
    self.assertEqual(
        out_spec,
        spec.ActivationSpec(
            row_size=80,
            col_size=1280,
            dtype=jnp.dtype(jnp.bfloat16),
            quant=None,
        ),
    )

  def test_remote_scatter_divides_out_rows_by_size_ep(self):
    out_spec = _make_config(8, size_ep=4).out.spec
    self.assertEqual(out_spec.row_size, 20)
    self.assertEqual(out_spec.col_size, 1280)

  @parameterized.parameters(1024, 1536)
  def test_tile_size_quant_format_resolves_to_tile_n(self, size_n: int):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn, quant=out_format_lib.auto_quant
    )
    out_spec = _make_config(8, out_format=fmt, size_n=size_n).out.spec
    self.assertEqual(out_spec.dtype, jnp.dtype(jnp.float8_e4m3fn))
    self.assertEqual(
        out_spec.quant,
        spec.LaneStridedQuantSpec(block_col_size=size_n),
    )

  def test_explicit_quant_block_col_size_matching_tile_n_is_preserved(self):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn,
        quant=out_format_lib.LaneStrided(block_col_size=1024),
    )
    out_spec = _make_config(8, out_format=fmt, size_n=1024).out.spec
    self.assertEqual(
        out_spec.quant,
        spec.LaneStridedQuantSpec(block_col_size=1024),
    )

  def test_explicit_quant_block_col_size_mismatching_tile_n_raises(self):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn,
        quant=out_format_lib.LaneStrided(block_col_size=512),
    )
    with self.assertRaisesRegex(
        ValueError, "Output quantization block_col_size"
    ):
      _make_config(8, out_format=fmt, size_n=1024)

  def test_subchannel_out_quant_on_gmm_raises_not_implemented(self):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn,
        quant=out_format_lib.Subchannel(block_size=512),
    )
    with self.assertRaisesRegex(NotImplementedError, "Only LaneStrided"):
      _make_config(8, out_format=fmt, size_n=1024)

  @parameterized.named_parameters(
      dict(
          testcase_name="num_stash_buffers",
          custom_plan=gmm_plan.RoutingGmmPlan(num_stash_buffers=2),
      ),
      dict(
          testcase_name="stash_dtype",
          custom_plan=gmm_plan.RoutingGmmPlan(stash_dtype=jnp.bfloat16),
      ),
  )
  def test_stash_fields_on_unquantized_out_raise_value_error(
      self, custom_plan: gmm_plan.RoutingGmmPlan
  ):
    with self.assertRaisesRegex(
        ValueError, "has no effect when output is not quantized"
    ):
      _make_config(8, custom_plan)

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_buffers",
          custom_plan=gmm_plan.RoutingGmmPlan(num_stash_buffers=0),
          expected_regex="num_stash_buffers other than 2",
      ),
      dict(
          testcase_name="one_buffer",
          custom_plan=gmm_plan.RoutingGmmPlan(num_stash_buffers=1),
          expected_regex="num_stash_buffers other than 2",
      ),
      dict(
          testcase_name="three_buffers",
          custom_plan=gmm_plan.RoutingGmmPlan(num_stash_buffers=3),
          expected_regex="num_stash_buffers other than 2",
      ),
      dict(
          testcase_name="float32_stash_dtype",
          custom_plan=gmm_plan.RoutingGmmPlan(stash_dtype=jnp.float32),
          expected_regex="stash_dtype other than bfloat16",
      ),
  )
  def test_unsupported_stash_fields_on_quantized_out_raise_not_implemented(
      self, custom_plan: gmm_plan.RoutingGmmPlan, expected_regex: str
  ):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn, quant=out_format_lib.auto_quant
    )
    with self.assertRaisesRegex(NotImplementedError, expected_regex):
      _make_config(8, custom_plan, out_format=fmt, size_n=1024)

  @parameterized.named_parameters(
      dict(
          testcase_name="untiled_fused",
          fuse_act="silu",
          tile_n=None,
          expected=True,
      ),
      dict(
          testcase_name="unfused",
          fuse_act=None,
          tile_n=None,
          expected=False,
      ),
      dict(
          testcase_name="tiled_fused",
          fuse_act="silu",
          tile_n=256,
          expected=False,
      ),
      dict(
          testcase_name="untiled_fused_w2",
          fuse_act="silu",
          with_w2=True,
          tile_n=None,
          expected=True,
      ),
      dict(
          testcase_name="tiled_k2_fused_w2",
          fuse_act="silu",
          with_w2=True,
          tile_n=None,
          tile_k2=512,
          expected=False,
      ),
  )
  def test_contiguous_fused_w(
      self,
      fuse_act: str | None,
      tile_n: int | None,
      expected: bool,
      with_w2: bool = False,
      tile_k2: int | None = None,
  ):
    cfg = _make_config(
        8,
        gmm_plan.RoutingGmmPlan(
            tile_n=tile_n,
            tile_k2=tile_k2,
            fuse_up_and_down=True if with_w2 else None,
        ),
        size_n=2048 if with_w2 else 1536,
        w_q_block_size=512 if with_w2 else None,
        fuse_act=fuse_act,
        with_w2=with_w2,
    )
    self.assertEqual(cfg.contiguous_fused_w, expected)

  def _block_scaled_fixture(self):
    if not block_scaled.is_available():
      self.skipTest("Block-scaled matmul is not available in this build.")
    chip_8t = getattr(pltpu.ChipVersion, "TPU_8T", None)
    if chip_8t is None:
      self.skipTest("TPU_8T is not available in this JAX version.")
    x = spec.ActivationSpec(
        row_size=16,
        col_size=1024,
        dtype=jnp.dtype(jnp.bfloat16),
        quant=None,
    )
    w = spec.WeightSpec(
        group_size=8,
        in_size=1024,
        out_size=1024,
        dtype=jnp.dtype(jnp.float8_e5m2),
        quant=spec.BlockScaledQuantSpec(scale_rows=64),
        has_bias=False,
        fused_gate_up=False,
    )
    gmm_spec = gmm_spec_lib.RoutingGmmSpec(
        x=x,
        w=w,
        w2=None,
        gather=None,
        scatter=None,
        num_global_groups=8,
        fuse_act=None,
        size_ep=1,
        size_row_group=0,
    )
    resolved_fmt = out_format_lib.ResolvedActivationFormat.create(
        None,
        default_dtype=x.dtype,
        default_quant=out_format_lib.ResolvedLaneStrided(
            block_col_size=out_format_lib.tile_size
        ),
    )
    tpu_8t = pltpu.get_tpu_info_for_chip(chip_8t, 1)
    plan_8t = gmm_plan.ResolvedGmmPlan.create(
        None, spec=gmm_spec, out_format=resolved_fmt, tpu_info=tpu_8t
    )
    return gmm_spec, resolved_fmt, plan_8t, tpu_8t

  def test_block_scaled_config_resolves_w_quant(self):
    gmm_spec, resolved_fmt, plan_8t, tpu_8t = self._block_scaled_fixture()
    cfg = gmm_config.RoutingGmmConfig.create(
        spec=gmm_spec, out_format=resolved_fmt, plan=plan_8t, tpu_info=tpu_8t
    )
    self.assertEqual(
        cfg.compute.matmul_mode, gmm_config.MatmulMode.BLOCK_SCALED
    )
    self.assertIsNone(cfg.compute.dynamic_quant)
    self.assertEqual(
        cfg.compute.w_quant, spec.BlockScaledQuantSpec(scale_rows=64)
    )
    self.assertIsNone(cfg.compute2)
    self.assertEqual(cfg.compute.matmul_dtype, jnp.dtype(jnp.float8_e5m2))
    self.assertIsNone(cfg.plan.x_quant_block_size)

    w1 = dataclasses.replace(gmm_spec.w, fused_gate_up=True)
    w2 = spec.WeightSpec(
        group_size=8,
        in_size=512,
        out_size=1024,
        dtype=jnp.dtype(jnp.float8_e5m2),
        quant=spec.BlockScaledQuantSpec(scale_rows=32),
        has_bias=False,
        fused_gate_up=False,
    )
    up_down_spec = dataclasses.replace(gmm_spec, w=w1, w2=w2, fuse_act="silu")
    up_down_plan = gmm_plan.ResolvedGmmPlan.create(
        None, spec=up_down_spec, out_format=resolved_fmt, tpu_info=tpu_8t
    )
    up_down_cfg = gmm_config.RoutingGmmConfig.create(
        spec=up_down_spec,
        out_format=resolved_fmt,
        plan=up_down_plan,
        tpu_info=tpu_8t,
    )
    self.assertEqual(
        up_down_cfg.compute.matmul_mode, gmm_config.MatmulMode.BLOCK_SCALED
    )
    self.assertIsNone(up_down_cfg.compute.dynamic_quant)
    self.assertEqual(
        up_down_cfg.compute.w_quant,
        spec.BlockScaledQuantSpec(scale_rows=64),
    )
    assert up_down_cfg.compute2 is not None
    self.assertEqual(
        up_down_cfg.compute2.matmul_mode, gmm_config.MatmulMode.BLOCK_SCALED
    )
    self.assertIsNone(up_down_cfg.compute2.dynamic_quant)
    self.assertEqual(
        up_down_cfg.compute2.w_quant,
        spec.BlockScaledQuantSpec(scale_rows=32),
    )

  @parameterized.named_parameters(
      dict(testcase_name="tpu_7x", bad_chip=pltpu.ChipVersion.TPU_7X),
      dict(testcase_name="tpu_8i", bad_chip=pltpu.ChipVersion.TPU_8I),
  )
  def test_block_scaled_config_rejects_unsupported_chip(
      self, bad_chip: pltpu.ChipVersion
  ):
    gmm_spec, resolved_fmt, plan_8t, _ = self._block_scaled_fixture()
    with self.assertRaisesRegex(NotImplementedError, "requires TPU_8T"):
      gmm_config.RoutingGmmConfig.create(
          spec=gmm_spec,
          out_format=resolved_fmt,
          plan=plan_8t,
          tpu_info=pltpu.get_tpu_info_for_chip(bad_chip, 1),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="x_quant_block_size_set",
          custom_plan=gmm_plan.RoutingGmmPlan(x_quant_block_size=512),
          exc=ValueError,
          err="not supported with BlockScaledQuantSpec",
      ),
      dict(
          testcase_name="invalid_matmul_dtype",
          custom_plan=gmm_plan.RoutingGmmPlan(matmul_dtype=jnp.bfloat16),
          exc=ValueError,
          err="Block-scaled matmul_dtype must be in",
      ),
  )
  def test_block_scaled_config_rejects_invalid_plan_options(
      self, custom_plan: gmm_plan.RoutingGmmPlan, exc: type[Exception], err: str
  ):
    gmm_spec, resolved_fmt, _, tpu_8t = self._block_scaled_fixture()
    bad_plan = gmm_plan.ResolvedGmmPlan.create(
        custom_plan, spec=gmm_spec, out_format=resolved_fmt, tpu_info=tpu_8t
    )
    with self.assertRaisesRegex(exc, err):
      gmm_config.RoutingGmmConfig.create(
          spec=gmm_spec, out_format=resolved_fmt, plan=bad_plan, tpu_info=tpu_8t
      )

  def test_block_scaled_config_rejects_non_bf16_x(self):
    gmm_spec, _, plan_8t, tpu_8t = self._block_scaled_fixture()
    f32_x = dataclasses.replace(gmm_spec.x, dtype=jnp.dtype(jnp.float32))
    f32_spec = dataclasses.replace(gmm_spec, x=f32_x)
    f32_fmt = out_format_lib.ResolvedActivationFormat.create(
        None,
        default_dtype=f32_x.dtype,
        default_quant=out_format_lib.ResolvedLaneStrided(
            block_col_size=out_format_lib.tile_size
        ),
    )
    with self.assertRaisesRegex(NotImplementedError, "requires bfloat16 x"):
      gmm_config.RoutingGmmConfig.create(
          spec=f32_spec, out_format=f32_fmt, plan=plan_8t, tpu_info=tpu_8t
      )

  @parameterized.named_parameters(
      dict(testcase_name="w2_unquantized", w2_quant=None),
      dict(
          testcase_name="w2_subchannel",
          w2_quant=spec.SubchannelQuantSpec(block_size=512),
      ),
  )
  def test_block_scaled_config_rejects_mismatched_w2_quant(
      self, w2_quant: spec.WeightQuantSpec | None
  ):
    gmm_spec, resolved_fmt, _, tpu_8t = self._block_scaled_fixture()
    w1 = dataclasses.replace(gmm_spec.w, fused_gate_up=True)
    w2 = spec.WeightSpec(
        group_size=8,
        in_size=512,
        out_size=1024,
        dtype=jnp.dtype(jnp.float8_e5m2),
        quant=w2_quant,
        has_bias=False,
        fused_gate_up=False,
    )
    mismatch_spec = dataclasses.replace(gmm_spec, w=w1, w2=w2, fuse_act="silu")
    mismatch_plan = gmm_plan.ResolvedGmmPlan.create(
        None,
        spec=mismatch_spec,
        out_format=resolved_fmt,
        tpu_info=tpu_8t,
    )
    with self.assertRaisesRegex(
        NotImplementedError, "w2 must match the precision of w"
    ):
      gmm_config.RoutingGmmConfig.create(
          spec=mismatch_spec,
          out_format=resolved_fmt,
          plan=mismatch_plan,
          tpu_info=tpu_8t,
      )

  def test_block_scaled_config_rejects_unaligned_tile_k_and_tile_k2(self):
    gmm_spec, resolved_fmt, plan_8t, tpu_8t = self._block_scaled_fixture()
    unaligned_k_spec = dataclasses.replace(
        gmm_spec,
        x=dataclasses.replace(gmm_spec.x, col_size=128),
        w=dataclasses.replace(gmm_spec.w, in_size=128),
    )
    unaligned_k_plan = dataclasses.replace(plan_8t, tile_k=128)
    with self.assertRaisesRegex(ValueError, "tile_k .* compact chunk width"):
      gmm_config.RoutingGmmConfig.create(
          spec=unaligned_k_spec,
          out_format=resolved_fmt,
          plan=unaligned_k_plan,
          tpu_info=tpu_8t,
      )

    w1 = dataclasses.replace(gmm_spec.w, out_size=384, fused_gate_up=True)
    w2 = spec.WeightSpec(
        group_size=8,
        in_size=192,
        out_size=1024,
        dtype=jnp.dtype(jnp.float8_e5m2),
        quant=spec.BlockScaledQuantSpec(scale_rows=64),
        has_bias=False,
        fused_gate_up=False,
    )
    unaligned_k2_spec = dataclasses.replace(
        gmm_spec, w=w1, w2=w2, fuse_act="silu"
    )
    unaligned_k2_plan = gmm_plan.ResolvedGmmPlan.create(
        None, spec=unaligned_k2_spec, out_format=resolved_fmt, tpu_info=tpu_8t
    )
    with self.assertRaisesRegex(ValueError, "tile_k2 .* multiple of 128"):
      gmm_config.RoutingGmmConfig.create(
          spec=unaligned_k2_spec,
          out_format=resolved_fmt,
          plan=unaligned_k2_plan,
          tpu_info=tpu_8t,
      )

  def test_block_scaled_config_rejects_invalid_scale_rows(self):
    gmm_spec, resolved_fmt, plan_8t, tpu_8t = self._block_scaled_fixture()
    bad_w_spec = dataclasses.replace(
        gmm_spec,
        x=dataclasses.replace(gmm_spec.x, col_size=1536),
        w=dataclasses.replace(
            gmm_spec.w,
            in_size=1536,
            quant=spec.BlockScaledQuantSpec(scale_rows=64),
        ),
    )
    bad_w_plan = gmm_plan.ResolvedGmmPlan.create(
        None, spec=bad_w_spec, out_format=resolved_fmt, tpu_info=tpu_8t
    )
    with self.assertRaisesRegex(ValueError, "scale_rows must be 96 or 128"):
      gmm_config.RoutingGmmConfig.create(
          spec=bad_w_spec,
          out_format=resolved_fmt,
          plan=bad_w_plan,
          tpu_info=tpu_8t,
      )

    w1 = dataclasses.replace(gmm_spec.w, out_size=3072, fused_gate_up=True)
    w2 = spec.WeightSpec(
        group_size=8,
        in_size=1536,
        out_size=1024,
        dtype=jnp.dtype(jnp.float8_e5m2),
        quant=spec.BlockScaledQuantSpec(scale_rows=64),
        has_bias=False,
        fused_gate_up=False,
    )
    bad_w2_spec = dataclasses.replace(
        gmm_spec, w=w1, w2=w2, fuse_act="silu"
    )
    bad_w2_plan = gmm_plan.ResolvedGmmPlan.create(
        None, spec=bad_w2_spec, out_format=resolved_fmt, tpu_info=tpu_8t
    )
    with self.assertRaisesRegex(ValueError, "scale_rows must be 96 or 128"):
      gmm_config.RoutingGmmConfig.create(
          spec=bad_w2_spec,
          out_format=resolved_fmt,
          plan=bad_w2_plan,
          tpu_info=tpu_8t,
      )

    bad_k_spec = dataclasses.replace(
        gmm_spec,
        x=dataclasses.replace(gmm_spec.x, col_size=8),
        w=dataclasses.replace(gmm_spec.w, in_size=8),
    )
    bad_k_plan = dataclasses.replace(plan_8t, tile_k=8)
    with self.assertRaisesRegex(ValueError, "contracting size"):
      gmm_config.RoutingGmmConfig.create(
          spec=bad_k_spec,
          out_format=resolved_fmt,
          plan=bad_k_plan,
          tpu_info=tpu_8t,
      )

  def test_subchannel_config_rejects_missing_x_quant_block_size(self):
    cfg = _make_config(8)
    bad_plan = dataclasses.replace(cfg.plan, x_quant_block_size=None)
    resolved_fmt = out_format_lib.ResolvedActivationFormat.create(
        None,
        default_dtype=cfg.spec.x.dtype,
        default_quant=out_format_lib.ResolvedLaneStrided(
            block_col_size=out_format_lib.tile_size
        ),
    )
    with self.assertRaisesRegex(
        ValueError, "x_quant_block_size is required for subchannel"
    ):
      gmm_config.RoutingGmmConfig.create(
          spec=cfg.spec,
          out_format=resolved_fmt,
          plan=bad_plan,
          tpu_info=_tpu_info(),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="without_routing",
          num_tokens=8,
          routed=False,
          top_k=10,
          expected_regex="smem_indices=True requires gather or scatter",
      ),
      dict(
          testcase_name="zero_rows",
          num_tokens=8,
          routed=True,
          top_k=0,
          expected_regex="size_out_m > 0",
      ),
      dict(
          testcase_name="smem_overflow",
          num_tokens=65536,
          routed=True,
          top_k=10,
          expected_regex="SMEM allocation",
      ),
  )
  def test_invalid_smem_indices_raises_value_error(
      self,
      num_tokens: int,
      routed: bool,
      top_k: int,
      expected_regex: str,
  ):
    custom_plan = gmm_plan.RoutingGmmPlan(smem_indices=True)
    with self.assertRaisesRegex(ValueError, expected_regex):
      _make_config(num_tokens, custom_plan, routed=routed, top_k=top_k)

  @parameterized.parameters(
      (512, 32),
      (512, 64),
      (1536, 96),
      (1536, 128),
  )
  def test_block_scaled_scale_block_spec_uses_3d_shape(
      self, in_size: int, scale_rows: int
  ):
    gmm_spec, resolved_fmt, _, tpu_8t = self._block_scaled_fixture()
    w_spec = dataclasses.replace(
        gmm_spec.w,
        in_size=in_size,
        quant=spec.BlockScaledQuantSpec(scale_rows=scale_rows),
    )
    x_spec = dataclasses.replace(gmm_spec.x, col_size=in_size)
    cur_spec = dataclasses.replace(gmm_spec, x=x_spec, w=w_spec)
    cur_plan = gmm_plan.ResolvedGmmPlan.create(
        None, spec=cur_spec, out_format=resolved_fmt, tpu_info=tpu_8t
    )
    cfg = gmm_config.RoutingGmmConfig.create(
        spec=cur_spec, out_format=resolved_fmt, plan=cur_plan, tpu_info=tpu_8t
    )
    index_maps = memory_ref.PipelineIndexMaps(cast(Any, None), cfg)
    block_specs = memory_ref.PipelineBlockSpecs(cfg, index_maps)
    scale_spec = block_specs.get_scale_spec(w_spec, tile_k=in_size, tile_n=256)
    weight_spec = block_specs.get_weight_spec(tile_k=in_size, tile_n=256)
    self.assertEqual(scale_spec.block_shape, (None, scale_rows, 256))
    self.assertEqual(scale_spec.index_map, weight_spec.index_map)

  def test_block_scaled_scale_block_spec_rejects_unsupported_tile_k(self):
    gmm_spec, resolved_fmt, _, tpu_8t = self._block_scaled_fixture()
    w_spec = dataclasses.replace(
        gmm_spec.w,
        in_size=1536,
        quant=spec.BlockScaledQuantSpec(scale_rows=96),
    )
    x_spec = dataclasses.replace(gmm_spec.x, col_size=1536)
    cur_spec = dataclasses.replace(gmm_spec, x=x_spec, w=w_spec)
    cur_plan = gmm_plan.ResolvedGmmPlan.create(
        None, spec=cur_spec, out_format=resolved_fmt, tpu_info=tpu_8t
    )
    cfg = gmm_config.RoutingGmmConfig.create(
        spec=cur_spec, out_format=resolved_fmt, plan=cur_plan, tpu_info=tpu_8t
    )
    index_maps = memory_ref.PipelineIndexMaps(cast(Any, None), cfg)
    block_specs = memory_ref.PipelineBlockSpecs(cfg, index_maps)
    with self.assertRaises(ValueError):
      block_specs.get_scale_spec(w_spec, tile_k=1, tile_n=256)
    with self.assertRaisesRegex(NotImplementedError, "sub-K tiling"):
      block_specs.get_scale_spec(w_spec, tile_k=1024, tile_n=256)


class IndexWindowTest(parameterized.TestCase):

  def test_smem_indices_window_is_whole_table(self):
    custom_plan = gmm_plan.RoutingGmmPlan(smem_indices=True)
    cfg = _make_config(8, custom_plan)
    # 8 tokens * top-10 = 80 routed rows in one unbuffered window.
    self.assertEqual(
        cfg.idx_window,
        gmm_config.IndexWindow(num_buffers=1, buffer_size=0, ring_size=80),
    )

  def test_streamed_indices_window_uses_ring_buffers(self):
    custom_plan = gmm_plan.RoutingGmmPlan(smem_indices=False)
    cfg = _make_config(8, custom_plan)
    self.assertEqual(
        cfg.idx_window,
        gmm_config.IndexWindow(
            num_buffers=2,
            buffer_size=cfg.plan.idx_buffer_size,
            ring_size=cfg.plan.idx_ring_size,
        ),
    )

if __name__ == "__main__":
  absltest.main()
