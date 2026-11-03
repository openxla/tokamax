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
from tokamax._src.ops.experimental.routing_moe.common import dataclass_util
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.common import spec
from tokamax._src.ops.experimental.routing_moe.gmm import config as gmm_config
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.gmm import spec as gmm_spec_lib
from tokamax._src.ops.experimental.routing_moe.layer import plan as moe_plan


def _tpu_info() -> pltpu.TpuInfo:
  return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _make_config(
    num_tokens: int,
    custom_plan: gmm_plan.RoutingGmmPlan | None = None,
    *,
    quantized: bool = True,
    with_down: bool = False,
    out_format: out_format_lib.ActivationFormat | None = None,
    size_ep: int = 1,
    size_n: int = 1280,
    tpu_info: pltpu.TpuInfo | None = None,
    w_dtype: jnp.dtype | None = None,
) -> gmm_config.RoutingGmmConfig:
  """Builds a config for a 512-expert, top-10 MoE layer."""
  if tpu_info is None:
    tpu_info = _tpu_info()
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
      fuse_act="silu" if with_down else None,
      size_ep=size_ep,
  )
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


def _resolve(
    num_tokens: int,
    custom_plan: gmm_plan.RoutingGmmPlan | None = None,
    *,
    quantized: bool = True,
    with_down: bool = False,
) -> gmm_plan.ResolvedGmmPlan:
  return _make_config(
      num_tokens,
      custom_plan,
      quantized=quantized,
      with_down=with_down,
  ).plan


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
      _make_config(8, tpu_info=tpu_info, w_dtype=jnp.dtype(jnp.float8_e4m3fn))


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


class DataclassUtilTest(parameterized.TestCase):

  def test_require_none_default_rejects_missing_or_non_none_default(self):
    with self.assertRaisesRegex(ValueError, "must default to None"):

      @dataclass_util.require_none_default
      @dataclasses.dataclass(frozen=True, kw_only=True)
      class _BadUnresolved:  # pylint: disable=unused-variable
        x: int

  def test_require_no_default_rejects_inherited_default(self):
    @dataclass_util.require_none_default
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Base:
      x: int | None = None

    with self.assertRaisesRegex(ValueError, "must not have a default value"):

      @dataclass_util.require_no_default
      @dataclasses.dataclass(frozen=True, kw_only=True)
      class _BadResolved(_Base):  # pylint: disable=unused-variable
        x: int

  def test_resolves_from_accepts_matching_order_and_trailing_dropped(self):
    @dataclass_util.require_none_default
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Unresolved:
      a: int | None = None
      b: str | None = None
      hint: float | None = None

    @dataclass_util.resolves_from(_Unresolved, dropped=("hint",))
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Resolved:
      a: int
      b: str

    self.assertEqual(_Resolved(a=1, b="ok"), _Resolved(a=1, b="ok"))

  def test_resolves_from_rejects_reordered_or_mismatched_fields(self):
    @dataclass_util.require_none_default
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Unresolved:
      a: int | None = None
      b: str | None = None

    with self.assertRaisesRegex(ValueError, "must match _Unresolved fields"):

      @dataclass_util.resolves_from(_Unresolved)
      @dataclasses.dataclass(frozen=True, kw_only=True)
      class _Reordered:  # pylint: disable=unused-variable
        b: str
        a: int

    with self.assertRaisesRegex(ValueError, "must match _Unresolved fields"):

      @dataclass_util.resolves_from(_Unresolved)
      @dataclasses.dataclass(frozen=True, kw_only=True)
      class _MissingField:  # pylint: disable=unused-variable
        a: int

  def test_resolves_from_rejects_unknown_or_non_trailing_dropped(self):
    @dataclass_util.require_none_default
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Unresolved:
      hint: float | None = None
      a: int | None = None

    with self.assertRaisesRegex(ValueError, "Unknown dropped fields"):
      dataclass_util.resolves_from(_Unresolved, dropped=("nonexistent",))

    with self.assertRaisesRegex(ValueError, "must appear at the end"):
      dataclass_util.resolves_from(_Unresolved, dropped=("hint",))

  def test_resolves_from_enforces_default_invariants_on_both_classes(self):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _BadUnresolved:
      a: int = 1

    with self.assertRaisesRegex(ValueError, "must default to None"):
      dataclass_util.resolves_from(_BadUnresolved)

    @dataclass_util.require_none_default
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _GoodUnresolved:
      a: int | None = None

    with self.assertRaisesRegex(ValueError, "must not have a default value"):

      @dataclass_util.resolves_from(_GoodUnresolved)
      @dataclasses.dataclass(frozen=True, kw_only=True)
      class _BadResolved:  # pylint: disable=unused-variable
        a: int = 1


class FormatAndOutSpecResolutionTest(parameterized.TestCase):

  @parameterized.parameters(None, out_format_lib.ActivationFormat())
  def test_empty_or_none_activation_format_resolves_to_default(
      self, fmt: out_format_lib.ActivationFormat | None
  ):
    resolved = out_format_lib.ResolvedActivationFormat.create(
        fmt,
        default_dtype=jnp.bfloat16,
    )
    self.assertNotIsInstance(resolved, out_format_lib.ActivationFormat)
    self.assertEqual(
        resolved,
        out_format_lib.ResolvedActivationFormat(
            dtype=jnp.dtype(jnp.bfloat16), quant=None
        ),
    )

  def test_quant_without_dtype_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, "dtype must be specified"):
      out_format_lib.ActivationFormat(quant=out_format_lib.auto_quant)
    with self.assertRaisesRegex(ValueError, "dtype must be specified"):
      out_format_lib.ActivationFormat(quant=out_format_lib.LaneStrided())

  def test_quant_without_default_quant_raises_not_implemented(self):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn, quant=out_format_lib.auto_quant
    )
    with self.assertRaisesRegex(
        NotImplementedError, "Quantization is not supported"
    ):
      out_format_lib.ResolvedActivationFormat.create(
          fmt, default_dtype=jnp.bfloat16
      )

  @parameterized.parameters(
      out_format_lib.auto_quant,
      out_format_lib.LaneStrided(),
  )
  def test_resolved_activation_format_populates_quant_and_passes_through(
      self, quant_arg
  ):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn, quant=quant_arg
    )
    resolved_quant = out_format_lib.ResolvedActivationFormat.create(
        fmt,
        default_dtype=jnp.bfloat16,
        default_quant=out_format_lib.ResolvedLaneStrided(
            block_col_size=out_format_lib.tile_size
        ),
    )
    self.assertNotIsInstance(resolved_quant, out_format_lib.ActivationFormat)
    self.assertNotIsInstance(resolved_quant.quant, out_format_lib.LaneStrided)
    self.assertEqual(
        resolved_quant,
        out_format_lib.ResolvedActivationFormat(
            dtype=jnp.dtype(jnp.float8_e4m3fn),
            quant=out_format_lib.ResolvedLaneStrided(
                block_col_size=out_format_lib.tile_size
            ),
        ),
    )

    passthrough = out_format_lib.ResolvedActivationFormat.create(
        resolved_quant,
        default_dtype=jnp.float32,
    )
    self.assertIs(passthrough, resolved_quant)

  def test_cross_scheme_unset_field_raises_value_error(self):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn,
        quant=out_format_lib.Subchannel(),
    )
    with self.assertRaisesRegex(ValueError, "Subchannel.block_size must be"):
      out_format_lib.ResolvedActivationFormat.create(
          fmt,
          default_dtype=jnp.bfloat16,
          default_quant=out_format_lib.ResolvedLaneStrided(
              block_col_size=out_format_lib.tile_size
          ),
      )

  def test_resolved_moe_plan_populates_formats(self):
    resolved = moe_plan.ResolvedMoEPlan.create(
        moe_plan.RoutingMoEPlan(
            gather_format=out_format_lib.ActivationFormat(),
            scatter_format=out_format_lib.ActivationFormat(
                dtype=jnp.float8_e4m3fn, quant=out_format_lib.auto_quant
            ),
        ),
        x_dtype=jnp.bfloat16,
    )
    self.assertEqual(
        resolved.gather_format,
        out_format_lib.ResolvedActivationFormat(
            dtype=jnp.dtype(jnp.bfloat16), quant=None
        ),
    )
    self.assertEqual(
        resolved.scatter_format,
        out_format_lib.ResolvedActivationFormat(
            dtype=jnp.dtype(jnp.float8_e4m3fn),
            quant=out_format_lib.ResolvedLaneStrided(
                block_col_size=out_format_lib.tile_size
            ),
        ),
    )

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

  @parameterized.parameters(
      (None, None),
      (
          out_format_lib.ResolvedSubchannel(
              block_size=out_format_lib.tile_size
          ),
          spec.SubchannelQuantSpec(block_size=1024),
      ),
      (
          out_format_lib.ResolvedSubchannel(block_size=512),
          spec.SubchannelQuantSpec(block_size=512),
      ),
      (
          out_format_lib.ResolvedLaneStrided(
              block_col_size=out_format_lib.tile_size
          ),
          spec.LaneStridedQuantSpec(block_col_size=1024),
      ),
      (
          out_format_lib.ResolvedLaneStrided(block_col_size=512),
          spec.LaneStridedQuantSpec(block_col_size=512),
      ),
  )
  def test_to_quant_spec_materializes_all_variants(self, resolved, expected):
    self.assertEqual(
        out_format_lib.to_quant_spec(resolved, tile_col_size=1024), expected
    )


class RoutingGmmSpecTest(parameterized.TestCase):

  def test_derived_sizes_single_and_two_stage(self):
    single = _make_config(8, with_down=False).spec
    self.assertEqual(single.size_k, 2560)
    self.assertIsNone(single.size_k2)
    self.assertEqual(single.size_out_m, 80)
    self.assertEqual(single.size_out_n, 1280)
    self.assertEqual(single.size_local_groups, 128)
    self.assertFalse(single.is_fused_act)

    fused = _make_config(8, with_down=True).spec
    self.assertEqual(fused.size_k, 2560)
    self.assertEqual(fused.size_k2, 768)
    self.assertEqual(fused.size_out_m, 80)
    self.assertEqual(fused.size_out_n, 2560)
    self.assertTrue(fused.is_fused_act)

  def test_validation_rejects_mismatched_specs(self):
    base = _make_config(8, with_down=True).spec
    bad_x = dataclasses.replace(base.x, col_size=512)
    with self.assertRaisesRegex(ValueError, "must match w in_size"):
      dataclasses.replace(base, x=bad_x)

    with self.assertRaisesRegex(ValueError, "requires a remote scatter"):
      dataclasses.replace(base, size_ep=2)

    bad_scatter = spec.RoutingSpec(num_indices=79, remote_axis=None)
    with self.assertRaisesRegex(ValueError, "scatter must map every computed"):
      dataclasses.replace(base, scatter=bad_scatter)


if __name__ == "__main__":
  absltest.main()
