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
import functools
import math
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.common import spec
from tokamax._src.ops.experimental.routing_moe.gmm import config as gmm_config
from tokamax._src.ops.experimental.routing_moe.gmm import kernel
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
    size_k2: int = 768,
    size_k: int = 2560,
    num_experts: int = 512,
    ep_size: int = 4,
    top_k: int = 10,
    w_dtype: jnp.dtype | None = None,
    row_group: bool = True,
    with_scatter: bool = True,
    remote_scatter: bool = False,
    has_bias: bool = False,
    num_w_q_blocks: int = 1,
    w_q_block_size: int | None = None,
) -> gmm_spec_lib.RoutingGmmSpec:
  """Builds a RoutingGmmSpec, by default for a 512-expert, top-10 MoE layer."""
  if with_down:
    size_n = 2 * size_k2
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
    quant = None
    if quantized:
      quant = spec.SubchannelQuantSpec(
          block_size=w_q_block_size or in_size // num_w_q_blocks
      )
    return spec.WeightSpec(
        group_size=num_experts // ep_size,
        in_size=in_size,
        out_size=out_size,
        dtype=w_dtype,
        quant=quant,
        has_bias=has_bias,
        fused_gate_up=fused_gate_up,
    )

  # Gate/up projection, and optionally the down projection, which takes the
  # activation of the gate/up output (half its width).
  w = weight_spec(size_k, size_n, fused_gate_up=with_down)
  w2 = weight_spec(size_n // 2, size_k) if with_down else None
  routed_rows = num_tokens * top_k
  scatter_spec = None
  if with_scatter:
    scatter_spec = spec.RoutingSpec(
        num_indices=routed_rows,
        remote_axis="ep" if remote_scatter else None,
    )
  return gmm_spec_lib.RoutingGmmSpec(
      x=x,
      w=w,
      w2=w2,
      gather=spec.RoutingSpec(num_indices=routed_rows, remote_axis=None),
      scatter=scatter_spec,
      num_global_groups=num_experts,
      fuse_act="silu" if with_down else None,
      size_ep=ep_size if remote_scatter else 1,
      size_row_group=routed_rows if row_group else 0,
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
    row_group: bool = True,
    **spec_kwargs: Any,
) -> gmm_plan.ResolvedGmmPlan:
  """Resolves the plan of `_make_spec(num_tokens, ...)`."""
  gmm_spec = _make_spec(
      num_tokens,
      quantized=quantized,
      with_down=with_down,
      size_n=size_n,
      w_dtype=w_dtype,
      row_group=row_group,
      **spec_kwargs,
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


def _make_config(
    num_tokens: int,
    custom_plan: gmm_plan.RoutingGmmPlan | None = None,
    *,
    quantized: bool = True,
    row_group: bool = True,
    with_down: bool = False,
    with_scatter: bool = True,
) -> gmm_config.RoutingGmmConfig:
  """Builds a RoutingGmmConfig for a 512-expert, top-10 MoE layer."""
  gmm_spec = _make_spec(
      num_tokens,
      quantized=quantized,
      row_group=row_group,
      with_down=with_down,
      with_scatter=with_scatter,
  )
  resolved_out_format = out_format_lib.ResolvedActivationFormat.create(
      None,
      default_dtype=gmm_spec.x.dtype,
      default_quant=out_format_lib.ResolvedLaneStrided(
          block_col_size=out_format_lib.tile_size
      ),
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


class TileMTest(parameterized.TestCase):
  """Checks the tile_m the planner picks."""

  @parameterized.product(num_tokens=[1, 4, 8], quantized=[True, False])
  def test_decode_tile_is_one_sublane_tile(
      self, num_tokens: int, quantized: bool
  ):
    # An expert gets at most one row per token, so one sublane tile suffices.
    resolved = _resolve(num_tokens, quantized=quantized)
    min_tile_m = _tpu_info().num_sublanes
    self.assertEqual(resolved.tile_m, min_tile_m)
    self.assertEqual(resolved.bucket_rows, [min_tile_m])

  @parameterized.parameters(12, 64, 200)
  def test_tile_is_capped_at_tokens(self, num_tokens: int):
    min_tile_m = _tpu_info().num_sublanes
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


class FusedK2SlicingTest(parameterized.TestCase):
  """Checks when the fused kernel slices the intermediate (k2) dim."""

  def test_fits_whole_k2_when_vmem_allows(self):
    # A layer that fits fused with the whole k2 dim is never sliced.
    resolved = _resolve(8, with_down=True)
    self.assertTrue(resolved.fuse_up_and_down)
    # `_resolve(with_down=True)` uses an intermediate size of 1536 // 2.
    self.assertEqual(resolved.tile_k2, 768)

  def test_remote_scatter_selects_split_k2_plan(self):
    # With a remote unpermute the planner keeps tile_m=256 and slices k2.
    resolved = _resolve(
        256,
        with_down=True,
        size_k=7168,
        size_k2=3072,
        num_experts=384,
        ep_size=8,
        top_k=6,
        w_q_block_size=512,
        remote_scatter=True,
    )
    self.assertTrue(resolved.fuse_up_and_down)
    self.assertEqual(resolved.tile_m, 256)
    self.assertEqual(resolved.tile_k2, 512)
    self.assertEqual(resolved.tile_n, 7168)
    self.assertEqual(resolved.num_w_buffers, 2)

  def test_local_scatter_falls_back_to_unfused(self):
    # Without a remote unpermute, slicing k2 is not worth it: run unfused.
    resolved = _resolve(
        256,
        with_down=True,
        size_k=7168,
        size_k2=3072,
        num_experts=384,
        ep_size=8,
        top_k=6,
        w_q_block_size=512,
        remote_scatter=False,
    )
    self.assertFalse(resolved.fuse_up_and_down)


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

  def test_small_w_blocks_keep_x_unquantized(self):
    # size_k=2560 in 20 blocks gives 128-wide blocks, below the MXU width so we
    # dequant weight before the matmul for better mxu utilization.
    # In this case, we don't quantize the activations. Otherwise, fp8 @ bf16
    # costs precision and buys no speed.
    resolved = _resolve(8, num_w_q_blocks=20)
    self.assertEqual(resolved.matmul_dtype, jnp.bfloat16)
    self.assertIsNone(resolved.x_quant_block_size)

  def test_small_w_blocks_keep_explicit_matmul_dtype(self):
    custom_plan = gmm_plan.RoutingGmmPlan(matmul_dtype=jnp.float8_e4m3fn)
    resolved = _resolve(8, custom_plan, num_w_q_blocks=20)
    self.assertEqual(resolved.matmul_dtype, jnp.float8_e4m3fn)


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
    # but unfuses with 3 buffers or f32 stash dtype. 32 tokens give tile_m=32.
    fused_budget = dict(vmem_limit_bytes=13_550_000, vmem_usage_fraction=1.0)
    default_two_stage = _resolve(
        32,
        gmm_plan.RoutingGmmPlan(**fused_budget),
        with_down=True,
        out_format=fmt,
    )
    larger_stash_two_stage = _resolve(
        32,
        dataclasses.replace(extra_plan, **fused_budget),
        with_down=True,
        out_format=fmt,
    )
    self.assertTrue(default_two_stage.fuse_up_and_down)
    self.assertFalse(larger_stash_two_stage.fuse_up_and_down)


def _shape_bytes(shape: tuple[int, ...], dtype: jax.typing.DTypeLike) -> int:
  return math.prod(shape) * jnp.dtype(dtype).itemsize


def _leaf_name(path: jax.tree_util.KeyPath) -> str:
  return jax.tree_util.keystr(path, simple=True, separator=".")


def _traced_smem_allocations(
    cfg: gmm_config.RoutingGmmConfig,
    metadata_in: Any,
    metadata_scratch: Any,
    inputs: Any = None,
) -> dict[str, int]:
  """Logical bytes of each SMEM allocation."""
  if jax.typeof(metadata_in.group_cum_sum).memory_space != pltpu.SMEM:
    raise ValueError(
        "smem_allocations must be called under jit, where SMEM constraints are"
        " active."
    )

  allocs = {}
  for path, leaf in jax.tree_util.tree_leaves_with_path((metadata_in, inputs)):
    if jax.typeof(leaf).memory_space == pltpu.SMEM:
      key = _leaf_name(path[1:])
      assert key not in allocs
      allocs[key] = _shape_bytes(leaf.shape, leaf.dtype)
  for path, ref in jax.tree_util.tree_leaves_with_path(metadata_scratch):
    if ref.memory_space == pltpu.SMEM:
      key = _leaf_name(path)
      assert key not in allocs
      allocs[key] = _shape_bytes(ref.shape, ref.dtype)
  if inputs is not None:
    for field, idx in (
        ("gather_idx", inputs.gather_idx),
        ("scatter_idx", inputs.scatter_idx),
    ):
      if idx is not None and jax.typeof(idx).memory_space != pltpu.SMEM:
        shape = (cfg.plan.num_idx_buffers, cfg.plan.idx_buffer_size)
        key = f"{field}_buffers"
        assert key not in allocs
        allocs[key] = _shape_bytes(shape, idx.dtype)
  return allocs


def _weight_wrapper(
    weight_spec: spec.WeightSpec | None,
) -> arg.SingleWeightWrapper | None:
  if weight_spec is None:
    return None
  scale = (
      None
      if weight_spec.quant is None
      else jax.ShapeDtypeStruct(weight_spec.scale_shape, jnp.float32)
  )
  return arg.SingleWeightWrapper(
      weight=jax.ShapeDtypeStruct(weight_spec.shape, weight_spec.dtype),
      scale=scale,
      bias=None,
      spec=weight_spec,
  )


def _routing_wrapper(
    routing_spec: spec.RoutingSpec | None,
) -> arg.RoutingWrapper | None:
  if routing_spec is None:
    return None
  return arg.RoutingWrapper(
      indices=jax.ShapeDtypeStruct((routing_spec.num_indices,), jnp.int32),
      spec=routing_spec,
  )


def _trace_and_record(
    act_data,
    w_wrapper,
    w2_wrapper,
    group_sizes,
    row_group,
    gather,
    scatter,
    *,
    cfg: gmm_config.RoutingGmmConfig,
    allocs: dict[str, int],
) -> None:
  metadata_in, scratch, inputs, _ = kernel._prepare_kernel_args(
      act_data,
      w_wrapper,
      w2_wrapper,
      group_sizes,
      row_group=row_group,
      group_offset=None,
      gather=gather,
      scatter=scatter,
      cfg=cfg,
  )
  allocs.update(_traced_smem_allocations(cfg, metadata_in, scratch, inputs))


def _kernel_smem(cfg: gmm_config.RoutingGmmConfig) -> dict[str, int]:
  """SMEM allocations that `routing_gmm` hands its kernel for `cfg`."""
  sds = jax.ShapeDtypeStruct
  spec_val = cfg.spec
  allocs = {}
  act_wrapper = arg.ActivationWrapper(
      data=sds(spec_val.x.physical_compact_shape, spec_val.x.dtype),
      scale=None,
      spec=spec_val.x,
  )
  row_group = (
      sds((spec_val.size_row_group,), jnp.int32)
      if spec_val.size_row_group
      else None
  )
  trace_fn = functools.partial(_trace_and_record, cfg=cfg, allocs=allocs)
  jax.eval_shape(
      trace_fn,
      act_wrapper,
      _weight_wrapper(spec_val.w),
      _weight_wrapper(spec_val.w2),
      sds((spec_val.num_global_groups,), jnp.int32),
      row_group,
      _routing_wrapper(spec_val.gather),
      _routing_wrapper(spec_val.scatter),
  )
  return allocs


class SmemBudgetTest(parameterized.TestCase):
  """Checks the SMEM budget and kernel SMEM allocation accounting."""

  def test_budget(self):
    # int(0.6 * 1 MiB) - 32 KiB = 596,377 B on TPU v7x.
    self.assertEqual(gmm_plan.smem_budget_bytes(_tpu_info()), 596_377)

  def test_smem_allocations_outside_jit_raises(self):
    cfg = _make_config(8, quantized=False)
    spec_val = cfg.spec
    act_wrapper = arg.ActivationWrapper(
        data=jnp.zeros(spec_val.x.physical_compact_shape, spec_val.x.dtype),
        scale=None,
        spec=spec_val.x,
    )
    weight_wrapper = arg.SingleWeightWrapper(
        weight=jnp.zeros(spec_val.w.shape, spec_val.w.dtype),
        scale=None,
        bias=None,
        spec=spec_val.w,
    )
    metadata_in, scratch, inputs, _ = kernel._prepare_kernel_args(
        act_wrapper,
        weight_wrapper,
        None,
        jnp.zeros((spec_val.num_global_groups,), jnp.int32),
        row_group=None,
        group_offset=None,
        gather=None,
        scatter=None,
        cfg=cfg,
    )
    with self.assertRaisesRegex(
        ValueError, "smem_allocations must be called under jit"
    ):
      _traced_smem_allocations(cfg, metadata_in, scratch, inputs)

  def test_smem_total_is_derived_from_kernel_args(self):
    cfg = _make_config(4096, gmm_plan.RoutingGmmPlan(live_group_walk=True))
    allocs = _kernel_smem(cfg)
    self.assertEqual(allocs["row_group"], 4 * 40960)
    self.assertEqual(
        sum(allocs.values()),
        # group_cum_sum, group_offset, row_group, tile tables, num_gm, and the
        # gather and scatter index buffers.
        4 * 513
        + 4
        + 4 * 40960
        + 3 * (4 * cfg.max_num_gm)
        + 4
        + 2 * (4 * cfg.plan.num_idx_buffers * cfg.plan.idx_buffer_size),
    )
    est = gmm_plan.smem_estimate(cfg.spec, cfg.plan)
    self.assertEqual(sum(est.values()), sum(allocs.values()))
    self.assertEqual(est, allocs)

  def test_forced_walk_raises_under_static_budget(self):
    # 18000 tokens * top-k 10: row_group is 720,000 B, over the 596,377 B
    # budget.
    with self.assertRaisesRegex(ValueError, "row_group.*596377"):
      _make_config(18000, gmm_plan.RoutingGmmPlan(live_group_walk=True))

  def test_live_group_walk_requires_row_group(self):
    forced = gmm_plan.RoutingGmmPlan(live_group_walk=True)
    with self.assertRaisesRegex(ValueError, "requires the row_group"):
      _make_config(8, forced, row_group=False)

  def test_default_live_group_walk_without_row_group_uses_sorted_walk(self):
    self.assertTrue(_resolve(8, row_group=True).live_group_walk)
    self.assertFalse(_resolve(8, row_group=False).live_group_walk)


class SmemEstimateMatchesKernelTest(parameterized.TestCase):
  """Checks that smem_estimate equals the kernel's actual SMEM allocations."""

  @parameterized.named_parameters(
      dict(
          testcase_name="decode_1_up_walk",
          num_tokens=1,
          with_down=False,
          with_scatter=False,
          row_group=True,
          live_group_walk=True,
      ),
      dict(
          testcase_name="decode_16_up_sorted",
          num_tokens=16,
          with_down=False,
          with_scatter=False,
          row_group=True,
          live_group_walk=False,
      ),
      dict(
          testcase_name="decode_16_fused_walk",
          num_tokens=16,
          with_down=True,
          with_scatter=True,
          row_group=True,
          live_group_walk=True,
      ),
      dict(
          testcase_name="decode_16_up_no_row_group",
          num_tokens=16,
          with_down=False,
          with_scatter=False,
          row_group=False,
          live_group_walk=False,
      ),
      dict(
          testcase_name="prefill_256_up_walk",
          num_tokens=256,
          with_down=False,
          with_scatter=False,
          row_group=True,
          live_group_walk=True,
      ),
      dict(
          testcase_name="prefill_256_up_sorted",
          num_tokens=256,
          with_down=False,
          with_scatter=False,
          row_group=True,
          live_group_walk=False,
      ),
      dict(
          testcase_name="prefill_256_fused_sorted",
          num_tokens=256,
          with_down=True,
          with_scatter=True,
          row_group=True,
          live_group_walk=False,
      ),
      dict(
          testcase_name="prefill_4096_up_walk",
          num_tokens=4096,
          with_down=False,
          with_scatter=False,
          row_group=True,
          live_group_walk=True,
      ),
      dict(
          testcase_name="prefill_4096_fused_sorted",
          num_tokens=4096,
          with_down=True,
          with_scatter=True,
          row_group=True,
          live_group_walk=False,
      ),
      dict(
          testcase_name="prefill_4096_up_no_row_group",
          num_tokens=4096,
          with_down=False,
          with_scatter=False,
          row_group=False,
          live_group_walk=False,
      ),
  )
  def test_estimate_matches_traced_allocations(
      self,
      num_tokens: int,
      with_down: bool,
      with_scatter: bool,
      row_group: bool,
      live_group_walk: bool,
  ):
    custom_plan = gmm_plan.RoutingGmmPlan(live_group_walk=live_group_walk)
    cfg = _make_config(
        num_tokens,
        custom_plan,
        quantized=True,
        row_group=row_group,
        with_down=with_down,
        with_scatter=with_scatter,
    )
    traced = _kernel_smem(cfg)
    estimated = gmm_plan.smem_estimate(cfg.spec, cfg.plan)
    self.assertEqual(sum(estimated.values()), sum(traced.values()))
    self.assertEqual(estimated, traced)

  def test_smem_indices_estimate(self):
    cfg = _make_config(16, quantized=True)
    est_hbm = gmm_plan.smem_estimate(cfg.spec, cfg.plan, smem_indices=False)
    est_smem = gmm_plan.smem_estimate(cfg.spec, cfg.plan, smem_indices=True)
    self.assertIn("gather_idx_buffers", est_hbm)
    self.assertIn("scatter_idx_buffers", est_hbm)
    self.assertIn("gather_idx", est_smem)
    self.assertIn("scatter_idx", est_smem)
    assert cfg.spec.gather is not None and cfg.spec.scatter is not None
    num_idx = cfg.spec.gather.num_indices
    tile_m_idx = cfg.plan.tile_m_idx
    self.assertEqual(est_smem["gather_idx"], 4 * (num_idx + tile_m_idx))
    self.assertEqual(est_smem["scatter_idx"], 4 * (num_idx + tile_m_idx))


class VmemEstimateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="single", with_down=False),
      dict(testcase_name="up_down", with_down=True),
  )
  def test_vmem_estimate_bias(self, with_down: bool):
    num_w_buffers = 3

    def estimate(has_bias: bool) -> int:
      gmm_spec = _make_spec(8, with_down=with_down, has_bias=has_bias)
      return gmm_plan._vmem_estimate(
          x=gmm_spec.x,
          w=gmm_spec.w,
          w2=gmm_spec.w2,
          is_fused_act=gmm_spec.is_fused_act,
          matmul_dtype=jnp.dtype(jnp.float8_e4m3fn),
          acc_dtype=jnp.dtype(jnp.bfloat16),
          out_dtype=gmm_spec.x.dtype,
          tile_m=128,
          tile_k=gmm_spec.size_k,
          tile_k2=gmm_spec.size_k2,
          tile_n=gmm_spec.size_out_n,
          num_xo_buffers=2,
          num_w_buffers=num_w_buffers,
          tpu_info=_tpu_info(),
      )

    bytes_per_f32 = 4
    gmm_spec = _make_spec(8, with_down=with_down)
    bias_cols = gmm_spec.w.out_size
    if gmm_spec.w2 is not None:
      bias_cols += gmm_spec.w2.out_size
    self.assertEqual(
        estimate(True) - estimate(False),
        num_w_buffers * bias_cols * bytes_per_f32,
    )

  @parameterized.named_parameters(
      dict(testcase_name="single", with_down=False),
      dict(testcase_name="up_down", with_down=True),
  )
  def test_vmem_estimate_scale(self, with_down: bool):
    num_w_buffers = 3

    def estimate(num_w_q_blocks: int) -> int:
      gmm_spec = _make_spec(
          8, with_down=with_down, num_w_q_blocks=num_w_q_blocks
      )
      return gmm_plan._vmem_estimate(
          x=gmm_spec.x,
          w=gmm_spec.w,
          w2=gmm_spec.w2,
          is_fused_act=gmm_spec.is_fused_act,
          matmul_dtype=jnp.dtype(jnp.float8_e4m3fn),
          acc_dtype=jnp.dtype(jnp.bfloat16),
          out_dtype=gmm_spec.x.dtype,
          tile_m=128,
          tile_k=gmm_spec.size_k,
          tile_k2=gmm_spec.size_k2,
          tile_n=gmm_spec.size_out_n,
          num_xo_buffers=2,
          num_w_buffers=num_w_buffers,
          tpu_info=_tpu_info(),
      )

    # Only the scale term depends on the number of weight quant blocks, so one
    # more block adds one f32 scale row per weight buffer.
    bytes_per_f32 = 4
    gmm_spec = _make_spec(8, with_down=with_down)
    scale_cols = gmm_spec.w.out_size
    if gmm_spec.w2 is not None:
      scale_cols += gmm_spec.w2.out_size
    self.assertEqual(
        estimate(2) - estimate(1),
        num_w_buffers * scale_cols * bytes_per_f32,
    )

  def test_vmem_estimate_dequantized_w(self):
    num_w_buffers = 3

    def estimate(num_w_q_blocks: int) -> int:
      gmm_spec = _make_spec(8, num_w_q_blocks=num_w_q_blocks)
      return gmm_plan._vmem_estimate(
          x=gmm_spec.x,
          w=gmm_spec.w,
          matmul_dtype=gmm_spec.x.dtype,
          acc_dtype=jnp.dtype(jnp.bfloat16),
          out_dtype=gmm_spec.x.dtype,
          tile_m=128,
          tile_k=gmm_spec.size_k,
          tile_n=gmm_spec.size_out_n,
          num_xo_buffers=2,
          num_w_buffers=num_w_buffers,
          tpu_info=_tpu_info(),
      )

    # 128-wide blocks are below the MXU width, so the estimate adds a bf16
    # copy of the dequantized w tile on top of the extra scale rows.
    gmm_spec = _make_spec(8)
    size_k, size_n = gmm_spec.size_k, gmm_spec.size_out_n
    bytes_per_f32, bytes_per_bf16 = 4, 2
    self.assertEqual(
        estimate(20) - estimate(1),
        num_w_buffers * 19 * size_n * bytes_per_f32
        + size_k * size_n * bytes_per_bf16,
    )


if __name__ == "__main__":
  absltest.main()
