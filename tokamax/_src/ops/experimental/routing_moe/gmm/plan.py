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

"""Tunable parameters and heuristic planner of the routing GMM kernel.

`RoutingGmmPlan` exposes optional overrides where `None` delegates to the
heuristic planner, while `ResolvedGmmPlan` holds the concrete parameters
consumed by config construction and the kernel. Keeping them separate avoids
carrying `None` unions past planning.
"""

from collections.abc import Callable
import dataclasses
import functools
from typing import Self

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src import block_scaled
from tokamax._src.ops.experimental.routing_moe.common import dataclass_util
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.common import spec as tensor_spec
from tokamax._src.ops.experimental.routing_moe.gmm import spec as gmm_spec_lib


@dataclass_util.require_none_default
@dataclasses.dataclass(frozen=True, kw_only=True)
class RoutingGmmPlan:
  """User-provided plan where None fields will be determined by heuristic."""

  tile_m: int | None = None
  tile_k: int | None = None
  tile_k2: int | None = None
  tile_n: int | None = None
  bucket_base: int | None = None
  num_xo_buffers: int | None = None
  num_w_buffers: int | None = None
  num_stash_buffers: int | None = None
  # Accumulator dtype across quantization blocks and K tiles.
  acc_dtype: jax.typing.DTypeLike | None = None
  # Dtype in which inputs are fed to the MXU. Performs dynamic quantization if
  # the dtype does not match with storage dtype.
  matmul_dtype: jax.typing.DTypeLike | None = None
  # Dtype in which the intermediate is fed to the MXU for the fused down
  # projection `w2`.
  w2_matmul_dtype: jax.typing.DTypeLike | None = None
  # Block size if dynamic quantization is performed.
  x_quant_block_size: int | None = None
  # VMEM staging dtype for output tiles before output quantization.
  stash_dtype: jax.typing.DTypeLike | None = None
  # Whether gate/up and down projections run in one fused kernel when w2 is set.
  fuse_up_and_down: bool | None = None
  vmem_limit_bytes: int | None = None
  # Walk only the groups with rows when building the metadata (needs the
  # row_group input of routing_gmm).
  live_group_walk: bool | None = None
  # Keep routing gather/scatter indices in SMEM instead of streaming from HBM.
  smem_indices: bool | None = None
  # Scan ahead in weight BufferedRefs to deduplicate consecutive expert tiles.
  weight_lookahead: bool | None = None
  # These fields bound the heuristic search and are omitted once resolved.
  vmem_usage_fraction: float | None = None
  # Upper bound when num_w_buffers is grown into spare VMEM, so that the next
  # groups' weight DMAs stay in flight during the current group's compute.
  max_num_w_buffers: int | None = None
  # Upper bound on tile_m when it spans more than one bucket.
  max_tile_m: int | None = None
  # Upper bound on the number of buckets (tile_m / bucket_base).
  max_num_buckets: int | None = None
  # tile_m is kept at or above i_local // dma_floor_divisor, where i_local is
  # the per-shard intermediate width, so per-tile row gather/scatter stays
  # small next to the group's weight DMA.
  # TODO: The default of 16 was chosen empirically from TPUv7x sweeps
  # and is not derived. Revisit it on other chips and weight dtypes.
  dma_floor_divisor: int | None = None

  def __post_init__(self) -> None:
    if self.dma_floor_divisor is not None and self.dma_floor_divisor <= 0:
      raise ValueError(
          f"dma_floor_divisor must be positive, got {self.dma_floor_divisor}."
      )


def vmem_budget_bytes(tpu_info: pltpu.TpuInfo) -> int:
  """Bytes of TensorCore VMEM allocatable to a kernel after XLA reserves."""
  # XLA carves out top-of-VMEM regions before allocating scoped kernel VMEM:
  # - TPU <= 7: 16 `(num_sublanes, num_lanes)` f32 chunks (64 KiB on v5p-7x)
  #   for compiler overlay register spills.
  # - TPU >= 8: SparseCore subcore SPMEM mirrored in TensorCore VMEM (e.g.
  #   2 MiB on 8i, 8 MiB on 8t).
  # Compiler flags or future XLA changes can reserve additional module-level
  # VMEM; default planning absorbs that headroom via `vmem_usage_fraction`.
  if tpu_info.generation >= 8:
    sc = tpu_info.sparse_core
    reserved = (
        0
        if sc is None
        else sc.num_cores * sc.num_subcores * sc.vmem_capacity_bytes
    )
  else:
    reserved = 16 * tpu_info.num_sublanes * tpu_info.num_lanes * 4
  return tpu_info.vmem_capacity_bytes - reserved


def _resolve_smem_indices(
    plan: RoutingGmmPlan,
    *,
    spec: gmm_spec_lib.RoutingGmmSpec,
    live_group_walk: bool,
    tpu_info: pltpu.TpuInfo,
) -> bool:
  """Resolves whether routing indices reside entirely in SMEM."""
  if plan.smem_indices is not None:
    return plan.smem_indices
  if (spec.gather is None and spec.scatter is None) or spec.size_out_m <= 0:
    return False
  # Bound `gm_id_to_*` before tiling runs using minimum `tile_m` so SMEM
  # indices stay within smem_budget_bytes for any tiling.
  min_tile_m = plan.tile_m or tpu_info.num_sublanes
  allocs = _estimate_smem_allocations(
      spec,
      live_group_walk=live_group_walk,
      tile_m=min_tile_m,
      smem_indices=True,
  )
  return sum(allocs.values()) <= smem_budget_bytes(tpu_info)


@dataclass_util.resolves_from(
    RoutingGmmPlan,
    dropped=(
        "vmem_usage_fraction",
        "max_num_w_buffers",
        "max_tile_m",
        "max_num_buckets",
        "dma_floor_divisor",
    ),
)
@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedGmmPlan:
  """Plan with None fields from user-provided plan resolved by heuristic."""

  tile_m: int
  tile_k: int
  # It is None unless w2 is fused into the same kernel.
  tile_k2: int | None
  tile_n: int
  bucket_base: int
  num_xo_buffers: int
  num_w_buffers: int
  # It is None unless the output is quantized.
  num_stash_buffers: int | None
  acc_dtype: jnp.dtype
  matmul_dtype: jnp.dtype
  # It is None unless w2 is set.
  w2_matmul_dtype: jnp.dtype | None
  # It is None unless x is dynamically quantized.
  x_quant_block_size: int | None
  # It is None unless the output is quantized.
  stash_dtype: jnp.dtype | None
  fuse_up_and_down: bool
  vmem_limit_bytes: int
  live_group_walk: bool
  smem_indices: bool
  weight_lookahead: bool

  @property
  def bucket_rows(self) -> list[int]:
    bucket_base = self.bucket_base
    # Explicitly include tile_m in case tile_m % bucket_base != 0.
    return [*range(bucket_base, self.tile_m, bucket_base), self.tile_m]

  @property
  def tile_m_idx(self) -> int:
    # TODO: Fetch this from tpu_info instead of using magic numbers.
    num_lanes = 128
    return pl.cdiv(self.tile_m, num_lanes) * num_lanes

  @property
  def idx_ring_size(self) -> int:
    # Index uses 2 buffers. Despite this, we add at least one more gap to ensure
    # write slot does not overwrite read slot.
    return self.tile_m_idx * max(self.num_xo_buffers, 3)

  @property
  def idx_buffer_size(self) -> int:
    # Add tile_m_idx for "runaway" mentioned in IndexBufferedRef docstring.
    return self.idx_ring_size + self.tile_m_idx

  @property
  def num_idx_buffers(self) -> int:
    # We hardcode this and make it immutable since it is load bearing.
    # It is because "2" works correctly for determining a slot index even when
    # a dividend is underflowed uint.
    return 2

  @classmethod
  def create(
      cls,
      plan: RoutingGmmPlan | None,
      *,
      spec: gmm_spec_lib.RoutingGmmSpec,
      out_format: out_format_lib.ResolvedActivationFormat,
      tpu_info: pltpu.TpuInfo,
  ) -> Self:
    """Resolves unset plan fields into concrete execution parameters."""
    if plan is None:
      plan = RoutingGmmPlan()

    matmul_dtype, acc_dtype, x_quant_block_size = _resolve_dtypes(
        plan, x=spec.x, w=spec.w, tpu_info=tpu_info
    )
    num_stash_buffers, stash_dtype = _resolve_stash(
        plan, out_quant=out_format.quant
    )

    live_group_walk = _resolve_live_group_walk(plan, spec=spec)
    smem_indices = _resolve_smem_indices(
        plan, spec=spec, live_group_walk=live_group_walk, tpu_info=tpu_info
    )

    if spec.w2 is None and plan.tile_k2 is not None:
      raise ValueError("tile_k2 requires w2 to be set.")
    if spec.w2 is None and plan.fuse_up_and_down:
      raise ValueError("fuse_up_and_down=True requires w2 to be set.")
    if spec.w2 is None and plan.w2_matmul_dtype is not None:
      raise ValueError("w2_matmul_dtype requires w2 to be set.")

    vmem_usage_fraction = plan.vmem_usage_fraction or 0.9
    vmem_budget = vmem_budget_bytes(tpu_info)
    vmem_limit_bytes = min(plan.vmem_limit_bytes or vmem_budget, vmem_budget)
    tiling, fuse_up_and_down = resolve_tiling_and_fusion(
        plan,
        spec=spec,
        matmul_dtype=matmul_dtype,
        acc_dtype=acc_dtype,
        out_dtype=out_format.dtype,
        out_quant=out_format.quant,
        num_stash_buffers=num_stash_buffers,
        stash_dtype=stash_dtype,
        vmem_limit_bytes=int(vmem_usage_fraction * vmem_limit_bytes),
        tpu_info=tpu_info,
    )
    if not fuse_up_and_down and plan.w2_matmul_dtype is not None:
      raise ValueError("w2_matmul_dtype requires fuse_up_and_down=True.")
    w2_matmul_dtype = _resolve_w2_matmul_dtype(
        plan,
        x=spec.x,
        w2=spec.w2,
        matmul_dtype=matmul_dtype,
        tile_m=tiling.tile_m,
        tpu_info=tpu_info,
    )
    weight_lookahead = _resolve_weight_lookahead(
        plan,
        spec=spec,
        tile_m=tiling.tile_m,
        num_w_buffers=tiling.num_w_buffers,
    )
    resolved = cls(
        tile_m=tiling.tile_m,
        tile_k=tiling.tile_k,
        tile_k2=tiling.tile_k2,
        tile_n=tiling.tile_n,
        bucket_base=tiling.bucket_base,
        num_xo_buffers=tiling.num_xo_buffers,
        num_w_buffers=tiling.num_w_buffers,
        num_stash_buffers=num_stash_buffers,
        acc_dtype=acc_dtype,
        matmul_dtype=matmul_dtype,
        w2_matmul_dtype=w2_matmul_dtype,
        x_quant_block_size=x_quant_block_size,
        stash_dtype=stash_dtype,
        fuse_up_and_down=fuse_up_and_down,
        vmem_limit_bytes=vmem_limit_bytes,
        live_group_walk=live_group_walk,
        smem_indices=smem_indices,
        weight_lookahead=weight_lookahead,
    )
    if spec.w2 is None or fuse_up_and_down:
      check_smem(smem_estimate(spec, resolved), tpu_info)
    return resolved


def smem_budget_bytes(tpu_info: pltpu.TpuInfo) -> int:
  """Bytes of SMEM that routing gmm may allocate without slowing down."""
  # XLA gives a custom call this fraction of SMEM by default
  # (xla_tpu_impure_smem_callee_ratio); the caller keeps the rest. Past this
  # split, sreg spill slots go to HBM and routing gmm runs much slower. Under
  # dynamic SMEM negotiation the kernel gets at least as much, so this static
  # budget stays safe (only conservative).
  callee_ratio = 0.6
  # Left for sreg spill slots, the compiler's own SMEM, and Mosaic rounding
  # each kernel SMEM buffer up to 512 B (a few KiB at most).
  reserved_bytes = 32 * 1024
  return int(callee_ratio * tpu_info.smem_capacity_bytes) - reserved_bytes


def check_smem(allocations: dict[str, int], tpu_info: pltpu.TpuInfo) -> None:
  """Raises if the kernel's SMEM `allocations` (bytes) exceed the budget."""
  total = sum(allocations.values())
  budget = smem_budget_bytes(tpu_info)
  if total <= budget:
    return
  raise ValueError(
      f"Routing gmm SMEM allocations {allocations} total {total} bytes, over"
      f" the budget of {budget} bytes. Reduce the tokens, or set"
      " live_group_walk=False if row_group is large."
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Tiling:
  """Tile sizes and buffer counts of the routing GMM kernel."""

  tile_m: int
  tile_k: int
  tile_k2: int | None
  tile_n: int
  bucket_base: int
  num_xo_buffers: int
  num_w_buffers: int


def resolve_tiling_and_fusion(
    plan: RoutingGmmPlan,
    *,
    spec: gmm_spec_lib.RoutingGmmSpec,
    matmul_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    out_quant: out_format_lib.ResolvedQuantFormat | None,
    num_stash_buffers: int | None = None,
    stash_dtype: jnp.dtype | None = None,
    vmem_limit_bytes: int,
    tpu_info: pltpu.TpuInfo,
) -> tuple[Tiling, bool]:
  """Decides together whether to fuse the projections and how to tile them.

  With w2, try in order:
    1. Fused gate/up and down projections without k2 tiling with the largest
       possible tile_m.
    2. With a remote scatter (remote unpermute) only: fused projections with
       the largest possible tile_m and the k2 dim sliced as little as fits VMEM.
    3. Two unfused kernels.

  Args:
    plan: User plan. Fields that are set are kept.
    spec: Validated input specifications and operator semantics.
    matmul_dtype: Dtype in which inputs are fed to the MXU.
    acc_dtype: Accumulator dtype.
    out_dtype: Output dtype of the layer.
    out_quant: Optional output quantization format of the layer.
    num_stash_buffers: Stash buffer count when `out_quant` is set, else None.
    stash_dtype: Stash VMEM dtype when `out_quant` is set, else None.
    vmem_limit_bytes: VMEM budget of the kernel.
    tpu_info: TPU the kernel runs on.

  Returns:
    The tiling, and whether the gate/up and down projections are fused.

  Raises:
    ValueError: If `plan.fuse_up_and_down` is True but no fused tiling fits.
  """
  if spec.w2 is not None and (
      plan.fuse_up_and_down is None or plan.fuse_up_and_down
  ):
    tiling = _fused_tiling(
        plan,
        spec=spec,
        matmul_dtype=matmul_dtype,
        acc_dtype=acc_dtype,
        out_dtype=out_dtype,
        out_quant=out_quant,
        num_stash_buffers=num_stash_buffers,
        stash_dtype=stash_dtype,
        vmem_limit_bytes=vmem_limit_bytes,
        tpu_info=tpu_info,
    )
    if tiling is not None:
      return tiling, True
    if plan.fuse_up_and_down:
      raise ValueError(
          "fuse_up_and_down=True, but no fused tiling fits the VMEM budget of"
          f" {vmem_limit_bytes} bytes (even with the intermediate dim sliced)."
      )
  if plan.tile_k2 is not None:
    raise ValueError("tile_k2 requires fuse_up_and_down=True.")
  tiling = calculate_tiling(
      plan,
      spec=spec,
      matmul_dtype=matmul_dtype,
      acc_dtype=acc_dtype,
      out_dtype=out_dtype,
      out_quant=out_quant,
      num_stash_buffers=num_stash_buffers,
      stash_dtype=stash_dtype,
      vmem_limit_bytes=vmem_limit_bytes,
      tpu_info=tpu_info,
  )
  return tiling, False


def _row_align(
    spec: gmm_spec_lib.RoutingGmmSpec,
    tpu_info: pltpu.TpuInfo,
) -> int:
  """Returns row alignment (`2 * num_sublanes` on multi-tile routed ops)."""
  min_tile_m = tpu_info.num_sublanes
  return (
      2 * min_tile_m
      if (spec.has_routed_io and spec.x.row_size > min_tile_m)
      else min_tile_m
  )


def _max_group_rows(
    spec: gmm_spec_lib.RoutingGmmSpec,
    tpu_info: pltpu.TpuInfo,
) -> int:
  """Rows a tile ever needs, aligned to packed sublane width."""
  row_align = _row_align(spec, tpu_info)
  return row_align * max(1, pl.cdiv(spec.x.row_size, row_align))


def _tile_m_limits(plan: RoutingGmmPlan) -> tuple[int, int]:
  """Returns the max tile_m and max number of buckets of the heuristic."""
  # max_tile_m Defaults to 256.
  # TODO: Lift the 256 cap once larger tiles are validated.

  # max_num_buckets Defaults to 2.
  # TODO: The default was chosen empirically. Revisit it based on
  # instruction memory size.
  return plan.max_tile_m or 256, plan.max_num_buckets or 2


def _tile_m_candidates(
    plan: RoutingGmmPlan,
    *,
    bucket_base: int,
    max_group_rows: int,
    single_routed_bucket: bool = False,
) -> list[int]:
  """The tile_m values the heuristic may pick, largest first."""
  if plan.tile_m is not None:
    return [plan.tile_m]
  max_tile_m, max_num_buckets = _tile_m_limits(plan)
  if single_routed_bucket:
    max_num_buckets = 1
  candidates = set()
  for num_buckets in range(1, max_num_buckets + 1):
    tile_m = num_buckets * bucket_base
    # Like _resolve_tile_m, a single bucket is always a candidate.
    if num_buckets > 1 and tile_m > max_tile_m:
      break
    if plan.bucket_base is None:
      tile_m = min(tile_m, max_group_rows)
    candidates.add(tile_m)
  return sorted(candidates, reverse=True)


def _tile_k2_candidates(
    size_k2: int, w2: tensor_spec.WeightSpec, tpu_info: pltpu.TpuInfo
) -> list[int]:
  """The k2 tilings the fused kernel accepts, fewest first."""
  candidates = []
  for num_k2_tiles in range(1, pl.cdiv(size_k2, tpu_info.num_lanes) + 1):
    if size_k2 % num_k2_tiles != 0:
      continue
    tile_k2 = size_k2 // num_k2_tiles
    if tile_k2 % tpu_info.num_lanes != 0:
      continue
    if w2.quant is not None:
      if not isinstance(w2.quant, tensor_spec.SubchannelQuantSpec):
        continue
      sizes = (tile_k2, w2.quant.block_size)
      if max(sizes) % min(sizes) != 0:
        continue
    candidates.append(tile_k2)
  return candidates


def _fits_in(src: jnp.dtype, dst: jnp.dtype) -> bool:
  """Returns whether every value of floating-point `src` is exact in `dst`."""
  s, d = jnp.finfo(src), jnp.finfo(dst)
  return (
      d.nmant >= s.nmant  # number of mantissa bits
      and float(d.max) >= float(s.max)
      # smallest_subnormal is the smallest positive value the format can
      # represent.
      and float(d.smallest_subnormal) <= float(s.smallest_subnormal)
  )


def is_block_scaled_tpu(tpu_info: pltpu.TpuInfo) -> bool:
  """Returns whether `tpu_info` supports block-scaled matmul."""
  tpu_8t = getattr(pltpu.ChipVersion, "TPU_8T", None)
  return (tpu_8t is not None and tpu_info.chip_version == tpu_8t) or (
      tpu_info.generation > 8
  )


def _w_has_small_quant_block(
    w: tensor_spec.WeightSpec, tpu_info: pltpu.TpuInfo
) -> bool:
  """Returns whether w's quant blocks are narrower than the MXU."""
  if not isinstance(w.quant, tensor_spec.SubchannelQuantSpec):
    return False
  return w.quant.block_size < tpu_info.mxu_column_size


def dequantizes_w_before_matmul(
    *,
    x_dtype: jnp.dtype,
    matmul_dtype: jnp.dtype,
    w: tensor_spec.WeightSpec,
    tpu_info: pltpu.TpuInfo,
) -> bool:
  """Returns whether w is dequantized in VMEM before the matmul."""
  return matmul_dtype == x_dtype and _w_has_small_quant_block(w, tpu_info)


def _quantized_x_dtype(
    w_dtype: jnp.dtype, tpu_info: pltpu.TpuInfo
) -> jnp.dtype:
  """Returns the narrowest x dtype the MXU can pair with quantized w."""
  # matmul_dtype is the dtype x is converted to right before it goes
  # into the MXU. See RoutingGmmConfig.create(). compute.py converts the
  # weight dtype to matmul_dtype before the matmul if
  # is_matmul_supported(matmul_dtype, w.dtype) is false.
  # We pick the narrowest x dtype that satisfies is_matmul_supported.
  if not jnp.issubdtype(w_dtype, jnp.floating):
    return w_dtype
  for dtype in (jnp.float8_e4m3fn, jnp.float8_e4m3b11fnuz):
    dtype = jnp.dtype(dtype)
    if tpu_info.is_matmul_supported(dtype, w_dtype) or (
        _fits_in(w_dtype, dtype) and tpu_info.is_matmul_supported(dtype, dtype)
    ):
      return dtype
  raise NotImplementedError(
      f"No supported fp8 activation dtype for {w_dtype} weights on"
      f" {tpu_info.chip_version}"
  )


def _resolve_dtypes(
    plan: RoutingGmmPlan,
    *,
    x: tensor_spec.ActivationSpec,
    w: tensor_spec.WeightSpec,
    tpu_info: pltpu.TpuInfo,
) -> tuple[jnp.dtype, jnp.dtype, int | None]:
  """Resolves execution dtypes and `x` quantization block size."""
  x_quant_block_size = plan.x_quant_block_size
  match w.quant:
    case None:
      default_matmul_dtype = x.dtype
    case tensor_spec.BlockScaledQuantSpec():
      if not is_block_scaled_tpu(tpu_info):
        raise NotImplementedError(
            "Block-scaled quantization requires TPU_8T or newer. Got"
            f" {tpu_info.chip_version}."
        )
      default_matmul_dtype = w.dtype if x.quant is None else x.dtype
    case tensor_spec.SubchannelQuantSpec():
      if x.quant is None and not _w_has_small_quant_block(w, tpu_info):
        default_matmul_dtype = _quantized_x_dtype(w.dtype, tpu_info)
      else:
        default_matmul_dtype = x.dtype

  matmul_dtype = (
      default_matmul_dtype
      if plan.matmul_dtype is None
      else jnp.dtype(plan.matmul_dtype)
  )
  if (
      x_quant_block_size is None
      and matmul_dtype != x.dtype
      and not isinstance(w.quant, tensor_spec.BlockScaledQuantSpec)
  ):
    # TODO: Choose different block size based on HW.
    x_quant_block_size = 512

  acc_dtype = (
      jnp.bfloat16.dtype
      if plan.acc_dtype is None
      else jnp.dtype(plan.acc_dtype)
  )
  return matmul_dtype, acc_dtype, x_quant_block_size


def _resolve_live_group_walk(
    plan: RoutingGmmPlan, *, spec: gmm_spec_lib.RoutingGmmSpec
) -> bool:
  """Resolves whether metadata walks only the groups with rows."""
  live_group_walk = plan.live_group_walk
  if live_group_walk is None:
    # The walk pays off while most of this shard's groups are empty.
    # size_out_m counts the routed rows of all shards, i.e. (token, pick)
    # pairs, num_tokens * top-k. A shard gets size_out_m * group_size /
    # num_global_groups of them on average, which is fewer than its
    # group_size groups exactly when size_out_m < num_global_groups, under TP
    # or EP alike. This also keeps row_group, one int32 per routed row in
    # SMEM, small.
    live_group_walk = spec.size_out_m < spec.num_global_groups
  if live_group_walk and not spec.size_row_group:
    if plan.live_group_walk:
      raise ValueError("live_group_walk requires the row_group input.")
    # The walk reads row_group; without it, use the sorted walk.
    live_group_walk = False
  return live_group_walk


def _resolve_stash(
    plan: RoutingGmmPlan,
    *,
    out_quant: out_format_lib.ResolvedQuantFormat | None,
) -> tuple[int | None, jnp.dtype | None]:
  """Resolves VMEM stash parameters for quantized output."""
  num_stash_buffers = plan.num_stash_buffers
  stash_dtype = (
      None if plan.stash_dtype is None else jnp.dtype(plan.stash_dtype)
  )
  if out_quant is not None:
    if num_stash_buffers is None:
      num_stash_buffers = 2
    if stash_dtype is None:
      stash_dtype = jnp.bfloat16.dtype
  return num_stash_buffers, stash_dtype


def _resolve_w2_matmul_dtype(
    plan: RoutingGmmPlan,
    *,
    x: tensor_spec.ActivationSpec,
    w2: tensor_spec.WeightSpec | None,
    matmul_dtype: jnp.dtype,
    tile_m: int,
    tpu_info: pltpu.TpuInfo,
) -> jnp.dtype | None:
  """Resolves the LHS execution dtype for the fused down projection `w2`.

  Args:
    plan: User plan. `w2_matmul_dtype` and `matmul_dtype` overrides are kept.
    x: Input activation spec.
    w2: Down-projection weight spec, or None if there is no `w2`.
    matmul_dtype: Resolved MXU input dtype of the gate/up projection.
    tile_m: Resolved row tile size.
    tpu_info: TPU the kernel runs on.

  Returns:
    None when there is no `w2`, otherwise the dtype the intermediate is fed to
    the MXU in.
  """
  if w2 is None:
    return None
  if plan.w2_matmul_dtype is not None:
    return jnp.dtype(plan.w2_matmul_dtype)
  if plan.matmul_dtype is not None:
    return matmul_dtype
  # On small tiles, dynamically quantizing the intermediate activation costs
  # more on VPU than feeding it directly to a hardware mixed-precision MXU,
  # provided w2's quantization block is at least mxu_column_size (narrower
  # blocks would dequantize the full w2 weight tensor before matmul).
  if (
      isinstance(w2.quant, tensor_spec.SubchannelQuantSpec)
      and w2.quant.block_size >= tpu_info.mxu_column_size
      and tile_m <= 2 * tpu_info.num_sublanes
      and tpu_info.is_matmul_supported(x.dtype, w2.dtype)
  ):
    return x.dtype
  return matmul_dtype


def _supports_fusion(
    plan: RoutingGmmPlan, *, spec: gmm_spec_lib.RoutingGmmSpec
) -> bool:
  """Returns whether the fused kernel supports the weights and K/N tiling."""
  w, w2 = spec.w, spec.w2
  assert w2 is not None
  if w.dtype != w2.dtype or type(w.quant) is not type(w2.quant):
    # The intermediate is fed to the down projection in x's dtype, which is
    # only valid when both weights share a precision.
    return False
  # Fused execution does not support weight tiling along K or N.
  tile_k = plan.tile_k or spec.size_k
  tile_n = plan.tile_n or spec.size_out_n
  return tile_k == spec.size_k and tile_n == spec.size_out_n


def _w_buffer_range(
    plan: RoutingGmmPlan,
    *,
    spec: gmm_spec_lib.RoutingGmmSpec,
    tile_m: int,
) -> tuple[int, int]:
  """Returns the min and max number of weight buffers to try."""
  min_num_w_buffers = plan.num_w_buffers or 2
  if plan.num_w_buffers is not None:
    return min_num_w_buffers, min_num_w_buffers
  # TODO: On fused two-projection kernels at tile_m >= 64, tile
  # compute empirically covers weight DMA latency at 4 buffers while avoiding
  # extra prologue weight prefetches; on large (>= 12 MiB) expert weights, cap
  # at 3 buffers so prologue weight DMAs do not stall slot 0.
  if _is_large_expert_weight(spec):
    default_max = 3
  elif spec.w2 is not None and tile_m >= 64:
    default_max = 4
  else:
    default_max = 8
  return min_num_w_buffers, plan.max_num_w_buffers or default_max


def _max_fitting_w_buffers(
    fits: Callable[[int], bool],
    *,
    min_num_w_buffers: int,
    max_num_w_buffers: int,
) -> int | None:
  """Returns the most weight buffers that fit, or None if the min doesn't."""
  if not fits(min_num_w_buffers):
    return None
  # Spend spare VMEM on deeper weight prefetch.
  num_w_buffers = min_num_w_buffers
  while num_w_buffers < max_num_w_buffers and fits(num_w_buffers + 1):
    num_w_buffers += 1
  return num_w_buffers


def _may_slice_k2(
    plan: RoutingGmmPlan, *, spec: gmm_spec_lib.RoutingGmmSpec
) -> bool:
  """Returns whether the fused kernel may slice the k2 dim."""
  # Output rows are DMAed straight to the owning device (remote unpermute).
  remote_scatter = (
      spec.scatter is not None and spec.scatter.remote_axis is not None
  )
  return bool(
      remote_scatter or plan.fuse_up_and_down or plan.tile_k2 is not None
  )


def _sliced_tile_k2_candidates(
    plan: RoutingGmmPlan,
    *,
    size_k2: int,
    w2: tensor_spec.WeightSpec,
    tpu_info: pltpu.TpuInfo,
) -> list[int]:
  """The tile_k2 values to try with k2 sliced, fewest slices first."""
  if plan.tile_k2 is not None:
    return [plan.tile_k2]
  return [
      tile_k2
      for tile_k2 in _tile_k2_candidates(size_k2, w2, tpu_info)
      if tile_k2 < size_k2
  ]


def _fused_tiling(
    plan: RoutingGmmPlan,
    *,
    spec: gmm_spec_lib.RoutingGmmSpec,
    matmul_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    out_quant: out_format_lib.ResolvedQuantFormat | None,
    num_stash_buffers: int | None = None,
    stash_dtype: jnp.dtype | None = None,
    vmem_limit_bytes: int,
    tpu_info: pltpu.TpuInfo,
) -> Tiling | None:
  """Returns the tiling of the fused kernel, or None if none fits VMEM.

  Tries in order:
    1. No k2 tiling with the largest possible tile_m (stepping tile_m down when
       `plan.fuse_up_and_down` is True).
    2. Largest possible tile_m with the fewest k2 slices that fit.

  Args:
    plan: User plan. Fields that are set are kept.
    spec: Validated input specifications and operator semantics. Must have w2.
    matmul_dtype: Dtype in which inputs are fed to the MXU.
    acc_dtype: Accumulator dtype.
    out_dtype: Output dtype.
    out_quant: Optional output quantization format.
    num_stash_buffers: Stash buffer count when `out_quant` is set, else None.
    stash_dtype: Stash VMEM dtype when `out_quant` is set, else None.
    vmem_limit_bytes: VMEM budget of the kernel.
    tpu_info: TPU the kernel runs on.

  Returns:
    The fused kernel tiling, or None if no fused tiling fits VMEM.
  """
  x, w, w2 = spec.x, spec.w, spec.w2
  size_k2 = spec.size_k2
  assert w2 is not None and size_k2 is not None
  if not _supports_fusion(plan, spec=spec):
    return None
  tile_k, tile_n = spec.size_k, spec.size_out_n

  min_tile_m = tpu_info.num_sublanes
  max_group_rows = _max_group_rows(spec, tpu_info)
  num_xo_buffers = plan.num_xo_buffers or 3

  estimate_vmem = functools.partial(
      _vmem_estimate,
      x=x,
      w=w,
      w2=w2,
      is_fused_act=spec.is_fused_act,
      matmul_dtype=matmul_dtype,
      acc_dtype=acc_dtype,
      out_dtype=out_dtype,
      out_quant=out_quant,
      tile_k=tile_k,
      tile_n=tile_n,
      num_xo_buffers=num_xo_buffers,
      num_stash_buffers=num_stash_buffers,
      stash_dtype=stash_dtype,
      tpu_info=tpu_info,
  )

  def try_fit(tile_m: int, tile_k2: int, bucket_base: int) -> Tiling | None:
    w2_matmul_dtype = _resolve_w2_matmul_dtype(
        plan,
        x=x,
        w2=w2,
        matmul_dtype=matmul_dtype,
        tile_m=tile_m,
        tpu_info=tpu_info,
    )
    min_num_w_buffers, max_num_w_buffers = _w_buffer_range(
        plan, spec=spec, tile_m=tile_m
    )
    num_w_buffers = _max_fitting_w_buffers(
        lambda n: estimate_vmem(
            tile_m=tile_m,
            tile_k2=tile_k2,
            num_w_buffers=n,
            w2_matmul_dtype=w2_matmul_dtype,
        )
        <= vmem_limit_bytes,
        min_num_w_buffers=min_num_w_buffers,
        max_num_w_buffers=max_num_w_buffers,
    )
    if num_w_buffers is None:
      return None
    return Tiling(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_k2=tile_k2,
        tile_n=tile_n,
        bucket_base=bucket_base,
        num_xo_buffers=num_xo_buffers,
        num_w_buffers=num_w_buffers,
    )

  bucket_base, single_routed_bucket = _resolve_bucket_base(
      plan,
      spec=spec,
      matmul_dtype=matmul_dtype,
      max_group_rows=max_group_rows,
      min_tile_m=min_tile_m,
  )
  largest_tile_m = _tile_m_candidates(
      plan,
      bucket_base=bucket_base,
      max_group_rows=max_group_rows,
      single_routed_bucket=single_routed_bucket,
  )[0]

  # Step 1: Try fusion + largest possible tile_m.
  # tile_m has more impact on performance than fusion based on some experiments.
  # Heuristic: It may not worth to use fusion with a smaller tile_m vs unfused
  # with a larger tile_m.
  if plan.tile_k2 is None or plan.tile_k2 == size_k2:
    tiling = try_fit(largest_tile_m, size_k2, bucket_base)
    if tiling is not None:
      return tiling
    if plan.fuse_up_and_down and plan.tile_m is None:
      floor_m = max(min_tile_m, plan.bucket_base or 0)
      cand_m = largest_tile_m
      while cand_m > floor_m:
        cand_m = max(1 << ((cand_m - 1).bit_length() - 1), floor_m)
        tiling = try_fit(cand_m, size_k2, min(bucket_base, cand_m))
        if tiling is not None:
          return tiling

  # Step 2: Slice k2 with largest possible tile_m.
  if not _may_slice_k2(plan, spec=spec):
    return None
  tile_k2_candidates = _sliced_tile_k2_candidates(
      plan, size_k2=size_k2, w2=w2, tpu_info=tpu_info
  )

  # If fusion requires k2 tiling, the weights can't be reused across m tiles,
  # so we only attempt to do fusion + k2 tiling when remote scatter (to overlap
  # ICI with a larger block of computation) and tile_m is sufficiently large
  # (for HBM bw and compute overlap).
  bucket_base, _ = _resolve_bucket_base(
      dataclasses.replace(plan, tile_m=largest_tile_m),
      spec=spec,
      matmul_dtype=matmul_dtype,
      max_group_rows=max_group_rows,
      min_tile_m=min_tile_m,
  )
  for tile_k2 in tile_k2_candidates:
    tiling = try_fit(largest_tile_m, tile_k2, bucket_base)
    if tiling is not None:
      return tiling
  return None


def _estimate_smem_allocations(
    spec: gmm_spec_lib.RoutingGmmSpec,
    *,
    live_group_walk: bool,
    tile_m: int,
    smem_indices: bool,
    num_idx_buffers: int = 0,
    idx_buffer_size: int = 0,
) -> dict[str, int]:
  """Estimates SMEM allocation sizes in bytes for routing GMM."""
  allocs = {}
  allocs["group_sizes"] = 4 * spec.num_global_groups
  allocs["group_offset"] = 4
  if live_group_walk and spec.size_row_group:
    allocs["row_group"] = 4 * spec.size_row_group

  max_num_gm = spec.size_local_groups + pl.cdiv(spec.size_out_m, tile_m)
  allocs["gm_id_to_group_id"] = 4 * max_num_gm
  allocs["gm_id_to_m_offset"] = 4 * max_num_gm
  allocs["gm_id_to_num_rows"] = 4 * max_num_gm
  allocs["num_gm"] = 4

  # smem_indices covers indices passed as SMEM kernel inputs.
  for name, routing in (
      ("gather_idx", spec.gather),
      ("scatter_idx", spec.scatter),
  ):
    if routing is not None:
      if smem_indices:
        allocs[name] = 4 * routing.num_indices
      else:
        allocs[f"{name}_buffers"] = 4 * num_idx_buffers * idx_buffer_size

  return allocs


def smem_estimate(
    spec: gmm_spec_lib.RoutingGmmSpec,
    plan: ResolvedGmmPlan,
    *,
    smem_indices: bool | None = None,
) -> dict[str, int]:
  """Estimates SMEM allocation sizes in bytes for routing GMM.

  Args:
    spec: Problem spec.
    plan: Resolved plan.
    smem_indices: Whether routing indices are SMEM kernel inputs. Defaults to
      `plan.smem_indices`.

  Returns:
    SMEM bytes of each kernel allocation, keyed by name.
  """
  if smem_indices is None:
    smem_indices = plan.smem_indices
  return _estimate_smem_allocations(
      spec,
      live_group_walk=plan.live_group_walk,
      tile_m=plan.tile_m,
      smem_indices=smem_indices,
      num_idx_buffers=plan.num_idx_buffers,
      idx_buffer_size=plan.idx_buffer_size,
  )


def _matmul_vmem_bytes(
    weight: tensor_spec.WeightSpec,
    *,
    x_dtype: jnp.dtype,
    lhs_dtype: jnp.dtype,
    acc_bits: int,
    tile_m: int,
    k_tile: int,
    n_tile: int,
    num_w_buffers: int,
    num_w_refs: int,
    tpu_info: pltpu.TpuInfo,
) -> int:
  """Estimates VMEM bytes for one matmul's weights, scales, and accumulators."""
  w_bits = jax.dtypes.itemsize_bits(weight.dtype)
  w_vmem = num_w_refs * num_w_buffers * k_tile * n_tile * w_bits // 8
  scale_vmem = 0
  bytes_per_fp32 = 4
  match weight.quant:
    case None:
      pass
    case tensor_spec.SubchannelQuantSpec(block_size=block_size):
      num_blocks = pl.cdiv(k_tile, block_size)
      scale_vmem = (
          num_w_refs * num_w_buffers * num_blocks * n_tile * bytes_per_fp32
      )
    case tensor_spec.BlockScaledQuantSpec():
      scale_vmem = (
          num_w_refs
          * num_w_buffers
          * block_scaled.kernel_scale_rows(k_tile, dtype=weight.dtype)
          * n_tile
      )
  bias_vmem = 0
  if weight.has_bias:
    bias_vmem = num_w_refs * num_w_buffers * n_tile * bytes_per_fp32
  acc_vmem = num_w_refs * tile_m * n_tile * acc_bits // 8
  lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
  if lhs_dtype != x_dtype:
    acc_vmem += tile_m * k_tile * lhs_bits // 8
    if isinstance(weight.quant, tensor_spec.BlockScaledQuantSpec):
      acc_vmem += block_scaled.activation_scale_vmem_bytes(
          tile_m, k_tile, weight.dtype
      )
  elif dequantizes_w_before_matmul(
      x_dtype=x_dtype, matmul_dtype=lhs_dtype, w=weight, tpu_info=tpu_info
  ):
    acc_vmem += num_w_refs * k_tile * n_tile * lhs_bits // 8
  return w_vmem + scale_vmem + bias_vmem + acc_vmem


# SMEM is estimated from the spec and plan by smem_estimate, and checked
# against smem_budget_bytes in ResolvedGmmPlan.create.
def _vmem_estimate(
    *,
    x: tensor_spec.ActivationSpec,
    w: tensor_spec.WeightSpec,
    w2: tensor_spec.WeightSpec | None = None,
    is_fused_act: bool = False,
    matmul_dtype: jnp.dtype,
    w2_matmul_dtype: jnp.dtype | None = None,
    acc_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    out_quant: out_format_lib.ResolvedQuantFormat | None = None,
    tile_m: int,
    tile_k: int,
    tile_k2: int | None = None,
    tile_n: int,
    num_xo_buffers: int,
    num_w_buffers: int,
    num_stash_buffers: int | None = None,
    stash_dtype: jnp.dtype | None = None,
    tpu_info: pltpu.TpuInfo,
) -> int:
  """Estimates VMEM usage in bytes for routing GMM."""
  acc_bits = jax.dtypes.itemsize_bits(acc_dtype)

  # Chunk size equals the bytes of a single sublane.
  chunk_bytes = 4 * tpu_info.num_lanes
  # NOTE: x_chunk size stays as-is since we don't support K tiling yet.
  x_chunks = layout_util.to_bank_conflict_aware_stride(x.num_physical_chunks)
  out_chunk_cols = layout_util.get_packing(out_dtype) * tpu_info.num_lanes
  out_logical_chunks = pl.cdiv(tile_n, out_chunk_cols)
  out_physical_chunks = out_logical_chunks + (1 if out_quant is not None else 0)
  out_chunks = layout_util.to_bank_conflict_aware_stride(out_physical_chunks)
  x_vmem = num_xo_buffers * tile_m * x_chunks * chunk_bytes
  out_vmem = num_xo_buffers * tile_m * out_chunks * chunk_bytes
  if out_quant is not None:
    assert num_stash_buffers is not None and stash_dtype is not None
    out_bits = jax.dtypes.itemsize_bits(out_dtype)
    stash_bits = jax.dtypes.itemsize_bits(stash_dtype)
    stash_ratio = max(stash_bits // out_bits, 1)
    out_vmem += num_stash_buffers * tile_m * tile_n * stash_bits // 8
    out_vmem += num_stash_buffers * (1 + stash_ratio) * tile_m * chunk_bytes

  num_w1_refs = 2 if is_fused_act else 1
  w1_n_tile = tile_n if w2 is None else tile_k2
  assert w1_n_tile is not None
  matmul_vmem = _matmul_vmem_bytes(
      w,
      x_dtype=x.dtype,
      lhs_dtype=matmul_dtype,
      acc_bits=acc_bits,
      tile_m=tile_m,
      k_tile=tile_k,
      n_tile=w1_n_tile,
      num_w_buffers=num_w_buffers,
      num_w_refs=num_w1_refs,
      tpu_info=tpu_info,
  )
  if w2 is not None:
    assert tile_k2 is not None
    w2_lhs_dtype = matmul_dtype if w2_matmul_dtype is None else w2_matmul_dtype
    matmul_vmem += _matmul_vmem_bytes(
        w2,
        x_dtype=x.dtype,
        lhs_dtype=w2_lhs_dtype,
        acc_bits=acc_bits,
        tile_m=tile_m,
        k_tile=tile_k2,
        n_tile=tile_n,
        num_w_buffers=num_w_buffers,
        num_w_refs=1,
        tpu_info=tpu_info,
    )
    matmul_vmem += tile_m * tile_k2 * jax.dtypes.itemsize_bits(x.dtype) // 8
    if tile_k2 < w2.in_size:
      # Output accumulator kept across k2 slices (PipelineAllocs.out_acc).
      matmul_vmem += tile_m * tile_n * acc_bits // 8

  return x_vmem + out_vmem + matmul_vmem


def _dma_floor(
    plan: RoutingGmmPlan,
    *,
    spec: gmm_spec_lib.RoutingGmmSpec,
    min_tile_m: int,
) -> int:
  """Minimum tile_m so per-tile row gather/scatter stays small next to DMA."""
  if spec.size_k2 is not None:
    i_local = spec.size_k2
  elif spec.w.fused_gate_up:
    i_local = spec.w.out_size // 2
  elif spec.gather is None and spec.scatter is not None:
    i_local = spec.size_k
  else:
    i_local = min(spec.size_k, spec.w.out_size)
  return max(min_tile_m, i_local // (plan.dma_floor_divisor or 16))


def _resolve_bucket_base(
    plan: RoutingGmmPlan,
    *,
    spec: gmm_spec_lib.RoutingGmmSpec,
    matmul_dtype: jnp.dtype,
    max_group_rows: int,
    min_tile_m: int,
) -> tuple[int, bool]:
  """Resolves `(bucket_base, single_routed_bucket)` from plan and group load."""
  if plan.bucket_base is not None:
    return plan.bucket_base, False
  w = spec.w
  is_mxu_quantized = matmul_dtype not in (jnp.bfloat16.dtype, jnp.float32.dtype)
  w_bits = jax.dtypes.itemsize_bits(w.dtype)
  # When using bf16 for x and w, 128 is the largest tile_m value that is safe
  # to use for most scenarios. But if lower bitwidth is used, we need to tweak
  # tile_m to account for using faster hardware unit.
  bucket_base = 128 // min(pl.cdiv(16, w_bits), 2)
  if is_mxu_quantized:
    bucket_base *= 2
  if plan.tile_m is not None:
    return min(bucket_base, plan.tile_m), False
  if not spec.has_routed_io:
    return min(bucket_base, max_group_rows), False
  pair_tile_m = 2 * min_tile_m
  if (
      spec.x.row_size <= pair_tile_m
      or spec.num_global_groups <= 0
      or spec.size_out_m <= 0
  ):
    return min(bucket_base, max_group_rows), True
  # Align to `2 * num_sublanes` so 2-packed compact VMEM loads avoid crosslane
  # shuffles. `9/8 * mean + min_tile_m` was tuned empirically (comparing `{9/8,
  # 8/7, 7/6, 5/4}` up to 4x skew and `9/8` vs `1.5` on `gmm_prod_test`
  # 128-512-expert suites): `min_tile_m` covers small-mean variance while `9/8`
  # drops steps tightly (`mean=20`->`32`, `mean=80`->`112`, `mean=160`->`192`
  # instead of `48`/`208` at `5/4` or `128`/`240` at `1.5`).
  scaled_mean = pl.cdiv(9 * spec.size_out_m, 8 * spec.num_global_groups)
  dma_floor = _dma_floor(plan, spec=spec, min_tile_m=min_tile_m)
  target = pair_tile_m * pl.cdiv(
      max(dma_floor, scaled_mean + min_tile_m), pair_tile_m
  )
  # When `target` already spans at least two `bucket_base` buckets (`>= 256`
  # rows) or expert weights are large (`>= 12 MiB`), keep
  # `[bucket_base, 2 * bucket_base]` so multi-tile groups avoid reloading large
  # weights and short tail tiles halve MXU/DMA work. For `< 12 MiB` weights
  # strictly between `bucket_base` and `2 * bucket_base`, a single exact tile
  # avoids both `2 * bucket_base` padding and `lax.switch` overhead.
  if target > bucket_base and (
      target >= 2 * bucket_base or _is_large_expert_weight(spec)
  ):
    return min(bucket_base, max_group_rows), False
  floor_m = min(max_group_rows, pair_tile_m)
  ceil_m = min(256, max_group_rows)
  return max(floor_m, min(ceil_m, target)), True


def _resolve_w_tile_n(
    w_plan_tile_n: int | None,
    *,
    w_out_n: int,
    out_chunk_cols: int,
    min_tile_m: int,
    estimate_vmem: Callable[..., int],
    vmem_limit_bytes: int,
) -> int:
  """Splits `w`'s output width into the fewest whole-chunk tiles in VMEM."""
  if w_plan_tile_n is not None:
    return w_plan_tile_n
  num_n_tiles = 1
  w_tile_n = w_out_n
  max_num_n_tiles = w_out_n // out_chunk_cols
  while (
      estimate_vmem(tile_m=min_tile_m, tile_n=w_tile_n) > vmem_limit_bytes
      and num_n_tiles < max_num_n_tiles
  ):
    num_n_tiles += 1
    # w_tile_n must divide w_out_n into whole compact chunks.
    if w_out_n % (num_n_tiles * out_chunk_cols) == 0:
      w_tile_n = w_out_n // num_n_tiles
  return w_tile_n


def _resolve_tile_m(
    plan: RoutingGmmPlan,
    *,
    bucket_base: int,
    single_routed_bucket: bool,
    max_group_rows: int,
    w_tile_n: int,
    estimate_vmem: Callable[..., int],
    vmem_limit_bytes: int,
) -> int:
  """Resolves `tile_m` from user plan or VMEM-fitting bucket multiples."""
  if plan.tile_m is not None:
    return plan.tile_m
  # BranchlessBufferedRef unrolls `tile_m` row DMAs regardless of bucket, so a
  # single bucket avoids `lax.switch` code duplication and VREG pressure. For
  # multi-bucket IO, `max_num_buckets=2` limits instruction-memory growth.
  max_tile_m, max_num_buckets = _tile_m_limits(plan)
  if single_routed_bucket:
    max_num_buckets = 1
  tile_m = bucket_base
  for num_buckets in range(2, max_num_buckets + 1):
    candidate = num_buckets * bucket_base
    if (
        candidate > max_tile_m
        or estimate_vmem(tile_m=candidate, tile_n=w_tile_n) > vmem_limit_bytes
    ):
      break
    tile_m = candidate
  if plan.bucket_base is None:
    tile_m = min(tile_m, max_group_rows)
  return tile_m


def _expert_weight_bytes(spec: gmm_spec_lib.RoutingGmmSpec) -> int:
  """Returns HBM bytes of one expert's weight tensor(s)."""
  w = spec.w
  total = w.in_size * w.out_size * jax.dtypes.itemsize_bits(w.dtype) // 8
  if spec.w2 is not None:
    w2 = spec.w2
    total += w2.in_size * w2.out_size * jax.dtypes.itemsize_bits(w2.dtype) // 8
  return total


def _is_large_expert_weight(spec: gmm_spec_lib.RoutingGmmSpec) -> bool:
  """Whether one expert's weights are large enough to dominate HBM DMA."""
  # Expert weights >= 12 MiB cap prefetch at 3 buffers (`_w_buffer_range`) and
  # keep `[bucket_base, 2 * bucket_base]` (`_resolve_bucket_base`) so overflow
  # tiles do not reload large weights from HBM.
  return _expert_weight_bytes(spec) >= 12 * 1024 * 1024


def _resolve_weight_lookahead(
    plan: RoutingGmmPlan,
    *,
    spec: gmm_spec_lib.RoutingGmmSpec,
    tile_m: int,
    num_w_buffers: int,
) -> bool:
  """Resolves whether weight BufferedRefs scan ahead across expert tiles."""
  if plan.weight_lookahead is not None:
    return plan.weight_lookahead
  if spec.size_local_groups <= 1 or spec.x.row_size <= tile_m:
    return False
  if (
      (spec.gather is not None or spec.scatter is not None)
      and spec.num_global_groups > 0
      and num_w_buffers >= 4
  ):
    # TODO: Per-row routed DMAs saturate the scalar ALU when
    # BufferedRef lookahead scans run on short groups; when fewer than 4 weight
    # buffers fit, keep lookahead enabled so multi-tile hot groups never starve
    # next-expert prefetch.
    return spec.size_out_m > 4 * tile_m * spec.num_global_groups
  return True


def _grow_num_w_buffers(
    plan: RoutingGmmPlan,
    *,
    spec: gmm_spec_lib.RoutingGmmSpec,
    num_w_buffers: int,
    tile_m: int,
    w_tile_n: int,
    estimate_vmem: Callable[..., int],
    vmem_limit_bytes: int,
) -> int:
  """Grows weight buffer count into spare VMEM unless the plan fixes it."""
  if plan.num_w_buffers is not None:
    return num_w_buffers
  _, max_num_w_buffers = _w_buffer_range(plan, spec=spec, tile_m=tile_m)
  while num_w_buffers < max_num_w_buffers and (
      estimate_vmem(
          tile_m=tile_m, tile_n=w_tile_n, num_w_buffers=num_w_buffers + 1
      )
      <= vmem_limit_bytes
  ):
    num_w_buffers += 1
  return num_w_buffers


def calculate_tiling(
    plan: RoutingGmmPlan,
    *,
    spec: gmm_spec_lib.RoutingGmmSpec,
    matmul_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    out_dtype: jnp.dtype | None = None,
    out_quant: out_format_lib.ResolvedQuantFormat | None = None,
    num_stash_buffers: int | None = None,
    stash_dtype: jnp.dtype | None = None,
    vmem_limit_bytes: int,
    tpu_info: pltpu.TpuInfo,
) -> Tiling:
  """Resolves tile sizes and buffer counts of the unfused routing GMM.

  Args:
    plan: User plan. Fields that are set are kept.
    spec: Validated input specifications and operator semantics.
    matmul_dtype: Dtype in which inputs are fed to the MXU.
    acc_dtype: Accumulator dtype.
    out_dtype: Output dtype of the kernel, or None to default to `spec.x.dtype`.
    out_quant: Optional output quantization format.
    num_stash_buffers: Stash buffer count when `out_quant` is set, else None.
    stash_dtype: Stash VMEM dtype when `out_quant` is set, else None.
    vmem_limit_bytes: VMEM budget of the kernel.
    tpu_info: TPU the kernel runs on.

  Returns:
    The tiling of the unfused kernel.
  """
  x, w = spec.x, spec.w
  # Step 1: Set default values.
  min_tile_m = tpu_info.num_sublanes
  max_group_rows = _max_group_rows(spec, tpu_info)
  num_xo_buffers = plan.num_xo_buffers or 3
  num_w_buffers = plan.num_w_buffers or 2
  # TODO: Support K dim tiling.
  tile_k = plan.tile_k or spec.size_k
  bucket_base, single_routed_bucket = _resolve_bucket_base(
      plan,
      spec=spec,
      matmul_dtype=matmul_dtype,
      max_group_rows=max_group_rows,
      min_tile_m=min_tile_m,
  )

  # Size w's output tile against size_k2 when w2 is present, or size_out_n when
  # w is the only weight.
  if spec.size_k2 is None:
    w_out_n = spec.size_out_n
    w_plan_tile_n = plan.tile_n
    w_out_dtype = x.dtype if out_dtype is None else out_dtype
    w_out_quant = out_quant
    w_num_stash_buffers, w_stash_dtype = num_stash_buffers, stash_dtype
  else:
    w_out_n = spec.size_k2
    w_plan_tile_n = plan.tile_k2
    w_out_dtype = x.dtype
    w_out_quant = None
    w_num_stash_buffers, w_stash_dtype = None, None

  out_chunk_cols = layout_util.get_packing(w_out_dtype) * tpu_info.num_lanes

  # Step 2: Bind VMEM calculation function to static problem parameters.
  estimate_vmem = functools.partial(
      _vmem_estimate,
      x=x,
      w=w,
      is_fused_act=spec.is_fused_act,
      matmul_dtype=matmul_dtype,
      acc_dtype=acc_dtype,
      out_dtype=w_out_dtype,
      out_quant=w_out_quant,
      tile_k=tile_k,
      num_xo_buffers=num_xo_buffers,
      num_w_buffers=num_w_buffers,
      num_stash_buffers=w_num_stash_buffers,
      stash_dtype=w_stash_dtype,
      tpu_info=tpu_info,
  )

  # Split w's output width into the fewest tiles that fit VMEM. This is chosen
  # before tile_m because the weight block usually dominates VMEM.
  w_tile_n = _resolve_w_tile_n(
      w_plan_tile_n,
      w_out_n=w_out_n,
      out_chunk_cols=out_chunk_cols,
      min_tile_m=plan.tile_m or bucket_base,
      estimate_vmem=estimate_vmem,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  tile_m = _resolve_tile_m(
      plan,
      bucket_base=bucket_base,
      single_routed_bucket=single_routed_bucket,
      max_group_rows=max_group_rows,
      w_tile_n=w_tile_n,
      estimate_vmem=estimate_vmem,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  if plan.bucket_base is None:
    bucket_base = min(bucket_base, tile_m)

  # Spend spare VMEM on deeper weight prefetch.
  num_w_buffers = _grow_num_w_buffers(
      plan,
      spec=spec,
      num_w_buffers=num_w_buffers,
      tile_m=tile_m,
      w_tile_n=w_tile_n,
      estimate_vmem=estimate_vmem,
      vmem_limit_bytes=vmem_limit_bytes,
  )

  estimate = estimate_vmem(
      tile_m=tile_m, tile_n=w_tile_n, num_w_buffers=num_w_buffers
  )
  if estimate > vmem_limit_bytes:
    raise ValueError(
        f"Tiles ({tile_m=}, {tile_k=}, tile_n={w_tile_n}) need an estimated"
        f" {estimate} bytes of VMEM, exceeding the budget of {vmem_limit_bytes}"
    )

  if spec.size_k2 is None:
    tile_k2 = None
    tile_n = w_tile_n
  else:
    tile_k2 = w_tile_n
    tile_n = plan.tile_n or spec.size_out_n

  return Tiling(
      tile_m=tile_m,
      tile_k=tile_k,
      tile_k2=tile_k2,
      tile_n=tile_n,
      bucket_base=bucket_base,
      num_xo_buffers=num_xo_buffers,
      num_w_buffers=num_w_buffers,
  )
