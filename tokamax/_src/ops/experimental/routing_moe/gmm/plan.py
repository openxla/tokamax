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
from typing import Any, Self

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
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
  # Accumulator dtype across quantization blocks and K tiles.
  acc_dtype: jax.typing.DTypeLike | None = None
  # Dtype in which inputs are fed to the MXU. Performs dynamic quantization if
  # the dtype does not match with storage dtype.
  matmul_dtype: jax.typing.DTypeLike | None = None
  # Block size if dynamic quantization is performed.
  x_quant_block_size: int | None = None
  # Whether gate/up and down projections run in one fused kernel when w2 is set.
  fuse_up_and_down: bool | None = None
  vmem_limit_bytes: int | None = None
  # Walk only the groups with rows when building the metadata (needs the
  # row_group input of routing_gmm).
  live_group_walk: bool | None = None
  # These fields bound the heuristic search and are omitted once resolved.
  vmem_usage_fraction: float | None = None
  # Upper bound when num_w_buffers is grown into spare VMEM, so that the next
  # groups' weight DMAs stay in flight during the current group's compute.
  # TODO: The default of 3 was chosen empirically (on decode tiles
  # 3 buffers were faster than 2, 4 were even with 3 and 6 were slower).
  # Revisit it based on per-group DMA vs. compute time.
  max_num_w_buffers: int | None = None


@dataclass_util.resolves_from(
    RoutingGmmPlan,
    dropped=("vmem_usage_fraction", "max_num_w_buffers"),
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
  acc_dtype: jnp.dtype
  matmul_dtype: jnp.dtype
  # It is None unless x is dynamically quantized.
  x_quant_block_size: int | None
  fuse_up_and_down: bool
  vmem_limit_bytes: int
  live_group_walk: bool

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

    if spec.w2 is None and plan.tile_k2 is not None:
      raise ValueError("tile_k2 requires w2 to be set.")
    if spec.w2 is None and plan.fuse_up_and_down:
      raise ValueError("fuse_up_and_down=True requires w2 to be set.")

    vmem_usage_fraction = plan.vmem_usage_fraction or 0.9
    vmem_limit_bytes = plan.vmem_limit_bytes or tpu_info.vmem_capacity_bytes
    capped_vmem_limit_bytes = int(vmem_usage_fraction * vmem_limit_bytes)
    tiling_fn = functools.partial(
        calculate_tiling,
        plan,
        spec=spec,
        matmul_dtype=matmul_dtype,
        acc_dtype=acc_dtype,
        out_dtype=out_format.dtype,
        out_quant=out_format.quant,
        vmem_limit_bytes=capped_vmem_limit_bytes,
        tpu_info=tpu_info,
    )
    can_fuse = None
    if spec.w2 is not None:
      can_fuse = functools.partial(
          can_fuse_up_and_down,
          spec=spec,
          matmul_dtype=matmul_dtype,
          acc_dtype=acc_dtype,
          out_dtype=out_format.dtype,
          out_quant=out_format.quant,
          vmem_limit_bytes=capped_vmem_limit_bytes,
          tpu_info=tpu_info,
      )
    tiling, fuse_up_and_down = _choose_tiling_and_fusion(
        plan, tiling_fn=tiling_fn, can_fuse=can_fuse
    )
    (
        tile_m,
        tile_k,
        tile_k2,
        tile_n,
        bucket_base,
        num_xo_buffers,
        num_w_buffers,
    ) = tiling
    if not fuse_up_and_down and plan.tile_k2 is not None:
      raise ValueError("tile_k2 requires fuse_up_and_down=True.")

    return cls(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_k2=tile_k2,
        tile_n=tile_n,
        bucket_base=bucket_base,
        num_xo_buffers=num_xo_buffers,
        num_w_buffers=num_w_buffers,
        acc_dtype=acc_dtype,
        matmul_dtype=matmul_dtype,
        x_quant_block_size=x_quant_block_size,
        fuse_up_and_down=fuse_up_and_down,
        vmem_limit_bytes=vmem_limit_bytes,
        live_group_walk=live_group_walk,
    )


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


def _resolve_dtypes(
    plan: RoutingGmmPlan,
    *,
    x: tensor_spec.ActivationSpec,
    w: tensor_spec.WeightSpec,
    tpu_info: pltpu.TpuInfo,
) -> tuple[jnp.dtype, jnp.dtype, int | None]:
  """Resolves execution dtypes and `x` quantization block size."""
  # A pre-quantized x already fixes the MXU dtype. Otherwise w's storage
  # dtype is used so that both operands enter the MXU in the same dtype.
  if plan.matmul_dtype is not None:
    matmul_dtype = jnp.dtype(plan.matmul_dtype)
  elif x.quant is None and w.quant is not None:
    matmul_dtype = w.dtype
    # matmul_dtype is the dtype x is converted to right before it goes
    # into the MXU. See RoutingGmmConfig.create(). compute.py converts the
    # weight dtype to matmul_dtype before the matmul if
    # is_matmul_supported(matmul_dtype, w.dtype) is false.
    # We pick the narrowest x dtype that satisfies is_matmul_supported.
    if jnp.issubdtype(w.dtype, jnp.floating):
      for dtype in (jnp.float8_e4m3fn, jnp.float8_e4m3b11fnuz):
        dtype = jnp.dtype(dtype)
        if tpu_info.is_matmul_supported(dtype, w.dtype) or (
            _fits_in(w.dtype, dtype)
            and tpu_info.is_matmul_supported(dtype, dtype)
        ):
          matmul_dtype = dtype
          break
      else:
        raise NotImplementedError(
            f"No supported fp8 activation dtype for {w.dtype} weights on"
            f" {tpu_info.chip_version}"
        )
  else:
    matmul_dtype = x.dtype

  x_quant_block_size = plan.x_quant_block_size
  if x_quant_block_size is None and matmul_dtype != x.dtype:
    # TODO: Choose different block size based on HW.
    x_quant_block_size = 512

  if plan.acc_dtype is None:
    acc_dtype = jnp.bfloat16.dtype
  else:
    acc_dtype = jnp.dtype(plan.acc_dtype)
  return matmul_dtype, acc_dtype, x_quant_block_size


def _choose_tiling_and_fusion(
    plan: RoutingGmmPlan,
    *,
    tiling_fn: Callable[..., tuple[Any, ...]],
    can_fuse: Callable[..., bool] | None,
) -> tuple[tuple[Any, ...], bool]:
  """Picks the tiling and whether the gate/up and down projections are fused.

  Args:
    plan: User plan.
    tiling_fn: `calculate_tiling` with the problem bound. Takes
      `allow_vmem_growth`.
    can_fuse: `can_fuse_up_and_down` with everything but the tiling bound, or
      None if there is no down projection.

  Returns:
    The tiling as returned by `calculate_tiling`, and whether to fuse.
  """
  tiling = tiling_fn()
  # TODO: fusion-or-not, number of buffers for weights and
  # activations, and tiling should be decided together by a single function
  # with VMEM limit instead of separately.
  if can_fuse is None:
    return tiling, False
  if plan.fuse_up_and_down is not None and not plan.fuse_up_and_down:
    return tiling, False

  def fits(tiling: tuple[Any, ...]) -> bool:
    tile_m, tile_k, tile_k2, tile_n, _, num_xo_buffers, num_w_buffers = tiling
    return can_fuse(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_k2=tile_k2,
        tile_n=tile_n,
        num_xo_buffers=num_xo_buffers,
        num_w_buffers=num_w_buffers,
    )

  if fits(tiling):
    return tiling, True
  # calculate_tiling only budgets w, so heuristics that use more VMEM than the
  # default tiling (e.g. more weight buffers) can push the fused kernel past
  # VMEM. Losing fusion costs more than they save, so keep the default tiling
  # if it fuses, or if fusion is requested.
  default_tiling = tiling_fn(allow_vmem_growth=False)
  if plan.fuse_up_and_down or fits(default_tiling):
    return default_tiling, True
  return tiling, False


def _vmem_estimate(
    *,
    x: tensor_spec.ActivationSpec,
    w: tensor_spec.WeightSpec,
    w2: tensor_spec.WeightSpec | None = None,
    is_fused_act: bool = False,
    matmul_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    out_quant: out_format_lib.ResolvedQuantFormat | None = None,
    tile_m: int,
    tile_k: int,
    tile_k2: int | None = None,
    tile_n: int,
    num_xo_buffers: int,
    num_w_buffers: int,
    tpu_info: pltpu.TpuInfo,
) -> int:
  """Estimates VMEM usage in bytes for routing GMM."""
  is_dynamic_quantized = matmul_dtype != x.dtype
  matmul_bits = jax.dtypes.itemsize_bits(matmul_dtype)
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
    stash_ratio = max(layout_util.get_packing(out_dtype) // 2, 1)
    out_vmem += 2 * tile_m * tile_n * 2
    out_vmem += 2 * (1 + stash_ratio) * tile_m * tpu_info.num_lanes * 4

  def _matmul_vmem(
      weight: tensor_spec.WeightSpec, k_tile: int, n_tile: int, num_w_refs: int
  ) -> int:
    w_bits = jax.dtypes.itemsize_bits(weight.dtype)
    w_vmem = num_w_refs * num_w_buffers * k_tile * n_tile * w_bits // 8
    scale_vmem = 0
    if weight.quant is not None:
      num_blocks = pl.cdiv(k_tile, weight.quant.block_size)
      scale_vmem = num_w_refs * num_w_buffers * num_blocks * n_tile * 4
    acc_vmem = num_w_refs * tile_m * n_tile * acc_bits // 8
    # Dynamic quantization materializes a quantized copy of the x tile.
    if is_dynamic_quantized:
      acc_vmem += tile_m * k_tile * matmul_bits // 8
    return w_vmem + scale_vmem + acc_vmem

  num_w1_refs = 2 if is_fused_act else 1
  if w2 is None:
    matmul_vmem = _matmul_vmem(w, tile_k, tile_n, num_w1_refs)
  else:
    assert tile_k2 is not None
    matmul_vmem = _matmul_vmem(w, tile_k, tile_k2, num_w1_refs)
    matmul_vmem += _matmul_vmem(w2, tile_k2, tile_n, 1)
    matmul_vmem += tile_m * tile_k2 * jax.dtypes.itemsize_bits(x.dtype) // 8

  return x_vmem + out_vmem + matmul_vmem


def _resolve_bucket_base(
    plan: RoutingGmmPlan,
    *,
    w: tensor_spec.WeightSpec,
    matmul_dtype: jnp.dtype,
    max_group_rows: int,
) -> int:
  """Resolves `bucket_base` from user plan or bitwidth heuristics."""
  if plan.bucket_base is not None:
    return plan.bucket_base
  is_mxu_quantized = matmul_dtype not in (jnp.bfloat16.dtype, jnp.float32.dtype)
  w_bits = jax.dtypes.itemsize_bits(w.dtype)
  # When using bf16 for x and w, 128 is the largest tile_m value that is safe
  # to use for most scenarios. But if lower bitwidth is used, we need to tweak
  # tile_m to account for using faster hardware unit.
  bucket_base = 128 // min(pl.cdiv(16, w_bits), 2)
  if is_mxu_quantized:
    bucket_base *= 2
  return min(bucket_base, plan.tile_m or max_group_rows)


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
    max_group_rows: int,
    w_tile_n: int,
    estimate_vmem: Callable[..., int],
    vmem_limit_bytes: int,
) -> int:
  """Resolves `tile_m` from user plan or VMEM-fitting bucket multiples."""
  if plan.tile_m is not None:
    return plan.tile_m
  # TODO: Lift the 256 cap once larger tiles are validated.
  max_tile_m = 256
  # TODO: max number of bucket was choosen empiirically.
  # Revisit the value to be based on number of instruction memory size.
  max_num_buckets = 2
  # Largest multiple of bucket_base that stays within max_tile_m and VMEM.
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


def _grow_num_w_buffers(
    plan: RoutingGmmPlan,
    *,
    num_w_buffers: int,
    allow_vmem_growth: bool,
    tile_m: int,
    w_tile_n: int,
    estimate_vmem: Callable[..., int],
    vmem_limit_bytes: int,
) -> int:
  """Grows weight buffer count into spare VMEM when permitted."""
  if plan.num_w_buffers is not None or not allow_vmem_growth:
    return num_w_buffers
  max_num_w_buffers = plan.max_num_w_buffers or 3
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
    vmem_limit_bytes: int,
    tpu_info: pltpu.TpuInfo,
    allow_vmem_growth: bool = True,
) -> tuple[int, int, int | None, int, int, int, int]:
  """Resolves tile sizes and buffer counts for routing GMM.

  Args:
    plan: User plan. Fields that are set are kept.
    spec: Validated input specifications and operator semantics.
    matmul_dtype: Dtype in which inputs are fed to the MXU.
    acc_dtype: Accumulator dtype.
    out_dtype: Output dtype of the kernel, or None to default to `spec.x.dtype`.
    out_quant: Optional output quantization format.
    vmem_limit_bytes: VMEM budget of the kernel.
    tpu_info: TPU the kernel runs on.
    allow_vmem_growth: Whether heuristics may pick settings that use more VMEM
      than the default tiling (e.g. more weight buffers).

  Returns:
    (tile_m, tile_k, tile_k2, tile_n, bucket_base, num_xo_buffers,
    num_w_buffers).
  """
  x, w = spec.x, spec.w
  # Step 1: Set default values.
  # A group gets at most one row per input token, so a tile never needs more
  # rows than there are tokens (e.g. decode). Tiles stay a whole number of
  # sublane tiles of x and, with dynamic quantization, of x in matmul_dtype.
  min_tile_m = max(
      tpu_info.get_sublane_tiling(x.dtype),
      tpu_info.get_sublane_tiling(matmul_dtype),
  )
  max_group_rows = min_tile_m * pl.cdiv(x.row_size, min_tile_m)
  num_xo_buffers = plan.num_xo_buffers or 3
  num_w_buffers = plan.num_w_buffers or 2
  # TODO: Support K dim tiling.
  tile_k = plan.tile_k or spec.size_k
  bucket_base = _resolve_bucket_base(
      plan, w=w, matmul_dtype=matmul_dtype, max_group_rows=max_group_rows
  )

  # Size w's output tile against size_k2 when w2 is present, or size_out_n when
  # w is the only weight.
  if spec.size_k2 is None:
    w_out_n = spec.size_out_n
    w_plan_tile_n = plan.tile_n
    w_out_dtype = x.dtype if out_dtype is None else out_dtype
    w_out_quant = out_quant
  else:
    w_out_n = spec.size_k2
    w_plan_tile_n = plan.tile_k2
    w_out_dtype = x.dtype
    w_out_quant = None

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
      max_group_rows=max_group_rows,
      w_tile_n=w_tile_n,
      estimate_vmem=estimate_vmem,
      vmem_limit_bytes=vmem_limit_bytes,
  )

  # Spend spare VMEM on deeper weight prefetch.
  num_w_buffers = _grow_num_w_buffers(
      plan,
      num_w_buffers=num_w_buffers,
      allow_vmem_growth=allow_vmem_growth,
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

  return (
      tile_m,
      tile_k,
      tile_k2,
      tile_n,
      bucket_base,
      num_xo_buffers,
      num_w_buffers,
  )


def can_fuse_up_and_down(
    *,
    spec: gmm_spec_lib.RoutingGmmSpec,
    matmul_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    out_quant: out_format_lib.ResolvedQuantFormat | None = None,
    tile_m: int,
    tile_k: int,
    tile_k2: int,
    tile_n: int,
    num_xo_buffers: int,
    num_w_buffers: int,
    vmem_limit_bytes: int,
    tpu_info: pltpu.TpuInfo,
) -> bool:
  """Whether gate/up and down projections should run as one fused kernel."""
  x, w, w2 = spec.x, spec.w, spec.w2
  assert w2 is not None
  if w.dtype != w2.dtype or (w.quant is None) != (w2.quant is None):
    return False
  # Fused execution does not support weight tiling along K, K2, or N.
  if tile_k != x.col_size or tile_k2 != w2.in_size or tile_n != w2.out_size:
    return False
  estimate = _vmem_estimate(
      x=x,
      w=w,
      w2=w2,
      is_fused_act=spec.is_fused_act,
      matmul_dtype=matmul_dtype,
      acc_dtype=acc_dtype,
      out_dtype=out_dtype,
      out_quant=out_quant,
      tile_m=tile_m,
      tile_k=tile_k,
      tile_k2=tile_k2,
      tile_n=tile_n,
      num_xo_buffers=num_xo_buffers,
      num_w_buffers=num_w_buffers,
      tpu_info=tpu_info,
  )
  return estimate <= vmem_limit_bytes
