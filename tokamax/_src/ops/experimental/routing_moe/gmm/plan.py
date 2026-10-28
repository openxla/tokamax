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

"""Tunable parameters of the routing GMM kernel.

Plan fields affect performance and numerics but not the output shape, dtype,
or placement. Unset fields are resolved by heuristics.
"""

import dataclasses
import functools
from typing import Self

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import layout_util


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
  vmem_usage_fraction: float | None = None
  # Size of gap to try skipping when writing to metadata.
  skip_metadata_size: int | None = None
  # Walk only the groups with rows when building the metadata (needs the
  # row_group input of routing_gmm).
  live_group_walk: bool | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedGmmPlan(RoutingGmmPlan):
  """Plan with None fields from user-provided plan resolved by heuristic."""

  tile_m: int
  tile_k: int
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
  skip_metadata_size: int
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
      x: arg.ActivationSpec,
      w: arg.WeightSpec,
      w2: arg.WeightSpec | None = None,
      is_fused_act: bool = False,
      out_dtype: jax.typing.DTypeLike | None = None,
      size_out_m: int,
      size_global_groups: int,
      size_k: int,
      size_k2: int | None = None,
      size_out_n: int,
      tpu_info: pltpu.TpuInfo,
  ) -> Self:
    """Fills unset plan fields with heuristics."""
    if plan is None:
      plan = RoutingGmmPlan()

    # A pre-quantized x already fixes the MXU dtype. Otherwise w's storage
    # dtype is used so that both operands enter the MXU in the same dtype.
    if plan.matmul_dtype is not None:
      matmul_dtype = jnp.dtype(plan.matmul_dtype)
    elif x.quant is None and w.quant is not None:
      # TODO: Support mixed precision and choose by hardware support.
      matmul_dtype = w.dtype
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

    skip_metadata_size = plan.skip_metadata_size
    if skip_metadata_size is None:
      # Use the largest power of 2 <= expected gap between touched experts.
      gap = max(1, size_global_groups // size_out_m)
      skip_metadata_size = (1 << (gap.bit_length() - 1)) - 1
    if skip_metadata_size < 0:
      raise ValueError(
          f"skip_metadata_size must be >= 0. Got {skip_metadata_size}."
      )

    live_group_walk = plan.live_group_walk
    if live_group_walk is None:
      live_group_walk = size_out_m < w.group_size

    if w2 is None:
      if plan.tile_k2 is not None:
        raise ValueError("tile_k2 requires w2 to be set.")
      if plan.fuse_up_and_down:
        raise ValueError("fuse_up_and_down=True requires w2 to be set.")

    vmem_usage_fraction = plan.vmem_usage_fraction or 0.9
    vmem_limit_bytes = plan.vmem_limit_bytes or tpu_info.vmem_capacity_bytes
    capped_vmem_limit_bytes = int(vmem_usage_fraction * vmem_limit_bytes)
    (
        tile_m,
        tile_k,
        tile_k2,
        tile_n,
        bucket_base,
        num_xo_buffers,
        num_w_buffers,
    ) = calculate_tiling(
        plan,
        x=x,
        w=w,
        is_fused_act=is_fused_act,
        matmul_dtype=matmul_dtype,
        acc_dtype=acc_dtype,
        size_k=size_k,
        size_k2=size_k2,
        size_out_n=size_out_n,
        vmem_limit_bytes=capped_vmem_limit_bytes,
        tpu_info=tpu_info,
    )
    # TODO: fusion-or-not, number of buffers for weights and
    # activations, and tiling should be decided together by a single function
    # with VMEM limit instead of separately.
    if w2 is None:
      fuse_up_and_down = False
    elif plan.fuse_up_and_down is not None:
      fuse_up_and_down = plan.fuse_up_and_down
    else:
      assert tile_k2 is not None
      fuse_up_and_down = can_fuse_up_and_down(
          x=x,
          w=w,
          w2=w2,
          is_fused_act=is_fused_act,
          matmul_dtype=matmul_dtype,
          acc_dtype=acc_dtype,
          out_dtype=x.dtype if out_dtype is None else jnp.dtype(out_dtype),
          tile_m=tile_m,
          tile_k=tile_k,
          tile_k2=tile_k2,
          tile_n=tile_n,
          num_xo_buffers=num_xo_buffers,
          num_w_buffers=num_w_buffers,
          vmem_limit_bytes=capped_vmem_limit_bytes,
          tpu_info=tpu_info,
      )
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
        skip_metadata_size=skip_metadata_size,
        live_group_walk=live_group_walk,
    )


def _vmem_estimate(
    *,
    x: arg.ActivationSpec,
    w: arg.WeightSpec,
    w2: arg.WeightSpec | None = None,
    is_fused_act: bool = False,
    matmul_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
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
  x_chunks = layout_util.to_bank_conflict_aware_stride(x.compact_shape[1])
  out_shape = layout_util.to_compact_shape((tile_m, tile_n), out_dtype)
  out_chunks = layout_util.to_bank_conflict_aware_stride(out_shape[1])
  x_vmem = num_xo_buffers * tile_m * x_chunks * chunk_bytes
  out_vmem = num_xo_buffers * tile_m * out_chunks * chunk_bytes

  def _matmul_vmem(
      weight: arg.WeightSpec, k_tile: int, n_tile: int, num_w_refs: int
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


def calculate_tiling(
    plan: RoutingGmmPlan,
    *,
    x: arg.ActivationSpec,
    w: arg.WeightSpec,
    is_fused_act: bool = False,
    matmul_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    size_k: int,
    size_k2: int | None = None,
    size_out_n: int,
    vmem_limit_bytes: int,
    tpu_info: pltpu.TpuInfo,
) -> tuple[int, int, int | None, int, int, int, int]:
  """Resolves tile sizes and buffer counts for routing GMM."""
  # Slightly lower vmem limit than total required amount to be conservative.

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
  tile_k = plan.tile_k or size_k
  is_mxu_quantized = matmul_dtype not in (jnp.bfloat16.dtype, jnp.float32.dtype)
  w_bits = jax.dtypes.itemsize_bits(w.dtype)

  if plan.bucket_base is not None:
    bucket_base = plan.bucket_base
  else:
    # When using bf16 for x and w, 128 is the largest tile_m value that is safe
    # to use for most scenarios. But if lower bitwidth is used, we need to tweak
    # tile_m to account for using faster hardware unit.
    bucket_base = 128 // min(pl.cdiv(16, w_bits), 2)
    if is_mxu_quantized:
      bucket_base *= 2
    bucket_base = min(bucket_base, plan.tile_m or max_group_rows)

  # Size w's output tile against size_k2 when w2 is present, or size_out_n when
  # w is the only weight.
  w_out_n = size_out_n if size_k2 is None else size_k2
  w_plan_tile_n = plan.tile_n if size_k2 is None else plan.tile_k2

  # out_dtype is not known to the plan, so x's dtype is used as the estimate.
  # TODO: Add ability to pass out_dtype.
  out_shape = layout_util.to_compact_shape((1, w_out_n), x.dtype)
  out_chunk_cols = out_shape[2] * out_shape[3]

  # Step 2: Bind VMEM calculation function to static problem parameters.
  estimate_vmem = functools.partial(
      _vmem_estimate,
      x=x,
      w=w,
      is_fused_act=is_fused_act,
      matmul_dtype=matmul_dtype,
      acc_dtype=acc_dtype,
      out_dtype=x.dtype,
      tile_k=tile_k,
      num_xo_buffers=num_xo_buffers,
      num_w_buffers=num_w_buffers,
      tpu_info=tpu_info,
  )

  # Split w's output width into the fewest tiles that fit VMEM. This is chosen
  # before tile_m because the weight block usually dominates VMEM.
  if w_plan_tile_n is not None:
    w_tile_n = w_plan_tile_n
  else:
    num_n_tiles = 1
    w_tile_n = w_out_n
    max_num_n_tiles = w_out_n // out_chunk_cols
    min_tile_m = plan.tile_m or bucket_base
    while (
        estimate_vmem(tile_m=min_tile_m, tile_n=w_tile_n) > vmem_limit_bytes
        and num_n_tiles < max_num_n_tiles
    ):
      num_n_tiles += 1
      # w_tile_n must divide w_out_n into whole compact chunks.
      if w_out_n % (num_n_tiles * out_chunk_cols) == 0:
        w_tile_n = w_out_n // num_n_tiles

  if plan.tile_m is not None:
    tile_m = plan.tile_m
  else:
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

  estimate = estimate_vmem(tile_m=tile_m, tile_n=w_tile_n)
  if estimate > vmem_limit_bytes:
    raise ValueError(
        f"Tiles ({tile_m=}, {tile_k=}, tile_n={w_tile_n}) need an estimated"
        f" {estimate} bytes of VMEM, exceeding the budget of {vmem_limit_bytes}"
    )

  if size_k2 is None:
    tile_k2 = None
    tile_n = w_tile_n
  else:
    tile_k2 = w_tile_n
    tile_n = plan.tile_n or size_out_n

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
    x: arg.ActivationSpec,
    w: arg.WeightSpec,
    w2: arg.WeightSpec,
    is_fused_act: bool = True,
    matmul_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
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
  if w.dtype != w2.dtype or (w.quant is None) != (w2.quant is None):
    return False
  # Fused execution does not support weight tiling along K, K2, or N.
  if tile_k != x.col_size or tile_k2 != w2.in_size or tile_n != w2.out_size:
    return False
  estimate = _vmem_estimate(
      x=x,
      w=w,
      w2=w2,
      is_fused_act=is_fused_act,
      matmul_dtype=matmul_dtype,
      acc_dtype=acc_dtype,
      out_dtype=out_dtype,
      tile_m=tile_m,
      tile_k=tile_k,
      tile_k2=tile_k2,
      tile_n=tile_n,
      num_xo_buffers=num_xo_buffers,
      num_w_buffers=num_w_buffers,
      tpu_info=tpu_info,
  )
  return estimate <= vmem_limit_bytes
