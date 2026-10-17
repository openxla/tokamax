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

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import layout_util


@dataclasses.dataclass(frozen=True, kw_only=True)
class RoutingGmmPlan:
  """User provided plan where `None` field will be determined by heuristic."""

  tile_m: int | None = None
  tile_k: int | None = None
  tile_n: int | None = None
  bucket_base: int | None = None
  num_xo_buffers: int | None = None
  num_w_buffers: int | None = None
  # Accumulator dtype across quantization blocks and K tiles.
  acc_dtype: jnp.dtype | None = None
  # Dtype in which inputs are fed to the MXU. Performs dynamic quantization if
  # the dtype does not match with storage dtype.
  matmul_dtype: jnp.dtype | None = None
  # Block size if dynamic quantization is performed.
  x_quant_block_size: int | None = None
  vmem_limit_bytes: int | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedPlan(RoutingGmmPlan):
  """Plan that used heuristic to resolve `None` fields from user's plan."""

  tile_m: int
  tile_k: int
  tile_n: int
  bucket_base: int
  num_xo_buffers: int
  num_w_buffers: int
  acc_dtype: jnp.dtype
  matmul_dtype: jnp.dtype
  # None unless x is dynamically quantized.
  x_quant_block_size: int | None
  vmem_limit_bytes: int

  @property
  def bucket_rows(self):
    bucket_base = self.bucket_base
    # Explicitly include tile_m in case tile_m % bucket_base != 0.
    return [*range(bucket_base, self.tile_m, bucket_base), self.tile_m]

  @property
  def tile_m_idx(self):
    # TODO: Fetch this from tpu_info instead of using magic numbers.
    num_lanes = 128
    return pl.cdiv(self.tile_m, num_lanes) * num_lanes

  @property
  def idx_ring_size(self):
    # Index uses 2 buffers. Despite this, we add at least one more gap to ensure
    # write slot does not overwrite read slot.
    return self.tile_m_idx * max(self.num_xo_buffers, 3)

  @property
  def idx_buffer_size(self):
    # Add tile_m_idx for "runaway" mentioned in IndexBufferedRef docstring.
    return self.idx_ring_size + self.tile_m_idx

  @property
  def num_idx_buffers(self):
    # TODO: Make this configurable?
    return 2

  @classmethod
  def create(
      cls,
      plan: RoutingGmmPlan | None,
      *,
      x: arg.ActivationSpec,
      w: arg.WeightSpec,
      size_out_m: int,
      size_k: int,
      size_out_n: int,
      tpu_info: pltpu.TpuInfo,
  ):
    """Fills unset plan fields with heuristics."""
    if plan is None:
      plan = RoutingGmmPlan()

    # A pre-quantized x already fixes the MXU dtype. Otherwise w's storage
    # dtype is used so that both operands enter the MXU in the same dtype.
    if plan.matmul_dtype is not None:
      matmul_dtype = plan.matmul_dtype
    elif x.quant is None and w.quant is not None:
      # TODO: Support mixed precision and choose by hardware support.
      matmul_dtype = w.dtype
    else:
      matmul_dtype = x.dtype
    x_quant_block_size = plan.x_quant_block_size
    if x_quant_block_size is None and matmul_dtype != x.dtype:
      # TODO: Choose different block size based on HW.
      x_quant_block_size = 512

    acc_dtype = plan.acc_dtype
    if acc_dtype is None:
      acc_dtype = jnp.bfloat16.dtype

    vmem_limit_bytes = plan.vmem_limit_bytes or tpu_info.vmem_capacity_bytes
    tile_m, tile_k, tile_n, bucket_base, num_xo_buffers, num_w_buffers = (
        calculate_tiling(
            plan,
            x=x,
            w=w,
            matmul_dtype=matmul_dtype,
            acc_dtype=acc_dtype,
            size_out_m=size_out_m,
            size_k=size_k,
            size_out_n=size_out_n,
            vmem_limit_bytes=vmem_limit_bytes,
            tpu_info=tpu_info,
        )
    )
    return cls(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        bucket_base=bucket_base,
        num_xo_buffers=num_xo_buffers,
        num_w_buffers=num_w_buffers,
        acc_dtype=acc_dtype,
        matmul_dtype=matmul_dtype,
        x_quant_block_size=x_quant_block_size,
        vmem_limit_bytes=vmem_limit_bytes,
    )


def calculate_tiling(
    plan: RoutingGmmPlan,
    *,
    x: arg.ActivationSpec,
    w: arg.WeightSpec,
    matmul_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    size_out_m: int,
    size_k: int,
    size_out_n: int,
    vmem_limit_bytes: int,
    tpu_info: pltpu.TpuInfo,
) -> tuple[int, int, int, int, int, int]:
  """Returns (tile_m, tile_k, tile_n, bucket_base, num_xo_buffers, num_w_buffers)."""
  # Slightly lower vmem limit than total required amount to be conservative.
  vmem_budget = int(vmem_limit_bytes * 0.9)

  # Step 1: Set default values.
  num_xo_buffers = plan.num_xo_buffers or 3
  num_w_buffers = plan.num_w_buffers or 2
  # TODO: Support K dim tiling.
  tile_k = plan.tile_k or size_k
  num_w_refs = w.out_size // size_out_n
  is_mxu_quantized = matmul_dtype not in (jnp.bfloat16.dtype, jnp.float32.dtype)
  is_dynamic_quantized = matmul_dtype != x.dtype
  w_bits = jax.dtypes.itemsize_bits(w.dtype)
  matmul_bits = jax.dtypes.itemsize_bits(matmul_dtype)
  acc_bits = jax.dtypes.itemsize_bits(acc_dtype)

  if plan.bucket_base is not None:
    bucket_base = plan.bucket_base
  else:
    # When using bf16 for x and w, 128 is the largest tile_m value that is safe
    # to use for most scenarios. But if lower bitwidth is used, we need to tweak
    # tile_m to account for using faster hardware unit.
    bucket_base = 128 // min(pl.cdiv(16, w_bits), 2)
    if is_mxu_quantized:
      bucket_base *= 2
    if plan.tile_m is not None:
      bucket_base = min(bucket_base, plan.tile_m)

  # Chunk size equals the bytes of a single sublane.
  chunk_bytes = 4 * tpu_info.num_lanes
  # NOTE: x_chunk size stays as-is since we don't support K tiling yet.
  x_chunks = layout_util.to_bank_conflict_aware_stride(x.compact_shape[1])
  # out_dtype is not known to the plan, so x's dtype is used as the estimate.
  # TODO: Add ability to pass out_dtype.
  out_shape = layout_util.to_compact_shape((size_out_m, size_out_n), x.dtype)
  out_chunk_cols = out_shape[2] * out_shape[3]

  # Step 2: Define VMEM calculation function.
  def _vmem_estimate(new_tile_m: int, new_tile_n: int) -> int:
    out_chunks = layout_util.to_bank_conflict_aware_stride(
        new_tile_n // out_chunk_cols
    )
    x_vmem = num_xo_buffers * new_tile_m * x_chunks * chunk_bytes
    out_vmem = num_xo_buffers * new_tile_m * out_chunks * chunk_bytes

    w_vmem = num_w_refs * num_w_buffers * tile_k * new_tile_n * w_bits // 8
    scale_vmem = 0
    if w.quant is not None:
      num_blocks = pl.cdiv(tile_k, w.quant.block_size)
      scale_vmem = num_w_refs * num_w_buffers * num_blocks * new_tile_n * 4

    acc_vmem = num_w_refs * new_tile_m * new_tile_n * acc_bits // 8
    # Dynamic quantization materializes a quantized copy of the x tile.
    if is_dynamic_quantized:
      acc_vmem += new_tile_m * tile_k * matmul_bits // 8

    return x_vmem + out_vmem + w_vmem + scale_vmem + acc_vmem

  # Split N into the fewest tiles that fit VMEM. tile_n is chosen before tile_m
  # because the weight block usually dominates VMEM.
  if plan.tile_n is not None:
    tile_n = plan.tile_n
  else:
    num_n_tiles = 1
    tile_n = size_out_n
    max_num_n_tiles = size_out_n // out_chunk_cols
    min_tile_m = plan.tile_m or bucket_base
    while (
        _vmem_estimate(min_tile_m, tile_n) > vmem_budget
        and num_n_tiles < max_num_n_tiles
    ):
      num_n_tiles += 1
      # tile_n must divide size_out_n into whole compact chunks.
      if size_out_n % (num_n_tiles * out_chunk_cols) == 0:
        tile_n = size_out_n // num_n_tiles

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
          or _vmem_estimate(candidate, tile_n) > vmem_budget
      ):
        break
      tile_m = candidate

  estimate = _vmem_estimate(tile_m, tile_n)
  if estimate > vmem_budget:
    raise ValueError(
        f"Tiles ({tile_m=}, {tile_k=}, {tile_n=}) need an estimated"
        f" {estimate} bytes of VMEM, exceeding the budget of {vmem_budget}"
        f" bytes (limit: {vmem_limit_bytes})."
    )

  return tile_m, tile_k, tile_n, bucket_base, num_xo_buffers, num_w_buffers
