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

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg


@dataclasses.dataclass(frozen=True, kw_only=True)
class RoutingGmmPlan:
  """User provided plan where `None` field will be determined by heuristic."""

  tile_m: int | None = None
  tile_k: int | None = None
  tile_n: int | None = None
  bucket_base: int | None = None
  # Buffer counts have fixed default by the kernel's design.
  num_xo_buffers: int = 3
  num_w_buffers: int = 2
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
  def idx_buffer_size(self):
    # idx buffer requires at least one gap between read range and write range
    # to avoid corruption.
    return self.tile_m_idx * max(self.num_xo_buffers, 3)

  @property
  def num_max_buffers(self):
    return max(self.num_xo_buffers, self.num_w_buffers)

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
    tile_m, tile_k, tile_n, bucket_base = calculate_tiling(
        plan,
        x=x,
        w=w,
        matmul_dtype=matmul_dtype,
        size_out_m=size_out_m,
        size_k=size_k,
        size_out_n=size_out_n,
        vmem_limit_bytes=vmem_limit_bytes,
        tpu_info=tpu_info,
    )
    return cls(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        bucket_base=bucket_base,
        num_xo_buffers=plan.num_xo_buffers,
        num_w_buffers=plan.num_w_buffers,
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
    size_out_m: int,
    size_k: int,
    size_out_n: int,
    vmem_limit_bytes: int,
    tpu_info: pltpu.TpuInfo,
) -> tuple[int, int, int, int]:
  """Returns (tile_m, tile_k, tile_n, bucket_base) while keeping user fields."""
  del x, w, matmul_dtype, vmem_limit_bytes, tpu_info

  # TODO: Add more sophisticated tile selection logic.
  tile_m = min(256, size_out_m) if plan.tile_m is None else plan.tile_m
  # TODO: Support N/K dim tiling.
  tile_k = plan.tile_k or size_k
  tile_n = plan.tile_n or size_out_n
  bucket_base = (
      min(128, tile_m) if plan.bucket_base is None else plan.bucket_base
  )
  return tile_m, tile_k, tile_n, bucket_base
