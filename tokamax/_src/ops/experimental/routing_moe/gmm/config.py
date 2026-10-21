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

"""Resolved static configuration of the routing GMM kernel.

Config is the single source of truth that kernel modules read: specs, derived
sizes, and the resolved plans. No execution decision is made by config.
"""

import dataclasses
import enum

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.gmm import plan as plan_lib


class RoutingMode(enum.StrEnum):
  """Instructions on how DMA for input and output should be done."""

  DIRECT = enum.auto()  # DMA is performed on continuous memory region.
  LOCAL = enum.auto()  # Local DMA is performed in per-row basis.
  # Remote DMA to the device that owns the row, per row. Output only.
  # TODO: Unlike DIRECT and LOCAL, REMOTE also changes the output
  # layout (each device keeps only its rows). Consider renaming LOCAL to HBM and
  # moving remote-ness into ActivationConfig, so remote vs local differs only in
  # the per-row DMA.
  REMOTE = enum.auto()


@dataclasses.dataclass(frozen=True)
class DynamicQuantConfig:
  """Dynamic quantization of an activation as seen by compute."""

  dtype: jnp.dtype
  block_size: int


@dataclasses.dataclass(frozen=True)
class ActivationConfig:
  """User provided spec of an activation and the kernel's decisions on it."""

  spec: arg.ActivationSpec
  mode: RoutingMode
  dynamic_quant: DynamicQuantConfig | None = None


@dataclasses.dataclass(frozen=True)
class Dimension:
  """Problem sizes derived from the specs."""

  size_m: int  # Rows of x.
  size_k: int
  size_n: int  # Columns of w, including both halves when gate/up fused.
  size_out_m: int
  size_out_n: int
  size_local_groups: int
  size_global_groups: int
  # Devices that output rows are split across by a remote scatter.
  size_ep: int = 1

  @classmethod
  def create(
      cls,
      *,
      x: arg.ActivationSpec,
      w: arg.WeightSpec,
      num_global_groups: int,
      gather: arg.RoutingSpec | None,
      scatter: arg.RoutingSpec | None,
      fuse_act: str | None,
      size_ep: int = 1,
  ):
    """Validates the specs against each other and derives problem sizes.

    `size_ep` is the size of `scatter.remote_axis`, 1 for a local scatter.
    """
    if x.col_size != w.in_size:
      raise ValueError(
          f"x col_size ({x.col_size}) must match w in_size ({w.in_size})."
      )
    if num_global_groups < w.group_size:
      raise ValueError(
          f"group_sizes must cover at least {w.group_size} groups. Got"
          f" {num_global_groups}."
      )
    if fuse_act is not None and not w.fused_gate_up:
      raise ValueError("fuse_act requires a weight with fused_gate_up=True.")
    if w.has_bias:
      raise NotImplementedError("Bias is not supported yet.")
    if gather is not None and scatter is not None:
      raise NotImplementedError(
          "Simultaneous gather and scatter is not supported."
      )

    size_m = x.row_size
    # The fused activation combines a gate column and an up column into one, so
    # the output halves only when fuse_act is applied. fused_gate_up alone does
    # not change the width.
    size_out_n = w.out_size // 2 if fuse_act is not None else w.out_size
    size_out_m = size_m
    if gather is not None:
      if gather.remote_axis is not None:
        raise NotImplementedError("Remote gather is not supported.")
      size_out_m = gather.num_indices
    if scatter is not None:
      if scatter.num_indices != size_m:
        raise ValueError(
            f"scatter must map every x row ({size_m}). Got num_indices"
            f" {scatter.num_indices}."
        )
    if (scatter is None or scatter.remote_axis is None) and size_ep != 1:
      raise ValueError(f"size_ep ({size_ep}) requires a remote scatter.")
    if size_out_m % size_ep != 0:
      raise ValueError(
          f"Output rows ({size_out_m}) must divide evenly across"
          f" {size_ep} devices."
      )

    return cls(
        size_m=size_m,
        size_k=x.col_size,
        size_n=w.out_size,
        size_out_m=size_out_m,
        size_out_n=size_out_n,
        size_local_groups=w.group_size,
        size_global_groups=num_global_groups,
        size_ep=size_ep,
    )


@dataclasses.dataclass(frozen=True)
class RoutingGmmConfig:
  """Stores compile time static information of routing gmm."""

  dim: Dimension
  plan: plan_lib.ResolvedPlan
  x: ActivationConfig
  w: arg.WeightSpec
  out: ActivationConfig
  gather: arg.RoutingSpec | None
  scatter: arg.RoutingSpec | None
  fuse_act: str | None
  tpu_info: pltpu.TpuInfo

  @classmethod
  def create(
      cls,
      *,
      x: arg.ActivationSpec,
      w: arg.WeightSpec,
      dim: Dimension,
      plan: plan_lib.ResolvedPlan,
      gather: arg.RoutingSpec | None,
      scatter: arg.RoutingSpec | None,
      fuse_act: str | None,
      out_dtype: jnp.dtype | None,
      tpu_info: pltpu.TpuInfo,
  ):
    """Assembles the config and validates the plan against specs and HW."""
    if x.quant is not None:
      # TODO: Load pre-computed scales in compute.
      raise NotImplementedError("Pre-quantized x is not supported yet.")

    # Spec states dtype of the input and plan states conversion during compute.
    # Neither alone states the dtype compute sees. Config derives it from both.
    x_quant = None
    if plan.matmul_dtype != x.dtype:
      assert plan.x_quant_block_size is not None
      x_quant = DynamicQuantConfig(
          dtype=plan.matmul_dtype, block_size=plan.x_quant_block_size
      )

    # A remote scatter leaves each device only the rows it owns.
    out_spec = arg.ActivationSpec(
        row_size=dim.size_out_m // dim.size_ep,
        col_size=dim.size_out_n,
        dtype=x.dtype if out_dtype is None else out_dtype,
        row_shard_axis=None if scatter is None else scatter.remote_axis,
    )

    def _resolve_mode(routing: arg.RoutingSpec | None) -> RoutingMode:
      if routing is None:
        return RoutingMode.DIRECT
      if routing.remote_axis is None:
        return RoutingMode.LOCAL
      return RoutingMode.REMOTE

    x_mode = _resolve_mode(gather)
    out_mode = _resolve_mode(scatter)

    cfg = cls(
        dim=dim,
        plan=plan,
        x=ActivationConfig(spec=x, mode=x_mode, dynamic_quant=x_quant),
        w=w,
        out=ActivationConfig(spec=out_spec, mode=out_mode),
        gather=gather,
        scatter=scatter,
        fuse_act=fuse_act,
        tpu_info=tpu_info,
    )

    x_bits = jax.dtypes.itemsize_bits(cfg.plan.matmul_dtype)
    if x_bits < jax.dtypes.itemsize_bits(cfg.w.dtype):
      raise ValueError(
          f"x_exec_dtype ({cfg.plan.matmul_dtype}) must not be narrower"
          f" than w dtype ({cfg.w.dtype})."
      )
    if cfg.x.dynamic_quant is None and cfg.plan.x_quant_block_size is not None:
      raise ValueError(
          "plan.x_quant_block_size has no effect when x is not quantized."
      )
    if cfg.plan.acc_dtype not in (jnp.bfloat16.dtype, jnp.float32.dtype):
      raise ValueError(
          "acc_dtype must be bfloat16 or float32. Got"
          f" {cfg.plan.acc_dtype}."
      )
    if cfg.plan.tile_k != cfg.dim.size_k:
      raise NotImplementedError("K dim tiling is not supported yet.")
    if cfg.dim.size_out_n % cfg.plan.tile_n != 0:
      raise ValueError(
          f"tile_n ({cfg.plan.tile_n}) must divide the output width"
          f" ({cfg.dim.size_out_n})."
      )
    if cfg.out.spec.compact_shape[1] % cfg.num_n != 0:
      raise ValueError(
          f"tile_n ({cfg.plan.tile_n}) must be a multiple of the output's"
          " compact chunk width."
      )
    if not 0 < cfg.plan.bucket_base <= cfg.plan.tile_m:
      raise ValueError(
          f"bucket_base ({cfg.plan.bucket_base}) must be in (0, tile_m="
          f"{cfg.plan.tile_m}]."
      )
    if cfg.plan.num_xo_buffers < 3:
      raise ValueError(
          "To overlap DMA SALU, activation buffer counts must be at least 3."
      )
    if cfg.plan.num_w_buffers < 2:
      raise ValueError("Buffer counts must be at least 2.")
    smem_threshold = int(0.45 * tpu_info.smem_capacity_bytes)
    # 3: _to_group_id, _to_m_offset, _to_num_rows. 4: int32.
    smem_allocation = (cfg.max_num_gm * 3) * 4
    if smem_allocation >= smem_threshold:
      raise ValueError(
          f"Total number of SMEM allocation must be less than {smem_threshold}"
          f" to avoid OOM. Got {smem_allocation}"
      )
    return cfg

  @property
  def max_num_gm(self) -> int:
    """Returns maximum number of possible tiles."""
    max_num_single_group = pl.cdiv(self.dim.size_out_m, self.plan.tile_m)
    return self.dim.size_local_groups + max_num_single_group

  @property
  def num_n(self) -> int:
    return pl.cdiv(self.dim.size_out_n, self.plan.tile_n)

  @property
  def num_k(self) -> int:
    num_k = pl.cdiv(self.dim.size_k, self.plan.tile_k)
    # TODO: Add support for K dim tiling.
    assert num_k == 1
    return num_k

  @property
  def kernel_name(self):
    name = f"routing_gmm-{self.x.mode}-{self.out.mode}"
    name += f"-g{self.dim.size_local_groups}-m{self.dim.size_out_m}"
    name += f"-k{self.dim.size_k}-n{self.dim.size_out_n}"
    name += f"-tm{self.plan.tile_m}-tn{self.plan.tile_n}"
    if self.x.dynamic_quant is not None:
      name += f"-xq{self.x.dynamic_quant.dtype.name}"
    return name
