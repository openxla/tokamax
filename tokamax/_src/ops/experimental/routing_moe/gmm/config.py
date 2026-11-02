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

Config is the single source of truth that kernel modules read. It assembles
the problem spec, output format, and resolved plan without applying tuning
heuristics or format defaults.
"""

import dataclasses
import enum
import json
from typing import Self

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import dataclass_util
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.common import spec as tensor_spec
from tokamax._src.ops.experimental.routing_moe.gmm import plan as plan_lib
from tokamax._src.ops.experimental.routing_moe.gmm import spec as gmm_spec_lib


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


@dataclass_util.require_no_default
@dataclasses.dataclass(frozen=True)
class ComputeConfig:
  """Resolved configuration for a single matrix multiplication."""

  matmul_dtype: jnp.dtype
  acc_dtype: jnp.dtype
  tpu_info: pltpu.TpuInfo
  dynamic_quant: tensor_spec.SubchannelQuantSpec | None


def _resolve_mode(routing: tensor_spec.RoutingSpec | None) -> RoutingMode:
  """Resolves the DMA routing mode from a routing specification."""
  if routing is None:
    return RoutingMode.DIRECT
  if routing.remote_axis is None:
    return RoutingMode.LOCAL
  return RoutingMode.REMOTE


def _resolve_out_spec(
    *,
    spec: gmm_spec_lib.RoutingGmmSpec,
    plan: plan_lib.ResolvedGmmPlan,
    out_format: out_format_lib.ResolvedActivationFormat,
) -> tensor_spec.ActivationSpec:
  """Materializes `out_format` against the problem spec and tiling."""
  # A remote scatter leaves each device only the rows it owns.
  return tensor_spec.ActivationSpec(
      row_size=spec.size_out_m // spec.size_ep,
      col_size=spec.size_out_n,
      dtype=out_format.dtype,
      quant=out_format_lib.to_quant_spec(
          out_format.quant, tile_col_size=plan.tile_n
      ),
  )


@dataclass_util.require_no_default
@dataclasses.dataclass(frozen=True)
class ActivationConfig:
  """Resolved spec and routing mode of an activation.

  When `spec.quant` is `LaneStridedQuantSpec`, each row in compact layout
  `[m, n // (packing * lanes) + 1, packing, lanes]` stores its logical chunks
  followed by one trailing chunk of `f32` per-lane scales (`absmax /
  dtype_max`, with lane `l` covering columns `l, l + lanes, ...`). Because the
  trailing chunk stores `f32` bit patterns inside a narrow-dtype buffer,
  consumers must reinterpret that chunk as raw bytes rather than narrow-float
  values.
  """

  spec: tensor_spec.ActivationSpec
  mode: RoutingMode


@dataclass_util.require_no_default
@dataclasses.dataclass(frozen=True)
class RoutingGmmConfig:
  """Compile-time static configuration of the routing GMM kernel."""

  spec: gmm_spec_lib.RoutingGmmSpec
  plan: plan_lib.ResolvedGmmPlan
  x: ActivationConfig
  compute: ComputeConfig
  compute2: ComputeConfig | None
  out: ActivationConfig
  tpu_info: pltpu.TpuInfo

  @classmethod
  def create(
      cls,
      *,
      spec: gmm_spec_lib.RoutingGmmSpec,
      out_format: out_format_lib.ResolvedActivationFormat,
      plan: plan_lib.ResolvedGmmPlan,
      tpu_info: pltpu.TpuInfo,
  ) -> Self:
    """Assembles the config and validates the plan against specs and HW."""
    if spec.x.quant is not None:
      # TODO: Load pre-computed scales in compute.
      raise NotImplementedError("Pre-quantized x is not supported yet.")

    # Because x only states the input storage dtype while plan states the MXU
    # execution dtype, neither alone defines whether dynamic quantization runs.
    # Config reconciles both into ComputeConfig.
    x_quant = None
    if plan.matmul_dtype != spec.x.dtype:
      assert plan.x_quant_block_size is not None
      x_quant = tensor_spec.SubchannelQuantSpec(
          block_size=plan.x_quant_block_size
      )

    compute_cfg = ComputeConfig(
        matmul_dtype=plan.matmul_dtype,
        acc_dtype=plan.acc_dtype,
        tpu_info=tpu_info,
        dynamic_quant=x_quant,
    )
    out_spec = _resolve_out_spec(spec=spec, plan=plan, out_format=out_format)

    cfg = cls(
        spec=spec,
        plan=plan,
        x=ActivationConfig(spec=spec.x, mode=_resolve_mode(spec.gather)),
        compute=compute_cfg,
        compute2=None if spec.w2 is None else compute_cfg,
        out=ActivationConfig(spec=out_spec, mode=_resolve_mode(spec.scatter)),
        tpu_info=tpu_info,
    )
    cfg._validate_dtypes()
    cfg._validate_out()
    cfg._validate_tiling_and_buffers()
    return cfg

  def _validate_dtypes(self) -> None:
    """Validates precision and quantization invariants."""
    w, w2 = self.spec.w, self.spec.w2
    if w2 is not None and (
        w2.dtype != w.dtype or (w2.quant is None) != (w.quant is None)
    ):
      # The intermediate is fed to the second matmul in x's execution dtype,
      # which is only valid when both weights share a precision.
      raise NotImplementedError("w2 must match the precision of w.")
    if (
        self.compute.dynamic_quant is None
        and self.plan.x_quant_block_size is not None
    ):
      raise ValueError(
          "plan.x_quant_block_size has no effect when x is not quantized."
      )
    if self.plan.acc_dtype not in (jnp.bfloat16.dtype, jnp.float32.dtype):
      raise ValueError(
          f"acc_dtype must be bfloat16 or float32. Got {self.plan.acc_dtype}."
      )
    x_bits = jax.dtypes.itemsize_bits(self.plan.matmul_dtype)
    if x_bits < jax.dtypes.itemsize_bits(w.dtype):
      raise ValueError(
          f"matmul_dtype ({self.plan.matmul_dtype}) must not be narrower"
          f" than w dtype ({w.dtype})."
      )

  def _validate_out(self) -> None:
    """Validates output dtype, quantization, and chunk alignment."""
    out_spec = self.out.spec
    if out_spec.quant is None:
      if out_spec.dtype != self.x.spec.dtype:
        raise NotImplementedError(
            "Unquantized output dtype conversion is not supported yet."
        )
      return

    if self.spec.scatter is None:
      raise NotImplementedError("Quantized output needs a scatter.")
    if not isinstance(out_spec.quant, tensor_spec.LaneStridedQuantSpec):
      raise NotImplementedError(
          "Only LaneStrided quantization is supported for quantized output."
      )
    if out_spec.dtype.itemsize != 1:
      raise ValueError(
          f"Quantized output requires a 1-byte dtype. Got {out_spec.dtype}."
      )
    chunk_cols = (
        layout_util.get_packing(out_spec.dtype) * self.tpu_info.num_lanes
    )
    if self.spec.size_out_n % chunk_cols:
      raise ValueError(
          f"Output cols {self.spec.size_out_n} must fill whole"
          f" {out_spec.dtype} chunks."
      )
    if self.num_n != 1:
      # The per-lane scale is taken over the whole output row.
      raise NotImplementedError(
          "A quantized output does not support N dim tiling yet."
      )
    if out_spec.quant.block_col_size != self.plan.tile_n:
      raise ValueError(
          "Output quantization block_col_size"
          f" ({out_spec.quant.block_col_size}) must equal tile_n"
          f" ({self.plan.tile_n})."
      )

  def _validate_tiling_and_buffers(self) -> None:
    """Validates tile sizes, buffer counts, and SMEM footprint."""
    if self.plan.tile_k != self.spec.size_k:
      raise NotImplementedError("K dim tiling is not supported yet.")
    if self.spec.w2 is not None and self.plan.tile_k2 != self.spec.size_k2:
      raise NotImplementedError("K2 dim tiling is not supported yet.")
    if self.spec.w2 is not None and self.plan.tile_n != self.spec.size_out_n:
      raise NotImplementedError(
          "N dim tiling is not supported when w2 is fused."
      )
    if self.spec.size_out_n % self.plan.tile_n != 0:
      raise ValueError(
          f"tile_n ({self.plan.tile_n}) must divide the output width"
          f" ({self.spec.size_out_n})."
      )
    if self.out.spec.num_logical_chunks % self.num_n != 0:
      raise ValueError(
          f"tile_n ({self.plan.tile_n}) must be a multiple of the output's"
          " compact chunk width."
      )
    if not 0 < self.plan.bucket_base <= self.plan.tile_m:
      raise ValueError(
          f"bucket_base ({self.plan.bucket_base}) must be in (0, tile_m="
          f"{self.plan.tile_m}]."
      )
    if self.plan.num_xo_buffers < 3:
      raise ValueError(
          "To overlap DMA SALU, activation buffer counts must be at least 3."
      )
    if self.plan.num_w_buffers < 2:
      raise ValueError("Buffer counts must be at least 2.")
    smem_threshold = int(0.45 * self.tpu_info.smem_capacity_bytes)
    # 3: _to_group_id, _to_m_offset, _to_num_rows. 4: int32.
    smem_allocation = (self.max_num_gm * 3) * 4
    if smem_allocation >= smem_threshold:
      raise ValueError(
          f"Total number of SMEM allocation must be less than {smem_threshold}"
          f" to avoid OOM. Got {smem_allocation}"
      )

  @property
  def max_num_gm(self) -> int:
    """Maximum number of possible tiles."""
    max_num_single_group = pl.cdiv(self.spec.size_out_m, self.plan.tile_m)
    return self.spec.size_local_groups + max_num_single_group

  @property
  def num_n(self) -> int:
    return pl.cdiv(self.spec.size_out_n, self.plan.tile_n)

  @property
  def num_k(self) -> int:
    num_k = pl.cdiv(self.spec.size_k, self.plan.tile_k)
    # TODO: Add support for K dim tiling.
    assert num_k == 1
    return num_k

  @property
  def kernel_name(self) -> str:
    name = f"routing_gmm-{self.x.mode}-{self.out.mode}"
    name += f"-g{self.spec.size_local_groups}-m{self.spec.size_out_m}"
    name += f"-k{self.spec.size_k}-n{self.spec.size_out_n}"
    name += f"-tm{self.plan.tile_m}-tn{self.plan.tile_n}"
    return name

  @property
  def kernel_metadata(self) -> dict[str, str]:
    """The JSON-serialized config metadata for XProf."""

    def to_json(obj: object) -> object:
      if isinstance(obj, pltpu.TpuInfo):
        return str(obj.chip_version)
      if dataclasses.is_dataclass(obj):
        return vars(obj)
      return str(obj)

    return {"xprof_metadata": json.dumps(self, default=to_json)}
