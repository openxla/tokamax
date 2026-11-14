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


class MatmulMode(enum.StrEnum):
  """How x and w are fed to the MXU and how their scales are applied."""

  # x and w are both passed in unquantized.
  UNQUANTIZED = enum.auto()
  # w is passed in quantized with scales. x is passed in unquantized, and the
  # kernel quantizes it before the matmul. The output is scaled by both x and w
  # scales.
  QUANTIZED = enum.auto()
  # w is passed in quantized with scales. x is passed in unquantized and stays
  # unquantized. The output is scaled by w scales.
  WEIGHT_ONLY_SCALE_AFTER = enum.auto()
  # Like WEIGHT_ONLY_SCALE_AFTER, but w is dequantized before the matmul. Used
  # when w quant blocks are narrower than the MXU.
  WEIGHT_ONLY_SCALE_BEFORE = enum.auto()


@dataclass_util.require_no_default
@dataclasses.dataclass(frozen=True)
class ComputeConfig:
  """Resolved configuration for a single matrix multiplication."""

  matmul_mode: MatmulMode
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

    if x_quant is not None and spec.w.quant is not None:
      matmul_mode = MatmulMode.QUANTIZED
    elif plan_lib.dequantizes_w_before_matmul(
        x_dtype=spec.x.dtype,
        matmul_dtype=plan.matmul_dtype,
        w=spec.w,
        tpu_info=tpu_info,
    ):
      matmul_mode = MatmulMode.WEIGHT_ONLY_SCALE_BEFORE
    elif x_quant is None and spec.w.quant is not None:
      matmul_mode = MatmulMode.WEIGHT_ONLY_SCALE_AFTER
    elif x_quant is None and spec.w.quant is None:
      matmul_mode = MatmulMode.UNQUANTIZED
    else:
      raise NotImplementedError("Quantizing x requires a quantized w.")

    compute_cfg = ComputeConfig(
        matmul_mode=matmul_mode,
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
    cfg._validate_tiling()
    cfg._validate_buffers()
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
      if self.plan.num_stash_buffers is not None:
        raise ValueError(
            "plan.num_stash_buffers has no effect when output is not quantized."
        )
      if self.plan.stash_dtype is not None:
        raise ValueError(
            "plan.stash_dtype has no effect when output is not quantized."
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
    if self.plan.num_stash_buffers != 2:
      raise NotImplementedError(
          "num_stash_buffers other than 2 is not supported yet. Got"
          f" {self.plan.num_stash_buffers}."
      )
    if self.plan.stash_dtype != jnp.bfloat16.dtype:
      raise NotImplementedError(
          "stash_dtype other than bfloat16 is not supported yet. Got"
          f" {self.plan.stash_dtype}."
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

  def _validate_tiling(self) -> None:
    """Validates tile sizes and sublane alignment."""
    self._validate_k_tiling()
    self._validate_w2_tiling()
    self._validate_n_tiling()
    self._validate_m_tiling()

  def _validate_k_tiling(self) -> None:
    """Validates that the K dim is not tiled."""
    is_k_tiled = self.plan.tile_k != self.spec.size_k
    if is_k_tiled and self.num_k2 > 1:
      raise NotImplementedError(
          "K dim tiling is not supported together with k2 tiling."
      )
    if is_k_tiled:
      raise NotImplementedError("K dim tiling is not supported yet.")

  def _validate_w2_tiling(self) -> None:
    """Validates the k2 tiling of the fused w2."""
    if self.spec.w2 is None:
      return
    size_k2 = self.spec.size_k2
    assert size_k2 is not None and self.plan.tile_k2 is not None
    if size_k2 % self.plan.tile_k2 != 0:
      raise ValueError(
          f"tile_k2 ({self.plan.tile_k2}) must divide size_k2 ({size_k2})."
      )
    if self.plan.tile_k2 % self.tpu_info.num_lanes != 0:
      raise ValueError(
          f"tile_k2 ({self.plan.tile_k2}) must be a multiple of"
          f" {self.tpu_info.num_lanes}."
      )
    w2_quant = self.spec.w2.quant
    if w2_quant is not None:
      if not isinstance(w2_quant, tensor_spec.SubchannelQuantSpec):
        raise NotImplementedError(
            f"Fused w2 does not support {type(w2_quant).__name__}."
        )
      sizes = (self.plan.tile_k2, w2_quant.block_size)
      if max(sizes) % min(sizes) != 0:
        raise ValueError(
            f"tile_k2 ({self.plan.tile_k2}) and the w2 quantization block"
            f" size ({w2_quant.block_size}) must divide one"
            " another."
        )
    if self.plan.tile_n != self.spec.size_out_n:
      raise NotImplementedError(
          "N dim tiling is not supported when w2 is fused."
      )

  def _validate_n_tiling(self) -> None:
    """Validates that tile_n evenly splits the output."""
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

  def _validate_m_tiling(self) -> None:
    """Validates tile_m and bucket_base sizes and sublane alignment."""
    if not 0 < self.plan.bucket_base <= self.plan.tile_m:
      raise ValueError(
          f"bucket_base ({self.plan.bucket_base}) must be in (0, tile_m="
          f"{self.plan.tile_m}]."
      )
    sublanes = self.tpu_info.num_sublanes
    if (
        self.plan.tile_m % sublanes != 0
        or self.plan.bucket_base % sublanes != 0
    ):
      raise ValueError(
          f"tile_m ({self.plan.tile_m}) and bucket_base"
          f" ({self.plan.bucket_base}) must be multiples of {sublanes}."
      )

  def _validate_buffers(self) -> None:
    """Validates pipeline buffer counts."""
    if self.plan.num_xo_buffers < 3:
      raise ValueError(
          "To overlap DMA SALU, activation buffer counts must be at least 3."
      )
    if self.plan.num_w_buffers < 2:
      raise ValueError("Buffer counts must be at least 2.")

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
  def num_k2(self) -> int:
    """Number of intermediate (k2) slices the fused down projection runs."""
    size_k2 = self.spec.size_k2
    if size_k2 is None:
      return 1
    assert self.plan.tile_k2 is not None
    return pl.cdiv(size_k2, self.plan.tile_k2)

  @property
  def kernel_name(self) -> str:
    """Kernel name encoding the modes, shapes and tiling."""
    name = f"routing_gmm-{self.x.mode}-{self.out.mode}"
    name += f"-g{self.spec.size_local_groups}-m{self.spec.size_out_m}"
    name += f"-k{self.spec.size_k}"
    if self.spec.size_k2 is not None:
      name += f"-k2{self.spec.size_k2}"
    name += f"-n{self.spec.size_out_n}"
    name += f"-tm{self.plan.tile_m}"
    if self.plan.tile_k2 is not None:
      name += f"-tk2{self.plan.tile_k2}"
    name += f"-tn{self.plan.tile_n}"
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
