# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Ragged dot Pallas-Mosaic-TPU implementation."""

import dataclasses
import itertools
import types
from typing import Any, ClassVar
import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pydantic
import qwix
from tokamax._src import precision as precision_lib
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_gmm_kernel as gmm_backend
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_backend
from typing_extensions import override

# TODO: Directly import ManualAxisType JAX is upgraded.
try:
  from jax.sharding import ManualAxisType
except ImportError:
  ManualAxisType = Any

QArray = base.QArray
AsQArray = base.AsQArray
Residuals = types.NoneType

@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Pallas Mosaic TPU Ragged Dot config holding the kernel tuning
  parameters.
  """

  tile_m: pydantic.PositiveInt = 128
  tile_k: pydantic.PositiveInt = 128
  tile_n: pydantic.PositiveInt = 128

DEFAULT_RAGGED_DOT_DIM_NUMS = base.DEFAULT_RAGGED_DOT_DIM_NUMS

DLHS_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [2]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)

DRHS_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[],
)

UNSUPPORTED_DIMENSIONS_MSG = (
    "Specified ragged_dot_dimension_numbers `{}` not supported. Supported"
    f" dimensions include: {DEFAULT_RAGGED_DOT_DIM_NUMS},"
    f" {DLHS_RAGGED_DOT_DIM_NUMS}, {DRHS_RAGGED_DOT_DIM_NUMS}"
)


def _has_manual_axes(manual_axis_type) -> bool:
  """Returns whether `manual_axis_type` names any manual axes.

  The base VJP always forwards a `manual_axis_type` to its sub-calls, derived
  from the input's type. For unsharded arrays this is a *trivial* value (all
  axis sets empty), which is semantically equivalent to `None`. Only a value
  that actually names axes implies shard_map, which v2 does not support yet.
  """
  if manual_axis_type is None:
    return False
  return bool(
      getattr(manual_axis_type, "varying", None)
      or getattr(manual_axis_type, "unreduced", None)
      or getattr(manual_axis_type, "reduced", None)
  )

@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicTpuV2RaggedDot(base.RaggedDot[Config, None]):
  """Pallas-Mosaic-TPU ragged dot implementation v2.

  TPU Implementation of the Megablocks Paper https://arxiv.org/abs/2211.15841.
  """

  config_cls: ClassVar[type[Config]] = Config
  qdtype: jax.typing.DTypeLike | None = None
  # Local weight group count for the drhs (tgmm) path. Under expert
  # parallelism this differs from `group_sizes.shape[0]` (which is global).
  # Set by the custom vjp (see `__post_init__`); `None` for the forward and
  # for direct (non-autodiff) drhs calls.
  num_actual_groups: int | None = None

  def __post_init__(self):
    qdtype: str | None = (
        self.qdtype if self.qdtype is None else jnp.dtype(self.qdtype).name
    )
    if self.vjp is None:

      # Build a fresh sub-op for the backward sub-calls. `num_actual_groups`
      # only matters for the drhs (tgmm) path; leave it `None` otherwise.
      def make_fn(num_actual_groups=None):
        return lambda *args, **kw: PallasMosaicTpuV2RaggedDot(  # pylint: disable=unnecessary-lambda
            qdtype=qdtype,
            num_actual_groups=num_actual_groups,
        )(*args, **kw)

      def _vjp(residuals, out, dout, lhs, rhs, **kwargs):
        # Under expert parallelism `group_sizes` is global, but the original
        # local weights `rhs` (available here on the backward path) carry the
        # actual local group count. Capture it and thread it into the tgmm
        # sub-op, since the drhs `_fwd` only sees `lhs`/`dout`, not the weights.
        num_actual_groups = rhs.shape[0]
        return base.vjp(
            residuals,
            out,
            dout,
            lhs,
            rhs,
            dlhs_ragged_dot=make_fn(),
            drhs_ragged_dot=make_fn(num_actual_groups=num_actual_groups),
            **kwargs,
        )

      object.__setattr__(self, "vjp", _vjp)

  @override
  def _fwd(
      self,
      lhs: jax.Array | QArray | AsQArray,
      rhs: jax.Array | QArray | AsQArray,
      *,
      group_sizes: jax.Array | base.GroupSizes,
      ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
      precision: base.CanonicalPrecision,
      preferred_element_type: jax.typing.DTypeLike | None,
      return_residuals: bool = False,
      config: Config,
      group_offset: jax.Array | None = None,
      activation: base.ActivationFunction | None = None,
      manual_axis_type: ManualAxisType | None = None,
      rhs_scale: jax.Array | None = None,
      rhs_bias: jax.Array | None = None,
      maybe_quantize_lhs: bool = False,
      zero_initialize: bool = True,
      fuse_act: str | None = None,
  ) -> tuple[jax.Array, base.Residuals]:
    if isinstance(lhs, (QArray, AsQArray)) or isinstance(rhs, (QArray, AsQArray)):
      raise NotImplementedError("v2 accepts only raw arrays; pass quantization via the rhs_scale/rhs_bias API kwargs instead.")
    if _has_manual_axes(manual_axis_type):
      raise NotImplementedError(
          "v2 does not support manual_axis_type yet. But got"
          f" manual_axis_type={manual_axis_type}"
      )
    if activation is not None:
      raise NotImplementedError("v2 does not support activation.")

    if isinstance(group_sizes, base.GroupSizes):
      group_sizes = jnp.array(group_sizes)
    # The v2 kernel's metadata code does `pl.cdiv(group_size, tile_m)` with an
    # int32 `tile_m`, and `lax.div` rejects mixed signedness. Callers may pass
    # `uint32` group sizes (e.g. `jax.lax.ragged_dot`'s default), so normalize
    # to the int32 contract the kernel documents.
    group_sizes = group_sizes.astype(jnp.int32)

    vmem_limit_bytes = None
    acc_dtype = None
    if ragged_dot_dimension_numbers == DEFAULT_RAGGED_DOT_DIM_NUMS:  # gmm fwd
      out = gmm_backend.gmm_v2(
          lhs,
          rhs,
          group_sizes,
          rhs_scale,
          rhs_bias,
          group_offset,
          tile_info=gmm_backend.calculate_tiling,
          vmem_limit_bytes=vmem_limit_bytes,
          precision=precision,
          preferred_element_type=preferred_element_type,
          acc_dtype=acc_dtype,
          maybe_quantize_lhs=maybe_quantize_lhs,
          zero_initialize=zero_initialize,
          fuse_act=fuse_act,
      )
    elif ragged_dot_dimension_numbers == DLHS_RAGGED_DOT_DIM_NUMS:  # dlhs
      out = gmm_backend.gmm_v2(
          lhs,  # [m, n]
          rhs.swapaxes(1, 2),  # [num_groups, n, k]
          group_sizes,
          None,  # rhs_scale
          None,  # rhs_bias
          group_offset,
          tile_info=gmm_backend.calculate_tiling,
          vmem_limit_bytes=vmem_limit_bytes,
          precision=precision,
          preferred_element_type=lhs.dtype,
          acc_dtype=acc_dtype,
          maybe_quantize_lhs=maybe_quantize_lhs,
          zero_initialize=zero_initialize,
          fuse_act=None,
      )
    elif ragged_dot_dimension_numbers == DRHS_RAGGED_DOT_DIM_NUMS:  # drhs
      # Captured from the original local weights in the custom vjp. Under
      # expert parallelism this is the local group count, which `group_sizes`
      # (global) cannot provide. A direct (non-autodiff) drhs call has no
      # weight context, so it is unsupported.
      if self.num_actual_groups is None:
        raise NotImplementedError(
            "Direct drhs (tgmm) calls are not supported; `num_actual_groups`"
            " is only available on the autodiff backward path."
        )
      num_actual_groups = self.num_actual_groups
      # tgmm computes lhs^T @ rhs on the MXU and requires both operands to share
      # a sublane tiling, i.e. the same dtype (it asserts
      # size_lhs_sublane == size_rhs_sublane).
      rhs = rhs.astype(lhs.dtype)
      out = tgmm_backend.tgmm_v2(
          lhs,  # [m, k]
          rhs,  # [m, n]
          group_sizes,
          num_actual_groups,
          group_offset,
          # TODO: consider letting users provide tiling for bwd.
          tile_info=tgmm_backend.calculate_tgmm_tiling,
          vmem_limit_bytes=vmem_limit_bytes,
          precision=precision,
          preferred_element_type=rhs.dtype,
          acc_dtype=acc_dtype,
      )
    else:
      raise NotImplementedError(
          UNSUPPORTED_DIMENSIONS_MSG.format(ragged_dot_dimension_numbers)
      )
    
    # xw32: under what condition is return_residuals True?
    residuals = out
    return out, residuals if return_residuals else None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    # The v2 kernel computes its own VMEM-aware tiling internally (see the
    # `tile_info=...calculate_tiling` calls in `_fwd`), so there is nothing to
    # search here. Return the default `Config`; its tile fields are not consumed
    # by `_fwd` and exist only to satisfy the op-framework config contract.
    del ba
    return Config()

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 6
