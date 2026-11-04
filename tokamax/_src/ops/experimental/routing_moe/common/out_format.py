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

"""Target representation formats for kernel outputs.

`ActivationFormat` and `QuantFormat` expose optional fields where `None` uses
caller-provided defaults, while `Resolved*` types hold populated values before
concrete `spec.*Spec` types are materialized from tile sizes.
"""

import dataclasses
from typing import Self

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import dataclass_util
from tokamax._src.ops.experimental.routing_moe.common import spec


@dataclasses.dataclass(frozen=True)
class TileSize:
  """Sizes each quantization block to the kernel's output tile width."""


tile_size = TileSize()


@dataclasses.dataclass(frozen=True)
class AutoQuant:
  """Selects the kernel's default quantization format."""


auto_quant = AutoQuant()


@dataclass_util.require_none_default
@dataclasses.dataclass(frozen=True, kw_only=True)
class Subchannel:
  """User-provided subchannel quantization format where None uses defaults."""

  block_size: int | TileSize | None = None

  def __post_init__(self) -> None:
    _validate_block_size(self.block_size)


@dataclass_util.resolves_from(Subchannel)
@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedSubchannel:
  """Subchannel quantization format with None fields resolved by defaults."""

  block_size: int | TileSize

  def __post_init__(self) -> None:
    _validate_block_size(self.block_size)


@dataclass_util.require_none_default
@dataclasses.dataclass(frozen=True, kw_only=True)
class LaneStrided:
  """User-provided lane-strided quantization format where None uses defaults."""

  block_col_size: int | TileSize | None = None

  def __post_init__(self) -> None:
    _validate_block_col_size(self.block_col_size)


@dataclass_util.resolves_from(LaneStrided)
@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedLaneStrided:
  """Lane-strided quantization format with None fields resolved by defaults."""

  block_col_size: int | TileSize

  def __post_init__(self) -> None:
    _validate_block_col_size(self.block_col_size)


type QuantFormat = AutoQuant | Subchannel | LaneStrided
type ResolvedQuantFormat = ResolvedSubchannel | ResolvedLaneStrided


@dataclass_util.require_none_default
@dataclasses.dataclass(frozen=True, kw_only=True)
class ActivationFormat:
  """User-provided activation format where None fields use defaults."""

  dtype: jax.typing.DTypeLike | None = None
  quant: QuantFormat | None = None

  def __post_init__(self) -> None:
    if self.quant is not None and self.dtype is None:
      raise ValueError("dtype must be specified when quant is set.")


@dataclass_util.resolves_from(ActivationFormat)
@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedActivationFormat:
  """Activation format with None fields resolved by defaults."""

  dtype: jnp.dtype
  quant: ResolvedQuantFormat | None

  @classmethod
  def create(
      cls,
      fmt: ActivationFormat | Self | None,
      *,
      default_dtype: jax.typing.DTypeLike,
      default_quant: ResolvedQuantFormat | None = None,
  ) -> Self:
    """Resolves unset format fields using default_dtype and default_quant."""
    if isinstance(fmt, cls):
      return fmt
    dtype = (
        jnp.dtype(default_dtype)
        if fmt is None or fmt.dtype is None
        else jnp.dtype(fmt.dtype)
    )
    quant = None
    if fmt is not None and fmt.quant is not None:
      quant = _resolve_quant(fmt.quant, default_quant)
    return cls(dtype=dtype, quant=quant)


type ActivationFormatLike = ActivationFormat | ResolvedActivationFormat


def _validate_block_size(block_size: int | TileSize | None) -> None:
  """Validates a subchannel quantization `block_size`."""
  if isinstance(block_size, bool) or (
      isinstance(block_size, int) and block_size <= 0
  ):
    raise ValueError(
        f"block_size must be a positive int or tile_size. Got {block_size}."
    )


def _validate_block_col_size(block_col_size: int | TileSize | None) -> None:
  """Validates a lane-strided quantization `block_col_size`."""
  lanes = 128
  if isinstance(block_col_size, bool) or (
      isinstance(block_col_size, int)
      and (block_col_size <= 0 or block_col_size % lanes != 0)
  ):
    raise ValueError(
        f"block_col_size must be a positive multiple of {lanes} or tile_size."
        f" Got {block_col_size}."
    )


def _resolve_quant(
    quant: QuantFormat,
    default_quant: ResolvedQuantFormat | None,
) -> ResolvedQuantFormat:
  """Populates unset quantization fields from `default_quant`."""
  if default_quant is None:
    raise NotImplementedError("Quantization is not supported.")
  match quant:
    case AutoQuant():
      return default_quant
    case Subchannel(block_size=block_size):
      if block_size is None:
        if not isinstance(default_quant, ResolvedSubchannel):
          raise ValueError(
              "Subchannel.block_size must be set when default quant is"
              f" {type(default_quant).__name__}."
          )
        block_size = default_quant.block_size
      return ResolvedSubchannel(block_size=block_size)
    case LaneStrided(block_col_size=block_col_size):
      if block_col_size is None:
        if not isinstance(default_quant, ResolvedLaneStrided):
          raise ValueError(
              "LaneStrided.block_col_size must be set when default quant is"
              f" {type(default_quant).__name__}."
          )
        block_col_size = default_quant.block_col_size
      return ResolvedLaneStrided(block_col_size=block_col_size)


def _resolve_tile_size(size: int | TileSize, tile_col_size: int) -> int:
  """Replaces symbolic `TileSize` with `tile_col_size`."""
  return tile_col_size if isinstance(size, TileSize) else size


def to_quant_spec(
    quant: ResolvedQuantFormat | None, *, tile_col_size: int
) -> spec.QuantSpec | None:
  """Materializes a resolved quantization format for `tile_col_size`."""
  match quant:
    case None:
      return None
    case ResolvedSubchannel(block_size=block_size):
      return spec.SubchannelQuantSpec(
          block_size=_resolve_tile_size(block_size, tile_col_size)
      )
    case ResolvedLaneStrided(block_col_size=block_col_size):
      return spec.LaneStridedQuantSpec(
          block_col_size=_resolve_tile_size(block_col_size, tile_col_size)
      )
