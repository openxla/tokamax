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

"""Static specifications of materialized kernel inputs and outputs.

A spec describes the concrete shape, dtype, and layout of a tensor and is
defined by the tensor's producer.
"""

import dataclasses
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import dataclass_util
from tokamax._src.ops.experimental.routing_moe.common import layout_util


@dataclass_util.require_no_default
@dataclasses.dataclass(frozen=True)
class SubchannelQuantSpec:
  """Contiguous blocks of `block_size` elements share a scale."""

  block_size: int

  def __post_init__(self) -> None:
    if self.block_size <= 0:
      raise ValueError(f"block_size must be positive. Got {self.block_size}.")


@dataclass_util.require_no_default
@dataclasses.dataclass(frozen=True)
class LaneStridedQuantSpec:
  """Elements strided by 128 lanes in `block_col_size` columns share a scale."""

  # Unlike regular subchannel quantization - where a single scale is shared by a
  # continuous region of memory - in lane-strided quantization a single scale is
  # shared by elements with a stride of the hardware lane size. Computing the
  # scale therefore requires no expensive cross-lane reduction and eliminates
  # lane padding.
  #
  # For example, assume the following data with a HW lane size of 2, where ||
  # denotes a HW lane boundary and | denotes an element boundary.
  # || a | b || c | d || e | f || g | h ||
  #
  # Regular subchannel quantization with block size of 4 runs:
  #
  # Step 1: Split along blocks. (no-op: register aliasing)
  # block 0: || a | b || c | d ||
  # block 1: || e | f || g | h ||
  #
  # Step 2: Split along lanes. (no-op: register aliasing)
  # block 0.0: || a | b ||
  # block 0.1: || c | d ||
  # block 1.0: || e | f ||
  # block 1.1: || g | h ||
  #
  # Step 3: Elementwise reduction. (cheap)
  # block 0: || a/c | b/d ||
  # block 1: || e/g | f/h ||
  #
  # Step 4: Cross lane reduction. (expensive)
  # block 0: || a/c/b/d | N/A ||
  # block 1: || e/g/f/h | N/A ||
  #
  # Lane-strided quantization instead runs:
  #
  # Step 1: Split along lanes. (no-op: register aliasing)
  # 0: || a | b ||
  # 1: || c | d ||
  # 2: || e | f ||
  # 3: || g | h ||
  #
  # Step 2: Elementwise reduction. (cheap)
  # || a/c/e/g | b/d/f/h ||
  #
  # Compared to subchannel quantization, this approach has two benefits:
  # 1. There is no expensive cross-lane reduction.
  # 2. Every lane entry stores useful scale data.
  #
  # Note that unlike subchannel quantization, the number of elements sharing a
  # scale (block_col_size // num_lanes) scales with block_col_size.
  block_col_size: int

  def __post_init__(self) -> None:
    lanes = 128
    if self.block_col_size <= 0 or self.block_col_size % lanes != 0:
      raise ValueError(
          f"block_col_size must be a positive multiple of {lanes}. Got"
          f" {self.block_col_size}."
      )


type QuantSpec = SubchannelQuantSpec | LaneStridedQuantSpec


@dataclass_util.require_no_default
@dataclasses.dataclass(frozen=True)
class BlockScaledQuantSpec:
  """Strided block-scaled weights with 3D `[group, scale_rows, out]` scales.

  Unlike contiguous `SubchannelQuantSpec` or single-element
  `LaneStridedQuantSpec`, each `float8_e8m0fnu` scale groups strided
  multi-element segments along `in_size`, and `scale_rows` may include hardware
  alignment padding.
  """

  scale_rows: int

  def __post_init__(self) -> None:
    if self.scale_rows <= 0:
      raise ValueError(f"scale_rows must be positive. Got {self.scale_rows}.")


type WeightQuantSpec = SubchannelQuantSpec | BlockScaledQuantSpec


@dataclass_util.require_no_default
@dataclasses.dataclass(frozen=True)
class ActivationSpec:
  """Static shape, dtype, and quantization spec of a compact activation."""

  row_size: int
  # Logical column size. Physical column size is `col_size + quant_col_size`.
  col_size: int
  dtype: jnp.dtype
  quant: QuantSpec | None

  def __post_init__(self) -> None:
    match self.quant:
      case None:
        return
      case SubchannelQuantSpec(block_size=block_size):
        if self.col_size % block_size != 0:
          raise ValueError(
              f"quant block_size ({block_size}) must divide col_size"
              f" ({self.col_size})."
          )
      case LaneStridedQuantSpec(block_col_size=block_col_size):
        if self.col_size % block_col_size != 0:
          raise ValueError(
              f"quant block_col_size ({block_col_size}) must divide col_size"
              f" ({self.col_size})."
          )

  @property
  def quant_col_size(self) -> int:
    """Extra physical column width in dtype elements for embedded scales."""
    match self.quant:
      case None:
        return 0
      case SubchannelQuantSpec():
        raise NotImplementedError(
            "Only LaneStridedQuantSpec is supported for compact activations."
        )
      case LaneStridedQuantSpec(block_col_size=block_col_size):
        if block_col_size != self.col_size:
          raise NotImplementedError(
              "Sub-row LaneStridedQuantSpec is not supported yet."
          )
        lanes = 128
        return lanes * layout_util.get_packing(self.dtype)

  @property
  def num_logical_chunks(self) -> int:
    return self.logical_compact_shape[1]

  @property
  def num_physical_chunks(self) -> int:
    return self.physical_compact_shape[1]

  @property
  def logical_compact_shape(self) -> tuple[int, int, int, int]:
    return layout_util.to_compact_shape(
        (self.row_size, self.col_size), self.dtype
    )

  @property
  def physical_compact_shape(self) -> tuple[int, int, int, int]:
    return layout_util.to_compact_shape(
        (self.row_size, self.col_size + self.quant_col_size), self.dtype
    )

  @property
  def large_shape(self) -> tuple[int, int]:
    # Embedded scales only exist in compact layout, so 2D large shape is
    # always logical.
    return (self.row_size, self.col_size)


@dataclass_util.require_no_default
@dataclasses.dataclass(frozen=True)
class WeightSpec:
  """Static description of a grouped expert weight tensor."""

  group_size: int
  in_size: int
  out_size: int
  dtype: jnp.dtype
  quant: WeightQuantSpec | None
  has_bias: bool
  # Columns hold the gate and up weights concatenated, each out_size // 2 wide.
  fused_gate_up: bool

  def __post_init__(self) -> None:
    match self.quant:
      case None:
        pass
      case SubchannelQuantSpec(block_size=block_size):
        if self.in_size % block_size != 0:
          raise ValueError(
              f"quant block_size ({block_size}) must divide in_size"
              f" ({self.in_size})."
          )
      case BlockScaledQuantSpec():
        fp8_dtypes = (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2))
        if self.dtype not in fp8_dtypes:
          raise ValueError(
              f"BlockScaledQuantSpec requires dtype in {fp8_dtypes}. Got"
              f" {self.dtype}."
          )
    if self.fused_gate_up and self.out_size % 2 != 0:
      raise ValueError(
          f"Fused gate/up weight needs an even out_size. Got {self.out_size}."
      )

  @property
  def has_scale(self) -> bool:
    return self.quant is not None

  @property
  def shape(self) -> tuple[int, int, int]:
    return (self.group_size, self.in_size, self.out_size)

  @property
  def scale_dtype(self) -> jnp.dtype:
    assert self.quant is not None
    match self.quant:
      case SubchannelQuantSpec():
        return jnp.dtype(jnp.float32)
      case BlockScaledQuantSpec():
        return jnp.dtype(jnp.float8_e8m0fnu)

  @property
  def scale_shape(self) -> tuple[int, int, int] | tuple[int, int, int, int]:
    assert self.quant is not None
    match self.quant:
      case SubchannelQuantSpec(block_size=block_size):
        num_blocks = self.in_size // block_size
        return (self.group_size, num_blocks, 1, self.out_size)
      case BlockScaledQuantSpec(scale_rows=scale_rows):
        return (self.group_size, scale_rows, self.out_size)

  @property
  def bias_shape(self) -> tuple[int, int, int]:
    return (self.group_size, 1, self.out_size)


@dataclass_util.require_no_default
@dataclasses.dataclass(frozen=True)
class RoutingSpec:
  """Static description of a row-index map."""

  num_indices: int
  # Mesh axis across which indices address remote devices via collective DMA.
  remote_axis: str | None
