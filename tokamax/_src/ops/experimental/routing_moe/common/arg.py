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

"""Static specifications and PyTree wrappers of kernel inputs and outputs.

A spec describes the shape, dtype, and layout of a tensor and is defined by the
tensor's producer. A wrapper pairs the tensor data with its spec as static
PyTree metadata.
"""

import dataclasses

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import layout_util


@dataclasses.dataclass(frozen=True)
class QuantSpec:
  """Granularity of the scales that accompany a quantized tensor."""

  # Contracting-dim elements sharing one scale value.
  block_size: int


@dataclasses.dataclass(frozen=True)
class ActivationSpec:
  """Static description of an activation tensor stored in compact layout."""

  row_size: int
  col_size: int
  dtype: jnp.dtype
  quant: QuantSpec | None = None
  # When set, rows are this device's token shard along this mesh axis, not the
  # full activation.
  row_shard_axis: str | None = None

  def __post_init__(self):
    if self.quant is not None and self.col_size % self.quant.block_size != 0:
      raise ValueError(
          f"quant block_size ({self.quant.block_size}) must divide col_size"
          f" ({self.col_size})."
      )

  @property
  def compact_shape(self) -> tuple[int, int, int, int]:
    return layout_util.to_compact_shape(
        (self.row_size, self.col_size), self.dtype
    )

  @property
  def large_shape(self) -> tuple[int, int]:
    return (self.row_size, self.col_size)


@dataclasses.dataclass(frozen=True)
class WeightSpec:
  """Static description of a grouped expert weight tensor."""

  group_size: int
  in_size: int
  out_size: int
  dtype: jnp.dtype
  quant: QuantSpec | None = None
  has_bias: bool = False
  # Columns hold the gate and up weights concatenated, each out_size // 2 wide.
  fused_gate_up: bool = False

  def __post_init__(self):
    if self.quant is not None and self.in_size % self.quant.block_size != 0:
      raise ValueError(
          f"quant block_size ({self.quant.block_size}) must divide in_size"
          f" ({self.in_size})."
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
  def scale_shape(self) -> tuple[int, int, int, int]:
    assert self.quant is not None
    num_blocks = self.in_size // self.quant.block_size
    return (self.group_size, num_blocks, 1, self.out_size)

  @property
  def bias_shape(self) -> tuple[int, int, int]:
    return (self.group_size, 1, self.out_size)


@dataclasses.dataclass(frozen=True)
class RoutingSpec:
  """Static description of a row-index map."""

  num_indices: int
  # When set, output will be sharded on batch dim along this mesh axis.
  remote_axis: str | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ActivationWrapper[T]:
  """Activation data in compact layout together with its spec."""

  data: T
  scale: T | None
  spec: ActivationSpec = dataclasses.field(metadata=dict(static=True))

  def __post_init__(self):
    # Skip validation for non jax.Array.
    if not isinstance(self.data, jax.Array):
      return
    if self.data.shape != self.spec.compact_shape:
      raise ValueError(
          f"Activation data must have shape {self.spec.compact_shape}. Got"
          f" {self.data.shape}."
      )
    if self.data.dtype != self.spec.dtype:
      raise ValueError(
          f"Activation data must have dtype {self.spec.dtype}. Got"
          f" {self.data.dtype}."
      )
    if (self.scale is None) != (self.spec.quant is None):
      raise ValueError(
          "Activation scale must be present iff spec.quant is set. Got"
          f" scale={self.scale is not None}, quant={self.spec.quant}."
      )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SingleWeightWrapper[T]:
  """Expert weight, scale, and bias together with their spec."""

  weight: T
  scale: T | None
  bias: T | None
  spec: WeightSpec = dataclasses.field(metadata=dict(static=True))

  def __post_init__(self):
    if not isinstance(self.weight, jax.Array):
      return
    if self.weight.shape != self.spec.shape:
      raise ValueError(
          f"Weight must have shape {self.spec.shape}. Got {self.weight.shape}."
      )
    if self.weight.dtype != self.spec.dtype:
      raise ValueError(
          f"Weight must have dtype {self.spec.dtype}. Got {self.weight.dtype}."
      )
    if (self.scale is None) != (self.spec.quant is None):
      raise ValueError(
          "Weight scale must be present iff spec.quant is set. Got"
          f" scale={self.scale is not None}, quant={self.spec.quant}."
      )
    if self.scale is not None:
      assert isinstance(self.scale, jax.Array)
      if self.scale.shape != self.spec.scale_shape:
        raise ValueError(
            f"Weight scale must have shape {self.spec.scale_shape}. Got"
            f" {self.scale.shape}."
        )
    if (self.bias is not None) != self.spec.has_bias:
      raise ValueError(
          "Weight bias must be present iff spec.has_bias is set. Got"
          f" bias={self.bias is not None}, has_bias={self.spec.has_bias}."
      )
    if self.bias is not None:
      assert isinstance(self.bias, jax.Array)
      if self.bias.shape != self.spec.bias_shape:
        raise ValueError(
            f"Weight bias must have shape {self.spec.bias_shape}. Got"
            f" {self.bias.shape}."
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RoutingWrapper[T]:
  """Row indices together with the spec of the map they define."""

  indices: T
  spec: RoutingSpec = dataclasses.field(metadata=dict(static=True))

  def __post_init__(self):
    if not isinstance(self.indices, jax.Array):
      return
    if self.indices.ndim != 1 or self.indices.shape[0] != self.spec.num_indices:
      raise ValueError(
          f"Routing indices must be 1D with at least {self.spec.num_indices}"
          f" entries. Got shape {self.indices.shape}."
      )
    if self.indices.dtype != jnp.int32:
      raise ValueError(
          f"Routing indices must be int32. Got {self.indices.dtype}."
      )


def make_weight_wrapper(
    weight: jax.Array,  # [group, in, out]
    scale: jax.Array | None = None,  # [group, num_blocks, 1, out]
    bias: jax.Array | None = None,  # [group, 1, out]
    *,
    fused_gate_up: bool = False,
) -> SingleWeightWrapper[jax.Array]:
  """Builds a weight wrapper whose spec is derived from the given arrays."""
  group_size, in_size, out_size = weight.shape
  quant = None
  if scale is not None:
    quant = QuantSpec(block_size=in_size // scale.shape[1])
  spec = WeightSpec(
      group_size=group_size,
      in_size=in_size,
      out_size=out_size,
      dtype=weight.dtype,
      quant=quant,
      has_bias=bias is not None,
      fused_gate_up=fused_gate_up,
  )
  return SingleWeightWrapper(weight=weight, scale=scale, bias=bias, spec=spec)


def make_routing_wrapper(
    indices: jax.Array,  # [num_indices]
    *,
    remote_axis: str | None = None,
) -> RoutingWrapper[jax.Array]:
  """Builds a routing wrapper for a row-index map, local unless remote_axis."""
  spec = RoutingSpec(num_indices=indices.shape[0], remote_axis=remote_axis)
  return RoutingWrapper(indices=indices, spec=spec)
