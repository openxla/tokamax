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

"""PyTree wrappers pairing kernel tensor data with static specifications."""

import dataclasses
from typing import Self

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import spec as spec_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ActivationWrapper[T]:
  """Activation data in compact layout together with its spec."""

  data: T
  scale: T | None
  spec: spec_lib.ActivationSpec = dataclasses.field(metadata=dict(static=True))

  def __post_init__(self) -> None:
    # Skip validation for non jax.Array.
    if not isinstance(self.data, jax.Array):
      return
    if self.data.shape != self.spec.physical_compact_shape:
      raise ValueError(
          "Activation data must have shape"
          f" {self.spec.physical_compact_shape}. Got {self.data.shape}."
      )
    if self.data.dtype != self.spec.dtype:
      raise ValueError(
          f"Activation data must have dtype {self.spec.dtype}. Got"
          f" {self.data.dtype}."
      )
    if self.scale is not None and self.spec.quant is None:
      raise ValueError(
          "Activation scale requires spec.quant to be set. Got"
          f" scale={self.scale is not None}, quant={self.spec.quant}."
      )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SingleWeightWrapper[T]:
  """Expert weight, scale, and bias together with their spec."""

  weight: T
  scale: T | None
  bias: T | None
  spec: spec_lib.WeightSpec = dataclasses.field(metadata=dict(static=True))

  def __post_init__(self) -> None:
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

  @classmethod
  def create(
      cls,
      weight: jax.Array,  # [group, in, out]
      scale: jax.Array | None = None,  # [group, num_blocks, 1, out]
      bias: jax.Array | None = None,  # [group, 1, out]
      *,
      fused_gate_up: bool = False,
  ) -> Self:
    """Builds a weight wrapper whose spec is derived from the given arrays."""
    group_size, in_size, out_size = weight.shape
    quant = None
    if scale is not None:
      quant = spec_lib.SubchannelQuantSpec(block_size=in_size // scale.shape[1])
    weight_spec = spec_lib.WeightSpec(
        group_size=group_size,
        in_size=in_size,
        out_size=out_size,
        dtype=weight.dtype,
        quant=quant,
        has_bias=bias is not None,
        fused_gate_up=fused_gate_up,
    )
    return cls(weight=weight, scale=scale, bias=bias, spec=weight_spec)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RoutingWrapper[T]:
  """Row indices together with the spec of the map they define."""

  indices: T
  spec: spec_lib.RoutingSpec = dataclasses.field(metadata=dict(static=True))

  def __post_init__(self) -> None:
    if not isinstance(self.indices, jax.Array):
      return
    if self.indices.ndim != 1 or self.indices.shape[0] != self.spec.num_indices:
      raise ValueError(
          f"Routing indices must be 1D of length {self.spec.num_indices}. Got"
          f" shape {self.indices.shape}."
      )
    if self.indices.dtype != jnp.int32:
      raise ValueError(
          f"Routing indices must be int32. Got {self.indices.dtype}."
      )

  @classmethod
  def create(
      cls,
      indices: jax.Array,  # [num_indices]
      *,
      remote_axis: str | None = None,
  ) -> Self:
    """Builds a routing wrapper whose spec is derived from indices."""
    routing_spec = spec_lib.RoutingSpec(
        num_indices=indices.shape[0], remote_axis=remote_axis
    )
    return cls(indices=indices, spec=routing_spec)
