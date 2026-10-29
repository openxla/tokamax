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

"""Resolved static configuration of the routing MoE layer."""

import dataclasses
import enum
from typing import Self

from jax import sharding
from tokamax._src.ops.experimental.routing_moe.layer import plan as plan_lib

Mesh = sharding.Mesh
P = sharding.PartitionSpec


class OutputCollective(enum.StrEnum):
  """Cross-device collective applied to unpacked MoE outputs."""

  NONE = enum.auto()
  ALL_GATHER = enum.auto()
  REDUCE_SCATTER = enum.auto()
  ALL_REDUCE = enum.auto()


@dataclasses.dataclass(frozen=True)
class RoutingMoEConfig:
  """Compile-time static configuration of the routing MoE layer.

  Attributes:
    tp_axis: Tensor parallelism mesh axis name, or `None` if disabled.
    ep_axis: Expert parallelism mesh axis name, or `None` if disabled.
    use_sp: Whether inputs are sequence-parallel along the reduction axis.
    reduce_output: Whether to reduce the output across devices.
    activation: Activation function applied to the gate/up projection.
    plan: Resolved execution plan for the MoE layer.
  """

  tp_axis: str | None
  ep_axis: str | None
  use_sp: bool
  reduce_output: bool
  activation: str
  plan: plan_lib.ResolvedMoEPlan

  @classmethod
  def create(
      cls,
      *,
      tp_axis: str | None,
      ep_axis: str | None,
      use_sp: bool,
      reduce_output: bool,
      activation: str,
      plan: plan_lib.ResolvedMoEPlan,
      mesh: Mesh,
  ) -> Self:
    """Assembles the config and validates the plan against layer parameters."""

    if tp_axis is not None and tp_axis not in mesh.shape:
      raise ValueError(
          f"{tp_axis=} is provided but it is not present at {mesh.shape=}"
      )
    if ep_axis is not None and ep_axis not in mesh.shape:
      raise ValueError(
          f"{ep_axis=} is provided but it is not present at {mesh.shape=}"
      )
    if plan.remote_scatter:
      if ep_axis is None:
        raise ValueError("remote_scatter requires ep_axis.")
      if tp_axis is not None:
        raise NotImplementedError(
            "remote_scatter with hybrid parallelism is not yet supported."
        )
      if not reduce_output:
        raise ValueError(
            "remote_scatter cannot be True when reduce_output is False."
        )

    if ep_axis is not None and tp_axis is not None:
      # Supporting (TP, EP) is possible but number of potential use cases is too
      # small to justify complexities on supporting both (EP, TP) and (TP, EP).
      ep_axis_idx = mesh.axis_names.index(ep_axis)
      tp_axis_idx = mesh.axis_names.index(tp_axis)

      if ep_axis_idx > tp_axis_idx:
        raise NotImplementedError(
            "Only supports hybrid parallelism where ep_axis comes before"
            f" tp_axis. Currently {ep_axis_idx=} and {tp_axis_idx=}"
        )

    return cls(
        tp_axis=tp_axis,
        ep_axis=ep_axis,
        use_sp=use_sp,
        reduce_output=reduce_output,
        activation=activation,
        plan=plan,
    )

  @property
  def reduction_axis(self) -> tuple[str, ...]:
    """The mesh axes over which partial MoE outputs are combined."""
    return tuple(ax for ax in (self.ep_axis, self.tp_axis) if ax is not None)

  @property
  def act_spec(self) -> P:
    """The PartitionSpec of the input activations and top-k routing tensors."""
    return P(self.reduction_axis) if self.use_sp else P(None)

  @property
  def out_spec(self) -> P:
    """The PartitionSpec of the layer output."""
    if self.reduce_output:
      return self.act_spec
    # Ideally, we want to use P(unreduced={axis}) but that is only supported in
    # explicit mesh mode. Instead, we create an axis where output is unreduced.
    # TODO: Make gmm work in explicit mesh mode.
    return P(self.reduction_axis)

  @property
  def output_collective(self) -> OutputCollective:
    """The cross-device collective applied to unpacked MoE outputs."""
    unreduced = set(self.reduction_axis)
    if self.plan.remote_scatter:
      output_spec = P(self.reduction_axis)
    else:
      output_spec = P(None, unreduced=unreduced)

    if self.reduce_output:
      expected_out_spec = self.act_spec
    else:
      expected_out_spec = P(None, unreduced=unreduced)

    if output_spec == expected_out_spec:
      return OutputCollective.NONE
    if not output_spec.unreduced:
      return OutputCollective.ALL_GATHER
    if expected_out_spec == P(self.reduction_axis):
      return OutputCollective.REDUCE_SCATTER
    return OutputCollective.ALL_REDUCE
