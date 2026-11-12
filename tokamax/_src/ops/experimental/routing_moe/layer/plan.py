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

"""Tunable parameters of the routing MoE layer.

`RoutingMoEPlan` exposes optional overrides where `None` delegates to
heuristics, while `ResolvedMoEPlan` holds the concrete parameters consumed by
layer config construction. Plan fields affect performance without changing the
output value, shape, dtype, or sharding of MoE.
"""

import dataclasses
from typing import Self
import jax
from tokamax._src.ops.experimental.routing_moe.common import dataclass_util
from tokamax._src.ops.experimental.routing_moe.common import out_format


@dataclass_util.require_none_default
@dataclasses.dataclass(frozen=True, kw_only=True)
class RoutingMoEPlan:
  """User-provided plan where None fields will be determined by heuristic."""

  # Whether routing_gmm scatters output rows directly to each token's owner.
  remote_scatter: bool | None = None
  # Output format of input_pack fed into routing_gmm's gather input.
  gather_format: out_format.ActivationFormat | None = None
  # Output format of routing_gmm fed into output_reduce_unpack.
  scatter_format: out_format.ActivationFormat | None = None


@dataclass_util.resolves_from(RoutingMoEPlan)
@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedMoEPlan:
  """Plan with None fields from user-provided plan resolved by heuristic."""

  remote_scatter: bool
  gather_format: out_format.ResolvedActivationFormat
  scatter_format: out_format.ResolvedActivationFormat

  @classmethod
  def create(
      cls,
      plan: RoutingMoEPlan | None,
      *,
      x_dtype: jax.typing.DTypeLike,
  ) -> Self:
    """Resolves unset plan fields into concrete execution parameters."""
    if plan is None:
      plan = RoutingMoEPlan()

    remote_scatter = plan.remote_scatter
    if remote_scatter is None:
      remote_scatter = False
    gather_format = out_format.ResolvedActivationFormat.create(
        plan.gather_format,
        default_dtype=x_dtype,
    )
    scatter_format = out_format.ResolvedActivationFormat.create(
        plan.scatter_format,
        default_dtype=x_dtype,
        default_quant=out_format.ResolvedLaneStrided(
            block_col_size=out_format.tile_size
        ),
    )

    return cls(
        remote_scatter=remote_scatter,
        gather_format=gather_format,
        scatter_format=scatter_format,
    )
