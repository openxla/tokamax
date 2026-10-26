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

Plan fields affect performance but not the output value, shape, dtype, or
sharding. Unset fields are resolved by heuristics.
"""

import dataclasses


@dataclasses.dataclass(frozen=True, kw_only=True)
class RoutingMoEPlan:
  """User-provided plan where `None` fields are resolved by heuristics.

  Attributes:
    remote_scatter: Whether `routing_gmm` scatters output rows directly to the
      device that owns each token.
  """

  remote_scatter: bool | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedMoEPlan(RoutingMoEPlan):
  """Plan with `None` fields from user-provided plan resolved by heuristics."""

  remote_scatter: bool

  @classmethod
  def create(cls, plan: RoutingMoEPlan | None):
    """Fills unset plan fields with heuristics."""
    if plan is None:
      plan = RoutingMoEPlan()

    remote_scatter = plan.remote_scatter
    if remote_scatter is None:
      remote_scatter = False
    return cls(remote_scatter=remote_scatter)
