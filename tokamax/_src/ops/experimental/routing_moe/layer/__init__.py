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

"""Public API of the routing MoE layer."""

from tokamax._src.ops.experimental.routing_moe.common import out_format
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.layer import modular
from tokamax._src.ops.experimental.routing_moe.layer import monolithic
from tokamax._src.ops.experimental.routing_moe.layer import plan as moe_plan

routing_moe = modular.routing_moe
gated_routing_moe = monolithic.gated_routing_moe

RoutingMoEPlan = moe_plan.RoutingMoEPlan
RoutingGmmPlan = gmm_plan.RoutingGmmPlan

ActivationFormat = out_format.ActivationFormat
QuantFormat = out_format.QuantFormat
AutoQuant = out_format.AutoQuant
auto_quant = out_format.auto_quant
Subchannel = out_format.Subchannel
LaneStrided = out_format.LaneStrided
TileSize = out_format.TileSize
tile_size = out_format.tile_size
