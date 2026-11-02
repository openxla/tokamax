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

"""Public API of the routing GMM kernel."""

from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import out_format
from tokamax._src.ops.experimental.routing_moe.common import spec as tensor_spec
from tokamax._src.ops.experimental.routing_moe.gmm import kernel
from tokamax._src.ops.experimental.routing_moe.gmm import plan

routing_gmm = kernel.routing_gmm
RoutingGmmPlan = plan.RoutingGmmPlan

ActivationWrapper = arg.ActivationWrapper
SingleWeightWrapper = arg.SingleWeightWrapper
RoutingWrapper = arg.RoutingWrapper

ActivationSpec = tensor_spec.ActivationSpec
WeightSpec = tensor_spec.WeightSpec
RoutingSpec = tensor_spec.RoutingSpec
QuantSpec = tensor_spec.QuantSpec
SubchannelQuantSpec = tensor_spec.SubchannelQuantSpec
LaneStridedQuantSpec = tensor_spec.LaneStridedQuantSpec

ActivationFormat = out_format.ActivationFormat
QuantFormat = out_format.QuantFormat
AutoQuant = out_format.AutoQuant
auto_quant = out_format.auto_quant
Subchannel = out_format.Subchannel
LaneStrided = out_format.LaneStrided
TileSize = out_format.TileSize
tile_size = out_format.tile_size
