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
"""De-Dup MoE: a topology-aware de-duplicating expert-parallel MoE layer.

The layer dispatches tokens to their experts in expert waves over a
split-phase ragged all-to-all and, given a `BoardTopology`, de-duplicates the
inter-board traffic: a token needed by several devices of a remote board is
sent once to a gateway device on that board, which forwards it within the
board. See README.md for the design and the roadmap.

Entry points:
  * `dedup_moe(x, router_logits, prepared, cfg)`: the model-facing layer,
    which picks one configuration per t bucket (`DedupMoeConfig.policy`).
  * `dedup_ep_moe(...)`: the monolithic layer with every knob explicit.
"""

from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import plan as kernel_plan
from tokamax._src.ops.experimental.dedup_moe.layer import config as layer_config
from tokamax._src.ops.experimental.dedup_moe.layer import modular
from tokamax._src.ops.experimental.dedup_moe.layer import monolithic
from tokamax._src.ops.experimental.dedup_moe.layer import plan as layer_plan

BoardTopology = topology.BoardTopology
DedupMoeConfig = layer_config.DedupMoeConfig
PreparedMoe = layer_config.PreparedMoe
UnifiedPlan = layer_plan.UnifiedPlan
UnifiedRules = layer_config.UnifiedRules
WeightFormat = arg.WeightFormat
WEIGHT_FORMS = arg.WEIGHT_FORMS
dedup_ep_moe = monolithic.dedup_ep_moe
dedup_moe = modular.dedup_moe
prepare_weights = layer_config.prepare_weights
vmem_estimate_bytes = kernel_plan.vmem_estimate_bytes
vmem_limit = kernel_plan.vmem_limit

__all__ = [
    "BoardTopology",
    "DedupMoeConfig",
    "PreparedMoe",
    "UnifiedPlan",
    "UnifiedRules",
    "WEIGHT_FORMS",
    "WeightFormat",
    "dedup_ep_moe",
    "dedup_moe",
    "prepare_weights",
    "vmem_estimate_bytes",
    "vmem_limit",
]
