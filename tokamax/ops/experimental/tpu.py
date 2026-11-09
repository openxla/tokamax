# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tokamax experimental TPU operation implementations and configurations."""

# pylint: disable=g-importing-member,useless-import-alias
from tokamax._src.ops.experimental.fused_fp8_matmul.pallas_mosaic_tpu import Config as FusedFp8MatmulConfig
from tokamax._src.ops.experimental.fused_fp8_matmul.pallas_mosaic_tpu import PallasMosaicTpuFusedFp8Matmul as FusedFp8Matmul
from tokamax._src.ops.experimental.fused_moe.pallas_mosaic_tpu import Config as FusedMoeConfig
from tokamax._src.ops.experimental.fused_moe.pallas_mosaic_tpu import PallasMosaicTpuFusedMoe as FusedMoe
from tokamax._src.ops.experimental.mla.pallas_mosaic_tpu import Config as MultiHeadLatentAttentionConfig
from tokamax._src.ops.experimental.mla.pallas_mosaic_tpu import PallasTpuMultiHeadLatentAttention as MultiHeadLatentAttention
from tokamax._src.ops.experimental.mla.pallas_mosaic_tpu_v3 import Config as MultiHeadLatentAttentionV3Config
from tokamax._src.ops.experimental.mla.pallas_mosaic_tpu_v3 import PallasTpuMultiHeadLatentAttentionV3 as MultiHeadLatentAttentionV3
from tokamax._src.ops.experimental.mla_swa.pallas_mosaic_tpu import Config as MlaSwaConfig
from tokamax._src.ops.experimental.mla_swa.pallas_mosaic_tpu import PallasTpuMlaSwa as MlaSwa
from tokamax._src.ops.experimental.routing_moe_op.api import routing_moe as routing_moe
from tokamax._src.ops.experimental.routing_moe_op.api import routing_moe_from_logits as routing_moe_from_logits
from tokamax._src.ops.experimental.routing_moe_op.api import routing_topk as routing_topk
from tokamax._src.ops.experimental.routing_moe_op.api import RoutingMoeFromLogitsWarning as RoutingMoeFromLogitsWarning
from tokamax._src.ops.experimental.routing_moe_op.pallas_mosaic_tpu import Config as RoutingMoeConfig
from tokamax._src.ops.experimental.routing_moe_op.pallas_mosaic_tpu import PallasMosaicTpuRoutingMoe as RoutingMoe

# pylint: enable=g-importing-member,useless-import-alias

__all__ = (
    "FusedFp8Matmul",
    "FusedFp8MatmulConfig",
    "FusedMoe",
    "FusedMoeConfig",
    "MlaSwa",
    "MlaSwaConfig",
    "MultiHeadLatentAttention",
    "MultiHeadLatentAttentionConfig",
    "MultiHeadLatentAttentionV3",
    "MultiHeadLatentAttentionV3Config",
    "RoutingMoe",
    "RoutingMoeConfig",
    "RoutingMoeFromLogitsWarning",
    "routing_moe",
    "routing_moe_from_logits",
    "routing_topk",
)
