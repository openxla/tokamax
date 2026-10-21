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
"""Tokamax TPU operation implementations and configurations."""

# pylint: disable=g-importing-member,useless-import-alias
from tokamax._src.ops.attention.pallas_mosaic_tpu import Config as FlashAttentionConfig
from tokamax._src.ops.attention.pallas_mosaic_tpu import PallasMosaicTpuFlashAttention as FlashAttention
from tokamax._src.ops.batched_rpa.pallas_mosaic_tpu import Config as BatchedRpaConfig
from tokamax._src.ops.batched_rpa.pallas_mosaic_tpu import PallasTpuBatchedRpa as BatchedRpa
from tokamax._src.ops.causal_conv1d_gated_delta_rule.pallas_mosaic_tpu import Config as CausalConv1dGatedDeltaRuleConfig
from tokamax._src.ops.causal_conv1d_gated_delta_rule.pallas_mosaic_tpu import PallasMosaicTpuCausalConv1dGatedDeltaRule as CausalConv1dGatedDeltaRule
from tokamax._src.ops.csa_gather.pallas_mosaic_tpu import Config as CsaGatherConfig
from tokamax._src.ops.csa_gather.pallas_mosaic_tpu import PallasTpuCsaGather as CsaGather
from tokamax._src.ops.deepseek_rope.pallas_mosaic_tpu import Config as DeepseekRopeConfig
from tokamax._src.ops.deepseek_rope.pallas_mosaic_tpu import PallasTpuDeepseekRope as DeepseekRope
from tokamax._src.ops.dsv4_compressor.pallas_mosaic_tpu import Config as Dsv4CompressorConfig
from tokamax._src.ops.dsv4_compressor.pallas_mosaic_tpu import PallasTpuDsv4Compressor as Dsv4Compressor
from tokamax._src.ops.grouped_lowrank_proj.pallas_mosaic_tpu import Config as GroupedLowrankProjConfig
from tokamax._src.ops.grouped_lowrank_proj.pallas_mosaic_tpu import PallasTpuGroupedLowrankProj as GroupedLowrankProj
from tokamax._src.ops.kda.pallas_mosaic_tpu import Config as KimiDeltaAttentionConfig
from tokamax._src.ops.kda.pallas_mosaic_tpu import PallasMosaicTpuKimiDeltaAttention as KimiDeltaAttention
from tokamax._src.ops.lightning_indexer.pallas_mosaic_tpu import Config as LightningIndexerConfig
from tokamax._src.ops.lightning_indexer.pallas_mosaic_tpu import PallasTpuLightningIndexer as LightningIndexer
from tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu import Config as LinearSoftmaxCrossEntropyLossConfig
from tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu import PallasMosaicTpuLinearSoftmaxCrossEntropyLoss as LinearSoftmaxCrossEntropyLoss
from tokamax._src.ops.masked_dense_mla.pallas_mosaic_tpu import Config as MaskedDenseMlaConfig
from tokamax._src.ops.masked_dense_mla.pallas_mosaic_tpu import PallasTpuMaskedDenseMla as MaskedDenseMla
from tokamax._src.ops.mhc.pallas_mosaic_tpu import Config as MhcConfig
from tokamax._src.ops.mhc.pallas_mosaic_tpu import PallasTpuMhcFusedPostPre as MhcFusedPostPre
from tokamax._src.ops.mhc.pallas_mosaic_tpu import PallasTpuMhcPost as MhcPost
from tokamax._src.ops.mhc.pallas_mosaic_tpu import PallasTpuMhcPre as MhcPre
from tokamax._src.ops.proj_and_save_state.pallas_mosaic_tpu import Config as ProjAndSaveStateConfig
from tokamax._src.ops.proj_and_save_state.pallas_mosaic_tpu import PallasTpuProjAndSaveState as ProjAndSaveState
from tokamax._src.ops.ragged_dot.pallas_mosaic_tpu import Config as RaggedDotConfig
from tokamax._src.ops.ragged_dot.pallas_mosaic_tpu import PallasMosaicTpuRaggedDot as RaggedDot
from tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_v2 import Config as V2RaggedDotConfig
from tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_v2 import PallasMosaicTpuV2RaggedDot as V2RaggedDot
from tokamax._src.ops.ragged_gather.pallas_mosaic_tpu import PallasTpuRaggedGather as RaggedGather
from tokamax._src.ops.ragged_gather_reduce.pallas_mosaic_tpu import Config as RaggedGatherReduceConfig
from tokamax._src.ops.ragged_gather_reduce.pallas_mosaic_tpu import PallasTpuRaggedGatherReduce as RaggedGatherReduce
from tokamax._src.ops.router_topk.pallas_mosaic_tpu import Config as RouterTopKConfig
from tokamax._src.ops.router_topk.pallas_mosaic_tpu import PallasTpuRouterTopK as RouterTopK
from tokamax._src.ops.sparse_mla.pallas_mosaic_tpu import Config as SparseMlaConfig
from tokamax._src.ops.sparse_mla.pallas_mosaic_tpu import PallasTpuSparseMla as SparseMla
from tokamax._src.ops.topk.pallas_mosaic_tpu import Config as TopKConfig
from tokamax._src.ops.topk.pallas_mosaic_tpu import PallasTpuTopK as TopK

# pylint: enable=g-importing-member,useless-import-alias

__all__ = (
    "BatchedRpa",
    "BatchedRpaConfig",
    "CausalConv1dGatedDeltaRule",
    "CausalConv1dGatedDeltaRuleConfig",
    "CsaGather",
    "CsaGatherConfig",
    "DeepseekRope",
    "DeepseekRopeConfig",
    "Dsv4Compressor",
    "Dsv4CompressorConfig",
    "FlashAttention",
    "FlashAttentionConfig",
    "GroupedLowrankProj",
    "GroupedLowrankProjConfig",
    "KimiDeltaAttention",
    "KimiDeltaAttentionConfig",
    "LightningIndexer",
    "LightningIndexerConfig",
    "LinearSoftmaxCrossEntropyLoss",
    "LinearSoftmaxCrossEntropyLossConfig",
    "MaskedDenseMla",
    "MaskedDenseMlaConfig",
    "MhcConfig",
    "MhcFusedPostPre",
    "MhcPost",
    "MhcPre",
    "ProjAndSaveState",
    "ProjAndSaveStateConfig",
    "RaggedDot",
    "RaggedDotConfig",
    "RaggedGather",
    "RaggedGatherReduce",
    "RaggedGatherReduceConfig",
    "RouterTopK",
    "RouterTopKConfig",
    "SparseMla",
    "SparseMlaConfig",
    "TopK",
    "TopKConfig",
    "V2RaggedDot",
    "V2RaggedDotConfig",
)
