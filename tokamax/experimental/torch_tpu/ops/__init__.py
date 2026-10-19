# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tokamax integration with PyTorch on TPU."""

# pylint: disable=g-importing-member,useless-import-alias
from tokamax.experimental.torch_tpu.ops import torch_utils as tokamax_torch_utils
from tokamax.experimental.torch_tpu.ops.batched_rpa.torch_pallas_mosaic_tpu import PallasMosaicTpuBatchedRpa as batched_rpa
from tokamax.experimental.torch_tpu.ops.causal_conv1d_gated_delta_rule.torch_pallas_mosaic_tpu import PallasMosaicTpuCausalConv1dGatedDeltaRule as causal_conv1d_gated_delta_rule
from tokamax.experimental.torch_tpu.ops.csa_gather.torch_pallas_mosaic_tpu import PallasMosaicTpuCsaGather as csa_gather
from tokamax.experimental.torch_tpu.ops.deepseek_rope.torch_pallas_mosaic_tpu import PallasMosaicTpuDeepseekRope as deepseek_rope
from tokamax.experimental.torch_tpu.ops.dsv4_compressor.torch_pallas_mosaic_tpu import PallasMosaicTpuDsv4Compressor as dsv4_compressor
from tokamax.experimental.torch_tpu.ops.fused_moe.torch_pallas_mosaic_tpu import PallasMosaicTpuFusedMoe as fused_moe
from tokamax.experimental.torch_tpu.ops.grouped_lowrank_proj.torch_pallas_mosaic_tpu import PallasMosaicTpuGroupedLowrankProj as grouped_lowrank_proj
from tokamax.experimental.torch_tpu.ops.lightning_indexer.torch_pallas_mosaic_tpu import PallasMosaicTpuLightningIndexer as lightning_indexer
from tokamax.experimental.torch_tpu.ops.linear_softmax_cross_entropy.torch_pallas_mosaic_tpu import PallasMosaicTpuLinearSoftmaxCrossEntropyLoss as linear_softmax_cross_entropy_loss
from tokamax.experimental.torch_tpu.ops.masked_dense_mla.torch_pallas_mosaic_tpu import PallasMosaicTpuMaskedDenseMla as masked_dense_mla
from tokamax.experimental.torch_tpu.ops.mhc.torch_pallas_mosaic_tpu import PallasMosaicTpuMhcFusedPostPre as mhc_fused_post_pre
from tokamax.experimental.torch_tpu.ops.mhc.torch_pallas_mosaic_tpu import PallasMosaicTpuMhcPost as mhc_post
from tokamax.experimental.torch_tpu.ops.mhc.torch_pallas_mosaic_tpu import PallasMosaicTpuMhcPre as mhc_pre
from tokamax.experimental.torch_tpu.ops.proj_and_save_state.torch_pallas_mosaic_tpu import PallasMosaicTpuProjAndSaveState as proj_and_save_state
from tokamax.experimental.torch_tpu.ops.ragged_dot.torch_pallas_mosaic_tpu import PallasMosaicTpuRaggedDot as ragged_dot
from tokamax.experimental.torch_tpu.ops.ragged_gather.torch_pallas_mosaic_tpu import PallasMosaicTpuRaggedGather as ragged_gather
from tokamax.experimental.torch_tpu.ops.ragged_gather_reduce.torch_pallas_mosaic_tpu import PallasMosaicTpuRaggedGatherReduce as ragged_gather_reduce
from tokamax.experimental.torch_tpu.ops.router_topk.torch_pallas_mosaic_tpu import PallasMosaicTpuRouterTopK as router_topk
from tokamax.experimental.torch_tpu.ops.sparse_mla.torch_pallas_mosaic_tpu import PallasMosaicTpuSparseMla as sparse_mla
from tokamax.experimental.torch_tpu.ops.splash_attention.torch_pallas_mosaic_tpu import PallasMosaicTpuSplashAttention as splash_attention
from tokamax.experimental.torch_tpu.ops.topk.torch_pallas_mosaic_tpu import PallasMosaicTpuTopK as topk

# pylint: enable=g-importing-member,useless-import-alias
