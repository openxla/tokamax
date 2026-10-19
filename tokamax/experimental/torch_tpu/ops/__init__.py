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
from tokamax.experimental.torch_tpu.ops.fused_moe.torch_pallas_mosaic_tpu import PallasMosaicTpuFusedMoe as fused_moe
from tokamax.experimental.torch_tpu.ops.linear_softmax_cross_entropy.torch_pallas_mosaic_tpu import PallasMosaicTpuLinearSoftmaxCrossEntropyLoss as linear_softmax_cross_entropy_loss
from tokamax.experimental.torch_tpu.ops.ragged_dot.torch_pallas_mosaic_tpu import PallasMosaicTpuRaggedDot as ragged_dot
from tokamax.experimental.torch_tpu.ops.ragged_gather.torch_pallas_mosaic_tpu import PallasMosaicTpuRaggedGather as ragged_gather
from tokamax.experimental.torch_tpu.ops.topk.torch_pallas_mosaic_tpu import PallasMosaicTpuTopK as topk

# pylint: enable=g-importing-member,useless-import-alias
