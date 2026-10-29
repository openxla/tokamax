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
"""Tokamax GPU operation implementations and configurations."""

# pylint: disable=g-importing-member,useless-import-alias
from tokamax._src.ops.attention.pallas_mosaic_gpu import Config as PallasMosaicFlashAttentionConfig
from tokamax._src.ops.attention.pallas_mosaic_gpu import PallasMosaicGpuFlashAttention as PallasMosaicFlashAttention
from tokamax._src.ops.attention.pallas_triton import Config as PallasTritonFlashAttentionConfig
from tokamax._src.ops.attention.pallas_triton import PallasTritonFlashAttention as PallasTritonFlashAttention
from tokamax._src.ops.gated_linear_unit.pallas_mosaic_gpu import Config as PallasMosaicGatedLinearUnitConfig
from tokamax._src.ops.gated_linear_unit.pallas_mosaic_gpu import PallasMosaicGpuGatedLinearUnit as PallasMosaicGatedLinearUnit
from tokamax._src.ops.gated_linear_unit.triton import Config as PallasTritonGatedLinearUnitConfig
from tokamax._src.ops.gated_linear_unit.triton import TritonGatedLinearUnit as PallasTritonGatedLinearUnit
from tokamax._src.ops.normalization.triton import Config as NormalizationConfig
from tokamax._src.ops.normalization.triton import TritonNormalization as Normalization
from tokamax._src.ops.ragged_dot.pallas_mosaic_gpu import Config as PallasMosaicRaggedDotConfig
from tokamax._src.ops.ragged_dot.pallas_mosaic_gpu import PallasMosaicGpuRaggedDot as PallasMosaicRaggedDot
from tokamax._src.ops.ragged_dot.triton import Config as PallasTritonRaggedDotConfig
from tokamax._src.ops.ragged_dot.triton import TritonRaggedDot as PallasTritonRaggedDot

# pylint: enable=g-importing-member,useless-import-alias

__all__ = (
    "Normalization",
    "NormalizationConfig",
    "PallasMosaicFlashAttention",
    "PallasMosaicFlashAttentionConfig",
    "PallasMosaicGatedLinearUnit",
    "PallasMosaicGatedLinearUnitConfig",
    "PallasMosaicRaggedDot",
    "PallasMosaicRaggedDotConfig",
    "PallasTritonFlashAttention",
    "PallasTritonFlashAttentionConfig",
    "PallasTritonGatedLinearUnit",
    "PallasTritonGatedLinearUnitConfig",
    "PallasTritonRaggedDot",
    "PallasTritonRaggedDotConfig",
)
