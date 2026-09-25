# Copyright 2026 Google LLC. All Rights Reserved.
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

"""PyTorch Mixture of Block Attention (MoBA) operator."""

from tokamax.experimental.torch_tpu.ops.moba.torch_base import (
    _MixtureOfBlockAttention,
)
from tokamax.experimental.torch_tpu.ops.moba.torch_pallas_mosaic_tpu import (
    _PallasTpuMixtureOfBlockAttention,
    moba,
)

# Export functional and alias names
moba_pallas_attention = moba
flash_moba_pallas = moba

__all__ = [
    "_MixtureOfBlockAttention",
    "_PallasTpuMixtureOfBlockAttention",
    "moba",
    "moba_pallas_attention",
    "flash_moba_pallas",
]
