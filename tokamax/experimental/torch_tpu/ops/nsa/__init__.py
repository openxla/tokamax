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

"""PyTorch Native Sparse Attention (NSA) operator."""

from tokamax.experimental.torch_tpu.ops.nsa.torch_base import (
    _NativeSparseAttention,
)
from tokamax.experimental.torch_tpu.ops.nsa.torch_pallas_mosaic_tpu import (
    _PallasTpuNativeSparseAttention,
    nsa,
)

# Export functional and alias names
nsa_pallas_attention = nsa
flash_nsa_pallas = nsa

__all__ = [
    "_NativeSparseAttention",
    "_PallasTpuNativeSparseAttention",
    "nsa",
    "nsa_pallas_attention",
    "flash_nsa_pallas",
]
