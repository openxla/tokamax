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

"""Native Sparse Attention (NSA) Tokamax operators."""

from tokamax._src.ops.experimental.nsa import base
from tokamax._src.ops.experimental.nsa import pallas_mosaic_tpu
from tokamax._src.ops.experimental.nsa import pallas_mosaic_tpu_kernel
from tokamax._src.ops.experimental.nsa import reference

Config = pallas_mosaic_tpu.Config
NativeSparseAttention = base.NativeSparseAttention
PallasTpuNativeSparseAttention = (
    pallas_mosaic_tpu.PallasTpuNativeSparseAttention
)
nsa = PallasTpuNativeSparseAttention()
nsa_reference = reference.nsa_reference
nsa_pallas_fwd = pallas_mosaic_tpu_kernel.nsa_pallas_fwd

__all__ = [
    "Config",
    "NativeSparseAttention",
    "PallasTpuNativeSparseAttention",
    "nsa",
    "nsa_reference",
    "nsa_pallas_fwd",
]
