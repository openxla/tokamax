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

from tokamax._src.ops.experimental.nsa.api import native_sparse_attention
from tokamax._src.ops.experimental.nsa.arg_specs import ARG_SPECS
from tokamax._src.ops.experimental.nsa.base import (
    NativeSparseAttention,
    nsa_reference,
)
from tokamax._src.ops.experimental.nsa.pallas_mosaic_tpu import (
    Config,
    PallasTpuNativeSparseAttention,
)
from tokamax._src.ops.experimental.nsa.pallas_mosaic_tpu_kernel import (
    nsa_pallas_fwd,
)

nsa = PallasTpuNativeSparseAttention()

__all__ = [
    "ARG_SPECS",
    "Config",
    "NativeSparseAttention",
    "PallasTpuNativeSparseAttention",
    "native_sparse_attention",
    "nsa",
    "nsa_pallas_fwd",
    "nsa_reference",
]
