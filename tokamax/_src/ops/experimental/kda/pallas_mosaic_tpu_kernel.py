# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Low-level Pallas/Mosaic TPU kernels for Kimi Delta Attention."""

from tokamax._src.ops.experimental.kda.pallas_mosaic_tpu_bwd_kernel import (
    chunk_kda_bwd_custom,
)
from tokamax._src.ops.experimental.kda.pallas_mosaic_tpu_fwd_kernel import (
    chunk_kda_fwd_custom,
)

__all__ = [
    "chunk_kda_bwd_custom",
    "chunk_kda_fwd_custom",
]
