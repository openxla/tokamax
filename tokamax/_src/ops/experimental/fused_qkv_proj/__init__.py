# Copyright 2026 Google LLC
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
"""Pallas Mosaic TPU kernels for attention projections around batched_rpa."""

# pylint: disable=g-importing-member,useless-import-alias
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gdn.kernel import fused_gdn as fused_gdn
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa.kernel import fused_qkv as fused_qkv
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj.kernel import fused_out_proj as fused_out_proj

# pylint: enable=g-importing-member,useless-import-alias

__all__ = (
    "fused_gdn",
    "fused_out_proj",
    "fused_qkv",
)
