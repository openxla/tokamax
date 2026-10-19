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

"""PyTorch RWKV-7 (DPLR delta rule) operator."""

from tokamax.experimental.torch_tpu.ops.rwkv7.torch_base import _RWKV7
from tokamax.experimental.torch_tpu.ops.rwkv7.torch_pallas_mosaic_tpu import (
    _PallasTpuRWKV7,
    rwkv7,
)

# Export functional and alias names
rwkv7_pallas_delta_rule = rwkv7

__all__ = [
    "_RWKV7",
    "_PallasTpuRWKV7",
    "rwkv7",
    "rwkv7_pallas_delta_rule",
]
