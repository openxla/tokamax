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

"""PyTorch Raven Gated Slot Attention (GSA) operator."""

from tokamax.experimental.torch_tpu.ops.raven.torch_base import _RavenGSA
from tokamax.experimental.torch_tpu.ops.raven.torch_pallas_mosaic_tpu import (
    _PallasTpuRavenGSA,
    raven_gsa,
)

raven_pallas_gsa = raven_gsa

__all__ = [
    "_RavenGSA",
    "_PallasTpuRavenGSA",
    "raven_gsa",
    "raven_pallas_gsa",
]
