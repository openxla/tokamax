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

"""Gated Delta Net 2 (GDN-2) Tokamax operator."""

from tokamax._src.ops.experimental.gdn2.api import (
    gated_delta_net_2,
)
from tokamax._src.ops.experimental.gdn2.arg_specs import (
    ARG_SPECS,
)
from tokamax._src.ops.experimental.gdn2.base import (
    GatedDeltaNet2,
    gdn2_reference,
)
from tokamax._src.ops.experimental.gdn2.pallas_mosaic_tpu import (
    Config,
    PallasTpuGatedDeltaNet2,
)

__all__ = [
    "ARG_SPECS",
    "Config",
    "GatedDeltaNet2",
    "PallasTpuGatedDeltaNet2",
    "gated_delta_net_2",
    "gdn2_reference",
]
