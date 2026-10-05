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
"""Causal Conv1D Gated Delta Rule Op API."""

from typing import Final

import immutabledict
from tokamax._src.ops.causal_conv1d_gated_delta_rule import base

_IMPLEMENTATIONS: dict[str, base.CausalConv1dGatedDeltaRule] = dict(
    xla=base.CausalConv1dGatedDeltaRule(),
)

try:
  from tokamax._src.ops.causal_conv1d_gated_delta_rule import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasMosaicTpuCausalConv1dGatedDeltaRule()
  )
except ImportError:
  pass

IMPLEMENTATIONS: Final[
    immutabledict.immutabledict[str, base.CausalConv1dGatedDeltaRule]
] = immutabledict.immutabledict(_IMPLEMENTATIONS)

