# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""FlexAttention API."""

from typing import Final
from tokamax._src.ops.flex_attention import base


IMPLEMENTATIONS: Final[dict[str, base.FlexAttention]] = dict(
    xla=base.FlexAttention(),
)

try:
  from tokamax._src.ops.flex_attention import pallas_triton  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS['triton'] = pallas_triton.PallasTritonGatedLinearUnit()
except ImportError:
  pass
