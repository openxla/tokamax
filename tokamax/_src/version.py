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

"""Tokamax version information."""

from typing import Final

TOKAMAX_VERSION: Final[str] = "0.0.12"


def _version_as_tuple(version_str: str) -> tuple[int, int, int]:
  return tuple(int(i) for i in version_str.split(".") if i.isdigit())


TOKAMAX_VERSION_INFO: Final[tuple[int, int, int]] = _version_as_tuple(
    TOKAMAX_VERSION
)
