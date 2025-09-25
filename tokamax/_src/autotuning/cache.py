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
"""Tokamax autotuning cache."""
from typing import Any, TypeAlias

from tokamax._src.autotuning import autotuner

AutotuningData = autotuner.AutotuningData
DeviceKind = str
DeviceAutotuningCache: TypeAlias = dict[Any, AutotuningData[Any]]


class AutotuningCache(dict[DeviceKind, DeviceAutotuningCache]):
  """Cache of autotuning data.

  Autotuning data is read lazily from the cache files upon first access. The
  directory containing the cache files can be specified using TODO!!!
  """

  def __init__(self, op):
    super().__init__()
    self.op = op

  def __missing__(self, device_kind: DeviceKind) -> DeviceAutotuningCache:
    self[device_kind] = (cache := self._load_cache(device_kind))
    return cache

  def _load_cache(self, device_kind: DeviceKind) -> DeviceAutotuningCache:
    """Loads autotuning cache from corresponding JSON file."""
    return {}
