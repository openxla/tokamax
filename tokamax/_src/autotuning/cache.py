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

from importlib import resources
import re
from typing import Annotated
from typing import Any, Final, Sequence, TypeAlias

from absl import logging
import immutabledict
import pydantic
from tokamax._src import pydantic as pydantic_lib
from tokamax._src.autotuning import autotuner

AutotuningData = autotuner.AutotuningData
DeviceKind = str
DeviceAutotuningCache: TypeAlias = dict[Any, AutotuningData[Any]]


_CACHE_PATHS: Final[tuple[str, ...]] = (
    "data/autotuning",
)


def _get_cache_adapter(op) -> pydantic.TypeAdapter:
  model_name = f"{type(op).__name__}Spec"
  spec = pydantic_lib.get_arg_spec_model(model_name, op._fwd_signature)  # pylint: disable=protected-access
  return pydantic.TypeAdapter(
      dict[
          Annotated[
              pydantic.Json[spec],
              pydantic.AfterValidator(lambda d: immutabledict.immutabledict(d)),  # pylint: disable=unnecessary-lambda
              pydantic.WrapSerializer(lambda d, handler: handler(dict(d))),
          ],
          autotuner.AutotuningData[op.config_cls],
      ]
  )


class AutotuningCache(dict[DeviceKind, DeviceAutotuningCache]):
  """Cache of autotuning data.

  Autotuning data is read lazily from the cache files upon first access.
  """

  def __init__(self, op):
    super().__init__()
    self.op = op

  def __missing__(self, device_kind: DeviceKind) -> DeviceAutotuningCache:
    self[device_kind] = (cache := self._load_cache(device_kind))
    return cache

  def _load_cache(
      self, device_kind: DeviceKind, cache_path: Sequence[str] = _CACHE_PATHS
  ) -> DeviceAutotuningCache:
    """Loads autotuning cache from corresponding JSON files."""

    device_kind = device_kind.lower().replace(" ", "_")
    # Convert to snake case.
    op_name = re.sub(r"(?!^)([A-Z])", r"_\1", type(self.op).__name__).lower()

    tokamax_files = resources.files("tokamax")
    out = {}

    for base_dir in cache_path:
      path = tokamax_files.joinpath(base_dir, device_kind, f"{op_name}.json")
      try:
        json_data = path.read_text()
      except FileNotFoundError:
        logging.info("Autotuning cache file not found: %s", path)
        json_data = "{}"
      # Cache paths later in the list will override earlier ones.
      data = _get_cache_adapter(self.op).validate_json(json_data)
      out |= {
          self.op.bind(**k).autotuning_cache_key: cache
          for k, cache in data.items()
      }
    return out
