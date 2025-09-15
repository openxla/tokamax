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
"""Configuration options."""

import contextlib
import dataclasses
import threading
from typing import Generic, TypeVar

from absl import flags


_ABSENT = object()
_STATE = threading.local()
_T = TypeVar("_T")


@contextlib.contextmanager
def _option_override_scope(name, value):
  prev_value = getattr(_STATE, name, _ABSENT)
  try:
    setattr(_STATE, name, value)
    yield
  finally:
    if prev_value is _ABSENT:
      delattr(_STATE, name)
    else:
      setattr(_STATE, name, prev_value)


@dataclasses.dataclass(frozen=True)
class _ConfigOption(Generic[_T]):
  """A configuration option."""

  flag: flags.FlagHolder[_T]

  def __call__(self, value: _T) -> contextlib.AbstractContextManager[None]:
    name = self.flag.name
    flag = self.flag._flagvalues[name]
    try:
      value = flag.parser.parse(value)
    except ValueError as e:
      raise ValueError(f"Invalid value for config `{name}`: {value}") from e

    return _option_override_scope(name, value)

  @property
  def value(self) -> _T:
    return getattr(_STATE, self.flag.name, self.flag.value)


autotuning_cache_miss_fallback = _ConfigOption(
    flags.DEFINE_enum(
        "tokamax_autotuning_cache_miss_fallback",
        "heuristics",
        ("heuristics", "autotune", "error"),
        "Fallback when no config is found in the autotuning cache by"
        " `BoundArguments.default_config` ('heuristics' - use heuristics to"
        " create a config; 'autotune' - autotune over the default autotuning"
        " configs and use the fastest; 'error' - raise an error)",
    )
)


cross_compile = _ConfigOption(
    flags.DEFINE_bool(
        "tokamax_cross_compile",
        False,
        "With this option the user can disable checks like"
        " `has_triton_support()` and `has_mosaic_gpu_support()` that check that"
        " the correct hardware is present. It is possible that the machine"
        " where kernels are lowered and compiled is not the same as the machine"
        " where they are run. `has_mosaic_gpu_support()` that check that the"
        " correct hardware is present.",
    )
)
