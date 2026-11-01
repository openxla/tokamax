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

"""Decorator that ensures default fileds are None and tunable by heuristic."""

import dataclasses
from typing import Any


def require_none_default[T: type[Any]](cls: T) -> T:
  """Verifies that every field of a plan dataclass defaults to None."""
  for f in dataclasses.fields(cls):
    if f.default is not None or f.default_factory is not dataclasses.MISSING:
      raise ValueError(
          f"{cls.__name__}.{f.name} must default to None so heuristics can"
          " distinguish unset fields from user overrides."
      )
  return cls
