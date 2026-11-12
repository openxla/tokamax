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

"""Decorators enforcing invariants on unresolved and resolved dataclasses."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any


def require_none_default[T: type[Any]](cls: T) -> T:
  """Verifies that every field of a dataclass defaults to None."""
  for f in dataclasses.fields(cls):
    if f.default is not None or f.default_factory is not dataclasses.MISSING:
      raise ValueError(
          f"{cls.__name__}.{f.name} must default to None so unset fields can"
          " be distinguished from explicit user overrides."
      )
  return cls


def require_no_default[T: type[Any]](cls: T) -> T:
  """Verifies that no field of a dataclass has a default value."""
  for f in dataclasses.fields(cls):
    if (
        f.default is not dataclasses.MISSING
        or f.default_factory is not dataclasses.MISSING
    ):
      raise ValueError(
          f"{cls.__name__}.{f.name} must not have a default value so every"
          " field is populated explicitly."
      )
  return cls


def resolves_from[T: type[Any]](
    unresolved_cls: type[Any],
    *,
    dropped: Sequence[str] = (),
) -> Callable[[T], T]:
  """Verifies that a resolved dataclass mirrors `unresolved_cls` in order.

  Args:
    unresolved_cls: Unresolved dataclass whose unset fields default to `None`.
    dropped: Trailing field names on `unresolved_cls` omitted once resolved.

  Returns:
    Class decorator enforcing default, membership, and field-order invariants.

  Raises:
    ValueError: If defaults, `dropped` placement, or field order are invalid.
  """
  require_none_default(unresolved_cls)
  unresolved_names = tuple(f.name for f in dataclasses.fields(unresolved_cls))
  dropped_names = tuple(dropped)

  unknown = [name for name in dropped_names if name not in unresolved_names]
  if unknown:
    raise ValueError(
        f"Unknown dropped fields {unknown} not on {unresolved_cls.__name__}."
    )
  if (
      dropped_names
      and unresolved_names[-len(dropped_names) :] != dropped_names
  ):
    raise ValueError(
        f"Dropped fields {dropped_names} must appear at the end of"
        f" {unresolved_cls.__name__}, got {unresolved_names}."
    )
  expected_names = unresolved_names[
      : len(unresolved_names) - len(dropped_names)
  ]

  def decorator(resolved_cls: T) -> T:
    require_no_default(resolved_cls)
    resolved_names = tuple(f.name for f in dataclasses.fields(resolved_cls))
    if resolved_names != expected_names:
      raise ValueError(
          f"{resolved_cls.__name__} fields {resolved_names} must match"
          f" {unresolved_cls.__name__} fields {expected_names} in exact order."
      )
    return resolved_cls

  return decorator
