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

from collections.abc import Callable, Sequence
from typing import TypeVar


T = TypeVar('T')


def exact_div(a: int | tuple[int, str], b: int | tuple[int, str]) -> int:
  """Returns `a // b`, raising a `ValueError` if there is a remainder."""
  a, a_name = (a, '`a`') if isinstance(a, int) else a
  b, b_name = (b, '`b`') if isinstance(b, int) else b
  quotient, remainder = divmod(a, b)
  if remainder:
    raise ValueError(f'{a_name} ({a}) must divide exactly by {b_name} ({b})')
  return quotient


# Adapted from jax._src.util.split_merge in JAX v0.6.0.
def split_merge(
    predicate: Callable[[T], bool], xs: Sequence[T]
) -> tuple[list[T], list[T], Callable[[Sequence[T], Sequence[T]], list[T]]]:
  """Splits a sequence based on a predicate, and returns a merge function."""
  sides = list(map(predicate, xs))
  lhs = [x for x, s in zip(xs, sides) if s]
  rhs = [x for x, s in zip(xs, sides) if not s]

  def merge(new_lhs: Sequence[T], new_rhs: Sequence[T]) -> list[T]:
    out = []
    for s in sides:
      if s:
        out.append(new_lhs[0])
        new_lhs = new_lhs[1:]
      else:
        out.append(new_rhs[0])
        new_rhs = new_rhs[1:]
    assert not new_rhs
    assert not new_lhs
    return out

  return lhs, rhs, merge
