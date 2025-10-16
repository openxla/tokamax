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
"""Tokamax utility functions."""

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import jax


_T = TypeVar('_T')


def exact_div(a: int | tuple[int, str], b: int | tuple[int, str]) -> int:
  """Returns `a // b`, raising a `ValueError` if there is a remainder."""
  a, a_name = (a, '`a`') if isinstance(a, int) else a
  b, b_name = (b, '`b`') if isinstance(b, int) else b
  quotient, remainder = divmod(a, b)
  if remainder:
    raise ValueError(f'{a_name} ({a}) must divide exactly by {b_name} ({b})')
  return quotient


def is_array_like(x: Any) -> bool:
  """Returns whether `x` is an array-like value."""
  return hasattr(x, 'shape') and hasattr(x, 'dtype')


def flatten_arrays(
    x: _T, is_leaf: Callable[[Any], bool] | None = None
) -> tuple[list[Any], Callable[[Sequence[Any]], _T]]:
  """Flattens value to a list of "array-like" values and recompose function."""
  flat, tree = jax.tree.flatten(x, is_leaf=is_leaf)
  arrays, other, merge = split_merge(is_array_like, flat)
  recompose = lambda arrays: tree.unflatten(merge(arrays, other))
  return arrays, recompose


# Adapted from jax._src.util.split_merge in JAX v0.6.0.
def split_merge(
    predicate: Callable[[_T], bool], xs: Sequence[_T]
) -> tuple[
    list[_T], list[_T], Callable[[Sequence[_T], Sequence[_T]], list[_T]]
]:
  """Splits a sequence based on a predicate, and returns a merge function."""
  sides = list(map(predicate, xs))
  lhs = [x for x, s in zip(xs, sides) if s]
  rhs = [x for x, s in zip(xs, sides) if not s]
  def merge(new_lhs: Sequence[_T], new_rhs: Sequence[_T]) -> list[_T]:
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
