# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
from typing import Any, Final, Literal, TypeAlias

import immutabledict
import jax
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.dot import base


Implementation: TypeAlias = Literal["cutedsl", "xla"]

IMPLEMENTATIONS = dict(xla=base.Dot())
_DEFAULT_IMPLEMENTATION = ("xla",)

try:
  from tokamax._src.ops.dot import cutedsl  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["cutedsl"] = cutedsl.CutedslDot()
  _DEFAULT_IMPLEMENTATION = ("cutedsl",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass


IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(IMPLEMENTATIONS)
)


def dot(
    lhs: Float[Array, "M K"],
    rhs: Float[Array, "N K"],
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    *,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ) = None,
) -> Float[Array, "M N"]:  # pylint: disable=g-doc-args
  return dot_general(
      lhs,
      rhs,
      dimension_numbers=base.DEFAULT_DOT_DIM_NUMS,
      precision=precision,
      preferred_element_type=preferred_element_type,
      implementation=implementation,
  )


def dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    *,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ) = None,
) -> Float[Array, "..."]:  # pylint: disable=g-doc-args
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATION

  if not isinstance(implementation, (tuple, list)):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  # check that reduction dimensions sizes match
  reduction_dims = dimension_numbers[0]
  for dim1, dim2 in zip(*reduction_dims, strict=True):
    if lhs.shape[dim1] != rhs.shape[dim2]:
      raise ValueError(
          f"The reduction dimension {dim1} of lhs={jax.typeof(lhs)} equal to"
          f" {lhs.shape[dim1]} does not match the reduction dimension {dim2} of"
          f" rhs={jax.typeof(rhs)} equal to {rhs.shape[dim2]} for"
          f" dimension_numbers={dimension_numbers}."
      )

  errors = []
  for impl in implementation:
    if isinstance(impl, str):
      if impl not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown implementation: {impl}")

      impl = IMPLEMENTATIONS[impl]

    try:
      return impl(
          lhs,
          rhs,
          dimension_numbers=dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
