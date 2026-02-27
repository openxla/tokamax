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
import jaxtyping
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.matmul import base


Implementation: TypeAlias = Literal["cute_dsl", "xla"]

IMPLEMENTATIONS = dict(xla=base.Matmul())
_DEFAULT_IMPLEMENTATION = ("xla",)

try:
  from tokamax._src.ops.matmul import cute_dsl  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["cute_dsl"] = cute_dsl.CuteDslMatmul()
  _DEFAULT_IMPLEMENTATION = ("cute_dsl",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass


IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(IMPLEMENTATIONS)
)


@jaxtyping.jaxtyped
def matmul(
    a: Float[Array, "M K"],
    b: Float[Array, "N K"],
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    *,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ) = None,
) -> Float[Array, "M N"]:  # pylint: disable=g-doc-args
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATION

  if not isinstance(implementation, (tuple, list)):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  errors = []
  for impl in implementation:
    if isinstance(impl, str):
      if impl not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown implementation: {impl}")

      impl = IMPLEMENTATIONS[impl]

    try:
      return impl(
          a,
          b,
          precision=precision,
          preferred_element_type=preferred_element_type,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
