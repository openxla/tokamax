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
import dataclasses
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import precision as precision_lib
from tokamax._src.ops import op
from typing_extensions import override


_Config = TypeVar("_Config")
_Key = TypeVar("_Key")
Residuals = type(None)
CanonicalPrecision = precision_lib.CanonicalPrecision

DIMENSION_NUMBERS = jax.lax.DotDimensionNumbers((((1,), (1,)), ((), ())))


@dataclasses.dataclass(frozen=True)
class Matmul(op.Op[Any, jax.Array, Residuals, _Config, _Key]):

  @override
  def bind(
      self,
      a: Float[Array, "M K"],
      b: Float[Array, "N K"],
      *,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    # check that reduction dimensions sizes match
    (((lhs_contracting_dim,), (rhs_contracting_dim,)), _) = DIMENSION_NUMBERS
    if a.shape[lhs_contracting_dim] != b.shape[rhs_contracting_dim]:
      raise ValueError(
          f"The reduction dimension {lhs_contracting_dim} of a={jax.typeof()}"
          f" equal to {a.shape[lhs_contracting_dim]} does not match the"
          f" reduction dimension {rhs_contracting_dim} of b={jax.typeof(b)}"
          f" equal to {b.shape[rhs_contracting_dim]} for assumed"
          f" dimension_numbers={DIMENSION_NUMBERS}."
      )
    if preferred_element_type is not None:
      preferred_element_type = jnp.dtype(preferred_element_type)
    return super().bind(
        a,
        b,
        precision=precision_lib.canonicalize_precision(precision),
        preferred_element_type=preferred_element_type,
        return_residuals=return_residuals,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      a: Float[Array, "M K"],
      b: Float[Array, "N K"],
      *,
      precision: CanonicalPrecision,
      preferred_element_type: jnp.dtype | None,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[jax.Array, None]:
    del config  # Unused.

    return jax.lax.dot_general(
        a,
        b,
        dimension_numbers=DIMENSION_NUMBERS,
        precision=precision,
        preferred_element_type=preferred_element_type,
    ), None
