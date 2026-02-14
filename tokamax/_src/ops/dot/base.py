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
from tokamax._src import precision as precision_lib
from tokamax._src.ops import op
from typing_extensions import override


_Config = TypeVar("_Config")
_Key = TypeVar("_Key")
Residuals = jax.Array | None
CanonicalPrecision = precision_lib.CanonicalPrecision

DEFAULT_DOT_DIM_NUMS = jax.lax.DotDimensionNumbers((((1,), (1,)), ((), ())))


@dataclasses.dataclass(frozen=True)
class Dot(op.Op[Any, jax.Array, Residuals, _Config, _Key]):

  @override
  def bind(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      *,
      dimension_numbers: (
          jax.lax.DotDimensionNumbers | None
      ) = None,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    if dimension_numbers is None:
      dimension_numbers = DEFAULT_DOT_DIM_NUMS

    if preferred_element_type is not None:
      preferred_element_type = jnp.dtype(preferred_element_type)
    return super().bind(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision_lib.canonicalize_precision(precision),
        preferred_element_type=preferred_element_type,
    )

  @override
  def _fwd(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      *,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      precision: CanonicalPrecision,
      preferred_element_type: jnp.dtype | None,
      return_residuals: bool,
      config: _Config,
  ) -> jax.Array:
    del config  # Unused.

    return jax.lax.dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    ), None
