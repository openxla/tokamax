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
"""Base class for Ragged Gather."""

from typing import Any, TypeVar, override
import jax
from jaxtyping import Array, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op


_Config = TypeVar("_Config")


def ragged_gather(
    x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array
) -> jax.Array:
  del start, end  # Integer scalar arrays can be converted to a scalar index.
  return x[indices]


class RaggedGather(op.Op[Any, jax.Array, None, _Config, Any]):
  """Tokamax operator for Ragged Gather."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      x: Shaped[Array, "in_size hidden_size"],
      indices: Int[Array, "out_size"],
      start: Int[Array, "1"],
      end: Int[Array, "1"],
      *,
      return_residuals: bool = False,
  ) -> op.BoundArguments:

    return super().bind(
        x=x,
        indices=indices,
        start=start,
        end=end,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Shaped[Array, "in_size hidden_size"],
      indices: Int[Array, "out_size"],
      start: Int[Array, "1"],
      end: Int[Array, "1"],
      *,
      return_residuals: bool = False,
      config: _Config | None = None,
  ) -> tuple[jax.Array, None]:
    return (
        ragged_gather(x, indices, start, end),
        None,
    )
