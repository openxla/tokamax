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
"""Baseline JAX implementation of Ragged Scatter."""

from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Shaped
from tokamax._src.ops import op
from tokamax._src import jaxtyping

_Config = TypeVar("_Config")


def ragged_scatter(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
) -> jax.Array:
  """Pure JAX reference implementation for ragged scatter."""
  out = x[indices]
  mask = (indices >= start) & (indices < end)
  return jnp.where(mask[:, None], out, 0)


class RaggedScatter(op.Op[Any, jax.Array, None, _Config, Any]):
  """Tokamax operator for Ragged Scatter."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      x: Shaped[Array, "num_rows hidden_size"],
      indices: Int[Array, "output_size"],
      start: Int[Array, "1"] | Int[Array, ""],
      end: Int[Array, "1"] | Int[Array, ""],
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

  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Shaped[Array, "num_rows hidden_size"],
      indices: Int[Array, "output_size"],
      start: Int[Array, "1"] | Int[Array, ""],
      end: Int[Array, "1"] | Int[Array, ""],
      *,
      return_residuals: bool = False,
      config: _Config | None = None,
  ) -> tuple[jax.Array, None]:
    del return_residuals, config
    return ragged_scatter(x, indices, start, end), None
