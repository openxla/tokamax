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

from typing import Any, override

import jax
from jaxtyping import Array, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray


def ragged_gather(
    x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array
) -> jax.Array:
  """Gathers `x[indices]` for output rows in `[start, end)`."""
  # Mask indices for output rows outside [start, end) to -1.
  row_ids = jax.lax.iota(jax.numpy.int32, indices.shape[0])
  start_idx = jax.lax.reshape(start, ())
  end_idx = jax.lax.reshape(end, ())
  in_range = jax.lax.bitwise_and(
      jax.lax.ge(row_ids, start_idx), jax.lax.lt(row_ids, end_idx)
  )
  masked_indices = jax.lax.select(
      in_range, indices, jax.lax.full_like(indices, -1)
  )
  # Gather slices of x along axis 0 using the masked 1D indices.
  dnums = jax.lax.GatherDimensionNumbers(
      offset_dims=tuple(range(1, x.ndim)),
      collapsed_slice_dims=(0,),
      start_index_map=(0,),
  )
  return jax.lax.gather(
      x,
      jax.lax.expand_dims(masked_indices, (-1,)),
      dimension_numbers=dnums,
      slice_sizes=(1, *x.shape[1:]),
  )


class RaggedGather[C](op.Op[Any, jax.Array, None, C, Any]):
  """Tokamax operator for Ragged Gather."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      x: Shaped[Array | AbstractArray, "in_size hidden_size"],
      indices: Int[Array | AbstractArray, "out_size"],
      start: Int[Array | AbstractArray, "1"],
      end: Int[Array | AbstractArray, "1"],
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
      config: C | None = None,
  ) -> tuple[jax.Array, None]:
    return (
        ragged_gather(x, indices, start, end),
        None,
    )
