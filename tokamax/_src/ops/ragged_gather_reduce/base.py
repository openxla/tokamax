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
"""Base class for Ragged Gather Reduce.

See `reference` for the semantics of the op.
"""

from typing import Any, override

import jax
from jaxtyping import Array, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.ragged_gather_reduce import reference

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray


class RaggedGatherReduce[C](op.Op[Any, jax.Array, None, C, Any]):
  """Tokamax operator for Ragged Gather Reduce."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      x: Shaped[Array | AbstractArray, "num_rows hidden_size"],
      indices: Int[Array | AbstractArray, "input_size"],
      topk_weights: Shaped[Array | AbstractArray, "input_size"],
      valid_rows_mask: Shaped[Array | AbstractArray, "input_size"],
      *,
      reduce_group_size: int,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    if reduce_group_size <= 0:
      raise ValueError(
          f"reduce_group_size must be positive, got {reduce_group_size}."
      )
    if indices.shape[0] % reduce_group_size:
      raise ValueError(
          f"Number of routes ({indices.shape[0]}) must be a multiple of"
          f" reduce_group_size ({reduce_group_size})."
      )
    return super().bind(
        x=x,
        indices=indices,
        topk_weights=topk_weights,
        valid_rows_mask=valid_rows_mask,
        reduce_group_size=reduce_group_size,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Shaped[Array, "num_rows hidden_size"],
      indices: Int[Array, "input_size"],
      topk_weights: Shaped[Array, "input_size"],
      valid_rows_mask: Shaped[Array, "input_size"],
      *,
      reduce_group_size: int,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[jax.Array, None]:
    return (
        reference.ragged_gather_reduce(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        ),
        None,
    )
