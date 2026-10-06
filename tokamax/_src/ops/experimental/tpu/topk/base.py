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
"""Base class for TopK operator."""

from typing import Any, override

import jax
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.topk import reference

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray


class TopK[C](
    op.Op[Any, jax.Array | tuple[jax.Array, jax.Array], None, C, Any]
):
  """Tokamax operator for TopK."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      scores: (
          Float[Array | AbstractArray, "b n"]
          | Int[Array | AbstractArray, "b n"]
      ),
      k: int,
      row_lengths: Int[Array | AbstractArray, "b"] | None = None,
      *,
      return_scores: bool = False,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    if k <= 0:
      raise ValueError(f"k must be positive, got {k}.")
    if scores.shape[-1] < k:
      raise ValueError(
          f"Last dimension of scores ({scores.shape[-1]}) must be >= k ({k})."
      )
    if row_lengths is not None and row_lengths.shape != (scores.shape[0],):
      raise ValueError(
          f"row_lengths shape {row_lengths.shape} must match batch dimension"
          f" ({scores.shape[0]},)."
      )
    return super().bind(
        scores=scores,
        k=k,
        row_lengths=row_lengths,
        return_scores=return_scores,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      scores: Float[Array, "b n"] | Int[Array, "b n"],
      k: int,
      row_lengths: Int[Array, "b"] | None = None,
      *,
      return_scores: bool = False,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[jax.Array | tuple[jax.Array, jax.Array], None]:
    del config, return_residuals
    return (
        reference.topk(
            scores,
            k,
            row_lengths,
            return_scores=return_scores,
        ),
        None,
    )
