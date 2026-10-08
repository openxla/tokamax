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
"""Base class for the sort-free MoE router top-k.

See `reference` for the algorithm.
"""

from typing import Any, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.router_topk import reference

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray | np.ndarray

# Score dtypes the op accepts. Each is cast to float32 (exactly) before the
# selection, as upstream's torch bridge does with `scores.float()`.
SCORE_DTYPES: tuple[jnp.dtype, ...] = (
    jnp.dtype(jnp.float32),
    jnp.dtype(jnp.bfloat16),
    jnp.dtype(jnp.float16),
)

type RouterTopKOutput = tuple[Float[Array, "T K"], Int[Array, "T K"]]


class RouterTopK[C](op.Op[Any, RouterTopKOutput, None, C, Any]):
  """Tokamax operator for the sort-free MoE router top-k.

  Selects the `k` largest scores of each row of `[num_tokens, num_experts]`
  router scores with `k` passes of (row max -> lowest matching column -> mask
  that column out), `O(k * num_experts)` per row instead of a full sort.

  Semantics differ from `jax.lax.top_k` on two kinds of row:

  *   Ties resolve to the lowest expert id.
  *   Scores at or below `reference.NEG` (NaN, `-inf`, `-FLT_MAX`) are never
      selected over a finite score. A row made entirely of them gets NaN
      weights and the experts `0..k-1`.

  Scores are cast to float32 before the selection, so the weights are always
  float32 and the indices int32.
  """

  @jaxtyping.jaxtyped
  def bind(
      self,
      scores: Shaped[Array | AbstractArray, "..."],
      k: int,
      *,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Validates the arguments.

    Args:
      scores: `(num_tokens, num_experts)` float32, bfloat16 or float16 router
        scores.
      k: The number of experts to select per token, in `[1, num_experts]`.
      return_residuals: Unused; the op has no residuals.

    Returns:
      The bound arguments.

    Raises:
      ValueError: If `scores` is not rank 2 or not of a `SCORE_DTYPES` dtype,
        or `k` is out of range.
    """
    if scores.ndim != 2:
      raise ValueError(
          "scores must be rank 2 (num_tokens, num_experts), got shape"
          f" {scores.shape}."
      )
    if jnp.dtype(scores.dtype) not in SCORE_DTYPES:
      raise ValueError(
          "scores must be one of"
          f" {tuple(d.name for d in SCORE_DTYPES)}, got {scores.dtype}."
      )
    num_experts = scores.shape[1]
    if not 1 <= k <= num_experts:
      raise ValueError(f"k must be in [1, num_experts={num_experts}], got {k}.")
    return super().bind(scores=scores, k=k, return_residuals=return_residuals)

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      scores: Float[Array, "T E"],
      k: int,
      *,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[RouterTopKOutput, None]:
    """Selects the top `k` experts of each token.

    Runs `reference.rowmax_topk` on the float32 scores.

    Args:
      scores: `(num_tokens, num_experts)` router scores.
      k: The number of experts to select per token.
      return_residuals: Unused; the op has no residuals.
      config: Unused.

    Returns:
      `((weights, indices), None)`: float32 `weights`, descending, and int32
      expert `indices`, both `(num_tokens, k)`.
    """
    del return_residuals, config  # Unused.
    return reference.rowmax_topk(scores.astype(jnp.float32), k), None
