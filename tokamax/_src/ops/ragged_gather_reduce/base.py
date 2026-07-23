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
"""Base class for Ragged Gather Reduce."""

from typing import Any, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op


def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
  """Pure JAX reference implementation for ragged gather reduce."""
  out = x[indices] * topk_weights[:, None].astype(jnp.float32)
  out = jnp.where(valid_rows_mask[:, None], out, 0)
  out = out.reshape(-1, reduce_group_size, out.shape[-1])
  return jnp.sum(out, axis=1).astype(x.dtype)


class RaggedGatherReduce[C](op.Op[Any, jax.Array, None, C, Any]):
  """Tokamax operator for Ragged Gather Reduce."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      x: Shaped[Array, "input_size hidden_size"],
      indices: Int[Array, "input_size"],
      topk_weights: Shaped[Array, "input_size"],
      valid_rows_mask: Shaped[Array, "input_size"],
      *,
      reduce_group_size: int,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
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
      x: Shaped[Array, "input_size hidden_size"],
      indices: Int[Array, "input_size"],
      topk_weights: Shaped[Array, "input_size"],
      valid_rows_mask: Shaped[Array, "input_size"],
      *,
      reduce_group_size: int,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[jax.Array, None]:
    return (
        ragged_gather_reduce(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        ),
        None,
    )
