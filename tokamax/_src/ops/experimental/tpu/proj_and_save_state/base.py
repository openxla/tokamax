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
"""Base class for the DeepSeek-V4 compressor projection.

See `reference` for the state and cache layouts.
"""

from typing import Any, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.proj_and_save_state import reference

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray | np.ndarray

_SLAB_ROWS = 4  # uint8 sub-rows per 32-bit row.
_F32_BYTES = 4


class ProjAndSaveState[C](op.Op[Any, jax.Array, None, C, Any]):
  """Tokamax operator for the DeepSeek-V4 compressor projection."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      hidden_states: Float[Array | AbstractArray, "T H"],
      wkv_wgate: Float[Array | AbstractArray, "H D"],
      ape: Float[Array | AbstractArray, "R W"],
      positions: Int[Array | AbstractArray, "T"],
      slot_mapping: Int[Array | AbstractArray, "T"],
      cache: Shaped[Array | AbstractArray, "num_pages page_size *slab"],
      *,
      compress_ratio: int,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    state_dim = wkv_wgate.shape[1]
    if state_dim % 2:
      raise ValueError(f"wkv_wgate must have an even width, got {state_dim}.")
    state_width = state_dim // 2
    if ape.shape != (compress_ratio, state_width):
      raise ValueError(
          "ape must have shape (compress_ratio, state_width) ="
          f" ({compress_ratio}, {state_width}), got {ape.shape}."
      )
    if cache.dtype == jnp.uint8 and cache.ndim == 4:
      if cache.shape[2] != _SLAB_ROWS:
        raise ValueError(
            f"A uint8 cache must be (num_pages, page_size, {_SLAB_ROWS},"
            f" lanes), got {cache.shape}."
        )
    elif cache.dtype != jnp.int32 or cache.ndim != 3:
      raise ValueError(
          "cache must be (num_pages, page_size, 4, lanes) uint8 or (num_pages,"
          f" page_size, lanes) int32, got {cache.shape} {cache.dtype}."
      )
    lanes = cache.shape[-1]
    if lanes % 128 or state_width % lanes:
      raise ValueError(
          f"The cache lane count ({lanes}) must be a multiple of 128 that"
          f" divides state_width ({state_width})."
      )
    rows_per_token = state_dim * _F32_BYTES // (_SLAB_ROWS * lanes)
    if cache.shape[1] % rows_per_token:
      raise ValueError(
          f"page_size ({cache.shape[1]}) must be a multiple of the"
          f" {rows_per_token} cache rows a token's state occupies."
      )
    return super().bind(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        positions=positions,
        slot_mapping=slot_mapping,
        cache=cache,
        compress_ratio=compress_ratio,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      hidden_states: Float[Array, "T H"],
      wkv_wgate: Float[Array, "H D"],
      ape: Float[Array, "R W"],
      positions: Int[Array, "T"],
      slot_mapping: Int[Array, "T"],
      cache: Shaped[Array, "num_pages page_size *slab"],
      *,
      compress_ratio: int,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[jax.Array, None]:
    """Projects the tokens and writes their state into the cache.

    See `reference` for the state and cache layouts.

    Args:
      hidden_states: `(num_tokens, hidden_size)` hidden states.
      wkv_wgate: `(hidden_size, 2 * state_width)` fused `kv` and `score`
        projection weights.
      ape: `(compress_ratio, state_width)` absolute position embeddings, added
        to `score`.
      positions: `(num_tokens,)` int32 token positions; token `t` adds APE row
        `positions[t] % compress_ratio`.
      slot_mapping: `(num_tokens,)` int32 first cache row (`page * page_size +
        row`) of each token's state, a multiple of the rows per token. Negative
        entries skip the token.
      cache: `(num_pages, page_size, 4, lanes)` uint8 or `(num_pages, page_size,
        lanes)` int32 state cache.
      compress_ratio: Number of APE rows.
      return_residuals: Unused; the op has no residuals.
      config: Unused; the reference has no config.

    Returns:
      `(new_cache, None)`, where `new_cache` is `cache` with the state of every
      token with a non-negative slot written.
    """
    return (
        reference.proj_and_save_state(
            hidden_states,
            wkv_wgate,
            ape,
            positions,
            slot_mapping,
            cache,
            compress_ratio=compress_ratio,
        ),
        None,
    )
