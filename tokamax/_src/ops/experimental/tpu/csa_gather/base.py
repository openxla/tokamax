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
"""Base class for the DeepSeek-V4 CSA cache gather.

See `reference` for the cache and output layouts.
"""

from typing import Any, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.csa_gather import reference

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray


class CsaGather[C](op.Op[Any, tuple[jax.Array, jax.Array], None, C, Any]):
  """Tokamax operator for the DeepSeek-V4 CSA cache gather."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      nope_cache: Int[Array | AbstractArray, "num_pages page_size 128"],
      rope_cache: Int[Array | AbstractArray, "num_pages rope_rows 128"],
      indices: Int[Array | AbstractArray, "N"],
      num_valid_indices: Int[Array | AbstractArray, "*#nv"] | None = None,
      *,
      top_k: int = 1024,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    if nope_cache.dtype != jnp.int32 or rope_cache.dtype != jnp.int32:
      raise ValueError(
          "Caches must be int32, got"
          f" {nope_cache.dtype} and {rope_cache.dtype}."
      )
    if top_k <= 0 or top_k % 128:
      raise ValueError(
          f"top_k must be a positive multiple of 128, got {top_k}."
      )
    if indices.shape[0] % top_k:
      raise ValueError(
          f"Number of indices ({indices.shape[0]}) must be a multiple of top_k"
          f" ({top_k})."
      )
    if num_valid_indices is not None and num_valid_indices.size != 1:
      raise ValueError(
          "num_valid_indices must hold a single value, got shape"
          f" {num_valid_indices.shape}."
      )
    return super().bind(
        nope_cache=nope_cache,
        rope_cache=rope_cache,
        indices=indices,
        num_valid_indices=num_valid_indices,
        top_k=top_k,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      nope_cache: Int[Array, "num_pages page_size 128"],
      rope_cache: Int[Array, "num_pages rope_rows 128"],
      indices: Int[Array, "N"],
      num_valid_indices: Int[Array, "*#nv"] | None = None,
      *,
      top_k: int = 1024,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    return (
        reference.csa_gather(
            nope_cache, rope_cache, indices, num_valid_indices, top_k=top_k
        ),
        None,
    )
