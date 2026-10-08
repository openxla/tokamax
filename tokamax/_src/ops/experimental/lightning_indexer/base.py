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
"""Base class for Lightning Indexer operator."""

import dataclasses
from typing import Any, ClassVar, override

import jax
from jaxtyping import Array, Float, Int, UInt8  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.lightning_indexer import reference
from tokamax._src.ops.experimental.lightning_indexer.kernel import config as kernel_config

KVLayout = kernel_config.KVLayout
AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray


@dataclasses.dataclass(frozen=True)
class LightningIndexer[C](
    op.Op[Any, jax.Array | tuple[jax.Array, jax.Array], None, C, Any]
):
  """Tokamax base operator for Lightning Indexer retrieval."""

  supports_batched_args_capture: ClassVar[bool] = False

  @override
  @jaxtyping.jaxtyped
  def bind(
      self,
      q: Float[Array | AbstractArray, "T H D"],
      indexer_weights: Float[Array | AbstractArray, "T H"],
      cache_kv: UInt8[Array | AbstractArray, "P _ 4 _"],
      seq_lens: Int[Array | AbstractArray, "B"],
      page_indices: Int[Array | AbstractArray, "_"],
      cu_q_lens: Int[Array | AbstractArray, "_"],
      distribution: Int[Array | AbstractArray, "3"],
      *,
      k: int,
      compression_ratio: int = 1,
      kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
      cp_size: int = 1,
      cp_rank: Int[Array | AbstractArray, ""] | int = 0,
      interleave_size: int = 1,
      return_scores: bool = False,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Validates input shapes and binds arguments for Lightning Indexer."""
    if k <= 0:
      raise ValueError(f"k must be positive, got {k}.")
    if (
        compression_ratio < 1
        or (compression_ratio & (compression_ratio - 1)) != 0
    ):
      raise ValueError("compression_ratio must be a power of 2.")
    if cp_size < 1:
      raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
    if cp_size > 1 and interleave_size % compression_ratio != 0:
      raise ValueError(
          f"interleave_size ({interleave_size}) must be a multiple of "
          f"compression_ratio ({compression_ratio})."
      )
    if cu_q_lens.shape[0] != seq_lens.shape[0] + 1:
      raise ValueError(
          f"cu_q_lens length ({cu_q_lens.shape[0]}) must be"
          f" seq_lens length ({seq_lens.shape[0]}) + 1."
      )
    if page_indices.shape[0] % seq_lens.shape[0] != 0:
      raise ValueError(
          f"page_indices length ({page_indices.shape[0]}) must be divisible by"
          f" max_num_seqs ({seq_lens.shape[0]})."
      )
    kv_layout = KVLayout.parse(kv_layout)

    return super().bind(
        q=q,
        indexer_weights=indexer_weights,
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        k=k,
        compression_ratio=compression_ratio,
        kv_layout=kv_layout,
        cp_size=cp_size,
        cp_rank=cp_rank,
        interleave_size=interleave_size,
        return_scores=return_scores,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array, "T H D"],
      indexer_weights: Float[Array, "T H"],
      cache_kv: UInt8[Array, "P _ 4 _"],
      seq_lens: Int[Array, "B"],
      page_indices: Int[Array, "_"],
      cu_q_lens: Int[Array, "_"],
      distribution: Int[Array, "3"],
      *,
      k: int,
      compression_ratio: int = 1,
      kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
      cp_size: int = 1,
      cp_rank: Int[Array, ""] | int = 0,
      interleave_size: int = 1,
      return_scores: bool = False,
      return_residuals: bool = False,
      config: C,
  ) -> tuple[jax.Array | tuple[jax.Array, jax.Array], None]:
    del return_residuals, config
    out = reference.lightning_indexer(
        q=q,
        indexer_weights=indexer_weights,
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        k=k,
        compression_ratio=compression_ratio,
        kv_layout=kv_layout,
        cp_size=cp_size,
        cp_rank=cp_rank,
        interleave_size=interleave_size,
        return_scores=return_scores,
    )
    return out, None
