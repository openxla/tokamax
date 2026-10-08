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
"""Lightning Indexer Op API."""

from collections.abc import Sequence
from typing import Final, Literal

import immutabledict
import jax
from jaxtyping import Array, Float, Int, UInt8  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.lightning_indexer import base
from tokamax._src.ops.experimental.lightning_indexer.kernel import config

type Implementation = Literal["mosaic", "xla"]

_IMPLEMENTATIONS: dict[str, base.LightningIndexer] = dict(
    xla=base.LightningIndexer()
)
_DEFAULT_IMPLEMENTATION: tuple[str, ...] = ("xla",)

try:
  from tokamax._src.ops.experimental.lightning_indexer import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic"] = pallas_mosaic_tpu.PallasTpuLightningIndexer()
  _DEFAULT_IMPLEMENTATION = ("mosaic",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass

IMPLEMENTATIONS: Final[
    immutabledict.immutabledict[str, base.LightningIndexer]
] = immutabledict.immutabledict(_IMPLEMENTATIONS)
del _IMPLEMENTATIONS


def lightning_indexer(
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
    kv_layout: config.KVLayout | str = config.KVLayout.HEAD_ALONG_SUBLANE,
    cp_size: int = 1,
    cp_rank: Int[Array, ""] | int = 0,
    interleave_size: int = 1,
    return_scores: bool = False,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Computes Lightning Indexer retrieval using the selected implementation.

  Args:
    q: Query tensor of shape [max_num_tokens, num_q_heads, head_dim].
    indexer_weights: Indexer weights of shape [max_num_tokens, num_q_heads].
    cache_kv: Packed uint8 KV cache in HEAD_ALONG_SUBLANE or SEQ_ALONG_LANE
      layout.
    seq_lens: Uncompressed KV sequence lengths.
    page_indices: Flattened page look-up table
    cu_q_lens: Cumulative query token counts.
    distribution: Batch split counts.
    k: Number of top-K compressed KV positions to retrieve per query token.
    compression_ratio: KV cache compression ratio (must be a power of 2).
    kv_layout: Memory layout of cache_kv (either HEAD_ALONG_SUBLANE or
      SEQ_ALONG_LANE).
    cp_size: Context-parallel world size (1 means unsharded).
    cp_rank: Context-parallel rank of the current shard.
    interleave_size: Context-parallel chunk-interleave width in uncompressed
      tokens.
    return_scores: If True, also returns the selected scores as int32 holding
      raw float32` its (-inf bits in padded slots).
    implementation: Implementation(s) to try ("mosaic" or "xla"). If None, tries
      the default implementation order.

  Returns:
    Top-K compressed KV indices of shape [max_num_tokens, k] with int32
    dtype (-1 suffix-padded), or (indices, scores_bits) if
    return_scores=True.
  """
  kwargs = dict(
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
  if implementation is None:
    implementations = _DEFAULT_IMPLEMENTATION
  elif isinstance(implementation, str) or callable(implementation):
    implementations = (implementation,)
  elif not implementation:
    raise ValueError("The implementation must not be an empty sequence.")
  else:
    implementations = tuple(implementation)

  errors = []
  for impl in implementations:
    if isinstance(impl, str):
      if impl not in IMPLEMENTATIONS:
        continue
      impl_fn = IMPLEMENTATIONS[impl]
    else:
      impl_fn = impl

    try:
      return impl_fn(**kwargs)
    except NotImplementedError as e:
      if len(implementations) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("All implementations failed", errors)
