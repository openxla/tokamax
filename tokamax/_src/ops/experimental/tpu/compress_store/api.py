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
"""Compress-and-store API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
from tokamax._src.ops.experimental.tpu.compress_store import base

type Implementation = Literal["xla", "mosaic_tpu"]

_IMPLEMENTATIONS = dict(xla=base.CompressStore())
_DEFAULT_IMPLEMENTATIONS = ("xla",)

try:
  from tokamax._src.ops.experimental.tpu.compress_store import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuCompressStore()
  _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def compress_store(
    cache: jax.Array,
    positions: jax.Array,
    block_table: jax.Array,
    token_to_req_indices: jax.Array,
    kv_slot_mapping: jax.Array,
    rms_weight: jax.Array,
    *,
    cos_sin_cache: jax.Array,
    block_table_stride: int,
    state_block_size: int,
    compress_ratio: int,
    overlap: bool,
    state_cache: jax.Array | None = None,
    rope_cache: jax.Array | None = None,
    quant_block: int = base.CSA_QUANT_BLOCK,
    rms_eps: float = 1e-6,
    implementation: (
        Implementation | Sequence[Implementation | Callable[..., Any]] | None
    ) = None,
) -> tuple[jax.Array, jax.Array | None]:
  """Compresses DeepSeek-V4 boundary tokens' windows into the KV cache.

  For each token with a compressed-KV slot, softmax-pools the f32 states of its
  window from the state cache, applies RMSNorm and interleaved RoPE, packs the
  record and stores it into `cache` (and, for CSA, `rope_cache`): the caches
  that CSA Gather and the sparse attention read. See `reference` for the modes
  and cache layouts.

  Args:
    cache: The compressed KV cache: uint8 `(num_pages, rows, 4, 128)` (HCA),
      int32 `(num_pages, rows, 128)` (CSA) or uint8 `(num_pages, rows, 4, 256)`
      (CSA indexer). For CSA and the indexer it also hosts the f32 state.
    positions: `(N,)` int32 token positions.
    block_table: `(num_reqs * block_table_stride,)` int32 state pages of each
      request.
    token_to_req_indices: `(N,)` int32 request of each token.
    kv_slot_mapping: `(N,)` int32 compressed-KV slot of each token, or -1 to
      skip it. Only boundary tokens (`(position + 1) % compress_ratio == 0`) may
      have a slot.
    rms_weight: `(head_dim,)` f32 RMSNorm weight.
    cos_sin_cache: `(max_pos, 64)` f32 RoPE `[cos | sin]` table.
    block_table_stride: Row stride of `block_table`.
    state_block_size: Token states per state page.
    compress_ratio: Tokens compressed into one record.
    overlap: Whether windows overlap; selects CSA (or the indexer when `head_dim
      == 128`) over HCA.
    state_cache: The separate uint8 f32-state array (HCA), or `None` when
      `cache` hosts the state.
    rope_cache: `(num_pages, rows // 4, 128)` int32 RoPE cache, required for CSA
      and ignored otherwise.
    quant_block: FP8 quantization block: the scale lane period (64) for CSA, the
      scale block for the indexer. Unused for HCA.
    rms_eps: RMSNorm epsilon.
    implementation: The implementation to use. By default, `None` is used, which
      selects the best available backend. If a sequence is passed, the first
      implementation that doesn't raise a `NotImplementedError` is used.

  Returns:
    The updated `(cache, rope_cache)`; `rope_cache` is `None` outside CSA. As
    upstream, the Pallas implementation donates `cache` and `rope_cache`.

  Raises:
    ExceptionGroup: If multiple implementations are provided and all of them
      raise `NotImplementedError`.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  errors = []
  for impl in implementation:
    if isinstance(impl, str):
      if impl not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation: {impl}. You may need to add a dependency"
            " on the corresponding backend."
        )
      impl = IMPLEMENTATIONS[impl]

    try:
      return impl(
          cache,
          positions,
          block_table,
          token_to_req_indices,
          kv_slot_mapping,
          rms_weight,
          cos_sin_cache=cos_sin_cache,
          block_table_stride=block_table_stride,
          state_block_size=state_block_size,
          compress_ratio=compress_ratio,
          overlap=overlap,
          state_cache=state_cache,
          rope_cache=rope_cache,
          quant_block=quant_block,
          rms_eps=rms_eps,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
