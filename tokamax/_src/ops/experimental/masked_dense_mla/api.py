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
"""Masked-Dense Multi-Head Latent Attention (MaskedDenseMLA) API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
from jaxtyping import Array, Float, Int, UInt8  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.masked_dense_mla import base

type Implementation = Literal["xla", "mosaic"]

_IMPLEMENTATIONS = dict(xla=base.MaskedDenseMla())
_DEFAULT_IMPLEMENTATIONS = ("xla",)

try:
  from tokamax._src.ops.experimental.masked_dense_mla import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic"] = pallas_mosaic_tpu.PallasTpuMaskedDenseMla()
  _DEFAULT_IMPLEMENTATIONS = ("mosaic",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def masked_dense_mla(
    q: Float[Array, "T H D"],
    cache_kv_nope: UInt8[Array, "P S 4 128"],
    cache_kv_rope: UInt8[Array, "P R 4 128"],
    kv_lens: Int[Array, "M"],
    topk_indices: Int[Array, "T K"],
    page_indices: Int[Array, "N"],
    cu_q_lens: Int[Array, "B"],
    distribution: Int[Array, "3"],
    *,
    sm_scale: float = 1.0,
    k_scale: float = 1.0,
    mask_value: float | None = None,
    max_kv_len: int | None = None,
    chunk_prefill_size: int | None = None,
    sequence_start: Int[Array, ""] | None = None,
    implementation: (
        Implementation | Sequence[Implementation | Callable[..., Any]] | None
    ) = None,
) -> jax.Array:
  """Computes Masked-Dense Multi-Head Latent Attention using the selected backend.

  Args:
    q: Query tensor
    cache_kv_nope: Quantized NoPE KV cache
    cache_kv_rope: Quantized RoPE KV cache
    kv_lens: Per-sequence KV lengths
    topk_indices: Selected top-k KV token indices per query token
    page_indices: Flattened page table
    cu_q_lens: Cumulative query token lengths
    distribution: Batch distribution
    sm_scale: Softmax scale applied to Q @ K^T
    k_scale: Per-tensor dequantization scale for the FP8 caches
    mask_value: Score written where the mask excludes a KV position
    max_kv_len: Optional upper bound on every sequence's kv_len
    chunk_prefill_size: Optional static query length for the prefill-only
      segment
    sequence_start: Optional first sequence index to attend
    implementation: The implementation to use. By default, None is used, which
      selects the best available backend. If a sequence is passed, the first
      implementation that doesn't raise a NotImplementedError is used.

  Returns:
    Output attention tensor

  Raises:
    ExceptionGroup: If multiple implementations are provided and all of them
      raise NotImplementedError
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str) or callable(implementation):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("implementation must not be an empty sequence.")

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
          q,
          cache_kv_nope,
          cache_kv_rope,
          kv_lens,
          topk_indices,
          page_indices,
          cu_q_lens,
          distribution,
          sm_scale=sm_scale,
          k_scale=k_scale,
          mask_value=mask_value,
          max_kv_len=max_kv_len,
          chunk_prefill_size=chunk_prefill_size,
          sequence_start=sequence_start,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("All implementations failed", errors)
