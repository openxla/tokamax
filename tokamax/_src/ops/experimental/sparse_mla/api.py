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
"""Sparse Multi-Head Latent Attention (SparseMLA) API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.sparse_mla import base

type Implementation = Literal["xla", "mosaic_tpu"]

_IMPLEMENTATIONS = dict(xla=base.SparseMla())
_DEFAULT_IMPLEMENTATIONS = ("xla",)

try:
  from tokamax._src.ops.experimental.sparse_mla import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuSparseMla()
  _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def sparse_mla(
    q: Float[Array, "T H D"],
    cache_kv_nope: Int[Array, "P S 128"],
    cache_kv_rope: Int[Array, "P R 128"],
    topk_indices: Int[Array, "T K"],
    page_indices: Int[Array, "N"],
    cu_q_lens: Int[Array, "B"],
    distribution: Int[Array, "3"],
    attention_sinks: Float[Array, "H"],
    swa_accumution: Float[Array, "T H D"],
    swa_l: Float[Array, "T H"],
    swa_m: Float[Array, "T H"],
    *,
    sm_scale: float = 1.0,
    implementation: (
        Implementation | Sequence[Implementation | Callable[..., Any]] | None
    ) = None,
) -> jax.Array:
  """Computes Sparse Multi-Head Latent Attention using the selected backend.

  Args:
    q: Query tensor
    cache_kv_nope: Quantized NoPE KV cache
    cache_kv_rope: Packed RoPE KV cache
    topk_indices: Selected top-k KV token indices per query token
    page_indices: Flattened page table
    cu_q_lens: Cumulative query token lengths
    distribution: Batch distribution
    attention_sinks: Attention sink logits
    swa_accumution: Sliding window attention numerator accumulator
    swa_l: Sliding window attention denominator sum
    swa_m: Sliding window attention row max
    sm_scale: Softmax scale applied to Q @ K^T
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
          topk_indices,
          page_indices,
          cu_q_lens,
          distribution,
          attention_sinks,
          swa_accumution,
          swa_l,
          swa_m,
          sm_scale=sm_scale,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("All implementations failed", errors)
