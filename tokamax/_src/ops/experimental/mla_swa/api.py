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
"""Multi-Head Latent Sliding Window Attention (MLA-SWA) API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
from jaxtyping import Array, Float, Int, UInt8  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.mla_swa import base

type Implementation = Literal["xla", "mosaic"]

_IMPLEMENTATIONS = dict(xla=base.MlaSwa())
_DEFAULT_IMPLEMENTATIONS = ("xla",)

try:
  from tokamax._src.ops.experimental.mla_swa import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic"] = pallas_mosaic_tpu.PallasTpuMlaSwa()
  _DEFAULT_IMPLEMENTATIONS = ("mosaic",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def mla_swa(
    q: Float[Array, "T H D"],
    new_kv: Float[Array, "T D"],
    cache_kv: UInt8[Array, "P S 4 128"] | Int[Array, "P S 128"],
    kv_lens: Int[Array, "M"],
    page_indices: Int[Array, "N"],
    cu_q_lens: Int[Array, "B"],
    distribution: Int[Array, "3"],
    attention_sinks: Float[Array, "H"],
    *,
    sliding_window: int,
    logical_page_size: int,
    sm_scale: float = 1.0,
    chunk_prefill_size: int | None = None,
    unnormalized_output: bool = False,
    non_causal_block: bool = False,
    implementation: (
        Implementation | Sequence[Implementation | Callable[..., Any]] | None
    ) = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Computes Multi-Head Latent Sliding Window Attention.

  Args:
    q: Query tensor
    new_kv: New KV tokens.
    cache_kv: Paged SWA KV cache.
    kv_lens: Per-sequence KV lengths.
    page_indices: Flattened page table.
    cu_q_lens: Cumulative query token lengths.
    distribution: Batch distribution.
    attention_sinks: Attention sink logits.
    sliding_window: Sliding window size in tokens.
    logical_page_size: Logical page capacity in tokens.
    sm_scale: Softmax scale.
    chunk_prefill_size: Optional static query length for the prefill-only
      segment.
    unnormalized_output: Whether to return the unnormalized numerator
      accumulator instead of dividing by the softmax denominator and sinks.
    non_causal_block: If true, disables causal masking.
    implementation: The implementation to use. By default, None is used, which
      selects the best available backend. If a sequence is passed, the first
      implementation that doesn't raise a NotImplementedError is used.

  Returns:
    Tuple (output, updated_cache_kv, out_l, out_m).

  Raises:
    ExceptionGroup: If multiple implementations are provided and all of them
      raise NotImplementedError.
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
          new_kv,
          cache_kv,
          kv_lens,
          page_indices,
          cu_q_lens,
          distribution,
          attention_sinks,
          sliding_window=sliding_window,
          logical_page_size=logical_page_size,
          sm_scale=sm_scale,
          chunk_prefill_size=chunk_prefill_size,
          unnormalized_output=unnormalized_output,
          non_causal_block=non_causal_block,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("All implementations failed", errors)
