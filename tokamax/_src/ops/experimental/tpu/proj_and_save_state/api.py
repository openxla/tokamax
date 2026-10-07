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
"""DeepSeek-V4 compressor projection (proj_and_save_state) API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
from tokamax._src.ops.experimental.tpu.proj_and_save_state import base

type Implementation = Literal["xla", "mosaic_tpu"]

_IMPLEMENTATIONS = dict(xla=base.ProjAndSaveState())
_DEFAULT_IMPLEMENTATIONS = ("xla",)

try:
  from tokamax._src.ops.experimental.tpu.proj_and_save_state import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuProjAndSaveState()
  _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def proj_and_save_state(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    ape: jax.Array,
    positions: jax.Array,
    slot_mapping: jax.Array,
    cache: jax.Array,
    *,
    compress_ratio: int,
    implementation: (
        Implementation | Sequence[Implementation | Callable[..., Any]] | None
    ) = None,
) -> jax.Array:
  """Projects DeepSeek-V4 compressor states and writes them into the cache.

  Computes each token's f32 state `[kv, score + ape[positions %
  compress_ratio]]`
  from `hidden_states @ wkv_wgate` and writes it to the cache rows starting at
  its `slot_mapping` entry. See `reference` for the state and cache layouts.

  Args:
    hidden_states: `(num_tokens, hidden_size)` hidden states.
    wkv_wgate: `(hidden_size, 2 * state_width)` fused `kv` and `score`
      projection weights.
    ape: `(compress_ratio, state_width)` absolute position embeddings.
    positions: `(num_tokens,)` int32 token positions.
    slot_mapping: `(num_tokens,)` int32 first cache row of each token's state, a
      multiple of the rows per token. Negative entries skip the token.
    cache: `(num_pages, page_size, 4, lanes)` uint8 or `(num_pages, page_size,
      lanes)` int32 state cache.
    compress_ratio: Number of APE rows.
    implementation: The implementation to use. By default, `None` is used, which
      selects the best available backend. If a sequence is passed, the first
      implementation that doesn't raise a `NotImplementedError` is used.

  Returns:
    The updated cache, with the shape and dtype of `cache`.

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
          hidden_states,
          wkv_wgate,
          ape,
          positions,
          slot_mapping,
          cache,
          compress_ratio=compress_ratio,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
