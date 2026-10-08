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
"""Sort-free MoE router top-k API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
from tokamax._src.ops.experimental.tpu.router_topk import base

type Implementation = Literal["xla", "mosaic_tpu"]

_IMPLEMENTATIONS = dict(xla=base.RouterTopK())
_DEFAULT_IMPLEMENTATIONS = ("xla",)

try:
  from tokamax._src.ops.experimental.tpu.router_topk import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuRouterTopK()
  _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def router_topk(
    scores: jax.Array,
    k: int,
    *,
    implementation: (
        Implementation | Sequence[Implementation | Callable[..., Any]] | None
    ) = None,
) -> base.RouterTopKOutput:
  """Selects the top `k` experts of each token without a sort.

  Equivalent to `jax.lax.top_k(scores.astype(jnp.float32), k)` except on two
  kinds of row:

  *   Ties resolve to the lowest expert id.
  *   NaN, `-inf` and `-FLT_MAX` scores are never selected over a finite score.
      A row made entirely of them gets NaN weights and the experts `0..k-1`.

  The selection runs `k` passes of (row max -> lowest matching column -> mask
  that column out), `O(k * num_experts)` per token instead of the
  `O(num_experts * log^2(num_experts))` of a sort.

  Args:
    scores: `(num_tokens, num_experts)` float32, bfloat16 or float16 router
      scores. They are cast to float32 (exactly) before the selection.
    k: The number of experts to select per token, in `[1, num_experts]`.
    implementation: The implementation to use. By default, `None` is used, which
      selects the best available backend. If a sequence is passed, the first
      implementation that doesn't raise a `NotImplementedError` is used.

  Returns:
    `(weights, indices)`: float32 `weights`, descending, and int32 expert
    `indices`, both `(num_tokens, k)`.

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
      return impl(scores, k)
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
