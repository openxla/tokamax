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
"""TopK Op API."""

from collections.abc import Callable, Sequence
from typing import Literal
import immutabledict
import jax
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.tpu.topk import base

type Implementation = Literal["mosaic_tpu", "xla"]

_implementations = dict(xla=base.TopK())
_DEFAULT_IMPLEMENTATION = ("xla",)

try:
  from tokamax._src.ops.experimental.tpu.topk import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _implementations["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuTopK()
  _DEFAULT_IMPLEMENTATION = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass

IMPLEMENTATIONS = immutabledict.immutabledict(_implementations)


def top_k(
    scores: Float[Array, "b n"] | Int[Array, "b n"],
    k: int,
    row_lengths: Int[Array, "b"] | None = None,
    *,
    return_scores: bool = False,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ) = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Selects exact top-k indices per row.

  Args:
    scores: 2D array of shape (b, n), with float32 dtype or int32 holding raw
      float32 bits.
    k: Integer specifying the number of top entries per row.
    row_lengths: Optional int32 array of shape (b,) specifying effective row
      lengths. Defaults to n for all rows.
    return_scores: If True, also returns the selected scores as int32 raw
      float32 bits (-inf bits in padded slots).
    implementation: Can be set to "mosaic_tpu" or "xla" explicitly.

  Returns:
    Top-k column indices of shape (b, k) with int32 dtype, -1 suffix-padded, or
    a tuple (indices, scores_bits) if `return_scores=True`.
  """
  if implementation is not None:
    if isinstance(implementation, str):
      if implementation in IMPLEMENTATIONS:
        return IMPLEMENTATIONS[implementation](
            scores,
            k,
            row_lengths,
            return_scores=return_scores,
        )
      else:
        raise ValueError(f"Unsupported implementation: {implementation}")
    impl_seq = implementation
  else:
    impl_seq = _DEFAULT_IMPLEMENTATION

  errors = []
  for impl in impl_seq:
    if isinstance(impl, str):
      if impl not in IMPLEMENTATIONS:
        continue
      impl_fn = IMPLEMENTATIONS[impl]
    else:
      impl_fn = impl

    try:
      return impl_fn(
          scores,
          k,
          row_lengths,
          return_scores=return_scores,
      )
    except NotImplementedError as e:
      if len(impl_seq) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("All implementations failed", errors)
