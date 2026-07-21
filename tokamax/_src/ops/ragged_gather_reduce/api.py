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
"""Ragged gather reduce API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, TypeAlias

import immutabledict
import jax
from tokamax._src.ops.ragged_gather_reduce import base

Implementation: TypeAlias = Literal["xla", "mosaic", "mosaic_tpu"]

_IMPLEMENTATIONS = dict(xla=base.RaggedGatherReduce())
_DEFAULT_IMPLEMENTATIONS = ("xla",)

try:
  from tokamax._src.ops.ragged_gather_reduce import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  _IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasTpuRaggedGatherReduce()
  )
  _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass


IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    *,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ) = None,
) -> jax.Array:
  """Ragged gather reduce operation.

  Args:
    x: Input array of shape (in_size, hidden_size).
    indices: 1D array of indices of shape (in_size,).
    topk_weights: 1D array of weights of shape (in_size,).
    valid_rows_mask: 1D boolean array indicating valid rows of shape (in_size,).
    reduce_group_size: Number of consecutive rows to reduce (sum) together.
    implementation: The implementation to use.

  Returns:
    Reduced array of shape (in_size // reduce_group_size, hidden_size).
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
      if impl in ("mosaic", "mosaic_tpu"):
        impl = "mosaic_tpu"
      if impl not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation: {impl}. You may need to add a dependency"
            " on the corresponding backend."
        )
      impl = IMPLEMENTATIONS[impl]

    try:
      return impl(
          x=x,
          indices=indices,
          topk_weights=topk_weights,
          valid_rows_mask=valid_rows_mask,
          reduce_group_size=reduce_group_size,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
