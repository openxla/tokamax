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
"""Ragged gather API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
from tokamax._src.ops.ragged_gather import base

type Implementation = Literal["xla", "mosaic", "mosaic_tpu", "mosaic_tpu_v2"]

_IMPLEMENTATIONS = dict(xla=base.RaggedGather())
_DEFAULT_IMPLEMENTATIONS = ("xla",)

try:
  from tokamax._src.ops.ragged_gather import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuRaggedGather()
  _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

try:
  from tokamax._src.ops.ragged_gather import pallas_mosaic_v2_tpu  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  _IMPLEMENTATIONS["mosaic_tpu_v2"] = pallas_mosaic_v2_tpu.PallasV2TpuRaggedGather()
  if "mosaic_tpu_v2" not in _DEFAULT_IMPLEMENTATIONS:
    _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu_v2",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def ragged_gather(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    *,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ) = None,
) -> jax.Array:
  """Ragged gather operation.

  Args:
    x: Input array of shape (in_size, hidden_size).
    indices: 1D array of indices of shape (out_size,).
    start: 1D scalar array indicating sequence start index.
    end: 1D scalar array indicating sequence end index.
    implementation: The implementation to use.

  Returns:
    Gathered array of shape (out_size, hidden_size).
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
      if impl == "mosaic":
        impl = (
            "mosaic_tpu_v2"
            if "mosaic_tpu_v2" in IMPLEMENTATIONS
            else "mosaic_tpu"
        )
      if impl not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation: {impl}. You may need to add a dependency"
            " on the corresponding backend."
        )
      impl = IMPLEMENTATIONS[impl]

    try:
      return impl(x=x, indices=indices, start=start, end=end)
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
