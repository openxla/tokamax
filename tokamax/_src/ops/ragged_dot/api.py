# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Ragged dot API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, TypeAlias

import immutabledict
import jax
from jax.extend import backend
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import base

GroupSizes = base.GroupSizes
QuantizedArray = quantization.QuantizedArray
Implementation: TypeAlias = Literal["mosaic", "triton", "xla"]

IMPLEMENTATIONS = dict(xla=base.RaggedDot())
_DEFAULT_IMPLEMENTATION = ("xla",)

try:
  from tokamax._src.ops.ragged_dot import pallas_triton  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["triton"] = pallas_triton.PallasTritonRaggedDot()
  _DEFAULT_IMPLEMENTATION = ("triton",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass

try:
  from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["mosaic_gpu"] = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot()
  _DEFAULT_IMPLEMENTATION = ("mosaic",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass

try:
  from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()
  if "mosaic" not in _DEFAULT_IMPLEMENTATION:
    _DEFAULT_IMPLEMENTATION = ("mosaic",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass


IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(IMPLEMENTATIONS)
)


def ragged_dot(
    lhs: Float[Array | QuantizedArray, "M K"],
    rhs: Float[Array | QuantizedArray, "G K N"],
    group_sizes: Int[Array, "G"] | base.GroupSizes,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    group_offset: Array | None = None,
    *,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ) = None,
) -> Float[Array, "M N"]:  # pylint: disable=g-doc-args
  """Ragged matrix multiplication.

  This has the same API as `jax.lax.ragged_dot`.

  Args:
    lhs: (m, k) shaped array.
    rhs: (g, k, n) shaped array.
    group_sizes: (g,) shaped array with integer element type, where g denotes
      number of groups. The ith element indicates the size of ith group.
    precision: Optional. Consistent with precision argument for
      :func:`jax.lax.dot`.
    preferred_element_type: Optional. Consistent with precision argument for
      :func:`jax.lax.dot`.
    group_offset: Optional. (1,) shaped array that indicates the group in
      group_sizes to start computing from. If not specified, defaults to [0].
    implementation: The implementation to use. By default, `None` is used, which
      will automatically select the best available backend, and is guaranteed to
      work on all platforms. If a sequence is passed, the first implementation
      that doesn't raise a `NotImplementedError` is used.

  Returns:
    (m, n) shaped array with `preferred_element_type` element type.
  """
  if group_offset is not None:
    raise NotImplementedError("`group_offset` is not yet supported.")

  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATION

  if not isinstance(implementation, (tuple, list)):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  errors = []
  for impl in implementation:
    if isinstance(impl, str):
      if impl == "mosaic":
        impl = (
            "mosaic_gpu"
            if "NVIDIA" in backend.get_default_device().device_kind
            else "mosaic_tpu"
        )
      if impl not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown implementation: {impl}")

      impl = IMPLEMENTATIONS[impl]

    try:
      return impl(
          lhs,
          rhs,
          group_sizes=group_sizes,
          precision=precision,
          preferred_element_type=preferred_element_type,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
