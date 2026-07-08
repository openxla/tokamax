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
from typing import Literal, TypeAlias
import immutabledict
import jax
from jaxtyping import Array, Int, Shaped
from tokamax._src.ops.experimental.tpu.topk import base

Implementation: TypeAlias = Literal["mosaic_tpu", "xla"]

_implementations = {
    "xla": base.TopK(),
}
_DEFAULT_IMPLEMENTATION = ("xla",)

try:
  from tokamax._src.ops.experimental.tpu.topk import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  _implementations["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuTopK()
  _DEFAULT_IMPLEMENTATION = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass

IMPLEMENTATIONS = immutabledict.immutabledict(_implementations)


def top_k(
    operand: Shaped[Array, "*batch N"],
    k: int,
    values: Int[Array, "*batch N"] | None = None,
    *,
    axis: int = -1,
    is_stable: bool = True,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., tuple[jax.Array, jax.Array]]]
        | None
    ) = None,
) -> tuple[jax.Array, jax.Array]:
  """Based on the jax.lax.top_k API.

  Returns top ``k`` values and their indices along the specified axis of
  ``operand``.

  Args:
    operand: N-dimensional array of non-complex type. Shape (*batch_dims, N).
    k: Integer specifying the number of top entries.
    values: Optional input values array of shape (*batch_dims, N) with integer
      dtype. If None, 0..N-1 indices along the last dimension are used.
    axis: Optional integer specifying the axis along which to compute the top
      ``k`` entries. Default is -1, indicating the last axis.
    is_stable: Optional boolean specifying whether to preserve the relative
      order of equal elements. If True (default), equal elements preserve their
      relative order from the input; otherwise, their order is unspecified.
    implementation: By default `None` will be used to pick the best available
      backend. Can be set to "mosaic_tpu" or "xla" explicitly.

  Returns:
    A tuple (values, indices) where

    - values is an array containing the top k values along the specified axis.
    - indices is an array containing the indices corresponding to values.

  Raises:
    ValueError: If an unsupported implementation is specified.
    NotImplementedError: If an unsupported axis is specified.
    ExceptionGroup: If all implementations fail.
  """
  if axis != -1 and axis != operand.ndim - 1:
    raise NotImplementedError(
        f"Only axis=-1 is currently supported, got axis={axis}."
    )

  if implementation is not None:
    if isinstance(implementation, str):
      if implementation in IMPLEMENTATIONS:
        return IMPLEMENTATIONS[implementation](
            operand, k, values, axis=axis, is_stable=is_stable
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
      return impl_fn(operand, k, values, axis=axis, is_stable=is_stable)
    except NotImplementedError as e:
      if len(impl_seq) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
