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
"""Base class for TopK operator."""

import functools
from typing import Any, TypeVar, override
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Shaped
from tokamax._src import jaxtyping
from tokamax._src.ops import op

_Config = TypeVar("_Config")


@functools.partial(jax.jit, static_argnames=["k", "axis", "is_stable"])
def topk(
    operand: jax.Array,
    k: int,
    values: jax.Array | None = None,
    *,
    axis: int = -1,
    is_stable: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """Pure JAX reference implementation for TopK.

  Args:
    operand: Input operand array of shape (*batch_dims, N).
    k: Number of top elements to select.
    values: Optional input values of shape (*batch_dims, N) with int32 dtype. If
      None, 0..N-1 indices along the last dimension are used as values.
    axis: Optional integer specifying the axis along which to compute top k.
    is_stable: Optional boolean specifying whether to preserve relative order.

  Returns:
    A tuple (topk_values, topk_indices), both of shape (*batch_dims, k).
  """
  if values is None:
    top_keys, top_indices = jax.lax.top_k(operand, k, axis=axis)
    return top_keys, top_indices.astype(jnp.int32)
  else:
    top_keys, top_indices = jax.lax.top_k(operand, k, axis=axis)
    top_values = jnp.take_along_axis(values, top_indices, axis=axis)
    return top_keys, top_values


class TopK(op.Op[Any, tuple[jax.Array, jax.Array], None, _Config, Any]):
  """Tokamax operator for TopK."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      operand: Shaped[Array, "*batch N"],
      k: int,
      values: Int[Array, "*batch N"] | None = None,
      *,
      axis: int = -1,
      is_stable: bool = True,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    if k <= 0:
      raise ValueError(f"k must be positive, got {k}.")
    if axis != -1 and axis != operand.ndim - 1:
      raise NotImplementedError(
          f"Only axis=-1 is currently supported, got axis={axis}."
      )
    if operand.shape[axis] < k:
      raise ValueError(
          f"Dimension {axis} of operand ({operand.shape[axis]}) must be >= k"
          f" ({k})."
      )
    if values is not None:
      if values.shape != operand.shape:
        raise ValueError(
            f"values shape {values.shape} must match operand shape"
            f" {operand.shape}."
        )
    return super().bind(
        operand=operand,
        k=k,
        values=values,
        axis=axis,
        is_stable=is_stable,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      operand: Shaped[Array, "*batch N"],
      k: int,
      values: Int[Array, "*batch N"] | None = None,
      *,
      axis: int = -1,
      is_stable: bool = True,
      return_residuals: bool = False,
      config: _Config | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    return (
        topk(operand, k, values, axis=axis, is_stable=is_stable),
        None,
    )
