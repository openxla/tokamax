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
"""Normalization op."""

from collections.abc import Callable
import types
from typing import Any, TypeAlias, TypeVar

import jax
from jax.experimental.pallas import fuser
import jax.numpy as jnp
from tokamax._src.ops import op


_Config = TypeVar("_Config")
_Key = TypeVar("_Key")
FusedInputArray: TypeAlias = fuser.Fusion[[], jax.Array]  # pytype: disable=invalid-annotation
Residuals: TypeAlias = tuple[jax.Array | None, jax.Array]  # mean, rstddev


class Normalization(op.Op[Any, jax.Array, Residuals, _Config, _Key]):
  """Normalization op."""

  def bind(
      self,
      x: jax.Array | Callable[[], jax.Array],
      scale: jax.Array | None,
      offset: jax.Array | None,
      *,
      axis: int = -1,
      epsilon: float = 1e-6,
      scale_offset: float = 0.0,
      subtract_mean: bool = True,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds normalization op to the given arguments.

    The op is equivalent to the following JAX code:
    ```
    mean = None if subtract_mean else 0.0
    x_norm = jax.nn.standardize(x, axis=axis, mean=mean, epsilon=epsilon)
    result = x_norm * (scale + scale_offset) + offset
    ```

    Args:
      x: The input array.
      scale: An optional one-dimensional array of scale factors.
      offset: An optional one-dimensional array of offsets.
      axis: The axis along which to normalize.
      epsilon: Epsilon value add to denominator to avoid division by zero.
      scale_offset: An offset added to the scale factors before scaling.
      subtract_mean: If `True`, use standard variance calculation. If `False`,
        assume mean is zero (i.e. RMS norm).
      return_residuals: If `True`, return the residuals.

    Returns:
      The bound arguments.
    """
    if callable(x) and not isinstance(x, fuser.Fusion):
      x = fuser.Fusion(x, ((), {}), jax.eval_shape(x))

    if scale is not None:
      (scale_dim,) = scale.shape
      if scale_dim != x.shape[axis]:
        raise ValueError(
            "`scale` must have the same length as `x` along `axis`."
        )
    if offset is not None:
      (offset_dim,) = offset.shape
      if offset_dim != x.shape[axis]:
        raise ValueError(
            "`offset` must have the same length as `x` along `axis`."
        )

    return super().bind(
        x=x,
        scale=scale,
        offset=offset,
        axis=axis,
        epsilon=epsilon,
        scale_offset=scale_offset,
        subtract_mean=subtract_mean,
        return_residuals=return_residuals,
    )

  def _fwd(
      self,
      x: jax.Array | FusedInputArray,
      scale: jax.Array | None,
      offset: jax.Array | None,
      *,
      axis: int,
      epsilon: float,
      scale_offset: float,
      subtract_mean: bool,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[jax.Array, Residuals | None]:
    if callable(x):
      x = x()
    axis = axis if axis >= 0 else (x.ndim + axis)
    # Always calculate in at least `float32` precision.
    y = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    if subtract_mean:
      mean = jnp.mean(y, axis, keepdims=True)
      y -= mean
    else:
      mean = None

    var = jnp.mean(jnp.square(y), axis, keepdims=True)
    rstddev = jax.lax.rsqrt(var + epsilon)
    y *= rstddev

    param_bcast_axes = [a for a in range(x.ndim) if a != axis]
    if scale is not None:
      y *= jnp.expand_dims(scale + scale_offset, param_bcast_axes)
    if offset is not None:
      y += jnp.expand_dims(offset, param_bcast_axes)

    return y.astype(x.dtype), (mean, rstddev) if return_residuals else None


class NormalizationVjp(op.Op[Any, Any, types.NoneType, _Config, _Key]):
  """Normalization VJP."""

  def _fwd(
      self,
      residuals: Residuals,
      out: jax.Array,
      dout: jax.Array,
      x: jax.Array,
      scale: jax.Array | None,
      offset: jax.Array | None,
      *,
      axis: int,
      epsilon: float,
      scale_offset: float,
      subtract_mean: bool,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[tuple[jax.Array, jax.Array | None, jax.Array | None], None]:
    """Computes normalization VJP `(dx, dscale, doffset)`."""
    del out, config  # Unused.

    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")

    mean, rstddev = residuals
    if (mean is not None) != subtract_mean:
      raise ValueError("`mean` residual inconsistent with `subtract_mean`.")

    axis = axis if axis >= 0 else (x.ndim + axis)
    dtype = jnp.promote_types(x.dtype, jnp.float32)
    if mean is not None and mean.dtype != dtype:
      raise ValueError("`mean` dtype must be at least `float32`")
    if rstddev.dtype != dtype:
      raise ValueError("`rstddev` dtype must be at least `float32`")
    dout = dout.astype(dtype)

    x_norm = x.astype(dtype)
    if mean is not None:
      x_norm -= mean
    x_norm *= rstddev

    param_bcast_axes = [a for a in range(x.ndim) if a != axis]
    if offset is None:
      doffset = None
    else:
      doffset = jnp.sum(dout, param_bcast_axes).astype(offset.dtype)

    if scale is None:
      dscale = None
    else:
      dscale = jnp.sum(dout * x_norm, param_bcast_axes).astype(scale.dtype)
      dout *= scale.astype(dtype) + scale_offset

    dx = dout - jnp.mean(dout * x_norm, axis=axis, keepdims=True) * x_norm
    if mean is not None:
      dx -= jnp.mean(dout, axis=axis, keepdims=True)
    dx *= rstddev
    return (dx.astype(x.dtype), dscale, doffset), None
