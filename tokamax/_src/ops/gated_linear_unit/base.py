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
"""Gated linear unit op."""

from collections.abc import Callable
from typing import Any, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src import ad
from tokamax._src import jaxtyping
from tokamax._src import precision as precision_lib
from tokamax._src.ops import op
from typing_extensions import override


_T = TypeVar('_T')
_Config = TypeVar('_Config')
_Key = TypeVar('_Key')
_FwdFn = Callable[[jax.Array, jax.Array], _T]
FusedWeights: TypeAlias = Float[Array, 'K 2 N']
UnfusedWeights: TypeAlias = tuple[Float[Array, 'K N'], Float[Array, 'K N']]
Residuals: TypeAlias = Float[Array, '*B M 2 N']
CanonicalPrecision = precision_lib.CanonicalPrecision


class GatedLinearUnit(op.Op[Any, jax.Array, Residuals, _Config, _Key]):
  """Gated linear unit."""

  def __post_init__(self):
    if (self.vjp is None) and (type(self) is not GatedLinearUnit):  # pylint: disable=unidiomatic-typecheck
      object.__setattr__(self, 'vjp', GatedLinearUnitVjp())

  @override
  def bind(
      self,
      x: Float[Array, '*B M K'],
      weights: FusedWeights | UnfusedWeights,
      *,
      activation: Callable[[jax.Array], jax.Array] | None = None,
      precision: jax.lax.PrecisionLike = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds the arguments for the gated linear unit function."""

    return super().bind(
        x=x,
        weights=weights,
        activation=activation,
        precision=precision_lib.canonicalize_precision(precision),
        return_residuals=return_residuals,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      x: Float[Array, '*B M K'],
      weights: FusedWeights | UnfusedWeights,
      *,
      activation: Callable[[jax.Array], jax.Array] | None,
      precision: CanonicalPrecision,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[Float[Array, '*B M N'], Residuals | None]:
    """Applies a gated linear unit (https://arxiv.org/abs/1612.08083).

    Computes `activation(x @ weights[:, 0]) * x @ weights[:, 1]`.

    Args:
      x: the input array.
      weights: the weights for the linear transformations. Can be a single fused
        array of shape `(K, 2, N)`, or a tuple of unfused arrays, each of shape
        `(K, N)`.
      activation: optional activation function.
      precision: the matrix multiplication precision.
      return_residuals: if True, returns the residuals in addition to the
        output.
      config: the configuration of the op.

    Returns:
      The output array.
    """
    del config  # Unused.
    if isinstance(weights, tuple):
      gate_weights, projection_weights = weights
      projection = jnp.matmul(x, projection_weights, precision=precision)
      gate = jnp.matmul(x, gate_weights, precision=precision)
      dtype = jnp.promote_types(x.dtype, jnp.float32)
      if return_residuals:
        y = jnp.stack([gate, projection], axis=gate.ndim - 1)
      else:
        y = None
      gate = gate.astype(dtype)
      projection = projection.astype(dtype)
      if activation is not None:
        gate = activation(gate)
      out = (gate * projection).astype(x.dtype)
      return out, y

    y = jnp.einsum('...k,kjn->...jn', x, weights, precision=precision)
    # Apply activation and compute product of FP8/FP16/BF16 in FP32.
    dtype = jnp.promote_types(x.dtype, jnp.float32)
    gate, projection = jnp.split(
        jax.lax.collapse(y, -2).astype(dtype),
        2,
        axis=-1,
    )
    if activation is not None:
      gate = activation(gate)
    out = (gate * projection).astype(x.dtype)
    return out, (y if return_residuals else None)

  def _with_vmap(
      self, fwd: _FwdFn, *, fallback_to_sequential: bool = True
  ) -> _FwdFn:
    """Applies a vmap rule to a fwd function."""
    fwd_vmap = jax.custom_batching.custom_vmap(fwd)

    def vmap_rule(axis_size, in_batched, x, weights):
      # `vmap` of `x` is trivially supported for all backends.
      if in_batched == [True, False]:
        out = fwd_vmap(x, weights)
        return out, jax.tree.map(lambda _: True, out)

      if fallback_to_sequential:
        sequential_vmap = jax.custom_batching.sequential_vmap(fwd)
        return sequential_vmap.vmap_rule(axis_size, in_batched, x, weights)

      # Otherwise, fallback to inner `vmap` rule.
      in_axes = [0 if b else None for b in in_batched]
      out = jax.vmap(fwd, in_axes=in_axes)(x, weights)
      return out, jax.tree.map(lambda _: True, out)

    fwd_vmap.def_vmap(vmap_rule)
    return fwd_vmap


class GatedLinearUnitVjp(
    op.Op[
        Any,
        tuple[
            Float[Array, '*B M K'],
            FusedWeights | UnfusedWeights,
        ],
        Any,
        _Config,
        _Key,
    ]
):
  """Gated linear unit VJP."""

  @override
  def _fwd(
      self,
      residuals: Residuals,
      out: Float[Array, '*B M N'],
      dout: Float[Array, '*B M N'],
      x: Float[Array, '*B M K'],
      weights: FusedWeights | UnfusedWeights,
      *,
      activation: Callable[[jax.Array], jax.Array] | None,
      precision: CanonicalPrecision,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[
      tuple[
          Float[Array, '*B M K'],
          FusedWeights | UnfusedWeights,
      ],
      None,
  ]:
    """Gated linear unit VJP."""
    if return_residuals:
      raise NotImplementedError('`return_residuals=True` is not supported.')

    def fn(x, weights):
      out, residuals = GatedLinearUnit()(
          x,
          weights,
          activation=activation,
          precision=precision,
          return_residuals=True,
      )
      return out, (x, weights, residuals)

    vjp_fn = ad.get_vjp_taking_residuals(fn, x, weights)
    return vjp_fn((x, weights, residuals), dout), None
