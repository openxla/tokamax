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

from collections.abc import Callable, Hashable
from typing import Any, ClassVar, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src import ad
from tokamax._src import jaxtyping
from tokamax._src import precision as precision_lib
from tokamax._src.ops import op

type AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray
type FusedWeights = Float[Array, 'K 2 N']
type UnfusedWeights = tuple[Float[Array, 'K N'], Float[Array, 'K N']]
type Residuals = Float[Array, '*B M 2 N']
CanonicalPrecision = precision_lib.CanonicalPrecision


class GatedLinearUnit[C, K: Hashable](op.Op[Any, jax.Array, Residuals, C, K]):
  """Gated linear unit."""

  def __post_init__(self):
    if (self.vjp is None) and (type(self) is not GatedLinearUnit):  # pylint: disable=unidiomatic-typecheck
      object.__setattr__(self, 'vjp', GatedLinearUnitVjp())

  @override
  def bind(
      self,
      x: Float[Array | AbstractArray, '*B M K'],
      weights: (
          Float[Array | AbstractArray, 'K 2 N']
          | tuple[
              Float[Array | AbstractArray, 'K N'],
              Float[Array | AbstractArray, 'K N'],
          ]
      ),
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
      config: C,
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
      if not return_residuals:
        dtype = jnp.promote_types(x.dtype, jnp.float32)
        gate = gate.astype(dtype)
        projection = projection.astype(dtype)
        if activation is not None:
          gate = activation(gate)
        return (gate * projection).astype(x.dtype), None
      y = jnp.stack([gate, projection], axis=-2)
    else:
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

  def _with_vmap[W, T](
      self,
      fwd: Callable[[jax.Array, W], T],
      *,
      fallback_to_sequential: bool = True,
  ) -> Callable[[jax.Array, W], T]:
    """Applies a vmap rule to a fwd function."""
    fwd_vmap = jax.custom_batching.custom_vmap(fwd)

    def vmap_rule(axis_size, in_batched, x, weights):
      x_batched, weights_batched = in_batched
      # `vmap` of `x` is trivially supported for all backends.
      if x_batched and not any(jax.tree.leaves(weights_batched)):
        out = fwd_vmap(x, weights)
        return out, jax.tree.map(lambda _: True, out)

      if fallback_to_sequential:
        sequential_vmap = jax.custom_batching.sequential_vmap(fwd)
        return sequential_vmap.vmap_rule(axis_size, in_batched, x, weights)

      # Otherwise, fallback to inner `vmap` rule.
      in_axes = jax.tree.map(lambda b: 0 if b else None, tuple(in_batched))
      out = jax.vmap(fwd, in_axes=in_axes, axis_size=axis_size)(x, weights)
      return out, jax.tree.map(lambda _: True, out)

    fwd_vmap.def_vmap(vmap_rule)
    return fwd_vmap


type Grads = tuple[Float[Array, '*B M K'], FusedWeights | UnfusedWeights]


class GatedLinearUnitVjp[C, K](op.Op[Any, Grads, Any, C, K]):
  """Gated linear unit VJP."""

  supports_batched_args_capture: ClassVar[bool] = False

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
      config: C,
  ) -> tuple[Grads, None]:
    """Gated linear unit VJP."""
    del out, config  # Unused.
    if return_residuals:
      raise NotImplementedError('`return_residuals=True` is not supported.')

    def vjp_core(residuals, dout, x, weights):
      def fn(x_in, w_in):
        out_in, res_in = GatedLinearUnit()(
            x_in,
            w_in,
            activation=activation,
            precision=precision,
            return_residuals=True,
        )
        return out_in, (x_in, w_in, res_in)

      assert (
          vjp_fn := ad.get_vjp_taking_residuals(fn, x, weights)
      ) is not None
      return vjp_fn((x, weights, residuals), dout)

    vjp_vmap = jax.custom_batching.custom_vmap(vjp_core)

    def _sum_unmapped(leaf: jax.Array) -> jax.Array:
      acc_dtype = jnp.promote_types(leaf.dtype, jnp.float32)
      return jnp.sum(leaf.astype(acc_dtype), axis=0).astype(leaf.dtype)

    @vjp_vmap.def_vmap
    def _vmap_rule(axis_size, in_batched, residuals, dout, x, weights):
      res_b, dout_b, x_b, w_b = in_batched
      if not any(jax.tree.leaves(w_b)):
        bcast = lambda t, b: (
            t
            if b
            else jnp.broadcast_to(jnp.expand_dims(t, 0), (axis_size, *t.shape))
        )
        dx, dw = vjp_vmap(
            bcast(residuals, res_b),
            bcast(dout, dout_b),
            bcast(x, x_b),
            weights,
        )
        if not x_b:
          dx = _sum_unmapped(dx)
        return (dx, dw), (x_b, jax.tree.map(lambda _: False, dw))

      in_axes = (
          0 if res_b else None,
          0 if dout_b else None,
          0 if x_b else None,
          jax.tree.map(lambda b: 0 if b else None, w_b),
      )
      dx, dw = jax.vmap(vjp_core, in_axes=in_axes, axis_size=axis_size)(
          residuals, dout, x, weights
      )
      if not x_b:
        dx = _sum_unmapped(dx)
      dw = jax.tree.map(
          lambda g, b: g if b else _sum_unmapped(g), dw, w_b
      )
      return (dx, dw), (x_b, w_b)

    return vjp_vmap(residuals, dout, x, weights), None
