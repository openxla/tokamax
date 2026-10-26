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
"""Batching (a.k.a. `vmap`) utilities."""

from collections.abc import Callable, Sequence
import copy
import functools
from typing import Any, Self, override

import jax
from jax.experimental import hijax
import jax.numpy as jnp


def _broadcast_prefix(prefix_tree, full_tree):
  result = []
  num_leaves = lambda t: jax.tree.structure(t).num_leaves
  add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))
  jax.tree.map(add_leaves, prefix_tree, full_tree, is_leaf=lambda x: x is None)
  return result


def vmap_maybe_bcast[*Ts, R](
    f: Callable[[*Ts], R], in_axes: Any
) -> Callable[[*Ts], R]:
  """`vmap`s `f` over (possibly broadcast) axes of its arguments."""

  @functools.wraps(f)
  def vmapped(*args: *tuple[*Ts]) -> R:  # pylint: disable=g-one-element-tuple
    in_axes_flat = _broadcast_prefix(in_axes, args)
    args_flat, args_tree = jax.tree.flatten(args)

    def arg_axis(x, axis):
      if axis is None:
        return x, None
      return (jnp.squeeze(x, axis), None) if x.shape[axis] == 1 else (x, axis)

    args_flat, in_axes_flat = zip(*map(arg_axis, args_flat, in_axes_flat))
    args = args_tree.unflatten(args_flat)

    if all(axis is None for axis in in_axes_flat):  # i.e. axis size is 1
      return jax.tree.map(lambda x: jnp.expand_dims(x, 0), f(*args))

    return jax.vmap(f, args_tree.unflatten(in_axes_flat))(*args)

  return vmapped


def _split_dim(x, axis, num_parts):
  shape = list(x.shape)
  if shape[axis] == 1:
    shape[axis : axis + 1] = (1, 1)
  elif shape[axis] % num_parts == 0:
    shape[axis : axis + 1] = (num_parts, shape[axis] // num_parts)
  else:
    raise ValueError('dim not divisible by `num_parts`')
  return x.reshape(shape)


def vmap_split[*Ts, R](
    f: Callable[[*Ts], R], in_axes: Any, *, num_parts: int
) -> Callable[[*Ts], R]:
  """`vmap`s `f` over (possibly broadcast) parts of axes of its arguments."""

  @functools.wraps(f)
  def vmapped(*args: *tuple[*Ts]) -> R:  # pylint: disable=g-one-element-tuple
    in_axes_flat = _broadcast_prefix(in_axes, args)
    args_flat, args_tree = jax.tree.flatten(args)

    def arg_axis(x, axis):
      if axis is None:
        return x, None
      axis = axis if axis >= 0 else (axis + x.ndim)
      return _split_dim(x, axis, num_parts), axis

    args_flat, in_axes_flat = zip(*map(arg_axis, args_flat, in_axes_flat))
    args = args_tree.unflatten(args_flat)
    return vmap_maybe_bcast(f, args_tree.unflatten(in_axes_flat))(*args)

  return vmapped


class BatchedShapeDtype(jax.ShapeDtypeStruct):
  """A container for the shape, dtype, and vmap axes of an array."""

  __slots__ = ('vmap_axes',)

  # `(axis, axis_size)` pairs (or `None` when the value is broadcast).
  vmap_axes: tuple[tuple[int, int] | None, ...]

  def __init__(self, shape, dtype, vmap_axes: Sequence[tuple[int, int] | None]):
    super().__init__(shape, dtype)
    self.vmap_axes = tuple(vmap_axes)

  @property
  def vmap_shape(self) -> tuple[int, ...]:
    """Returns the shape of the array with the `vmap` axes added."""
    shape = list(self.shape)
    for axis_size in reversed(self.vmap_axes):
      if axis_size is not None:
        shape.insert(*axis_size)
    return tuple(shape)

  @property
  def vmap_axis_sizes(self) -> tuple[int | None, ...]:
    return tuple(
        None if axis_size is None else axis_size[1]
        for axis_size in self.vmap_axes
    )

  def __repr__(self):
    return f'{super().__repr__()[:-1]}, vmap_axes={self.vmap_axes})'

  __str__ = __repr__

  def __eq__(self, other):
    if not isinstance(other, jax.ShapeDtypeStruct):
      return False
    if isinstance(other, BatchedShapeDtype):
      return super().__eq__(other) and (self.vmap_axes == other.vmap_axes)
    return super().__eq__(other) and not self.vmap_axes

  def __hash__(self):
    sds_hash = super().__hash__()
    return hash((sds_hash, self.vmap_axes)) if self.vmap_axes else sds_hash


def get_vmap_axis_sizes(values) -> tuple[int, ...]:
  values_flat = jax.tree.leaves(values)
  arrays = tuple(a for a in values_flat if isinstance(a, BatchedShapeDtype))
  if not arrays:
    return ()
  array_axis_sizes = (a.vmap_axis_sizes for a in arrays)
  return tuple(map(_unique_not_none_value, *array_axis_sizes))


def _unique_not_none_value(*args):
  (value,) = {arg for arg in args if arg is not None}
  return value


class ExpandWithVmapData(hijax.HiPrim):
  """Primitive to make `vmap` data available when `expand`ing."""

  vmap_data: tuple[tuple[Any, Any], ...] = ()

  @override
  def expand(self, *args):
    fn = functools.partial(self.expand_with_vmap_data, self.vmap_data)
    for axis_data, in_dims in reversed(self.vmap_data):
      out_dim = self.batch_dim_rule(axis_data, in_dims)
      fn = hijax.vmap_rule(axis_data, fn, in_dims, out_dim)
    return fn(*map(jnp.asarray, args))

  def expand_with_vmap_data(self, vmap_data, *args):
    """Expand the primitive with the given vmap data (axis data and dims)."""
    raise NotImplementedError(
        f'subclass {type(self)} must implement `expand_with_vmap_data`'
    )

  @override
  def vjp_bwd_retval(self, res, outgrad):
    fn = functools.partial(self.vjp_bwd_retval_with_vmap_data, self.vmap_data)
    for axis_data, in_dims in reversed(self.vmap_data):
      res_dim = self.res_batch_dim_rule(axis_data, in_dims)
      out_dim = self.batch_dim_rule(axis_data, in_dims)
      fn = hijax.vmap_rule(
          axis_data, fn, (res_dim, out_dim), in_dims, sum_match=True
      )
    return fn(jax.tree.map(jnp.asarray, res), outgrad)

  def vjp_bwd_retval_with_vmap_data(self, vmap_data, res, outgrad):
    raise NotImplementedError(
        f'subclass {type(self)} must implement `vjp_bwd_retval` or'
        ' `vjp_bwd_retval_with_vmap_data`'
    )

  @override
  def batch(self, axis_data, args, dims):
    prim, out_dim = self.batch_prim(axis_data, dims)
    return prim(*args), out_dim

  def batch_prim(self, axis_data, in_dims) -> tuple[Self, Any]:
    out_dim = self.batch_dim_rule(axis_data, in_dims)
    prim = copy.copy(self)
    prim.in_avals = hijax.unmap_avals(axis_data, self.in_avals, in_dims)
    prim.out_aval = hijax.unmap_avals(axis_data, self.out_aval, out_dim)
    prim.vmap_data = ((axis_data, in_dims), *self.vmap_data)
    prim.params = dict(self.params, vmap_data=prim.vmap_data)
    super(ExpandWithVmapData, prim).__init__()
    return prim, out_dim

  def batch_dim_rule(self, axis_data, dims):
    raise NotImplementedError(
        f'for vmap support, subclass {type(self)} must implement `batch` or'
        ' `batch_dim_rule`'
    )

  def res_batch_dim_rule(self, axis_data, dims):
    raise NotImplementedError(
        f'for vmap + vjp support, subclass {type(self)} must implement'
        ' `vjp_bwd_retval` or `res_batch_dim_rule`'
    )


# TODO: Remove this once fixed in JAX (`_shard_map_to_lojax` omits
# `in_specs` for `consts` separated from `lo_jaxpr_`).
_orig_shard_map_to_lojax = jax._src.shard_map.shard_map_p.to_lojax  # pylint: disable=protected-access


def _shard_map_to_lojax(*args, **kwargs):
  out = _orig_shard_map_to_lojax(*args, **kwargs)
  eqn = jax._src.core.trace_ctx.trace.frame.tracing_eqns[-1]  # pylint: disable=protected-access
  in_specs = eqn.params['in_specs']
  if num_consts := (len(eqn.invars) - len(in_specs)):
    const_specs = tuple(
        jax._src.shard_map._repspec(t.aval) for t in eqn.invars[:num_consts]  # pylint: disable=protected-access
    )
    eqn.params['in_specs'] = (*const_specs, *in_specs)
  return out


jax._src.shard_map.shard_map_p.to_lojax = _shard_map_to_lojax  # pylint: disable=protected-access
