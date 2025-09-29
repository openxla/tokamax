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

from collections.abc import Callable
import dataclasses
import functools
import inspect
from typing import Any, Generic, ParamSpec, TypeVar

import jax
import jax.numpy as jnp
from tokamax._src import utils


_P = ParamSpec('_P')
_T = TypeVar('_T')
_zip = functools.partial(zip, strict=True)


def _broadcast_prefix(prefix_tree, full_tree):
  result = []
  num_leaves = lambda t: jax.tree.structure(t).num_leaves
  add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))
  jax.tree.map(add_leaves, prefix_tree, full_tree, is_leaf=lambda x: x is None)
  return result


# TODO: `ParamSpec` -> `TypeVarTuple` when pytype supports it.
def vmap_maybe_bcast(f: Callable[_P, _T], in_axes: Any) -> Callable[_P, _T]:
  """`vmap`s `f` over (possibly broadcast) axes of its arguments."""

  @functools.wraps(f)
  def vmapped(*args: _P.args, **kwargs: _P.kwargs):
    assert not kwargs
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


def vmap_split(
    f: Callable[_P, _T], in_axes: Any, *, num_parts: int
) -> Callable[_P, _T]:
  """`vmap`s `f` over (possibly broadcast) parts of axes of its arguments."""

  @functools.wraps(f)
  def vmapped(*args: _P.args, **kwargs: _P.kwargs):
    assert not kwargs
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

  vmap_axes: tuple[int | None, ...]

  def __init__(self, shape, dtype, vmap_axes: tuple[int | None, ...]):
    super().__init__(shape, dtype)
    self.vmap_axes = vmap_axes

  @property
  def vmap_axis_sizes(self) -> tuple[int | None, ...]:
    """Returns the vmap axis sizes of the given array."""
    shape = list(self.shape)
    return tuple(None if a is None else shape.pop(a) for a in self.vmap_axes)

  @property
  def inner_shape(self) -> tuple[int, ...]:
    """Returns the shape of the array inside of the `vmap`ped context."""
    shape = list(self.shape)
    for axis in self.vmap_axes:
      if axis is not None:
        shape.pop(axis)
    return tuple(shape)

  def __repr__(self):
    return f'{super().__repr__()[:-1]}, vmap_axes={self.vmap_axes})'

  def __eq__(self, other):
    if not isinstance(other, jax.ShapeDtypeStruct):
      return False
    if isinstance(other, BatchedShapeDtype):
      return super().__eq__(other) and (self.vmap_axes == other.vmap_axes)
    return super().__eq__(other) and not self.vmap_axes

  def __hash__(self):
    return hash((super().__hash__(), self.vmap_axes))


def _unique_not_none_value(*args):
  (value,) = {arg for arg in args if arg is not None}
  return value


@dataclasses.dataclass(frozen=True)
class Batched(Generic[_T]):
  """Wrapper for `vmap` axes on some values."""

  values: _T

  @property
  def vmap_axis_sizes(self) -> tuple[int, ...]:
    values = self.values
    if isinstance(values, inspect.BoundArguments):
      values = values.arguments
    values_flat = jax.tree.leaves(values)
    arrays = (a for a in values_flat if isinstance(a, BatchedShapeDtype))
    array_axis_sizes = (a.vmap_axis_sizes for a in arrays)
    return tuple(map(_unique_not_none_value, *array_axis_sizes))

  def __getattr__(self, name: str) -> Any:
    return getattr(self.values, name)


def capture_batched_args(fn: Callable[..., Any]) -> Callable[..., Any]:
  """Captures the batched arguments shapes and passes them to the function."""

  def bind(*args, **kwargs):
    kwargs.setdefault('batched_args', None)
    ba = inspect.signature(fn).bind(*args, **kwargs)
    ba.apply_defaults()
    ba.arguments.pop('batched_args')
    return ba

  @functools.wraps(fn)
  def wrapped(*args, batched_args=None, **kwargs):
    ba = bind(*args, batched_args=batched_args, **kwargs)
    args_flat, args_tree = jax.tree.flatten((ba.args, ba.kwargs))
    is_array = lambda x: isinstance(x, jax.Array)
    arrays, other, merge = utils.split_merge(is_array, args_flat)

    if batched_args is None:
      shapes = [BatchedShapeDtype(a.shape, a.dtype, ()) for a in arrays]
    else:
      bargs_bkwargs = (batched_args.args, batched_args.kwargs)
      bargs_flat = args_tree.flatten_up_to(bargs_bkwargs)
      shapes = [s for s, a in _zip(bargs_flat, args_flat) if is_array(a)]

    fn_flat_sig = inspect.Signature.from_callable(lambda *arrays: None)
    batched_args = Batched(fn_flat_sig.bind(*shapes))

    def fn_flat(*arrays, batched_args=batched_args):
      args, kwargs = args_tree.unflatten(merge(arrays, other))
      bargs, bkwargs = args_tree.unflatten(merge(batched_args.args, other))
      batched_args = Batched(bind(*bargs, **bkwargs))
      return fn(*args, batched_args=batched_args, **kwargs)

    def vmap_rule(axis_size, in_batched, *args):
      in_axes = [0 if b else None for b in in_batched]
      new_shapes = []
      for arg, axis, shape in _zip(args, in_axes, shapes):
        exp_shape = (() if axis is None else (axis_size,)) + shape.shape
        assert (arg.shape, arg.dtype) == (exp_shape, shape.dtype)
        new_vmap_axes = (axis, *shape.vmap_axes)
        new_shape = BatchedShapeDtype(arg.shape, arg.dtype, new_vmap_axes)
        new_shapes.append(new_shape)

      new_batched_args = Batched(fn_flat_sig.bind(*new_shapes))

      @capture_batched_args
      def vmap_fn_flat(*arrays, batched_args):
        fn_flat_closed = functools.partial(fn_flat, batched_args=batched_args)
        return jax.vmap(fn_flat_closed, in_axes=in_axes)(*arrays)

      out = vmap_fn_flat(*args, batched_args=new_batched_args)
      return out, jax.tree.map(lambda _: True, out)

    fn_vmap = jax.custom_batching.custom_vmap(fn_flat)
    fn_vmap.def_vmap(vmap_rule)
    return fn_vmap(*arrays)

  orig_sig = inspect.signature(fn)
  params = orig_sig.parameters.copy()
  params['batched_args'] = params['batched_args'].replace(
      kind=inspect.Parameter.KEYWORD_ONLY, default=None
  )
  wrapped.__signature__ = orig_sig.replace(parameters=tuple(params.values()))
  return wrapped
