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
"""Utilities for numerics."""

import abc
from concurrent import futures
import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import qwix
from tokamax._src import batching
from tokamax._src import utils

type PyTree = Any


@dataclasses.dataclass(frozen=True, slots=True)
class NumericSummary:
  """Summary properties of an array."""

  has_inf: bool
  has_nan: bool
  min: float
  max: float
  mean: float
  mean_abs: float


@dataclasses.dataclass(frozen=True, slots=True)
class DiffSummary:
  """Summary of the difference of two arrays."""

  max_absolute_diff_values: tuple[float, float]
  percent_close: float
  allclose: bool
  l2_diff: float

  @property
  def max_absolute_diff(self) -> float:
    a, b = self.max_absolute_diff_values
    return abs(a - b)


# Defaults taken from:
# https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.isclose.html
def array_diff_summary(
    expected: jax.Array | np.ndarray,
    actual: jax.Array | np.ndarray,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> DiffSummary:
  """Produce a summary of the numerics of two arrays."""

  if expected.shape != actual.shape:
    raise ValueError('Arrays x and y must have the same shape.')
  if expected.dtype.name != actual.dtype.name:
    raise ValueError('Arrays x and y must have the same dtype.')

  # Calculate statistics with increased precision.
  expected = np.array(expected).astype(np.float64)
  actual = np.array(actual).astype(np.float64)

  diff = expected - actual
  abs_diff = np.abs(diff)
  l2_diff = float(np.linalg.norm(diff))

  # It's useful logging out the values that caused the max absolute
  # difference.
  max_diff_index = np.nanargmax(abs_diff) if equal_nan else np.argmax(abs_diff)

  expected_max_val = float(np.ravel(expected)[max_diff_index])
  actual_max_val = float(np.ravel(actual)[max_diff_index])

  # Note: this is not symmetric: "For finite values, isclose uses the following
  # equation to test whether two floating point values are equivalent:
  # absolute(a - b) <= (atol + rtol * absolute(b)). Unlike the built-in
  # math.isclose, the above equation is not symmetric in a and b – it assumes
  # b is the reference value."
  # https://numpy.org/doc/stable/reference/generated/numpy.isclose.html#numpy.isclose
  # As allclose calls isclose, the order matters.
  num_close = np.sum(
      np.isclose(
          actual,
          expected,
          rtol=rtol,
          atol=atol,
          equal_nan=equal_nan,
      )
  )
  percent_close = num_close / actual.size

  return DiffSummary(
      max_absolute_diff_values=(expected_max_val, actual_max_val),
      percent_close=percent_close,
      allclose=num_close == actual.size,
      l2_diff=l2_diff,
  )


def array_numeric_summary(x: jax.Array) -> NumericSummary:
  """Produce a numerical summary of an array."""
  # Convert to numpy fp64 array to avoid issues with XLA:GPU numerics.
  x_ = np.array(x).astype(np.float64)
  return NumericSummary(
      has_inf=bool(np.isinf(x_).any()),
      has_nan=bool(np.isnan(x_).any()),
      min=float(np.nanmin(x_)),
      max=float(np.nanmax(x_)),
      mean=float(np.nanmean(x_)),
      mean_abs=float(np.nanmean(np.abs(x_))),
  )


class ArrayInitializer(abc.ABC):
  """A callable that returns an array."""

  sharding: jax.sharding.Sharding | None = None
  format: Any = None

  @abc.abstractmethod
  def __call__(self, rng: np.random.Generator) -> np.ndarray | jax.Array:
    ...

  @property
  @abc.abstractmethod
  def shape(self) -> tuple[int, ...]:
    ...

  @property
  @abc.abstractmethod
  def dtype(self) -> jnp.dtype:
    ...

  def with_sharding(
      self, sharding: jax.sharding.Sharding
  ) -> 'ArrayInitializer':
    """Returns a copy of this initializer with the given sharding."""
    raise NotImplementedError(
        f'{type(self).__name__} must implement with_sharding to be used with a'
        ' sharded abstract mesh.'
    )


class RangedArrayInitializer(jax.ShapeDtypeStruct, ArrayInitializer):
  """A abstract array with a known range."""

  def __init__(self, shape, dtype, minval, maxval):
    jax.ShapeDtypeStruct.__init__(self, shape, dtype)
    self.minval = minval
    self.maxval = maxval

  def __call__(self, rng: np.random.Generator) -> np.ndarray:
    return _int_initializer(
        rng, self.shape, self.dtype, self.minval, self.maxval
    )


class FloatRangedArrayInitializer(jax.ShapeDtypeStruct, ArrayInitializer):
  """An abstract float array initialized uniformly in `[minval, maxval)`."""

  def __init__(self, shape, dtype, minval: float, maxval: float):
    jax.ShapeDtypeStruct.__init__(self, shape, dtype)
    self.minval = minval
    self.maxval = maxval

  def __call__(self, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(self.minval, self.maxval, self.shape).astype(self.dtype)


# Threshold (in number of elements) above which arrays are initialized directly
# on device via `jax.jit` + `jax.random` instead of on the host with NumPy.
# Small arrays stay on the host path to avoid per-shape XLA compile overhead
# across parameterized unit tests.
_DEVICE_INIT_THRESHOLD = 1024 * 1024


def _int_initializer(rng, shape, dtype, minval=None, maxval=None):
  """Default int initializer for `random_initialize`."""
  iinfo = jnp.iinfo(jnp.dtype(dtype))
  if maxval is None:
    maxval = min(iinfo.max + 1, 128)
  if minval is None:
    minval = max(iinfo.min, -maxval)
  if isinstance(rng, np.random.Generator):
    return rng.integers(minval, maxval, shape).astype(dtype)
  return jax.random.randint(rng, shape, minval, maxval).astype(dtype)


def _as_vmap_shape(x: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
  if isinstance(x, batching.BatchedShapeDtype):
    return jax.ShapeDtypeStruct(x.vmap_shape, x.dtype)
  return x


def _init_array(
    rng: np.random.Generator, x: jax.ShapeDtypeStruct
) -> jax.Array:
  """Initializes a `ShapeDtypeStruct` on host (small) or device (large)."""
  x = _as_vmap_shape(x)
  dtype = jnp.dtype(x.dtype)
  try:
    sharding = x.sharding
  except AttributeError:
    sharding = None
  if sharding is None:
    out_sharding = None
  else:
    try:
      out_sharding = x.format
    except AttributeError:
      out_sharding = None

  if jnp.issubdtype(dtype, jnp.floating):
    try:
      dtype_ = jnp.promote_types(dtype, jnp.float32)
    except jax.dtypes.TypePromotionError:
      dtype_ = jnp.float32
    init_host = lambda: rng.standard_normal(size=x.shape, dtype=dtype_).astype(
        dtype
    )
    init_dev = lambda k: jax.random.normal(k, x.shape, dtype=dtype_).astype(
        dtype
    )
  elif dtype.name == 'bool':
    init_host = lambda: rng.binomial(n=1, p=0.5, size=x.shape).astype(dtype)
    init_dev = lambda k: jax.random.bernoulli(k, 0.5, x.shape)
  elif 'int' in dtype.name:
    init_host = lambda: _int_initializer(rng, x.shape, dtype)
    init_dev = lambda k: _int_initializer(k, x.shape, dtype)
  else:
    raise NotImplementedError(f'dtype {dtype.name} not supported.')

  if x.size >= _DEVICE_INIT_THRESHOLD:
    seed = jnp.int32(rng.integers(0, 2**31))
    return jax.jit(
        lambda s: init_dev(jax.random.key(s)), out_shardings=out_sharding
    )(seed)
  # TODO: Can we consolidate `device_put` into a single call?
  return jax.device_put(init_host(), out_sharding)


def _is_tree_leaf(leaf: Any) -> bool:
  return isinstance(leaf, (ArrayInitializer, qwix.QArray))


def _is_abstract_leaf(leaf: Any) -> bool:
  """Returns True if leaf is an abstract array or initializer."""
  if isinstance(leaf, (jax.ShapeDtypeStruct, ArrayInitializer)):
    return True
  if isinstance(leaf, qwix.QArray):
    return isinstance(leaf.qvalue, jax.ShapeDtypeStruct) or isinstance(
        leaf.scale, jax.ShapeDtypeStruct
    )
  return False


def _init_qarray(rng: np.random.Generator, qarray: qwix.QArray) -> qwix.QArray:
  """Initializes an abstract QArray."""
  if qarray.zero_point is not None:
    raise NotImplementedError('Zero point not supported.')
  if qarray.qvalue.dtype != qarray.qtype:
    raise NotImplementedError('`qvalue.dtype` and `qtype` must match.')

  abstract_qvalue = isinstance(qarray.qvalue, jax.ShapeDtypeStruct)
  abstract_scale = isinstance(qarray.scale, jax.ShapeDtypeStruct)

  if not abstract_qvalue and not abstract_scale:
    return qarray
  if not (abstract_qvalue and abstract_scale):
    raise ValueError(
        '`QArray` values and scales must both be abstract or both concrete.'
    )

  qvalue_struct = cast(jax.ShapeDtypeStruct, qarray.qvalue)
  scale_struct = cast(jax.ShapeDtypeStruct, qarray.scale)
  vmap_qvalue: Any = _as_vmap_shape(qvalue_struct)
  vmap_scale: Any = _as_vmap_shape(scale_struct)

  updated_qarray = dataclasses.replace(
      qarray,
      qvalue=vmap_qvalue,
      scale=vmap_scale,
  )
  values = _init_array(
      rng, jax.ShapeDtypeStruct(updated_qarray.shape, updated_qarray.dtype)
  )
  tiled_axes = {i: dim for i, dim in enumerate(updated_qarray.scale_tile_shape)}
  return qwix.quantize(values, updated_qarray.qtype, tiled_axes=tiled_axes)


def _init_leaf(rng: np.random.Generator, leaf: Any) -> Any:
  """Initializes a single abstract array leaf with layout and sharding."""
  if isinstance(leaf, ArrayInitializer):
    leaf_dict = vars(leaf)
    sharding = leaf_dict.get('sharding', None)
    out_sharding = (
        leaf_dict.get('format', sharding) if sharding is not None else None
    )
    return jax.device_put(leaf(rng), out_sharding)

  if isinstance(leaf, qwix.QArray):
    return _init_qarray(rng, leaf)

  assert isinstance(leaf, jax.ShapeDtypeStruct)
  return _init_array(rng, leaf)


def random_initialize(x: PyTree, seed: int = 0) -> PyTree:
  """Randomly initialize all abstract arrays in a PyTree.

  Abstract arrays can be represented as `ShapeDtypeStruct` or
  `BatchedShapeDtype` objects. All `ArrayInitializer` callables will be replaced
  by the output of the call.

  Arguments:
    x: a PyTree.
    seed: the random seed to initialize the arrays.

  Returns:
    A new PyTree with each abstract array replaced by a randomly initialized
    `jax.Array`.
  """
  x_flat, tree = jax.tree.flatten(x, is_leaf=_is_tree_leaf)
  abstract_arrays, other, merge = utils.split_merge(_is_abstract_leaf, x_flat)
  rngs = np.random.default_rng(seed).spawn(len(abstract_arrays))

  if jax.process_count() > 1:
    # Initializing arrays sharded across processes launches a program on every
    # process. These must be launched in the same order on all processes, so we
    # can't initialize concurrently.
    arrays = list(map(_init_leaf, rngs, abstract_arrays))
  else:
    with futures.ThreadPoolExecutor() as executor:
      arrays = list(executor.map(_init_leaf, rngs, abstract_arrays))

  return tree.unflatten(merge(arrays, other))
