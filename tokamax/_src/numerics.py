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
from typing import Any

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


# Process large arrays in ~128 MiB float64 chunks so summary reductions do not
# materialize multiple full-size float64 copies on host RAM.
_DIFF_CHUNK_SIZE = 16 * 1024 * 1024


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

  # Keep the original dtype until each chunk is cast to float64.
  exp_flat = np.asarray(expected).ravel()
  act_flat = np.asarray(actual).ravel()
  size = exp_flat.size
  if size == 0:
    raise ValueError('Cannot compute diff summary of empty arrays.')

  sum_sq = 0.0
  num_close = 0
  max_abs_diff = -1.0
  has_max = False
  found_nan = False
  expected_max_val = 0.0
  actual_max_val = 0.0

  for start in range(0, size, _DIFF_CHUNK_SIZE):
    end = start + _DIFF_CHUNK_SIZE
    exp_chunk = exp_flat[start:end].astype(np.float64)
    act_chunk = act_flat[start:end].astype(np.float64)

    diff = exp_chunk - act_chunk
    abs_diff = np.abs(diff)
    sum_sq += float(np.sum(diff * diff))
    # Note: this is not symmetric: "For finite values, isclose uses the
    # following equation to test whether two floating point values are
    # equivalent: absolute(a - b) <= (atol + rtol * absolute(b)). Unlike the
    # built-in math.isclose, the above equation is not symmetric in a and b - it
    # assumes b is the reference value."
    # https://numpy.org/doc/stable/reference/generated/numpy.isclose.html#numpy.isclose
    # As allclose calls isclose, the order matters.
    num_close += int(
        np.sum(
            np.isclose(
                act_chunk,
                exp_chunk,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )
        )
    )

    if found_nan:
      continue
    if equal_nan:
      try:
        idx = int(np.nanargmax(abs_diff))
      except ValueError:
        continue
      chunk_max = float(abs_diff[idx])
    else:
      idx = int(np.argmax(abs_diff))
      chunk_max = float(abs_diff[idx])
      if np.isnan(chunk_max):
        found_nan = True
        expected_max_val = float(exp_chunk[idx])
        actual_max_val = float(act_chunk[idx])
        continue
    if not has_max or chunk_max > max_abs_diff:
      has_max = True
      max_abs_diff = chunk_max
      expected_max_val = float(exp_chunk[idx])
      actual_max_val = float(act_chunk[idx])

  if equal_nan and not has_max:
    raise ValueError('All-NaN slice encountered')

  percent_close = num_close / size
  return DiffSummary(
      max_absolute_diff_values=(expected_max_val, actual_max_val),
      percent_close=percent_close,
      allclose=num_close == size,
      l2_diff=float(np.sqrt(sum_sq)),
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
  sharding = getattr(x, 'sharding', None)
  out_sharding = None if sharding is None else getattr(x, 'format', None)

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

  def init_with_layout(rng, x):
    if isinstance(x, ArrayInitializer):
      return jax.device_put(x(rng))
    if isinstance(x, qwix.QArray):
      if x.zero_point is not None:
        raise NotImplementedError('Zero point not supported.')
      if x.qvalue.dtype != x.qtype:
        raise NotImplementedError('`qvalue.dtype` and `qtype` must match.')

      abstract_qvalue = isinstance(x.qvalue, jax.ShapeDtypeStruct)
      abstract_scale = isinstance(x.scale, jax.ShapeDtypeStruct)

      if abstract_qvalue and abstract_scale:
        x = dataclasses.replace(
            x, qvalue=_as_vmap_shape(x.qvalue), scale=_as_vmap_shape(x.scale)  # pyrefly: ignore[bad-argument-type]
        )
        values = _init_array(rng, jax.ShapeDtypeStruct(x.shape, x.dtype))
        tiled_axes = {i: d for i, d in enumerate(x.scale_tile_shape)}
        return qwix.quantize(values, x.qtype, tiled_axes=tiled_axes)
      elif not abstract_qvalue and not abstract_scale:
        return x
      else:
        raise ValueError(
            '`QArray` values and scales must both be abstract or both concrete.'
        )
    assert isinstance(x, jax.ShapeDtypeStruct)
    return _init_array(rng, x)

  is_leaf = lambda x: isinstance(x, (ArrayInitializer, qwix.QArray))
  x_flat, tree = jax.tree.flatten(x, is_leaf=is_leaf)

  def is_abstract(x):
    return isinstance(x, (jax.ShapeDtypeStruct, ArrayInitializer)) or (
        isinstance(x, qwix.QArray)
        and (
            isinstance(x.qvalue, jax.ShapeDtypeStruct)
            or isinstance(x.scale, jax.ShapeDtypeStruct)
        )
    )

  abstract_arrays, other, merge = utils.split_merge(is_abstract, x_flat)
  rngs = np.random.default_rng(seed).spawn(len(abstract_arrays))

  if jax.process_count() > 1:
    # Initializing arrays sharded across processes launches a program on every
    # process. These must be launched in the same order on all processes, so we
    # can't initialize concurrently.
    arrays = list(map(init_with_layout, rngs, abstract_arrays))
  else:
    with futures.ThreadPoolExecutor() as executor:
      arrays = list(executor.map(init_with_layout, rngs, abstract_arrays))

  return tree.unflatten(merge(arrays, other))
