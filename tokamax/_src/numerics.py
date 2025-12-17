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
import dataclasses
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import qwix
from tokamax._src import batching


PyTree: TypeAlias = Any


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
  x = np.array(x).astype(np.float64)
  return NumericSummary(
      has_inf=np.isinf(x).any(),
      has_nan=np.isnan(x).any(),
      min=np.nanmin(x),
      max=np.nanmax(x),
      mean=np.nanmean(x),
      mean_abs=np.nanmean(np.abs(x)),
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


def _int_initializer(rng, shape, dtype, minval=None, maxval=None):
  """Default int initializer for `random_initialize`."""
  iinfo = jnp.iinfo(jnp.dtype(dtype))
  if maxval is None:
    maxval = min(iinfo.max + 1, 128)
  if minval is None:
    minval = max(iinfo.min, -maxval)
  return rng.integers(minval, maxval, shape).astype(dtype)


def _as_vmap_shape(x: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
  if isinstance(x, batching.BatchedShapeDtype):
    return jax.ShapeDtypeStruct(x.vmap_shape, x.dtype)
  return x


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
  rng = np.random.default_rng(seed)

  def init_with_layout(x):
    if isinstance(x, ArrayInitializer):
      return jax.device_put(x(rng))
    if isinstance(x, qwix.QArray):
      abstract_qvalue = isinstance(x.qvalue, jax.ShapeDtypeStruct)
      abstract_scale = isinstance(x.scale, jax.ShapeDtypeStruct)

      if abstract_qvalue and abstract_scale:
        x = qwix.QArray(_as_vmap_shape(x.qvalue), _as_vmap_shape(x.scale))  # pytype: disable=wrong-arg-types
        dtype = jnp.promote_types(x.dtype, jnp.float32)
        values = rng.standard_normal(size=x.shape, dtype=dtype).astype(x.dtype)
        tiled_axes = {i: d for i, d in enumerate(x.scale_tile_shape)}
        return qwix.quantize(values, x.qvalue.dtype, tiled_axes=tiled_axes)
      elif not abstract_qvalue and not abstract_scale:
        return x
      else:
        raise ValueError(
            '`QArray` values and scales must both be abstract or both concrete.'
        )
    if not isinstance(x, jax.ShapeDtypeStruct):
      return x

    x = _as_vmap_shape(x)
    dtype = jnp.dtype(x.dtype)

    if 'float' in dtype.name:
      dtype_ = jnp.promote_types(dtype, jnp.float32)
      y = rng.standard_normal(size=x.shape, dtype=dtype_).astype(dtype)
    elif dtype.name == 'bool':
      y = rng.binomial(n=1, p=0.5, size=x.shape).astype(dtype)
    elif 'int' in dtype.name:
      y = _int_initializer(rng, x.shape, dtype)
    else:
      raise NotImplementedError(f'dtype {dtype.name} not supported.')

    sharding = getattr(x, 'sharding', None)
    # TODO: Can we consolidate `device_put` into a single call?
    return jax.device_put(y, None if sharding is None else x.format)

  is_leaf = lambda x: isinstance(x, (ArrayInitializer, qwix.QArray))
  return jax.tree.map(init_with_layout, x, is_leaf=is_leaf)
