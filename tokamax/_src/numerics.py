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

PyTree: TypeAlias = Any


@dataclasses.dataclass(frozen=True)
class NumericSummary:
  """Summary properties of an array."""

  has_inf: bool
  has_nan: bool
  min: float
  max: float
  mean: float
  mean_abs: float


@dataclasses.dataclass(frozen=True)
class DiffSummary:
  """Summary of the difference of two arrays."""

  max_absolute_diff_values: tuple[float, float]
  percent_close: float
  allclose: bool

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

  abs_diff = np.abs(expected - actual)
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


RngKey: TypeAlias = jax.Array


class ArrayInitializer(abc.ABC):
  """A callable that returns an array."""

  @abc.abstractmethod
  def __call__(self, key: RngKey) -> jax.Array:
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

  def __call__(self, key: RngKey) -> jax.Array:
    return _int_initializer(
        key, self.shape, self.dtype, self.minval, self.maxval
    )


def _int_initializer(key, shape, dtype, minval=None, maxval=None):
  """Default int initializer for `random_initialize`."""
  dtype = jnp.dtype(dtype)
  if maxval is None:
    maxval = min(jnp.iinfo(dtype).max + 1, 128)
  if minval is None:
    minval = max(jnp.iinfo(dtype).min, -maxval)

  return jax.random.randint(
      key, shape=shape, minval=minval, maxval=maxval, dtype=dtype
  )


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

  key = jax.random.PRNGKey(seed)

  def init_with_layout(x):
    if isinstance(x, ArrayInitializer):
      init = x
    elif isinstance(x, jax.ShapeDtypeStruct):
      dtype = jnp.dtype(x.dtype)

      if 'float' in dtype.name:
        init = lambda key: jax.random.normal(key, shape=x.shape, dtype=dtype)
      elif dtype.name == 'bool':
        init = lambda key: jax.random.bernoulli(key, shape=x.shape)
      elif 'int' in dtype.name:
        init = lambda key: _int_initializer(key, x.shape, dtype)
      else:
        raise NotImplementedError(f'dtype {dtype.name} not supported.')
    else:
      return x

    if getattr(x, 'sharding', None) is not None:
      init = jax.jit(init, out_shardings=x.format)

    nonlocal key
    curr_key, key = jax.random.split(key)
    return init(curr_key)

  is_leaf = lambda x: isinstance(x, ArrayInitializer)
  return jax.tree.map(init_with_layout, x, is_leaf=is_leaf)
