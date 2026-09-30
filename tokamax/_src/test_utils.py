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
"""Tokamax test utilities."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np


def assert_trees_all_close(
    *trees: chex.ArrayTree,
    rtol: float = 1e-06,
    atol: float | dict[float, float] = 0.0,
):
  """Similar to `chex.assert_trees_all_close`, but with relaxed tolerances.

  The implementation attempts to emulate `chex._assert_trees_all_close_static`
  as much as possible.

  Args:
    *trees: Trees to compare.
    rtol: Relative tolerance.
    atol: Absolute tolerance. When a dict, its keys represent the fraction of
      values that will be compared using the corresponding tolerance value.

  Example:
  ```
  def test_foo(self, foo):
    test_utils.assert_trees_all_close(foo, bar, atol={0.99: 1e-5, 1.0: 1e-4})
  ```
  Means that 99% of the values must have an absolute difference of at most
  1e-5 and all 100% of the values must have an absolute difference of at most
  1e-4.
  """
  if isinstance(atol, float):
    chex.assert_trees_all_close(*trees, rtol=rtol, atol=atol)
    return
  assert isinstance(atol, dict)

  def jnp_to_np_array(arr) -> np.ndarray:
    if getattr(arr, "dtype", None) == jnp.bfloat16:
      # Numpy does not support `bfloat16`.
      arr = arr.astype(jnp.float32)
    return jax.device_get(arr)

  def assert_fn(arr_1, arr_2):
    arr_1 = jnp_to_np_array(arr_1)
    arr_2 = jnp_to_np_array(arr_2)
    msg = "Error in value equality check: Values not approximately equal"
    # Sort the tolerances in descending order, so that the first tolerance
    # that exceeds the absolute difference is the most relaxed one.
    atol_sorted = sorted(atol.items(), key=lambda x: x[1], reverse=True)
    for fraction, tolerance in atol_sorted:
      matches = np.isclose(arr_1, arr_2, rtol=rtol, atol=tolerance)
      if np.count_nonzero(matches) < np.ceil(fraction * arr_1.size):
        np.testing.assert_allclose(  # Recreate original error behaviour.
            arr_1, arr_2, rtol=rtol, atol=tolerance, err_msg=msg
        )

  def cmp_fn(arr_1, arr_2) -> bool:
    try:
      # Raises an AssertionError if values are not close.
      assert_fn(arr_1, arr_2)
    except AssertionError:
      return False
    return True

  def err_msg_fn(arr_1, arr_2) -> str:
    try:
      assert_fn(arr_1, arr_2)
    except AssertionError as e:
      return (
          f"{str(e)} \nOriginal dtypes: "
          f"{np.asarray(arr_1).dtype}, {np.asarray(arr_2).dtype}"
      )
    return ""

  chex.assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


def get_tpu_version() -> int:
  """Returns the numeric version of the TPU, or -1 if not on TPU."""
  kind = jax.devices()[0].device_kind
  if "TPU" not in kind:
    return -1
  if kind.endswith(" lite"):
    kind = kind[: -len(" lite")]
  if kind.endswith("p") or kind.endswith("e"):
    kind = kind[:-1]
  if kind == "TPU7x":
    return 7
  assert kind[:-1] == "TPU v", kind
  return int(kind[-1])


# Stand-ins for the parts of `jax._src.test_util` that batched RPA tests use.
# `jax._src` is not visible outside JAX, so those tests import this module as
# `jtu` instead.

JaxTestLoader = absltest.TestLoader

# Default tolerances when a caller passes neither atol nor rtol, mirroring
# `jax._src.test_util`'s per-dtype defaults for the dtypes used here.
_DEFAULT_TOLERANCE = {
    np.dtype(jnp.float8_e4m3fn): 1e-1,
    np.dtype(jnp.float8_e5m2): 1e-1,
    np.dtype(jnp.bfloat16): 1e-2,
    np.dtype(np.float16): 1e-3,
    np.dtype(np.float32): 1e-6,
    np.dtype(np.float64): 1e-15,
}


def _tolerance(x, y) -> float:
  tols = [
      _DEFAULT_TOLERANCE.get(np.dtype(getattr(a, "dtype", np.float32)), 0.0)
      for a in (x, y)
  ]
  return max(tols)


def is_device_tpu_at_least(version: int) -> bool:
  if jax.default_backend() != "tpu":
    return False
  try:
    return get_tpu_version() >= version
  except (AssertionError, ValueError):
    return False


def with_config(**config):
  """Class decorator that applies `jax.config` values for each test."""

  def decorator(cls):
    original_setup = cls.setUp
    original_teardown = cls.tearDown

    @functools.wraps(original_setup)
    def setUp(self):  # pylint: disable=invalid-name
      self._jtu_saved_config = {k: getattr(jax.config, k) for k in config}
      for k, v in config.items():
        jax.config.update(k, v)
      original_setup(self)

    @functools.wraps(original_teardown)
    def tearDown(self):  # pylint: disable=invalid-name
      original_teardown(self)
      for k, v in self._jtu_saved_config.items():
        jax.config.update(k, v)

    cls.setUp = setUp
    cls.tearDown = tearDown
    return cls

  return decorator


class JaxTestCase(parameterized.TestCase):
  """Test case with the `assertAllClose`/`assertArraysEqual` helpers."""

  def assertAllClose(self, x, y, *, atol=None, rtol=None, err_msg=""):  # pylint: disable=invalid-name
    tol = _tolerance(x, y)
    np.testing.assert_allclose(
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        atol=tol if atol is None else atol,
        rtol=tol if rtol is None else rtol,
        err_msg=err_msg,
    )

  def assertArraysEqual(self, x, y, *, err_msg=""):  # pylint: disable=invalid-name
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=err_msg)
