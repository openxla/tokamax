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
"""Ragged dot test base."""

import functools

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics
from tokamax._src import quantization
from tokamax._src import test_utils
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import arg_specs


def ref(lhs, rhs, group_sizes, preferred_element_type=None):
  """Reference implementation of ragged dot."""
  if isinstance(lhs, quantization.QuantizedArray):
    lhs = lhs.recompose()

  if isinstance(rhs, quantization.QuantizedArray):
    rhs = rhs.recompose()

  if jnp.result_type(lhs, rhs) == jnp.float32:
    precision = jax.lax.Precision.HIGHEST
  else:
    precision = None

  if lhs.dtype != rhs.dtype:
    input_dtype = jnp.result_type(lhs.dtype, rhs.dtype)
    lhs, rhs = lhs.astype(input_dtype), rhs.astype(input_dtype)

  return jax.lax.ragged_dot(
      lhs,
      rhs,
      group_sizes=jnp.asarray(group_sizes),
      precision=precision,
      preferred_element_type=preferred_element_type,
  )


NAMED_ARG_SPECS = {
    s.full_name: s.args for s in arg_specs.ARG_SPECS if "primary" in s.tags
}


# pylint: disable=missing-function-docstring
class RaggedDotTestBase(parameterized.TestCase):
  """Base class for ragged dot op tests."""

  def __init__(self, *args, dot_fn):
    super().__init__(*args)
    self._dot_fn = dot_fn

    # Allow redefining the tolerance in subclasses.
    self.tol = dict(atol=5e-2)

    # Allow redefining the numerical comparison metric in subclasses.
    self.assert_close = lambda a, b, **tol: chex.assert_trees_all_close(
        a, b, **dict(self.tol, **tol)
    )

  def _create_inputs(self, num_groups, m, k, n, dtype, random_groups=False):
    rng = np.random.default_rng(sum(self._testMethodName.encode()))
    a = jnp.array(rng.standard_normal((m, k), np.float32), dtype)
    b = jnp.array(rng.standard_normal((num_groups, k, n), np.float32), dtype)
    if random_groups:
      group_sizes = rng.integers(0, m // num_groups, (num_groups,), np.int32)
      group_sizes = jnp.array(group_sizes)
    else:
      group_sizes = jnp.array([m // num_groups] * num_groups, jnp.uint32)
    return a, b, group_sizes

  @parameterized.parameters(jnp.bfloat16, jnp.float32)
  def test_simple(self, dtype):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(num_groups, m, k, n, dtype)
    actual = self._dot_fn(a, b, group_sizes=group_sizes)
    self.assert_close(actual, ref(a, b, group_sizes))

  def test_padded(self):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(
        num_groups, m, k, n, jnp.float32, random_groups=True
    )
    expected = ref(a, b, group_sizes)
    actual = self._dot_fn(a, b, group_sizes=group_sizes)
    count = sum(group_sizes)
    self.assert_close(actual[:count], expected[:count])

  @parameterized.product(
      dtype=("int8", "int4"),
      a_tile_shape=(None, (1, 128), (1, 16), (256, 1), (16, 1)),
      b_tile_shape=((1, 1, 16), (1, 1, 128), (1, 256, 1), (1, 16, 1)),
  )
  def test_quantized(self, dtype, a_tile_shape, b_tile_shape):
    dtype = jnp.dtype(dtype)
    num_groups, m, k, n = 8, 512, 256, 512
    a, b, group_sizes = self._create_inputs(
        num_groups, m, k, n, jnp.bfloat16, random_groups=True
    )

    if a_tile_shape is not None:
      a_quant = quantization.quantize_as(
          dtype, tile_shape=a_tile_shape, scale_dtype=a.dtype
      )(a)
      a = a_quant.recompose()
    else:
      a_quant = a

    b_quant = quantization.quantize_as(
        dtype, tile_shape=b_tile_shape, scale_dtype=b.dtype
    )(b)
    expected = ref(a, b_quant.recompose(), group_sizes)
    actual = self._dot_fn(a_quant, b_quant, group_sizes=group_sizes)
    count = sum(group_sizes)
    self.assert_close(actual[:count], expected[:count])

  @parameterized.parameters(None, jnp.bfloat16, jnp.float32)
  def test_preferred_element_type(self, out_type):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(num_groups, m, k, n, jnp.bfloat16)
    actual = self._dot_fn(
        a, b, group_sizes=group_sizes, preferred_element_type=out_type
    )
    expected = ref(a, b, group_sizes, preferred_element_type=out_type)
    self.assertEqual(actual.dtype, expected.dtype)
    self.assert_close(actual, expected)

  @parameterized.parameters((8, 1024, 128, 256), (8, 128, 64, 128))
  def test_vjp(self, num_groups, m, k, n):
    a, b, group_sizes = self._create_inputs(num_groups, m, k, n, jnp.float32)
    f = functools.partial(self._dot_fn, group_sizes=group_sizes)
    f_ref = functools.partial(ref, group_sizes=group_sizes)
    self.assert_close(f(a, b), f_ref(a, b))

    actual, f_vjp = jax.vjp(f, a, b)
    expected, f_ref_vjp = jax.vjp(f_ref, a, b)
    self.assert_close(actual, expected)

    dout = jax.nn.standardize(expected).astype(actual.dtype)
    self.assert_close(f_vjp(dout), f_ref_vjp(dout.astype(expected.dtype)))

  def test_group_sizes(self):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(num_groups, m, k, n, jnp.float32)
    expected = ref(a, b, group_sizes=group_sizes)
    group_sizes = base.GroupSizes(group_sizes, (1,) * num_groups)
    actual = self._dot_fn(a, b, group_sizes=group_sizes)  # pytype: disable=wrong-arg-types
    self.assert_close(actual, expected)

  @parameterized.named_parameters(NAMED_ARG_SPECS.items())
  def test_bench(self, spec):
    kwargs = numerics.random_initialize(spec)
    expected = ref(**kwargs)
    actual = self._dot_fn(**kwargs)
    count = sum(spec["group_sizes"].representative_value)
    self.assert_close(actual[:count], expected[:count], atol=0.05, rtol=0.05)


def base_names_and_params(test_name: str) -> list[tuple[str, str]]:
  return test_utils.get_names_and_params(RaggedDotTestBase, test_name)
