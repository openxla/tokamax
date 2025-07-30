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
from tokamax._src import numerics
from tokamax._src import quantization
from tokamax._src import test_utils
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import bench_arg_specs


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

  return jax.lax.ragged_dot(
      lhs,
      rhs,
      group_sizes=jnp.asarray(group_sizes),
      precision=precision,
      preferred_element_type=preferred_element_type,
  )


# pylint: disable=missing-function-docstring
class RaggedDotTestBase(parameterized.TestCase):
  """Base class for ragged dot op tests."""

  def __init__(self, *args, dot_fn):
    super().__init__(*args)
    self._dot_fn = dot_fn

  @parameterized.parameters(jnp.bfloat16, jnp.float32)
  def test_simple(self, dtype):
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
    num_groups, m, k, n = 8, 1024, 128, 256
    a = jax.random.normal(rng0, (m, k), dtype=dtype)
    b = jax.random.normal(rng1, (num_groups, k, n), dtype=dtype)
    group_sizes = jnp.array([m // num_groups] * num_groups, jnp.uint32)

    actual = self._dot_fn(a, b, group_sizes=group_sizes)
    chex.assert_trees_all_close(actual, ref(a, b, group_sizes), atol=5e-2)

  def test_padded(self):
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(0), 3)
    num_groups, m, k, n = 8, 1024, 128, 256
    a = jax.random.normal(rng0, (m, k))
    b = jax.random.normal(rng1, (num_groups, k, n))
    max_group_size = m // num_groups
    group_sizes = jax.random.randint(
        rng2, (num_groups,), 0, max_group_size, dtype=jnp.uint32
    )

    expected = ref(a, b, group_sizes)
    actual = self._dot_fn(a, b, group_sizes=group_sizes)
    count = sum(group_sizes)
    chex.assert_trees_all_close(actual[:count], expected[:count], atol=5e-2)

  @parameterized.product(
      dtype=("int8", "int4"),
      a_tile_shape=(None, (1, 128), (1, 16), (256, 1), (16, 1)),
      b_tile_shape=((1, 1, 16), (1, 1, 128), (1, 256, 1), (1, 16, 1)),
  )
  def test_quantized(self, dtype, a_tile_shape, b_tile_shape):
    dtype = jnp.dtype(dtype)
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(0), 3)
    num_groups, m, k, n = 8, 512, 256, 512
    a = jax.random.normal(rng0, (m, k), dtype=jnp.bfloat16)
    b = jax.random.normal(rng1, (num_groups, k, n), dtype=jnp.bfloat16)
    max_group_size = m // num_groups
    group_sizes = jax.random.randint(
        rng2, (num_groups,), 0, max_group_size, dtype=jnp.uint32
    )

    if a_tile_shape is not None:
      a_quant = quantization.quantize_as(dtype, tile_shape=a_tile_shape)(a)
      a = a_quant.recompose()
    else:
      a_quant = a

    b_quant = quantization.quantize_as(dtype, tile_shape=b_tile_shape)(b)
    expected = ref(a, b_quant.recompose(), group_sizes)
    actual = self._dot_fn(a_quant, b_quant, group_sizes=group_sizes)
    count = sum(group_sizes)
    chex.assert_trees_all_close(actual[:count], expected[:count], atol=5e-2)

  @parameterized.parameters(None, jnp.bfloat16, jnp.float32)
  def test_preferred_element_type(self, out_type):
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
    num_groups, m, k, n = 8, 1024, 128, 256
    a = jax.random.normal(rng0, (m, k), dtype=jnp.bfloat16)
    b = jax.random.normal(rng1, (num_groups, k, n), dtype=jnp.bfloat16)
    group_sizes = jnp.array([m // num_groups] * num_groups, jnp.uint32)

    actual = self._dot_fn(
        a, b, group_sizes=group_sizes, preferred_element_type=out_type
    )
    expected = ref(a, b, group_sizes, preferred_element_type=out_type)
    self.assertEqual(actual.dtype, expected.dtype)
    chex.assert_trees_all_close(actual, expected, atol=5e-2)

  @parameterized.parameters((8, 1024, 128, 256), (8, 128, 64, 128))
  def test_vjp(self, num_groups, m, k, n):
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(0), 3)
    a = jax.random.normal(rng0, (m, k))
    b = jax.random.normal(rng1, (num_groups, k, n))
    group_sizes = jnp.array([m // num_groups] * num_groups, jnp.uint32)
    f = functools.partial(self._dot_fn, group_sizes=group_sizes)
    f_ref = functools.partial(ref, group_sizes=group_sizes)
    chex.assert_trees_all_close(f(a, b), f_ref(a, b), atol=5e-2)

    actual, f_vjp = jax.vjp(f, a, b)
    expected, f_ref_vjp = jax.vjp(f_ref, a, b)
    chex.assert_trees_all_close(actual, expected, atol=5e-2)

    dout = jax.random.normal(rng2, (m, n), dtype=expected.dtype)
    chex.assert_trees_all_close(f_vjp(dout), f_ref_vjp(dout), atol=5e-2)

  def test_group_sizes(self):
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
    num_groups, m, k, n = 8, 1024, 128, 256
    a = jax.random.normal(rng0, (m, k))
    b = jax.random.normal(rng1, (num_groups, k, n))
    group_sizes = jnp.array([m // num_groups] * num_groups, jnp.int32)
    expected = ref(a, b, group_sizes=group_sizes)
    group_sizes = base.GroupSizes(group_sizes, (1,) * num_groups)
    actual = self._dot_fn(a, b, group_sizes=group_sizes)  # pytype: disable=wrong-arg-types
    chex.assert_trees_all_close(actual, expected, atol=5e-2)

  @parameterized.named_parameters(bench_arg_specs.ARG_SPECS.items())
  def test_bench(self, spec):
    kwargs = numerics.random_initialize(spec)
    expected = ref(**kwargs)
    actual = self._dot_fn(**kwargs)
    count = sum(spec["group_sizes"].representative_value)
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.05, rtol=0.05
    )


def base_names_and_params(test_name: str) -> list[tuple[str, str]]:
  return test_utils.get_names_and_params(RaggedDotTestBase, test_name)
