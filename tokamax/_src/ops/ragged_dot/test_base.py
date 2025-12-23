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

import dataclasses
import functools
from unittest import mock

from absl.testing import parameterized
import chex
import jax
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np
import qwix
from tokamax._src import numerics
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import arg_specs
from tokamax._src.ops.ragged_dot import base
ARG_SPECS = arg_specs.ARG_SPECS


def _dot_fn_f32(dot_fn):
  """Wraps a dot_fn to ensure that the output is always f32."""

  def wrapped(lhs, rhs, group_sizes, **kwargs):
    if jnp.result_type(lhs, rhs) == jnp.float32:
      kwargs["precision"] = jax.lax.Precision.HIGHEST
    kwargs["preferred_element_type"] = jnp.float32
    return dot_fn(lhs, rhs, group_sizes=group_sizes, **kwargs)

  return wrapped


_jax_ragged_dot_f32 = _dot_fn_f32(jax.lax.ragged_dot)


def ref(lhs, rhs, group_sizes, activation=None):
  """Reference implementation of ragged dot."""
  lhs, rhs = map(quantization.as_array, (lhs, rhs))

  if lhs.dtype != rhs.dtype:
    input_dtype = jnp.result_type(lhs.dtype, rhs.dtype)
    lhs, rhs = lhs.astype(input_dtype), rhs.astype(input_dtype)

  result = _jax_ragged_dot_f32(lhs, rhs, group_sizes=jnp.asarray(group_sizes))
  if activation is not None:
    result = activation(result)
  return result


def override_chex_args(**kwargs):
  orig_assert_close = chex.assert_trees_all_close
  assert_close = lambda *a, **kw: orig_assert_close(*a, **(kw | kwargs))
  return mock.patch.object(chex, "assert_trees_all_close", assert_close)


NAMED_ARG_SPECS = {
    s.full_name: s.args for s in ARG_SPECS if "primary" in s.tags
}


# TODO: `jax.nn.relu` is annotated with `custom_jvp_call`
# which isn't compatible with `_estimate_resources` in the mosaic lowering.
# It would be nice in the future to support this, if possible.
def relu(x):
  return jnp.maximum(x, 0)


# pylint: disable=missing-function-docstring
class RaggedDotTestBase(parameterized.TestCase):
  """Base class for ragged dot op tests."""

  def __init__(self, *args, dot_fn):
    super().__init__(*args)
    self._dot_fn = dot_fn

  @property
  def _dot_fn_f32(self):
    return _dot_fn_f32(self._dot_fn)

  def _create_inputs(
      self, num_groups, m, k, n, dtype, random_groups=False, std=0.1
  ):
    rng = np.random.default_rng(sum(self._testMethodName.encode()))
    a = jnp.array(
        rng.normal(loc=0.0, scale=std, size=(m, k)).astype(np.float32), dtype
    )
    b = jnp.array(
        rng.normal(loc=0.0, scale=std, size=(num_groups, k, n)).astype(
            np.float32
        ),
        dtype,
    )
    if random_groups:
      group_sizes = rng.integers(0, m // num_groups, (num_groups,), np.int32)
      group_sizes = jnp.array(group_sizes)
    else:
      group_sizes = jnp.array([m // num_groups] * num_groups, jnp.uint32)
    return a, b, group_sizes

  def _test_simple(self, dtype):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(num_groups, m, k, n, dtype)
    actual = self._dot_fn_f32(a, b, group_sizes=group_sizes)
    chex.assert_trees_all_close(actual, ref(a, b, group_sizes))

  @parameterized.parameters(jnp.bfloat16, jnp.float32)
  def test_simple(self, dtype):
    self._test_simple(dtype)

  def test_padded(self):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(
        num_groups, m, k, n, jnp.bfloat16, random_groups=True
    )
    expected = ref(a, b, group_sizes)
    actual = self._dot_fn_f32(a, b, group_sizes=group_sizes)
    count = sum(group_sizes)
    chex.assert_trees_all_close(actual[:count], expected[:count])

  def test_zero_group_sizes(self):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(
        num_groups, m, k, n, jnp.bfloat16, random_groups=True
    )

    # Test all possible patterns of zero group sizes.
    for i in range(2**num_groups):
      group_sizes_ = jnp.where(jnp.unpackbits(jnp.uint8(i)), group_sizes, 0)
      with self.subTest(f"group_sizes={group_sizes_.tolist()}"):
        expected = ref(a, b, group_sizes_)
        actual = self._dot_fn_f32(a, b, group_sizes=group_sizes_)
        count = sum(group_sizes_)
        chex.assert_trees_all_close(actual[:count], expected[:count])

  @parameterized.product(
      dtype=("int8", "int4"),
      a_tile_shape=(None, (1, 128), (1, 16), (256, 1), (16, 1)),
      b_tile_shape=((1, 1, 16), (1, 1, 128), (1, 256, 1), (1, 16, 1)),
      use_as_qarray=(True, False),
      activation=(None, relu, jax.nn.tanh),
  )
  def test_quantized(
      self,
      dtype,
      a_tile_shape,
      b_tile_shape,
      use_as_qarray,
      activation,
  ):
    self._test_quantized(
        dtype, a_tile_shape, b_tile_shape, use_as_qarray, activation
    )

  def _test_quantized(
      self,
      dtype,
      a_tile_shape,
      b_tile_shape,
      use_as_qarray,
      activation,
  ):
    dtype = jnp.dtype(dtype)
    num_groups, m, k, n = 8, 512, 256, 512
    a, b, group_sizes = self._create_inputs(
        num_groups, m, k, n, jnp.bfloat16, random_groups=True
    )

    def quantize(x, tile_shape):
      tiled_axes = {i: d for i, d in enumerate(tile_shape)}
      if use_as_qarray:
        return quantization.AsQArray(x, dtype, tiled_axes=tiled_axes)
      return qwix.quantize(x, dtype, tiled_axes=tiled_axes)

    if a_tile_shape is not None:
      a = quantize(a, a_tile_shape)
    b = quantize(b, b_tile_shape)

    expected = ref(a, b, group_sizes, activation)
    # TODO: preferred_element_type to f32 and tighten tolerances.
    actual = self._dot_fn(
        a, b, group_sizes=group_sizes, activation=activation
    )
    count = sum(group_sizes)
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.01, rtol=0.005
    )

  @parameterized.parameters(None, jnp.bfloat16, jnp.float32)
  def test_preferred_element_type(self, out_type):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(num_groups, m, k, n, jnp.bfloat16)
    expected = ref(a, b, group_sizes)
    actual = self._dot_fn(
        a, b, group_sizes=group_sizes, preferred_element_type=out_type
    )
    self.assertEqual(actual.dtype, out_type or jnp.bfloat16)
    tol = dict(atol=0.01, rtol=0.005) if actual.dtype == jnp.bfloat16 else {}
    chex.assert_trees_all_close(actual, expected, **tol)

  @parameterized.product(
      num_groups=(8,),
      m=(1024, 128),
      k=(128, 64),
      n=(256, 128),
      activation=(None, relu, jax.nn.tanh),
  )
  def test_vjp(self, num_groups, m, k, n, activation=None):
    a, b, group_sizes = self._create_inputs(num_groups, m, k, n, jnp.bfloat16)
    a_ref = a.astype(jnp.float32)
    b_ref = b.astype(jnp.float32)
    f = functools.partial(
        self._dot_fn_f32,
        group_sizes=group_sizes,
        activation=activation,
        precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
    )
    f_ref = functools.partial(
        ref, group_sizes=group_sizes, activation=activation
    )
    chex.assert_trees_all_close(f(a, b), f_ref(a_ref, b_ref), atol=1e-5)

    actual, f_vjp = jax.vjp(f, a, b)
    expected, f_ref_vjp = jax.vjp(f_ref, a_ref, b_ref)
    chex.assert_trees_all_close(actual, expected, atol=1e-5)

    dout = jax.nn.standardize(expected)
    # NOTE: The reference doesn't correctly support `BF16_BF16_F32`, so we
    # explicitly downcast `dout` to bfloat16 to bottleneck the precision.
    dout = dout.astype(jnp.bfloat16).astype(actual.dtype)
    expected = f_ref_vjp(dout.astype(expected.dtype))
    chex.assert_trees_all_close(f_vjp(dout), expected, atol=0.02, rtol=0.005)

  def test_group_sizes(self):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(num_groups, m, k, n, jnp.bfloat16)
    expected = ref(a, b, group_sizes=group_sizes)
    group_sizes = base.GroupSizes(group_sizes, (1,) * num_groups)
    actual = self._dot_fn_f32(a, b, group_sizes=group_sizes)
    chex.assert_trees_all_close(actual, expected)

  @parameterized.parameters(((2, 3, -1, 4),), ((1022, 1, 1, 1),))
  def test_invalid_group_sizes(self, group_sizes):
    if not isinstance(self._dot_fn, base.RaggedDot):
      self.skipTest("Requires a bare `RaggedDot` implementation.")

    num_groups, m, k, n = 4, 1024, 128, 256
    a, b, _ = self._create_inputs(num_groups, m, k, n, jnp.bfloat16)
    fn = dataclasses.replace(self._dot_fn, checkify_group_sizes=True)
    fn = jax.jit(checkify.checkify(fn))
    with self.assertRaises(checkify.JaxRuntimeError):
      err, _ = fn(a, b, group_sizes=jnp.array(group_sizes, jnp.int32))
      err.throw()

  @parameterized.named_parameters(NAMED_ARG_SPECS.items())
  def test_bench(self, spec):
    self._test_bench(spec)

  def _test_bench(self, spec):
    kwargs = numerics.random_initialize(spec)
    # Skip the test if the reference implementation fails,
    # e.g out-of-memory.
    try:
      expected = ref(**kwargs)
    except Exception as e:  # pylint: disable=broad-exception-caught
      if "RESOURCE_EXHAUSTED: Out of memory while trying to allocate" in str(e):
        self.skipTest("Out of memory")
      raise
    actual = self._dot_fn(**kwargs)
    count = sum(spec["group_sizes"].representative_value)
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.01, rtol=0.005
    )
