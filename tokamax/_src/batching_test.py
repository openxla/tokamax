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
"""Batching utility tests."""

import math

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import batching


def _bsd(shape, dtype, vmap_axes):
  shape = list(shape)
  vmap_axes = tuple(None if a is None else (a, shape.pop(a)) for a in vmap_axes)
  return batching.BatchedShapeDtype(tuple(shape), dtype, vmap_axes)


class BatchingTest(parameterized.TestCase):

  @parameterized.parameters(
      (((0, 0),),),
      (((0, None),),),
      (((0, 0), (0, 0)),),
      (((0, 0), (None, 0)),),
      (((None, 0), (None, 0)),),
  )
  def test_capture_batched_args(self, in_axes):
    seen_inner_shapes = []
    seen_batched_args = []
    seen_batched_kwargs = []

    @batching.capture_batched_args
    def f(a0: jax.Array, a1: jax.Array, *, kw: jax.Array, batched_args):
      vmap_axis_sizes = batching.get_vmap_axis_sizes(batched_args.arguments)
      ret = a0 * a1 * kw * math.prod(vmap_axis_sizes)
      seen_inner_shapes.append((a0.shape, a1.shape, kw.shape))
      seen_batched_args.append(batched_args.args)
      seen_batched_kwargs.append(batched_args.kwargs)
      return ret

    shape = [2, 3, 4]
    a0_shape = shape[:]
    a1_shape = shape[:]

    for i, (a0_axis, a1_axis) in enumerate(in_axes[::-1]):
      if a0_axis is None:
        del a0_shape[len(in_axes) - i - 1]
      if a1_axis is None:
        del a1_shape[len(in_axes) - i - 1]
      f = jax.vmap(f, in_axes=(a0_axis, a1_axis))

    def _init_arr(shape, offs):
      return jnp.arange(np.prod(shape)).reshape(shape) + offs

    a0 = _init_arr(a0_shape, 0)
    a1 = _init_arr(a1_shape, 42)
    kw = _init_arr(shape, 100)

    ref_a0 = a0.reshape([s if s in a0_shape else 1 for s in shape])
    ref_a1 = a1.reshape([s if s in a1_shape else 1 for s in shape])
    num_vmaps = len(in_axes) if in_axes is not None else 0

    ref = ref_a0 * ref_a1 * kw * math.prod(shape[:num_vmaps])
    chex.assert_trees_all_equal(ref, f(a0, a1, kw=kw))

    expected_a0_inner_shape = a0_shape[:]
    expected_a1_inner_shape = a1_shape[:]
    expected_kw_inner_shape = shape[num_vmaps:]
    for a0_axis, a1_axis in in_axes:
      if a0_axis is not None:
        del expected_a0_inner_shape[a0_axis]
      if a1_axis is not None:
        del expected_a1_inner_shape[a1_axis]

    expected_shapes = (
        tuple(expected_a0_inner_shape),
        tuple(expected_a1_inner_shape),
        tuple(expected_kw_inner_shape),
    )
    self.assertEqual(seen_inner_shapes, [expected_shapes] * (num_vmaps + 1))

    a0_axes, a1_axes = zip(*in_axes)
    expected_batched_args = (
        _bsd(a0.shape, a0.dtype, a0_axes), _bsd(a1_shape, a1.dtype, a1_axes)
    )
    expected_batched_kwargs = dict(
        kw=_bsd(kw.shape, kw.dtype, (0,) * num_vmaps)
    )
    self.assertEqual(seen_batched_args[-1], expected_batched_args)
    self.assertEqual(seen_batched_kwargs[-1], expected_batched_kwargs)

    seen_inner_shapes.clear()
    seen_batched_args.clear()
    seen_batched_kwargs.clear()

    _ = jax.vjp(lambda a0, a1: f(a0, a1, kw=kw), a0, a1)
    self.assertEqual(seen_inner_shapes, [expected_shapes] * (num_vmaps + 1))
    self.assertEqual(seen_batched_args[-1], expected_batched_args)
    self.assertEqual(seen_batched_kwargs[-1], expected_batched_kwargs)

  def test_capture_batched_args_squeezes_unit_axes(self):
    """Checks that the wrapper squeezes a vmap axis of size 1.

    Some primitives have no general batching rule. For example,
    `ragged_dot_general` requires every operand to be batched at dim 0. Under
    `vmap(grad(...))` the cotangent is not batched. Under `shard_map` with
    `spmd_axis_name`, each device can hold one replica. The wrapper squeezes a
    vmap axis of size 1 and applies the function once, so such primitives do
    not see a batch axis. The inner `custom_vmap` rule below raises if JAX
    batches the function, so the test detects any attempt to batch it.
    """

    @jax.custom_batching.custom_vmap
    def inner(a, b):
      return a * b

    @inner.def_vmap
    def _(axis_size, in_batched, a, b):
      del axis_size, in_batched, a, b  # Unused.
      raise AssertionError('inner vmap rule must not run for a unit axis')

    seen_batched_args = []

    @batching.capture_batched_args
    def f(a, b, *, batched_args):
      seen_batched_args.append(batched_args.args)
      return inner(a, b)

    a = jnp.arange(12).reshape(3, 4)
    b = jnp.flip(a)
    expected = a * b

    def expected_batched_args(vmap_axes):
      return tuple(
          batching.BatchedShapeDtype(x.shape, x.dtype, vmap_axes)
          for x in (a, b)
      )

    with self.subTest('unit_axis'):
      out = jax.vmap(f)(a[None], b[None])
      chex.assert_trees_all_equal(out, expected[None])
      self.assertEqual(seen_batched_args[-1], expected_batched_args(((0, 1),)))

    with self.subTest('unit_axis_with_unbatched_arg'):
      # This is the `vmap(grad(...))` case: the vmap axis runs over the
      # residuals, but not over the cotangent. The squeeze applies to the
      # batched arguments only.
      out = jax.vmap(f, in_axes=(0, None))(a[None], b)
      chex.assert_trees_all_equal(out, expected[None])
      self.assertEqual(
          seen_batched_args[-1],
          (
              batching.BatchedShapeDtype(a.shape, a.dtype, ((0, 1),)),
              batching.BatchedShapeDtype(b.shape, b.dtype, (None,)),
          ),
      )

    with self.subTest('axis_of_size_2_still_batches'):
      with self.assertRaisesRegex(
          AssertionError, 'inner vmap rule must not run for a unit axis'
      ):
        jax.vmap(f)(jnp.stack([a, b]), jnp.stack([b, a]))

    with self.subTest('nested_unit_axes'):
      out = jax.vmap(jax.vmap(f))(a[None, None], b[None, None])
      chex.assert_trees_all_equal(out, expected[None, None])
      self.assertEqual(
          seen_batched_args[-1], expected_batched_args(((0, 1), (0, 1)))
      )

    with self.subTest('nested_axis_of_size_2_still_batches'):
      # A unit axis at either nesting level does not stop the batching of the
      # size-2 axis.
      stacked = (jnp.stack([a, b]), jnp.stack([b, a]))
      for unit_axis in (0, 1):
        args = tuple(jnp.expand_dims(x, unit_axis) for x in stacked)
        with self.assertRaisesRegex(
            AssertionError, 'inner vmap rule must not run for a unit axis'
        ):
          jax.vmap(jax.vmap(f))(*args)

  def test_batched_shape_dtype_equality(self):
    x = jax.ShapeDtypeStruct((3, 4), jnp.float32)
    x_batched = batching.BatchedShapeDtype((3, 4), jnp.float32, vmap_axes=())
    x_batched_vmap = batching.BatchedShapeDtype(
        (4,), jnp.float32, vmap_axes=((0, 3),)
    )

    self.assertEqual(x_batched, x_batched)
    self.assertEqual(x_batched_vmap, x_batched_vmap)
    self.assertEqual(x_batched, x)
    self.assertEqual(x, x_batched)
    self.assertNotEqual(x_batched, x_batched_vmap)
    self.assertNotEqual(x_batched_vmap, x_batched)
    self.assertNotEqual(x, x_batched_vmap)
    self.assertNotEqual(x_batched_vmap, x)


if __name__ == '__main__':
  absltest.main()
