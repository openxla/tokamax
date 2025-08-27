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
      ret = a0 * a1 * kw * math.prod(batched_args.vmap_axis_sizes)
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
        batching.BatchedShapeDtype(a0.shape, a0.dtype, a0_axes),
        batching.BatchedShapeDtype(a1.shape, a1.dtype, a1_axes),
    )
    expected_batched_kwargs = dict(
        kw=batching.BatchedShapeDtype(kw.shape, kw.dtype, (0,) * num_vmaps)
    )
    self.assertEqual(seen_batched_args[-1], expected_batched_args)
    self.assertEqual(seen_batched_kwargs[-1], expected_batched_kwargs)

  def test_batched_shape_dtype_equality(self):
    x = jax.ShapeDtypeStruct((3, 4), jnp.float32)
    x_batched = batching.BatchedShapeDtype((3, 4), jnp.float32, vmap_axes=())
    x_batched_vmap = batching.BatchedShapeDtype(
        (3, 4), jnp.float32, vmap_axes=(0,)
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
