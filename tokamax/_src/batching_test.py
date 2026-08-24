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

from absl.testing import absltest
import jax
from jax.experimental import hijax
import jax.numpy as jnp
import numpy as np
from tokamax._src import batching


def _bsd(shape, dtype, vmap_axes):
  shape = list(shape)
  vmap_axes = tuple(None if a is None else (a, shape.pop(a)) for a in vmap_axes)
  return batching.BatchedShapeDtype(tuple(shape), dtype, vmap_axes)


class _RecordVmapPrimitive(batching.ExpandWithVmapData):
  """Test primitive that records the captured vmap stack upon expansion."""

  def __init__(self, in_aval, recorder):
    self.in_avals = (in_aval,)
    self.out_aval = in_aval
    self.params = {}
    self._recorder = recorder
    super().__init__()

  def expand_with_vmap_data(self, vmap_data, *args):
    (x,) = args
    self._recorder.append(
        tuple(
            (
                axis_data.size,
                axis_data.name if isinstance(axis_data.name, str) else None,
                in_dims,
            )
            for axis_data, in_dims in vmap_data
        )
    )
    return x * 2.0

  def batch_dim_rule(self, axis_data, in_dims):
    return in_dims[0]

  def jvp(self, primals, tangents):
    return jax.jvp(self.expand, primals, tangents)

  vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp


class BatchingTest(absltest.TestCase):

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

  def test_vmap_stack_no_vmap(self):
    recorded = []
    x = jnp.ones((4, 8), jnp.float32)
    prim = _RecordVmapPrimitive(
        jax.core.ShapedArray((8,), jnp.float32), recorded
    )
    np.testing.assert_allclose(prim(x), x * 2.0)
    self.assertEqual(recorded, [()])

  def test_vmap_stack_single_vmap(self):
    recorded = []
    x = jnp.ones((4, 8), jnp.float32)
    prim = _RecordVmapPrimitive(
        jax.core.ShapedArray((8,), jnp.float32), recorded
    )
    out = jax.vmap(prim, in_axes=0)(x)
    np.testing.assert_allclose(out, x * 2.0)
    self.assertEqual(recorded, [((4, None, (0,)),)])

  def test_vmap_stack_nested_vmap(self):
    recorded = []
    x = jnp.ones((2, 3, 4), jnp.float32)
    prim = _RecordVmapPrimitive(
        jax.core.ShapedArray((4,), jnp.float32), recorded
    )
    out = jax.vmap(jax.vmap(prim, in_axes=0), in_axes=1)(x)
    self.assertEqual(out.shape, (3, 2, 4))
    np.testing.assert_allclose(out, jnp.transpose(x * 2.0, (1, 0, 2)))
    self.assertEqual(recorded, [((3, None, (1,)), (2, None, (0,)))])

  def test_vmap_stack_jvp(self):
    recorded = []
    x = jnp.ones((4, 8), jnp.float32)
    x_dot = jnp.ones((4, 8), jnp.float32)
    prim = _RecordVmapPrimitive(
        jax.core.ShapedArray((8,), jnp.float32), recorded
    )
    primals_out, tangents_out = jax.jvp(
        jax.vmap(prim, in_axes=0), (x,), (x_dot,)
    )
    np.testing.assert_allclose(primals_out, x * 2.0)
    np.testing.assert_allclose(tangents_out, x_dot * 2.0)
    self.assertEqual(recorded, [((4, None, (0,)),)])

  def test_vmap_stack_vjp(self):
    recorded = []
    x = jnp.ones((4, 8), jnp.float32)
    prim = _RecordVmapPrimitive(
        jax.core.ShapedArray((8,), jnp.float32), recorded
    )
    primals_out, vjp_fn = jax.vjp(jax.vmap(prim, in_axes=0), x)
    (grad,) = vjp_fn(jnp.ones((4, 8), jnp.float32))
    np.testing.assert_allclose(primals_out, x * 2.0)
    np.testing.assert_allclose(grad, jnp.full((4, 8), 2.0))
    self.assertEqual(recorded, [((4, None, (0,)),), ((4, None, (0,)),)])

  def test_vmap_stack_named_axes(self):
    recorded = []
    x = jnp.ones((4, 8), jnp.float32)
    prim = _RecordVmapPrimitive(
        jax.core.ShapedArray((8,), jnp.float32), recorded
    )
    out = jax.vmap(prim, in_axes=0, axis_name='i')(x)
    np.testing.assert_allclose(out, x * 2.0)
    self.assertEqual(recorded, [((4, 'i', (0,)),)])


if __name__ == '__main__':
  absltest.main()
