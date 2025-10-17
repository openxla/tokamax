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

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from tokamax._src import shape


class ShapeTest(absltest.TestCase):

  def test_upcast_broadcast(self):
    x = jnp.ones((3,), jnp.bfloat16)
    with shape.upcast_broadcast():
      y, vjp_fn = jax.vjp(lambda x: jnp.broadcast_to(x, (1024, 3)), x)
    dy = jax.random.normal(jax.random.PRNGKey(0), y.shape, jnp.bfloat16)
    chex.assert_trees_all_equal(vjp_fn(dy), (jnp.sum(dy, axis=0),))

  def test_contains_symbolic_shape(self):
    (a,) = jax.export.symbolic_shape('a')
    x1 = {'a': jax.ShapeDtypeStruct((2, 3, a), jnp.bfloat16), 'b': (2, 3)}
    x2 = {'a': jax.ShapeDtypeStruct((2, 3, 3), jnp.bfloat16), 'b': (2, 3)}
    self.assertTrue(shape.contains_symbolic_shape(x1))
    self.assertFalse(shape.contains_symbolic_shape(x2))


if __name__ == '__main__':
  absltest.main()
