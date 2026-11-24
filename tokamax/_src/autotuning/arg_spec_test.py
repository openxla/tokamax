# Copyright 2026 Google LLC
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
"""Tests for ArgSpec and mesh concretization."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec

P = jax.sharding.PartitionSpec


class ArgSpecTest(absltest.TestCase):

  def test_concretize_meshes(self):
    devices = jax.devices()
    mesh = jax.sharding.AbstractMesh((1,), ('d',))
    sharding = jax.sharding.NamedSharding(mesh, P('d'))
    args = {
        'mesh': mesh,
        'x': jax.ShapeDtypeStruct((4, 8), jnp.float32, sharding=sharding),
    }
    concrete = arg_spec.concretize_meshes(args, devices=devices)
    self.assertIsInstance(concrete['mesh'], jax.sharding.Mesh)
    self.assertEqual(concrete['x'].sharding.mesh, concrete['mesh'])

  def test_concretize_mesh_raises_when_insufficient_devices(self):
    mesh = jax.sharding.AbstractMesh((len(jax.devices()) + 1,), ('d',))
    with self.assertRaisesRegex(ValueError, 'requires'):
      arg_spec.concretize_mesh(mesh)


if __name__ == '__main__':
  absltest.main()
