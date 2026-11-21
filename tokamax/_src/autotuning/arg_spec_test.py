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
"""Tests for ArgSpec, ShardMapSpec, and mesh concretization."""

from typing import Any, ClassVar

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import benchmarking
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops import op as op_lib

P = jax.sharding.PartitionSpec


class _ShardMapOp(op_lib.Op[Any, jax.Array, None, op_lib.NullConfig, Any]):
  """Simple op that expects to run inside `jax.shard_map`."""

  config_cls: ClassVar[type[op_lib.NullConfig]] = op_lib.NullConfig

  def _fwd(
      self,
      x: jax.Array,
      w: jax.Array,
      *,
      scale: float = 1.0,
      axis_name: str = 'd',
      return_residuals: bool = False,
      config: op_lib.NullConfig,
  ) -> tuple[jax.Array, None]:
    del return_residuals, config
    idx = jax.lax.axis_index(axis_name)
    return (x + w) * scale + idx, None


class ArgSpecTest(absltest.TestCase):

  def test_num_devices_from_shard_map(self):
    mesh = jax.sharding.AbstractMesh((8,), ('d',))
    spec = arg_spec.ArgSpec(
        args={'x': jax.ShapeDtypeStruct((4, 8), jnp.float32)},
        shard_map=arg_spec.ShardMapSpec(mesh=mesh, in_specs={'x': P('d')}),
    )
    self.assertEqual(spec.num_devices, 8)

  def test_concretize_meshes(self):
    devices = jax.devices()
    mesh = jax.sharding.AbstractMesh((1,), ('d',))
    sharding = jax.sharding.NamedSharding(mesh, P('d'))
    args = {
        'mesh': mesh,
        'x': jax.ShapeDtypeStruct((4, 8), jnp.float32, sharding=sharding),
        'idx': numerics.RangedArrayInitializer(
            (4,), jnp.int32, 0, 10, sharding=sharding
        ),
    }
    concrete = arg_spec.concretize_meshes(args, devices=devices)
    self.assertIsInstance(concrete['mesh'], jax.sharding.Mesh)
    self.assertEqual(concrete['x'].sharding.mesh, concrete['mesh'])
    self.assertEqual(concrete['idx'].sharding.mesh, concrete['mesh'])

  def test_lift_shard_map_args_scales_sharded_dims(self):
    devices = [jax.devices()[0]] * 2
    mesh = jax.sharding.AbstractMesh((2,), ('d',))
    sm = arg_spec.ShardMapSpec(
        mesh=mesh,
        in_specs={'x': P('d'), 'w': P(), 'ids': P('d')},
        out_specs=P('d'),
    )
    local_args = {
        'x': jax.ShapeDtypeStruct((4, 8), jnp.float32),
        'w': jax.ShapeDtypeStruct((4, 8), jnp.float32),
        'ids': numerics.RangedArrayInitializer((4, 2), jnp.int32, 0, 16),
        'scale': 2.0,
    }
    lifted = arg_spec.lift_shard_map_args(local_args, sm, devices=devices)
    self.assertEqual(lifted['x'].shape, (8, 8))
    self.assertEqual(lifted['x'].sharding.spec, P('d'))
    self.assertEqual(lifted['w'].shape, (4, 8))
    self.assertEqual(lifted['w'].sharding.spec, P())
    self.assertEqual(lifted['ids'].shape, (8, 2))
    self.assertEqual(lifted['ids'].sharding.spec, P('d'))
    self.assertEqual(lifted['scale'], 2.0)

  def test_wrap_shard_map_executes_and_standardizes(self):
    mesh = jax.sharding.AbstractMesh((1,), ('d',))
    sm = arg_spec.ShardMapSpec(
        mesh=mesh,
        in_specs={'x': P('d'), 'w': P()},
        out_specs=P('d'),
        check_vma=False,
    )
    local_args = {
        'x': jax.ShapeDtypeStruct((4, 8), jnp.float32),
        'w': jax.ShapeDtypeStruct((4, 8), jnp.float32),
        'scale': 2.0,
        'axis_name': 'd',
    }
    op = _ShardMapOp()
    lifted = arg_spec.lift_shard_map_args(local_args, sm)
    wrapped = arg_spec.wrap_shard_map(op, sm)
    fn, arrays = benchmarking.standardize_function(wrapped, kwargs=lifted)
    out = jax.jit(fn)(arrays)
    self.assertEqual(out.shape, (4, 8))
    np.testing.assert_allclose(
        np.asarray(out), (np.asarray(arrays[0]) + np.asarray(arrays[1])) * 2.0
    )


if __name__ == '__main__':
  absltest.main()
