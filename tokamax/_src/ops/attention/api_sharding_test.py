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
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp

from tokamax._src import gpu_utils
from tokamax._src.ops.attention import api

Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
PartitionSpec = jax.sharding.PartitionSpec


class ApiShardingTest(parameterized.TestCase):

  @parameterized.parameters(*api.IMPLEMENTATIONS)
  def test_dot_product_attention_sharding(self, implementation):
    if implementation == 'mosaic' and not gpu_utils.has_mosaic_gpu_support():
      self.skipTest('Mosaic not supported on this platform.')
    if implementation == 'triton' and not gpu_utils.has_triton_support():
      self.skipTest('Triton not supported on this platform.')
    if implementation == 'cudnn' and jax.default_backend() != 'gpu':
      self.skipTest('cuDNN only supported on GPU')

    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices for sharding.')

    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names='x')
    sharding = NamedSharding(mesh, PartitionSpec('x'))

    dtype = jnp.bfloat16
    b, s, t, n, h = jax.device_count(), 128, 128, 4, 64
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(keys[0], (b, t, n, h), dtype)
    k = jax.random.normal(keys[1], (b, s, n, h), dtype)
    v = jax.random.normal(keys[2], (b, s, n, h), dtype)

    sdpa_ref = functools.partial(
        jax.nn.dot_product_attention, implementation='xla'
    )
    sdpa_ans = functools.partial(
        api.dot_product_attention, implementation=implementation
    )

    out_ref = sdpa_ref(q, k, v)

    @functools.partial(
        jax.jit,
        in_shardings=(sharding, sharding, sharding),
        out_shardings=sharding,
    )
    def f(q, k, v):
      return sdpa_ans(q, k, v, q_sharding=sharding)

    q_sharded = jax.device_put(q, sharding)
    k_sharded = jax.device_put(k, sharding)
    v_sharded = jax.device_put(v, sharding)
    out_ans = f(q_sharded, k_sharded, v_sharded)
    self.assertEqual(out_ans.sharding, sharding)
    chex.assert_trees_all_close(out_ans, out_ref, atol=0.01, rtol=0.01)


if __name__ == '__main__':
  absltest.main()
