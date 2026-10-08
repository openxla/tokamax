# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
from typing import override
from unittest import mock

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from tokamax._src.ops.normalization import pallas_triton_config
from tokamax._src.ops.normalization import test_base

try:
  from tokamax._src.ops.normalization import triton  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]
except ImportError:
  triton = None  # pyrefly: ignore[assignment]


@absltest.skipIf(triton is None, 'Requires jax_triton.')
class TritonNormalizationTest(test_base.NormalizationTestBase):

  def __init__(self, *args):
    norm_fn = triton.TritonNormalization() if triton is not None else None
    super().__init__(*args, norm_fn=norm_fn)  # pyrefly: ignore[bad-argument-type]

  def setUp(self):
    if jax.default_backend() == 'tpu':
      self.skipTest('Not supported on TPUs.')
    super().setUp()

  @override
  def _test_layer_norm_vmap(self, axis, vmap_in_axes):
    x_shape = [24, 32, 40]
    vmap_axis_sizes = tuple(
        x_shape.pop(in_axes[0]) for in_axes in vmap_in_axes[::-1]
    )

    seen_vmap_axis_sizes = []
    get_heuristics_config = pallas_triton_config.get_heuristics_config

    def my_heuristics_config(*args, **kwargs):
      seen_vmap_axis_sizes.append(kwargs['vmap_axis_sizes'])
      return get_heuristics_config(*args, **kwargs)

    with mock.patch.object(
        pallas_triton_config, 'get_heuristics_config', my_heuristics_config
    ):
      super()._test_layer_norm_vmap(axis, vmap_in_axes)

    # We expect to see a shape for non-vmapped and each layer of vmap.
    seen_vmap_axis_sizes = seen_vmap_axis_sizes[-1 :: -(len(vmap_in_axes) + 1)]
    # We expect three calls from fwd, fwd res, and VJP.
    self.assertEqual(seen_vmap_axis_sizes, [vmap_axis_sizes] * 3)

  def _assert_one_call_per_kernel(self, lowered):
    hlo = str(lowered.compiler_ir('stablehlo'))
    ffi_calls = [
        line for line in hlo.splitlines() if '@triton_kernel_call_ffi' in line
    ]
    self.assertLen(ffi_calls, 3, msg=hlo)

    def count(*patterns):
      return sum(all(p in c for p in patterns) for c in ffi_calls)

    fwd = 'TritonNormalization\\22'
    self.assertEqual(count(fwd, 'return_residuals\\22:false'), 1)
    self.assertEqual(count(fwd, 'return_residuals\\22:true'), 1)
    self.assertEqual(count('TritonNormalizationVjp\\22'), 1)

  def test_remat(self):
    rngs = list(jax.random.split(jax.random.PRNGKey(0), 4))

    shape = (128, 32)
    x = jax.random.normal(rngs.pop(), shape)
    scale = jax.random.uniform(rngs.pop(), (shape[-1],))
    offset = jax.random.uniform(rngs.pop(), (shape[-1],))
    epsilon = 1e-6

    f = functools.partial(self._norm_fn, epsilon=epsilon)
    g_ref = jax.value_and_grad(lambda *args: f(*args).sum())
    g_remat = jax.value_and_grad(lambda *args: jax.remat(f)(*args).sum())
    g_remat_lowered = jax.jit(g_remat).lower(x, scale, offset)
    self._assert_one_call_per_kernel(g_remat_lowered)

    g_out = g_remat_lowered.compile()(x, scale, offset)
    chex.assert_trees_all_equal(g_out, g_ref(x, scale, offset))

  def test_remat_with_vmap(self):
    rngs = list(jax.random.split(jax.random.PRNGKey(0), 4))

    shape = (3, 128, 32)
    x = jax.random.normal(rngs.pop(), shape)
    scale = jax.random.uniform(rngs.pop(), (shape[0], shape[-1]))
    offset = jax.random.uniform(rngs.pop(), (shape[0], shape[-1]))
    epsilon = 1e-6

    def f(x, scale, offset):
      return self._norm_fn(x, scale, offset, epsilon=epsilon)

    g_ref = jax.vmap(jax.value_and_grad(lambda *args: f(*args).sum()))
    g_remat = jax.vmap(
        jax.value_and_grad(lambda *args: jax.remat(f)(*args).sum())
    )
    g_remat_lowered = jax.jit(g_remat).lower(x, scale, offset)
    self._assert_one_call_per_kernel(g_remat_lowered)

    g_out = g_remat_lowered.compile()(x, scale, offset)
    chex.assert_trees_all_equal(g_out, g_ref(x, scale, offset))

  def test_vmap_large_batch(self):
    # More blocks than CUDA allows along grid axes 1 and 2.
    self._run_test(
        jax.ShapeDtypeStruct((70_000, 32), jnp.float32),
        jax.ShapeDtypeStruct((32,), jnp.float32),
        jax.ShapeDtypeStruct((32,), jnp.float32),
        vmap_in_axes=((0, None, None),),
    )


if __name__ == '__main__':
  absltest.main()
