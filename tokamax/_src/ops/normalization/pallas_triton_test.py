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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from tokamax._src.ops.normalization import pallas_triton
from tokamax._src.ops.normalization import pallas_triton_config
from tokamax._src.ops.normalization import test_base


class PallasTritonNormalizationTest(test_base.NormalizationTestBase):

  def __init__(self, *args):
    super().__init__(*args, norm_fn=pallas_triton.PallasTritonNormalization())

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  def test_layer_norm_with_pre_scale(self):
    rngs = list(jax.random.split(jax.random.PRNGKey(0), 4))

    shape = (128, 32)
    x = jax.random.normal(rngs.pop(), shape)
    scale = jax.random.uniform(rngs.pop(), (shape[-1],))
    offset = jax.random.uniform(rngs.pop(), (shape[-1],))
    pre_scale = jax.random.uniform(rngs.pop(), (shape[-1],))
    epsilon = 1e-6

    y_expected = jax.nn.standardize(x * pre_scale, epsilon=epsilon) * scale
    y_expected += offset
    y_actual = self._norm_fn(
        lambda: x * pre_scale, scale, offset, epsilon=epsilon
    )
    chex.assert_trees_all_close(y_actual, y_expected, atol=1e-6)

  @parameterized.parameters(
      *test_base.base_names_and_params('test_layer_norm_vmap')
  )
  def test_layer_norm_vmap(self, test_name, kwargs):
    kwargs = eval(f'dict{kwargs}')  # pylint: disable=eval-used
    vmap_in_axes = kwargs['vmap_in_axes']

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
      getattr(super(), test_name)()

    # We expect to see a shape for non-vmapped and each layer of vmap.
    seen_vmap_axis_sizes = seen_vmap_axis_sizes[-1 :: -(len(vmap_in_axes) + 1)]
    # We expect three calls from fwd, fwd res, and VJP.
    self.assertEqual(seen_vmap_axis_sizes, [vmap_axis_sizes] * 3)

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
    # TODO: Re-enable checks after bug is fixed.
    _ = """
    hlo = str(g_remat_lowered.compiler_ir('stablehlo'))
    self.assertEqual(hlo.count('name = "pallas_layer_norm"'), 1)
    self.assertEqual(hlo.count('name = "pallas_layer_norm_fwd_res"'), 1)
    self.assertEqual(hlo.count('name = "pallas_layer_norm_vjp"'), 1)
    """

    g_out = g_remat_lowered.compile()(x, scale, offset)
    chex.assert_trees_all_equal(g_out, g_ref(x, scale, offset))

  def test_remat_with_vmap(self):
    rngs = list(jax.random.split(jax.random.PRNGKey(0), 4))

    shape = (3, 128, 32)
    x = jax.random.normal(rngs.pop(), shape)
    scale = jax.random.uniform(
        rngs.pop(),
        (
            shape[0],
            shape[-1],
        ),
    )
    offset = jax.random.uniform(
        rngs.pop(),
        (
            shape[0],
            shape[-1],
        ),
    )
    epsilon = 1e-6

    def f(x, scale, offset):
      return self._norm_fn(x, scale, offset, epsilon=epsilon)

    g_ref = jax.vmap(jax.value_and_grad(lambda *args: f(*args).sum()))
    g_remat = jax.vmap(
        jax.value_and_grad(lambda *args: jax.remat(f)(*args).sum())
    )
    g_remat_lowered = jax.jit(g_remat).lower(x, scale, offset)
    # TODO: Re-enable checks after bug is fixed.
    _ = """
    hlo = str(g_remat_lowered.compiler_ir('stablehlo'))
    self.assertEqual(
        hlo.count('name = "pallas_layer_norm_batched"'), 1, msg=hlo
    )
    self.assertEqual(hlo.count('name = "pallas_layer_norm_fwd_res_batched"'), 1)
    self.assertEqual(hlo.count('name = "pallas_layer_norm_vjp_batched"'), 1)
    """

    g_out = g_remat_lowered.compile()(x, scale, offset)
    chex.assert_trees_all_equal(g_out, g_ref(x, scale, offset))


if __name__ == '__main__':
  absltest.main()
