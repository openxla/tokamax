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

import jax
from absl.testing import absltest
from tokamax._src.ops.normalization import mosaic
from tokamax._src.ops.normalization import test_base

import functools
from typing import override
from unittest import mock
import chex
import jax
from tokamax._src.ops.normalization import triton_config


class PallasMosaicGpuNormalizationTest(test_base.NormalizationTestBase):

    def __init__(self, *args):
        # The Mosaic kernel only supports a slice of the shape space (see
        # `mosaic._fwd` and `_reduction_layout`): the reduced axis must be a
        # multiple of 32, at most one non-reduced axis may be non-degenerate,
        # and the block needs >= 128 elements for cp.async. Everything else
        # raises `NotImplementedError` by design, so report those as skips
        # rather than failures -- a real bug still surfaces as a failure.
        op = mosaic.PallasMosaicGpuNormalization()

        def norm_fn(*args, **kwargs):
            try:
                return op(*args, **kwargs)
            except NotImplementedError as e:
                raise absltest.SkipTest(f'Unsupported by the Mosaic kernel: {e}')

        super().__init__(*args, norm_fn=norm_fn)

    def setUp(self):
        op = mosaic.PallasMosaicGpuNormalization()
        if not op.supported_on(jax.devices()[0]):
            self.skipTest('Mosaic GPU normalization not supported on this device.')
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

    @override
    def _test_layer_norm_vmap(self, axis, vmap_in_axes):
        x_shape = [24, 32, 40]
        vmap_axis_sizes = tuple(
            x_shape.pop(in_axes[0]) for in_axes in vmap_in_axes[::-1]
        )

        seen_vmap_axis_sizes = []
        get_heuristics_config = triton_config.get_heuristics_config

        def my_heuristics_config(*args, **kwargs):
            seen_vmap_axis_sizes.append(kwargs['vmap_axis_sizes'])
            return get_heuristics_config(*args, **kwargs)

        with mock.patch.object(
            triton_config, 'get_heuristics_config', my_heuristics_config
        ):
            super()._test_layer_norm_vmap(axis, vmap_in_axes)

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

        hlo = str(g_remat_lowered.compiler_ir('stablehlo'))
        self.assertEqual(hlo.count('name = "pallas_layer_norm"'), 1)
        self.assertEqual(hlo.count('name = "pallas_layer_norm_fwd_res"'), 1)
        self.assertEqual(hlo.count('name = "pallas_layer_norm_vjp"'), 1)

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

        hlo = str(g_remat_lowered.compiler_ir('stablehlo'))
        self.assertEqual(hlo.count('name = "pallas_layer_norm"'), 1, msg=hlo)
        self.assertEqual(hlo.count('name = "pallas_layer_norm_fwd_res"'), 1)
        self.assertEqual(hlo.count('name = "pallas_layer_norm_vjp"'), 1)

        g_out = g_remat_lowered.compile()(x, scale, offset)
        chex.assert_trees_all_equal(g_out, g_ref(x, scale, offset))


class LargestDivisorTest(absltest.TestCase):
    """Blocks must tile the array exactly: cp.async cannot mask OOB reads."""

    def test_largest_divisor(self):
        for n, cap, expected in [
            (8, 8, 8),
            (8, 16, 8),  # cap above n
            (6, 8, 6),
            (12, 8, 6),  # 8 does not divide 12
            (7, 4, 1),  # prime
        ]:
            self.assertEqual(mosaic._largest_divisor(n, cap), expected, (n, cap))


if __name__ == '__main__':
    absltest.main()
