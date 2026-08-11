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

import contextlib
import functools
from typing import override
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import config as config_lib
from tokamax._src.ops.normalization import mosaic
from tokamax._src.ops.normalization import mosaic_tiling
from tokamax._src.ops.normalization import mosaic_vjp
from tokamax._src.ops.normalization import pallas_triton_config as triton_config
from tokamax._src.ops.normalization import pallas_triton_vjp_config as vjp_config
from tokamax._src.ops.normalization import test_base


@contextlib.contextmanager
def _skip_if_unsupported():
    """Reports the kernel's by-design `NotImplementedError`s as skips.

    A real bug still surfaces as a failure. This has to wrap the lowering call
    as well as the op call: `jax.remat` re-traces its function while `lower`
    runs, which is outside the dynamic extent of `norm_fn`.
    """
    try:
        yield
    except NotImplementedError as e:
        raise absltest.SkipTest(f'Unsupported by the Mosaic kernel: {e}')


class PallasMosaicGpuNormalizationTest(test_base.NormalizationTestBase):

    def __init__(self, *args):
        # The Mosaic kernel only supports a slice of the shape space (see
        # `mosaic_tiling.plan`): the reduced axis must be a multiple of 32, at
        # most one non-reduced axis may be non-degenerate, and the blocks have
        # to tile the array exactly. Everything else raises
        # `NotImplementedError` by design, so report those as skips rather than
        # failures -- a real bug still surfaces as a failure.
        op = mosaic.PallasMosaicGpuNormalization()

        def norm_fn(*args, **kwargs):
            with _skip_if_unsupported():
                return op(*args, **kwargs)

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

    @parameterized.named_parameters(('_plain', False), ('_under_vmap', True))
    def test_grad_gradients(self, under_vmap):
        """Gradients against plain JAX, with and without an enclosing `vmap`.

        The pair is deliberate: it separates a wrong kernel from wrong batching.
        `test_base` only ever takes `jax.vjp` of an already-vmapped function, so
        `scale` is shared and `dscale` is summed over the batch; the `_under_vmap`
        case is the other order, where each element owns its gradient. Configs are
        explicit so which case of `mosaic_tiling.plan` runs does not depend on the
        device's core count.
        """
        rows = (4, 4096) if under_vmap else (4096,)
        shape = (*rows, 128)
        rngs = list(jax.random.split(jax.random.PRNGKey(0), 3))
        x = jax.random.normal(rngs.pop(), shape)
        scale = jax.random.uniform(rngs.pop(), (shape[-1],))
        offset = jax.random.uniform(rngs.pop(), (shape[-1],))
        epsilon = 1e-6

        config = dict(block_m=32, block_n=None, num_warps=4)
        op = mosaic.PallasMosaicGpuNormalization(
            config=triton_config.Config(**config),
            vjp=mosaic_vjp.PallasMosaicGpuNormalizationVjp(
                config=vjp_config.Config(**config)
            ),
        )
        ref = lambda x, s, o: jax.nn.standardize(x, epsilon=epsilon) * s + o

        def grads(fn):
            loss = lambda x, s, o: jnp.sum(fn(x, s, o))
            f = jax.grad(loss, argnums=(0, 1, 2))
            if under_vmap:
                f = jax.vmap(f, in_axes=(0, None, None))
            return f(x, scale, offset)

        with _skip_if_unsupported():
            actual = grads(lambda x, s, o: op(x, s, o, epsilon=epsilon))
        dx, dscale, doffset = grads(ref)
        chex.assert_trees_all_close(actual[0], dx, atol=1e-5)
        # Each is a sum over 4096 rows, so a looser bound than `dx` needs.
        chex.assert_trees_all_close(actual[1], dscale, atol=1e-3)
        chex.assert_trees_all_close(actual[2], doffset, atol=1e-3)

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
        # `plgpu.kernel` takes no `name`, so unlike `pallas_triton_test` there is
        # nothing to count in the HLO; the numbers below are the real check.
        with _skip_if_unsupported():
            g_remat_lowered = jax.jit(g_remat).lower(x, scale, offset)
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
        with _skip_if_unsupported():
            g_remat_lowered = jax.jit(g_remat).lower(x, scale, offset)
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
            got = mosaic_tiling.largest_divisor(n, cap)
            self.assertEqual(got, expected, (n, cap))

    def test_warp_aligned_block(self):
        for rows, cap, expected in [
            (256, 32, 32),
            (12, 8, 4),  # 6 divides 12 and is under the cap, but is not a warp
            (12, 32, 12),  # cap above rows
            (4096, 4, 4),
            (2, 32, None),  # 2 rows cannot fill four warps
            (7, 32, None),  # odd
        ]:
            got = mosaic_tiling.warp_aligned_block(rows, cap)
            self.assertEqual(got, expected, (rows, cap))

    def test_with_usable_block_m_floors_at_four(self):
        """The Triton heuristic bottoms out at 1, which cannot fill the warps."""
        for block_m, expected in [(1, 4), (2, 4), (4, 4), (32, 32)]:
            config = triton_config.Config(
                block_m=block_m, block_n=None, num_warps=4
            )
            got = mosaic_tiling.with_usable_block_m(config)
            self.assertEqual(got.block_m, expected)
            # Everything else is left alone.
            self.assertEqual((got.block_n, got.num_warps), (None, 4))


class AmpereLoweringTest(parameterized.TestCase):
    """Lowers the kernels for Ampere, on whatever machine is running the test.

    `--tokamax_cross_compile` drops the hardware checks and `lowering_platforms`
    picks the target, so no GPU is needed: `.lower()` runs the whole Mosaic
    pipeline -- layout inference, the SMEM scratch for a cross-warp reduction,
    NVVM -- which is where a bad layout shows up. Only `.compile()`, i.e. PTX and
    the register allocator, needs the real device.

    Configs are explicit because the heuristics ask the local device for its core
    count, and because the block sizes are what select the case in
    `mosaic_tiling.plan`.
    """

    def _op(self, *, block_m, block_n):
        config = dict(block_m=block_m, block_n=block_n, num_warps=4)
        return mosaic.PallasMosaicGpuNormalization(
            config=triton_config.Config(**config),
            vjp=mosaic_vjp.PallasMosaicGpuNormalizationVjp(
                config=vjp_config.Config(**config)
            ),
        )

    def _lower(self, shape, *, axis, block_m, block_n, use_params, subtract_mean):
        op = self._op(block_m=block_m, block_n=block_n)
        x = jnp.zeros(shape, jnp.float32)
        params = jnp.zeros((shape[axis],), jnp.float32) if use_params else None

        def loss(x, scale, offset):
            y = op(x, scale, offset, axis=axis, subtract_mean=subtract_mean)
            return jnp.sum(y)

        # Take the gradient so that the VJP kernel lowers too.
        argnums = (0, 1, 2) if use_params else (0,)
        f = jax.jit(jax.grad(loss, argnums=argnums))
        with config_lib.cross_compile(True):
            return f.trace(x, params, params).lower(
                lowering_platforms=('cuda',)
            ).as_text()

    @parameterized.product(
        (
            # One case per branch of `mosaic_tiling.plan`.
            dict(shape=(4096, 128), axis=-1, block_m=32, block_n=None),
            dict(shape=(8, 128, 256), axis=1, block_m=32, block_n=32),
            # 6 rows: no multiple of 4 divides them, so `A` supplies the warps.
            dict(shape=(6, 256), axis=-1, block_m=32, block_n=None),
        ),
        use_params=(False, True),
        subtract_mean=(False, True),
    )
    def test_lowers(self, **kwargs):
        hlo = self._lower(**kwargs)
        # The forward and the VJP, and nothing that fell back to XLA.
        self.assertEqual(hlo.count('custom_call @mosaic_gpu_v2'), 2, msg=hlo)

    @parameterized.named_parameters(
        # `dscale` per batch element: `scale` is shared, but each element's
        # gradient is asked for separately.
        ('_vmap_of_grad', True, ((4, 4096, 128), (4, 128), (4, 128))),
        # `dscale` summed over the batch: one gradient for the shared `scale`.
        ('_grad_of_vmap', False, ((4, 4096, 128), (128,), (128,))),
    )
    def test_vmap_lowers_with_the_gradients_the_caller_asked_for(
        self, grad_inside, expected
    ):
        """Both `vmap` orders, which disagree about the shape of `dscale`.

        Folding the batch into `M` can only ever produce the summed form, so
        under `vmap` the VJP leaves batching to JAX -- see
        `mosaic.PallasMosaicGpuNormalization._rule`. Getting this wrong is not a
        crash: the fold
        returns a summed gradient where a per-element one belongs, which JAX
        broadcasts, so every element silently gets the batch's total.
        """
        op = self._op(block_m=32, block_n=None)
        x = jnp.zeros((4, 4096, 128), jnp.float32)
        params = jnp.zeros((128,), jnp.float32)
        loss = lambda x, scale, offset: jnp.sum(op(x, scale, offset))
        if grad_inside:
            f = jax.vmap(jax.grad(loss, argnums=(0, 1, 2)), in_axes=(0, None, None))
        else:
            batched = lambda x, s, o: jnp.sum(jax.vmap(lambda xi: op(xi, s, o))(x))
            f = jax.grad(batched, argnums=(0, 1, 2))

        with config_lib.cross_compile(True):
            hlo = str(
                jax.jit(f)
                .trace(x, params, params)
                .lower(lowering_platforms=('cuda',))
                .as_text()
            )
            grads = jax.eval_shape(f, x, params, params)

        self.assertEqual(hlo.count('custom_call @mosaic_gpu_v2'), 2, msg=hlo)
        shapes = tuple(g.shape for g in jax.tree.leaves(grads))
        self.assertEqual(shapes, expected)


    def test_batched_params_lower(self):
        """`vmap` may give each element its own `scale`/`offset`.

        The kernel then reads the row's own element rather than a shared param,
        which is why blocks must not straddle a batch boundary. Both directions
        have to handle it: a forward that folds and a VJP that raised would push
        the failure to grad time.
        """
        op = self._op(block_m=32, block_n=None)
        x = jnp.zeros((3, 128, 32), jnp.float32)
        params = jnp.zeros((3, 32), jnp.float32)
        loss = lambda x, scale, offset: jnp.sum(op(x, scale, offset))
        f = jax.vmap(jax.grad(loss, argnums=(0, 1, 2)))

        with config_lib.cross_compile(True):
            hlo = str(
                jax.jit(f)
                .trace(x, params, params)
                .lower(lowering_platforms=('cuda',))
                .as_text()
            )
            grads = jax.eval_shape(f, x, params, params)

        self.assertEqual(hlo.count('custom_call @mosaic_gpu_v2'), 2, msg=hlo)
        shapes = tuple(g.shape for g in jax.tree.leaves(grads))
        self.assertEqual(shapes, ((3, 128, 32), (3, 32), (3, 32)))


class PlanTest(absltest.TestCase):
    """`mosaic_tiling.plan` is pure Python, so it is testable without a GPU."""

    F32 = 4

    def test_degenerate_1d_is_declined(self):
        """A single row leaves a statistic Mosaic cannot store. See `plan`."""
        with self.assertRaisesRegex(NotImplementedError, 'single-element'):
            mosaic_tiling.plan((1, 128, 1), self.F32, block_m=32, block_b=1)

    def test_reduced_axis_needs_multiple_of_128_for_the_row_layout(self):
        with self.assertRaisesRegex(NotImplementedError, 'multiple of 128'):
            mosaic_tiling.plan((6, 96, 1), self.F32, block_m=32, block_b=1)

    def test_short_m_uses_the_warpgroup_row_layout(self):
        # 6 rows: no multiple of 4 divides them, so the fast path is out, but
        # the shape is no longer rejected.
        p = mosaic_tiling.plan((6, 256, 1), self.F32, block_m=32, block_b=1)
        self.assertEqual((p.block_m, p.grid), (6, (1, 1)))
        self.assertEqual(
            p.layout, mosaic_tiling.warpgroup_row_layout(256, self.F32)
        )

    def test_reduced_axis_contiguous(self):
        p = mosaic_tiling.plan((256, 128, 1), self.F32, block_m=32, block_b=1)
        self.assertEqual((p.block_m, p.block_b), (32, 1))
        self.assertEqual(p.x_shape, (256, 128))
        self.assertEqual(p.stat_shape, (256,))
        self.assertEqual(p.grid, (8, 1))
        self.assertEqual(p.dparam_shape, (8, 1, 128))
        # `A` is the tile's fast axis, so it is reduced within a warp, and the
        # params are reduced over `M`, which crosses warps.
        self.assertEqual((p.reduce_axis, p.stat_axes), (1, (0,)))

    def test_reduced_axis_strided(self):
        p = mosaic_tiling.plan((4, 32, 256), self.F32, block_m=32, block_b=128)
        # `M` goes entirely to the grid.
        self.assertEqual((p.block_m, p.block_b), (1, 128))
        self.assertEqual(p.x_shape, (4, 32, 256))
        self.assertEqual(p.stat_shape, (4, 256))
        self.assertEqual(p.grid, (4, 2))
        self.assertEqual(p.dparam_shape, (4, 2, 32))
        self.assertEqual((p.reduce_axis, p.stat_axes), (0, (1,)))

    def test_layouts_are_constructible(self):
        for x_shape in [(256, 128, 1), (4, 32, 256), (6, 256, 1)]:
            p = mosaic_tiling.plan(x_shape, self.F32, block_m=32, block_b=128)
            # `to_mgpu` is where a bad tiling or partitioning would blow up.
            p.layout.to_mgpu()
            p.stat_layout.to_mgpu()
            p.param_layout.to_mgpu()

    def test_row_grid_splits_over_two_axes(self):
        grid_m = 1 << 17  # Over the 65535 cap on gridDim.y/.z.
        p = mosaic_tiling.plan((4 * grid_m, 128, 1), self.F32, block_m=4, block_b=1)
        self.assertEqual(p.grid_names, ('mo', 'm', 'b'))
        outer, inner, _ = p.grid
        self.assertLessEqual(max(outer, inner), 65535)
        self.assertEqual(inner, p.split_m)
        self.assertEqual(outer * p.split_m, grid_m)

    def test_unsplittable_row_grid(self):
        prime = 65599  # Over the cap, and its only divisor below it is 1.
        with self.assertRaises(NotImplementedError):
            mosaic_tiling.plan((4 * prime, 128, 1), self.F32, block_m=4, block_b=1)

    def test_block_m_is_warp_aligned(self):
        """A divisor under the cap is not enough; it has to fill four warps."""
        # 6 is the largest divisor of 12 under the cap, but it is not four
        # warps' worth, so 4 is taken instead.
        p = mosaic_tiling.plan((12, 128, 1), self.F32, block_m=8, block_b=1)
        self.assertEqual(p.block_m, 4)
        self.assertEqual(p.grid, (3, 1))

    def test_no_warp_aligned_block_falls_back(self):
        """2 rows cannot fill four warps, so `A` supplies them instead."""
        p = mosaic_tiling.plan((2, 128, 1), self.F32, block_m=32, block_b=1)
        self.assertEqual((p.block_m, p.grid), (2, (1, 1)))
        self.assertEqual(
            p.layout, mosaic_tiling.warpgroup_row_layout(128, self.F32)
        )

    def test_rows_per_element_keeps_blocks_inside_one_element(self):
        """A folded `vmap` batch: blocks must not straddle a batch boundary."""
        p = mosaic_tiling.plan(
            (24, 128, 1), self.F32, block_m=32, block_b=1, rows_per_element=12
        )
        self.assertEqual(p.block_m, 12)
        self.assertEqual(12 % p.block_m, 0)  # Blocks divide one element's rows.
        self.assertEqual(p.grid, (2, 1))

    def test_register_budget(self):
        with self.assertRaisesRegex(NotImplementedError, 'spill'):
            mosaic_tiling.plan(
                (256, 1 << 16, 1), self.F32, block_m=32, block_b=1
            )


if __name__ == '__main__':
    absltest.main()
