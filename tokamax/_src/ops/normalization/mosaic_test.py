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
from jax.experimental.mosaic.gpu import core as mgpu_core
import jax.numpy as jnp
from tokamax._src import config as config_lib
from tokamax._src.ops.normalization import mosaic
from tokamax._src.ops.normalization import mosaic_tiling
from tokamax._src.ops.normalization import pallas_triton_vjp
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
        # `mosaic_tiling.plan`): whichever axis the tile blocks needs a divisor
        # that is a multiple of the tiling, and the tile has to fit SMEM.
        # Everything else raises `NotImplementedError` by design, so report those
        # as skips rather than failures -- a real bug still surfaces as a failure.
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

    @parameterized.named_parameters(
        ('_plain', False, 4096),
        ('_under_vmap', True, 4096),
        # 4096 is a multiple of `block_m`; 100 is not, so the last block slides
        # back over the previous one and `dscale`/`doffset` would double-count
        # the repeated rows if the kernel did not mask them.
        ('_overlapping_blocks', False, 100),
    )
    def test_grad_gradients(self, under_vmap, num_rows):
        """Gradients against plain JAX, with and without an enclosing `vmap`.

        The pair is deliberate: it separates a wrong kernel from wrong batching.
        `test_base` only ever takes `jax.vjp` of an already-vmapped function, so
        `scale` is shared and `dscale` is summed over the batch; the `_under_vmap`
        case is the other order, where each element owns its gradient. Configs are
        explicit so which case of `mosaic_tiling.plan` runs does not depend on the
        device's core count.
        """
        rows = (4, num_rows) if under_vmap else (num_rows,)
        shape = (*rows, 128)
        rngs = list(jax.random.split(jax.random.PRNGKey(0), 3))
        x = jax.random.normal(rngs.pop(), shape)
        scale = jax.random.uniform(rngs.pop(), (shape[-1],))
        offset = jax.random.uniform(rngs.pop(), (shape[-1],))
        epsilon = 1e-6

        config = dict(block_m=32, block_n=None, num_warps=4)
        op = mosaic.PallasMosaicGpuNormalization(
            config=triton_config.Config(**config),
            vjp=pallas_triton_vjp.PallasTritonNormalizationVjp(
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


class DivisorsTest(absltest.TestCase):
    """Blocks must tile `M` exactly: cp.async cannot predicate an OOB copy."""

    def test_divisors_descend_and_divide(self):
        got = list(mosaic_tiling.divisors(96, 32, multiple_of=8))
        self.assertEqual(got, [32, 24, 16, 8])

    def test_divisors_respects_the_cap_and_the_multiple(self):
        self.assertEqual(list(mosaic_tiling.divisors(12, 8, multiple_of=8)), [])
        self.assertEqual(list(mosaic_tiling.divisors(64, 128, multiple_of=8)),
                         [64, 32, 16, 8])


@contextlib.contextmanager
def _target_ampere():
    """Makes Mosaic lower for sm_80 rather than for the local device.

    Mosaic takes the target from the default device's `compute_capability` and
    falls back to `(9, 0)` when there is none (`mosaic_gpu/core.py:_infer_arch`).
    On a machine with no GPU -- every machine that runs this test in CI --
    `lowering_platforms=('cuda',)` alone would stamp the module as Hopper, and
    the kernel would take the TMA path rather than the `cp.async` one it will
    actually run on. Overriding the private `_infer_arch` is the only way in:
    `AbstractDevice` has no `compute_capability` field.
    """
    with mock.patch.object(mgpu_core, '_infer_arch', lambda: (8, 0)):
        yield


class AmpereLoweringTest(parameterized.TestCase):
    """Lowers the kernels for Ampere, on whatever machine is running the test.

    `--tokamax_cross_compile` drops the hardware checks, `lowering_platforms`
    picks the backend and `_target_ampere` picks the arch, so no GPU is needed:
    `.lower()` runs the whole Mosaic pipeline -- layout inference, the SMEM
    scratch for a cross-warp reduction, NVVM -- which is where a bad layout shows
    up. Only `.compile()`, i.e. PTX and the register allocator, needs the real
    device.

    Configs are explicit because the heuristics ask the local device for its core
    count, and because the block sizes are what select the case in
    `mosaic_tiling.plan`.
    """

    def _op(self, *, block_m, block_n):
        config = dict(block_m=block_m, block_n=block_n, num_warps=4)
        return mosaic.PallasMosaicGpuNormalization(
            config=triton_config.Config(**config),
            vjp=pallas_triton_vjp.PallasTritonNormalizationVjp(
                config=vjp_config.Config(**config)
            ),
        )

    def _lower(self, shape, *, axis, block_m, block_n, use_params, subtract_mean):
        """Lowers the forward kernel for sm_80.

        Forward only: gradients go through the Triton VJP until the Mosaic one
        is ported, and the Triton backend cannot lower without a real device.
        """
        op = self._op(block_m=block_m, block_n=block_n)
        x = jnp.zeros(shape, jnp.float32)
        params = jnp.zeros((shape[axis],), jnp.float32) if use_params else None
        f = jax.jit(
            lambda x, s, o: op(x, s, o, axis=axis, subtract_mean=subtract_mean)
        )
        with config_lib.cross_compile(True), _target_ampere():
            return f.trace(x, params, params).lower(
                lowering_platforms=('cuda',)
            ).as_text()

    @parameterized.product(
        (
            # One case per branch of `mosaic_tiling.plan`.
            dict(shape=(4096, 128), axis=-1, block_m=32, block_n=None),
            # `block_m` shrinks to fit SMEM as the reduced axis grows.
            dict(shape=(4096, 1024), axis=-1, block_m=32, block_n=None),
            # Rows that the cap does not divide: a smaller divisor is taken.
            dict(shape=(104, 128), axis=-1, block_m=32, block_n=None),
            # Strided: the reduced axis is not the contiguous one, so the tile
            # is `(A, block_b)` and the reduction runs down its slow axis.
            dict(shape=(8, 128, 256), axis=1, block_m=32, block_n=32),
            # `block_n` that the trailing axis does not divide.
            dict(shape=(4, 32, 96), axis=1, block_m=32, block_n=64),
        ),
        use_params=(False, True),
        subtract_mean=(False, True),
    )
    def test_lowers(self, **kwargs):
        hlo = self._lower(**kwargs)
        # The forward kernel, and nothing that fell back to XLA.
        self.assertEqual(hlo.count('custom_call @mosaic_gpu_v2'), 1, msg=hlo)

    def test_batched_params_lower(self):
        """`vmap` may give each element its own `scale`/`offset`."""
        op = self._op(block_m=32, block_n=None)
        x = jnp.zeros((3, 128, 32), jnp.float32)
        params = jnp.zeros((3, 32), jnp.float32)
        f = jax.vmap(lambda x, s, o: op(x, s, o))

        with config_lib.cross_compile(True), _target_ampere():
            hlo = str(
                jax.jit(f)
                .trace(x, params, params)
                .lower(lowering_platforms=('cuda',))
                .as_text()
            )
            out = jax.eval_shape(f, x, params, params)

        self.assertEqual(hlo.count('custom_call @mosaic_gpu_v2'), 1, msg=hlo)
        self.assertEqual(out.shape, (3, 128, 32))


class PlanTest(absltest.TestCase):
    """`mosaic_tiling.plan` is pure Python, so it is testable without a GPU."""

    F32 = 4

    def test_block_divides_the_rows(self):
        p = mosaic_tiling.plan(4096, 128, self.F32, block_m=32)
        self.assertEqual(p.block_m, 32)
        self.assertEqual(4096 % p.block_m, 0)
        self.assertEqual(p.steps, 128)

    def test_block_falls_to_a_divisor_under_the_cap(self):
        # 32 does not divide 104; 8 is the largest multiple of 8 that does.
        p = mosaic_tiling.plan(104, 128, self.F32, block_m=32)
        self.assertEqual(p.block_m, 8)
        self.assertEqual(p.steps, 13)

    def test_rows_must_be_a_multiple_of_the_tiling(self):
        with self.assertRaisesRegex(NotImplementedError, 'multiple of 8'):
            mosaic_tiling.plan(100, 128, self.F32, block_m=32)

    def test_reduced_axis_must_be_a_multiple_of_32(self):
        with self.assertRaisesRegex(NotImplementedError, 'multiple of 32'):
            mosaic_tiling.plan(4096, 48, self.F32, block_m=32)

    def test_block_shrinks_to_fit_smem(self):
        """A long row is paid for in `block_m`, not declined outright."""
        wide = mosaic_tiling.plan(4096, 2048, self.F32, block_m=32)
        narrow = mosaic_tiling.plan(4096, 128, self.F32, block_m=32)
        self.assertLess(wide.block_m, narrow.block_m)
        self.assertLessEqual(wide.smem_bytes(), 227 * 1024)

    def test_reduced_axis_can_be_too_wide_for_smem(self):
        with self.assertRaisesRegex(NotImplementedError, 'SMEM'):
            mosaic_tiling.plan(4096, 8192, self.F32, block_m=32)

    def test_ctas_cover_every_step(self):
        p = mosaic_tiling.plan(4096, 128, self.F32, block_m=32)
        self.assertGreaterEqual(p.num_ctas * p.steps_per_cta, p.steps)
        # A CTA handed fewer steps re-reads its last tile rather than going OOB.
        self.assertLess((p.num_ctas - 1) * p.steps_per_cta, p.steps)

    def test_strided_blocks_the_trailing_axis(self):
        """`B > 1`: a tile holds one row of `M` and a block of `B`."""
        p = mosaic_tiling.plan(24, 32, self.F32, block_m=32, num_b=256,
                               block_b=32)
        self.assertEqual((p.block_m, p.block_b), (1, 32))
        self.assertEqual(p.tile_shape, (32, 32))
        self.assertEqual(p.block_shape, (None, 32, 32))
        self.assertEqual(p.steps, 24 * 8)
        # The reduction runs down the tile's slow axis, which `A` spans.
        self.assertEqual(p.reduce_axis, 0)

    def test_strided_reduced_axis_must_be_a_multiple_of_8(self):
        # `A` is the tiled axis here, so 8 divides it rather than 32.
        mosaic_tiling.plan(24, 8, self.F32, block_m=32, num_b=256, block_b=32)
        with self.assertRaisesRegex(NotImplementedError, 'multiple of 8'):
            mosaic_tiling.plan(24, 36, self.F32, block_m=32, num_b=256,
                               block_b=32)

    def test_strided_trailing_axis_needs_a_usable_block(self):
        with self.assertRaisesRegex(NotImplementedError, 'no divisor'):
            mosaic_tiling.plan(24, 32, self.F32, block_m=32, num_b=48,
                               block_b=32)

    def test_every_tile_is_visited_exactly_once(self):
        """The flat step index splits into a block of `M` and a block of `B`.

        The whole array has to be covered, and -- because the statistics are
        written per tile -- no tile may be visited twice, other than the clamped
        repeats of the very last one.
        """
        for kwargs in (
            dict(num_b=256, block_b=32),  # Strided.
            dict(),  # Contiguous.
            dict(num_b=96, block_b=64),  # Steps the CTAs do not divide evenly.
        ):
            with self.subTest(**kwargs):
                p = mosaic_tiling.plan(24, 32, self.F32, block_m=32, **kwargs)
                seen = []
                for cta in range(p.num_ctas):
                    with mock.patch.object(
                        jax.lax, 'axis_index', lambda _, cta=cta: cta
                    ):
                        for step in range(p.steps_per_cta):
                            i, j = p.tile_indices(step)
                            seen.append((int(i), int(j)))
                expected = [
                    (i, j)
                    for i in range(p.num_m // p.block_m)
                    for j in range(p.steps_b)
                ]
                self.assertEqual(sorted(set(seen)), expected)
                # Only the last tile may repeat, and only as the clamped tail.
                repeats = len(seen) - len(set(seen))
                self.assertEqual(
                    seen[len(seen) - repeats:], [expected[-1]] * repeats
                )

    def test_a_single_block_needs_no_pipeline(self):
        p = mosaic_tiling.plan(8, 128, self.F32, block_m=32)
        self.assertEqual((p.block_m, p.steps, p.num_ctas), (8, 1, 1))
        self.assertEqual(p.num_stages, 1)


if __name__ == '__main__':
    absltest.main()
