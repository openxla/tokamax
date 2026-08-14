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
"""Tests for the Mosaic GPU normalization kernel that need no GPU.

`mosaic_test.py` mirrors `pallas_triton_test.py` and so covers numerics, which
needs the hardware. Everything here runs on any machine: the blocking is pure
Python, and `.lower()` runs the whole Mosaic pipeline -- layout inference, the
SMEM scratch for a cross-warp reduction, NVVM -- which is where a bad layout
shows up. Only `.compile()`, i.e. PTX and the register allocator, needs a device.
"""

import contextlib
import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.mosaic.gpu import core as mgpu_core
import jax.numpy as jnp
from tokamax._src import config as config_lib
from tokamax._src.ops.normalization import mosaic
from tokamax._src.ops.normalization import mosaic_tiling
from tokamax._src.ops.normalization import pallas_triton_config as triton_config
from tokamax._src.ops.normalization import pallas_triton_vjp
from tokamax._src.ops.normalization import pallas_triton_vjp_config as vjp_config


@contextlib.contextmanager
def _target_ampere():
  """Makes Mosaic lower for sm_80 rather than for the local device.

  Mosaic takes the target from the default device's `compute_capability` and
  falls back to `(9, 0)` when there is none (`mosaic_gpu/core.py:_infer_arch`).
  On a machine with no GPU -- every machine that runs this test in CI --
  `lowering_platforms=('cuda',)` alone would stamp the module as Hopper, and the
  kernel would take the TMA path rather than the `cp.async` one it will actually
  run on. Overriding the private `_infer_arch` is the only way in:
  `AbstractDevice` has no `compute_capability` field.
  """
  with mock.patch.object(mgpu_core, '_infer_arch', lambda: (8, 0)):
    yield


class AmpereLoweringTest(parameterized.TestCase):
  """Lowers the kernel for Ampere, on whatever machine is running the test.

  `--tokamax_cross_compile` drops the hardware checks, `lowering_platforms`
  picks the backend and `_target_ampere` picks the arch, so no GPU is needed.

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

    Forward only: gradients go through the Triton VJP until the Mosaic one is
    ported, and the Triton backend cannot lower without a real device.
    """
    op = self._op(block_m=block_m, block_n=block_n)
    x = jnp.zeros(shape, jnp.float32)
    params = jnp.zeros((shape[axis],), jnp.float32) if use_params else None
    f = jax.jit(
        lambda x, s, o: op(x, s, o, axis=axis, subtract_mean=subtract_mean)
    )
    with config_lib.cross_compile(True), _target_ampere():
      return (
          f.trace(x, params, params)
          .lower(lowering_platforms=('cuda',))
          .as_text()
      )

  @parameterized.product(
      (
          # One case per branch of `mosaic_tiling.plan`.
          dict(shape=(4096, 128), axis=-1, block_m=32, block_n=None),
          # `block_m` shrinks to fit SMEM as the reduced axis grows.
          dict(shape=(4096, 1024), axis=-1, block_m=32, block_n=None),
          # Rows that the cap does not divide: a smaller divisor is taken.
          dict(shape=(104, 128), axis=-1, block_m=32, block_n=None),
          # A rank-3 tile: the reduced axis is not the minor one, so `B` is
          # blocked and the reduction runs down the tile's middle axis. Needs
          # the four `jax` fixes in `mgpu-rank3-transfer-probe/README.md`.
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


class DivisorsTest(absltest.TestCase):
  """Blocks must tile their axis exactly: cp.async cannot predicate an OOB copy."""

  def test_divisors_descend_and_divide(self):
    got = list(mosaic_tiling.divisors(96, 32, multiple_of=8))
    self.assertEqual(got, [32, 24, 16, 8])

  def test_divisors_respects_the_cap_and_the_multiple(self):
    self.assertEqual(list(mosaic_tiling.divisors(12, 8, multiple_of=8)), [])
    self.assertEqual(
        list(mosaic_tiling.divisors(64, 128, multiple_of=8)), [64, 32, 16, 8]
    )


class PlanTest(absltest.TestCase):
  """`mosaic_tiling.plan` is pure Python, so it is testable without a GPU."""

  F32 = 4

  def _plan(self, shape, *, block_m=32, block_b=None):
    return mosaic_tiling.plan(
        shape, self.F32, block_m=block_m, block_b=block_b
    )

  def test_block_divides_the_rows(self):
    p = self._plan((4096, 128))
    self.assertEqual(p.block, (32, 128))
    self.assertEqual(p.steps, 128)

  def test_block_falls_to_a_divisor_under_the_cap(self):
    # 32 does not divide 104; 8 is the largest multiple of 8 that does.
    p = self._plan((104, 128))
    self.assertEqual(p.block, (8, 128))
    self.assertEqual(p.steps, 13)

  def test_rows_must_be_a_multiple_of_the_tiling(self):
    with self.assertRaisesRegex(NotImplementedError, 'multiple of 8'):
      self._plan((100, 128))

  def test_reduced_axis_must_be_a_multiple_of_32(self):
    with self.assertRaisesRegex(NotImplementedError, 'multiple of 32'):
      self._plan((4096, 48))

  def test_block_shrinks_to_fit_smem(self):
    """A long row is paid for in `block_m`, not declined outright."""
    wide = self._plan((4096, 2048))
    narrow = self._plan((4096, 128))
    self.assertLess(wide.block[0], narrow.block[0])
    self.assertLessEqual(wide.smem_bytes(), 227 * 1024)

  def test_reduced_axis_can_be_too_wide_for_smem(self):
    with self.assertRaisesRegex(NotImplementedError, 'SMEM'):
      self._plan((4096, 8192))

  def test_ctas_cover_every_step(self):
    p = self._plan((4096, 128))
    self.assertGreaterEqual(p.num_ctas * p.steps_per_cta, p.steps)
    # A CTA handed fewer steps re-reads its last tile rather than going OOB.
    self.assertLess((p.num_ctas - 1) * p.steps_per_cta, p.steps)

  def test_the_reduced_axis_is_never_blocked(self):
    for shape in ((4096, 128), (24, 32, 256)):
      with self.subTest(shape=shape):
        p = self._plan(shape, block_b=32)
        axis = mosaic_tiling.REDUCE_AXIS
        self.assertEqual(p.block[axis], shape[axis])
        self.assertEqual(p.grid[axis], 1)
        self.assertEqual(
            mosaic_tiling.drop_reduced(shape), shape[:1] + shape[2:]
        )

  def test_trailing_axis_takes_the_lane_tiling(self):
    """With a `B`, `B` is what the lanes tile and `A` takes the 8 rows."""
    p = self._plan((24, 32, 256), block_m=8, block_b=32)
    self.assertEqual(p.block, (8, 32, 32))
    self.assertEqual(p.grid, (3, 1, 8))
    # `A` needs only `% 8` here, rather than the `% 32` a minor axis needs.
    self._plan((24, 8, 256), block_b=32)
    with self.assertRaisesRegex(NotImplementedError, 'multiple of 8'):
      self._plan((24, 36, 256), block_b=32)

  def test_trailing_axis_needs_a_usable_block(self):
    with self.assertRaisesRegex(NotImplementedError, 'no divisor'):
      self._plan((24, 32, 48), block_b=32)

  def test_every_tile_is_visited_exactly_once(self):
    """The flat step index splits into one block index per axis.

    The whole array has to be covered, and -- because the statistics are written
    per tile -- no tile may be visited twice, other than the clamped repeats of
    the very last one.
    """
    for shape, block_b in (
        ((24, 32, 256), 32),
        ((24, 32), None),
        ((24, 32, 96), 32),  # Steps the CTAs do not divide evenly.
    ):
      with self.subTest(shape=shape):
        p = self._plan(shape, block_m=8, block_b=block_b)
        seen = []
        for cta in range(p.num_ctas):
          with mock.patch.object(
              jax.lax, 'axis_index', lambda _, cta=cta: cta
          ):
            for step in range(p.steps_per_cta):
              seen.append(tuple(int(i) for i in p.block_indices(step)))
        expected = sorted(itertools.product(*map(range, p.grid)))
        self.assertEqual(sorted(set(seen)), expected)
        # Only the last tile may repeat, and only as the clamped tail.
        repeats = len(seen) - len(set(seen))
        self.assertEqual(seen[len(seen) - repeats :], [expected[-1]] * repeats)

  def test_a_single_block_needs_no_pipeline(self):
    p = self._plan((8, 128))
    self.assertEqual((p.block, p.steps, p.num_ctas), ((8, 128), 1, 1))
    self.assertEqual(p.num_stages, 1)


if __name__ == '__main__':
  absltest.main()
