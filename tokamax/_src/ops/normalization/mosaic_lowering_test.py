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
  store would take the `stmatrix` path rather than the plain vector stores it
  will actually run on (`fragmented_array.py`, `TxMatrixIneligible`). Overriding
  the private `_infer_arch` is the only way in: `AbstractDevice` has no
  `compute_capability` field.
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
          # `block_m` shrinks to fit the register budget as the row grows.
          dict(shape=(4096, 1024), axis=-1, block_m=32, block_n=None),
          # Rows that the block does not divide: the last tile slides back.
          dict(shape=(100, 128), axis=-1, block_m=32, block_n=None),
          # A rank-3 tile: the reduced axis is not the minor one, so `N` is
          # blocked, the reduction runs down the tile's middle axis in registers,
          # and the slice this loads is strided. `A` is bounded here by the
          # register budget; see
          # `PlanTest.test_a_register_resident_reduced_axis_is_bounded`.
          dict(shape=(8, 64, 256), axis=1, block_m=32, block_n=32),
          # `block_n` that the trailing axis does not divide.
          dict(shape=(4, 32, 96), axis=1, block_m=32, block_n=64),
          # Reducing the major axis, so `M` is 1 and too short for the warps:
          # they come from `N` instead and `M` is not blocked.
          dict(shape=(32, 256), axis=0, block_m=32, block_n=32),
          # A trailing axis too short for the lanes to tile: they go back on the
          # reduced axis and `N` rides inside each lane.
          dict(shape=(24, 32, 3), axis=1, block_m=8, block_n=32),
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
  """Every axis but `M` must be tiled exactly: only `M`'s last tile can slide."""

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

  def _plan(self, shape, *, block_m=32, block_n=None):
    return mosaic_tiling.plan(
        shape, self.F32, block_m=block_m, block_n=block_n
    )

  def test_block_divides_the_rows(self):
    p = self._plan((4096, 128))
    self.assertEqual(p.block, (32, 128))
    self.assertEqual(p.grid, (128, 1))

  def test_the_cap_is_taken_whether_or_not_it_divides_the_rows(self):
    # 32 does not divide 104, and no longer has to: the last tile slides back.
    p = self._plan((104, 128))
    self.assertEqual(p.block, (32, 128))
    self.assertEqual(p.grid, (4, 1))

  def test_the_last_row_block_slides_back(self):
    p = self._plan((100, 128))
    self.assertEqual((p.block, p.grid), ((32, 128), (4, 1)))
    starts = [int(p.tile_starts((i, 0))[0]) for i in range(p.grid[0])]
    # The tail overlaps its predecessor rather than overhanging `M`.
    self.assertEqual(starts, [0, 32, 64, 68])

  def test_rows_must_feed_the_warps(self):
    with self.assertRaisesRegex(NotImplementedError, 'at least 4'):
      self._plan((3, 128))

  def test_short_rows_give_the_warps_to_the_trailing_axis(self):
    """With an `N` to take them, `M` shorter than the warp rows is fine."""
    p = self._plan((1, 32, 256), block_n=32)
    # `M` is not blocked, and `N` is blocked by the whole warpgroup rather than
    # by the 32 lanes -- which overrides the smaller `block_n` asked for.
    self.assertEqual(p.block, (1, 32, 128))
    self.assertEqual(p.grid, (1, 1, 2))
    with self.assertRaisesRegex(NotImplementedError, 'multiple of 128'):
      self._plan((1, 32, 96), block_n=32)

  def test_reduced_axis_must_be_a_multiple_of_32(self):
    with self.assertRaisesRegex(NotImplementedError, 'multiple of 32'):
      self._plan((4096, 48))

  def test_block_shrinks_to_fit_the_registers(self):
    """A long row is paid for in `block_m`, not declined outright."""
    wide = self._plan((4096, 2048))
    narrow = self._plan((4096, 128))
    self.assertLess(wide.block[0], narrow.block[0])
    self.assertLessEqual(wide.tile_regs(), 64)

  def test_reduced_axis_can_be_too_wide_for_the_registers(self):
    with self.assertRaisesRegex(NotImplementedError, 'registers per thread'):
      self._plan((4096, 4096))

  def test_the_reduced_axis_is_never_blocked(self):
    for shape in ((4096, 128), (24, 32, 256)):
      with self.subTest(shape=shape):
        p = self._plan(shape, block_n=32)
        axis = mosaic_tiling.REDUCE_AXIS
        self.assertEqual(p.block[axis], shape[axis])
        self.assertEqual(p.grid[axis], 1)
        self.assertEqual(
            mosaic_tiling.drop_reduced(shape), shape[:1] + shape[2:]
        )

  def test_trailing_axis_takes_the_lane_tiling(self):
    """With an `N`, `N` is what the lanes tile and `M` still feeds the warps."""
    p = self._plan((24, 32, 256), block_m=8, block_n=32)
    self.assertEqual(p.block, (8, 32, 32))
    self.assertEqual(p.grid, (3, 1, 8))
    # `A` is tiled by neither lanes nor warps here, so it is unconstrained: it
    # lives in registers, one whole reduced row per thread.
    for num_a in (4, 5, 34):
      self.assertEqual(self._plan((24, num_a, 256), block_n=32).block[1], num_a)

  def test_a_register_resident_reduced_axis_is_bounded(self):
    """`A` in registers costs `block_m * A * block_n / 128` per thread.

    `block_m` cannot go under the four warp rows and `block_n` not under the 32
    lanes, so with an `N` there is nothing left to shrink and a long `A` is
    declined rather than blocked.
    """
    self.assertEqual(self._plan((8, 64, 256), block_n=32).tile_regs(), 64)
    with self.assertRaisesRegex(NotImplementedError, 'registers per thread'):
      self._plan((8, 128, 256), block_n=32)

  def test_a_trailing_axis_the_lanes_cannot_tile_rides_in_them(self):
    """A block of `N` has to divide it exactly, so 32 lanes need 32 elements.

    Below that -- or at any `N` with no 32-multiple divisor -- the lanes go back
    on the reduced axis and each lane takes the whole of `N`.
    """
    for num_n in (1, 3, 4, 48):
      with self.subTest(num_n=num_n):
        p = self._plan((24, 32, num_n), block_m=8, block_n=32)
        # `N` is not blocked at all, so it has no grid axis of its own.
        self.assertEqual(p.block[1:], (32, num_n))
        self.assertEqual(p.grid[1:], (1, 1))

  def test_nothing_fits_a_short_m_and_a_short_n(self):
    """`M` too short for the warps and `N` too short to take them instead."""
    with self.assertRaisesRegex(NotImplementedError, 'No thread mapping fits'):
      self._plan((2, 32, 48), block_n=32)

  def test_every_tile_is_visited_exactly_once(self):
    """The grid index turns into one element offset per axis.

    One CTA takes one tile, so the whole array has to be covered and no tile may
    be started twice -- the statistics are written per tile. A slid tile overlaps
    its predecessor, which is harmless: it recomputes those rows to the same
    values.
    """
    for shape, block_n in (
        ((24, 32, 256), 32),
        ((24, 32), None),
        ((24, 32, 96), 32),
        ((100, 32), None),  # Rows the block does not divide.
    ):
      with self.subTest(shape=shape):
        p = self._plan(shape, block_m=8, block_n=block_n)
        tiles = list(itertools.product(*map(range, p.grid)))
        seen = [tuple(int(i) for i in p.tile_starts(t)) for t in tiles]
        expected = sorted(
            itertools.product(*(
                sorted({min(i * b, s - b) for i in range(n)})
                for s, b, n in zip(p.shape, p.block, p.grid, strict=True)
            ))
        )
        self.assertEqual(sorted(seen), expected)
        self.assertLen(set(seen), len(tiles))

  def test_a_single_tile_is_a_single_cta(self):
    p = self._plan((8, 128))
    self.assertEqual((p.block, p.grid), ((8, 128), (1, 1)))


if __name__ == '__main__':
  absltest.main()
