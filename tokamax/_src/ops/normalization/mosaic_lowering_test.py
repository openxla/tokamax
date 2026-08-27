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
import math
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.mosaic.gpu import core as mgpu_core
import jax.numpy as jnp
from tokamax._src import config as config_lib
from tokamax._src import gpu_utils
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

  A config is passed explicitly because the heuristics ask the local device for
  its core count. Its block sizes reach only the Triton VJP: the forward kernel
  takes none, `mosaic_tiling.plan` settling the tile from the shape alone, so
  the shape is the whole of what selects a case below.
  """

  def _op(self):
    config = dict(block_m=32, block_n=None, num_warps=4)
    return mosaic.PallasMosaicGpuNormalization(
        config=triton_config.Config(**config),
        vjp=pallas_triton_vjp.PallasTritonNormalizationVjp(
            config=vjp_config.Config(**config)
        ),
    )

  def _lower(self, shape, *, axis, use_params, subtract_mean):
    """Lowers the forward kernel for sm_80.

    Forward only: gradients go through the Triton VJP until the Mosaic one is
    ported, and the Triton backend cannot lower without a real device.
    """
    op = self._op()
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
          # No `N`, so all 32 lanes tile the reduced axis.
          dict(shape=(4096, 128), axis=-1),
          # The rows give way to the register budget as the row length grows.
          dict(shape=(4096, 1024), axis=-1),
          # Rows whose divisors are few: the block is the largest of them the
          # warps can take, since every axis is tiled exactly.
          dict(shape=(100, 128), axis=-1),
          # A rank-3 tile: the reduced axis is not the minor one, so the lanes
          # split between the two and `N` is blocked.
          dict(shape=(8, 64, 256), axis=1),
          # A trailing axis a whole cache line does not divide.
          dict(shape=(4, 32, 96), axis=1),
          # Rows to spare, but a reduced axis long enough that four warps on `M`
          # could not afford a cache line of columns: the warps go to `N` and the
          # rows block below the four they would otherwise need.
          dict(shape=(8, 128, 256), axis=1),
          # Reducing the major axis, so `M` is 1 and too short for the warps:
          # they come from `N` too and `M` is not blocked.
          dict(shape=(32, 256), axis=0),
          # The same, with `M` long enough to feed two of the four warps.
          dict(shape=(2, 32, 256), axis=1),
          # A trailing axis with no run for the lanes to tile: they all go back
          # on the reduced axis and `N` rides inside each lane.
          dict(shape=(24, 32, 3), axis=1),
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
    op = self._op()
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
  """Every axis is tiled exactly, so every block is a divisor of its axis."""

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

  def _plan(self, shape):
    return mosaic_tiling.plan(shape, self.F32)

  def test_the_rows_take_the_whole_register_budget(self):
    """Nothing else can spend it: the reduced axis is whole and `N` has a line."""
    p = self._plan((4096, 128))
    self.assertEqual((p.block, p.grid()), ((64, 128), (64, 1)))
    self.assertEqual(p.tile_regs(), 64)

  def test_the_row_block_drops_to_a_divisor_of_the_rows(self):
    """The block is the largest divisor of `M` the register budget affords.

    Every axis is tiled exactly, so a block that does not divide its axis is not
    a block at all -- there is no masked GMEM store to write a partial tile with.
    """
    # The budget affords 64 rows, and 52 is the largest divisor of 104 under it.
    p = self._plan((104, 128))
    self.assertEqual((p.block, p.grid()), ((52, 128), (2, 1)))
    p = self._plan((100, 128))
    self.assertEqual((p.block, p.grid()), ((20, 128), (5, 1)))

  def test_short_rows_give_the_warps_to_the_trailing_axis(self):
    """With an `N` to take them, `M` shorter than the warp rows is fine."""
    p = self._plan((1, 32, 256))
    # `M` feeds no warp of its own, so it is not blocked at all.
    self.assertEqual(p.block, (1, 32, 32))
    self.assertEqual(p.grid(), (1, 1, 8))

  def test_short_rows_still_feed_the_warps_they_can(self):
    """`M` takes as many of the four warps as divide it, and `N` takes the rest.

    Only the warps `N` feeds constrain it, so two rows halve what `N` has to be a
    multiple of -- and `M` is then blocked in twos rather than left whole.
    """
    p = self._plan((2, 32, 48))
    self.assertEqual(p.block, (2, 32, 16))
    self.assertEqual(p.grid(), (1, 1, 3))
    # `N` is tiled twice: 2 warps, then 2 of the lanes, then a 4-element vector.
    # (m_warps, a_lanes, n_warps, sub_a, n_lanes, vec).
    layout = p.layout.to_mgpu()
    self.assertEqual(layout.tiled_tiling_shape, (2, 16, 2, 2, 2, 4))
    self.assertEqual((layout.warp_dims, layout.lane_dims), ((-6, -4), (-5, -2)))

  def test_a_tile_over_the_budget_puts_every_lane_back_on_the_reduced_axis(self):
    """Giving `N` lanes as well as warps raises the least `N` a block can take.

    The base tile spans `n_warps * n_lanes * vec` of `N` rather than
    `n_warps * vec`, so where the sector-wide run does not fit the budget the
    narrower one still can. `plan` can only shrink `M`, and `M` is already down
    to what the warps take, so this is the last thing left to give up.
    """
    wide = self._plan((1, 128, 4096))
    # (m_warps, a_lanes, n_warps, sub_a, n_lanes, vec); `N` feeds a lane dim.
    self.assertEqual(wide.layout.to_mgpu().lane_dims, (-5, -2))
    self.assertEqual((wide.block[2], wide.tile_regs()), (32, 32))
    narrow = self._plan((1, 384, 4096))
    # The same block would need 96 registers, so every lane goes back on `A`.
    self.assertEqual(narrow.layout.to_mgpu().lane_dims, (-4,))
    self.assertEqual((narrow.block[2], narrow.tile_regs()), (16, 48))

  def test_short_rows_pay_for_a_long_reduced_axis_in_columns(self):
    """`M` is at its floor here, so `block_n` is the only thing left to shrink.

    `M` feeds no warp of its own, so `plan` cannot take a row off it the way it
    would for a longer one; without `block_n` giving way too, this shape has no
    tile at all.
    """
    p = self._plan((1, 512, 4096))
    self.assertEqual((p.block, p.tile_regs()), ((1, 512, 16), 64))
    self.assertEqual(p.grid(), (1, 1, 256))

  def test_the_vector_narrows_to_let_the_lanes_tile_the_reduced_axis(self):
    """With no `N`, all 32 lanes tile `A`, so `32 * vec` has to divide it.

    The vector is what gives to make that true -- and where nothing does, the
    shape is declined rather than read in overlapping loads: a tail costs a mask
    on every reduction, and registers for elements the load already held.
    """
    self.assertEqual(self._plan((4096, 128)).layout.to_mgpu().vector_length, 4)
    # 32 four-element vectors overrun 96, and 64 does not divide it either.
    self.assertEqual(self._plan((4096, 96)).layout.to_mgpu().vector_length, 1)

  def test_block_shrinks_to_fit_the_registers(self):
    """A long row is paid for in `block_m`, not declined outright."""
    wide = self._plan((4096, 2048))
    narrow = self._plan((4096, 128))
    self.assertLess(wide.block[0], narrow.block[0])
    self.assertLessEqual(wide.tile_regs(), 64)


  def test_the_reduced_axis_is_never_blocked(self):
    for shape in ((4096, 128), (24, 32, 256)):
      with self.subTest(shape=shape):
        p = self._plan(shape)
        axis = 1
        self.assertEqual(p.block[axis], shape[axis])
        self.assertEqual(p.grid()[axis], 1)
        self.assertEqual(
            mosaic_tiling.drop_reduced(shape), shape[:1] + shape[2:]
        )

  def test_trailing_axis_takes_a_share_of_the_lanes(self):
    """With an `N`, the lanes split between it and `A`, and `M` feeds the warps."""
    p = self._plan((24, 32, 256))
    self.assertEqual(p.block, (8, 32, 32))
    self.assertEqual(p.grid(), (3, 1, 8))
    # The reduced axis is never blocked, whatever share of the lanes tiles it --
    # but a share that does not divide it is no mapping at all.
    for num_a in (4, 8, 32):
      self.assertEqual(self._plan((24, num_a, 256)).block[1], num_a)

  def test_the_lane_split_holds_a_whole_sector_per_lane_group(self):
    """The run the lanes read contiguously is what sets the register cost.

    `n_lanes * vec` elements are contiguous, and one 32-byte sector of them is
    the shortest run that wastes no bandwidth; `a_lanes` takes the rest of the
    warp. Here that is 8 `float32`s, so 2 lanes of 4-element vectors on `N` and
    the other 16 on `A`.
    """
    layout = self._plan((24, 32, 256)).layout.to_mgpu()
    self.assertEqual(layout.tiling.tiles, ((4, 32, 8), (2, 4)))
    # (_WARP_ROWS, a_lanes, n_lanes, sub_a, vec).
    self.assertEqual(layout.tiled_tiling_shape, (4, 16, 2, 2, 4))
    self.assertEqual((layout.warp_dims, layout.lane_dims), ((-5,), (-4, -3)))

  def test_every_layout_is_a_legal_tiled_layout(self):
    """`TiledLayout.__post_init__` is the check; building one is running it.

    It is where a layout that places the wrong number of threads, or that names
    a size-1 dimension and so is not canonical, is caught -- and it is pure
    Python, so it runs here rather than needing a device. `registers_shape` then
    has to agree with `tile_regs` on what a thread actually holds.
    """
    shapes = [
        (m, a, *n)
        for m in (1, 2, 3, 4, 24, 100)
        for a in (24, 32, 40, 42, 128)
        for n in ((), (1,), (3,), (4,), (40,), (48,), (256,))
    ]
    for shape in shapes:
      with self.subTest(shape=shape):
        try:
          p = self._plan(shape)
        except NotImplementedError:
          continue  # Declined outright, which the caller falls back from.
        layout = p.layout.to_mgpu()  # Raises if the layout is not legal.
        held = math.prod(layout.registers_shape(p.block))
        self.assertEqual(held * layout.vector_length, p.tile_regs())

  def test_a_register_resident_reduced_axis_is_paid_for_in_both_blocks(self):
    """`A` in registers costs `block_m * A * block_n / 128` per thread.

    The budget bounds the product, so a reduced axis twice as long is afforded by
    halving either factor -- and where a mapping cannot halve either, another
    mapping of the same shape can.
    """
    p = self._plan((8, 64, 256))
    self.assertEqual((p.block, p.tile_regs()), ((4, 64, 32), 64))
    # Twice the reduced axis, so half the tile. The columns are a cache line and
    # will not give, so the rows do -- which takes a mapping whose warps are not
    # on `M`, since four rows are the fewest four warps of `M` can have.
    p = self._plan((8, 128, 256))
    self.assertEqual((p.block, p.tile_regs()), ((2, 128, 32), 64))
    self.assertEqual(p.layout.to_mgpu().lane_dims, (-5, -2))
    # Eight times that, and the sector-wide run is out of reach at any blocking:
    # every lane goes back on `A` for a quarter of the registers, and the run is
    # a bare vector. Only the columns pay for it, `M` being down to one row.
    p = self._plan((8, 512, 256))
    self.assertEqual((p.block, p.tile_regs()), ((1, 512, 16), 64))
    self.assertEqual(p.layout.to_mgpu().lane_dims, (-4,))

  def test_a_trailing_axis_with_no_run_to_tile_rides_in_the_lanes(self):
    """A lane group's run has to divide `N`, so an odd `N` leaves no run at all.

    There the lanes all go back on the reduced axis and each takes the whole of
    `N`, which is contiguous with it -- a `float32` `N` of 4 is already a whole
    16-byte vector, so it takes the lanes too.
    """
    for num_n in (3, 4):
      with self.subTest(num_n=num_n):
        p = self._plan((24, 32, num_n))
        # `N` is not blocked at all, so it has no grid axis of its own.
        self.assertEqual(p.block[1:], (32, num_n))
        self.assertEqual(p.grid()[1:], (1, 1))

  def test_a_degenerate_trailing_axis_is_dropped(self):
    """An `N` of one is no axis at all, and dropping it puts `A` back in minor.

    That is where the vector can come from `A`, so this is strictly the better
    path -- and it is why the rank-2 case is not folded the other way, into an
    `N` of one.
    """
    p = self._plan((24, 128, 1))
    self.assertEqual((p.shape, p.block, p.grid()), ((24, 128), (24, 128), (1, 1)))
    # Four `float32`s, a whole 16-byte vector, where an `N` of one would have
    # pinned the vector to it and left the lanes reading `A` one element at a time.
    self.assertEqual(p.layout.to_mgpu().vector_length, 4)

  def test_a_trailing_axis_the_block_does_not_divide_is_blocked_smaller(self):
    """`N` is tiled exactly, so the block drops to a divisor of it."""
    p = self._plan((24, 32, 48))
    self.assertEqual(p.block, (8, 32, 24))
    self.assertEqual(p.grid(), (3, 1, 2))

  def test_every_tile_is_visited_exactly_once(self):
    """The grid index turns into one element offset per axis.

    One CTA takes one tile, so the whole array has to be covered and no tile may
    be started twice -- the statistics are written per tile. Every axis is tiled
    exactly, so the tiles partition the array and none of them overhangs it.
    """
    for shape, block_n in (
        ((24, 32, 256), 32),
        ((24, 32), None),
        ((24, 32, 96), 32),
        ((100, 32), None),  # Rows the block does not divide.
    ):
      with self.subTest(shape=shape):
        p = self._plan(shape)
        tiles = list(itertools.product(*map(range, p.grid())))
        seen = [tuple(int(i) for i in p.tile_starts(t)) for t in tiles]
        expected = sorted(
            itertools.product(*(
                range(0, s, b) for s, b in zip(p.shape, p.block, strict=True)
            ))
        )
        self.assertEqual(sorted(seen), expected)
        self.assertLen(set(seen), len(tiles))

  def test_the_shapes_the_op_is_benchmarked_on_all_plan(self):
    """`arg_specs`, canonicalized. These are the ones worth being fast on."""
    for shape in ((147456, 64), (147456, 128), (589824, 128)):
      with self.subTest(shape=shape):
        self.assertEqual(self._plan(shape).block[1], shape[1])
    # `axis=0`, so `M` is 1 and the warps come from `N` -- and so do two of the
    # lanes, which is what keeps the load reading a whole sector at a time.
    # `axis=0`, so `M` is 1 -- or the `vmap` batch -- and the warps come from `N`
    # either way, as do two of the lanes; that is what keeps a lane group reading
    # a whole sector and the four warps together a whole cache line. Rows to
    # spare do not change it: the reduced axis is resident, so four warps on `M`
    # would leave the budget nothing to buy the rest of the line with.
    for num_m in (1, 8):
      for num_n in (147456, 589824):
        with self.subTest(num_m=num_m, num_n=num_n):
          p = self._plan((num_m, 128, num_n))
          self.assertEqual(p.block[1], 128)
          self.assertEqual(p.block[2] * self.F32, gpu_utils.CACHE_LINE_SIZE_BYTES)
          self.assertLessEqual(p.tile_regs(), 64)
          self.assertEqual(p.layout.to_mgpu().lane_dims, (-5, -2))

  def test_a_single_tile_is_a_single_cta(self):
    p = self._plan((8, 128))
    self.assertEqual((p.block, p.grid()), ((8, 128), (1, 1)))


if __name__ == '__main__':
  absltest.main()
