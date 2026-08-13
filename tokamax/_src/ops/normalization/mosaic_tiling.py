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
"""Blocking, grid and register layouts shared by the Mosaic GPU norm kernels.

The forward and the VJP walk the same canonical `(M, A, B)` form, want the same
coalesced register layout, and hit the same grid limits. Only the kernel bodies
differ, so everything else lives here.
"""

from collections.abc import Callable
import dataclasses
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp


# CUDA allows 2**31-1 blocks in gridDim.x but only 65535 in .y and .z.
_MAX_GRID_DIM = 65535

LANE_SEMANTICS = plgpu.CompilerParams(
  lowering_semantics=plgpu.LoweringSemantics.Lane
)


def _ceil_div(a: int, b: int) -> int:
  return -(-a // b)


def largest_divisor(n: int, cap: int) -> int:
  "Returns the largest divisor of `n` that is at most `cap`"
  return next(b for b in range(min(cap, n), 0, -1) if n % b == 0)


def warp_aligned_block(rows: int, cap: int) -> int:
  """Returns the largest multiple of 4 at most `min(cap, rows)`, else that.

  `make_layout` takes the four warps from `M` only when the block is a multiple
  of 4 rows, which is the arrangement to prefer. Blocks need not divide `rows`
  -- the last one is slid back to fit, see `Plan.indices` -- so this is just a
  rounding, except when there are not 4 rows to round to and the warps have to
  come from the reduced axis instead.
  """
  return min(cap, rows) // 4 * 4 or min(cap, rows)


def _largest_pow2(cap: int, ok: Callable[[int], bool]) -> int | None:
  """Returns the largest power of two `<= cap` satisfying `ok`, or `None`."""
  v = 1 << (cap.bit_length() - 1)
  while not ok(v):
    if v == 1:
      return None
    v //= 2
  return v


def make_layout(block_slow: int, block_fast: int, itemsize: int):
  """Register layout for a 2D `(slow, fast)` tile, `fast` being contiguous.

  Lanes and the vector always come from `fast`, so a warp covers `32 * vector`
  contiguous elements -- one unbroken memory transaction. What varies is where
  the four warps come from, and since a layout has to place all 128 threads of
  the warpgroup, they have to come from somewhere:

  - `slow`, when it has 4 rows to give them. A row then lives inside a single
    warp, so a reduction along `fast` is a lane shuffle.
  - `fast` otherwise, spreading one row over the whole warpgroup. Still
    perfectly coalesced, but a reduction along `fast` now crosses warps and
    takes the SMEM scratch path.
  """
  max_vec = max(1, 16 // itemsize)  # 16-byte (128-bit) register vectors.
  if block_slow % 4 == 0:
    v = _largest_pow2(max_vec, lambda v: block_fast % (32 * v) == 0)
    if v is None:
      raise NotImplementedError(
        f'Contiguous axis ({block_fast}) must be a multiple of 32 so the lanes'
        ' cover it.'
      )
    return plgpu.Layout.TILED(
      plgpu.Tiling(((4, 32 * v), (4, v))), (-2,), (-3,), -1
    )

  v = _largest_pow2(max_vec, lambda v: block_fast % (128 * v) == 0)
  if v is None:
    raise NotImplementedError(
      f'Contiguous axis ({block_fast}) must be a multiple of 128 so the whole'
      f' warpgroup covers it, as {block_slow} rows cannot feed the warps.'
    )
  return plgpu.Layout.TILED(
    plgpu.Tiling(((1, 128 * v), (32 * v,), (v,))), (-3,), (-2,), -1
  )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Plan:
  """How one canonical `(M, A, B)` normalization is spread over the device.

  `A`, the reduced axis, is never blocked: a whole row lives in registers, so
  the reduction needs no cross-block communication. `M` and `B` are blocked and
  the rest of the extent goes to the grid.

  A tile in registers is always 2D: `A` and whichever of `M`/`B` is blocked,
  the other one being indexed by a scalar and so dropping out.

  Blocks need not tile `M` exactly. Mosaic has no masked GMEM load or store, so
  the last block instead slides back to end at the last row, overlapping its
  predecessor. Rows are normalized independently, so the overlap is recomputed
  to the same values and rewritten with them -- harmless for everything except a
  sum over rows, which would count them twice; `drop_duplicate_rows` is how the
  VJP's partials avoid that.

  Attributes:
    num_m: Rows of `M` in the whole array.
    block_m: Rows of `M` per block.
    block_b: Elements of `B` per block.
    overlaps: Whether the blocks along `M` overrun `num_m`, so that the last one
      slides back and repeats rows.
    x_shape: Shape the `x`-like refs are given, rank-reduced to the case so the
      kernel never indexes a rank away.
    stat_shape: Shape of the `mean`/`rstddev` refs.
    dparam_shape: Shape of the per-block `dscale`/`doffset` partials, one `A`-
      indexed row per block, to be summed over the leading two axes.
    layout: Register layout of an `x` tile.
    reduce_axis: Axis of the 2D tile holding `A`.
    grid: The `plgpu.kernel` grid.
    grid_names: Names of `grid`'s axes. `'m'`/`'b'` index `M`/`B`; a leading
      `'mo'` appears only when `M` needs two grid axes (see `plan`). They are
      read by name, so a `vmap` prepending an axis of its own is harmless.
    split_m: Stride of the `'mo'` axis, i.e. the size of the `'m'` axis.
  """

  num_m: int
  block_m: int
  block_b: int
  overlaps: bool
  x_shape: tuple[int, ...]
  stat_shape: tuple[int, ...]
  dparam_shape: tuple[int, int, int]
  layout: Any
  reduce_axis: int
  grid: tuple[int, ...]
  grid_names: tuple[str, ...]
  split_m: int

  @property
  def stat_axis(self) -> int:
    return 1 - self.reduce_axis

  def indices(self):
    """Returns how the current program indexes each of the refs.

    In order:
     - A `x`-shaped ref, giving this block's tile
     - A `mean`/`rstddev`-shaped ref
     - A `dscale`/`doffset` partials ref
     - How many leading rows of the tile this block shares with its
       predecessor, for `drop_duplicate_rows`; a constant `0` unless the blocks
       overrun `M`.
    """
    pid_m = jax.lax.axis_index('m')
    if 'mo' in self.grid_names:
      pid_m += jax.lax.axis_index('mo') * self.split_m
    pid_b = jax.lax.axis_index('b')
    dparam = (pid_m, pid_b)

    start_m, skip = pid_m * self.block_m, 0
    if self.overlaps:
      # Slide the overrunning block back to end at the last row. A block past
      # the end entirely -- which the `'mo'` split can produce -- lands on the
      # last one, so `skip` reaches `block_m` and it contributes nothing.
      start_m = jnp.minimum(start_m, self.num_m - self.block_m)
      skip = pid_m * self.block_m - start_m

    if len(self.x_shape) == 2:
      m_slice = pl.ds(start_m, self.block_m)
      return m_slice, m_slice, dparam, skip
    b_slice = pl.ds(pid_b * self.block_b, self.block_b)
    return (start_m, slice(None), b_slice), (start_m, b_slice), dparam, skip

  def drop_duplicate_rows(self, a: jax.Array, skip) -> jax.Array:
    """Zeros the rows of a tile that its predecessor already accounted for.

    Only for values about to be summed over `M`; anything else is idempotent
    under the overlap and wants the real rows.
    """
    if not self.overlaps:
      return a
    if self.block_m == 1:
      # `M` is indexed by a scalar and is not part of the tile, so a repeated
      # block repeats all of it.
      return jnp.where(skip == 0, a, 0.0)
    # `M` is the tile's slow axis, so it is the one a per-row statistic keeps.
    rows = jax.lax.broadcasted_iota(jnp.int32, a.shape, self.stat_axis)
    return jnp.where(plgpu.layout_cast(rows, self.layout) >= skip, a, 0.0)

  def _spread(self, a: jax.Array, shape, dims) -> jax.Array:
    return plgpu.layout_cast(
      jax.lax.broadcast_in_dim(a, shape, dims), self.layout
    )

  def bcast(self, a: jax.Array, shape) -> jax.Array:
    """Broadcasts a per-row statistic back over a tile of `shape`."""
    return self._spread(a, shape, (self.stat_axis,))

  def read_param(self, ref, shape) -> jax.Array:
    """Reads a 1D param, laid out to match a tile of `shape` along `A`."""
    layout = self.layout.reduce((self.stat_axis,))
    p = plgpu.load(ref, layout=layout, optimized=False)
    return self._spread(p.astype(jnp.float32), shape, (self.reduce_axis,))


def plan(
  x_shape: tuple[int, int, int],
  itemsize: int,
  *,
  block_m: int,
  block_b: int,
  max_regs: int = 256,
) -> Plan:
  """Plans a canonical `(M, A, B)` normalization.

  Args:
    x_shape: The canonical `(M, A, B)` shape, `A` being the reduced axis.
    itemsize: Bytes per element of `x`.
    block_m: Upper bound on `block_m`; the largest warp-aligned block at or
      below it is used. It need not divide `M`; see `Plan`.
    block_b: Upper bound on `block_b`; the largest divisor at or below it is
      used, as `B` is the contiguous axis and a block sliding back off a
      multiple of the load's vector width would misalign it.
    max_regs: Register budget per thread, in elements of a single block. The
      real footprint is a small multiple of this -- reductions run in `float32`
      and the kernels keep temporaries live -- so this is a knob to turn if a
      kernel spills, not a hardware limit.
  """
  num_m, num_a, num_b = x_shape
  if num_b == 1:
    # `A` is contiguous *and* reduced, so the lanes land on it. Preferably the
    # warps come from `M`, which keeps the reduction inside a warp -- but that
    # needs 4 rows to give them, so a short `M` spreads one row over the whole
    # warpgroup instead and pays a cross-warp reduction.
    block_m = warp_aligned_block(num_m, block_m)
    block_b = 1
    tile_shape, stat_shape, reduce_axis = (num_m, num_a), (num_m,), 1
    layout = make_layout(block_m, num_a, itemsize)
  else:
    # `B` is contiguous, so `A` is the slow axis: it feeds the warps when it is
    # long enough, and the reduction along it then takes the scratch path. `M`
    # goes entirely to the grid -- it adds nothing to coalescing and only
    # inflates the register footprint -- so it is indexed by a scalar and the
    # tile is 2D even though the ref is 3D.
    block_m, block_b = 1, largest_divisor(num_b, block_b)
    tile_shape, stat_shape, reduce_axis = (num_m, num_a, num_b), (num_m, num_b), 0
    layout = make_layout(num_a, block_b, itemsize)

  if block_m * block_b == 1:
    # Mosaic's scalar path cannot yet write a tiled value to a GMEM ref.
    raise NotImplementedError(
      f'A block of ({block_m}, {num_a}, {block_b}) leaves a single-element'
      ' statistic, which cannot be stored.'
    )

  if (regs := block_m * num_a * block_b // 128) > max_regs:
    raise NotImplementedError(
      f'Block ({block_m}, {num_a}, {block_b}) puts {regs} elements in each'
      f' thread, over the budget of {max_regs}, so it will spill.'
    )

  # CUDA caps gridDim.y/.z at 65535 (only .x is 2**31-1) and Mosaic does not
  # map the leading axis to .x, so too many rows aborts the launch with
  # `cuLaunchKernelEx: invalid argument`. Splitting the row grid over two axes
  # need not be exact either: a leftover block lands back on the last one and,
  # like any other overlap, contributes nothing new.
  grid_m, grid_b = _ceil_div(num_m, block_m), num_b // block_b
  split_m, grid, grid_names = 1, (grid_m, grid_b), ('m', 'b')
  if grid_m > _MAX_GRID_DIM:
    outer = _ceil_div(grid_m, _MAX_GRID_DIM)
    split_m = _ceil_div(grid_m, outer)
    grid_m, grid = outer * split_m, (outer, split_m, grid_b)
    grid_names = ('mo', 'm', 'b')

  return Plan(
    num_m=num_m,
    overlaps=grid_m * block_m > num_m,
    block_m=block_m,
    block_b=block_b,
    x_shape=tile_shape,
    stat_shape=stat_shape,
    dparam_shape=(grid_m, grid_b, num_a),
    layout=layout,
    reduce_axis=reduce_axis,
    grid=grid,
    grid_names=grid_names,
    split_m=split_m,
  )


def with_usable_block_m(config):
  """Raises a borrowed Triton config's `block_m` to something this kernel can use.
  A `block_m` below 4 leaves nothing to give the warps, so `make_layout` takes
  them from the reduced axis instead and reduces across warps through SMEM when
  a lane shuffle would have done.
  """
  return dataclasses.replace(config, block_m=max(4, config.block_m))
