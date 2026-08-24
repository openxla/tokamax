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
"""Blocking and register layout for the Mosaic GPU normalization kernels.

Tiles go straight from GMEM into registers -- no SMEM, no pipeline.
The reduced axis is never blocked -- a whole row has to be resident for the
reduction -- so what is left to decide is where the warpgroup's 128 threads sit.
That is a *thread mapping*, and there is one per way the canonical shape can
place them; `plan` tries them in order and takes the first that fits, so the
mappings are also a preference order. Given `x` canonicalized to `(M, A)` or
`(M, A, N)`:

  - `_lanes_on_reduced`, for `(M, A)`. Nothing is minor to the reduced axis, so
    the lanes tile it and the reduction is a shuffle inside a warp.
  - `_lanes_on_minor`, for `(M, A, N)`. The lanes tile `N`, `A` is tiled by
    nothing and so lives in registers, and the reduction is a register loop.
  - `_warpgroup_on_minor`, when `M` is too short to give the four warps a row
    each: they come from `N` too, and `M` goes untiled.
  - `_minor_in_vector`, when `N` is too short for the lanes to tile: they go back
    on the reduced axis and `N` rides inside each lane.

In every one, the warps and the lanes take whole rows and contiguous runs
respectively, which is what makes the load and the store coalesce; and `M` -- the
axis with nothing riding on it -- is what `plan` shrinks to fit the register
budget.

Blocks tile their axis exactly, apart from `M`, whose last tile slides back to
end flush with the axis (see `Plan.tile_starts`). Nothing here can read or write
out of bounds: Mosaic has no masked GMEM load and no masked GMEM store, so every
access has to be provably in bounds, and sliding is what makes that true without
constraining `M`.

The reduced axis borrows the same trick where the lanes tile it, since there a
base tile of 32 elements has to divide it: it is read in several loads, the last
sliding back, and the duplicated elements are dropped from the reduction in
registers. See `Plan.a_tiles`.
"""

import dataclasses
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from tokamax._src import gpu_utils


# The layout's base tile is 4 rows tall, one per warp, so the axis feeding the
# warps -- and any block of it -- has to be a multiple of this.
_WARP_ROWS = 4

# The 32 lanes tile the tile's minor axis, so it has to be a multiple of this.
_LANES = 32

# Threads a layout has to place: four warps of 32 lanes.
_WARPGROUP = _WARP_ROWS * _LANES

# Tile elements per thread, in `float32` registers. The real footprint is a
# small multiple of this -- the reduction runs in `float32` while the loaded
# tile is still live, and temporaries stay live across the two passes -- so this
# is the knob to turn down if a kernel spills, not a hardware limit. 64 leaves
# room for two warpgroups per SM at 128 registers each.
_MAX_TILE_REGS = 64

WARPGROUP_SEMANTICS = plgpu.CompilerParams(
  lowering_semantics=plgpu.LoweringSemantics.Warpgroup
)


def ceil_div(a: int, b: int) -> int:
  return -(-a // b)


def divisors(n: int, cap: int, *, multiple_of: int = 1):
  """Yields divisors of `n` at most `cap` and a multiple of `multiple_of`, descending."""
  start = min(cap, n) // multiple_of * multiple_of
  for b in range(start, 0, -multiple_of):
    if n % b == 0:
      yield b


# The reduced axis of the canonical shape; see the module docstring.
REDUCE_AXIS = 1


def _vector_length(cols: int, bitwidth: int, *, lanes: int = _LANES) -> int:
  """The widest 16-byte-or-less vector `lanes` lanes can tile `cols` with.

  `cols` has to be a multiple of `lanes * vector_length` for some power of two
  `vector_length`, which every caller below has already made true.
  """
  vector_length = 128 // bitwidth  # 16-byte vectors.
  while cols % (lanes * vector_length):
    vector_length //= 2
  return vector_length


def _tiled(tiles, *, warp_dim: int, lane_dim: int):
  """A tiled layout whose vector is always the minormost tiled dimension."""
  return plgpu.Layout.TILED(
    plgpu.Tiling(tiles),
    warp_dims=(warp_dim,),
    lane_dims=(lane_dim,),
    vector_dim=-1,
  )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class _Mapping:
  """Where one thread mapping puts the warpgroup, and what it costs.

  Attributes:
    tail: The block of every axis but `M`. `M` is the axis the register budget is
      spent on, so `plan` is what picks it.
    m_multiple: `M` is blocked in multiples of this: the rows the warps take, or
      1 when the warps come from elsewhere and `M` is untiled.
    a_tile: How much of the reduced axis one load takes. The whole of it, unless
      the lanes tile it and their base tile does not divide it; see
      `Plan.a_tiles`.
    layout: The register layout of a tile, as loaded straight out of GMEM. Stated
      by hand because inference has no candidate this short; see the module
      docstring.
  """

  tail: tuple[int, ...]
  m_multiple: int
  a_tile: int
  layout: Any


def _lanes_on_reduced(shape, *, bitwidth, cap_n) -> _Mapping | str:
  """`(M, A)`: the lanes tile the reduced axis itself.

  Nothing is minor to `A`, so `A` takes both the lanes and the vector: a warp
  covers `32 * vector_length` contiguous elements, one unbroken transaction in
  either direction, and the reduction stays inside a warp as a shuffle. The warps
  take four rows of `M`.
  """
  del cap_n
  num_m, num_a, *rest = shape
  if rest:
    return 'the reduced axis is not the minor one'
  if num_m < _WARP_ROWS:
    return f'rows ({num_m}) must be at least {_WARP_ROWS}, one per warp'
  if num_a < _LANES:
    return f'the reduced axis ({num_a}) must be at least {_LANES}, one per lane'
  if num_a % _LANES:
    # No base tile divides the axis, so it takes several loads with the last
    # sliding back; see `Plan.a_tiles`. The vector is as wide as the *axis*
    # allows, rather than as wide as a tile of it allows, because it is the slid
    # offset -- a multiple of neither the tile nor the axis -- that has to stay
    # aligned.
    vec = _vector_length(num_a, bitwidth, lanes=1)
    while _LANES * vec > num_a:
      vec //= 2
    a_tile = _LANES * vec
  else:
    vec = _vector_length(num_a, bitwidth)
    a_tile = num_a
  return _Mapping(
    tail=(num_a,),
    m_multiple=_WARP_ROWS,
    a_tile=a_tile,
    layout=_tiled(
      ((_WARP_ROWS, _LANES * vec), (_WARP_ROWS, vec)), warp_dim=-2, lane_dim=-3
    ),
  )


def _lanes_on_minor(shape, *, bitwidth, cap_n) -> _Mapping | str:
  """`(M, A, N)`: the lanes tile the minor axis, a reduced row per thread.

  `A` is tiled by neither the lanes nor the warps, so a whole reduced row lives
  in one thread's registers and the reduction is a register loop with no data
  movement at all. Giving `A` to the warps instead -- which is what a base tile
  over the two minor axes alone would do -- makes the reduction cross warps, and
  so go through SMEM.

  The base tile spans all three axes, which is also what lets a statistic be
  broadcast back over the tile: `FragmentedArray.broadcast_in_dim` wants the base
  tile and the array to be of equal rank. Naming `A` in the tiling costs nothing,
  as `A` is never blocked.
  """
  num_m, num_a, *rest = shape
  if not rest:
    return 'there is no axis minor to the reduced one'
  if num_m < _WARP_ROWS:
    return f'rows ({num_m}) must be at least {_WARP_ROWS}, one per warp'
  (num_n,) = rest
  block_n = next(divisors(num_n, cap_n, multiple_of=_LANES), None)
  if block_n is None:
    return (
      f'the minor axis ({num_n}) has no divisor that is a multiple of {_LANES}'
      f' and at most {cap_n}'
    )
  vec = _vector_length(block_n, bitwidth)
  return _Mapping(
    tail=(num_a, block_n),
    m_multiple=_WARP_ROWS,
    a_tile=num_a,
    layout=_tiled(
      ((_WARP_ROWS, num_a, _LANES * vec), (vec,)), warp_dim=-4, lane_dim=-2
    ),
  )


def _warpgroup_on_minor(shape, *, bitwidth, cap_n) -> _Mapping | str:
  """`(M, A, N)`: every thread comes from the minor axis, so `M` goes untiled.

  For an `M` too short to give the four warps a row each -- or a tile that only
  fits the register budget with fewer rows than that, since an untiled `M` may be
  blocked as finely as one row. The warps take four consecutive
  `32 * vector_length` runs of the minor axis, so the load is one 4x longer
  unbroken transaction; the price is that the minor axis has to be blocked in
  multiples of the whole warpgroup, which is a floor on the block rather than a
  cap, and so overrides `cap_n`.
  """
  num_a, *rest = shape[REDUCE_AXIS:]
  if not rest:
    return 'there is no axis minor to the reduced one'
  (num_n,) = rest
  block_n = next(
    divisors(num_n, max(cap_n, _WARPGROUP), multiple_of=_WARPGROUP), None
  )
  if block_n is None:
    return (
      f'the minor axis ({num_n}) has no divisor that is a multiple of'
      f' {_WARPGROUP}'
    )
  vec = _vector_length(block_n, bitwidth, lanes=_WARPGROUP)
  return _Mapping(
    tail=(num_a, block_n),
    m_multiple=1,
    a_tile=num_a,
    layout=_tiled(
      ((1, num_a, _WARPGROUP * vec), (_LANES * vec,), (vec,)),
      warp_dim=-3,
      lane_dim=-2,
    ),
  )


def _minor_in_vector(shape, *, bitwidth, cap_n) -> _Mapping | str:
  """`(M, A, N)`: the minor axis is too short for the lanes, so it rides in them.

  A block of the minor axis has to divide it exactly, so the 32 lanes need 32
  elements of it to tile. Below that they go back on the reduced axis, as in the
  contiguous case, and each lane takes the whole minor axis instead. The two are
  adjacent in memory, so a warp still covers `32 * N` contiguous elements, and
  the reduction is a shuffle again.
  """
  del cap_n
  num_m, num_a, *rest = shape
  if not rest:
    return 'there is no axis minor to the reduced one'
  if num_m < _WARP_ROWS:
    return f'rows ({num_m}) must be at least {_WARP_ROWS}, one per warp'
  if num_a < _LANES:
    return f'the reduced axis ({num_a}) must be at least {_LANES}, one per lane'
  (num_n,) = rest
  base = (_WARP_ROWS, _LANES, num_n)
  vec = _vector_length(num_n, bitwidth, lanes=1)
  if vec == num_n:
    # The minor axis is one whole vector, so there is nothing to split it by: a
    # second tile of `(num_n,)` would only add a size-1 dimension, which is not
    # a canonical tiling.
    layout = _tiled((base,), warp_dim=-3, lane_dim=-2)
  else:
    layout = _tiled((base, (vec,)), warp_dim=-4, lane_dim=-3)
  # The vector comes from `N` here, so the lanes take a bare `_LANES` of the
  # reduced axis and a ragged one is covered by several loads; see `Plan.a_tiles`.
  return _Mapping(
    tail=(num_a, num_n),
    m_multiple=_WARP_ROWS,
    a_tile=num_a if num_a % _LANES == 0 else _LANES,
    layout=layout,
  )


# In preference order; `plan` takes the first that fits.
_MAPPINGS = (
  _lanes_on_reduced,
  _lanes_on_minor,
  _warpgroup_on_minor,
  _minor_in_vector,
)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Plan:
  """How one canonical normalization is spread over the device.

  One CTA handles one tile, so `grid` is both the blocking and the launch.

  Attributes:
    shape: The canonical shape of `x`, `(M, A)` or `(M, A, N)`.
    block: One tile of `shape`. The reduced axis is not blocked, so
      `block[REDUCE_AXIS] == shape[REDUCE_AXIS]`. Every axis but `M` is tiled
      exactly; `M` need only be at least `block[0]` (see `tile_starts`).
    a_tile: How much of the reduced axis one load takes; see `a_tiles`. The whole
      of it, unless the lanes tile it and their base tile does not divide it.
    layout: The register layout of one load, from the thread mapping that chose
      the block; see the module docstring.
  """

  shape: tuple[int, ...]
  block: tuple[int, ...]
  a_tile: int
  layout: Any

  @property
  def grid(self) -> tuple[int, ...]:
    """Blocks along each axis; 1 along the reduced axis, which is not blocked."""
    return tuple(
      ceil_div(s, b) for s, b in zip(self.shape, self.block, strict=True)
    )

  @property
  def grid_names(self) -> tuple[str, ...]:
    """A `plgpu.kernel` grid name per axis, in the order of `shape`."""
    return ('m', 'a', 'n')[: len(self.shape)]

  @property
  def a_tiles(self) -> tuple[tuple[int, int], ...]:
    """`(offset, duplicates)` per load along the reduced axis.

    A base tile has to divide the block exactly, so when `a_tile` does not divide
    the reduced axis the last load slides back to end flush with it -- the same
    trick `M`'s last tile uses, and for the same reason: Mosaic has no masked GMEM
    access, so every read has to be provably in bounds. `duplicates` is how many
    of a load's leading elements its predecessor already covered. A reduction has
    to drop them, or it would count them twice; a store may simply rewrite them,
    as they take the same value.

    Duplicated reads cost no bandwidth -- the previous load pulled those lines
    into L1 a moment earlier -- but they do cost registers, which `tile_regs`
    counts.
    """
    num_a = self.shape[REDUCE_AXIS]
    offsets = [
      min(i * self.a_tile, num_a - self.a_tile)
      for i in range(ceil_div(num_a, self.a_tile))
    ]
    return tuple(
      (offset, max(0, previous + self.a_tile - offset))
      for previous, offset in zip([-self.a_tile, *offsets], offsets)
    )

  def tile_regs(self) -> int:
    """Tile elements each thread holds, in `float32` registers.

    The whole reduced axis is resident, rounded up to whole `a_tile`s. The layout
    places a whole warpgroup and every axis it tiles is a multiple of its share
    of it, so this divides exactly.
    """
    rows, _, *minor = self.block
    resident = rows * len(self.a_tiles) * self.a_tile * math.prod(minor)
    return resident // _WARPGROUP

  def tile_starts(self, indices) -> tuple[jax.Array, ...]:
    """Returns a tile's element offset per axis, given its block index per axis.

    The last tile along `M` slides back to end flush with `M` rather than
    overhang it, so `block[0]` need not divide `M`. Every other axis is tiled
    exactly, so only `M` can overhang. Rows are normalized independently, so the
    rows an overlapping tile revisits are recomputed to the same values.

    Sliding, rather than reading out of bounds and masking, is forced: Mosaic has
    no masked GMEM load or store. `M` is also the only axis it is safe along --
    it shifts the address by whole rows, so every transfer keeps the 16-byte
    alignment its vector width demands, which a slide along a tiled axis would
    not.
    """
    starts = [i * b for i, b in zip(indices, self.block, strict=True)]
    starts[0] = jnp.minimum(starts[0], self.shape[0] - self.block[0])
    return tuple(starts)

  def out_index(self) -> tuple[Any, ...]:
    """Returns the slice of `x` this CTA covers, as one `pl.ds` per axis."""
    starts = self.tile_starts(map(jax.lax.axis_index, self.grid_names))
    return tuple(pl.ds(s, b) for s, b in zip(starts, self.block, strict=True))


def drop_reduced(xs: tuple[Any, ...]) -> tuple[Any, ...]:
  """Drops the reduced axis' entry, mapping `x` to the statistics beside it.

  Takes a shape or an index tuple: the statistics have one entry per axis of `x`
  except the reduced one, which they are the reduction over.
  """
  return xs[:REDUCE_AXIS] + xs[REDUCE_AXIS + 1 :]


def plan(
  shape: tuple[int, ...], itemsize: int, *, block_m: int, block_n: int | None
) -> Plan:
  """Plans a normalization of a canonical `(M, A)` or `(M, A, N)` array.

  Args:
    shape: The canonical shape of `x`; see the module docstring.
    itemsize: Bytes per element of `x`.
    block_m: Upper bound on the rows of `M` per tile.
    block_n: Upper bound on the columns of `N` per tile.

  Each thread mapping in `_MAPPINGS` says what it needs of `shape` and how the
  axes other than `M` are blocked; all that is left here is `M`, which is blocked
  as coarsely as the register budget allows. A mapping that cannot fit the budget
  even at its smallest is passed over for the next one.

  Raises:
    NotImplementedError: if no mapping fits, reporting what each one wanted. This
    allows us to fall back to another implementation.
  """
  shape = tuple(shape)
  num_m = shape[0]
  cap_n = (
    gpu_utils.CACHE_LINE_SIZE_BYTES // itemsize if block_n is None else block_n
  )
  reasons = []
  for mapping in _MAPPINGS:
    name = mapping.__name__.lstrip('_')
    fit = mapping(shape, bitwidth=itemsize * 8, cap_n=cap_n)
    if isinstance(fit, str):
      reasons.append(f'{name}: {fit}')
      continue
    # A whole reduced row has to be resident, so a long one is paid for in rows.
    # `tile_regs` is linear in them, so the most that fit is arithmetic rather
    # than a search. One block's worth is always tried, even if `block_m` asks
    # for less than that.
    step = fit.m_multiple
    # The reduced axis is resident rounded up to whole loads of it; see
    # `Plan.a_tiles`.
    resident = ceil_div(fit.tail[0], fit.a_tile) * fit.a_tile
    row_regs = resident * math.prod(fit.tail[1:])
    budget_rows = _MAX_TILE_REGS * _WARPGROUP // row_regs
    rows = min(max(step, min(block_m, num_m)), budget_rows) // step * step
    if rows == 0:
      reasons.append(
        f'{name}: its smallest tile, {(step, *fit.tail)}, needs'
        f' {step * row_regs // _WARPGROUP} registers per thread, over the budget'
        f' of {_MAX_TILE_REGS}'
      )
      continue
    return Plan(
      shape=shape,
      block=(rows, *fit.tail),
      a_tile=fit.a_tile,
      layout=fit.layout,
    )

  raise NotImplementedError(
    f'No thread mapping fits {shape}: ' + '; '.join(reasons) + '.'
  )


def with_usable_block_m(config):
  """Raises a borrowed Triton config's `block_m` to something this kernel can use.

  A block has to be a multiple of 4 rows, one per warp, and `plan` would raise
  the cap to that anyway; doing it here keeps the config honest.
  """
  return dataclasses.replace(config, block_m=max(_WARP_ROWS, config.block_m))
