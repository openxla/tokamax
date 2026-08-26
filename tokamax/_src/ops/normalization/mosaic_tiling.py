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

Tiles go straight from GMEM into registers -- no SMEM, no pipeline. The reduced
axis is never blocked -- a whole row has to be resident for the reduction -- so
what is left to decide is where the warpgroup's 128 threads sit. Given `x`
canonicalized to `(M, A)` or `(M, A, N)`, the four warps come from `M`, from `N`,
or from both, and the 32 lanes then split between `A` and `N`:

  - `_warps_on_m`. The warps take four rows of `M` and the 32 lanes are split
    between the reduced axis and the minor one; `_lane_split` picks the split.
    With no `N` to take any, all 32 go on `A` and the reduction is a shuffle
    inside a warp. Needs `M >= 4`, and is the only mapping a rank-2 shape has.
  - `_lanes_split_under_warps_on_n`. `N` feeds some of the warps, so `M` may be
    blocked as finely as one row, and `N` is tiled twice: once for the warps and
    again for the lanes, which still take a sector-wide run each.
  - `_all_lanes_on_reduced`. The warps as above, but every lane back on `A`, so a
    lane group reads a bare vector and the base tile costs a fraction of the
    registers. What a tile over the register budget falls back on.

`_mappings` lists every one a shape admits and `plan` blocks each, taking the
tile that reads best: the widest requests, then the most of a cache line covered
by them (`_score`). Where the warps go is not a free choice between equals. The
smallest tile costs the same wherever they come from -- `m_multiple * n_multiple`
is four runs either way -- but four warps on `N` sit on adjacent runs and so
span a whole line, where four on `M` span one run of four rows and have to buy
the rest of the line with registers. So a reduced axis long enough to leave the
budget nothing to buy columns with gives its warps to `N`, rows to spare or not.

No mapping ever reduces across warps, so none needs SMEM: a warp always
holds whole rows of the statistic it is computing, and what it does hold is
reduced by a register loop and, where the lanes tile `A`, a shuffle.

Every axis is tiled exactly: a block divides its axis, and the lanes' share of
the reduced axis divides that. Nothing here can read or write out of bounds, and
that is forced rather than chosen -- Mosaic has no masked GMEM load and no masked
GMEM store, so every access has to be provably in bounds. Exact tiling is the
whole of how that is met, which is why a shape it cannot be met for is declined
(`_decline`) for the caller to fall back on: an axis read in overlapping pieces
would buy those shapes at the price of a mask on every reduction, and of registers
spent on elements the load already had.
"""

import dataclasses
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from tokamax._src import gpu_utils


# The layout's base tile is 4 rows tall, one per warp, so the axis feeding the
# warps -- and any block of it -- has to be a multiple of this.
_WARP_ROWS = 4

# Lanes in a warp. Whatever the mapping, the dimensions the lane dims name have
# to multiply to exactly this.
_LANES = 32

# Threads a layout has to place: four warps of 32 lanes.
_WARPGROUP = _WARP_ROWS * _LANES

# The memory system moves 32 bytes at a time, so this many bits is the least a
# run of lanes should read contiguously; below it a load fetches bytes it will
# not use. See `_lane_split`.
_SECTOR_BITS = 256

# Tile elements per thread, in `float32` registers. The real footprint is a
# small multiple of this -- the reduction runs in `float32` while the loaded
# tile is still live, and temporaries stay live across the two passes -- so this
# is the knob to turn down if a kernel spills, not a hardware limit. 64 leaves
# room for two warpgroups per SM at 128 registers each.
_MAX_TILE_REGS = 64

WARPGROUP_SEMANTICS = plgpu.CompilerParams(
  lowering_semantics=plgpu.LoweringSemantics.Warpgroup
)


def divisors(n: int, cap: int, *, multiple_of: int = 1):
  """Yields divisors of `n` at most `cap` and a multiple of `multiple_of`, descending."""
  start = min(cap, n) // multiple_of * multiple_of
  for b in range(start, 0, -multiple_of):
    if n % b == 0:
      yield b


# The reduced axis of the canonical shape; see the module docstring.
REDUCE_AXIS = 1


def _pow2_part(n: int) -> int:
  """The largest power of two dividing `n`."""
  return n & -n


def _vector_length(cols: int, bitwidth: int, *, lanes: int = _LANES) -> int:
  """The widest 16-byte-or-less vector `lanes` lanes can tile `cols` with."""
  vector_length = 128 // bitwidth  # 16-byte vectors.
  while cols % (lanes * vector_length):
    vector_length //= 2
  return vector_length


def _tiled(tiles, *, warp_dims, lane_dims):
  """A tiled layout whose vector is always the minormost tiled dimension."""
  return plgpu.Layout.TILED(
    plgpu.Tiling(tiles),
    warp_dims=tuple(warp_dims),
    lane_dims=tuple(lane_dims),
    vector_dim=-1,
  )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class _Mapping:
  """Where one thread mapping puts the warpgroup, and what it costs.

  Attributes:
    m_multiple: `M` is blocked in multiples of this: the rows the warps take, or
      fewer when some of the warps come from `N` instead.
    n_multiple: `N` is blocked in multiples of this, the base tile's extent along
      it, and so it is also the least a block of it can be. `None` when there is
      no `N`. Both multiples are floors rather than sizes: what a mapping settles
      is where the threads go, and `plan` spends the register budget on the
      blocks.
    run: Elements a lane group reads contiguously: the run of `_lane_split`, or a
      bare vector where every lane is on `A`. This is what one request covers, so
      it is how many requests a cache line costs; `plan` prefers the mapping
      whose run is a whole sector. See `_score`.
    layout: The register layout of a tile, as loaded straight out of GMEM. Stated
      by hand because inference has no candidate this short; see the module
      docstring.
  """

  m_multiple: int
  n_multiple: int | None
  run: int
  layout: Any


def _min_tile_regs(mapping: _Mapping, num_a: int) -> int:
  """Registers per thread of the smallest tile the mapping admits.

  Both blocks at their floor. `plan` can only ever grow them from here, so a
  mapping over the budget at this size has nothing left to give.
  """
  cols = mapping.n_multiple or 1
  return mapping.m_multiple * num_a * cols // _WARPGROUP


def _lane_split(num_a: int, num_n: int, bitwidth: int) -> tuple[int, int, int]:
  """Splits the 32 lanes between the reduced axis and the minor one.

  Returns `(a_lanes, n_lanes, vec)`, where `a_lanes * n_lanes == 32`: `a_lanes`
  lanes tile `A` and the other `n_lanes` tile `N`, each holding a `vec`-long
  vector of it.

  Consecutive lanes run along `N`, so a group of `n_lanes` covers `n_lanes * vec`
  contiguous elements -- call it the run -- and the warp reads `a_lanes` such runs
  at a stride of `N`. Registers per thread work out to

      (block_m / 4) * (A / a_lanes) * (block_n / run) * vec

  and with `block_n` at its smallest, one run, that collapses to
  `(block_m / 4) * A * run / 32`: the cost depends on the run alone, not on
  how the run is divided into lanes and vector. So the run is the whole trade --
  wider is better coalesced and dearer in registers -- and one 32-byte sector is
  where it should sit, the shortest run that wastes no bandwidth.

  That leaves `vec` free, and the widest one gives the fewest, widest load
  instructions. It is only bounded away from the widest when `a_lanes` has to
  shrink to divide `A`, which comes first: the lanes tile the reduced axis
  exactly or the mapping is declined, so a split that does not divide it is no
  split at all.
  """
  max_vec = 128 // bitwidth
  # A run cannot outrun the sector, nor the alignment `N` actually has: the run
  # has to divide `N` for a block of it to.
  run = min(_SECTOR_BITS // bitwidth, _pow2_part(num_n))
  splits = []
  vec = min(max_vec, run)
  while vec >= 1:
    n_lanes = run // vec
    if _LANES % n_lanes == 0:
      splits.append((_LANES // n_lanes, n_lanes, vec))
    vec //= 2
  # Widest vector first, so the first split that divides `A` is also the one with
  # the fewest, widest loads. `run` is a power of two at most `_LANES`, so the
  # narrowest vector always leaves a lane count that divides the warp and the
  # list is never empty; falling back on it puts the fewest lanes on `A`, which is
  # the likeliest to divide it, and the caller declines if even that does not.
  return next((s for s in splits if num_a % s[0] == 0), splits[-1])


def _lanes_on_reduced_layout(vec: int):
  """`(M, A)`: all 32 lanes tile the reduced axis, which is also the vector.

  Nothing is minor to `A`, so a warp covers `32 * vec` contiguous elements -- one
  unbroken transaction in either direction -- and the reduction is a shuffle.
  """
  # Tiled: `(_WARP_ROWS, _LANES, vec)`.
  return _tiled(
    ((_WARP_ROWS, _LANES * vec), (vec,)), warp_dims=(-3,), lane_dims=(-2,)
  )


def _warps_on_m(shape, *, bitwidth) -> _Mapping | None:
  """`M >= 4`: the warps take four rows and the lanes split between `A` and `N`.

  `None` when the lanes have no element of the reduced axis each; `_decline` says
  as much when nothing else fits either.
  """
  _, num_a, *rest = shape
  if not rest:
    # No `N`, so all 32 lanes go on `A` and the vector comes from it too. They
    # tile the axis whole, in one load, so 32 of it is both the least there can
    # be and the multiple it has to come in.
    if num_a % _LANES:
      return None
    return _Mapping(
      m_multiple=_WARP_ROWS,
      n_multiple=None,
      # Nothing is minor to `A`, so the whole load is one contiguous run.
      run=num_a,
      layout=_lanes_on_reduced_layout(_vector_length(num_a, bitwidth)),
    )

  (num_n,) = rest
  a_lanes, n_lanes, vec = _lane_split(num_a, num_n, bitwidth)

  if n_lanes == 1:
    # `N` is too short, or too oddly shaped, for any lane to be spared for it:
    # all 32 go back on `A` and the whole of `N` rides inside each lane. The two
    # are adjacent in memory, so a warp still covers `32 * N` contiguous
    # elements, and the reduction is a shuffle again.
    if num_a % _LANES:
      return None
    base = (_WARP_ROWS, _LANES, num_n)
    vec = _vector_length(num_n, bitwidth, lanes=1)
    if vec == num_n:
      # `N` is one whole vector, so there is nothing to split it by: a second
      # tile of `(num_n,)` would only add a size-1 dimension, which is not a
      # canonical tiling.
      layout = _tiled((base,), warp_dims=(-3,), lane_dims=(-2,))
    else:
      layout = _tiled((base, (vec,)), warp_dims=(-4,), lane_dims=(-3,))
    return _Mapping(
      m_multiple=_WARP_ROWS,
      # The base tile spans the whole of `N`, so `N` is not blocked at all.
      n_multiple=num_n,
      # `N` rides whole inside a lane, and is contiguous with `A` besides.
      run=num_n,
      layout=layout,
    )

  # `_lane_split` returns the widest-vectored split that divides `A`, so one that
  # does not divide it means none of them would have.
  if num_a % a_lanes:
    return None
  run = n_lanes * vec
  sub_a = num_a // a_lanes
  if sub_a > 1:
    # Tiled: `(m_tile, a_lanes, n_lanes, sub_a, vec)`.
    layout = _tiled(
      ((_WARP_ROWS, num_a, run), (sub_a, vec)),
      warp_dims=(-5,),
      lane_dims=(-4, -3),
    )
  else:
    # One reduced element per lane, so there is nothing left of `A` to split:
    # a second tile of `(1, vec)` is not canonical. Tiled:
    # `(m_tile, a_lanes, n_lanes, vec)`.
    layout = _tiled(
      ((_WARP_ROWS, num_a, run), (vec,)), warp_dims=(-4,), lane_dims=(-3, -2)
    )
  return _Mapping(
    m_multiple=_WARP_ROWS, n_multiple=run, run=run, layout=layout
  )


def _all_lanes_on_reduced(
  num_a, num_n, *, bitwidth, m_warps, n_warps
) -> _Mapping | None:
  """Every lane on `A`, with only the warps taking any of `N`.

  A lane's `vec` elements are contiguous but its neighbour's are a whole `N`
  away, so a warp asks for 32 sectors where four would do. The four warps sit on
  adjacent runs of `N` and issue together, so L1 serves all but the first of each
  sector and the bandwidth is not actually wasted -- what is, is the transactions
  it takes to ask. That buys the cheapest tile there is, though: a lane group's
  run is a bare `vec`, half a sector, and the run is what registers go as. So
  this is what a sector-wide run over the budget falls back on.

  `None` unless all 32 lanes have a whole share of the reduced axis: they tile it
  between them, so 32 has to divide it.
  """
  if num_a % _LANES:
    return None
  vec = _vector_length(num_n, bitwidth, lanes=n_warps)
  run = n_warps * vec
  sub_a = num_a // _LANES
  # A leading 1 is only canonical in the base tile, so an `M` feeding no warp of
  # its own is left out of the warp dims rather than named as a size-1 one.
  m_dims = (-5,) if m_warps > 1 else ()
  if sub_a > 1:
    # Tiled: `(m_warps, _LANES, n_warps, sub_a, vec)`.
    layout = _tiled(
      ((m_warps, num_a, run), (sub_a, vec)),
      warp_dims=(*m_dims, -3),
      lane_dims=(-4,),
    )
  else:
    # Tiled: `(m_warps, _LANES, n_warps, vec)`.
    layout = _tiled(
      ((m_warps, num_a, run), (vec,)),
      warp_dims=(*(d + 1 for d in m_dims), -2),
      lane_dims=(-3,),
    )
  return _Mapping(
    m_multiple=m_warps,
    n_multiple=run,
    # The warps read adjacent runs, but a lane group reads a bare vector.
    run=vec,
    layout=layout,
  )


def _lanes_split_under_warps_on_n(
  num_a, num_n, *, bitwidth, m_warps, n_warps
) -> _Mapping | None:
  """`M < 4`, with `N` tiled twice: once for the warps and again for the lanes.

  The warps take `n_warps` runs of `N` and the lanes still split between `A` and
  `N`, so a lane group covers a whole sector and the transactions come back down
  to what `_warps_on_m` gets. Nothing is replicated: every thread of the
  warpgroup holds a distinct part of the tile.

  The price is `N`'s divisibility. A block of it has to be a multiple of
  `n_warps * n_lanes * vec` rather than of `n_warps * vec`, so the run the lanes
  need has to divide `N / n_warps`; `None` when it cannot, or when `a_lanes` does
  not divide the reduced axis.
  """
  a_lanes, n_lanes, vec = _lane_split(num_a, num_n // n_warps, bitwidth)
  if n_lanes == 1 or num_a % a_lanes:
    return None
  # The base tile's whole extent along `N`: warps, then lanes, then the vector.
  base_n = n_warps * n_lanes * vec
  sub_a = num_a // a_lanes
  m_dims = (-6,) if m_warps > 1 else ()
  if sub_a > 1:
    # Tiled: `(m_warps, a_lanes, n_warps, sub_a, n_lanes, vec)`.
    layout = _tiled(
      ((m_warps, num_a, base_n), (sub_a, n_lanes * vec), (vec,)),
      warp_dims=(*m_dims, -4),
      lane_dims=(-5, -2),
    )
  else:
    # One reduced element per lane, so there is nothing left of `A` to split.
    # Tiled: `(m_warps, a_lanes, n_warps, n_lanes, vec)`.
    layout = _tiled(
      ((m_warps, num_a, base_n), (n_lanes * vec,), (vec,)),
      warp_dims=(*(d + 1 for d in m_dims), -3),
      lane_dims=(-4, -2),
    )
  return _Mapping(
    m_multiple=m_warps,
    n_multiple=base_n,
    run=n_lanes * vec,
    layout=layout,
  )


def _mappings(shape, *, bitwidth) -> list[_Mapping]:
  """Every thread mapping `shape` admits.

  The four warps split between `M` and `N` -- `m_warps * n_warps == 4`, both
  powers of two -- and every split `M` has the rows for is a candidate, with the
  two lane mappings under it. What rules most of them out is divisibility: `N` has
  to be a multiple of what a base tile spans along it, and the reduced axis a
  multiple of the lanes' share of it, every axis being tiled exactly.

  Unordered: it is what `plan` can block from a mapping, not the mapping itself,
  that says which is best. See `_score`.
  """
  num_m, num_a, *rest = shape
  if not rest:
    # Nothing is minor to the reduced axis, so `M` is the only place the warps can
    # go, and a short `M` has nowhere to put them at all.
    fits = [_warps_on_m(shape, bitwidth=bitwidth)] if num_m >= _WARP_ROWS else []
  else:
    (num_n,) = rest
    fits = []
    for n_warps in (1, 2, _WARP_ROWS):
      m_warps = _WARP_ROWS // n_warps
      # A warp with no element of `N` to itself has no distinct work, and
      # replicating it would cost four times the registers.
      if num_m < m_warps or num_n % n_warps:
        continue
      if n_warps == 1:
        fits.append(_warps_on_m(shape, bitwidth=bitwidth))
        continue
      warps = dict(bitwidth=bitwidth, m_warps=m_warps, n_warps=n_warps)
      fits.append(_lanes_split_under_warps_on_n(num_a, num_n, **warps))
      fits.append(_all_lanes_on_reduced(num_a, num_n, **warps))
  # `N` is tiled exactly, so a base tile that does not divide it has no block.
  return [
    f
    for f in fits
    if f is not None
    and (f.n_multiple is None or shape[-1] % f.n_multiple == 0)
  ]


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Plan:
  """How one canonical normalization is spread over the device.

  One CTA handles one tile, so `grid` is both the blocking and the launch.

  Attributes:
    shape: The canonical shape of `x`, `(M, A)` or `(M, A, N)`.
    block: One tile of `shape`, which it divides exactly on every axis. The
      reduced axis is not blocked at all, so
      `block[REDUCE_AXIS] == shape[REDUCE_AXIS]`.
    layout: The register layout of a tile, from the thread mapping that chose the
      block; see the module docstring.
  """

  shape: tuple[int, ...]
  block: tuple[int, ...]
  layout: Any

  @property
  def grid(self) -> tuple[int, ...]:
    """Blocks along each axis; 1 along the reduced axis, which is not blocked."""
    return tuple(
      s // b for s, b in zip(self.shape, self.block, strict=True)
    )

  @property
  def grid_names(self) -> tuple[str, ...]:
    """A `plgpu.kernel` grid name per axis, in the order of `shape`."""
    return ('m', 'a', 'n')[: len(self.shape)]

  def tile_regs(self) -> int:
    """Tile elements each thread holds, in `float32` registers.

    Every thread of the warpgroup holds a distinct part of the tile -- no mapping
    here replicates -- and every axis a layout tiles is a multiple of its share of
    it, so this divides exactly.
    """
    return math.prod(self.block) // _WARPGROUP

  def tile_starts(self, indices) -> tuple[Any, ...]:
    """Returns a tile's element offset per axis, given its block index per axis.

    Every axis is tiled exactly, so a tile starts where its index puts it and no
    tile overhangs its axis. That is what makes every access provably in bounds,
    which Mosaic requires: it has no masked GMEM load and no masked GMEM store.
    """
    return tuple(i * b for i, b in zip(indices, self.block, strict=True))

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


def _block(
  fit: _Mapping, shape, *, itemsize: int, block_m: int, block_n: int | None
) -> tuple[int, ...] | None:
  """The largest tile a mapping and the register budget both admit, or `None`.

  The register budget bounds the *product* of the blocks: a tile costs
  `block_m * A * block_n / 128` per thread, with `A` fixed, as the reduced axis is
  never blocked. Each block is the widest divisor of its axis that the budget and
  the bound allow -- a divisor because every axis is tiled exactly, and it is `N`
  that gets first call on the budget, having also to be a multiple of what the
  base tile spans. `None` when what is left admits no block of `M`: the smallest
  tile is over the budget, or no divisor of `M` is a multiple of the rows the
  warps take.
  """
  num_a = shape[REDUCE_AXIS]
  budget = _MAX_TILE_REGS * _WARPGROUP
  if fit.n_multiple is None:
    tail = ()
  else:
    # The floor wins over both bounds: a block below what the base tile spans has
    # no layout, so asking for one is asking for nothing.
    cap_n = (
      gpu_utils.CACHE_LINE_SIZE_BYTES // itemsize if block_n is None else block_n
    )
    cap = min(cap_n, budget // (fit.m_multiple * num_a))
    tail = (
      next(
        divisors(shape[-1], max(cap, fit.n_multiple), multiple_of=fit.n_multiple)
      ),
    )

  # `M` takes the rest, in multiples of the rows the warps want. One whole
  # multiple is always tried, even if `block_m` asks for less than that.
  row_regs = num_a * math.prod(tail)
  cap_m = min(max(fit.m_multiple, min(block_m, shape[0])), budget // row_regs)
  rows = next(divisors(shape[0], cap_m, multiple_of=fit.m_multiple), None)
  return None if rows is None else (rows, num_a, *tail)


def _score(fit: _Mapping, block: tuple[int, ...], itemsize: int):
  """How well a mapping's tile reads, largest first. Coalescing, in two parts.

  A load costs requests, and it costs lines. The run a lane group reads is what
  one request covers, and a sector is as much of one as a request can be, so a
  run short of a sector spends two requests where one would do. The tile's extent
  along the minor axis is then how much of each line it touches it actually
  wants, and a whole line is as much as there is to want: a narrower tile leaves
  the rest of every line to some other CTA, and a wider one buys nothing more.

  Both are capped, so neither can be run up at the other's expense, and what is
  left over breaks ties towards the most warps on `M` -- the fewest, widest
  pieces the same span can be read in, and so the fewest instructions.
  """
  return (
    min(fit.run * itemsize, _SECTOR_BITS // 8),
    min(block[-1] * itemsize, gpu_utils.CACHE_LINE_SIZE_BYTES),
    fit.m_multiple,
  )


def _decline(shape, fits: list[_Mapping]) -> str:
  """Why `plan` has no tile for `shape`, for the caller that falls back to say.

  Either the shape admits mappings and none of them can be blocked -- the budget
  will not have the smallest tile, or `M` has no divisor the warps can take --
  or it admits none at all, and then what they wanted is threads: four warps with
  rows or columns of their own, and 32 lanes with an equal share of the reduced
  axis.
  """
  num_m, num_a = shape[0], shape[REDUCE_AXIS]
  if fits:
    cheapest = min(fits, key=lambda f: _min_tile_regs(f, num_a))
    regs = _min_tile_regs(cheapest, num_a)
    if regs > _MAX_TILE_REGS:
      tail = () if cheapest.n_multiple is None else (cheapest.n_multiple,)
      return (
        f'its smallest tile, {(cheapest.m_multiple, num_a, *tail)}, needs {regs}'
        f' registers per thread, over the budget of {_MAX_TILE_REGS}'
      )
    return (
      f'no divisor of the rows ({num_m}) is both a multiple of the'
      f' {cheapest.m_multiple} the warps take and small enough to afford'
    )
  if len(shape) == 2:
    # All a rank-2 shape has is four warps of rows with every lane on the reduced
    # axis, so there are only the two ways to fail it.
    if num_m < _WARP_ROWS:
      return (
        f'rows ({num_m}) are fewer than the {_WARP_ROWS} warps and there is no'
        ' axis minor to the reduced one to take the rest'
      )
    return (
      f'the reduced axis ({num_a}) must be a multiple of the {_LANES} lanes that'
      ' tile it'
    )
  return (
    f'no split of the {_WARPGROUP} threads fits: the {_WARP_ROWS} warps need rows'
    f' ({num_m}) or columns ({shape[-1]}) of their own, and the lanes a share of'
    f' the reduced axis ({num_a}) that divides it'
  )


def plan(
  shape: tuple[int, ...], itemsize: int, *, block_m: int, block_n: int | None
) -> Plan:
  """Plans a normalization of a canonical `(M, A)` or `(M, A, N)` array.

  Args:
    shape: The canonical shape of `x`; see the module docstring.
    itemsize: Bytes per element of `x`.
    block_m: Upper bound on the rows of `M` per tile.
    block_n: Upper bound on the columns of `N` per tile.

  Every mapping the shape admits is blocked (`_mappings`, `_block`) and the tile
  that reads best wins (`_score`). Nothing else is compared: a mapping is a place
  to put 128 threads, they all put every one of them to work, and what is left to
  tell them apart is the shape of the loads they make.

  Raises:
    NotImplementedError: if no mapping fits, saying what they wanted. This allows
    us to fall back to another implementation.
  """
  shape = tuple(shape)
  # A trailing axis of one is no axis at all, and dropping it puts the reduced
  # axis back in the minor position, where the vector can come from it. Going the
  # other way -- treating `(M, A)` as `(M, A, 1)` to be rid of the rank-2 case --
  # would cost exactly that: with the vector pinned to a minor axis of one
  # element, the lanes would read `A` a scalar at a time.
  if len(shape) == 3 and shape[2] == 1:
    shape = shape[:2]
  fits = _mappings(shape, bitwidth=itemsize * 8)
  blocks = dict(block_m=block_m, block_n=block_n, itemsize=itemsize)
  viable = [
    (fit, block)
    for fit in fits
    if (block := _block(fit, shape, **blocks)) is not None
  ]
  if not viable:
    raise NotImplementedError(
      f'No thread mapping fits {shape}: {_decline(shape, fits)}.'
    )

  fit, block = max(viable, key=lambda fb: _score(*fb, itemsize))
  return Plan(shape=shape, block=block, layout=fit.layout)


