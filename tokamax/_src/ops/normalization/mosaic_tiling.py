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

Blocks tile their axis exactly, apart from `M`, whose last tile slides back to
end flush with the axis (see `Plan.tile_starts`). Nothing here can read or write
out of bounds: Mosaic has no masked GMEM load and no masked GMEM store, so every
access has to be provably in bounds, and sliding is what makes that true without
constraining `M`.

The reduced axis borrows the same trick where the lanes tile it and their share
of it does not divide it: it is read in several loads, the last sliding back, and
the duplicated elements are dropped from the reduction in registers. See
`Plan.a_tiles`.
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
    a_tile: How much of the reduced axis one load takes. The whole of it, unless
      the lanes tile it and their share does not divide it; see `Plan.a_tiles`.
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
  a_tile: int
  run: int
  layout: Any


def _resident(a_tile: int, num_a: int) -> int:
  """The reduced axis rounded up to whole loads of it.

  The duplicates an overlapping load reads cost registers, even though they cost
  no bandwidth; see `Plan.a_tiles`.
  """
  return ceil_div(num_a, a_tile) * a_tile


def _min_tile_regs(mapping: _Mapping, num_a: int) -> int:
  """Registers per thread of the smallest tile the mapping admits.

  Both blocks at their floor. `plan` can only ever grow them from here, so a
  mapping over the budget at this size has nothing left to give.
  """
  cols = mapping.n_multiple or 1
  resident = _resident(mapping.a_tile, num_a)
  return mapping.m_multiple * resident * cols // _WARPGROUP


def _lane_split(num_a: int, num_n: int, bitwidth: int) -> tuple[int, int, int]:
  """Splits the 32 lanes between the reduced axis and the minor one.

  Returns `(a_lanes, n_lanes, vec)`, where `a_lanes * n_lanes == 32`: `a_lanes`
  lanes tile `A` and the other `n_lanes` tile `N`, each holding a `vec`-long
  vector of it.

  Consecutive lanes run along `N`, so a group of `n_lanes` covers `n_lanes * vec`
  contiguous elements -- call it the run -- and the warp reads `a_lanes` such runs
  at a stride of `N`. Registers per thread work out to

      (block_m / 4) * (a_tile / a_lanes) * (block_n / run) * vec

  and with `block_n` at its smallest, one run, that collapses to
  `(block_m / 4) * a_tile * run / 32`: the cost depends on the run alone, not on
  how the run is divided into lanes and vector. So the run is the whole trade --
  wider is better coalesced and dearer in registers -- and one 32-byte sector is
  where it should sit, the shortest run that wastes no bandwidth.

  That leaves `vec` free, and the widest one gives the fewest, widest load
  instructions. It is only bounded away from the widest when `a_lanes` has to
  shrink to divide `A`, which is worth doing: an `a_lanes` that divides is an
  `a_tile` of the whole reduced axis, and so a single load rather than two
  overlapping ones.
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
  # list is never empty; falling back on it puts the fewest lanes on `A`, and so
  # leaves the least of it over to slide.
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
    # No `N`, so all 32 lanes go on `A` and the vector comes from it too.
    if num_a < _LANES:
      return None
    if num_a % _LANES:
      # No base tile divides the axis, so it takes several loads with the last
      # sliding back; see `Plan.a_tiles`. The vector is as wide as the *axis*
      # allows, rather than as wide as a tile of it allows, because it is the
      # slid offset -- a multiple of neither the tile nor the axis -- that has to
      # stay aligned.
      vec = _vector_length(num_a, bitwidth, lanes=1)
      while _LANES * vec > num_a:
        vec //= 2
      a_tile = _LANES * vec
    else:
      vec = _vector_length(num_a, bitwidth)
      a_tile = num_a
    return _Mapping(
      m_multiple=_WARP_ROWS,
      n_multiple=None,
      a_tile=a_tile,
      # Nothing is minor to `A`, so the whole load is one contiguous run.
      run=a_tile,
      layout=_lanes_on_reduced_layout(vec),
    )

  (num_n,) = rest
  a_lanes, n_lanes, vec = _lane_split(num_a, num_n, bitwidth)

  if n_lanes == 1:
    # `N` is too short, or too oddly shaped, for any lane to be spared for it:
    # all 32 go back on `A` and the whole of `N` rides inside each lane. The two
    # are adjacent in memory, so a warp still covers `32 * N` contiguous
    # elements, and the reduction is a shuffle again.
    if num_a < _LANES:
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
      # The vector comes from `N` here, so the lanes take a bare `_LANES` of the
      # reduced axis and a ragged one is covered by several loads.
      a_tile=num_a if num_a % _LANES == 0 else _LANES,
      # `N` rides whole inside a lane, and is contiguous with `A` besides.
      run=num_n,
      layout=layout,
    )

  if num_a < a_lanes:
    return None
  # `a_lanes` has to divide the load, so a reduced axis it does not divide is
  # covered by several overlapping ones; see `Plan.a_tiles`.
  a_tile = num_a // a_lanes * a_lanes
  run = n_lanes * vec
  sub_a = a_tile // a_lanes
  if sub_a > 1:
    # Tiled: `(m_tile, a_lanes, n_lanes, sub_a, vec)`.
    layout = _tiled(
      ((_WARP_ROWS, a_tile, run), (sub_a, vec)),
      warp_dims=(-5,),
      lane_dims=(-4, -3),
    )
  else:
    # One reduced element per lane, so there is nothing left of `A` to split:
    # a second tile of `(1, vec)` is not canonical. Tiled:
    # `(m_tile, a_lanes, n_lanes, vec)`.
    layout = _tiled(
      ((_WARP_ROWS, a_tile, run), (vec,)), warp_dims=(-4,), lane_dims=(-3, -2)
    )
  return _Mapping(
    m_multiple=_WARP_ROWS,
    n_multiple=run,
    a_tile=a_tile,
    run=run,
    layout=layout,
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

  `None` when the reduced axis cannot give all 32 lanes an element each, which is
  the one thing this mapping cannot slide its way out of: the lanes tile `A`
  whole, so a load below 32 of it has no base tile at all.
  """
  if num_a < _LANES:
    return None
  vec = _vector_length(num_n, bitwidth, lanes=n_warps)
  run = n_warps * vec
  a_tile = num_a // _LANES * _LANES
  sub_a = a_tile // _LANES
  # A leading 1 is only canonical in the base tile, so an `M` feeding no warp of
  # its own is left out of the warp dims rather than named as a size-1 one.
  m_dims = (-5,) if m_warps > 1 else ()
  if sub_a > 1:
    # Tiled: `(m_warps, _LANES, n_warps, sub_a, vec)`.
    layout = _tiled(
      ((m_warps, a_tile, run), (sub_a, vec)),
      warp_dims=(*m_dims, -3),
      lane_dims=(-4,),
    )
  else:
    # Tiled: `(m_warps, _LANES, n_warps, vec)`.
    layout = _tiled(
      ((m_warps, a_tile, run), (vec,)),
      warp_dims=(*(d + 1 for d in m_dims), -2),
      lane_dims=(-3,),
    )
  return _Mapping(
    m_multiple=m_warps,
    n_multiple=run,
    a_tile=a_tile,
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
  need has to divide `N / n_warps`; `None` when it cannot, or when `A` is too
  short to give `a_lanes` an element each.
  """
  a_lanes, n_lanes, vec = _lane_split(num_a, num_n // n_warps, bitwidth)
  if n_lanes == 1 or num_a < a_lanes:
    return None
  a_tile = num_a // a_lanes * a_lanes
  # The base tile's whole extent along `N`: warps, then lanes, then the vector.
  base_n = n_warps * n_lanes * vec
  sub_a = a_tile // a_lanes
  m_dims = (-6,) if m_warps > 1 else ()
  if sub_a > 1:
    # Tiled: `(m_warps, a_lanes, n_warps, sub_a, n_lanes, vec)`.
    layout = _tiled(
      ((m_warps, a_tile, base_n), (sub_a, n_lanes * vec), (vec,)),
      warp_dims=(*m_dims, -4),
      lane_dims=(-5, -2),
    )
  else:
    # One reduced element per lane, so there is nothing left of `A` to split.
    # Tiled: `(m_warps, a_lanes, n_warps, n_lanes, vec)`.
    layout = _tiled(
      ((m_warps, a_tile, base_n), (n_lanes * vec,), (vec,)),
      warp_dims=(*(d + 1 for d in m_dims), -3),
      lane_dims=(-4, -2),
    )
  return _Mapping(
    m_multiple=m_warps,
    n_multiple=base_n,
    a_tile=a_tile,
    run=n_lanes * vec,
    layout=layout,
  )


def _mappings(shape, *, bitwidth) -> list[_Mapping]:
  """Every thread mapping `shape` admits.

  The four warps split between `M` and `N` -- `m_warps * n_warps == 4`, both
  powers of two -- and every split `M` has the rows for is a candidate, with the
  two lane mappings under it. `M` needs only as many rows as it feeds warps,
  since its last block slides (`Plan.tile_starts`), where `N` has to be a
  multiple of what a base tile spans along it, so it is `N`'s divisibility that
  rules most of the candidates out.

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
    block: One tile of `shape`. The reduced axis is not blocked, so
      `block[REDUCE_AXIS] == shape[REDUCE_AXIS]`. Every axis but `M` is tiled
      exactly; `M` need only be at least `block[0]` (see `tile_starts`).
    a_tile: How much of the reduced axis one load takes; see `a_tiles`. The whole
      of it, unless the lanes tile it and their share does not divide it.
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

    The whole reduced axis is resident, rounded up to whole `a_tile`s. Every
    thread of the warpgroup holds a distinct part of the tile -- no mapping here
    replicates -- and every axis a layout tiles is a multiple of its share of it,
    so this divides exactly.
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


def _block(
  fit: _Mapping, shape, *, itemsize: int, block_m: int, block_n: int | None
) -> tuple[int, ...] | None:
  """The largest tile a mapping and the register budget both admit, or `None`.

  The register budget bounds the *product* of the blocks: a tile costs
  `block_m * resident * block_n / 128` per thread, with `resident` fixed, as the
  reduced axis is never blocked. `N` is spent first because it has the fewer
  legal values -- it has to tile exactly, where `M`'s last block may slide (see
  `Plan.tile_starts`) -- and then `M` takes whatever budget is left. Both bounds
  are upper bounds, not sizes: a block below what the layout needs is raised to
  it, and one the budget cannot afford is lowered. `None` when that leaves `M`
  with no rows at all, the mapping's smallest tile being over the budget.
  """
  num_a = shape[REDUCE_AXIS]
  budget = _MAX_TILE_REGS * _WARPGROUP
  resident = _resident(fit.a_tile, num_a)
  if fit.n_multiple is None:
    tail = ()
  else:
    # `N` tiles exactly, so the block has to be a divisor of it as well as a
    # multiple of what the base tile spans. The floor wins over both bounds: a
    # smaller block has no layout, so asking for one is asking for nothing.
    cap_n = (
      gpu_utils.CACHE_LINE_SIZE_BYTES // itemsize if block_n is None else block_n
    )
    cap = min(cap_n, budget // (fit.m_multiple * resident))
    cap = max(cap, fit.n_multiple)
    tail = (next(divisors(shape[-1], cap, multiple_of=fit.n_multiple)),)

  # `M` takes the rest. `tile_regs` is linear in the rows, so the most that fit
  # is arithmetic rather than a search, and one whole multiple is always tried
  # even if `block_m` asks for less than that.
  step = fit.m_multiple
  row_regs = resident * math.prod(tail)
  rows = (
    min(max(step, min(block_m, shape[0])), budget // row_regs) // step * step
  )
  return (rows, num_a, *tail) if rows else None


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

  Either the shape admits mappings but the register budget cannot afford the
  smallest tile of any of them -- the cheapest says by how much -- or it admits
  none, and then what they all wanted is threads: four warps with rows or columns
  of their own, and 32 lanes with an element of the reduced axis each.
  """
  num_m, num_a = shape[0], shape[REDUCE_AXIS]
  if fits:
    cheapest = min(fits, key=lambda f: _min_tile_regs(f, num_a))
    tail = () if cheapest.n_multiple is None else (cheapest.n_multiple,)
    return (
      f'its smallest tile, {(cheapest.m_multiple, num_a, *tail)}, needs'
      f' {_min_tile_regs(cheapest, num_a)} registers per thread, over the budget'
      f' of {_MAX_TILE_REGS}'
    )
  if len(shape) == 2 and num_m < _WARP_ROWS:
    return (
      f'rows ({num_m}) are fewer than the {_WARP_ROWS} warps and there is no'
      ' axis minor to the reduced one to take the rest'
    )
  if num_a < _LANES:
    return f'the reduced axis ({num_a}) must be at least {_LANES}, one per lane'
  return (
    f'rows ({num_m}) and the minor axis ({shape[-1]}) cannot feed the'
    f' {_WARP_ROWS} warps between them'
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
  return Plan(shape=shape, block=block, a_tile=fit.a_tile, layout=fit.layout)


