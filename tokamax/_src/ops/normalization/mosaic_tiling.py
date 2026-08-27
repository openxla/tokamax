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
Given `x` canonicalized to `(M, A)` or `(M, A, N)`, there are 3 layout options:

  - `_warps_on_m`. Warps on `M`, lanes split between `A` and `N`; `_lane_split` picks the split.
  - `_warps_on_mn_split`. Warps on `M` and part of `N`, lanes on `N`.
  - `_warps_on_mn`. Warps on `M` and `N`, lanes on `A`.

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

_WARP_ROWS = 4
_LANES = 32
_WARPGROUP = _WARP_ROWS * _LANES

# The memory system moves 32 bytes at a time, so this many bits is the least a
# run of lanes should read contiguously; below it a load fetches bytes it will
# not use. See `_lane_split`.
_SECTOR_BITS = 256

# Lower bound on tile elements per thread, in `float32` registers.
_MAX_TILE_REGS = 64

def divisors(n: int, cap: int, *, multiple_of: int = 1):
  """Yields divisors of `n` at most `cap` and a multiple of `multiple_of`, descending."""
  start = min(cap, n) // multiple_of * multiple_of
  for b in range(start, 0, -multiple_of):
    if n % b == 0:
      yield b

def _pow2_part(n: int) -> int:
  """The largest power of two dividing `n`."""
  return n & -n


def _vector_length(cols: int, bitwidth: int, *, lanes: int = _LANES) -> int:
  """The widest 16-byte-or-less vector length allowing `lanes` to tile `cols`."""
  vector_length = (8 * 16) // bitwidth  # 16-byte vectors.
  while cols % (lanes * vector_length):
    vector_length //= 2
  return vector_length


def _tiled(tiles, *, warp_dims, lane_dims):
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

def _warps_on_m(shape: tuple[int, ...], bitwidth: int) -> _Mapping | None:
  _, num_a, *rest = shape
  if not rest:
    if num_a % _LANES:
      return None
    vec = _vector_length(num_a, bitwidth)
    return _Mapping(
      m_multiple=_WARP_ROWS,
      n_multiple=None,
      run=num_a,
      layout=_tiled(
        ((_WARP_ROWS, _LANES * vec), (vec,)), warp_dims=(-3,), lane_dims=(-2,)))

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


def _warps_on_mn(
  num_a, num_n, *, bitwidth, m_warps, n_warps
) -> _Mapping | None:
  if num_a % _LANES:
    return None
  vec = _vector_length(num_n, bitwidth, lanes=n_warps)
  run = n_warps * vec
  sub_a = num_a // _LANES
  # A leading 1 is only canonical in the base tile, so an `M` feeding no warp of
  # its own is left out of the warp dims rather than named as a size-1 one.
  m_dims = (-5,) if m_warps > 1 else ()
  if sub_a > 1:
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
    run=vec,
    layout=layout,
  )


def _warps_on_mn_split(
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

  The four warps split between `M` and `N` -- `m_warps * n_warps == 4`.
  `N` has to be a multiple of what a base tile spans along it, and the reduced axis a
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
      fits.append(_warps_on_mn_split(num_a, num_n, **warps))
      fits.append(_warps_on_mn(num_a, num_n, **warps))
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

  Attributes:
    shape: The canonical shape of `x`, `(M, A)` or `(M, A, N)`.
    block: One tile of `shape`, which it divides exactly on every axis.
    layout: The register layout of a tile
  """

  shape: tuple[int, ...]
  block: tuple[int, ...]
  layout: Any

  def grid(self) -> tuple[int, ...]:
    return tuple(s // b for s, b in zip(self.shape, self.block, strict=True))

  def grid_names(self) -> tuple[str, ...]:
    return ('m', 'a', 'n')[: len(self.shape)]

  def tile_regs(self) -> int:
    return math.prod(self.block) // _WARPGROUP

  def tile_starts(self, indices) -> tuple[Any, ...]:
    "Returns a tile's element offset per axis, given its block index per axis."
    return tuple(i * b for i, b in zip(indices, self.block, strict=True))

  def out_index(self) -> tuple[Any, ...]:
    """Returns the slice of `x` this CTA covers, as one `pl.ds` per axis."""
    starts = self.tile_starts(map(jax.lax.axis_index, self.grid_names()))
    return tuple(pl.ds(s, b) for s, b in zip(starts, self.block, strict=True))


def drop_reduced(xs: tuple[Any, ...]) -> tuple[Any, ...]:
  return xs[:1] + xs[2:]


def _block(fit: _Mapping, shape, itemsize: int) -> tuple[int, ...] | None:
  """The largest tile the mapping and the register budget admit, or `None`.

  The budget bounds the *product* of the blocks: a tile costs
  `block_m * A * block_n / 128` registers per thread, with `A` fixed, as the
  reduced axis is never blocked. Each block is then the widest divisor of its axis
  the budget allows -- a divisor because every axis is tiled exactly.

  `N` gets first call on the budget, and takes a cache line of it at most: a
  wider block buys no more coalescing (`_score`) and every column of it is a row
  of `M` unspent. `M` takes what is left, which is the whole of the rest -- there
  is nothing else to spend registers on, and rows are how a memory-bound kernel
  gets its work per CTA. `_MAX_TILE_REGS` is what bounds that, and so what to
  turn down if occupancy, not the tile, is what the kernel wants.

  `None` when the budget admits no block of `M` at all: the mapping's smallest
  tile is over it, or no divisor of `M` is a multiple of the rows the warps take.
  """
  num_a = shape[1]
  budget = _MAX_TILE_REGS * _WARPGROUP
  if fit.n_multiple is None:
    tail = ()
  else:
    # The floor wins over the cap: a block below what the base tile spans has no
    # layout, so asking for one is asking for nothing.
    cap_n = min(
      gpu_utils.CACHE_LINE_SIZE_BYTES // itemsize,
      budget // (fit.m_multiple * num_a),
    )
    tail = (
      next(
        divisors(
          shape[-1], max(cap_n, fit.n_multiple), multiple_of=fit.n_multiple
        )
      ),
    )

  row_regs = num_a * math.prod(tail)
  cap_m = min(shape[0], budget // row_regs)
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

def plan(shape: tuple[int, ...], itemsize: int) -> Plan:
  """Plans a normalization of a canonical `(M, A)` or `(M, A, N)` array.

  The plan is a function of the shape and the dtype alone: what a tile can be is
  settled by the layouts the shape admits and by the register budget, and there
  is no bound left for a caller to pass. See `_block`.

  Args:
    shape: The canonical shape of `x`; see the module docstring.
    itemsize: Bytes per element of `x`.

  Every mapping the shape admits is blocked (`_mappings`, `_block`) and the tile
  that reads best wins (`_score`). Nothing else is compared: a mapping is a place
  to put 128 threads, they all put every one of them to work, and what is left to
  tell them apart is the shape of the loads they make.

  Raises:
    NotImplementedError: if no mapping fits, saying what they wanted. This allows
    us to fall back to another implementation.
  """
  # Treat (M,A,1) as (M,A)
  if len(shape) == 3 and shape[2] == 1:
    shape = shape[:2]
  fits = _mappings(shape, bitwidth=itemsize * 8)
  viable = [
    (fit, block)
    for fit in fits
    if (block := _block(fit, shape, itemsize)) is not None
  ]
  if not viable:
    raise NotImplementedError(
      f'No thread mapping fits {shape}'
    )

  fit, block = max(viable, key=lambda fb: _score(*fb, itemsize))
  return Plan(shape=shape, block=block, layout=fit.layout)
