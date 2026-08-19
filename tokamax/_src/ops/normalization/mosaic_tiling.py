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

Tiles go straight from GMEM into registers -- no SMEM, no pipeline. One CTA
handles one tile, so the grid *is* the tiles. Two things follow:

  - The register layout is what makes the load coalesce, so it is stated by hand:
    `short_tile_layout` gives 32 lanes one contiguous run of the minor axis, and
    the four warps a row each, which keeps a reduction along the minor axis
    inside a warp. Layout inference offers nothing this short -- every candidate
    it has tiles the slow axis by 64 or 128, because they all exist to serve an
    MMA -- and a normalization cannot afford a tile that tall, since the whole
    reduced row has to be resident.
  - The tile lives in *registers*, so `plan`'s budget is registers per thread
    rather than SMEM. Unlike SMEM this is a heuristic (see `_MAX_TILE_REGS`):
    overflowing it spills to local memory, which is DRAM, and is a cliff rather
    than an error.

The layout's base tile is `(4, 32 * vector_length)` and applies to a suffix of
the shape, so only the two minormost axes are constrained. The reduced axis is
never blocked -- a whole row has to be resident for the reduction -- which leaves
the whole thing indifferent to the rank of `x`, which `plan` canonicalizes to
`(M, A)` or `(M, A, B)`:

  - `(M, A)`, the contiguous case. The reduced axis is the minor one, so it is
    what the 32 lanes tile, and `M` feeds the warps: it is blocked in multiples
    of 4.
  - `(M, A, B)`. `B` is the minor axis, so it is blocked in multiples of 32, `A`
    feeds the warps and so comes in multiples of 4, and `M`, tiled by neither,
    may be blocked freely.

Blocks tile their axis exactly, apart from `M`, whose last tile slides back to
end flush with the axis (see `Plan.tile_starts`). Nothing here can read or write
out of bounds: Mosaic has no masked GMEM load and no masked GMEM store, so every
access has to be provably in bounds, and sliding is what makes that true without
constraining `M`.

Outputs are stored by hand, straight from the layout the reduction produced --
consecutive lanes hold consecutive elements, so the store coalesces without
going through SMEM to be relaid out.
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


def short_tile_layout(cols: int, bitwidth: int):
  """A tiled layout whose base tile is only 4 rows tall.

  The four warps come from the 4-row dimension, so a row lives inside a single
  warp and a reduction along `cols` is a lane shuffle. Lanes and the vector both
  come from `cols`, so a warp covers `32 * vector_length` contiguous elements --
  one unbroken transaction, in either direction.

  `cols` has to be a multiple of `_LANES`, which is what tiles it; `plan` is what
  guarantees that.
  """
  vector_length = 128 // bitwidth  # 16-byte vectors.
  while cols % (_LANES * vector_length):
    vector_length //= 2
  return plgpu.Layout.TILED(
    plgpu.Tiling(
      ((_WARP_ROWS, _LANES * vector_length), (_WARP_ROWS, vector_length))
    ),
    warp_dims=(-2,),
    lane_dims=(-3,),
    vector_dim=-1,
  )


# The reduced axis of the canonical shape; see the module docstring.
REDUCE_AXIS = 1


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Plan:
  """How one canonical normalization is spread over the device.

  One CTA handles one tile, so `grid` is both the blocking and the launch.

  Attributes:
    shape: The canonical shape of `x`, `(M, A)` or `(M, A, B)`.
    block: One tile of `shape`. The reduced axis is not blocked, so
      `block[REDUCE_AXIS] == shape[REDUCE_AXIS]`. Every axis but `M` is tiled
      exactly; `M` need only be at least `block[0]` (see `tile_starts`).
    itemsize: Bytes per element of `x`.
  """

  shape: tuple[int, ...]
  block: tuple[int, ...]
  itemsize: int

  @property
  def grid(self) -> tuple[int, ...]:
    """Blocks along each axis; 1 along the reduced axis, which is not blocked."""
    return tuple(
      ceil_div(s, b) for s, b in zip(self.shape, self.block, strict=True)
    )

  @property
  def steps(self) -> int:
    """Tiles in total, i.e. the `plgpu.kernel` grid."""
    return math.prod(self.grid)

  @property
  def layout(self):
    """Register layout of a tile, as loaded straight out of GMEM.

    Stated by hand: inference has no candidate this short. See the module
    docstring.
    """
    return short_tile_layout(self.block[-1], self.itemsize * 8)

  def tile_regs(self) -> int:
    """Tile elements each thread holds, in `float32` registers.

    The layout places a whole warpgroup and both tiled axes are multiples of
    their share of it, so this divides exactly.
    """
    return math.prod(self.block) // _WARPGROUP

  def tile_starts(self, flat) -> tuple[jax.Array, ...]:
    """Returns the element offset of tile `flat` (row-major over `grid`), per axis.

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
    indices = []
    for n in reversed(self.grid):
      flat, i = jnp.divmod(flat, n)
      indices.append(i)
    starts = [i * b for i, b in zip(reversed(indices), self.block, strict=True)]
    starts[0] = jnp.minimum(starts[0], self.shape[0] - self.block[0])
    return tuple(starts)

  def out_index(self) -> tuple[Any, ...]:
    """Returns the slice of `x` this CTA covers, as one `pl.ds` per axis."""
    starts = self.tile_starts(jax.lax.axis_index('tile'))
    return tuple(pl.ds(s, b) for s, b in zip(starts, self.block, strict=True))


def drop_reduced(xs: tuple[Any, ...]) -> tuple[Any, ...]:
  """Drops the reduced axis' entry, mapping `x` to the statistics beside it.

  Takes a shape or an index tuple: the statistics have one entry per axis of `x`
  except the reduced one, which they are the reduction over.
  """
  return xs[:REDUCE_AXIS] + xs[REDUCE_AXIS + 1 :]


def plan(
  shape: tuple[int, ...], itemsize: int, *, block_m: int, block_b: int | None
) -> Plan:
  """Plans a normalization of a canonical `(M, A)` or `(M, A, B)` array.

  Args:
    shape: The canonical shape of `x`; see the module docstring.
    itemsize: Bytes per element of `x`.
    block_m: Upper bound on the rows of `M` per tile.
    block_b: Upper bound on the columns of `B` per tile. Unused, and may be
      `None`, when there is no `B`; defaults to one cache line's worth.

  `B` is blocked by the largest divisor at or below its cap that is a multiple of
  the lane tiling; see the module docstring for why it has to divide exactly. `M`
  need only be at least one block, because its last tile slides back rather than
  overhang, so the cap is taken as-is.

  The `NotImplementedError`s below all have to be raised here, even though the
  shapes they reject would fail later anyway: they fail as a `ValueError` from
  layout inference, or as a bare `AssertionError` from
  `pallas/mosaic_gpu/core.py`. `op.Op` dispatch and the tests both key off
  `NotImplementedError` in particular, so anything else is a hard failure rather
  than a fallback to another implementation.
  """
  num_m, num_a, *rest = shape
  # The layout's base tile covers the block's two minormost axes: the second
  # feeds the four warps, the minor one the 32 lanes.
  if rest:
    (num_b,) = rest
    if num_a % _WARP_ROWS:
      raise NotImplementedError(
        f'The reduced axis ({num_a}) must be a multiple of {_WARP_ROWS} to feed'
        ' the four warps, when it is not the minor axis.'
      )
    cap = gpu_utils.CACHE_LINE_SIZE_BYTES // itemsize if block_b is None else block_b
    minor = next(divisors(num_b, cap, multiple_of=_LANES), None)
    if minor is None:
      raise NotImplementedError(
        f'The trailing axis ({num_b}) has no divisor that is a multiple of'
        f' {_LANES} and at most {cap}, so the lanes cannot tile a block of it.'
      )
    minor_block, m_multiple = (minor,), 1
  else:
    if num_a % _LANES:
      raise NotImplementedError(
        f'The reduced axis ({num_a}) must be a multiple of {_LANES} so that the'
        ' 32 lanes tile it.'
      )
    if num_m < _WARP_ROWS:
      raise NotImplementedError(
        f'Rows ({num_m}) must be at least {_WARP_ROWS}, one per warp: a shorter'
        ' tile cannot feed them, and has nothing to slide back onto.'
      )
    minor_block, m_multiple = (), _WARP_ROWS

  # A whole row has to be resident, so a long one is paid for in `block_m`. Take
  # the largest block that fits the register budget rather than declining.
  smallest = None
  # One warp's worth of rows is always legal -- the checks above leave at least
  # that many -- so a `block_m` under it is raised rather than leaving nothing to
  # try.
  cap_m = max(m_multiple, min(block_m, num_m) // m_multiple * m_multiple)
  for candidate in range(cap_m, 0, -m_multiple):
    p = Plan(
      shape=tuple(shape),
      block=(candidate, num_a, *minor_block),
      itemsize=itemsize,
    )
    if p.tile_regs() <= _MAX_TILE_REGS:
      return p
    smallest = p

  assert smallest is not None, f'{num_m} >= {m_multiple}, so this holds'
  raise NotImplementedError(
    f'Even a tile of {smallest.block} needs {smallest.tile_regs()} registers per'
    f' thread, over the budget of {_MAX_TILE_REGS}. The reduced axis cannot be'
    ' blocked, so this bounds it.'
  )


def with_usable_block_m(config):
  """Raises a borrowed Triton config's `block_m` to something this kernel can use.

  In the contiguous case a block has to be a multiple of 4 rows, one per warp,
  and `plan` would raise the cap to that anyway; doing it here keeps the config
  honest.
  """
  return dataclasses.replace(config, block_m=max(_WARP_ROWS, config.block_m))
