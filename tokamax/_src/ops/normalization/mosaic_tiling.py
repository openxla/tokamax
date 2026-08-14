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
"""Blocking and the input pipeline for the Mosaic GPU normalization kernels.

Tiles reach the kernel through `plgpu.emit_pipeline`, which stages them into
SMEM with `cp.async` on Ampere. Two consequences shape everything here:

  - The SMEM ref carries tiling and swizzling transforms, and layout inference
    derives a register layout from them. Nothing states a layout by hand; that
    needs `fa.short_tile_layout` to be among the candidates offered for an
    optimized SMEM transfer, which is a local patch to `jax` at the time of
    writing (see `jax-bug-mgpu-keepdims/`).
  - `cp.async` cannot predicate an out-of-bounds access, so every copy has to be
    provably in bounds. So blocks tile their axis exactly, and the question of a
    partial block does not arise.

The tiling and the swizzle apply to the block's two minormost axes, and only
those two are constrained; the reduced axis is never blocked, because a whole row
has to be resident for the reduction. That leaves the whole thing indifferent to
the rank of `x`, which `plan` canonicalizes to `(M, A)` or `(M, A, B)`:

  - `(M, A)`, the contiguous case. The reduced axis is the minor one, so it is
    what the 32 lanes tile, and `M` is blocked in multiples of 8.
  - `(M, A, B)`. `B` is the minor axis, so it is blocked in multiples of 32, `A`
    takes the 8-row tiling, and `M` may be blocked freely.

Staging a rank-3 tile needs four local fixes in `jax`; see
`mgpu-rank3-transfer-probe/README.md` for what each one is and how it was found.
No layout is stated by hand -- inference finds one once it is offered a candidate
for a rank-3 value.

Outputs are not pipelined: `emit_pipeline` supports input-only pipelines
pre-Hopper. The kernels store them by hand, relaying out through SMEM so that
the store to GMEM coalesces -- as `pallas_mosaic_gpu_kernel_sm80` does.
"""

import dataclasses
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from tokamax._src import gpu_utils


# The tiled transform tiles the second minormost dimension by 8, so the tile's
# slow axis -- and any block of it -- has to be a multiple of this.
_TILING_ROWS = 8

# The 32 lanes tile the tile's minor axis, so it has to be a multiple of this.
_TILING_COLS = 32

# Per-CTA SMEM, as reported by Mosaic when a kernel asks for too much. The
# budget covers the pipeline's staging buffers *and* the store scratch.
_SMEM_BUDGET = 227 * 1024

# Tiles per CTA. Enough to give `cp.async` something to overlap with, without
# making the grid so small that the device runs out of blocks to schedule.
_STEPS_PER_CTA = 4

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


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Plan:
  """How one canonical normalization is spread over the device.

  The blocks are handed out to CTAs, and each CTA walks its own share as a
  pipeline.

  Attributes:
    shape: The canonical shape of `x`, `(M, A)` or `(M, A, B)`.
    block: One tile of `shape`. The reduced axis is not blocked, so
      `block[REDUCE_AXIS] == shape[REDUCE_AXIS]`.
    itemsize: Bytes per element of `x`.
  """

  shape: tuple[int, ...]
  block: tuple[int, ...]
  itemsize: int

  @property
  def swizzle(self) -> int:
    """SMEM swizzle. The block's minor axis is what it has to divide."""
    return plgpu.find_swizzle(
      self.block[-1] * self.itemsize * 8, 'normalization tile'
    )

  @property
  def grid(self) -> tuple[int, ...]:
    """Blocks along each axis; 1 along the reduced axis, which is not blocked."""
    return tuple(s // b for s, b in zip(self.shape, self.block, strict=True))

  @property
  def steps(self) -> int:
    """Tiles in total."""
    return math.prod(self.grid)

  @property
  def steps_per_cta(self) -> int:
    """Tiles each CTA walks."""
    return min(self.steps, _STEPS_PER_CTA)

  @property
  def num_ctas(self) -> int:
    """CTAs, i.e. the `plgpu.kernel` grid."""
    return ceil_div(self.steps, self.steps_per_cta)

  @property
  def num_stages(self) -> int:
    """Pipeline depth."""
    return min(2, self.steps_per_cta)

  @property
  def transforms(self):
    """Tiling and swizzling for the staged tile, and for the store scratch.

    This is what gives layout inference a tiled layout to work from; without it
    the register layout comes out strided and the reduction rejects it.
    """
    elem_bits = self.itemsize * 8
    return (
      plgpu.TilingTransform((_TILING_ROWS, 8 * self.swizzle // elem_bits)),
      plgpu.SwizzleTransform(self.swizzle),
    )

  def block_indices(self, step) -> tuple[jax.Array, ...]:
    """Returns the tile this CTA reads on `step`, as one index per axis.

    The last CTA may be handed fewer than `steps_per_cta` tiles; rather than
    read out of bounds -- which `cp.async` cannot predicate -- it re-reads the
    last tile. Rows are normalized independently, so recomputing one is
    idempotent: the same values are written back over themselves.
    """
    first = jax.lax.axis_index('cta') * self.steps_per_cta
    flat = jnp.minimum(first + step, self.steps - 1)
    indices = []
    for n in reversed(self.grid):
      flat, i = jnp.divmod(flat, n)
      indices.append(i)
    return tuple(reversed(indices))

  def out_index(self, step) -> tuple[Any, ...]:
    """Returns the slice of `x` that this CTA's `step` covers."""
    return tuple(
      pl.ds(i * b, b)
      for i, b in zip(self.block_indices(step), self.block, strict=True)
    )

  def smem_bytes(self) -> int:
    tile = math.prod(self.block) * self.itemsize
    return tile * (self.num_stages + 1)  # Staging buffers, plus the store scratch.


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

  The largest divisor of each blocked axis at or below its cap that is also a
  multiple of the tiling is used; see the module docstring for why it has to
  divide exactly.

  The `NotImplementedError`s below all have to be raised here, even though the
  shapes they reject would fail later anyway: they fail as a `ValueError` from
  layout inference, or as a bare `AssertionError` from
  `pallas/mosaic_gpu/core.py`. `op.Op` dispatch and the tests both key off
  `NotImplementedError` in particular, so anything else is a hard failure rather
  than a fallback to another implementation. `mgpu-rank3-transfer-probe/
  guard_probe.py` checks that, shape by shape.
  """
  num_m, num_a, *rest = shape
  # The two minormost axes of the block are what the tiling and the swizzle
  # apply to: the second by 8 rows, the minor one by the 32 lanes.
  if rest:
    (num_b,) = rest
    if num_a % _TILING_ROWS:
      raise NotImplementedError(
        f'The reduced axis ({num_a}) must be a multiple of {_TILING_ROWS}, which'
        ' is how the staged tile is tiled, when it is not the minor axis.'
      )
    cap = gpu_utils.CACHE_LINE_SIZE_BYTES // itemsize if block_b is None else block_b
    minor = next(divisors(num_b, cap, multiple_of=_TILING_COLS), None)
    if minor is None:
      raise NotImplementedError(
        f'The trailing axis ({num_b}) has no divisor that is a multiple of'
        f' {_TILING_COLS} and at most {cap}, so no tile of it can be staged.'
      )
    minor_block, m_multiple = (minor,), 1
  else:
    if num_a % _TILING_COLS:
      raise NotImplementedError(
        f'The reduced axis ({num_a}) must be a multiple of {_TILING_COLS} so'
        ' that the 32 lanes tile it.'
      )
    if num_m % _TILING_ROWS:
      raise NotImplementedError(
        f'Rows ({num_m}) must be a multiple of {_TILING_ROWS}, which is how the'
        ' staged tile is tiled.'
      )
    minor_block, m_multiple = (), _TILING_ROWS

  # A whole row has to be resident, so a long one is paid for in `block_m`. Take
  # the largest block that fits SMEM rather than declining outright.
  smallest = None
  for candidate in divisors(num_m, block_m, multiple_of=m_multiple):
    p = Plan(
      shape=tuple(shape),
      block=(candidate, num_a, *minor_block),
      itemsize=itemsize,
    )
    if p.smem_bytes() <= _SMEM_BUDGET:
      return p
    smallest = p

  assert smallest is not None, f'{m_multiple} divides {num_m}, so this holds'
  raise NotImplementedError(
    f'Even a tile of {smallest.block} over {smallest.num_stages} stages needs'
    f' {smallest.smem_bytes()} bytes of SMEM, over the budget of'
    f' {_SMEM_BUDGET}. The reduced axis cannot be blocked, so this bounds it.'
  )


def with_usable_block_m(config):
  """Raises a borrowed Triton config's `block_m` to something this kernel can use.

  In the contiguous case a block has to be a multiple of 8 rows, so anything
  below that would leave `divisors` nothing to find.
  """
  return dataclasses.replace(config, block_m=max(_TILING_ROWS, config.block_m))
