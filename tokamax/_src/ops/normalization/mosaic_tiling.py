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
    provably in bounds. The tiling already forces the blocked axis to be a
    multiple of 8 (resp. 32), so a block size that is both a multiple of that
    and a divisor of the axis always exists -- and then the blocks tile the axis
    exactly and the question of a partial block does not arise.
  - The tiled `cp.async` copy is 2D only (`launch_context.py`, `Only 2D copies
    implemented`), and the check runs on the rank *after* squeezing. So the
    staged tile is always 2D: when the reduced axis is not the minor one, the
    block is `(None, A, block_b)` and the degenerate leading axis is squeezed
    away, which also keeps the GMEM slice contiguous.

Outputs are not pipelined: `emit_pipeline` supports input-only pipelines
pre-Hopper. The kernels store them by hand, relaying out through SMEM so that
the store to GMEM coalesces -- as `pallas_mosaic_gpu_kernel_sm80` does.
"""

import dataclasses
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
# ponytail: fixed depth, make it a `Config` field if autotuning wants it.
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


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Plan:
  """How one canonical `(M, A, B)` normalization is spread over the device.

  `A`, the reduced axis, is never blocked: a whole row has to be resident for
  the reduction, so it is what bounds the other block size through the SMEM
  budget. Which axis *is* blocked depends on where `A` sits:

    - `B == 1`, the contiguous case. The tile is `(block_m, A)` and the
      reduction runs along the tile's minor axis.
    - `B > 1`, the strided case. `A` is no longer contiguous, so a tile can only
      hold one row of `M`: the tile is `(A, block_b)` and the reduction runs
      down its slow axis, crossing warps through the reduction scratch.

  Either way the tiles are handed out to CTAs, and each CTA walks its own share
  as a pipeline.

  Attributes:
    num_m: Rows of `M` in the whole array.
    num_a: The reduced axis, i.e. the width of a row.
    num_b: The axis after the reduced one; 1 in the contiguous case.
    block_m: Rows of `M` per tile. Divides `num_m`; forced to 1 when strided.
    block_b: Columns of `B` per tile. Divides `num_b`; 1 when contiguous.
    itemsize: Bytes per element of `x`.
    swizzle: SMEM swizzle, chosen for the tile's minor axis.
    steps: Tiles in total.
    steps_per_cta: Tiles each CTA walks.
    num_ctas: CTAs, i.e. the `plgpu.kernel` grid.
    num_stages: Pipeline depth.
  """

  num_m: int
  num_a: int
  num_b: int
  block_m: int
  block_b: int
  itemsize: int
  swizzle: int
  steps: int
  steps_per_cta: int
  num_ctas: int
  num_stages: int

  @property
  def strided(self) -> bool:
    """Whether the reduced axis is not the contiguous one."""
    return self.num_b > 1

  @property
  def reduce_axis(self) -> int:
    """The axis of the staged 2D tile that `A` spans, i.e. the reduced one."""
    return 0 if self.strided else 1

  @property
  def tile_shape(self) -> tuple[int, int]:
    """The staged tile, always 2D; see the module docstring."""
    if self.strided:
      return (self.num_a, self.block_b)
    return (self.block_m, self.num_a)

  @property
  def block_shape(self) -> tuple[int | None, ...]:
    """The `BlockSpec` block, with the squeezed axis that makes the copy 2D."""
    if self.strided:
      return (None, self.num_a, self.block_b)
    return (self.block_m, self.num_a)

  @property
  def steps_b(self) -> int:
    """Tiles needed to cover `B`."""
    return self.num_b // self.block_b

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

  @property
  def store_layout(self):
    """Register layout for the relayout that makes the GMEM store coalesce."""
    return plgpu.Layout.SMEM_GMEM_COPY(
      self.tile_shape, jnp.dtype(f'float{self.itemsize * 8}'), swizzle=self.swizzle
    )

  def block_index(self, step) -> jax.Array:
    """Returns the tile this CTA reads on `step`, as a flat tile index.

    The last CTA may be handed fewer than `steps_per_cta` tiles; rather than
    read out of bounds -- which `cp.async` cannot predicate -- it re-reads the
    last tile. Rows are normalized independently, so recomputing one is
    idempotent: the same values are written back over themselves.
    """
    first = jax.lax.axis_index('cta') * self.steps_per_cta
    return jnp.minimum(first + step, self.steps - 1)

  def tile_indices(self, step) -> tuple[jax.Array, jax.Array]:
    """Splits the flat tile index into a block of `M` and a block of `B`."""
    flat = self.block_index(step)
    return flat // self.steps_b, flat % self.steps_b

  def in_spec(self) -> plgpu.BlockSpec:
    def index_map(step):
      i, j = self.tile_indices(step)
      return (i, 0, j) if self.strided else (i, 0)

    return plgpu.BlockSpec(
      self.block_shape, index_map, transforms=self.transforms
    )

  def out_index(self, step) -> tuple[Any, ...]:
    """Returns the slice of the `(M, A, B)` output that this `step` covers."""
    i, j = self.tile_indices(step)
    if self.strided:
      return (i, slice(None), pl.ds(j * self.block_b, self.block_b))
    return (pl.ds(i * self.block_m, self.block_m), slice(None))

  def stat_index(self, step) -> tuple[Any, ...]:
    """Returns the slice of the `(M, B)` statistics that this `step` covers."""
    i, j = self.tile_indices(step)
    if self.strided:
      return (i, pl.ds(j * self.block_b, self.block_b))
    return (pl.ds(i * self.block_m, self.block_m),)

  def smem_bytes(self) -> int:
    tile = self.tile_shape[0] * self.tile_shape[1] * self.itemsize
    return tile * (self.num_stages + 1)  # Staging buffers, plus the store scratch.


def plan(
  num_m: int,
  num_a: int,
  itemsize: int,
  *,
  block_m: int,
  num_b: int = 1,
  block_b: int | None = None,
) -> Plan:
  """Plans a canonical `(M, A, B)` normalization.

  Args:
    num_m: Rows to normalize, i.e. the product of the axes before the reduced
      one.
    num_a: The reduced axis.
    itemsize: Bytes per element of `x`.
    block_m: Upper bound on the rows per tile. Ignored when `num_b > 1`, where a
      tile can only hold one row.
    num_b: The product of the axes after the reduced one.
    block_b: Upper bound on the columns of `B` per tile. Only used when
      `num_b > 1`; defaults to one cache line's worth.

  Whichever axis is blocked, the largest divisor of it at or below the cap that
  is also a multiple of the tiling is used; see the module docstring for why it
  has to divide exactly.
  """
  if num_b > 1:
    # The tile is `(A, block_b)`, so `A` is what the 8-row tiling applies to and
    # `block_b` is what the lanes tile.
    if num_a % _TILING_ROWS:
      raise NotImplementedError(
        f'The reduced axis ({num_a}) must be a multiple of {_TILING_ROWS}, which'
        ' is how the staged tile is tiled, when it is not the contiguous axis.'
      )
    if block_b is None:
      block_b = gpu_utils.CACHE_LINE_SIZE_BYTES // itemsize
    candidates = [
      (1, b) for b in divisors(num_b, block_b, multiple_of=_TILING_COLS)
    ]
  else:
    if num_m % _TILING_ROWS:
      raise NotImplementedError(
        f'Rows ({num_m}) must be a multiple of {_TILING_ROWS}, which is how the'
        ' staged tile is tiled.'
      )
    if num_a % _TILING_COLS:
      raise NotImplementedError(
        f'The reduced axis ({num_a}) must be a multiple of {_TILING_COLS} so'
        ' that the 32 lanes tile it.'
      )
    candidates = [
      (m, 1) for m in divisors(num_m, block_m, multiple_of=_TILING_ROWS)
    ]

  # A whole row has to be resident, so a long one is paid for in the other block
  # size. Take the largest block that fits SMEM rather than declining outright.
  smallest = None
  for candidate_m, candidate_b in candidates:
    steps = (num_m // candidate_m) * (num_b // candidate_b)
    steps_per_cta = min(steps, _STEPS_PER_CTA)
    minor = candidate_b if num_b > 1 else num_a
    p = Plan(
      num_m=num_m,
      num_a=num_a,
      num_b=num_b,
      block_m=candidate_m,
      block_b=candidate_b,
      itemsize=itemsize,
      swizzle=plgpu.find_swizzle(minor * itemsize * 8, 'normalization tile'),
      steps=steps,
      steps_per_cta=steps_per_cta,
      num_ctas=ceil_div(steps, steps_per_cta),
      num_stages=min(2, steps_per_cta),
    )
    if p.smem_bytes() <= _SMEM_BUDGET:
      return p
    smallest = p

  if smallest is None:
    raise NotImplementedError(
      f'The trailing axis ({num_b}) has no divisor that is a multiple of'
      f' {_TILING_COLS} and at most {block_b}, so no tile of it can be staged.'
    )
  rows, cols = smallest.tile_shape
  raise NotImplementedError(
    f'Even a tile of {rows}x{cols} over {smallest.num_stages} stages needs'
    f' {smallest.smem_bytes()} bytes of SMEM, over the budget of'
    f' {_SMEM_BUDGET}. The reduced axis cannot be blocked, so this bounds it.'
  )


def with_usable_block_m(config):
  """Raises a borrowed Triton config's `block_m` to something this kernel can use.

  A block has to be a multiple of 8 rows, so anything below that would leave
  `largest_divisor` nothing to find.
  """
  return dataclasses.replace(config, block_m=max(_TILING_ROWS, config.block_m))
