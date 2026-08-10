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
from typing import Any, NamedTuple

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp


# CUDA allows 2**31-1 blocks in gridDim.x but only 65535 in .y and .z.
_MAX_GRID_DIM = 65535

LANE_SEMANTICS = plgpu.CompilerParams(
  lowering_semantics=plgpu.LoweringSemantics.Lane
)


def largest_divisor(n: int, cap: int) -> int:
  """Returns the largest divisor of `n` that is at most `cap`.

  Always finds one, as 1 divides everything.
  """
  return next(b for b in range(min(cap, n), 0, -1) if n % b == 0)


def warp_aligned_block(rows: int, cap: int) -> int | None:
  """Returns the largest divisor of `rows` at most `cap` and a multiple of 4.

  The four warps of a block come from `M` whenever the reduced axis is the
  contiguous one, so a block of rows that is not a multiple of 4 cannot be
  lowered at all. Returns `None` when `rows` has no such divisor -- 2 rows
  cannot fill four warps however the blocks are chosen.
  """
  return next(
    (b for b in range(min(cap, rows) // 4 * 4, 0, -4) if rows % b == 0), None
  )


def _largest_pow2(cap: int, ok: Callable[[int], bool]) -> int | None:
  """Returns the largest power of two `<= cap` satisfying `ok`, or `None`."""
  v = 1 << (cap.bit_length() - 1)
  while not ok(v):
    if v == 1:
      return None
    v //= 2
  return v


def coalesced_layout(block_slow: int, block_fast: int, itemsize: int):
  """Register layout for a 2D `(slow, fast)` tile, `fast` being contiguous.

  Lanes and the vector both come from `fast`, so a warp covers
  `32 * vector` contiguous elements -- one unbroken memory transaction. The
  warps come from `slow`. Whether the reduction ends up crossing warps (and so
  needing SMEM scratch) depends on which axis is being reduced, and is
  deliberately not what this optimises for.
  """
  if block_slow % 4:
    raise NotImplementedError(
      f'Slow axis ({block_slow}) must be a multiple of 4 for the warps.'
    )
  max_vec = max(1, 16 // itemsize)  # 16-byte (128-bit) register vectors.
  v = _largest_pow2(max_vec, lambda v: block_fast % (32 * v) == 0)
  if v is None:
    raise NotImplementedError(
      f'Contiguous axis ({block_fast}) must be a multiple of 32 so the lanes'
      ' cover it.'
    )
  return plgpu.Layout.TILED(
    plgpu.Tiling(((4, 32 * v), (4, v))), (-2,), (-3,), -1
  )


def load(ref, layout) -> jax.Array:
  """Loads a whole `ref` into registers at `layout`, in `float32`.

  Both kernels reduce in `float32` regardless of the input dtype, as Triton
  does, so the cast belongs here rather than at every call site.
  """
  return plgpu.load(ref, layout=layout, optimized=False).astype(jnp.float32)


class Indices(NamedTuple):
  """How one program indexes the refs it was given.

  Attributes:
    x: Index into an `x`-shaped ref, giving this block's tile.
    stat: Index into a `mean`/`rstddev`-shaped ref.
    dparam: Index into a `dscale`/`doffset` partials ref.
    pid_m: This block's position along `M`, already reassembled if the row grid
      needed two axes.
  """

  x: Any
  stat: Any
  dparam: Any
  pid_m: Any


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Plan:
  """How one canonical `(M, A, B)` normalization is spread over the device.

  `A`, the reduced axis, is never blocked: a whole row lives in registers, so
  the reduction needs no cross-block communication. `M` and `B` are blocked and
  the rest of the extent goes to the grid.

  Attributes:
    num_a: Length of the reduced axis, i.e. the reduction denominator.
    block_m: Rows of `M` per block.
    block_b: Elements of `B` per block.
    x_shape: Shape the `x`-like refs are given, rank-reduced to the case so the
      kernel never indexes a rank away.
    stat_shape: Shape of the `mean`/`rstddev` refs.
    dparam_shape: Shape of the per-block `dscale`/`doffset` partials, one `A`-
      indexed row per block, to be summed over the leading two axes.
    ndim: Rank of a tile in registers. Note this is not `len(x_shape)`: when `B`
      is blocked, `M` is indexed by a scalar and so drops out of the tile.
    layout: Register layout of an `x` tile, or `None` for the 1D case, where the
      default strided layout is both usable and optimal.
    reduce_axis: Axis of the tile holding `A`. For a 1D tile that is its only
      axis, so the reduction lands on a scalar.
    grid: The `plgpu.kernel` grid.
    grid_names: Names of `grid`'s axes. `'m'`/`'b'` index `M`/`B`; a leading
      `'mo'` appears only when `M` needs two grid axes (see `plan`).
    split_m: Stride of the `'mo'` axis, i.e. the size of the `'m'` axis.
  """

  num_a: int
  block_m: int
  block_b: int
  x_shape: tuple[int, ...]
  stat_shape: tuple[int, ...]
  dparam_shape: tuple[int, int, int]
  ndim: int
  layout: Any | None
  reduce_axis: int
  grid: tuple[int, ...]
  grid_names: tuple[str, ...]
  split_m: int

  @property
  def stat_axes(self) -> tuple[int, ...]:
    """The tile axes other than `A`.

    A per-row statistic is indexed by exactly these, and an `A`-indexed value is
    reduced over exactly these, so one tuple serves both. For a 1D tile it is
    empty: the statistic is a scalar and a tile already is one value per `A`.
    """
    return tuple(i for i in range(self.ndim) if i != self.reduce_axis)

  @property
  def stat_layout(self):
    """Layout of a per-row statistic, laid out to match a tile."""
    return None if self.layout is None else self.layout.reduce((self.reduce_axis,))

  @property
  def param_layout(self):
    """Layout of an `A`-indexed param, laid out to match a tile."""
    return None if self.layout is None else self.layout.reduce(self.stat_axes)

  def indices(self) -> 'Indices':
    """Returns how the current program indexes each of the refs."""
    *pid_m_parts, pid_b = map(pl.program_id, range(len(self.grid)))
    pid_m = pid_m_parts[-1]
    if len(pid_m_parts) == 2:
      pid_m += pid_m_parts[0] * self.split_m
    dparam = (pid_m, pid_b)
    m_slice = pl.ds(pid_m * self.block_m, self.block_m)
    if len(self.x_shape) == 1:
      return Indices(slice(None), 0, dparam, pid_m)
    if len(self.x_shape) == 2:
      return Indices(m_slice, m_slice, dparam, pid_m)
    b_slice = pl.ds(pid_b * self.block_b, self.block_b)
    return Indices(
      (pid_m, slice(None), b_slice), (pid_m, b_slice), dparam, pid_m
    )

  def _spread(self, a: jax.Array, shape, dims) -> jax.Array:
    a = jax.lax.broadcast_in_dim(a, shape, dims)
    # A splat source needs no hint; the broadcast rule handles it.
    return a if self.layout is None else plgpu.layout_cast(a, self.layout)

  def bcast(self, a: jax.Array, shape) -> jax.Array:
    """Broadcasts a per-row statistic back over a tile of `shape`."""
    return self._spread(a, shape, self.stat_axes)

  def read_param(self, ref, shape) -> jax.Array:
    """Reads a 1D param, laid out to match a tile of `shape` along `A`."""
    # For a 1D tile this broadcast is the identity: the param already matches.
    return self._spread(load(ref, self.param_layout), shape, (self.reduce_axis,))

  def mean_over_a(self, a: jax.Array) -> jax.Array:
    """Averages a tile over `A`, giving one value per row."""
    return jnp.sum(a, axis=self.reduce_axis) / self.num_a

  def sum_over_rows(self, a: jax.Array) -> jax.Array:
    """Sums a tile over all but `A`, giving one value per `A` element."""
    return jnp.sum(a, axis=self.stat_axes)


def plan(
  x_shape: tuple[int, int, int],
  itemsize: int,
  *,
  block_m: int,
  block_b: int,
  rows_per_element: int | None = None,
  max_regs: int = 256,
) -> Plan:
  """Plans a canonical `(M, A, B)` normalization.

  Args:
    x_shape: The canonical `(M, A, B)` shape, `A` being the reduced axis.
    itemsize: Bytes per element of `x`.
    block_m: Upper bound on `block_m`; the largest usable divisor at or below it
      is used, as blocks must tile the array exactly.
    block_b: Upper bound on `block_b`, likewise.
    rows_per_element: Rows belonging to a single `vmap` element, when `M` holds a
      folded batch. Blocks are then kept from straddling a batch boundary, so
      that per-block partials can still be separated per element.
    max_regs: Register budget per thread, in elements of a single block. The
      real footprint is a small multiple of this -- reductions run in `float32`
      and the kernels keep temporaries live -- so this is a knob to turn if a
      kernel spills, not a hardware limit.

  Returns:
    The `Plan`.
  """
  num_m, num_a, num_b = x_shape
  block_m_cap = block_m
  if num_m == 1 and num_b == 1:
    # Wholly degenerate: the tile is 1D, so reducing its only axis *is* an
    # all-axes reduction -- the one case the strided layout handles, and the
    # params already match the tile so they need no broadcast.
    if num_a % 128:
      raise NotImplementedError(
        f'A 1D input needs its reduced axis ({num_a}) to be a multiple of 128'
        ' for the strided layout.'
      )
    block_m = block_b = 1
    tile_shape, stat_shape = (num_a,), (1,)
    layout, ndim, reduce_axis = None, 1, 0
  elif num_b == 1:
    # `A` is contiguous *and* reduced, so the lanes land on it and the
    # reduction is a lane shuffle. The four warps come from `M`, so the block
    # has to be a multiple of 4 rows -- picking any old divisor below the cap
    # and letting `coalesced_layout` reject it wastes most of the shape space.
    rows = num_m if rows_per_element is None else rows_per_element
    found = warp_aligned_block(rows, block_m_cap)
    if found is None:
      raise NotImplementedError(
        f'No multiple of 4 at or below {min(block_m_cap, rows)} divides the'
        f' {rows} rows, so the warps cannot be filled.'
      )
    block_m, block_b = found, 1
    tile_shape, stat_shape = (num_m, num_a), (num_m,)
    layout, ndim, reduce_axis = coalesced_layout(block_m, num_a, itemsize), 2, 1
  else:
    # `B` is contiguous, so the warps land on `A` and the reduction takes the
    # scratch path. `M` goes entirely to the grid: it adds nothing to
    # coalescing and only inflates the register footprint.
    block_m, block_b = 1, largest_divisor(num_b, block_b)
    tile_shape, stat_shape = (num_m, num_a, num_b), (num_m, num_b)
    # `M` is indexed by a scalar, so the tile is 2D even though the ref is 3D.
    layout, ndim, reduce_axis = coalesced_layout(num_a, block_b, itemsize), 2, 0

  if (regs := block_m * num_a * block_b // 128) > max_regs:
    raise NotImplementedError(
      f'Block ({block_m}, {num_a}, {block_b}) puts {regs} elements in each'
      f' thread, over the budget of {max_regs}, so it will spill.'
    )

  # CUDA caps gridDim.y/.z at 65535 (only .x is 2**31-1) and Mosaic does not
  # map the leading axis to .x, so too many rows aborts the launch with
  # `cuLaunchKernelEx: invalid argument`. Folding a vmap batch into `M`
  # multiplies the row grid by the batch size, which is how you get there.
  grid_m, grid_b = num_m // block_m, num_b // block_b
  split_m, grid, grid_names = 1, (grid_m, grid_b), ('m', 'b')
  if grid_m > _MAX_GRID_DIM:
    split_m = largest_divisor(grid_m, _MAX_GRID_DIM)
    if grid_m // split_m > _MAX_GRID_DIM:
      raise NotImplementedError(
        f'Row grid {grid_m} does not factor into two axes of at most'
        f' {_MAX_GRID_DIM}.'
      )
    grid, grid_names = (grid_m // split_m, split_m, grid_b), ('mo', 'm', 'b')

  return Plan(
    num_a=num_a,
    block_m=block_m,
    block_b=block_b,
    x_shape=tile_shape,
    stat_shape=stat_shape,
    dparam_shape=(grid_m, grid_b, num_a),
    ndim=ndim,
    layout=layout,
    reduce_axis=reduce_axis,
    grid=grid,
    grid_names=grid_names,
    split_m=split_m,
  )


def with_usable_block_m(config):
  """Raises a borrowed Triton config's `block_m` to something this kernel can use.

  `pallas_triton_config.get_heuristics_config` halves `block_m` until the launch
  has enough blocks to fill the device, which for anything but a very tall `M`
  bottoms out at 1. Triton is happy with that; this kernel is not, because when
  the reduced axis is contiguous the four warps of a block come from `M`, so a
  `block_m` below 4 cannot be lowered and the op declines a shape it could
  perfectly well have run at `block_m == 4`.

  This is a floor, not a retuning: it says nothing about what `block_m` *should*
  be, only what the layout can accept. `plan` still picks the largest usable
  divisor at or below it, and the two degenerate cases override it entirely.
  """
  return dataclasses.replace(config, block_m=max(4, config.block_m))


def broadcast_unbatched(args, in_batched, axis_size):
  """Gives every argument a leading batch axis, materializing the missing ones.

  A `custom_vmap` rule gets its batched arguments with the mapped axis at the
  front, but `vmap` leaves others alone -- the cotangent of a `sum` is a
  broadcast scalar, for one -- and folding or mapping needs them all uniform.
  """
  return [
    a if b else jnp.broadcast_to(a[jnp.newaxis], (axis_size, *a.shape))
    for a, b in zip(args, in_batched)
  ]
