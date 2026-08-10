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


def largest_divisor(n: int, cap: int) -> int:
  """Returns the largest divisor of `n` that is at most `cap`."""
  return next(b for b in range(min(cap, n), 0, -1) if n % b == 0)


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
    layout: Register layout of an `x` tile, or `None` for the 1D case, where the
      default strided layout is both usable and optimal.
    reduce_axis: Axis of the tile holding `A`; `None` means every axis (the 1D
      case), which reduces to a scalar.
    stat_dims: `broadcast_in_dim` dims mapping a per-row statistic to a tile.
    param_dims: `broadcast_in_dim` dims mapping an `A`-indexed param to a tile.
    param_axes: Tile axes to reduce to get one value per `A` element -- the
      complement of `reduce_axis`, and `None` in the 1D case, where a tile
      already is one value per `A` element.
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
  layout: Any | None
  reduce_axis: int | None
  stat_dims: tuple[int, ...]
  param_dims: tuple[int, ...] | None
  param_axes: tuple[int, ...] | None
  grid: tuple[int, ...]
  grid_names: tuple[str, ...]
  split_m: int

  @property
  def stat_layout(self):
    """Layout of a per-row statistic, laid out to match a tile."""
    if self.layout is None:
      return None
    return self.layout.reduce((self.reduce_axis,))

  @property
  def param_layout(self):
    """Layout of an `A`-indexed param, laid out to match a tile."""
    if self.layout is None:
      return None
    return self.layout.reduce(self.param_axes)

  def indices(self):
    """Returns `(x_idx, stat_idx, dparam_idx)` for the current program."""
    *pid_m_parts, pid_b = map(pl.program_id, range(len(self.grid)))
    pid_m = pid_m_parts[-1]
    if len(pid_m_parts) == 2:
      pid_m += pid_m_parts[0] * self.split_m
    dparam_idx = (pid_m, pid_b)
    m_slice = pl.ds(pid_m * self.block_m, self.block_m)
    if len(self.x_shape) == 1:
      return slice(None), 0, dparam_idx
    if len(self.x_shape) == 2:
      return m_slice, m_slice, dparam_idx
    b_slice = pl.ds(pid_b * self.block_b, self.block_b)
    return (pid_m, slice(None), b_slice), (pid_m, b_slice), dparam_idx

  def bcast(self, a: jax.Array, shape, dims) -> jax.Array:
    """Broadcasts a reduced value back over a tile of `shape`."""
    a = jax.lax.broadcast_in_dim(a, shape, dims)
    # A splat source needs no hint; the broadcast rule handles it.
    return a if self.layout is None else plgpu.layout_cast(a, self.layout)

  def read_param(self, ref, shape) -> jax.Array:
    """Reads a 1D param, laid out to match a tile of `shape` along `A`."""
    p = load(ref, self.param_layout)
    if self.layout is None:  # 1D tile: the param already matches it.
      return p
    return self.bcast(p, shape, self.param_dims)

  def mean_over_a(self, a: jax.Array) -> jax.Array:
    """Averages a tile over `A`, giving one value per row."""
    return jnp.sum(a, axis=self.reduce_axis) / self.num_a

  def sum_over_rows(self, a: jax.Array) -> jax.Array:
    """Sums a tile over all but `A`, giving one value per `A` element."""
    return a if self.param_axes is None else jnp.sum(a, axis=self.param_axes)


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
    block_m: Upper bound on `block_m`; the largest divisor of `M` at or below it
      is used, as blocks must tile the array exactly.
    block_b: Upper bound on `block_b`, likewise.
    max_regs: Register budget per thread, in elements of a single block. The
      real footprint is a small multiple of this -- reductions run in `float32`
      and the kernels keep temporaries live -- so this is a knob to turn if a
      kernel spills, not a hardware limit.

  Returns:
    The `Plan`.
  """
  num_m, num_a, num_b = x_shape
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
    layout, reduce_axis, stat_dims, param_dims, param_axes = (
      None, None, (), None, None
    )
  elif num_b == 1:
    # `A` is contiguous *and* reduced, so the lanes land on it and the
    # reduction is a lane shuffle.
    block_m, block_b = largest_divisor(num_m, block_m), 1
    tile_shape, stat_shape = (num_m, num_a), (num_m,)
    layout = coalesced_layout(block_m, num_a, itemsize)
    reduce_axis, stat_dims, param_dims, param_axes = 1, (0,), (1,), (0,)
  else:
    # `B` is contiguous, so the warps land on `A` and the reduction takes the
    # scratch path. `M` goes entirely to the grid: it adds nothing to
    # coalescing and only inflates the register footprint.
    block_m, block_b = 1, largest_divisor(num_b, block_b)
    tile_shape, stat_shape = (num_m, num_a, num_b), (num_m, num_b)
    layout = coalesced_layout(num_a, block_b, itemsize)
    reduce_axis, stat_dims, param_dims, param_axes = 0, (1,), (0,), (1,)

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
    layout=layout,
    reduce_axis=reduce_axis,
    stat_dims=stat_dims,
    param_dims=param_dims,
    param_axes=param_axes,
    grid=grid,
    grid_names=grid_names,
    split_m=split_m,
  )


def vmappable(launch):
  """Folds a `vmap` axis into the leading axis of the canonical `(M, A, B)`.

  `batching.capture_batched_args` records the batch sizes for the heuristics
  but still calls `jax.vmap` over `_fwd` (see `batching.py`), so without this
  JAX batches the `plgpu.kernel` launch itself. `pallas_call` has a batching
  rule that handles that well -- which is why `pallas_triton` needs nothing --
  but `plgpu.kernel` does not, and it measured ~2.7x worse per element than a
  single larger launch.

  Rows of the canonical form are normalized independently, so a batch of them
  is just more rows. Blocks may straddle a batch boundary, which is harmless
  precisely because the rows are independent -- but only while `scale`/`offset`
  are shared, which is the only case handled here.

  This is for the forward only; the VJP uses `vmappable_serially`. Every output
  here is indexed by row, so folding is just a reshape either way, whereas the
  VJP's `dscale`/`doffset` are reduced *over* rows.

  Args:
    launch: Takes `x` in canonical form followed by the shared params, and
      returns arrays whose leading axis indexes rows of `x`.

  Returns:
    `launch`, wrapped in a `custom_vmap` rule that folds rather than batches.
  """
  f = jax.custom_batching.custom_vmap(launch)

  @f.def_vmap
  def _rule(axis_size, in_batched, x, *params):
    x_batched, *params_batched = in_batched
    if any(params_batched):
      raise NotImplementedError(
        'The Mosaic GPU kernel does not support batched `scale`/`offset`.'
      )
    if not x_batched:
      out = f(x, *params)
      return out, jax.tree.map(lambda _: False, out)
    out = f(x.reshape(axis_size * x.shape[1], *x.shape[2:]), *params)
    unmerge = lambda o: o.reshape(axis_size, o.shape[0] // axis_size, *o.shape[1:])
    return jax.tree.map(unmerge, out), jax.tree.map(lambda _: True, out)

  return f


def vmappable_serially(launch):
  """Runs `launch` once per `vmap` element instead of once for the whole batch.

  This is the VJP's counterpart to `vmappable`, and it exists because all three
  cheaper options are wrong:

  * Folding the batch into `M` sums `dscale`/`doffset` over it as well -- right
    for `vjp(vmap(f))`, where the params are shared, and wrong for
    `vmap(grad(f))`, which wants a gradient per batch element. The rule cannot
    tell the two apart, and the wrong answer is silent: JAX broadcasts the summed
    value, so every element receives the batch total.
  * Leaving it to JAX's generic batching prepends a grid axis, which renumbers
    the `pl.program_id`s that `Plan.indices` reads. Only the first row-block of
    each element then gets written, and the rest of `dx` keeps whatever value the
    incoming cotangent had.
  * Sharing one param buffer across the batch is not available either: `scale`
    arrives batched regardless of the caller's `in_axes`, because `op.Op` keeps
    it in the `custom_vjp` residuals, which JAX batches wholesale.

  So each element gets its own launch, via `jax.lax.map` -- one kernel in the
  HLO however large the batch, rather than an unrolled one per element. Each
  launch still covers `M // block_m` blocks, normally more than enough to fill
  the device, so serializing a handful costs less than it sounds. Returning the
  results per element is what makes both call patterns come out right: for a
  shared param, transposing its broadcast sums the batch for us.

  Args:
    launch: Takes the arrays for one element and returns a tuple of arrays.

  Returns:
    `launch`, wrapped in a `custom_vmap` rule that maps rather than batches.
  """
  f = jax.custom_batching.custom_vmap(launch)

  @f.def_vmap
  def _rule(axis_size, in_batched, *args):
    if not any(in_batched):
      out = f(*args)
      return out, jax.tree.map(lambda _: False, out)
    # `lax.map` maps over every argument, and `vmap` leaves some unbatched -- the
    # cotangent of a `sum` is a broadcast scalar, for one.
    args = [
      a if b else jnp.broadcast_to(a[jnp.newaxis], (axis_size, *a.shape))
      for a, b in zip(args, in_batched)
    ]
    out = jax.lax.map(lambda args: launch(*args), tuple(args))
    return out, jax.tree.map(lambda _: True, out)

  return f
