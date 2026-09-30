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
"""Pallas-Mosaic-GPU normalization op.

Normalization is memory-bound, so the design serves coalesced GMEM access: one
CTA takes one tile straight from GMEM into registers, reduces it, and writes it
back. No SMEM, no pipeline.
"""

import dataclasses
import math
from typing import Any, ClassVar, override
from collections.abc import Sequence

import immutabledict
import jax
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.mosaic.gpu import TiledLayout
from jax.experimental import pallas as pl
import jax.numpy as jnp
import pydantic
from tokamax._src import gpu_utils
from tokamax._src import pydantic as pydantic_lib
from tokamax._src.ops import op
from tokamax._src.ops.normalization import base


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """The block shape. `block_n` is `None` when the reduced axis is trailing."""

  block_m: pydantic_lib.PowerOfTwo
  block_n: pydantic_lib.PowerOfTwo | None


# There is no autotuning cache for these ops, so the keys are only what `Op`
# builds out of the arguments by default. The VJP takes the same block shape as
# the forward, so it shares `Config` too.
type Key = immutabledict.immutabledict[str, Any]
FusedInputArray = base.FusedInputArray

def _vector_lengths(block_n: int, A: int, bitwidth: int):
  """The legal per-thread vector lengths, widest first.

  The kernel is memory-bound, so we want the widest load the hardware offers
  (16 bytes = 128 bits per thread). A vector is only legal if its elements are
  adjacent in memory: `x` is (M, A, N) with N innermost, so a vector either sits
  inside one N row (`block_n % vec == 0`) or covers whole N rows and spills into
  the A axis (`vec % block_n == 0`, with A divisible so the spill never crosses
  a block boundary). Neither holds -> halve and retry, down to a scalar load.
  """
  vec = (8 * 16) // bitwidth  # 16-byte vectors.
  while vec >= 1:
    if block_n % vec == 0 or (vec % block_n == 0 and A % vec == 0):
      yield vec
    vec //= 2

def _vector_length(block_n: int, A: int, bitwidth: int) -> int:
  """The widest legal vector, ignoring what it costs in rows."""
  return next(iter(_vector_lengths(block_n, A, bitwidth)))

def _lanes_along_a(vec: int, A: int) -> int | None:
  """How many of the 32 lanes go along A; the rest spread along M.

  A lane holds `A // la` elements of the reduced axis, which the vector then has
  to tile, so `la` must divide A and leave a multiple of `vec` behind. A is not
  always obliging -- A=40 with a 4-element vector admits la=2 at most, and A=42
  only la=1 -- and every lane A cannot take is a lane M has to, which is what
  makes `block_m` the heuristic's problem. See `_min_block_m`. `None` when no
  split works, which is a narrower vector's cue.
  """
  for la in (32, 16, 8, 4, 2, 1):
    if A % la == 0 and (A // la) % vec == 0:
      return la
  return None

def _lanes_along_n(vec: int, A: int, N: int) -> int | None:
  """How many of the 32 lanes go along N; the rest go along A.

  N first, since lanes adjacent along N issue one coalesced transaction, whereas
  lanes along A add reduction shuffles; we take the largest `ln` that divides
  evenly and only fall back to A for the leftovers. Divisibility is required
  both ways because the tiling has no support for partial tiles. `None` when no
  split works, which is a narrower vector's cue.
  """
  for ln in (32, 16, 8, 4, 2, 1):
    if N % (ln * vec) == 0 and A % (32 // ln) == 0:
      return ln
  return None

def _vector_for(block_n: int, bitwidth: int, M: int, A: int, N: int) -> int:
  """The widest legal vector whose lane split tiles one warp's block.

  Width is not free. The wider the vector, the fewer lanes are left for the axis
  it runs along, and the leftovers have to divide what remains: the warp's rows
  when the vector spans whole N rows, its N when it sits inside one. Narrowing
  is what lets an awkward block tile at all -- A=40 with 8 rows to a warp, A=24
  with N=8 -- and it costs a narrower load, nothing else. The search runs widest
  first and crosses from one branch to the other as `vec` falls below `block_n`.
  """
  for vec in _vector_lengths(block_n, A, bitwidth):
    if vec >= block_n:
      la = _lanes_along_a(vec, A)
      if la is not None and M % (32 // la) == 0:
        return vec
    elif _lanes_along_n(vec, A, N) is not None:
      return vec
  raise ValueError(f'Cannot tile {M=}, {A=}, {N=} with any vector.')

def _vec_along_a(vec: int, M: int, A: int, N: int) -> tuple[int, int, int]:
  """Tiles a warp's block when the vector spans whole N rows (vec >= block_n).
  The 32 lanes go along the reduced axis, as many as it can take, and the ones
  left over once it is exhausted spread along M.
  """
  la = _lanes_along_a(vec, A)
  if la is None or M % (32 // la):
    raise ValueError(f'Cannot spread 32 lanes over {M=}, {A=} with {vec=}.')
  return (M // (32 // la), A // la, N)

def _vec_along_n(vec: int, M: int, A: int, N: int) -> tuple[int, int, int]:
  """Tiles a warp's block when the vector fits inside an N row (vec < block_n).
  The 32 lanes go along N, as many as it can take, and the ones left over spread
  along the reduced axis.
  """
  ln = _lanes_along_n(vec, A, N)
  if ln is None:
    raise ValueError(f'Cannot spread 32 lanes over {A=}, {N=} with {vec=}.')
  return (M, A // (32 // ln), N // ln)

def _warp_blocks(block_m, block_n, a=1, split_a=False):
    """Splits the block over the warpgroup's 4 warps. Returns (m, a, n) counts.

    The forward only reduces A, so it splits M and/or N and keeps its reduction
    in-warp (`split_a=False`, the default).

    The VJP also reduces over M and N, for `dscale`/`doffset`. Splitting M or N
    leaves those partials warp-replicated and all four warps store the same
    values to the same addresses.
    """
    if split_a and a % 4 == 0:
      return (1, 4, 1)
    for amt_m in [4,2,1]:
      amt_n = 4 // amt_m
      if block_m % amt_m == 0 and block_n % amt_n == 0:
        return (amt_m, 1, amt_n)
    raise ValueError(f'Cannot find warp distribution for block_m {block_m}, block_n {block_n}.')

def _tiled_layout(
  block_m: int, a: int, block_n: int, bitwidth: int, split_a: bool = False
):
  """Builds the register layout for one (block_m, a, block_n) block.

  Shared by the forward and VJP kernels: both stream the same block shape
  straight from GMEM and reduce along `a`, so they want the same tiling.

  The layout has 14 dims. Each `plgpu.Tiling` entry replaces the trailing dims
  it covers with (how many tiles fit, tile shape), so the rank grows
  3 -> 6 -> 9 -> 12 -> 14 as we tile by block, by warp, by lane, and finally by
  vector. `warp_dims`/`lane_dims`/`vector_dim` index that final rank from the
  end; the warp dims always multiply to 4 and the lane dims to 32.

  Shared by both branches (tile 0, the CTA block; tile 1, the warp block):
    -14, -13, -12  block counts along M, A, N. All 1: the grid has already
                   selected a single block, and A is never blocked.
    -11, -10,  -9  warp counts along M, A, N, all three of which are WARP dims
                   (their product is always 4, and `canonicalize()` drops
                   whichever are 1). The forward splits M and/or N, leaving -10
                   at 1; `split_a` puts all four warps on A instead. See
                   `_warp_blocks` for why the choice matters.

  vec >= block_n (`_vec_along_a`; the vector spans whole N rows):
     -8,  -7,  -6  lane counts along M, A, N. LANE dims are -8 and -7 --
                   lanes fill A first and spill into M; N is consumed
                   entirely by the vector, so -6 is 1.
     -5           M extent held by one lane.
     -4,  -3      vectors along A per lane, and N chunks per lane (1).
     -2           VECTOR dim: `vec` contiguous elements along A.
     -1           the `block_n` elements of N each of those rows spans.

  vec < block_n (`_vec_along_n`; the vector sits inside one N row):
     -8,  -7,  -6  lane counts along M, A, N. LANE dims are -7 and -6 --
                   `la` lanes along A times `ln` lanes along N; lanes never
                   split M here, so -8 is 1.
     -5           M extent held by one lane (all of the warp's M).
     -4,  -3      A chunks per lane and vectors along N per lane.
     -2           the A extent one lane holds (A // la).
     -1           VECTOR dim: `vec` contiguous elements along N.

  `canonicalize()` then collapses the size-1 dims, which is why the layout is
  read back off `l` rather than reusing the values passed in.
  """
  warp_m, warp_a, warp_n = _warp_blocks(block_m, block_n, a, split_a)
  tile_spec = [
    (block_m, a, block_n),
    (block_m // warp_m, a // warp_a, block_n // warp_n),
  ]
  vec = _vector_for(block_n, bitwidth, *tile_spec[-1])
  if vec >= block_n:
    tile_spec.append(_vec_along_a(vec, *tile_spec[-1]))
    vector_dim = -2
    lane_dims = (-8, -7)  # The M and A tile counts.
    tile_spec.append((vec, block_n),)
  else:
    tile_spec.append(_vec_along_n(vec, *tile_spec[-1]))
    vector_dim = -1
    lane_dims = (-7, -6)  # The A and N tile counts.
    tile_spec.append((tile_spec[-1][1], vec),)

  l = TiledLayout(
    plgpu.Tiling(tuple(tile_spec)),
    warp_dims=(-11, -10, -9),
    lane_dims=lane_dims,
    vector_dim=vector_dim,
    _check_canonical=False).canonicalize()
  return plgpu.Layout.TILED(l.tiling, warp_dims=l.warp_dims,
                            lane_dims=l.lane_dims, vector_dim=l.vector_dim)

def canonicalize_shape_3d(
    shape: Sequence[int], axis: int
) -> tuple[int, int, int]:
  return (math.prod(shape[:axis]), shape[axis], math.prod(shape[axis:][1:]))

def _min_block_m(A: int, bitwidth: int, rows: int) -> int:
  """The fewest rows a block can have, with no N to block.

  All four warps go along M (`_warp_blocks` has nowhere else to put them), and
  so do the `32 // la` lanes A had no room for, so the block needs one row per
  warp per leftover lane before the layout can tile it at all. At the widest
  vector that is 8 rows for A=64, 64 for A=40, 128 for A=42.

  A narrower vector leaves A room for more lanes and so asks for fewer rows,
  which is the trade to make when `rows` cannot cover the widest one: A=40 wants
  64 rows at 16-byte loads and 32 at 8-byte. We take the widest vector the rows
  can afford, and fall back to the widest one outright when none fits, leaving
  `_launch` to pad.
  """
  floors = []
  for vec in _vector_lengths(1, A, bitwidth):
    la = _lanes_along_a(vec, A)
    if la is None:
      continue
    floors.append(4 * (32 // la))
    if floors[-1] <= rows:
      return floors[-1]
  return floors[0]

def _heuristics_config(x, scale, offset, *, axis, vmap_axis_sizes) -> Config:
  """Picks the block shape.

  The kernel is always a single warpgroup, so there is no warp count to pick.
  """
  m, a, n = canonicalize_shape_3d(x.shape, axis)
  # A block never exceeds the axis it tiles, and stays a power of two.
  prev_power_of_2 = lambda s: 1 << (s.bit_length() - 1)

  if len(x.shape[axis:]) > 1:
    if n % 128 == 0:
      block_n = 128  # 128 divided by 4 warps allows for 32 lanes per warp.
    else:
      # Read a full cache line at a time.
      els_per_cache_line = (
          gpu_utils.CACHE_LINE_SIZE_BYTES // jnp.dtype(x.dtype).itemsize
      )
      block_n = min(els_per_cache_line, prev_power_of_2(n))
    # Blocking N already gives each block a full cache line, so there is nothing
    # left for `block_m > 1` to re-use.
    return Config(block_m=1, block_n=block_n)

  # Reducing the trailing axis: there is no N to block, so the only re-use of
  # `scale`/`offset` across rows comes from M. Halve `block_m` until the block
  # fits in registers and enough blocks are launched to fill the device, but
  # never below what the tiling needs: M is carrying the warps and whatever
  # lanes A could not take.
  # A shape with fewer rows than the floor gets the floor anyway, and `_grid`
  # declines it. Raising here would pre-empt the `vmap` rule in `_fwd`, which
  # is the one that can still find the rows, in the batch axes.
  min_block_m = _min_block_m(a, jnp.dtype(x.dtype).itemsize * 8, m)
  block_m = min_block_m if (scale is None and offset is None) else max(
      min_block_m, min(32, prev_power_of_2(m))
  )
  block_size = block_m * pl.next_power_of_2(a)
  num_blocks = pl.cdiv(m, block_m) * math.prod(vmap_axis_sizes)
  max_block_size = gpu_utils.NUM_REGISTERS_PER_SM // 4
  min_num_blocks = 4 * jax.devices()[0].core_count
  while (block_m > min_block_m) and (
      (block_size > max_block_size) or (num_blocks < min_num_blocks)
  ):
    block_m //= 2
    block_size //= 2
    num_blocks *= 2
  return Config(block_m=block_m, block_n=None)

def _grid(x_shape, block) -> tuple[int, ...]:
  """Returns the launch grid, declining blocks the launch cannot express.

  A block wider than its axis is a shape this kernel has no tiling for, not a
  bug in the caller, so it declines and lets another impl take it.
  """
  for (s, b) in zip(x_shape, block):
    if s < b:
      raise NotImplementedError(
        f'Block {b} is larger than the axis it tiles ({s}).'
      )
  return tuple(pl.cdiv(s, b) for (s, b) in zip(x_shape, block))

def _block_index(x_shape, block, idx):
  """Start offsets for one CTA's block, clamped to keep the block in bounds.

  Shapes need not be multiples of the block: a trailing partial block is shifted
  back to end at the array's edge, so it overlaps its predecessor and recomputes
  the shared elements.
  """
  return tuple(
    pl.ds(jnp.minimum(i * b, s - b), b) for i, s, b in zip(idx, x_shape, block)
  )

@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicGpuNormalization(base.Normalization[Config, Key]):
  """Pallas-Mosaic-GPU normalization op."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  input_output_alias: bool | None = None

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, 'vjp', PallasMosaicGpuNormalizationVjp())

  @override
  def _fwd(
    self,
    x: jax.Array | FusedInputArray,
    scale: jax.Array | None,
    offset: jax.Array | None,
    *,
    axis: int,
    epsilon: float,
    scale_offset: float,
    subtract_mean: bool,
    return_residuals: bool,
    config: Config,
  ) -> tuple[jax.Array, base.Residuals | None]:
    """Launches the kernel, folding `vmap`'s axes into M where they can be.

    Left to itself, `vmap` gives each batch element its own grid slot, so the
    block is left tiling the inner rows alone -- fewer than the layout needs,
    for an A the lanes cannot cover on their own (see `_min_block_m`). But the
    batch axis lands left of the reduced one, where `canonicalize_shape_3d`
    folds it into M, and the block then draws rows from it like any other.

    Only `x` folds. A batched `scale`/`offset` varies along the rows one block
    spans and the kernel has a single param vector per launch, so those keep
    `vmap`'s own rule.
    """
    if callable(x):
      x = x()

    rest = dict(
      epsilon=epsilon,
      scale_offset=scale_offset,
      subtract_mean=subtract_mean,
      return_residuals=return_residuals,
    )

    def with_vmap(axis, config):
      def launch(x, scale, offset):
        return self._launch(x, scale, offset, axis=axis, config=config, **rest)

      fwd = jax.custom_batching.custom_vmap(launch)

      def vmap_rule(axis_size, in_batched, x, scale, offset):
        del axis_size
        x_batched, *params_batched = in_batched
        if x_batched and not any(jax.tree.leaves(params_batched)):
          # The batch arrives at axis 0, so the reduced axis has shifted right.
          # The config goes back to `None` to be re-derived: the one we were
          # handed describes the shape as it was before the fold. Recursing
          # through `with_vmap` leaves the new call batchable in turn, which is
          # what lets a second `vmap` fold its axis in as well.
          new_axis = axis + 1 if axis >= 0 else axis
          out = with_vmap(new_axis, None)(x, scale, offset)
        else:
          in_axes = [0 if b else None for b in in_batched]
          out = jax.vmap(launch, in_axes=in_axes)(x, scale, offset)
        return out, jax.tree.map(lambda _: True, out)

      fwd.def_vmap(vmap_rule)
      return fwd

    return with_vmap(axis, config)(x, scale, offset)

  def _launch(
    self,
    x: jax.Array,
    scale: jax.Array | None,
    offset: jax.Array | None,
    *,
    axis: int,
    epsilon: float,
    scale_offset: float,
    subtract_mean: bool,
    return_residuals: bool,
    config: Config | None,
  ) -> tuple[jax.Array, base.Residuals | None]:
    """One kernel launch. `config` is `None` when it has to be re-derived."""
    if config is None:
      config = _heuristics_config(
        x, scale, offset, axis=axis, vmap_axis_sizes=()
      )

    dtype = x.dtype
    orig_x_shape = x.shape
    x_shape = canonicalize_shape_3d(orig_x_shape, axis)

    return_mean = return_residuals and subtract_mean
    has_scale = scale is not None
    has_offset = offset is not None

    A = x_shape[1]
    block = (config.block_m, A, config.block_n or 1)
    block_m, _, block_n = block

    layout = _tiled_layout(block_m, A, block_n, dtype.itemsize * 8)

    # `_block_index` shifts a trailing partial block back to end at the array's
    # edge, which needs a whole block to shift within. Too few rows for even one
    # -- a shape below `_min_block_m`, which `vmap` hands over all the time --
    # and the rows are padded up to a block.
    rows = x_shape[0]
    pad_m = max(0, block_m - rows)
    x = x.reshape(x_shape)
    if pad_m:
      x = jnp.pad(x, ((0, pad_m), (0, 0), (0, 0)))
      x_shape = (block_m,) + x_shape[1:]

    # A trailing partial block re-reads rows a neighbouring CTA writes, so
    # writing `y` over `x` would race; fall back to a separate output. Padding
    # makes `x` a fresh buffer, so there is nothing left worth aliasing either.
    alias = (
      bool(self.input_output_alias)
      and not pad_m
      and all(s % b == 0 for (s, b) in zip(x_shape, block))
    )
    # A ref is written in place, so the kernel writes `y` over `x` and XLA gets
    # to drop the copy `new_ref` starts with whenever `x` is dead afterwards.
    x_operand = jax.new_ref(x) if alias else x

    def kernel(*refs):
      it = iter(refs)  # Inputs then outputs, optional ones only if present.
      take = lambda present: next(it) if present else None
      x_gmem, scale_ref, offset_ref = next(it), take(has_scale), take(has_offset)
      # When aliasing, `x_gmem` is a mutable ref and there is no `y` output:
      # each CTA has already read its tile into registers by the time it writes.
      y_gmem = x_gmem if alias else next(it)
      mean_gmem = take(return_mean)
      rstd_gmem = take(return_residuals)

      index = _block_index(
        x_shape, block, [jax.lax.axis_index(i) for i in 'man']
      )

      stat_index = index[:1] + index[2:]
      x = plgpu.load(x_gmem.at[index], layout=layout, optimized=False).astype(jnp.float32)

      if subtract_mean:
        mean = jnp.mean(x, axis=1)
        x -= jax.lax.broadcast_in_dim(mean, block, (0,2))
        if mean_gmem is not None:
          mean_gmem[stat_index] = mean
      rstddev = jax.lax.rsqrt(jnp.mean(x * x, axis=1) + epsilon)
      if rstd_gmem is not None:
        rstd_gmem[stat_index] = rstddev

      x = x * jax.lax.broadcast_in_dim(rstddev, block, (0, 2))
      if scale_ref is not None:
        scale = jax.lax.broadcast_in_dim(plgpu.load(scale_ref, optimized=False).astype(jnp.float32), block, (1,))
        x *= scale + scale_offset
      if offset_ref is not None:
        offset = jax.lax.broadcast_in_dim(plgpu.load(offset_ref, optimized=False).astype(jnp.float32), block, (1,))
        x += offset
      y_gmem[index] = x.astype(dtype)

    stat = jax.ShapeDtypeStruct(x_shape[:1] + x_shape[2:], jnp.float32)
    grid = _grid(x_shape, block)
    outs = plgpu.kernel(
      kernel,
      out_type=(
        *([] if alias else [jax.ShapeDtypeStruct(x_shape, dtype)]),
        *[stat] * (return_mean + return_residuals),
      ),
      grid=grid,
      grid_names=('m', 'a', 'n'),
      kernel_name=f"mosaic_norm_fwd_{dtype.name}_m{block_m}_n{block_n}{'_mean' if subtract_mean else ''}",
      compiler_params=plgpu.CompilerParams(
        lowering_semantics=plgpu.LoweringSemantics.Warpgroup)
    )(
      x_operand,
      *[a for a in (scale, offset) if a is not None],
    )

    if alias:
      y, stats = jax.freeze(x_operand), outs
    else:
      y, *stats = outs

    unpad = lambda a: a[:rows] if pad_m else a
    y = unpad(y).reshape(orig_x_shape)
    if not return_residuals:
      return y, None

    stat_shape = list(orig_x_shape)
    stat_shape[axis] = 1
    mean = unpad(stats[0]).reshape(stat_shape) if return_mean else None
    rstddev = unpad(stats[-1]).reshape(stat_shape)
    return y, (mean, rstddev)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return _heuristics_config(
      *ba.args, axis=ba.kwargs['axis'], vmap_axis_sizes=ba.vmap_axis_sizes
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicGpuNormalizationVjp(base.NormalizationVjp[Config, Key]):
  """Pallas-Mosaic-GPU normalization VJP.

  Same shape as the forward kernel: one CTA takes one (block_m, A, block_n)
  tile of `x` and `dout` straight from GMEM into registers, reduces along A,
  and writes `dx` back. `dscale`/`doffset` reduce over M and N as well, which
  spans CTAs, so each CTA writes a partial and the final sum is left to XLA.
  """

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _fwd(
    self,
    residuals: base.Residuals,
    out: jax.Array,
    dout: jax.Array,
    x: jax.Array,
    scale: jax.Array | None,
    offset: jax.Array | None,
    *,
    axis: int,
    epsilon: float,
    scale_offset: float,
    subtract_mean: bool,
    return_residuals: bool,
    config: Config,
  ) -> tuple[tuple[jax.Array, jax.Array | None, jax.Array | None], None]:

    del epsilon  # Unused: `rstddev` comes from the residuals, not recomputed.

    if return_residuals:
      raise NotImplementedError('`return_residuals` not supported.')

    mean, rstddev = residuals
    if (mean is not None) != subtract_mean:
      raise ValueError('`mean` residual inconsistent with `subtract_mean`.')

    dtype = x.dtype
    orig_x_shape = x.shape
    x_shape = canonicalize_shape_3d(orig_x_shape, axis)

    stat_shape = (x_shape[0], x_shape[2])
    if mean is not None:
      mean = mean.reshape(stat_shape)
    rstddev = rstddev.reshape(stat_shape)

    has_scale = scale is not None
    has_offset = offset is not None

    A = x_shape[1]
    block = (config.block_m, A, config.block_n or 1)
    block_m, _, block_n = block
    grid = _grid(x_shape, block)
    grid_n = grid[2]

    vec_bitwidth = 32
    layout = _tiled_layout(block_m, A, block_n, vec_bitwidth, split_a=True)

    # Reductions across singleton dimensions are not supported
    reduced = tuple(ax for ax in (2, 0) if block[ax] > 1)
    kept = tuple(ax for ax in range(3) if ax not in reduced)
    # Only the axes with a clamped trailing block need the dedup mask.
    dup_axes = tuple(ax for ax in reduced if x_shape[ax] % block[ax])

    def kernel(*refs):
      it = iter(refs)  # Inputs then outputs, optional ones only if present.
      take = lambda present: next(it) if present else None
      dout_gmem, x_gmem, scale_ref = next(it), next(it), take(has_scale)
      mean_gmem, rstd_gmem = take(subtract_mean), next(it)
      dx_gmem = next(it)
      dscale_gmem, doffset_gmem = take(has_scale), take(has_offset)

      m, _, n = [jax.lax.axis_index(i) for i in 'man']
      index = _block_index(x_shape, block, (m, 0, n))
      # How far each axis' block was shifted back to stay in bounds; the first
      # `dup` elements along that axis were already covered by the CTA before.
      dups = [i * b - s.start for (i, b, s) in zip((m, 0, n), block, index)]
      load = lambda ref: plgpu.load(
        ref.at[index], layout=layout, optimized=False
      ).astype(jnp.float32)
      bcast = lambda a: jax.lax.broadcast_in_dim(a, block, (0, 2))

      stat_index = index[:1] + index[2:]
      stat = lambda ref: plgpu.load(
        ref.at[stat_index], optimized=False
      ).astype(jnp.float32)

      rstddev = stat(rstd_gmem)
      x_norm = load(x_gmem)
      if mean_gmem is not None:
        x_norm -= bcast(stat(mean_gmem))
      x_norm *= bcast(rstddev)

      dout = load(dout_gmem)

      def reduce_mn(a): # Multi axis reduce on mosaic not yet implemented
        for ax in dup_axes:  # Count the re-read elements once, not twice.
          iota = plgpu.broadcasted_iota(jnp.int32, block, ax, layout=layout)
          a = jnp.where(iota >= dups[ax], a, 0.0)
        for ax in reduced:
          a = jnp.sum(a, axis=ax)  # N first, so M stays at axis 0.
        return a

      # Each CTA owns one (m, n) grid cell, hence one `A`-long run of the
      # partials; any degenerate axis `reduce_mn` left on is indexed away.
      dparam_index = tuple(
        pl.ds((m * grid_n + n) * A, A) if ax == 1 else pl.ds(0, 1)
        for ax in kept
      )

      if doffset_gmem is not None:
        doffset_gmem[dparam_index] = reduce_mn(dout)
      if dscale_gmem is not None:
        dscale_gmem[dparam_index] = reduce_mn(dout * x_norm)
        loaded = plgpu.load(scale_ref, optimized=False).astype(jnp.float32)
        dout *= jax.lax.broadcast_in_dim(loaded, block, (1,)) + scale_offset

      dx = dout - bcast(jnp.mean(dout * x_norm, axis=1)) * x_norm
      if mean_gmem is not None:
        dx -= bcast(jnp.mean(dout, axis=1))
      dx_gmem[index] = (dx * bcast(rstddev)).astype(dtype)

    # Shaped to match what `reduce_mn` leaves behind, with the A axis stacked
    # one run per (m, n) grid cell.
    dparam_shape = tuple(
      grid[0] * grid_n * A if ax == 1 else 1 for ax in kept
    )
    dparam = jax.ShapeDtypeStruct(dparam_shape, jnp.float32)
    dx, *dparams = plgpu.kernel(
      kernel,
      out_type=(
        jax.ShapeDtypeStruct(x_shape, dtype),
        *[dparam] * (has_scale + has_offset),
      ),
      grid=grid,
      kernel_name=f"mosaic_norm_bwd_{dtype.name}_m{block_m}_n{block_n}{'_mean' if subtract_mean else ''}",
      grid_names=('m', 'a', 'n'),
      compiler_params=plgpu.CompilerParams(
        lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        # The dparam reductions run over M and N, which are the warp dims, so
        # unlike the forward kernel's reduction over A they have to go via SMEM:
        # one f32 slot per warp, lane and vector element.
        reduction_scratch_bytes=(
          4 * 32 * _vector_length(block_n, A, vec_bitwidth) * 4
        ),
      )
    )(
      dout.reshape(x_shape),
      x.reshape(x_shape),
      *[a for a in (scale, mean) if a is not None],
      rstddev,
    )

    it = iter(dparams)

    total = lambda a, dt: jnp.sum(a.reshape(-1, A), axis=0).astype(dt)
    dscale = total(next(it), scale.dtype) if has_scale else None
    doffset = total(next(it), offset.dtype) if has_offset else None
    return (dx.reshape(orig_x_shape), dscale, doffset), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    _, _, _, x, scale, offset = ba.args
    axis = ba.kwargs['axis']
    config = _heuristics_config(
      x, scale, offset, axis=axis, vmap_axis_sizes=ba.vmap_axis_sizes
    )
    n = canonicalize_shape_3d(x.shape, axis)[2]
    if config.block_n is not None and n % 32 == 0:
      # The VJP holds more live values per element than the forward, so it takes
      # a narrower N block.
      config = dataclasses.replace(config, block_n=32)
    return config

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)
