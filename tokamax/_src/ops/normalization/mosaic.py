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

def _vector_length(r: int, c: int, bitwidth: int) -> int:
  """Elements in one thread's vector, for an (r, c) tile with `c` contiguous.

  The kernel is memory-bound, so we want the widest load the hardware offers
  (16 bytes = 128 bits per thread), capped by what the contiguous axis holds and
  by what is left once the tile is spread over the warpgroup's 128 threads.
  """
  return math.gcd(128 // bitwidth, c, (r * c) // 128)

def _tiled_layout(r: int, c: int, bitwidth: int, *, reduce_axis: int):
  """Builds the register layout for one (r, c) block, `c` contiguous in GMEM.

  The kernel blocks either M or N, never both, so the unblocked one of the two
  is indexed with a scalar and every tile is 2D: the reduced axis A and
  whichever of M/N is blocked, with the contiguous axis last.

  Warps go along the non-reduced axis first, which keeps a warp's reduction
  in-warp, and whatever that axis cannot take goes along the other. Lanes go
  along the contiguous axis first, which keeps the loads coalesced.

  `plgpu.Tiling` grows the rank 2 -> 4 -> 6 -> 7 as it splits the tile by warp,
  by lane and by vector: -7, -6 are tile counts, -5, -4 the warp dims, -3, -2
  the lane dims and -1 the vector. `canonicalize()` drops whichever are 1.
  """
  if (r * c) % 128 != 0:
    raise NotImplementedError(f'Tile ({r=}, {c=}) is not a multiple of 128.')
  vec = _vector_length(r, c, bitwidth)
  c_v = c // vec
  if reduce_axis == 1: # c is reduced → warps onto r first
    wr = math.gcd(4, r)
    wc = math.gcd(4 // wr, c_v)
  else:  # r is reduced → warps onto c first
    wc = math.gcd(4, c_v)
    wr = math.gcd(4 // wc, r)
  lc = math.gcd(32, c_v // wc)
  lr = math.gcd(32 // lc, r // wr)
  if wr * wc != 4 or lr * lc != 32:
    raise NotImplementedError(f'Cannot tile ({r=}, {c=}) across 128 threads.')

  l = TiledLayout(
      plgpu.Tiling(((wr * lr, wc * lc * vec), (lr, lc * vec), (vec,))),
      warp_dims=(-5, -4),
      lane_dims=(-3, -2),
      vector_dim=-1,
      _check_canonical=False,
  ).canonicalize()
  return plgpu.Layout.TILED(
      l.tiling, warp_dims=l.warp_dims, lane_dims=l.lane_dims, vector_dim=-1
  )

def canonicalize_shape_3d(
    shape: Sequence[int], axis: int
) -> tuple[int, int, int]:
  return (math.prod(shape[:axis]), shape[axis], math.prod(shape[axis:][1:]))

def _min_block_m(A: int, bitwidth: int) -> int:
  """The fewest rows a block can have, with no N to block.

  With only M and A to tile, the block has to hold enough for 128 threads to
  divide it. `_tiled_layout` is the authority on that, so we ask it rather than
  re-deriving the rule, and leave `_launch` to pad shapes with fewer rows.
  """
  for block_m in (1, 2, 4, 8, 16, 32, 64, 128):
    try:
      _tiled_layout(block_m, A, bitwidth, reduce_axis=1)
      return block_m
    except NotImplementedError:
      continue
  raise NotImplementedError(f'No tiling for {A=} at any block_m.')

def _heuristics_config(x, scale, offset, *, axis, vmap_axis_sizes) -> Config:
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
  min_block_m = _min_block_m(a, jnp.dtype(x.dtype).itemsize * 8)
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

def _block_index(x_shape, block, idx, squeeze):
  """Start offsets for one CTA's block, clamped to keep the block in bounds.

  Shapes need not be multiples of the block: a trailing partial block is shifted
  back to end at the array's edge, so it overlaps its predecessor and recomputes
  the shared elements. The loads stay unpredicated and the duplicated stores
  write the same values twice, which is harmless.

  Axis `squeeze` is blocked at 1 and gets a scalar index, which drops it from
  the tile; see `_tiled_layout`.
  """
  return tuple(
    jnp.minimum(i, s - 1) if ax == squeeze else pl.ds(jnp.minimum(i * b, s - b), b)
    for ax, (i, s, b) in enumerate(zip(idx, x_shape, block))
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

    # Either M or N is blocked, never both; the other is indexed with a scalar
    # so the tile is 2D with the contiguous axis last.
    squeeze = 2 if config.block_n is None else 0
    tile = tuple(b for ax, b in enumerate(block) if ax != squeeze)
    red = 1 if squeeze == 2 else 0  # Where A sits in the tile.
    keep = 1 - red

    layout = _tiled_layout(tile[0], tile[1], dtype.itemsize * 8,
                            reduce_axis=red)

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
        x_shape, block, [jax.lax.axis_index(i) for i in 'man'], squeeze
      )

      stat_index = index[:1] + index[2:]
      bcast_stat = lambda a: jax.lax.broadcast_in_dim(a, tile, (keep,))
      x = plgpu.load(x_gmem.at[index], layout=layout, optimized=False).astype(jnp.float32)

      if subtract_mean:
        mean = jnp.mean(x, axis=red)
        x -= bcast_stat(mean)
        if mean_gmem is not None:
          mean_gmem[stat_index] = mean
      rstddev = jax.lax.rsqrt(jnp.mean(x * x, axis=red) + epsilon)
      if rstd_gmem is not None:
        rstd_gmem[stat_index] = rstddev

      x = x * bcast_stat(rstddev)
      bcast_param = lambda a: jax.lax.broadcast_in_dim(a, tile, (red,))
      if scale_ref is not None:
        scale = bcast_param(plgpu.load(scale_ref, optimized=False).astype(jnp.float32))
        x *= scale + scale_offset
      if offset_ref is not None:
        offset = bcast_param(plgpu.load(offset_ref, optimized=False).astype(jnp.float32))
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
    # The forward takes shapes its block does not divide by shifting the
    # trailing block back over rows a neighbour already covered. Harmless there,
    # since the stores repeat values; here the `dscale`/`doffset` reductions
    # would count those rows twice, so this kernel keeps to whole blocks.
    if any(s % b for (s, b) in zip(x_shape, block)):
      raise NotImplementedError('Shapes the block does not divide.')
    grid = _grid(x_shape, block)
    grid_n = grid[2]

    vec_bitwidth = 32
    squeeze = 2 if config.block_n is None else 0
    tile = tuple(b for ax, b in enumerate(block) if ax != squeeze)
    red = 1 if squeeze == 2 else 0  # Where A sits in the tile.
    keep = 1 - red
    # The forward puts the warps on the non-reduced axis; here `dscale`/
    # `doffset` reduce over everything *except* A, so warps on anything but A
    # leave those partials warp-replicated -- all four warps storing the same
    # values to the same addresses, ~14x write amplification. Passing `keep` as
    # the reduced axis is what puts them on A instead.
    layout = _tiled_layout(tile[0], tile[1], vec_bitwidth, reduce_axis=keep)

    def kernel(*refs):
      it = iter(refs)  # Inputs then outputs, optional ones only if present.
      take = lambda present: next(it) if present else None
      dout_gmem, x_gmem, scale_ref = next(it), next(it), take(has_scale)
      mean_gmem, rstd_gmem = take(subtract_mean), next(it)
      dx_gmem = next(it)
      dscale_gmem, doffset_gmem = take(has_scale), take(has_offset)

      m, _, n = [jax.lax.axis_index(i) for i in 'man']
      index = tuple(
        i if ax == squeeze else pl.ds(i * b, b)
        for ax, (i, b) in enumerate(zip((m, 0, n), block))
      )
      load = lambda ref: plgpu.load(
        ref.at[index], layout=layout, optimized=False
      ).astype(jnp.float32)
      bcast = lambda a: jax.lax.broadcast_in_dim(a, tile, (keep,))

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

      # Reductions across singleton dimensions are not supported.
      reduce_mn = (
        (lambda a: jnp.sum(a, axis=keep)) if tile[keep] > 1 else (lambda a: a)
      )

      # Each CTA owns one (m, n) grid cell, hence one `A`-long run of the
      # partials. A singleton kept axis cannot be reduced over, so it stays and
      # the run is indexed inside it.
      run = pl.ds((m * grid_n + n) * A, A)
      dparam_index = run if tile[keep] > 1 else (
        (pl.ds(0, 1), run) if keep == 0 else (run, pl.ds(0, 1))
      )

      if doffset_gmem is not None:
        doffset_gmem[dparam_index] = reduce_mn(dout)
      if dscale_gmem is not None:
        dscale_gmem[dparam_index] = reduce_mn(dout * x_norm)
        loaded = plgpu.load(scale_ref, optimized=False).astype(jnp.float32)
        dout *= jax.lax.broadcast_in_dim(loaded, tile, (red,)) + scale_offset

      dx = dout - bcast(jnp.mean(dout * x_norm, axis=red)) * x_norm
      if mean_gmem is not None:
        dx -= bcast(jnp.mean(dout, axis=red))
      dx_gmem[index] = (dx * bcast(rstddev)).astype(dtype)

    # Shaped to match what `reduce_mn` leaves behind, with the A axis stacked
    # one run per (m, n) grid cell.
    runs = grid[0] * grid_n * A
    dparam_shape = (runs,) if tile[keep] > 1 else (
      (1, runs) if keep == 0 else (runs, 1)
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
          4 * 32 * _vector_length(tile[0], tile[1], vec_bitwidth) * 4
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
