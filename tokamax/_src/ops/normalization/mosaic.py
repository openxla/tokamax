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

def _tiled_layout(
    block_shape: tuple[int, int], bitwidth: int, *, reduce_axis: int
):
  """Builds the register layout for one 2D tile, trailing axis contiguous.

  Args:
    block_shape: The tile's (r, c), with `c` contiguous in GMEM.
    bitwidth: Bits per element.
    reduce_axis: The axis of `block_shape` the kernel reduces over.

  Warps go along the non-reduced axis first, which keeps a warp's reduction
  in-warp, and whatever that axis cannot take goes along the other. Lanes go
  along the contiguous axis first, which keeps the loads coalesced.
  """
  r, c = block_shape
  if (r * c) % 128 != 0:
    raise NotImplementedError(f'Tile ({r=}, {c=}) is not a multiple of 128.')
  vec = math.gcd(128 // bitwidth, c, (r * c) // 128)
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
    # The kernel does not mask and only M is padded, so `block_n` has to divide
    # `n`: the largest power-of-two divisor under the target. It also has to
    # carry the 128 threads alongside A; when no divisor does, this shape gets
    # no N block at all and M takes over below, strided loads and all.
    block_n = math.gcd(n, block_n)
    if (a * block_n) % 128 == 0:
      # Blocking N already gives each block a full cache line, so there is
      # nothing left for `block_m > 1` to re-use.
      return Config(block_m=1, block_n=block_n)

  # No N block, either because the reduced axis is trailing or because nothing
  # divided `n`: the only re-use of `scale`/`offset` across rows comes from M.
  # Halve `block_m` until the block fits in registers and enough blocks are
  # launched to fill the device, but never below what the tiling needs: M is
  # carrying the warps and whatever lanes A could not take.
  # A shape with fewer rows than the floor gets the floor anyway, and the
  # launch pads it up to one block.
  min_block_m = 128 // math.gcd(128, a)
  block_m = max(min_block_m, min(32, prev_power_of_2(m)))
  block_size = block_m * a  # One register per element; the tile is not padded.
  num_blocks = pl.cdiv(m, block_m) * n * math.prod(vmap_axis_sizes)
  max_block_size = gpu_utils.NUM_REGISTERS_PER_SM // 4
  min_num_blocks = 4 * jax.devices()[0].core_count
  while (block_m > min_block_m) and (
      (block_size > max_block_size) or (num_blocks < min_num_blocks)
  ):
    block_m //= 2
    block_size //= 2
    num_blocks *= 2
  return Config(block_m=block_m, block_n=None)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicGpuNormalization(base.Normalization[Config, Any]):
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
    x: jax.Array | base.FusedInputArray,
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
    if callable(x):
      x = x()

    dtype = x.dtype
    orig_x_shape = x.shape
    x_shape = canonicalize_shape_3d(orig_x_shape, axis)

    return_mean = return_residuals and subtract_mean
    A = x_shape[1]
    block = (config.block_m, A, config.block_n or 1)
    block_m, _, block_n = block

    # Either M or N is blocked, never both; the other is indexed with a scalar
    # so the tile is 2D with the contiguous axis last.
    squeeze = 2 if config.block_n is None else 0
    tile = tuple(b for ax, b in enumerate(block) if ax != squeeze)
    red = 1 if squeeze == 2 else 0  # Where A sits in the tile.
    keep = 1 - red

    layout = _tiled_layout(tile, dtype.itemsize * 8, reduce_axis=red)

    # A block that overruns its axis is not expressible without masked loads,
    # so M is padded up to a whole number of blocks. It is the only axis that
    # needs it: A is never blocked, and `block_n` divides `n` by construction.
    # Fewer rows than one block -- what `vmap` hands over all the time -- is the
    # same case, and pads the same way.
    rows = x_shape[0]
    pad_m = -rows % block_m
    x = x.reshape(x_shape)
    if pad_m:
      x = jnp.pad(x, ((0, pad_m), (0, 0), (0, 0)))
      x_shape = (rows + pad_m,) + x_shape[1:]

    # Padding makes `x` a fresh buffer, so there is nothing left worth aliasing.
    alias = bool(self.input_output_alias) and not pad_m
    # A ref is written in place, so the kernel writes `y` over `x` and XLA gets
    # to drop the copy `new_ref` starts with whenever `x` is dead afterwards.
    x_operand = jax.new_ref(x) if alias else x

    def kernel(x_gmem, scale_ref, offset_ref, outs):
      # When aliasing, `x_gmem` is a mutable ref and there is no `y` output:
      # each CTA has already read its tile into registers by the time it writes.
      y_gmem = x_gmem if outs['y'] is None else outs['y']
      mean_gmem, rstd_gmem = outs['mean'], outs['rstd']

      slc = tuple(
        i if ax == squeeze else pl.ds(i * b, b)
        for ax, (i, b) in enumerate(
          zip([jax.lax.axis_index(j) for j in 'man'], block))
      )

      stat_index = slc[:1] + slc[2:]
      bcast_stat = lambda a: jax.lax.broadcast_in_dim(a, tile, (keep,))
      x = plgpu.load(x_gmem.at[slc], layout=layout, optimized=False).astype(jnp.float32)

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
      y_gmem[slc] = x.astype(dtype)

    stat = jax.ShapeDtypeStruct(x_shape[:1] + x_shape[2:], jnp.float32)
    # `block_n` divides `n` and M is padded, so every block divides its axis.
    for (s, b) in zip(x_shape, block):
      assert s % b == 0
    # An absent input or output stays in place as `None` -- `plgpu.kernel`
    # passes the refs positionally, but it keeps the operands' tree structure,
    # so every ref reaches the kernel under the name it has here.
    outs = plgpu.kernel(
      kernel,
      out_type={
        'y': None if alias else jax.ShapeDtypeStruct(x_shape, dtype),
        'mean': stat if return_mean else None,
        'rstd': stat if return_residuals else None,
      },
      grid=tuple(s // b for (s, b) in zip(x_shape, block)),
      grid_names=('m', 'a', 'n'),
      kernel_name=f"mosaic_norm_fwd_{dtype.name}_m{block_m}_n{block_n}{'_mean' if subtract_mean else ''}",
      compiler_params=plgpu.CompilerParams(
        lowering_semantics=plgpu.LoweringSemantics.Warpgroup)
    )(x_operand, scale, offset)

    y = jax.freeze(x_operand) if alias else outs['y']

    unpad = lambda a: a[:rows] if pad_m else a
    y = unpad(y).reshape(orig_x_shape)
    if not return_residuals:
      return y, None

    stat_shape = list(orig_x_shape)
    stat_shape[axis] = 1
    mean = unpad(outs['mean']).reshape(stat_shape) if return_mean else None
    rstddev = unpad(outs['rstd']).reshape(stat_shape)
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
class PallasMosaicGpuNormalizationVjp(base.NormalizationVjp[Config, Any]):
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

    # Same as the forward: a block that overruns M is not expressible without
    # masked loads, so the rows are padded up to a whole number of blocks. A
    # padded row carries a zero `dout`, so it adds nothing to the `dscale` and
    # `doffset` partials, and its `dx` is sliced back off below. `block_n`
    # divides N by construction and A is never blocked, so M is the only one.
    rows = x_shape[0]
    pad_m = -rows % block_m
    dout, x = dout.reshape(x_shape), x.reshape(x_shape)
    if pad_m:
      pad = lambda a: jnp.pad(a, ((0, pad_m),) + ((0, 0),) * (a.ndim - 1))
      x_shape = (rows + pad_m,) + x_shape[1:]
      dout, x, rstddev = pad(dout), pad(x), pad(rstddev)
      mean = None if mean is None else pad(mean)

    grid = tuple(s // b for (s, b) in zip(x_shape, block))
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
    layout = _tiled_layout(tile, vec_bitwidth, reduce_axis=keep)

    def kernel(dout_gmem, x_gmem, scale_ref, mean_gmem, rstd_gmem, outs):
      dx_gmem = outs['dx']
      dscale_gmem, doffset_gmem = outs['dscale'], outs['doffset']

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
      def reduce_mn(a):
        return a if tile[keep] == 1 else jnp.sum(a, axis=keep)

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
    # An absent input or output stays in place as `None`: `plgpu.kernel` keeps
    # the operands' tree structure, so every ref reaches the kernel under the
    # name it has here.
    outs = plgpu.kernel(
      kernel,
      out_type={
        'dx': jax.ShapeDtypeStruct(x_shape, dtype),
        'dscale': dparam if has_scale else None,
        'doffset': dparam if has_offset else None,
      },
      grid=grid,
      kernel_name=f"mosaic_norm_bwd_{dtype.name}_m{block_m}_n{block_n}{'_mean' if subtract_mean else ''}",
      grid_names=('m', 'a', 'n'),
      compiler_params=plgpu.CompilerParams(
        lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        # The dparam reductions run over M and N, which are the warp dims, so
        # unlike the forward kernel's reduction over A they have to go via SMEM:
        # one f32 slot per warp, lane and vector element.
        reduction_scratch_bytes=(
          4 * 32 * layout.to_mgpu().vector_length * 4
        ),
      )
    )(dout, x, scale, mean, rstddev)

    total = lambda a, dt: jnp.sum(a.reshape(-1, A), axis=0).astype(dt)
    dscale = total(outs['dscale'], scale.dtype) if has_scale else None
    doffset = total(outs['doffset'], offset.dtype) if has_offset else None
    dx = outs['dx']
    dx = dx[:rows] if pad_m else dx
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
