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
from typing import ClassVar, override

import jax
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.mosaic.gpu import TiledLayout
from jax.experimental import pallas as pl
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src.ops import op
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import pallas_triton_config as triton_config
from tokamax._src.ops.normalization import pallas_triton_vjp_config as triton_vjp_config

# Reuse the Triton configs only for their cache keys
Config = triton_config.Config
Key = triton_config.Key
VjpConfig = triton_vjp_config.Config
VjpKey = triton_vjp_config.Key
FusedInputArray = base.FusedInputArray

def _vector_length(block_n: int, A: int, bitwidth: int) -> int:
  """Picks the widest per-thread vector that stays contiguous in GMEM.

  The kernel is memory-bound, so we want the widest load the hardware offers
  (16 bytes = 128 bits per thread). A vector is only legal if its elements are
  adjacent in memory: `x` is (M, A, N) with N innermost, so a vector either sits
  inside one N row (`block_n % vec == 0`) or covers whole N rows and spills into
  the A axis (`vec % block_n == 0`, with A divisible so the spill never crosses
  a block boundary). Neither holds -> halve and retry, down to a scalar load.
  """
  vec = (8 * 16) // bitwidth  # 16-byte vectors.
  while True:
    if block_n % vec == 0 or (vec % block_n == 0 and A % vec == 0):
      return vec
    vec //= 2

def _vec_along_a(vec: int, M: int, A: int, N: int) -> tuple[int, int, int]:
  """Tiles a warp's block when the vector spans whole N rows (vec >= block_n).
  The 32 lanes go along the reduced axis, one vector each, and any lanes left
  over once it is exhausted spread along M.
  """
  a = A // vec  # Vectors along the reduced axis.
  if a >= 32:
    return (M, A // 32, N)
  return (M // (32 // a), vec, N)

def _vec_along_n(vec: int, M: int, A: int, N: int) -> tuple[int, int, int]:
  """Tiles a warp's block when the vector fits inside an N row (vec < block_n).

  The 32 lanes are split `ln` along N and `la = 32 // ln` along A. N first, since
  lanes adjacent along N issue one coalesced transaction, whereas lanes along A
  add reduction shuffles; we take the largest `ln` that divides evenly and only
  fall back to A for the leftovers. Divisibility is required both ways because
  the tiling has no support for partial tiles.
  """
  for ln in (32, 16, 8, 4, 2, 1):
    la = 32 // ln
    if N % (ln * vec) == 0 and A % la == 0:
      return (M, A // la, N // ln)
  raise ValueError(f'Cannot spread 32 lanes over {A=}, {N=} with {vec=}.')

def _warp_blocks(block_m, block_n):
    """Splits the block over the warpgroup's 4 warps, along M and/or N.
    """
    for amt_m in [4,2,1]:
      amt_n = 4 // amt_m
      if block_m % amt_m == 0 and block_n % amt_n == 0:
        return (amt_m, amt_n)

def _tiled_layout(block_m: int, a: int, block_n: int, bitwidth: int):
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
    -11, -10,  -9  warp counts along M, A, N. WARP dims are -11 and -9 --
                   warps split M and/or N but never the reduced axis A, so
                   -10 is always 1.

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
  vec = _vector_length(block_n, a, bitwidth)
  warp_m, warp_n = _warp_blocks(block_m, block_n)
  tile_spec = [(block_m, a, block_n), (block_m // warp_m, a, block_n // warp_n)]
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
    warp_dims=(-11, -9),
    lane_dims=lane_dims,
    vector_dim=vector_dim,
    _check_canonical=False).canonicalize()
  return plgpu.Layout.TILED(l.tiling, warp_dims=l.warp_dims,
                            lane_dims=l.lane_dims, vector_dim=l.vector_dim)

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
    if self.input_output_alias:
      raise NotImplementedError(
        '`input_output_alias` is not supported by the Mosaic GPU kernel.'
      )

    if callable(x):
      x = x()

    dtype = x.dtype
    orig_x_shape = x.shape
    x_shape = triton_config.canonicalize_shape_3d(orig_x_shape, axis)

    return_mean = return_residuals and subtract_mean
    has_scale = scale is not None
    has_offset = offset is not None

    A = x_shape[1]
    block = (config.block_m, A, config.block_n or 1)
    block_m, _, block_n = block

    layout = _tiled_layout(block_m, A, block_n, dtype.itemsize * 8)

    def kernel(*refs):
      it = iter(refs)  # Inputs then outputs, optional ones only if present.
      take = lambda present: next(it) if present else None
      x_gmem, scale_ref, offset_ref = next(it), take(has_scale), take(has_offset)
      y_gmem, mean_gmem = next(it), take(return_mean)
      rstd_gmem = take(return_residuals)

      index = tuple(pl.ds(s * b, b) for (s, b) in
        zip([jax.lax.axis_index(i) for i in "man"], block))

      stat_index = index[:1] + index[2:]
      x = plgpu.load(x_gmem.at[index], layout=layout, optimized=False).astype(jnp.float32)
      bcast = lambda a: jax.lax.broadcast_in_dim(a, block, (0, 2))

      if subtract_mean:
        mean = jnp.mean(x, axis=1)
        x -= bcast(mean)
        if mean_gmem is not None:
          mean_gmem[stat_index] = mean
      rstddev = jax.lax.rsqrt(jnp.mean(x * x, axis=1) + epsilon)
      if rstd_gmem is not None:
        rstd_gmem[stat_index] = rstddev

      def param(ref):
        loaded = plgpu.load(ref, optimized=False).astype(jnp.float32)
        # The params span only the reduced axis, so they spread along the rest.
        return jax.lax.broadcast_in_dim(loaded, block, (1,))

      x = x * bcast(rstddev)
      if scale_ref is not None:
        x *= param(scale_ref) + scale_offset
      if offset_ref is not None:
        x += param(offset_ref)
      y_gmem[index] = x.astype(dtype)

    stat = jax.ShapeDtypeStruct(x_shape[:1] + x_shape[2:], jnp.float32)
    for (s,b) in zip(x_shape, block):
      assert s % b == 0
    outs = plgpu.kernel(
      kernel,
      out_type=(
        jax.ShapeDtypeStruct(x_shape, dtype),
        *[stat] * (return_mean + return_residuals),
      ),
      grid=tuple(s//b for (s,b) in zip(x_shape, block)),
      grid_names=('m', 'a', 'n'),
      compiler_params=plgpu.CompilerParams(
        lowering_semantics=plgpu.LoweringSemantics.Warpgroup)
    )(
      x.reshape(x_shape),
      *[a for a in (scale, offset) if a is not None],
    )

    y = outs[0].reshape(orig_x_shape)
    if not return_residuals:
      return y, None

    stat_shape = list(orig_x_shape)
    stat_shape[axis] = 1
    mean = outs[1].reshape(stat_shape) if return_mean else None
    rstddev = outs[-1].reshape(stat_shape)
    return y, (mean, rstddev)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    config = triton_config.get_heuristics_config(
      *ba.args, vmap_axis_sizes=ba.vmap_axis_sizes, **ba.kwargs
    )
    n = triton_config.canonicalize_shape_3d(
      ba.args[0].shape, ba.kwargs['axis']
    )[2]
    if config.block_n is not None and n % 128 == 0:
      # 128 divided by 4 warps allows for 32 lanes per warp.
      config = dataclasses.replace(config, block_n=128)
    return config

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    return triton_config.get_key(*ba.args, **ba.kwargs)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicGpuNormalizationVjp(base.NormalizationVjp[VjpConfig, VjpKey]):
  """Pallas-Mosaic-GPU normalization VJP.

  Same shape as the forward kernel: one CTA takes one (block_m, A, block_n)
  tile of `x` and `dout` straight from GMEM into registers, reduces along A,
  and writes `dx` back. `dscale`/`doffset` reduce over M and N as well, which
  spans CTAs, so each CTA writes a partial and the final sum is left to XLA.
  """

  config_cls: ClassVar[type[VjpConfig]] = VjpConfig

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
    config: VjpConfig,
  ) -> tuple[tuple[jax.Array, jax.Array | None, jax.Array | None], None]:

    if return_residuals:
      raise NotImplementedError('`return_residuals` not supported.')

    mean, _ = residuals
    if (mean is not None) != subtract_mean:
      raise ValueError('`mean` residual inconsistent with `subtract_mean`.')

    dtype = x.dtype
    orig_x_shape = x.shape
    x_shape = triton_config.canonicalize_shape_3d(orig_x_shape, axis)

    has_scale = scale is not None
    has_offset = offset is not None

    A = x_shape[1]
    block = (config.block_m, A, config.block_n or 1)
    block_m, _, block_n = block
    for (s, b) in zip(x_shape, block):
      assert s % b == 0
    grid = tuple(s // b for (s, b) in zip(x_shape, block))
    grid_n = grid[2]

    # If we need to reduce over M, use float32
    vec_bitwidth = dtype.itemsize * 8 if block_m == 1 else 32
    layout = _tiled_layout(block_m, A, block_n, vec_bitwidth)

    reduced = tuple(ax for ax in (2, 0) if block[ax] > 1)
    kept = tuple(ax for ax in range(3) if ax not in reduced)

    def kernel(*refs):
      it = iter(refs)  # Inputs then outputs, optional ones only if present.
      take = lambda present: next(it) if present else None
      dout_gmem, x_gmem, scale_ref = next(it), next(it), take(has_scale)
      dx_gmem = next(it)
      dscale_gmem, doffset_gmem = take(has_scale), take(has_offset)

      m, _, n = [jax.lax.axis_index(i) for i in 'man']
      index = tuple(pl.ds(i * b, b) for (i, b) in zip((m, 0, n), block))
      load = lambda ref: plgpu.load(
        ref.at[index], layout=layout, optimized=False
      ).astype(jnp.float32)
      bcast = lambda a: jax.lax.broadcast_in_dim(a, block, (0, 2))

      # The residuals are ignored and recomputed: `x` is in registers already,
      # so the two extra reductions are cheaper than the GMEM traffic of
      # loading `mean` and `rstddev` back.
      x = load(x_gmem)
      if subtract_mean:
        x -= bcast(jnp.mean(x, axis=1))
      rstddev = jax.lax.rsqrt(jnp.mean(x * x, axis=1) + epsilon)
      x_norm = x * bcast(rstddev)

      dout = load(dout_gmem)

      def reduce_mn(a):
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
        # The params span only the reduced axis, so they spread along the rest.
        loaded = plgpu.load(scale_ref, optimized=False).astype(jnp.float32)
        dout *= jax.lax.broadcast_in_dim(loaded, block, (1,)) + scale_offset

      dx = dout - bcast(jnp.mean(dout * x_norm, axis=1)) * x_norm
      if subtract_mean:
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
      *[a for a in (scale,) if a is not None],
    )

    it = iter(dparams)
    total = lambda a, dt: jnp.sum(a.reshape(-1, A), axis=0).astype(dt)
    dscale = total(next(it), scale.dtype) if has_scale else None
    doffset = total(next(it), offset.dtype) if has_offset else None
    return (dx.reshape(orig_x_shape), dscale, doffset), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> VjpConfig:
    config = triton_vjp_config.get_heuristics_config(
      *ba.args, vmap_axis_sizes=ba.vmap_axis_sizes, **ba.kwargs
    )
    n = triton_config.canonicalize_shape_3d(
      ba.args[3].shape, ba.kwargs['axis']
    )[2]
    if config.block_n is not None and n % 128 == 0:
      # 128 divided by 4 warps allows for 32 lanes per warp.
      config = dataclasses.replace(config, block_n=128)
    return config

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> VjpKey:
    return triton_vjp_config.get_key(*ba.args, **ba.kwargs)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)
