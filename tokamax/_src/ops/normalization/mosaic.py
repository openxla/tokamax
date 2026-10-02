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


# There is no autotuning cache for this op, so the key is only what `Op` builds
# out of the arguments by default.
type Key = immutabledict.immutabledict[str, Any]
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
    raise ValueError(f'Cannot find warp distribution for block_m {block_m}, block_n {block_n}.')

def _tiled_layout(block_m: int, a: int, block_n: int, bitwidth: int):
  """Builds the register layout for one (block_m, a, block_n) block.

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
  tile_spec = [
    (block_m, a, block_n),
    (block_m // warp_m, a, block_n // warp_n),
  ]
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

def canonicalize_shape_3d(
    shape: Sequence[int], axis: int
) -> tuple[int, int, int]:
  return (math.prod(shape[:axis]), shape[axis], math.prod(shape[axis:][1:]))

def _heuristics_config(x, scale, offset, *, axis, vmap_axis_sizes) -> Config:
  """Picks the block shape.

  The kernel is always a single warpgroup, so there is no warp count to pick.
  """
  m, a, n = canonicalize_shape_3d(x.shape, axis)

  if len(x.shape[axis:]) > 1:
    if n % 128 == 0:
      block_n = 128  # 128 divided by 4 warps allows for 32 lanes per warp.
    else:
      # Read a full cache line at a time.
      els_per_cache_line = (
          gpu_utils.CACHE_LINE_SIZE_BYTES // jnp.dtype(x.dtype).itemsize
      )
      block_n = min(els_per_cache_line, pl.next_power_of_2(n))
    # Blocking N already gives each block a full cache line, so there is nothing
    # left for `block_m > 1` to re-use.
    return Config(block_m=1, block_n=block_n)

  # Reducing the trailing axis: there is no N to block, so the only re-use of
  # `scale`/`offset` across rows comes from M. Halve `block_m` until the block
  # fits in registers and enough blocks are launched to fill the device.
  block_m = 1 if (scale is None and offset is None) else 32
  block_size = block_m * pl.next_power_of_2(a)
  num_blocks = pl.cdiv(m, block_m) * math.prod(vmap_axis_sizes)
  max_block_size = gpu_utils.NUM_REGISTERS_PER_SM // 4
  min_num_blocks = 4 * jax.devices()[0].core_count
  while (block_m > 1) and (
      (block_size > max_block_size) or (num_blocks < min_num_blocks)
  ):
    block_m //= 2
    block_size //= 2
    num_blocks *= 2
  return Config(block_m=block_m, block_n=None)

@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicGpuNormalization(base.Normalization[Config, Key]):
  """Pallas-Mosaic-GPU normalization op."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  input_output_alias: bool | None = None

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
    if callable(x):
      x = x()

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

    alias = bool(self.input_output_alias)
    # A ref is written in place, so the kernel writes `y` over `x` and XLA gets
    # to drop the copy `new_ref` starts with whenever `x` is dead afterwards.
    x_operand = jax.new_ref(x.reshape(x_shape)) if alias else x.reshape(x_shape)

    def kernel(*refs):
      it = iter(refs)  # Inputs then outputs, optional ones only if present.
      take = lambda present: next(it) if present else None
      x_gmem, scale_ref, offset_ref = next(it), take(has_scale), take(has_offset)
      # When aliasing, `x_gmem` is a mutable ref and there is no `y` output:
      # each CTA has already read its tile into registers by the time it writes.
      y_gmem = x_gmem if alias else next(it)
      mean_gmem = take(return_mean)
      rstd_gmem = take(return_residuals)

      index = tuple(pl.ds(s * b, b) for (s, b) in
        zip([jax.lax.axis_index(i) for i in "man"], block))

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
    for (s,b) in zip(x_shape, block):
      assert s % b == 0
    outs = plgpu.kernel(
      kernel,
      out_type=(
        *([] if alias else [jax.ShapeDtypeStruct(x_shape, dtype)]),
        *[stat] * (return_mean + return_residuals),
      ),
      grid=tuple(s//b for (s,b) in zip(x_shape, block)),
      grid_names=('m', 'a', 'n')
    )(
      x_operand,
      *[a for a in (scale, offset) if a is not None],
    )

    if alias:
      y, stats = jax.freeze(x_operand), outs
    else:
      y, *stats = outs

    y = y.reshape(orig_x_shape)
    if not return_residuals:
      return y, None

    stat_shape = list(orig_x_shape)
    stat_shape[axis] = 1
    mean = stats[0].reshape(stat_shape) if return_mean else None
    rstddev = stats[-1].reshape(stat_shape)
    return y, (mean, rstddev)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return _heuristics_config(
      *ba.args, axis=ba.kwargs['axis'], vmap_axis_sizes=ba.vmap_axis_sizes
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)
