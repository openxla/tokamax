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

def _heuristics_config(x, scale, offset, *, axis, vmap_axis_sizes) -> Config:
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

    # Either M or N is blocked, never both; the other is indexed with a scalar
    # so the tile is 2D with the contiguous axis last.
    squeeze = 2 if config.block_n is None else 0
    tile = tuple(b for ax, b in enumerate(block) if ax != squeeze)
    red = 1 if squeeze == 2 else 0  # Where A sits in the tile.
    keep = 1 - red

    layout = _tiled_layout(tile[0], tile[1], dtype.itemsize * 8,
                            reduce_axis=red)

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

      index = tuple(
        i if ax == squeeze else pl.ds(i * b, b)
        for ax, (i, b) in enumerate(
          zip([jax.lax.axis_index(j) for j in "man"], block))
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
