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
back. No SMEM, no pipeline -- blocking and the layout live in `mosaic_tiling`.

A tile is one load and one store. Every axis is tiled exactly -- the reduced axis
by the lanes, the rest by the block -- so there is no tail to mask, and a shape
that cannot be tiled exactly is declined for the caller to fall back on, Mosaic
having no masked GMEM access to offer instead.

The one layout annotation is on the loaded tile, and everything downstream
follows from it: it is what makes both the load and the store coalesce, and the
reduction needs a tiled layout that inference will not offer. Which layout that
is depends on the shape; see `mosaic_tiling`'s thread mappings.
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
import numpy as np
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import pallas_triton_config as triton_config
from tokamax._src.ops.normalization import pallas_triton_vjp

# Reuse the Triton config only for its cache key
Config = triton_config.Config
Key = triton_config.Key
FusedInputArray = base.FusedInputArray

def _vector_length(block_n: int, A: int, bitwidth: int) -> int:
  vec = (8 * 16) // bitwidth  # 16-byte vectors.
  while True:
    if block_n % vec == 0 or (vec % block_n == 0 and A % vec == 0):
      return vec
    vec //= 2

def _vec_along_a(vec: int, M: int, A: int, N: int) -> tuple[int, int, int]:
  a = A // vec
  if a >= 32:
    return (M, a // 32, N)
  else:
    f = 32 // a
    print("F ", f)
    return (f, a, N)

def _vec_along_n(vec, M, A, N):
  a = 32 // (N // vec)
  b = 32 // a
  return (M, A // b, N // a)

def _warp_blocks(block_m, block_n):
    for amt_m in [4,2,1]:
      amt_n = 4 // amt_m
      if block_m % amt_m == 0 and block_n % amt_n == 0:
        return (amt_m, amt_n)

@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicGpuNormalization(base.Normalization[Config, Key]):
  """Pallas-Mosaic-GPU normalization op."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  input_output_alias: bool | None = None

  def __post_init__(self):
    if self.vjp is None:
      # The Mosaic VJP has not been ported to the pipelined form yet, so borrow
      # the Triton one.
      object.__setattr__(self, 'vjp', pallas_triton_vjp.PallasTritonNormalizationVjp())

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

    # TODO: see if named arguments work here.
    def kernel(*refs):
      it = iter(refs)  # Inputs then outputs, optional ones only if present.
      take = lambda present: next(it) if present else None
      x_gmem, scale_ref, offset_ref = next(it), take(has_scale), take(has_offset)
      y_gmem, mean_gmem = next(it), take(return_mean)
      rstd_gmem = take(return_residuals)

      index = tuple(pl.ds(s * b, b) for (s, b) in
        zip([jax.lax.axis_index(i) for i in "man"], block))

      print("\nITEMSIZE ", dtype.itemsize, dtype.itemsize * 8)
      vec = _vector_length(block_n, A, dtype.itemsize * 8)
      tile_along_a = vec >= block_n
      print("TILE ALONG A ", tile_along_a)
      print("\nVEC ", vec)
      # Start with a block of size (block_m, A, block_n)
      warp_m, warp_n = _warp_blocks(block_m, block_n)
      print("AMT M ", warp_m)
      print("AMT N ", warp_n)
      tile_spec = [block, (block_m // warp_m, A, block_n // warp_n)]
      print("SO FAR ", plgpu.Tiling(tuple(tile_spec)).tile_shape((block_m, A, block_n)))
      if tile_along_a:
        tile_spec.append(_vec_along_a(vec, *tile_spec[-1]))
        print("GOT LANE BLOCKS ", tile_spec[-1])
        vector_dim = -2
        print("TILING ", plgpu.Tiling(tuple(tile_spec)).tile_shape((block_m, A, block_n)))
        tile_spec.append((vec, block_n),)
        print("--> ", plgpu.Tiling(tuple(tile_spec)).tile_shape((block_m, A, block_n)))
      else:
        tile_spec.append(_vec_along_n(vec, *tile_spec[-1]))
        print("GOT LANE BLOCKS ", tile_spec[-1])
        print("TILING ", plgpu.Tiling(tuple(tile_spec)).tile_shape((block_m, A, block_n)))
        vector_dim=-1
        tile_spec.append((A // 32, vec),)
        print("--> ", plgpu.Tiling(tuple(tile_spec)).tile_shape((block_m, A, block_n)))
      print("ORIG ", (block_m, A, block_n))
      print("SPEC ", tile_spec)

      # TODO: alphafold_alphafold_384res_128chan_axis0_forward
      # and alphafold_alphafold_768res_128chan_axis0_forward are too slow!
      l = TiledLayout(
        plgpu.Tiling(tuple(tile_spec)),
        warp_dims=(-11, -9),
        lane_dims=(-8, -7,),
        vector_dim=vector_dim,
        _check_canonical=False).canonicalize()
      layout = plgpu.Layout.TILED(l.tiling, warp_dims=l.warp_dims,
                           lane_dims=l.lane_dims, vector_dim=l.vector_dim)

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
    return triton_config.get_heuristics_config(
      *ba.args, vmap_axis_sizes=ba.vmap_axis_sizes, **ba.kwargs
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    return triton_config.get_key(*ba.args, **ba.kwargs)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)
