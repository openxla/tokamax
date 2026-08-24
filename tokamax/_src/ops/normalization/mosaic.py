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

The one layout annotation is on the loaded tile, and everything downstream
follows from it: it is what makes both the load and the store coalesce, and the
reduction needs a tiled layout that inference will not offer. Which layout that
is depends on the shape; see `mosaic_tiling`'s thread mappings.
"""

import dataclasses
from typing import ClassVar, override

import jax
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src.ops import op
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import mosaic_tiling
from tokamax._src.ops.normalization import pallas_triton_config as triton_config
from tokamax._src.ops.normalization import pallas_triton_vjp

# Reuse the Triton config only for its cache key
Config = triton_config.Config
Key = triton_config.Key
FusedInputArray = base.FusedInputArray


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
    # `(M, A)`, or `(M, A, N)` when the reduced axis is not the minor one.
    x_shape = triton_config.canonicalize_shape(orig_x_shape, axis)

    return_mean = return_residuals and subtract_mean
    has_scale = scale is not None
    has_offset = offset is not None

    p = mosaic_tiling.plan(
      x_shape, dtype.itemsize, block_m=config.block_m, block_n=config.block_n
    )

    axis_a = mosaic_tiling.REDUCE_AXIS

    def kernel(*refs):
      it = iter(refs)  # Inputs then outputs, optional ones only if present.
      take = lambda present: next(it) if present else None
      x_gmem, scale_ref, offset_ref = next(it), take(has_scale), take(has_offset)
      y_gmem, mean_gmem = next(it), take(return_mean)
      rstd_gmem = take(return_residuals)

      # The 1D params come straight from GMEM. `optimized=False` because they are
      # not in SMEM; inference picks their layout off the tile they multiply.
      load = lambda ref: plgpu.load(ref, optimized=False).astype(jnp.float32)
      scale_a = None if scale_ref is None else load(scale_ref)
      offset_a = None if offset_ref is None else load(offset_ref)

      index = p.out_index()
      stat_index = mosaic_tiling.drop_reduced(index)
      x = plgpu.load(x_gmem.at[index], layout=p.layout, optimized=False)
      x = x.astype(jnp.float32)
      # The stats span every axis but the reduced one.
      stat_dims = tuple(d for d in range(x.ndim) if d != axis_a)
      bcast = lambda a: jax.lax.broadcast_in_dim(a, x.shape, stat_dims)

      if subtract_mean:
        mean = jnp.mean(x, axis=axis_a)
        x -= bcast(mean)
        if mean_gmem is not None:
          mean_gmem[stat_index] = mean
      rstddev = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=axis_a) + epsilon)
      if rstd_gmem is not None:
        rstd_gmem[stat_index] = rstddev
      x *= bcast(rstddev)
      # `y = x_norm * (scale + scale_offset) + offset`; see `base.Normalization`.
      # The params span only the reduced axis, so they spread along the rest.
      bcast_a = lambda a: jax.lax.broadcast_in_dim(a, x.shape, (axis_a,))
      if scale_a is not None:
        x *= bcast_a(scale_a) + scale_offset
      if offset_a is not None:
        x += bcast_a(offset_a)

      y_gmem[index] = x.astype(dtype)

    stat = jax.ShapeDtypeStruct(mosaic_tiling.drop_reduced(p.shape), jnp.float32)
    outs = plgpu.kernel(
      kernel,
      out_type=(
        jax.ShapeDtypeStruct(x_shape, dtype),
        *[stat] * (return_mean + return_residuals),
      ),
      grid=p.grid,
      grid_names=p.grid_names,
      compiler_params=mosaic_tiling.WARPGROUP_SEMANTICS,
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
    return mosaic_tiling.with_usable_block_m(
      triton_config.get_heuristics_config(
        *ba.args, vmap_axis_sizes=ba.vmap_axis_sizes, **ba.kwargs
      )
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    return triton_config.get_key(*ba.args, **ba.kwargs)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    # `cp.async` staging and plain GMEM stores: no TMA or WGMMA needed.
    return gpu_utils.has_mosaic_gpu_support(device)
