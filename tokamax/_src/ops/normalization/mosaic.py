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

Normalization is memory-bound, so the design serves coalesced GMEM access:
blocking, layouts and the grid all live in `mosaic_tiling`, shared with the VJP.

Lane semantics is required: Warpgroup layout inference cannot solve a program
containing a partial reduction, so layouts are explicit.
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
from tokamax._src.ops.normalization import mosaic_vjp
from tokamax._src.ops.normalization import pallas_triton_config as triton_config

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
      object.__setattr__(self, 'vjp', mosaic_vjp.PallasMosaicGpuNormalizationVjp())

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
      # `plgpu.kernel` has no donation/aliasing argument.
      raise NotImplementedError(
        '`input_output_alias` is not supported by the Mosaic GPU kernel.'
      )

    if callable(x):
      x = x()

    dtype = x.dtype
    orig_x_shape = x.shape
    # Canonicalize to 3D, where the second axis is the reduced axis.
    num_m, num_a, num_b = triton_config.canonicalize_shape_3d(orig_x_shape, axis)

    return_mean = return_residuals and subtract_mean
    has_scale = scale is not None
    has_offset = offset is not None

    p = mosaic_tiling.plan(
      (num_m, num_a, num_b),
      dtype.itemsize,
      block_m=config.block_m,
      block_b=config.block_n or 1,
    )

    def kernel(*refs):
      it = iter(refs)  # Inputs then outputs, optional ones only if present.
      take = lambda present: next(it) if present else None
      x_ref, scale_ref, offset_ref = next(it), take(has_scale), take(has_offset)
      y_ref, mean_ref = next(it), take(return_mean)
      rstd_ref = take(return_residuals)
      x_idx, stat_idx, _, _ = p.indices()

      x = plgpu.load(x_ref.at[x_idx], layout=p.layout, optimized=False).astype(jnp.float32)
      bcast = lambda a: p.bcast(a, x.shape)

      if subtract_mean:
        mean = jnp.mean(x, axis=p.reduce_axis)
        x -= bcast(mean)
        if mean_ref is not None:
          mean_ref[stat_idx] = mean
      rstddev = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=p.reduce_axis) + epsilon)
      if rstd_ref is not None:
        rstd_ref[stat_idx] = rstddev
      x *= bcast(rstddev)
      # `y = x_norm * (scale + scale_offset) + offset`; see `base.Normalization`.
      if scale_ref is not None:
        x *= p.read_param(scale_ref, x.shape) + scale_offset
      if offset_ref is not None:
        x += p.read_param(offset_ref, x.shape)
      y_ref[x_idx] = x.astype(dtype)

    stat = jax.ShapeDtypeStruct(p.stat_shape, jnp.float32)
    # Under `vmap` this launch is the only thing needing a batching rule, and
    # `plgpu.kernel` brings its own: it prepends a grid axis and slices every
    # batched ref by it. `Plan.indices` reads the grid by name, so the extra
    # axis passes it by, and everything around here is plain JAX that `vmap`
    # batches itself. (`batching.capture_batched_args` records the batch sizes
    # for the heuristics but still calls `jax.vmap` over `_fwd`; see
    # `batching.py`.)
    outs = plgpu.kernel(
      kernel,
      out_type=(
        jax.ShapeDtypeStruct(p.x_shape, dtype),
        *[stat] * (return_mean + return_residuals),
      ),
      grid=p.grid,
      grid_names=p.grid_names,
      compiler_params=mosaic_tiling.LANE_SEMANTICS,
    )(x.reshape(p.x_shape), *[a for a in (scale, offset) if a is not None])

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
    # No TMA, WGMMA or async copies: plain GMEM loads and stores run on Ampere.
    return gpu_utils.has_mosaic_gpu_support(device)
