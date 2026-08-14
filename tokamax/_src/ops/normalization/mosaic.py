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

Normalization is memory-bound, so the design serves coalesced GMEM access: tiles
arrive through `plgpu.emit_pipeline`, which stages them with `cp.async` on
Ampere and gives the next tile somewhere to land while this one is reduced.
Blocking and the pipeline live in `mosaic_tiling`.

There are no layout annotations. Inference derives the register layout from the
staged SMEM ref's transforms, including for the partial reduction -- see
`mosaic_tiling` for the one thing that has to be true of `jax` for that to hold.
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
      # the Triton one: it takes the same residuals and runs on the same
      # hardware. Leaving this `None` would not fall back -- `op.Op` tries to
      # differentiate `_fwd` itself, which a Pallas kernel does not support, and
      # gradients would raise instead.
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
      # `plgpu.kernel` has no donation/aliasing argument.
      raise NotImplementedError(
        '`input_output_alias` is not supported by the Mosaic GPU kernel.'
      )

    if callable(x):
      x = x()

    dtype = x.dtype
    orig_x_shape = x.shape
    # `(M, A)`, or `(M, A, B)` when the reduced axis is not the minor one.
    x_shape = triton_config.canonicalize_shape(orig_x_shape, axis)

    return_mean = return_residuals and subtract_mean
    has_scale = scale is not None
    has_offset = offset is not None

    p = mosaic_tiling.plan(
      x_shape, dtype.itemsize, block_m=config.block_m, block_b=config.block_n
    )

    axis_a = mosaic_tiling.REDUCE_AXIS

    def kernel(*refs):
      it = iter(refs)  # Inputs then outputs, optional ones only if present.
      take = lambda present: next(it) if present else None
      x_gmem, scale_ref, offset_ref = next(it), take(has_scale), take(has_offset)
      y_gmem, mean_gmem = next(it), take(return_mean)
      rstd_gmem = take(return_residuals)
      out_smem = next(it)

      def param(ref, shape):
        """Reads a 1D param straight from GMEM and spreads it along `A`."""
        a = plgpu.load(ref, optimized=False).astype(jnp.float32)
        return jax.lax.broadcast_in_dim(a, shape, (axis_a,))

      def compute(step, x_smem):
        (step,) = step
        index = p.out_index(step)
        stat_index = mosaic_tiling.drop_reduced(index)
        x = x_smem[...].astype(jnp.float32)
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
        if scale_ref is not None:
          x *= param(scale_ref, x.shape) + scale_offset
        if offset_ref is not None:
          x += param(offset_ref, x.shape)

        # Outputs are not pipelined (`emit_pipeline` is input-only pre-Hopper),
        # so store by hand. Going through swizzled SMEM relayouts the tile so
        # that the store to GMEM coalesces; storing straight from the reduction
        # layout would emit scattered writes.
        out_smem[...] = x.astype(dtype)
        y_gmem[index] = plgpu.layout_cast(
          out_smem[...],
          plgpu.Layout.SMEM_GMEM_COPY(p.block, dtype, swizzle=p.swizzle),
        )

      plgpu.emit_pipeline(
        compute,
        grid=(p.steps_per_cta,),
        in_specs=[
          plgpu.BlockSpec(p.block, p.block_indices, transforms=p.transforms)
        ],
        max_concurrent_steps=p.num_stages,
      )(x_gmem)

    stat = jax.ShapeDtypeStruct(mosaic_tiling.drop_reduced(p.shape), jnp.float32)
    outs = plgpu.kernel(
      kernel,
      out_type=(
        jax.ShapeDtypeStruct(x_shape, dtype),
        *[stat] * (return_mean + return_residuals),
      ),
      scratch_types=[
        plgpu.SMEM(p.block, dtype, transforms=p.transforms),
      ],
      grid=(p.num_ctas,),
      grid_names=('cta',),
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
