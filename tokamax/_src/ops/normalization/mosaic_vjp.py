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
"""Pallas-Mosaic-GPU normalization VJP op.

One kernel for all three gradients. `dx` needs two reductions over the reduced
axis `A`, which the forward's layout already makes cheap; `dscale`/`doffset`
reduce over every *other* axis, which no single block owns, so each block writes
a partial sum and a small XLA reduction finishes the job -- as in
`pallas_triton_vjp`.
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
from tokamax._src.ops.normalization import pallas_triton_vjp_config as vjp_config


Config = vjp_config.Config
Key = vjp_config.Key
Residuals = base.Residuals


@dataclasses.dataclass(frozen=True, slots=True)
class PallasMosaicGpuNormalizationVjp(base.NormalizationVjp[Config, Key]):
  """Pallas-Mosaic-GPU normalization VJP."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  def _fwd(
    self,
    residuals: Residuals,
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
    """Computes normalization VJP `(dx, dscale, doffset)`."""
    del out, epsilon  # Unused: `rstddev` already absorbed `epsilon`.

    if return_residuals:
      raise NotImplementedError('`return_residuals` not supported.')

    mean, rstddev = residuals
    if (mean is not None) != subtract_mean:
      raise ValueError('`mean` residual inconsistent with `subtract_mean`.')

    dtype = x.dtype
    orig_x_shape = x.shape
    # Canonicalize to 3D, where the second axis is the reduced axis.
    num_m, num_a, num_b = triton_config.canonicalize_shape_3d(orig_x_shape, axis)

    has_scale = scale is not None
    has_offset = offset is not None

    # This gets `vmappable_serially`, not the forward's `vmappable`: `dscale` and
    # `doffset` are reduced over rows, so folding a batch into `M` would sum them
    # over the batch too. Its docstring has the full reasoning.
    def launch(dout, x, mean, rstddev, scale):
      num_m, num_a, num_b = x.shape
      # The register budget is deliberately the forward's: the two then support
      # exactly the same shapes, and a `NotImplementedError` raised here would
      # surface at grad time, long after the forward committed to this kernel.
      # The block does hold one array more live than the forward does, so this
      # is the knob to lower if it starts spilling.
      p = mosaic_tiling.plan(
        (num_m, num_a, num_b),
        dtype.itemsize,
        block_m=config.block_m,
        block_b=config.block_n or 1,
      )

      def kernel(*refs):
        it = iter(refs)  # Inputs then outputs, optional ones only if present.
        take = lambda present: next(it) if present else None
        dout_ref, x_ref = next(it), next(it)
        scale_ref, mean_ref = take(has_scale), take(subtract_mean)
        rstddev_ref, dx_ref = next(it), next(it)
        dscale_ref, doffset_ref = take(has_scale), take(has_offset)
        x_idx, stat_idx, dparam_idx = p.indices()

        def load_stat(ref):
          if p.layout is None:  # 1D tile: the statistic is a scalar.
            return ref[stat_idx].astype(jnp.float32)
          return mosaic_tiling.load(ref.at[stat_idx], p.stat_layout)

        x_norm = mosaic_tiling.load(x_ref.at[x_idx], p.layout)
        bcast = lambda a: p.bcast(a, x_norm.shape, p.stat_dims)
        if mean_ref is not None:
          x_norm -= bcast(load_stat(mean_ref))
        rstddev = load_stat(rstddev_ref)
        x_norm *= bcast(rstddev)

        dout = mosaic_tiling.load(dout_ref.at[x_idx], p.layout)
        # These two reduce over every row, so a block can only ever hold a
        # partial sum. Both read `dout` before `scale` is folded in.
        if doffset_ref is not None:
          doffset_ref[dparam_idx] = p.sum_over_rows(dout)
        if dscale_ref is not None:
          dscale_ref[dparam_idx] = p.sum_over_rows(dout * x_norm)
        if scale_ref is not None:
          dout *= p.read_param(scale_ref, dout.shape) + scale_offset

        # `dx = (dout - mean(dout * x_norm) * x_norm - mean(dout)) * rstddev`;
        # see `base.NormalizationVjp`. The last term is zero for RMS norm, whose
        # mean was never subtracted in the first place.
        dx = dout - bcast(p.mean_over_a(dout * x_norm)) * x_norm
        if mean_ref is not None:
          dx -= bcast(p.mean_over_a(dout))
        dx_ref[x_idx] = (dx * bcast(rstddev)).astype(dtype)

      dparam = jax.ShapeDtypeStruct(p.dparam_shape, jnp.float32)
      outs = plgpu.kernel(
        kernel,
        out_type=(
          jax.ShapeDtypeStruct(p.x_shape, dtype),
          *[dparam] * (has_scale + has_offset),
        ),
        grid=p.grid,
        grid_names=p.grid_names,
        compiler_params=mosaic_tiling.LANE_SEMANTICS,
      )(
        dout.reshape(p.x_shape),
        x.reshape(p.x_shape),
        *([scale] if has_scale else []),
        *([mean.reshape(p.stat_shape)] if mean is not None else []),
        rstddev.reshape(p.stat_shape),
      )

      # Finish the cross-block reduction here rather than in the kernel: XLA
      # does it in one pass over an array that is `grid` big, not `x` big.
      return (
        outs[0].reshape(num_m, num_a, num_b),
        *(jnp.sum(o, axis=(0, 1)) for o in outs[1:]),
      )

    for name, stat in (('mean', mean), ('rstddev', rstddev)):
      if stat is not None and stat.dtype != jnp.float32:
        raise ValueError(f'`{name}` must be `float32`, got {stat.dtype}.')

    canonical = lambda a: a.reshape(num_m, num_a, num_b)
    as_stat = lambda a: None if a is None else a.reshape(num_m, num_b)
    args = (
      canonical(dout), canonical(x), as_stat(mean), as_stat(rstddev), scale
    )
    # `custom_vmap` reports one `in_batched` flag per positional argument, so the
    # optional ones are dropped here and put back inside.
    present = [a is not None for a in args]

    def launch_present(*present_args):
      it = iter(present_args)
      return launch(*[next(it) if p else None for p in present])

    outs = list(
      mosaic_tiling.vmappable_serially(launch_present)(
        *[a for a in args if a is not None]
      )
    )

    dx = outs.pop(0).reshape(orig_x_shape)
    dscale = outs.pop(0).astype(scale.dtype) if has_scale else None
    doffset = outs.pop(0).astype(offset.dtype) if has_offset else None
    return (dx, dscale, doffset), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return vjp_config.get_heuristics_config(
      *ba.args, vmap_axis_sizes=ba.vmap_axis_sizes, **ba.kwargs
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    return vjp_config.get_key(*ba.args, **ba.kwargs)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    # No TMA, WGMMA or async copies: plain GMEM loads and stores run on Ampere.
    return gpu_utils.has_mosaic_gpu_support(device)
