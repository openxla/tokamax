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

    # `x` may hold `axis_size` `vmap` elements folded into its leading axis, in
    # which case `scale` has one row each and the param gradients come back one
    # per element. See `_rule` below; `axis_size == 1` is the unbatched call.
    def launch(dout, x, mean, rstddev, scale, axis_size):
      num_rows, num_a, num_b = x.shape
      rows_per_element = num_rows // axis_size
      # The register budget is deliberately the forward's: the two then support
      # exactly the same shapes, and a `NotImplementedError` raised here would
      # surface at grad time, long after the forward committed to this kernel.
      # The block does hold one array more live than the forward does, so this
      # is the knob to lower if it starts spilling. `rows_per_element` keeps a
      # block from straddling a batch boundary, which would mix two elements'
      # `dscale` into one partial.
      p = mosaic_tiling.plan(
        (num_rows, num_a, num_b),
        dtype.itemsize,
        block_m=config.block_m,
        block_b=config.block_n or 1,
        rows_per_element=rows_per_element,
      )
      blocks_per_element = rows_per_element // p.block_m

      def kernel(*refs):
        it = iter(refs)  # Inputs then outputs, optional ones only if present.
        take = lambda present: next(it) if present else None
        dout_ref, x_ref = next(it), next(it)
        scale_ref, mean_ref = take(has_scale), take(subtract_mean)
        rstddev_ref, dx_ref = next(it), next(it)
        dscale_ref, doffset_ref = take(has_scale), take(has_offset)
        idx = p.indices()
        x_idx, stat_idx, dparam_idx = idx.x, idx.stat, idx.dparam
        # `scale` has a row per batch element, and this block lies inside one.
        element = 0 if axis_size == 1 else idx.pid_m // blocks_per_element

        def load_stat(ref):
          if p.layout is None:  # 1D tile: the statistic is a scalar.
            return ref[stat_idx].astype(jnp.float32)
          return mosaic_tiling.load(ref.at[stat_idx], p.stat_layout)

        x_norm = mosaic_tiling.load(x_ref.at[x_idx], p.layout)
        bcast = lambda a: p.bcast(a, x_norm.shape)
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
          dout *= p.read_param(scale_ref.at[element], dout.shape) + scale_offset

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

      # Finish the cross-block reduction here rather than in the kernel: XLA does
      # it in one pass over an array that is `grid` big, not `x` big. Splitting
      # the block axis first keeps each batch element's partials to itself, which
      # is the whole reason blocks were kept from straddling a boundary.
      def sum_partials(o):
        o = o.reshape(axis_size, blocks_per_element, *o.shape[1:])
        return jnp.sum(o, axis=(1, 2))

      return (
        outs[0].reshape(num_rows, num_a, num_b),
        *(sum_partials(o) for o in outs[1:]),
      )

    for name, stat in (('mean', mean), ('rstddev', rstddev)):
      if stat is not None and stat.dtype != jnp.float32:
        raise ValueError(f'`{name}` must be `float32`, got {stat.dtype}.')

    def launch_one(dout, x, mean, rstddev, scale):
      """One batch element: 1D `scale` in, 1D param gradients out."""
      dx, *dparams = launch(
        dout, x, mean, rstddev, None if scale is None else scale[jnp.newaxis], 1
      )
      return (dx, *(p[0] for p in dparams))

    canonical = lambda a: a.reshape(num_m, num_a, num_b)
    as_stat = lambda a: None if a is None else a.reshape(num_m, num_b)
    # `scale` is last because it is the one argument not indexed by row.
    args = (
      canonical(dout), canonical(x), as_stat(mean), as_stat(rstddev), scale
    )
    # `custom_vmap` reports one `in_batched` flag per positional argument, so the
    # optional ones are dropped here and put back inside.
    present = [a is not None for a in args]

    def launch_present(*present_args):
      it = iter(present_args)
      return launch_one(*[next(it) if p else None for p in present])

    f = jax.custom_batching.custom_vmap(launch_present)

    @f.def_vmap
    def _rule(axis_size, in_batched, *args):
      """Folds the batch into `M`, as the forward's own `_rule` does.

      Two things stop this from being that function. `dscale`/`doffset` are
      reduced over rows, so the fold would sum them over the batch as well --
      right for `vjp(vmap(f))`, where the params are shared, wrong for
      `vmap(grad(f))`, which wants one per element. And `scale` arrives batched
      whatever the caller's `in_axes`, because `op.Op` keeps it in the
      `custom_vjp` residuals, which JAX batches wholesale. So the launch keeps
      each element's rows in its own blocks and hands back per-element gradients;
      for a shared param, transposing its broadcast sums them again for free.

      Leaving the batching to JAX is not an option: the generic rule prepends a
      grid axis, renumbering the `pl.program_id`s that `Plan.indices` reads, and
      only the first row-block of each element ends up written.
      """
      if not any(in_batched):
        out = f(*args)
        return out, jax.tree.map(lambda _: False, out)

      args = mosaic_tiling.broadcast_unbatched(args, in_batched, axis_size)
      rows, scale = (args[:-1], args[-1]) if has_scale else (args, None)
      fold = lambda a: a.reshape(axis_size * a.shape[1], *a.shape[2:])
      it = iter([fold(a) for a in rows])
      dout, x = next(it), next(it)
      mean = next(it) if subtract_mean else None
      rstddev = next(it)

      try:
        dx, *dparams = launch(dout, x, mean, rstddev, scale, axis_size)
      except NotImplementedError:
        # The folded rows do not tile -- a 1D input has one row per element, so
        # there is nothing to give the warps. Fall back to a launch per element.
        out = jax.lax.map(lambda a: launch_present(*a), tuple(args))
        return out, jax.tree.map(lambda _: True, out)

      dx = dx.reshape(axis_size, dx.shape[0] // axis_size, *dx.shape[1:])
      return (dx, *dparams), (True,) * (1 + len(dparams))

    outs = list(f(*[a for a in args if a is not None]))

    dx = outs.pop(0).reshape(orig_x_shape)
    dscale = outs.pop(0).astype(scale.dtype) if has_scale else None
    doffset = outs.pop(0).astype(offset.dtype) if has_offset else None
    return (dx, dscale, doffset), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    # Must match the forward's floor, or the forward would accept shapes whose
    # gradient then raises -- long after it committed to this kernel.
    return mosaic_tiling.with_usable_block_m(
      vjp_config.get_heuristics_config(
        *ba.args, vmap_axis_sizes=ba.vmap_axis_sizes, **ba.kwargs
      )
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    return vjp_config.get_key(*ba.args, **ba.kwargs)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    # No TMA, WGMMA or async copies: plain GMEM loads and stores run on Ampere.
    return gpu_utils.has_mosaic_gpu_support(device)
