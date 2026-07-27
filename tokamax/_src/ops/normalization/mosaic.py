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
"""Pallas-Mosaic-GPU normalization op implementation."""

import dataclasses
import functools
import math
from typing import ClassVar, TypeAlias

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src.ops import op
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import triton_config
from typing_extensions import override

# Reuse the Triton config only for its cache key / heuristics plumbing; this
# backend processes one row per program, so `block_m`/`block_n`/`num_warps` are
# all ignored by the kernel.
Config: TypeAlias = triton_config.Config
Key: TypeAlias = triton_config.Key
FusedInputArray = base.FusedInputArray

_WARPGROUP_SIZE = 128

def _vmappable(launch):
  """Adds a `jax.vmap` rule that folds the batch into the leading (batch) axis.

  Mosaic GPU's `pallas_call` cannot be vmapped (its lowering asserts no
  `vmapped_dims`). Normalization batching is just extra independent rows, so we
  merge each `vmap` axis into every argument's existing leading batch dimension
  and issue a single batch-aware launch (`grid = (batch, rows)`). This mirrors
  the `jax_triton` backend's `_vmappable`.
  """
  f = jax.custom_batching.custom_vmap(launch)

  @f.def_vmap
  def _rule(axis_size, in_batched, *arrays):
    b = axis_size

    def merge(a, batched):
      # Give `a` a leading `b` axis (broadcast if unbatched here), then fold it
      # into `a`'s existing leading batch dim.
      if not batched:
        a = jnp.broadcast_to(a[None], (b, *a.shape))
      return a.reshape(b * a.shape[1], *a.shape[2:])

    out = f(*(merge(a, bat) for a, bat in zip(arrays, in_batched)))
    # Split the folded batch back out for this `vmap` level.
    out = jax.tree.map(lambda o: o.reshape(b, o.shape[0] // b, *o.shape[1:]), out)
    return out, jax.tree.map(lambda _: True, out)

  return f


def _padded_len(n: int, vec_size: int = 1) -> int:
  """Round `n` up to a multiple of `vec_size * 128` (`WG_STRIDED` requires the
  row length to be a multiple of `vec_size * 128`)."""
  m = _WARPGROUP_SIZE * vec_size
  return m * ((n + m - 1) // m)


def _normalization_kernel(
    x_ref,
    scale_ref,
    offset_ref,
    y_ref,
    mean_ref,
    rstddev_ref,
    *,
    axis_len,
    epsilon,
    scale_offset,
    subtract_mean,
    block_m,
    vec_size,
):
  """Normalization kernel: `block_m` padded rows per program.

  All refs are whole-array GMEM refs (no SMEM staging, so no Hopper-only TMA);
  each program loads its rows straight to registers with a `WG_STRIDED` layout
  (`vec_size` contiguous elements per lane). Rows are zero-padded to a multiple
  of `vec_size * 128`, which both satisfies the layout constraint and lets us
  reduce with the identity `var = E[x^2] - E[x]^2` — the padding zeros drop out
  of both sums, so no in-kernel masking is needed. The padded tail of the output
  is dropped by the caller. Absent optional operands
  (`scale`/`offset`/`mean`/`rstddev`) are passed as `None` refs.

  The 2D grid is `(batch, row_blocks)`; each program handles `block_m` rows,
  amortizing launch overhead over more work than the one-row-per-program
  variant. `vmap` folds extra batch elements into the batch axis (see
  `_vmappable`). Inputs carry a leading batch dim (size 1 when unbatched);
  `scale`/`offset` are indexed per batch element.
  """
  bi, mi = pl.program_id(0), pl.program_id(1)
  padded_len = x_ref.shape[-1]
  layout = plgpu.Layout.WG_STRIDED((padded_len,), vec_size=vec_size)

  # Load scale/offset once; they are shared across the block's rows.
  s = o = None
  if scale_ref is not None:
    s = plgpu.load(scale_ref.at[bi], layout=layout).astype(jnp.float32)
  if offset_ref is not None:
    o = plgpu.load(offset_ref.at[bi], layout=layout).astype(jnp.float32)

  for j in range(block_m):
    ri = mi * block_m + j
    x = plgpu.load(x_ref.at[bi, ri], layout=layout).astype(jnp.float32)  # pad == 0

    # Padding zeros contribute nothing to either sum.
    mean = jnp.sum(x) / axis_len
    mean_sq = jnp.sum(x * x) / axis_len
    if subtract_mean:
      var = mean_sq - mean * mean
      if mean_ref is not None:
        mean_ref[bi, ri] = mean.astype(mean_ref.dtype)
    else:
      mean = 0.0
      var = mean_sq
    rstddev = jax.lax.rsqrt(var + epsilon)
    if rstddev_ref is not None:
      rstddev_ref[bi, ri] = rstddev.astype(rstddev_ref.dtype)

    y = (x - mean) * rstddev
    if s is not None:
      y = y * (s + scale_offset)
    if o is not None:
      y = y + o

    y_ref.at[bi, ri][...] = y.astype(y_ref.dtype)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicGpuNormalization(base.Normalization[Config, Key]):
  """Pallas-Mosaic-GPU normalization op."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  def __post_init__(self):
    if self.vjp is None:
      # TODO: a fused Mosaic backward kernel
      object.__setattr__(self, 'vjp', base.NormalizationVjp())

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
    # Materialize fused inputs instead of fusing into the kernel. Add
    # Pallas `fuser` support if input fusion turns out to matter.
    if callable(x):
      x = x()

    orig_shape = x.shape
    axis = axis % x.ndim

    # Canonicalize to 2D `(M, A)`, reducing the trailing axis. The transpose is
    # a no-op for the common `axis == -1` case.
    # TODO: reduce the middle axis in-kernel: fuse the transpose.
    x_t = jnp.moveaxis(x, axis, -1)
    perm_shape = x_t.shape
    axis_len = orig_shape[axis]
    rows = math.prod(perm_shape) // axis_len
    x_t = x_t.reshape(rows, axis_len)

    block_m = config.block_m
    vec_size = 1 if config.block_n is None else config.block_n

    # Zero-pad the reduction axis to a multiple of `vec_size * 128` so the
    # `WG_STRIDED` register layout is valid (see `_normalization_kernel`).
    padded_len = _padded_len(axis_len, vec_size)
    pad = padded_len - axis_len
    if pad:
      x_t = jnp.pad(x_t, ((0, 0), (0, pad)))
      if scale is not None:
        scale = jnp.pad(scale, (0, pad))
      if offset is not None:
        offset = jnp.pad(offset, (0, pad))

    # Zero-pad the row axis to a multiple of `block_m` so every program's
    # `block_m` rows stay in bounds (the padded rows' output is dropped below).
    padded_rows = ((rows + block_m - 1) // block_m) * block_m
    if padded_rows != rows:
      x_t = jnp.pad(x_t, ((0, padded_rows - rows), (0, 0)))

    return_mean = return_residuals and subtract_mean
    return_rstd = return_residuals

    kernel = functools.partial(
        _normalization_kernel,
        axis_len=axis_len,
        epsilon=epsilon,
        scale_offset=scale_offset,
        subtract_mean=subtract_mean,
        block_m=block_m,
        vec_size=vec_size,
    )

    # `pl.pallas_call` passes refs only for present operands and won't accept
    # `None` specs, so slot `None` back in for absent optional operands before
    # invoking the kernel. This is the Mosaic analogue of `filter_specs=True` in
    # Triton's `block.pallas_call` (which can't be reused here: it wraps refs in
    # `BlockRef` and patches Triton's `load`/`store`).
    def dispatch(*refs):
      refs = iter(refs)
      x_ref = next(refs)
      scale_ref = next(refs) if scale is not None else None
      offset_ref = next(refs) if offset is not None else None
      y_ref = next(refs)
      mean_ref = next(refs) if return_mean else None
      rstddev_ref = next(refs) if return_rstd else None
      kernel(x_ref, scale_ref, offset_ref, y_ref, mean_ref, rstddev_ref)

    name = 'mosaic_layer_norm' if subtract_mean else 'mosaic_rms_norm'

    # `block_m` rows per program: `grid=(b, padded_rows // block_m)`. Row padding
    # keeps every access in bounds so we need no masking.
    def launch(*arrays):
      b = arrays[0].shape[0]
      gmem = plgpu.BlockSpec(memory_space=plgpu.GMEM)
      stat_shape = jax.ShapeDtypeStruct((b, padded_rows), jnp.float32)
      out_shape = [jax.ShapeDtypeStruct((b, padded_rows, padded_len), x.dtype)]
      if return_mean:
        out_shape.append(stat_shape)
      if return_rstd:
        out_shape.append(stat_shape)
      return pl.pallas_call(
          dispatch,
          out_shape=out_shape,
          grid=(b, padded_rows // block_m),
          in_specs=[gmem] * len(arrays),
          out_specs=[gmem] * len(out_shape),
          name=name,
          compiler_params=plgpu.CompilerParams(
              lowering_semantics=plgpu.LoweringSemantics.Lane
          ),
      )(*arrays)

    # Give every argument a leading (size-1) batch dim; `_vmappable` folds any
    # `vmap` axes into it. `scale`/`offset` are indexed per batch element.
    in_arrays = [x_t] + [p for p in (scale, offset) if p is not None]
    outs = _vmappable(launch)(*(a[None] for a in in_arrays))

    # Drop the leading batch dim, the padded rows and padded tail, and undo the
    # canonicalizing transpose.
    y = jnp.moveaxis(outs[0].reshape(padded_rows, padded_len)[:rows, :axis_len]
                     .reshape(perm_shape), -1, axis)

    if not return_residuals:
      return y, None

    # Stats are laid out in the transposed (perm) order; move the reduced axis
    # (now trailing, size 1) back to its original position.
    def _unstat(flat):
      stat = flat.reshape(padded_rows)[:rows].reshape(perm_shape[:-1] + (1,))
      return jnp.moveaxis(stat, -1, axis)

    outs = list(outs[1:])
    mean = _unstat(outs.pop(0)) if return_mean else None
    rstddev = _unstat(outs.pop(0))
    return y, (mean, rstddev)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    cfg = triton_config.get_heuristics_config(
        *ba.args, vmap_axis_sizes=ba.vmap_axis_sizes, **ba.kwargs
    )
    # This backend flattens to 2D and interprets `block_n` as the `WG_STRIDED`
    # `vec_size`, not a trailing tile as in Triton — so drop Triton's 3D
    # `block_n` (`vec_size == 1`). Only `block_m` (rows per program) carries over.
    return Config(block_m=cfg.block_m, block_n=None, num_warps=cfg.num_warps)

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    return triton_config.get_key(*ba.args, **ba.kwargs)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    # This is a memory-bound, GMEM-direct kernel (no TMA, no MMA), so it also
    # runs on Ampere (sm80), not just the Hopper+ that Mosaic GPU officially
    # targets.
    return gpu_utils.is_sm80(device) or gpu_utils.has_mosaic_gpu_support(device)
