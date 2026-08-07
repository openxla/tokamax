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

Lane semantics is required: Warpgroup layout inference cannot solve a program
containing a partial reduction, so layouts are explicit.
"""

import dataclasses
from typing import ClassVar, override

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src.ops import op
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import triton_config

# Reuse the Triton config only for its cache key
Config = triton_config.Config
Key = triton_config.Key
FusedInputArray = base.FusedInputArray

# CUDA allows 2**31-1 blocks in gridDim.x but only 65535 in .y and .z.
_MAX_GRID_DIM = 65535

_LANE_SEMANTICS = plgpu.CompilerParams(
  lowering_semantics=plgpu.LoweringSemantics.Lane
)


def _largest_divisor(n: int, cap: int) -> int:
  """Returns the largest divisor of `n` that is at most `cap`."""
  return next(b for b in range(min(cap, n), 0, -1) if n % b == 0)


def _largest_pow2(cap: int, ok) -> int | None:
  """Returns the largest power of two `<= cap` satisfying `ok`, or `None`."""
  v = 1 << (cap.bit_length() - 1)
  while not ok(v):
    if v == 1:
      return None
    v //= 2
  return v


def _vmappable(launch):
  """Folds a `vmap` axis into the leading axis of the canonical `(M, A, N)`.

  `batching.capture_batched_args` records the batch sizes for the heuristics
  but still calls `jax.vmap` over `_fwd` (see `batching.py`), so without this
  JAX batches the `plgpu.kernel` launch itself. `pallas_call` has a batching
  rule that handles that well -- which is why `pallas_triton` needs nothing --
  but `plgpu.kernel` does not, and it measured ~2.7x worse per element than a
  single larger launch.

  Rows of the canonical form are normalized independently, so a batch of them
  is just more rows. Blocks may straddle a batch boundary, which is harmless
  precisely because the rows are independent -- but only while `scale`/`offset`
  are shared, which is the only case handled here.
  """
  f = jax.custom_batching.custom_vmap(launch)

  @f.def_vmap
  def _rule(axis_size, in_batched, x, *params):
    x_batched, *params_batched = in_batched
    if any(params_batched):
      raise NotImplementedError(
        'The Mosaic GPU kernel does not support batched `scale`/`offset`.'
      )
    if not x_batched:
      out = f(x, *params)
      return out, jax.tree.map(lambda _: False, out)
    out = f(x.reshape(axis_size * x.shape[1], *x.shape[2:]), *params)
    unmerge = lambda o: o.reshape(axis_size, o.shape[0] // axis_size, *o.shape[1:])
    return jax.tree.map(unmerge, out), jax.tree.map(lambda _: True, out)

  return f


def _coalesced_layout(block_slow: int, block_fast: int, itemsize: int):
  """Register layout for a 2D `(slow, fast)` tile, `fast` being contiguous.

  Lanes and the vector both come from `fast`, so a warp covers
  `32 * vector` contiguous elements -- one unbroken memory transaction. The
  warps come from `slow`. Whether the reduction ends up crossing warps (and so
  needing SMEM scratch) depends on which axis is being reduced, and is
  deliberately not what this optimises for.
  """
  if block_slow % 4:
    raise NotImplementedError(
      f'Slow axis ({block_slow}) must be a multiple of 4 for the warps.'
    )
  max_vec = max(1, 16 // itemsize)  # 16-byte (128-bit) register vectors.
  v = _largest_pow2(max_vec, lambda v: block_fast % (32 * v) == 0)
  if v is None:
    raise NotImplementedError(
      f'Contiguous axis ({block_fast}) must be a multiple of 32 so the lanes'
      ' cover it.'
    )
  return plgpu.Layout.TILED(
    plgpu.Tiling(((4, 32 * v), (4, v))), (-2,), (-3,), -1
  )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicGpuNormalization(base.Normalization[Config, Key]):
  """Pallas-Mosaic-GPU normalization op."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  input_output_alias: bool | None = None

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

    # Everything below depends on `M`, which `_vmappable` changes when it folds
    # a batch in, so it all has to live inside the launch.
    def launch(x, *params):
      num_m, num_a, num_b = x.shape

      # One dispatch for the whole case split. The refs are shaped to the case
      # so the kernel never indexes a rank away -- an integer index into a 3D
      # ref does not squeeze on the store side.
      if num_m == 1 and num_b == 1:
        # Wholly degenerate: the tile is 1D, so reducing its only axis *is* an
        # all-axes reduction -- the one case the strided layout handles, and
        # the params already match the tile so they need no broadcast.
        if num_a % 128:
          raise NotImplementedError(
            f'A 1D input needs its reduced axis ({num_a}) to be a multiple of'
            ' 128 for the strided layout.'
          )
        block_m = block_b = 1
        x_shape, stat_shape = (num_a,), (1,)
        layout, reduce_axis, stat_dims, param_dims, param_axes = (
          None, None, (), None, None
        )
      elif num_b == 1:
        # `A` is contiguous *and* reduced, so the lanes land on it and the
        # reduction is a lane shuffle.
        block_m, block_b = _largest_divisor(num_m, config.block_m), 1
        x_shape, stat_shape = (num_m, num_a), (num_m,)
        layout = _coalesced_layout(block_m, num_a, dtype.itemsize)
        reduce_axis, stat_dims, param_dims, param_axes = 1, (0,), (1,), (0,)
      else:
        # `N` is contiguous, so the warps land on `A` and the reduction takes
        # the scratch path. `M` goes entirely to the grid: it adds nothing to
        # coalescing and only inflates the register footprint.
        block_m, block_b = 1, _largest_divisor(num_b, config.block_n or 1)
        x_shape, stat_shape = (num_m, num_a, num_b), (num_m, num_b)
        layout = _coalesced_layout(num_a, block_b, dtype.itemsize)
        reduce_axis, stat_dims, param_dims, param_axes = 0, (1,), (0,), (1,)

      # Reductions run in f32 and the body keeps temporaries live, so the real
      # cost is a small multiple of this. Past a few hundred it spills, and the
      # whole point (staying bandwidth-bound) is lost.
      if (regs := block_m * num_a * block_b // 128) > 256:
        raise NotImplementedError(
          f'Block ({block_m}, {num_a}, {block_b}) puts {regs} elements in each'
          ' thread, which will spill.'
        )

      # CUDA caps gridDim.y/.z at 65535 (only .x is 2**31-1) and Mosaic does
      # not map the leading axis to .x, so too many rows aborts the launch with
      # `cuLaunchKernelEx: invalid argument`. Folding a vmap batch into `M`
      # multiplies the row grid by the batch size, which is how you get there.
      grid_m, grid_b = num_m // block_m, num_b // block_b
      split_m, grid, grid_names = 1, (grid_m, grid_b), ('m', 'n')
      if grid_m > _MAX_GRID_DIM:
        split_m = _largest_divisor(grid_m, _MAX_GRID_DIM)
        if grid_m // split_m > _MAX_GRID_DIM:
          raise NotImplementedError(
            f'Row grid {grid_m} does not factor into two axes of at most'
            f' {_MAX_GRID_DIM}.'
          )
        grid, grid_names = (grid_m // split_m, split_m, grid_b), ('mo', 'm', 'n')

      def kernel(*refs):
        it = iter(refs)  # Inputs then outputs, optional ones only if present.
        take = lambda present: next(it) if present else None
        x_ref, scale_ref, offset_ref = next(it), take(has_scale), take(has_offset)
        y_ref, mean_ref = next(it), take(return_mean)
        rstd_ref = take(return_residuals)
        *pid_m_parts, pid_b = map(pl.program_id, range(len(grid)))
        pid_m = pid_m_parts[-1]
        if len(pid_m_parts) == 2:
          pid_m += pid_m_parts[0] * split_m
        m_slice = pl.ds(pid_m * block_m, block_m)

        if len(x_shape) == 1:
          block_idx, stat_idx = slice(None), 0
        elif len(x_shape) == 2:
          block_idx = stat_idx = m_slice
        else:
          b_slice = pl.ds(pid_b * block_b, block_b)
          block_idx, stat_idx = (pid_m, slice(None), b_slice), (pid_m, b_slice)

        load = lambda ref, lo: plgpu.load(ref, layout=lo, optimized=False)
        # Reduce in f32 regardless of the input dtype, as Triton does.
        x = load(x_ref.at[block_idx], layout).astype(jnp.float32)

        def bcast(a, dims):
          a = jax.lax.broadcast_in_dim(a, x.shape, dims)
          # A splat source needs no hint; the broadcast rule handles it.
          return a if layout is None else plgpu.layout_cast(a, layout)

        def read_param(ref):
          """Reads a 1D param, laid out to match `x` along the reduced axis."""
          if layout is None:  # 1D tile: the param already matches it.
            return load(ref, None).astype(jnp.float32)
          p = load(ref, layout.reduce(param_axes)).astype(jnp.float32)
          return bcast(p, param_dims)

        # `reduce_axis is None` reduces every axis, giving a scalar.
        row_mean = lambda a: jnp.sum(a, axis=reduce_axis) / num_a

        if subtract_mean:
          mean = row_mean(x)
          x -= bcast(mean, stat_dims)
          if mean_ref is not None:
            mean_ref[stat_idx] = mean
        rstddev = jax.lax.rsqrt(row_mean(jnp.square(x)) + epsilon)
        if rstd_ref is not None:
          rstd_ref[stat_idx] = rstddev
        x *= bcast(rstddev, stat_dims)
        # `y = x_norm * (scale + scale_offset) + offset`; see `base.Normalization`.
        if scale_ref is not None:
          x *= read_param(scale_ref) + scale_offset
        if offset_ref is not None:
          x += read_param(offset_ref)
        y_ref[block_idx] = x.astype(dtype)

      stat = jax.ShapeDtypeStruct(stat_shape, jnp.float32)
      outs = plgpu.kernel(
        kernel,
        out_type=(
          jax.ShapeDtypeStruct(x_shape, dtype),
          *[stat] * (return_mean + return_residuals),
        ),
        grid=grid,
        grid_names=grid_names,
        compiler_params=_LANE_SEMANTICS,
      )(x.reshape(x_shape), *params)

      # Always hand back canonical shapes, so `_vmappable` can fold and unfold
      # the batch on the leading axis without knowing which case ran.
      return (
        outs[0].reshape(num_m, num_a, num_b),
        *(o.reshape(num_m, num_b) for o in outs[1:]),
      )

    params = [p for p in (scale, offset) if p is not None]
    outs = _vmappable(launch)(x.reshape(num_m, num_a, num_b), *params)

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
    # No TMA, WGMMA or async copies: plain GMEM loads and stores run on Ampere.
    return gpu_utils.has_mosaic_gpu_support(device, min_compute_capability=8.0)
