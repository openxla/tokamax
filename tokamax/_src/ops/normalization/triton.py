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
"""Triton normalization op implementation."""

import dataclasses
import math
from typing import Any, ClassVar, override

import jax
import jax.numpy as jnp
import jax_triton as jt
from tokamax._src import gpu_utils
from tokamax._src.ops import op
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import triton_config
from tokamax._src.ops.normalization import triton_vjp_config
import triton
import triton.language as tl


Config = triton_config.Config
Key = triton_config.Key
_NUM_REGISTERS_PER_SM = gpu_utils.NUM_REGISTERS_PER_SM


def _vmap_kernel(launch_fn, batch_shape=(), batched=()):
  """Wraps a kernel launch to support (nested) `jax.vmap`.

  `launch_fn(batch_shape, batch_strides, *args)` must return outputs with
  leading `batch_shape` dims. `batched[i][j]` is whether arg `j` is batched
  along `batch_shape[i]`; unbatched args get a zero batch stride.
  """

  @jax.custom_batching.custom_vmap
  def f(*args):
    strides = []
    for i, x in enumerate(args):
      x_strides = iter(() if x is None else jt.strides_from_shape(x.shape))
      strides.append(tuple(next(x_strides) if b[i] else 0 for b in batched))
    return launch_fn(batch_shape, strides, *args)

  @f.def_vmap
  def _(axis_size, in_batched, *args):
    out = _vmap_kernel(
        launch_fn, (axis_size, *batch_shape), (in_batched, *batched)
    )(*args)
    return out, jax.tree.map(lambda _: True, out)

  return f


@triton.jit
def _block_offsets_and_masks(
    pid_m, pid_n, m, a, n, block_m, block_a, block_n
):
  start_m = pid_m * block_m
  start_n = pid_n * block_n
  offs_m = tl.arange(0, block_m)[:, None, None]
  offs_a = tl.arange(0, block_a)
  offs_n = tl.arange(0, block_n)[None, None, :] if block_n > 1 else 0
  mask_m = True if m % block_m == 0 else (start_m + offs_m) < m
  mask_a = True if a == block_a else offs_a < a
  mask_n = True if n % block_n == 0 else (start_n + offs_n) < n
  stat_mask = mask_m & mask_n
  x_mask = stat_mask & (True if a == block_a else (offs_a < a)[None, :, None])
  # Only the scalar block bases may need 64 bits, per-element offsets don't.
  x_base = start_m * (a * n) + start_n
  x_offs = offs_m * (a * n) + offs_a[None, :, None] * n + offs_n
  stat_base = start_m * n + start_n
  stat_offs = offs_m * n + offs_n
  return x_base, x_offs, x_mask, stat_base, stat_offs, stat_mask, offs_a, mask_a


@triton.jit
def _program_ids(m, n, block_m, block_n, batch_shape):
  # Everything is folded into grid axis 0. Axes 1 and 2 are capped at 65535
  # blocks, and LLVM narrows div/mod on their 16-bit ids to `v2i16` ops, which
  # crash NVPTX isel.
  grid_m: tl.constexpr = tl.cdiv(m, block_m)
  grid_n: tl.constexpr = tl.cdiv(n, block_n)
  pid = tl.program_id(0)
  pid_m = (pid % grid_m).to(tl.int64)
  pid_n = (pid // grid_m % grid_n).to(tl.int64)
  pid_b = 0 if batch_shape == () else (pid // (grid_m * grid_n)).to(tl.int64)
  return pid_b, pid_m, pid_n


@triton.jit
def _batch_offset(pid_b, batch_shape, batch_strides):
  offset = 0
  for i in tl.static_range(len(batch_shape) - 1, -1, -1):
    offset += (pid_b % batch_shape[i]) * batch_strides[i]
    pid_b //= batch_shape[i]
  return offset


@triton.jit
def _normalization_kernel(
    x_ptr,
    scale_ptr,
    offset_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    batch_shape: tl.constexpr,
    m: tl.constexpr,
    a: tl.constexpr,
    n: tl.constexpr,
    x_batch_strides: tl.constexpr,
    scale_batch_strides: tl.constexpr,
    offset_batch_strides: tl.constexpr,
    epsilon: tl.constexpr,
    scale_offset: tl.constexpr,
    subtract_mean: tl.constexpr,
    block_m: tl.constexpr,
    block_a: tl.constexpr,
    block_n: tl.constexpr,
):
  """Normalization forward kernel."""
  pid_b, pid_m, pid_n = _program_ids(m, n, block_m, block_n, batch_shape)
  x_base, x_offs, x_mask, stat_base, stat_offs, stat_mask, offs_a, mask_a = (
      _block_offsets_and_masks(pid_m, pid_n, m, a, n, block_m, block_a, block_n)
  )

  dtype = tl.float64 if x_ptr.dtype.element_ty == tl.float64 else tl.float32
  x_ptr += _batch_offset(pid_b, batch_shape, x_batch_strides) + x_base
  x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(dtype)

  if subtract_mean:
    mean = tl.sum(x, axis=1, keep_dims=True) / a
    if mean_ptr is not None:
      mean_ptr += pid_b * (m * n) + stat_base
      tl.store(mean_ptr + stat_offs, mean, mask=stat_mask)
    x -= mean
    if a != block_a:
      x = tl.where(mask_a[None, :, None], x, 0.0)

  var = tl.sum(x * x, axis=1, keep_dims=True) / a
  rstddev = tl.rsqrt(var + epsilon)
  if rstd_ptr is not None:
    rstd_ptr += pid_b * (m * n) + stat_base
    tl.store(rstd_ptr + stat_offs, rstddev, mask=stat_mask)
  x *= rstddev

  if scale_ptr is not None:
    scale_ptr += _batch_offset(pid_b, batch_shape, scale_batch_strides)
    scale = tl.load(scale_ptr + offs_a, mask=mask_a, other=0.0).to(dtype)
    if scale_offset != 0.0:
      scale += scale_offset
    x *= scale[None, :, None]
  if offset_ptr is not None:
    offset_ptr += _batch_offset(pid_b, batch_shape, offset_batch_strides)
    offset = tl.load(offset_ptr + offs_a, mask=mask_a, other=0.0).to(dtype)
    x += offset[None, :, None]

  out_ptr = x_ptr if y_ptr is None else y_ptr + pid_b * (m * a * n) + x_base
  tl.store(out_ptr + x_offs, x.to(out_ptr.dtype.element_ty), mask=x_mask)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TritonNormalization(base.Normalization[Config, Key]):
  """Triton normalization op."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  # If `None`, `input_output_alias = not return_residuals`.
  input_output_alias: bool | None = None

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, 'vjp', TritonNormalizationVjp())

  @override
  def _fwd(
      self,
      x: jax.Array | base.FusedInputArray,
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
      raise NotImplementedError('Callable `x` is not supported.')

    input_output_alias = self.input_output_alias
    if input_output_alias is None:
      input_output_alias = not return_residuals

    orig_x_shape = x.shape
    m, a, n = triton_config.canonicalize_shape_3d(orig_x_shape, axis)
    x = x.reshape(m, a, n)

    block_m = config.block_m
    block_n = 1 if config.block_n is None else config.block_n
    block_a = triton.next_power_of_2(a)
    grid_m = triton.cdiv(m, block_m)
    grid_n = triton.cdiv(n, block_n)

    name = 'triton_layer_norm' if subtract_mean else 'triton_rms_norm'
    if return_residuals:
      name += '_fwd_res'

    @_vmap_kernel
    def fwd_call(batch_shape, batch_strides, x, scale, offset):
      x_strides, scale_strides, offset_strides = batch_strides
      # `x` cannot be overwritten if it is shared across the batch.
      alias = input_output_alias and all(x_strides)
      y_shape = jax.ShapeDtypeStruct((*batch_shape, m, a, n), x.dtype)
      stat_shape = jax.ShapeDtypeStruct((*batch_shape, m, 1, n), jnp.float32)
      out_type = (
          None if alias else y_shape,
          stat_shape if return_residuals and subtract_mean else None,
          stat_shape if return_residuals else None,
      )
      none_outs: dict[str, Any] = {
          k: None
          for k, v in zip(
              ('y_ptr', 'mean_ptr', 'rstd_ptr'), out_type, strict=True
          )
          if v is None
      }
      x_or_ref = jax.new_ref(x) if alias else x
      y, mean, rstddev = jt.triton_call(
          x_or_ref,
          scale,
          offset,
          kernel=_normalization_kernel,
          out_type=out_type,
          grid=(math.prod(batch_shape) * grid_m * grid_n,),
          name=name,
          num_warps=config.num_warps,
          batch_shape=batch_shape,
          m=m,
          a=a,
          n=n,
          x_batch_strides=x_strides,
          scale_batch_strides=scale_strides,
          offset_batch_strides=offset_strides,
          epsilon=epsilon,
          scale_offset=scale_offset,
          subtract_mean=subtract_mean,
          block_m=block_m,
          block_a=block_a,
          block_n=block_n,
          **none_outs,
      )
      if alias:
        y = jax.freeze(x_or_ref)
      return y, mean, rstddev

    y, mean, rstddev = fwd_call(x, scale, offset)
    y = y.reshape(orig_x_shape)
    stat_shape = list(orig_x_shape)
    stat_shape[axis] = 1
    if mean is not None:
      mean = mean.reshape(stat_shape)
    if rstddev is not None:
      rstddev = rstddev.reshape(stat_shape)

    return y, (mean, rstddev) if return_residuals else None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return triton_config.get_heuristics_config(
        *ba.args, vmap_axis_sizes=ba.vmap_axis_sizes, **ba.kwargs
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    # TODO: Use batched args.
    return triton_config.get_key(*ba.args, **ba.kwargs)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    x = ba.args[0]
    axis = ba.kwargs['axis']
    x_shape = triton_config.canonicalize_shape(x.shape, axis)
    configs = set()
    # `num_stages` has no effect, as there is no loop within kernel.
    for num_warps in [1, 2, 4, 8, 16]:
      for block_m in [1, 2, 4, 8, 16, 32, 64]:
        block_m = min(block_m, triton.next_power_of_2(x_shape[0]))

        if len(x_shape) > 2:
          for block_n in [16, 32, 64, 128]:
            block_n = min(block_n, triton.next_power_of_2(x_shape[2]))
            # Ensure block size fits in registers.
            if (block_m * x_shape[1] * block_n <= _NUM_REGISTERS_PER_SM) or (
                block_m == 1 and block_n <= 16
            ):
              configs.add(
                  Config(block_m=block_m, block_n=block_n, num_warps=num_warps)
              )
        # Ensure block size fits in registers.
        elif block_m * x_shape[1] <= _NUM_REGISTERS_PER_SM or block_m == 1:
          config = Config(block_m=block_m, block_n=None, num_warps=num_warps)
          configs.add(config)
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)


@triton.jit
def _normalization_vjp_kernel(
    dout_ptr,
    x_ptr,
    scale_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    dscale_ptr,
    doffset_ptr,
    batch_shape: tl.constexpr,
    m: tl.constexpr,
    a: tl.constexpr,
    n: tl.constexpr,
    dout_batch_strides: tl.constexpr,
    x_batch_strides: tl.constexpr,
    scale_batch_strides: tl.constexpr,
    mean_batch_strides: tl.constexpr,
    rstd_batch_strides: tl.constexpr,
    scale_offset: tl.constexpr,
    block_m: tl.constexpr,
    block_a: tl.constexpr,
    block_n: tl.constexpr,
):
  """Normalization VJP kernel."""
  pid_b, pid_m, pid_n = _program_ids(m, n, block_m, block_n, batch_shape)
  x_base, x_offs, x_mask, stat_base, stat_offs, stat_mask, offs_a, mask_a = (
      _block_offsets_and_masks(pid_m, pid_n, m, a, n, block_m, block_a, block_n)
  )

  dtype = tl.float64 if x_ptr.dtype.element_ty == tl.float64 else tl.float32
  x_ptr += _batch_offset(pid_b, batch_shape, x_batch_strides) + x_base
  x_norm = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(dtype)
  if mean_ptr is not None:
    mean_ptr += (
        _batch_offset(pid_b, batch_shape, mean_batch_strides) + stat_base
    )
    x_norm -= tl.load(mean_ptr + stat_offs, mask=stat_mask, other=0.0).to(dtype)
  rstd_ptr += _batch_offset(pid_b, batch_shape, rstd_batch_strides) + stat_base
  rstddev = tl.load(rstd_ptr + stat_offs, mask=stat_mask, other=0.0).to(dtype)
  x_norm *= rstddev

  dout_ptr += _batch_offset(pid_b, batch_shape, dout_batch_strides) + x_base
  dout = tl.load(dout_ptr + x_offs, mask=x_mask, other=0.0).to(dtype)
  dparam_offs = tl.program_id(0).to(tl.int64) * a + offs_a

  if doffset_ptr is not None:
    doffset = tl.sum(tl.sum(dout, axis=2), axis=0)
    tl.store(doffset_ptr + dparam_offs, doffset, mask=mask_a)

  if dscale_ptr is not None:
    dscale = tl.sum(tl.sum(dout * x_norm, axis=2), axis=0)
    tl.store(dscale_ptr + dparam_offs, dscale, mask=mask_a)
    scale_ptr += _batch_offset(pid_b, batch_shape, scale_batch_strides)
    scale = tl.load(scale_ptr + offs_a, mask=mask_a, other=0.0).to(dtype)
    if scale_offset != 0.0:
      scale += scale_offset
    dout *= scale[None, :, None]

  dx1 = -(tl.sum(dout * x_norm, axis=1, keep_dims=True) / a) * x_norm
  dx2 = (
      0.0 if mean_ptr is None else -(tl.sum(dout, axis=1, keep_dims=True) / a)
  )
  out_ptr = x_ptr if dx_ptr is None else dx_ptr + pid_b * (m * a * n) + x_base
  tl.store(
      out_ptr + x_offs,
      ((dout + dx1 + dx2) * rstddev).to(out_ptr.dtype.element_ty),
      mask=x_mask,
  )


@dataclasses.dataclass(frozen=True, slots=True)
class TritonNormalizationVjp(base.NormalizationVjp[Config, Key]):
  """Triton normalization VJP."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  def _fwd(
      self,
      residuals: base.Residuals,
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
    del out, epsilon  # Unused.

    if return_residuals:
      raise NotImplementedError('`return_residuals` not supported.')

    mean, rstddev = residuals
    if (mean is not None) != subtract_mean:
      raise ValueError('`mean` residual inconsistent with `subtract_mean`.')

    orig_x_shape = x.shape
    m, a, n = triton_config.canonicalize_shape_3d(orig_x_shape, axis)
    x = x.reshape(m, a, n)
    dout = dout.reshape(m, a, n)
    if mean is not None:
      mean = mean.reshape(m, 1, n)
    rstddev = rstddev.reshape(m, 1, n)

    block_m = config.block_m
    block_n = 1 if config.block_n is None else config.block_n
    block_a = triton.next_power_of_2(a)
    grid_m = triton.cdiv(m, block_m)
    grid_n = triton.cdiv(n, block_n)
    name = 'triton_rms_norm_vjp' if mean is None else 'triton_layer_norm_vjp'

    @_vmap_kernel
    def vjp_call(batch_shape, batch_strides, dout, x, scale, mean, rstddev):
      (
          dout_strides,
          x_strides,
          scale_strides,
          mean_strides,
          rstddev_strides,
      ) = batch_strides
      alias = all(x_strides)
      dx_shape = jax.ShapeDtypeStruct((*batch_shape, m, a, n), x.dtype)
      dparam_shape = jax.ShapeDtypeStruct(
          (*batch_shape, grid_m * grid_n, a), jnp.float32
      )
      out_type = (
          None if alias else dx_shape,
          None if scale is None else dparam_shape,
          None if offset is None else dparam_shape,
      )
      none_outs: dict[str, Any] = {
          k: None
          for k, v in zip(
              ('dx_ptr', 'dscale_ptr', 'doffset_ptr'), out_type, strict=True
          )
          if v is None
      }
      x_or_ref = jax.new_ref(x) if alias else x
      dx, dscale, doffset = jt.triton_call(
          dout,
          x_or_ref,
          scale,
          mean,
          rstddev,
          kernel=_normalization_vjp_kernel,
          out_type=out_type,
          grid=(math.prod(batch_shape) * grid_m * grid_n,),
          name=name,
          num_warps=config.num_warps,
          batch_shape=batch_shape,
          m=m,
          a=a,
          n=n,
          dout_batch_strides=dout_strides,
          x_batch_strides=x_strides,
          scale_batch_strides=scale_strides,
          mean_batch_strides=mean_strides,
          rstd_batch_strides=rstddev_strides,
          scale_offset=scale_offset,
          block_m=block_m,
          block_a=block_a,
          block_n=block_n,
          **none_outs,
      )
      if alias:
        dx = jax.freeze(x_or_ref)
      if scale is not None:
        dscale = jnp.sum(dscale, axis=-2).astype(scale.dtype)
      if offset is not None:
        doffset = jnp.sum(doffset, axis=-2).astype(offset.dtype)
      return dx, dscale, doffset

    dx, dscale, doffset = vjp_call(dout, x, scale, mean, rstddev)
    return (dx.reshape(orig_x_shape), dscale, doffset), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    _, _, _, x, scale, offset = ba.args
    return triton_config.get_heuristics_config(
        x,
        scale,
        offset,
        block_size_per_warp=2048,
        vmap_axis_sizes=ba.vmap_axis_sizes,
        **ba.kwargs,
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    return triton_vjp_config.get_key(*ba.args, **ba.kwargs)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    axis = ba.kwargs['axis']
    dout_shape = triton_config.canonicalize_shape(ba.args[1].shape, axis)
    configs = set()
    # `num_stages` has no effect, as there is no loop within kernel.
    for num_warps in [1, 2, 4, 8, 16]:
      for block_m in [1, 2, 4, 8, 16, 32, 64, 128]:
        if block_m > triton.next_power_of_2(dout_shape[0]):
          break

        config = Config(block_m=block_m, block_n=None, num_warps=num_warps)
        if len(dout_shape) > 2:
          for block_n in [16, 32, 64, 128]:
            if block_n > max(triton.next_power_of_2(dout_shape[2]), 16):
              break
            # Ensure two full blocks (`x` and `dout`) fit in registers.
            if 2 * block_m * dout_shape[1] * block_n <= _NUM_REGISTERS_PER_SM:
              configs.add(dataclasses.replace(config, block_n=block_n))
        # Ensure two full blocks (`x` and `dout`) fit in registers.
        elif 2 * block_m * dout_shape[1] <= _NUM_REGISTERS_PER_SM:
          configs.add(config)
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)
