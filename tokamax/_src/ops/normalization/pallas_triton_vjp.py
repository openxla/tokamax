# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Pallas-Triton normalization VJP implementation."""

import dataclasses
import functools
from typing import ClassVar, TypeAlias

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src.ops import op
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import pallas_triton_config
from tokamax._src.ops.normalization import pallas_triton_vjp_config
from tokamax._src.pallas import block
from typing_extensions import override


_NUM_REGISTERS_PER_SM = gpu_utils.NUM_REGISTERS_PER_SM


Config: TypeAlias = pallas_triton_vjp_config.Config
Key: TypeAlias = pallas_triton_vjp_config.Key
Residuals: TypeAlias = base.Residuals


def _normalization_vjp_kernel(
    dout_ref,
    x_ref,
    scale_ref,
    mean_ref,
    rstddev_ref,
    dx_ref,
    dscale_ref,
    doffset_ref,
    *,
    scale_offset,
):
  """Normalization VJP kernel."""
  dtype = jnp.promote_types(x_ref.dtype, jnp.float32)

  x_norm = x_ref.load().astype(dtype)
  if mean_ref is not None:
    x_norm -= mean_ref.load().astype(dtype)
  rstddev = rstddev_ref.load().astype(dtype)
  x_norm *= rstddev

  dout = dout_ref.load().astype(dtype)
  if doffset_ref is not None:
    # Split `sum` and keep dims to work around Triton bug.
    doffset = jnp.sum(dout, axis=2, keepdims=True)
    doffset_ref.store(jnp.sum(doffset, axis=0, keepdims=True))

  if dscale_ref is not None:
    # Split `sum` and keep dims to work around Triton bug.
    dscale = jnp.sum(dout * x_norm, axis=2, keepdims=True)
    dscale_ref.store(jnp.sum(dscale, axis=0, keepdims=True))
    dout *= scale_ref.load().astype(dtype) + scale_offset

  n = x_ref.full_shape[1]
  dx1 = -(jnp.sum(dout * x_norm, axis=1, keepdims=True) / n) * x_norm
  dx2 = 0.0 if mean_ref is None else -(jnp.sum(dout, axis=1, keepdims=True) / n)
  dx_ref.store(((dout + dx1 + dx2) * rstddev).astype(dx_ref.dtype))


@dataclasses.dataclass(frozen=True, slots=True)
class PallasTritonNormalizationVjp(base.NormalizationVjp[Config, Key]):
  """Pallas-Triton normalization VJP."""

  config_cls: ClassVar[type[Config]] = Config

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
    del out  # Unused.

    if return_residuals:
      raise NotImplementedError('`return_residuals` not supported.')

    mean, rstddev = residuals
    if (mean is not None) != subtract_mean:
      raise ValueError('`mean` residual inconsistent with `subtract_mean`.')

    kernel = functools.partial(
        _normalization_vjp_kernel, scale_offset=scale_offset
    )

    # Cananonicalize to 3D, where the second axis is the reduced axis.
    orig_x_shape = x.shape
    x = x.reshape(pallas_triton_config.canonicalize_shape_3d(x.shape, axis))
    dout = dout.reshape(x.shape)

    if scale is not None:
      scale = scale[:, None]

    stat_shape = pallas_triton_config.canonicalize_shape_3d(rstddev.shape, axis)
    if mean is not None:
      mean = mean.reshape(stat_shape)
    rstddev = rstddev.reshape(stat_shape)

    block_m = config.block_m
    block_n = 1 if config.block_n is None else config.block_n
    block_a = pl.next_power_of_2(x.shape[1])

    grid = (pl.cdiv(x.shape[0], block_m), pl.cdiv(x.shape[2], block_n))
    dparam_shape = (grid[0], x.shape[1], grid[1])
    dparam_shape = jax.ShapeDtypeStruct(dparam_shape, jnp.float32)
    out_shape = (
        x,
        None if scale is None else dparam_shape,
        None if offset is None else dparam_shape,
    )

    x_spec = pl.BlockSpec((block_m, block_a, block_n), lambda i, j: (i, 0, j))
    param_spec = pl.BlockSpec((block_a, 1), lambda i, j: (0, 0))
    stat_spec = pl.BlockSpec((block_m, 1, block_n), lambda i, j: (i, 0, j))
    dparam_spec = pl.BlockSpec((1, block_a, 1), lambda i, j: (i, 0, j))

    dx, dscale, doffset = block.pallas_call(
        kernel,
        name='pallas_rms_norm_vjp' if mean is None else 'pallas_layer_norm_vjp',
        out_shape=out_shape,
        grid=grid,
        in_specs=(x_spec, x_spec, param_spec, stat_spec, stat_spec),
        out_specs=(x_spec, dparam_spec, dparam_spec),
        filter_specs=True,
        input_output_aliases={1: 0},
        compiler_params=plgpu.CompilerParams(num_warps=config.num_warps),
    )(dout, x, scale, mean, rstddev)

    if scale is not None:
      dscale = jnp.sum(dscale, axis=(0, 2)).astype(scale.dtype)

    if offset is not None:
      doffset = jnp.sum(doffset, axis=(0, 2)).astype(offset.dtype)

    return (dx.reshape(orig_x_shape), dscale, doffset), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return pallas_triton_vjp_config.get_heuristics_config(
        *ba.args, vmap_axis_sizes=ba.vmap_axis_sizes, **ba.kwargs
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    # TODO: Use batched args.
    return pallas_triton_vjp_config.get_key(*ba.args, **ba.kwargs)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    axis = ba.kwargs['axis']
    dout_shape = pallas_triton_config.canonicalize_shape(ba.args[1].shape, axis)
    configs = set()
    # `num_stages` has no effect, as there is no loop within kernel.
    for num_warps in [1, 2, 4, 8, 16]:
      for block_m in [1, 2, 4, 8, 16, 32, 64, 128]:
        if block_m > pl.next_power_of_2(dout_shape[0]):
          break

        config = Config(block_m=block_m, block_n=None, num_warps=num_warps)
        if len(dout_shape) > 2:
          for block_n in [16, 32, 64, 128]:
            if block_n > max(pl.next_power_of_2(dout_shape[2]), 16):
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
