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
"""Pallas-Triton normalization op implementation."""

import dataclasses
import functools
from typing import TypeAlias

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
from tokamax._src.ops import op
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import pallas_triton_config
from tokamax._src.ops.normalization import pallas_triton_vjp
from tokamax._src.pallas import block


Config: TypeAlias = pallas_triton_config.Config
Key: TypeAlias = pallas_triton_config.Key
_NUM_REGISTERS_PER_SM = pallas_triton_config.NUM_REGISTERS_PER_SM


def _normalization_kernel(
    x_ref,
    scale_ref,
    offset_ref,
    pre_scale_ref,
    y_ref,
    mean_ref,
    rstddev_ref,
    *,
    epsilon,
    scale_offset,
    subtract_mean,
):
  """Normalization kernel."""
  dtype = jnp.promote_types(x_ref.dtype, jnp.float32)
  x = x_ref.load().astype(dtype)

  if pre_scale_ref is not None:
    x *= pre_scale_ref.load().astype(dtype)

  axis_len = x_ref.full_shape[1]
  if subtract_mean:
    mean = jnp.sum(x, axis=1, keepdims=True) / axis_len
    if mean_ref is not None:
      mean_ref.store(mean)
    x -= mean
    # # Zero invalid values (when axis is not a power of two).
    if x.shape[1] != axis_len:
      x *= (jnp.indices(x.shape, sparse=True)[1] < axis_len).astype(x.dtype)

  var = jnp.sum(jnp.square(x), axis=1, keepdims=True) / axis_len
  rstddev = jax.lax.rsqrt(var + epsilon)
  if rstddev_ref is not None:
    rstddev_ref.store(rstddev)
  x *= rstddev

  if scale_ref is not None:
    x *= scale_ref.load(bounds_check=False).astype(x.dtype) + scale_offset
  if offset_ref is not None:
    x += offset_ref.load().astype(x.dtype)

  y_ref.store(x.astype(y_ref.dtype))


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTritonNormalization(base.Normalization[Config, Key]):
  """Pallas-Triton normalization op."""

  # If `None`, `input_output_alias = not return_residuals`.
  input_output_alias: bool | None = None

  def __post_init__(self):
    if self.vjp is None:
      vjp = pallas_triton_vjp.PallasTritonNormalizationVjp()
      object.__setattr__(self, 'vjp', vjp)

  def _fwd(
      self,
      x: jax.Array,
      scale: jax.Array | None,
      offset: jax.Array | None,
      *,
      axis: int,
      epsilon: float,
      scale_offset: float,
      subtract_mean: bool,
      return_residuals: bool,
      pre_scale: jax.Array | None = None,
      config: Config,
  ) -> tuple[jax.Array, base.Residuals | None]:
    kernel = functools.partial(
        _normalization_kernel,
        epsilon=epsilon,
        scale_offset=scale_offset,
        subtract_mean=subtract_mean,
    )

    # Cananonicalize to 3D, where the second axis is the reduced axis.
    orig_x_shape = x.shape
    x = x.reshape(pallas_triton_config.canonicalize_shape_3d(x.shape, axis))
    param_bcast = lambda p: None if p is None else p[:, None]

    stat_shape = jax.ShapeDtypeStruct((x.shape[0], 1, x.shape[2]), jnp.float32)
    out_shape = (
        x,
        stat_shape if return_residuals and subtract_mean else None,
        stat_shape if return_residuals else None,
    )

    block_m = config.block_m
    block_n = 1 if config.block_n is None else config.block_n
    block_a = pl.next_power_of_2(x.shape[1])

    x_spec = pl.BlockSpec((block_m, block_a, block_n), lambda i, j: (i, 0, j))
    param_spec = pl.BlockSpec((block_a, 1), lambda i, j: (0, 0))
    stat_spec = pl.BlockSpec((block_m, 1, block_n), lambda i, j: (i, 0, j))

    input_output_alias = self.input_output_alias
    if input_output_alias is None:
      input_output_alias = not return_residuals

    name = 'pallas_layer_norm' if subtract_mean else 'pallas_rms_norm'
    if return_residuals:
      name += '_fwd_res'

    y, mean, rstddev = block.pallas_call(
        kernel,
        name=name,
        out_shape=out_shape,
        grid=(pl.cdiv(x.shape[0], block_m), pl.cdiv(x.shape[2], block_n)),
        in_specs=(x_spec, param_spec, param_spec, param_spec),
        out_specs=(x_spec, stat_spec, stat_spec),
        filter_specs=True,
        input_output_aliases={0: 0} if input_output_alias else {},
        compiler_params=plgpu.CompilerParams(num_warps=config.num_warps),
    )(x, param_bcast(scale), param_bcast(offset), param_bcast(pre_scale))

    y = y.reshape(orig_x_shape)
    stat_shape = list(orig_x_shape)
    stat_shape[axis] = 1
    if mean is not None:
      mean = mean.reshape(stat_shape)
    if rstddev is not None:
      rstddev = rstddev.reshape(stat_shape)

    return y, (mean, rstddev) if return_residuals else None

  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return pallas_triton_config.get_heuristics_config(
        *ba.args, vmap_axis_sizes=ba.batched.vmap_axis_sizes, **ba.kwargs
    )

  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    # TODO(cjfj): Use batched args.
    return pallas_triton_config.get_key(*ba.args, **ba.kwargs)

  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    x = ba.args[0]
    axis = ba.kwargs['axis']
    x_shape = pallas_triton_config.canonicalize_shape(x.shape, axis)
    configs = set()
    # `num_stages` has no effect, as there is no loop within kernel.
    for num_warps in [1, 2, 4, 8, 16]:
      for block_m in [1, 2, 4, 8, 16, 32, 64]:
        block_m = min(block_m, pl.next_power_of_2(x_shape[0]))

        if len(x_shape) > 2:
          for block_n in [16, 32, 64, 128]:
            block_n = min(block_n, pl.next_power_of_2(x_shape[2]))
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
