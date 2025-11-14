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
"""Ragged dot Pallas-Mosaic-GPU implementation."""

import dataclasses
from typing import ClassVar

import jax
from jax.extend import backend
import jax.numpy as jnp
import qwix
from tokamax._src import mosaic_gpu as mosaic_gpu_lib
from tokamax._src import precision as precision_lib
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.ragged_dot import base
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_common as common
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100 as sm100
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100_quant as sm100_quant
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm90 as sm90
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm90_quant as sm90_quant
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm90_quant_ws as sm90_quant_ws
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm90_quant_ws_async_store as sm90_quant_ws_async_store
from typing_extensions import override

Config = common.Config
QArray = base.QArray
GroupSizes = base.GroupSizes


# TODO: Natively support mk,ekn->mn.
@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicGpuRaggedDot(base.RaggedDot[Config, None]):
  """Pallas-Mosaic-GPU ragged dot implementation.

  The kernel is optimized for physical layout `mk,enk->mn`.
  """

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  def __post_init__(self):
    if self.vjp is None:
      # TODO: Use kernel for vjp.
      object.__setattr__(self, "vjp", base.vjp)

  @override
  def _fwd(
      self,
      lhs: jax.Array | QArray,
      rhs: jax.Array | QArray,
      *,
      group_sizes: jax.Array | GroupSizes,
      ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
      precision: base.CanonicalPrecision,
      preferred_element_type: jnp.dtype | None,
      return_residuals: bool,
      config: Config,
  ) -> tuple[jax.Array, None]:
    del return_residuals  # Unused.

    if not mosaic_gpu_lib.has_mosaic_gpu_support():
      raise NotImplementedError("Mosaic not supported on this platform.")

    if ragged_dot_dimension_numbers != base.DEFAULT_RAGGED_DOT_DIM_NUMS:
      raise NotImplementedError(
          "Only default `ragged_dot_dimension_numbers` supported."
      )

    if not precision_lib.is_default(lhs.dtype, rhs.dtype, precision):
      raise NotImplementedError(f"{precision=} not supported.")

    lhs = quantization.as_array(lhs)
    # None of the kernels support zero point yet.
    if isinstance(rhs, QArray) and rhs.zero_point is not None:
      rhs = qwix.dequantize(rhs)

    device = backend.get_default_device()

    if float(getattr(device, "compute_capability", "9.0")) >= 10.0:
      if isinstance(rhs, QArray):
        fn = sm100_quant.ragged_dot_gpu_quant_blackwell_kernel
      else:
        fn = sm100.ragged_dot_gpu_non_quant_blackwell_kernel
    elif isinstance(rhs, QArray):
      if config.async_store:
        fn = sm90_quant_ws_async_store.ragged_dot_quantized_ws_async_store_kernel  # pylint: disable=line-too-long
      elif config.warp_specialized:
        fn = sm90_quant_ws.ragged_dot_quantized_ws_kernel
      else:
        fn = sm90_quant.ragged_dot_quantized_kernel
    else:
      fn = sm90.ragged_dot_non_quantized_kernel

    if isinstance(group_sizes, GroupSizes):
      group_sizes = jnp.array(group_sizes)

    if preferred_element_type is None:
      preferred_element_type = jnp.promote_types(lhs.dtype, rhs.dtype)

    return fn(lhs, rhs, group_sizes, preferred_element_type, config), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    _, rhs = ba.args

    device = backend.get_default_device()
    if float(getattr(device, "compute_capability", "9.0")) >= 10.0:
      if isinstance(rhs, QArray):
        return Config(
            block_m=128,
            block_n=128,
            block_k=256,
            num_stages=2,
            split_k=1,
            grid_block_n=1,
        )
      else:
        return Config(
            block_m=64,
            block_n=128,
            block_k=256,
            num_stages=2,
            split_k=1,
            grid_block_n=1,
            warp_specialized=True,
            persistent=False,
            collective=True,
            grid_minor_dim=common.MatmulDimension.M,
            grid_tile_width=4,
        )
    return Config(
        block_m=64,
        block_n=64,
        block_k=rhs.scale_tile_shape[1] if isinstance(rhs, QArray) else 128,
        num_stages=4,
        split_k=1,
        grid_block_n=1,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    device = backend.get_default_device()
    if float(getattr(device, "compute_capability", "9.0")) >= 10.0:
      return self._get_sm100_autotuning_configs(ba)
    return self._get_sm90_autotuning_configs(ba)

  def _get_sm90_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    # Adjusted block_k for float16/bfloat16
    lhs, rhs = ba.args[:2]
    lhs_dtype_bits = jnp.finfo(lhs.dtype).bits
    if isinstance(rhs, QArray):
      rhs_dtype_bits = jnp.iinfo(rhs.values.dtype).bits
    else:
      rhs_dtype_bits = jnp.finfo(rhs.dtype).bits
    out_dtype = ba.kwargs["preferred_element_type"]
    if out_dtype is None:
      out_dtype = jnp.promote_types(lhs.dtype, rhs.dtype)
    out_dtype_bits = jnp.finfo(out_dtype).bits
    out_swizzle_elems = (128 * 8) // out_dtype_bits

    warp_specialized = [True, False] if isinstance(rhs, QArray) else [True]

    configs = set()
    # For prefill
    for persistent in [True, False]:
      for async_store in [True, False]:
        for ws in warp_specialized:
          for block_k in [128, 256]:
            if (block_k * rhs_dtype_bits) % (128 * 8) or (
                block_k * lhs_dtype_bits
            ) % (128 * 8):
              continue
            for block_m in [128, 64]:
              for num_stages in [4, 2, 1]:
                for grid_minor_dim in [
                    common.MatmulDimension.M,
                    common.MatmulDimension.N,
                ]:
                  for grid_tile_width in [1, 2, 4, 8]:
                    configs.add(
                        Config(
                            block_m=block_m,
                            block_n=out_swizzle_elems,
                            block_k=block_k,
                            num_stages=num_stages,
                            split_k=1,
                            async_store=async_store,
                            warp_specialized=ws,
                            persistent=persistent,
                            grid_block_n=grid_tile_width,
                            grid_minor_dim=grid_minor_dim,
                            grid_tile_width=grid_tile_width,
                        )
                    )
    # For generate
    for persistent in [True, False]:
      for async_store in [True, False]:
        for ws in warp_specialized:
          for block_k in [128, 256]:
            if (block_k * rhs_dtype_bits) % (128 * 8) or (
                block_k * lhs_dtype_bits
            ) % (128 * 8):
              continue
            for block_m in [64, 32, 24, 16]:
              for num_stages in [4]:
                for grid_minor_dim in [
                    common.MatmulDimension.M,
                    common.MatmulDimension.N,
                ]:
                  for grid_tile_width in [1, 2, 4, 8]:
                    configs.add(
                        Config(
                            block_m=block_m,
                            block_n=out_swizzle_elems,
                            block_k=block_k,
                            num_stages=num_stages,
                            split_k=1,
                            async_store=async_store,
                            warp_specialized=ws,
                            persistent=persistent,
                            grid_block_n=grid_tile_width,
                            grid_minor_dim=grid_minor_dim,
                            grid_tile_width=grid_tile_width,
                        )
                    )
    return configs

  def _get_sm100_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    del ba  # Unused.
    configs = set()
    # Configs for prefill
    for block_k in [128, 256]:
      for num_stages in [2, 3]:
        configs.add(
            Config(
                block_m=128,
                block_n=128,
                block_k=block_k,
                num_stages=num_stages,
                split_k=1,
                grid_block_n=1,
                warp_specialized=True,
                persistent=False,
                collective=True,
            )
        )

    # Config for generate
    for block_m in [8, 16, 32]:
      for num_stages in [2, 3]:
        for grid_block_n in [1, 4, 8]:
          for persistent in [False, True]:
            configs.add(
                Config(
                    block_m=block_m,
                    block_n=128,
                    block_k=256,
                    num_stages=num_stages,
                    split_k=1,
                    grid_block_n=grid_block_n,
                    warp_specialized=True,
                    persistent=persistent,
                    collective=False,
                )
            )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return mosaic_gpu_lib.has_mosaic_gpu_support(device)
