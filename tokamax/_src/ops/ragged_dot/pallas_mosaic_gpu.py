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
from tokamax._src import mosaic_gpu as mosaic_gpu_lib
from tokamax._src import precision as precision_lib
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.ragged_dot import base
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_common as common
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_non_quant_kernel as non_quant_kernel
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_quant_kernel as quant_kernel
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_quant_kernel_blackwell as quant_kernel_blackwell
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_quant_ws_kernel as quant_ws_kernel

Config = common.Config
QuantizedArray = quantization.QuantizedArray
GroupSizes = base.GroupSizes


# TODO: Natively support mk,ekn->mn.
@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicGpuRaggedDot(base.RaggedDot[common.Config, None]):
  """Pallas-Mosaic-GPU ragged dot implementation.

  The kernel is optimized for physical layout `mk,enk->mn`.
  """

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  def __post_init__(self):
    if self.vjp is None:
      # TODO: Use kernel for vjp.
      object.__setattr__(self, "vjp", base.vjp)

  def _fwd(
      self,
      lhs: jax.Array | QuantizedArray,
      rhs: jax.Array | QuantizedArray,
      *,
      group_sizes: jax.Array | GroupSizes,
      ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
      precision: base.CanonicalPrecision,
      preferred_element_type: jnp.dtype | None,
      return_residuals: bool,
      config: common.Config,
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

    if isinstance(lhs, QuantizedArray):
      lhs = lhs.recompose()

    if isinstance(rhs, QuantizedArray):
      if "b200" in backend.get_default_device().device_kind.lower():
        fn = quant_kernel_blackwell.ragged_dot_gpu_quant_blackwell_kernel
      elif config.warp_specialized:
        fn = quant_ws_kernel.ragged_dot_quantized_ws_kernel
      else:
        fn = quant_kernel.ragged_dot_quantized_kernel
    else:
      fn = non_quant_kernel.ragged_dot_non_quantized_kernel

    if isinstance(group_sizes, GroupSizes):
      group_sizes = jnp.array(group_sizes)

    if preferred_element_type is None:
      preferred_element_type = jnp.result_type(lhs.dtype, rhs.dtype)

    out = fn(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        config,
    )
    return out, None

  def _get_heuristics_config(self, ba: op.BoundArguments) -> common.Config:
    _, rhs = ba.args
    device_kind = backend.get_default_device().device_kind.lower()
    if "b200" in device_kind:
      return common.Config(
          block_m=128,
          block_n=128,
          block_k=256,
          num_stages=2,
          split_k=1,
          grid_block_n=1,
      )
    return common.Config(
        block_m=64,
        block_n=64,
        block_k=rhs.tile_shape[1] if isinstance(rhs, QuantizedArray) else 128,
        num_stages=4,
        split_k=1,
        grid_block_n=1,
    )

  def _get_autotuning_configs(
      self, ba: op.BoundArguments
  ) -> set[common.Config]:
    configs = set()
    # Adjusted block_k for float16/bfloat16
    lhs, rhs = ba.args[:2]
    warp_specialized = [True]
    if isinstance(rhs, quantization.QuantizedArray):
      warp_specialized = [True, False]

    out_dtype = ba.kwargs.get("preferred_element_type") or jnp.promote_types(
        lhs.dtype, rhs.dtype
    )
    out_dtype_bits = jnp.finfo(out_dtype).bits
    out_swizzle_elems = (128 * 8) // out_dtype_bits
    lhs_dtype_bits = jnp.finfo(lhs.dtype).bits
    if isinstance(rhs, quantization.QuantizedArray):
      rhs_dtype_bits = jnp.iinfo(rhs.values.dtype).bits
    else:
      rhs_dtype_bits = jnp.finfo(rhs.dtype).bits

    device_kind = backend.get_default_device().device_kind.lower()
    if "h100" in device_kind:
      for persistent in [True, False]:
        for ws in warp_specialized:
          for block_k in [128, 256]:
            if (block_k * rhs_dtype_bits) % (128 * 8) or (
                block_k * lhs_dtype_bits
            ) % (128 * 8):
              continue
            for block_m in [128, 64]:
              for num_stages in [4, 2, 1]:
                for grid_block_n in [1, 2, 4, 8]:
                  configs.add(
                      common.Config(
                          block_m=block_m,
                          block_n=out_swizzle_elems,
                          block_k=block_k,
                          num_stages=num_stages,
                          split_k=1,
                          grid_block_n=grid_block_n,
                          warp_specialized=ws,
                          persistent=persistent,
                      )
                  )
    elif "b200" in device_kind:
      # Configs for prefill
      block_m = 128
      block_n = 128
      for block_k in [128, 256]:
        for num_stages in [2, 3]:
          configs.add(
              common.Config(
                  block_m=block_m,
                  block_n=block_n,
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
                  common.Config(
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
    else:
      raise ValueError(
          f"Autotuning not supported for device kind: {device_kind}"
      )
    return configs
