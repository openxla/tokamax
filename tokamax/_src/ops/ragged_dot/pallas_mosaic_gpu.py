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
from functools import partial  # pylint: disable=g-importing-member
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src import precision as precision_lib
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.ragged_dot import base
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_common as common
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100 as sm100
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100_fp8_quant as sm100_fp8_quant
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100_i8_quant as sm100_i8_quant
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100_quant as sm100_quant
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100_quant_post_scale as sm100_quant_post_scale
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm90 as sm90
import tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm90_quant as sm90_quant
from typing_extensions import override

Config = common.Config
QArray = base.QArray
AsQArray = base.AsQArray
GroupSizes = base.GroupSizes


# Read quant metadata from a QArray, a lazy AsQArray, or a model-side
# PreQuantizedInputFusionGPU uniformly. Heuristics run on abstract values, so an
# AsQArray cannot be materialized to a QArray here.
def _is_quantized_input(x) -> bool:
  return (
      isinstance(x, (QArray, AsQArray))
      or x.__class__.__name__ == "PreQuantizedInputFusionGPU"
  )


def _input_qtype(x):
  if isinstance(x, (QArray, AsQArray)):
    return x.qtype
  if x.__class__.__name__ == "PreQuantizedInputFusionGPU":
    return x.buffer.dtype
  if isinstance(x, jax.Array):
    return x.dtype
  return None


def _input_quant_bits(x) -> int | None:
  qtype = _input_qtype(x)
  if qtype is None:
    return None
  try:
    return jnp.iinfo(qtype).bits
  except ValueError:
    return jnp.finfo(qtype).bits


def _input_subchannel(x) -> int | None:
  if isinstance(x, QArray):
    return x.scale_tile_shape[1]  # contraction-axis (index 1) tile
  if isinstance(x, AsQArray):
    return x.tiled_axes.get(1) if x.tiled_axes else None
  if x.__class__.__name__ == "PreQuantizedInputFusionGPU":
    return x.subchannel_size
  return None


# TODO: Directly import ManualAxisType JAX is upgraded.
try:
  from jax.sharding import ManualAxisType
except ImportError:
  ManualAxisType = Any

# TODO: Natively support mk,ekn->mn.
@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicGpuRaggedDot(base.RaggedDot[Config, None]):
  """Pallas-Mosaic-GPU ragged dot implementation.

  The kernel is optimized for physical layout `mk,enk->mn`.
  """

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  enable_fused_epilogue_quant: bool = False

  def __post_init__(self):
    if self.vjp is None:
      # Avoid infinite recursion.
      fn = lambda *a, **kw: PallasMosaicGpuRaggedDot()(*a, **kw)  # pylint: disable=unnecessary-lambda
      vjp = partial(base.vjp, dlhs_ragged_dot=fn, drhs_ragged_dot=fn)
      object.__setattr__(self, "vjp", vjp)

  @override
  def _fwd(
      self,
      lhs: jax.Array | QArray | AsQArray,
      rhs: jax.Array | QArray | AsQArray,
      *,
      group_sizes: jax.Array | GroupSizes,
      ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
      precision: base.CanonicalPrecision,
      preferred_element_type: jnp.dtype | None,
      return_residuals: bool,
      config: Config,
      activation: base.ActivationFunction | None = None,
      manual_axis_type: ManualAxisType | None = None,
      group_offset: jax.Array | None = None,
      rhs_scale: jax.Array | None = None,
      rhs_bias: jax.Array | None = None,
      maybe_quantize_lhs: bool = False,
      zero_initialize: bool = True,
      fuse_gateup_activation: str | None = None,
      lhs_quantization_dtype: jax.typing.DTypeLike | None = None,
      rhs_quantization_dtype: jax.typing.DTypeLike | None = None,
  ) -> tuple[jax.Array, base.Residuals]:
    # TODO: Support returning residuals from mosaic GPU kernel.

    if (
        group_offset is not None
        or rhs_scale is not None
        or rhs_bias is not None
        or maybe_quantize_lhs
        or not zero_initialize
        or fuse_gateup_activation is not None
        or lhs_quantization_dtype is not None
        or rhs_quantization_dtype is not None
    ):
      raise NotImplementedError(
          "The Pallas-Mosaic-GPU implementation does not support group_offset,"
          " rhs_scale, rhs_bias, maybe_quantize_lhs, zero_initialize,"
          " fuse_gateup_activation, lhs_quantization_dtype,"
          " or rhs_quantization_dtype."
      )

    if ragged_dot_dimension_numbers == base.TRANS_RHS_RAGGED_DOT_DIM_NUMS:
      rhs = rhs.mT  # TODO: Fuse transpose into kernel.
      ragged_dot_dimension_numbers = base.DEFAULT_RAGGED_DOT_DIM_NUMS

    # None of the kernels support zero point yet.
    lhs = quantization.as_array_or_qarray_without_zero_point(lhs)
    rhs = quantization.as_array_or_qarray_without_zero_point(rhs)

    fn = None

    if gpu_utils.is_sm90():
      # sm90 doesn't support lhs to be QArray.
      if isinstance(lhs, QArray):
        lhs = quantization.as_array(lhs)

      if ragged_dot_dimension_numbers == base.DEFAULT_RAGGED_DOT_DIM_NUMS:
        if isinstance(rhs, QArray):
          if not precision_lib.is_default(lhs.dtype, rhs.dtype, precision):
            raise NotImplementedError(f"{precision=} not supported.")

          fn = sm90_quant.ragged_dot_quantized_kernel
        else:
          if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
            lhs = lhs.astype(jnp.bfloat16)
            rhs = rhs.astype(jnp.bfloat16)
          elif not precision_lib.is_default(lhs.dtype, rhs.dtype, precision):
            raise NotImplementedError(f"{precision=} not supported.")

          fn = sm90.ragged_dot_kernel
      elif ragged_dot_dimension_numbers == base.RAGGED_CONTRACTING_DOT_DIM_NUMS:
        rhs = quantization.as_array(rhs)

        if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
          lhs = lhs.astype(jnp.bfloat16)
          rhs = rhs.astype(jnp.bfloat16)
        elif not precision_lib.is_default(lhs.dtype, rhs.dtype, precision):
          raise NotImplementedError(f"{precision=} not supported.")

        fn = sm90.ragged_contracting_dim_dot_kernel
      else:
        raise NotImplementedError("Unsupported ragged dot dimension numbers.")
    elif gpu_utils.is_sm100():
      if not precision_lib.is_default(lhs.dtype, rhs.dtype, precision):
        raise NotImplementedError(f"{precision=} not supported.")

      if ragged_dot_dimension_numbers != base.DEFAULT_RAGGED_DOT_DIM_NUMS:
        raise NotImplementedError(
            "Only default `ragged_dot_dimension_numbers` supported."
        )
      if isinstance(rhs, QArray):
        if isinstance(lhs, QArray):
          if lhs.qtype == jnp.int8:
            fn = sm100_i8_quant.ragged_dot_gpu_i8_quant_blackwell_kernel
          elif lhs.qtype == jnp.float8_e4m3fn:
            fn = sm100_fp8_quant.ragged_dot_gpu_fp8_quant_blackwell_kernel
            # make sure output is bfloat16 since we may want to store lhs.scale
            # as float32 to avoid in-kernel conversion.
            if (
                preferred_element_type is None
                or preferred_element_type == jnp.float8_e4m3fn
            ):
              preferred_element_type = rhs.dtype
          else:
            # dequantize lhs to fallback to non-lhs-quantized kernel
            lhs = quantization.as_array(lhs)
        if isinstance(lhs, jax.Array):
          if config.post_scale:
            fn = sm100_quant_post_scale.ragged_dot_gpu_quant_post_scale_blackwell_kernel  # pylint: disable=line-too-long
          else:
            fn = sm100_quant.ragged_dot_gpu_quant_blackwell_kernel
      else:
        fn = sm100.ragged_dot_gpu_non_quant_blackwell_kernel
    else:
      raise NotImplementedError("Unsupported GPU architecture.")

    if isinstance(group_sizes, GroupSizes):
      group_sizes = jnp.array(group_sizes)

    if preferred_element_type is None:
      preferred_element_type = jnp.promote_types(lhs.dtype, rhs.dtype)

    if fn is None:
      raise NotImplementedError(
          f"Unsupported config: {config=} {ragged_dot_dimension_numbers=}"
          f" {lhs.dtype=} {rhs.dtype=} {preferred_element_type=} {precision=}"
      )

    dot_out = fn(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        config,
        activation=activation if not return_residuals else None,
    )
    residuals = dot_out
    if activation is not None and return_residuals:
      dot_out = activation(dot_out)

    return dot_out, residuals if return_residuals else None

  # Because fp8 ragged dot kernel has special optimization to:
  # 1. offload subchannel rowsum (used in debiasing) to preceding quant kernel;
  # 2. pack rowsum to scale (to not break tokamax + qwix API);
  # 3. pre-convert scale to f32 (since fp8 ragged dot is generally ALU bound).
  # it will break autotuning cache lookup key because the special quant kernel
  # is not exposed to Tokamax, which solely rely on qwix to perform quantization
  # and dequantization. For example, autotuning will set the lookup key with
  # "scale:bf16(4096,8)" but when fp8 ragged dot kernel is used with special
  # quant kernel, the lookup key expects to be "scale:f32(4096,16)". This hack
  # fixes the cache lookup.
  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Any:
    lhs = ba.arguments.get("lhs")
    if isinstance(lhs, QArray) and lhs.qtype == jnp.float8_e4m3fn:
      if lhs.scale.dtype == jnp.bfloat16:
        new_scale = jax.ShapeDtypeStruct(
            lhs.scale.shape[:-1] + (lhs.scale.shape[-1] * 2,), jnp.float32
        )
        new_lhs = QArray(
            qvalue=lhs.qvalue,
            scale=new_scale,
            zero_point=lhs.zero_point,
            qtype=lhs.qtype,
        )
        new_arguments = dict(ba.arguments)
        new_arguments["lhs"] = new_lhs
        ba = dataclasses.replace(ba, arguments=new_arguments)
    return base.RaggedDot._get_autotuning_cache_key(self, ba)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    lhs, rhs = ba.args
    _is_fp8 = lambda x: _input_qtype(x) == jnp.float8_e4m3fn
    _is_int8 = lambda x: _input_qtype(x) == jnp.int8

    # avoid OOMs by having too large block_k
    rhs_subchannel = _input_subchannel(rhs)
    block_k = min(rhs_subchannel, 256) if rhs_subchannel is not None else 128
    if gpu_utils.is_sm100():
      # Clamp to the activation subchannel so a finer lhs (e.g. fp8 output of a
      # previous dot) keeps `tile_xk % block_k == 0`.
      lhs_subchannel = _input_subchannel(lhs)
      if lhs_subchannel is not None:
        block_k = min(block_k, lhs_subchannel)

    if gpu_utils.is_sm90():
      return Config(
          block_m=64,
          block_n=64,
          block_k=block_k,
          num_stages=2,
          split_k=1,
          persistent=_is_quantized_input(rhs),
          grid_minor_dim=common.MatmulDimension.M,
          grid_tile_width=1,
      )
    elif gpu_utils.is_sm100():
      if _is_quantized_input(rhs):
        if _is_int8(lhs):
          return Config(
              block_m=16,
              block_n=128,
              block_k=block_k,
              num_stages=2,
              split_k=1,
          )
        elif _is_fp8(lhs):
          preferred_element_type = ba.arguments.get("preferred_element_type")
          epilogue_quant_qtype = None
          epilogue_quant_subchannel_size = None
          if self.enable_fused_epilogue_quant and preferred_element_type == jnp.float8_e4m3fn:
            epilogue_quant_qtype = jnp.float8_e4m3fn
            epilogue_quant_subchannel_size = 128
          return Config(
              block_m=16,
              block_n=128,
              block_k=block_k,
              collective=False,
              num_stages=2,
              split_k=1,
              epilogue_quant_qtype=epilogue_quant_qtype,
              epilogue_quant_subchannel_size=epilogue_quant_subchannel_size,
          )
        return Config(
            block_m=64,
            block_n=128,
            block_k=block_k,
            num_stages=2,
            split_k=1,
        )
      else:
        return Config(
            block_m=64,
            block_n=128,
            block_k=256,
            num_stages=2,
            split_k=1,
            persistent=False,
            collective=True,
            grid_minor_dim=common.MatmulDimension.M,
            grid_tile_width=4,
        )
    else:
      raise NotImplementedError("Unsupported GPU architecture.")

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    if gpu_utils.is_sm100():
      return self._get_sm100_autotuning_configs(ba)
    return self._get_sm90_autotuning_configs(ba)

  def _get_sm90_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    # Adjusted block_k for float16/bfloat16
    lhs, rhs = ba.args[:2]
    lhs_dtype_bits = jnp.finfo(lhs.dtype).bits
    if _is_quantized_input(rhs):
      rhs_dtype_bits = _input_quant_bits(rhs)
      scale_tile_shape = _input_subchannel(rhs) or 0
    else:
      rhs_dtype_bits = jnp.finfo(rhs.dtype).bits
      scale_tile_shape = 0
    out_dtype = ba.kwargs["preferred_element_type"]
    if out_dtype is None:
      out_dtype = jnp.promote_types(lhs.dtype, rhs.dtype)
    out_dtype_bits = jnp.finfo(out_dtype).bits
    out_swizzle_elems = (128 * 8) // out_dtype_bits

    configs = set()
    for persistent in [True, False]:
      for block_k in [128, 256, 512]:
        if (block_k * rhs_dtype_bits) % (128 * 8) or (
            block_k * lhs_dtype_bits
        ) % (128 * 8):
          continue
        if scale_tile_shape != 0 and scale_tile_shape % block_k != 0:
          continue
        for block_m in [128, 64, 32, 16]:
          for num_stages in [4, 2]:
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
                        persistent=persistent,
                        split_k=1,
                        grid_minor_dim=grid_minor_dim,
                        grid_tile_width=grid_tile_width,
                    )
                )
    return configs

  def _get_sm100_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    lhs, rhs = ba.args[:2]
    lhs_dtype_bits = jnp.finfo(lhs.dtype).bits
    if _is_quantized_input(rhs):
      rhs_dtype_bits = _input_quant_bits(rhs)
      scale_tile_shape = _input_subchannel(rhs) or 0
    else:
      rhs_dtype_bits = jnp.finfo(rhs.dtype).bits
      scale_tile_shape = 0

    block_k_choices = []
    for block_k in [128, 256, 512]:
      if (block_k * rhs_dtype_bits) % (128 * 8) or (
          block_k * lhs_dtype_bits
      ) % (128 * 8):
        continue
      if scale_tile_shape != 0 and scale_tile_shape % block_k != 0:
        continue
      block_k_choices.append(block_k)
    grid_minor_dim_choices = [
        common.MatmulDimension.M,
        common.MatmulDimension.N,
    ]
    grid_tile_width_choices = [1, 2, 4, 8]

    configs = set()

    def _generate_configs(
        configs,
        block_m_choices: list[int],
        block_k_choices: list[int],
        num_stages_choices: list[int],
        grid_minor_dim_choices: list[common.MatmulDimension],
        grid_tile_width_choices: list[int],
        split_m_choices: list[int],
        collective_choices: list[bool],
        post_scale_choices: list[bool],
    ):
      epilogue_quant_qtype = None
      epilogue_quant_subchannel_size = None

      preferred_element_type = ba.arguments.get("preferred_element_type")
      if self.enable_fused_epilogue_quant and preferred_element_type == jnp.float8_e4m3fn:
        epilogue_quant_qtype = jnp.float8_e4m3fn
        epilogue_quant_subchannel_size = 128

      for block_m in block_m_choices:
        for block_k in block_k_choices:
          for num_stages in num_stages_choices:
            for grid_minor_dim in grid_minor_dim_choices:
              for grid_tile_width in grid_tile_width_choices:
                for split_m in split_m_choices:
                  for collective in collective_choices:
                    for post_scale in post_scale_choices:
                      configs.add(
                          Config(
                              block_m=block_m,
                              block_n=128,
                              block_k=block_k,
                              num_stages=num_stages,
                              split_k=1,
                              persistent=False,
                              collective=collective,
                              post_scale=post_scale,
                              grid_minor_dim=grid_minor_dim,
                              grid_tile_width=grid_tile_width,
                              split_m=split_m,
                              epilogue_quant_qtype=epilogue_quant_qtype,
                              epilogue_quant_subchannel_size=epilogue_quant_subchannel_size,
                          )
                      )
      return configs

    if _is_quantized_input(lhs):
      if _input_qtype(lhs) in [jnp.int8, jnp.float8_e4m3fn]:
        configs = _generate_configs(
            configs,
            block_m_choices=[8, 16, 32],
            block_k_choices=[128, 256, 512],
            num_stages_choices=[2],
            grid_minor_dim_choices=grid_minor_dim_choices,
            grid_tile_width_choices=grid_tile_width_choices,
            split_m_choices=[1],
            collective_choices=[False, True],
            post_scale_choices=[False],
        )
    else:
      # Configs for prefill
      configs = _generate_configs(
          configs,
          block_m_choices=[128],
          block_k_choices=block_k_choices,
          num_stages_choices=[2, 4],
          grid_minor_dim_choices=grid_minor_dim_choices,
          grid_tile_width_choices=grid_tile_width_choices,
          split_m_choices=[1],
          collective_choices=[True, False],
          post_scale_choices=[False],
      )
      # Config for generate
      configs = _generate_configs(
          configs,
          block_m_choices=[8, 16, 32, 64],
          block_k_choices=block_k_choices,
          num_stages_choices=[2, 4],
          grid_minor_dim_choices=grid_minor_dim_choices,
          grid_tile_width_choices=grid_tile_width_choices,
          split_m_choices=[1],
          collective_choices=[True, False],
          post_scale_choices=[True, False],
      )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)
