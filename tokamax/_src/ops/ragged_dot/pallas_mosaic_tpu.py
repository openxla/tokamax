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
"""Pallas Mosaic TPU Megablox."""

import dataclasses
from functools import partial  # pylint: disable=g-importing-member
import itertools
import logging
import types
from typing import ClassVar

import jax
import jax.numpy as jnp
import pydantic
from tokamax._src import mosaic_tpu
from tokamax._src import precision as precision_lib
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as backend
from typing_extensions import override


TilingTuple = tuple[
    pydantic.conint(ge=128, multiple_of=128),  # tile_m
    pydantic.conint(ge=128, multiple_of=128),  # tile_k
    pydantic.conint(ge=128, multiple_of=128),  # tile_n
]
InputBufferCount = pydantic.conint(ge=1, le=3, multiple_of=1)

QuantizedArray = quantization.QuantizedArray
Residuals = types.NoneType

LUTKey = tuple[
    pydantic.PositiveInt,  # m
    pydantic.PositiveInt,  # k
    pydantic.PositiveInt,  # n
    pydantic.PositiveInt,  # g
    bool,  # is_quantized
]
LUTValue = tuple[TilingTuple, InputBufferCount]


def _group_sizes_to_indices(gs: jax.Array, *, m: int) -> jax.Array:
  gsc = jnp.concat([jnp.zeros((1,), gs.dtype), jnp.cumsum(gs)])
  s, e = gsc[:-1], gsc[1:]
  iota, inc = jnp.arange(m), jnp.arange(gs.size)
  mask = ((iota[None, :] >= s[:, None]) & (iota[None, :] < e[:, None]))
  return jnp.sum(inc[:, None] * mask, axis=0)


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Pallas Mosaic TPU Ragged Dot config."""

  gmm_tiling: TilingTuple = (128, 128, 128)
  gmm_rhs_transpose_tiling: TilingTuple | None = None
  tgmm_tiling: TilingTuple | None = None
  input_buffer_count: InputBufferCount = 2


# A temporary lookup table for optimized configs.
# TODO: formally add autotuning to the vjp.
GMM_TILING_TUNED_LUT: dict[LUTKey, LUTValue] = {
    (262144, 7168, 2048, 256, False): ((256, 7168, 512), 2),
    (262144, 7168, 2048, 256, True): ((128, 7168, 2048), 2),
    (262144, 2048, 7168, 256, False): ((128, 2048, 3584), 2),
    (262144, 2048, 7168, 256, True): ((256, 2048, 3584), 3),
}
GMM_RHS_TRANSPOSE_TILING_TUNED_LUT: dict[LUTKey, LUTValue] = {
    (262144, 7168, 2048, 256, False): ((256, 2048, 1792), 2),
    (262144, 7168, 2048, 256, True): ((256, 2048, 3584), 2),
    (262144, 2048, 7168, 256, False): ((256, 7168, 512), 2),
    (262144, 2048, 7168, 256, True): ((256, 7168, 1024), 2),
}
TGMM_TILING_TUNED_LUT: dict[LUTKey, LUTValue] = {
    (262144, 7168, 2048, 256, False): ((512, 1024, 2048), 3),
    (262144, 7168, 2048, 256, True): ((512, 1024, 2048), 2),
    (262144, 2048, 7168, 256, False): ((256, 2048, 1024), 3),
    (262144, 2048, 7168, 256, True): ((512, 512, 3584), 2),
}

# Ragged dot dimension numbers supported by the megablox kernel.
DEFAULT_RAGGED_DOT_DIM_NUMS = base.DEFAULT_RAGGED_DOT_DIM_NUMS

DLHS_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [2]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)

DRHS_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[],
)

UNSUPPORTED_DIMENSIONS_MSG = (
    "Specified ragged_dot_dimension_numbers `{}` not supported. Supported"
    f" dimensions include: {DEFAULT_RAGGED_DOT_DIM_NUMS},"
    f" {DLHS_RAGGED_DOT_DIM_NUMS}, {DRHS_RAGGED_DOT_DIM_NUMS}"
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuRaggedDot(base.RaggedDot[Config, None]):
  """Pallas-Mosaic-TPU ragged dot implementation.

  TPU Implementation of the Megablocks Paper https://arxiv.org/abs/2211.15841.
  """

  config_cls: ClassVar[type[Config]] = Config
  qdtype: jax.typing.DTypeLike | None = None
  interpret: bool = False

  def __post_init__(self):
    qdtype: str | None = (
        self.qdtype if self.qdtype is None else jnp.dtype(self.qdtype).name
    )
    if self.vjp is None:
      # Avoid infinite recursion.
      fn = lambda *args, **kw: PallasMosaicTpuRaggedDot(  # pylint: disable=unnecessary-lambda
          config=self.config,
          qdtype=qdtype,
          interpret=self.interpret,
      )(*args, **kw)
      object.__setattr__(
          self, "vjp", partial(base.vjp, dlhs_ragged_dot=fn, drhs_ragged_dot=fn)
      )

  @override
  def _fwd(
      self,
      lhs: jax.Array | QuantizedArray,
      rhs: jax.Array | QuantizedArray,
      *,
      group_sizes: jax.Array | base.GroupSizes,
      ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers | None,
      precision: base.CanonicalPrecision,
      preferred_element_type: jax.typing.DTypeLike | None,
      return_residuals: bool = False,
      config: Config,
  ) -> tuple[jax.Array, None]:
    del return_residuals  # Unused.

    if not precision_lib.is_default(lhs.dtype, rhs.dtype, precision):
      logging.warning("Requested precision `%s`, but mosaic TPU ragged dot does"
                      " not currently support this precision. Proceeding with"
                      " default precision.", str(precision))

    # TODO: Support more ragged_dot_dimension_numbers
    # configurations.

    if any(  # pallas mosaic TPU requires all non-expert dimensions to be >= 128
        size < 128 for size in tuple(lhs.shape[-2:]) + tuple(rhs.shape[-2:])
    ):
      raise NotImplementedError(
          f"RaggedDot inputs must be >= 128, but {lhs.shape=}, {rhs.shape=}"
      )

    def maybe_quantize(x, tile_shape):
      if isinstance(x, QuantizedArray) or self.qdtype is None:
        return x
      return quantization.quantize_as(self.qdtype, tile_shape=tile_shape)(x)

    if isinstance(group_sizes, base.GroupSizes):
      group_sizes = jnp.array(group_sizes)

    if preferred_element_type is None:
      preferred_element_type = (
          precision_lib.default_output_dtype_from_input_dtypes(
              lhs.dtype, rhs.dtype
          )
      )

    if ragged_dot_dimension_numbers == DEFAULT_RAGGED_DOT_DIM_NUMS:  # gmm fwd
      # STRATEGY 1: full-channel quantization along the reduction dimension
      lhs = maybe_quantize(lhs, (1, lhs.shape[1]))
      rhs = maybe_quantize(rhs, (1, rhs.shape[1], 1))
      out = backend.gmm(
          lhs,
          rhs,
          group_sizes=group_sizes,
          out_dtype=preferred_element_type,
          tiling=config.gmm_tiling,
          interpret=self.interpret,  # pytype: disable=attribute-error
          input_buffer_count=config.input_buffer_count,
      )
    elif ragged_dot_dimension_numbers == DLHS_RAGGED_DOT_DIM_NUMS:  # dlhs
      # here, handle fast-path special cases that arise in backwards gmm
      if isinstance(lhs, jax.Array) and isinstance(rhs, QuantizedArray):
        if rhs.scales.shape[1] == 1:
          # STRATEGY 1: full-channel quantization along the reduction dimension
          # here, apply rhs scales to lhs and compute with rhs quant values
          indices = _group_sizes_to_indices(group_sizes, m=lhs.shape[0])
          lhs *= jnp.take_along_axis(rhs.scales[:, 0, :], indices[:, None], 0)
          rhs = rhs.values
          lhs = maybe_quantize(lhs, (1, lhs.shape[1]))
        else:
          rhs = maybe_quantize(rhs.recompose(), (1, 1, rhs.shape[2]))
      else:
        lhs = maybe_quantize(lhs, (1, lhs.shape[1]))
        rhs = maybe_quantize(rhs, (1, 1, rhs.shape[2]))
      out = backend.gmm(
          lhs,
          rhs,
          group_sizes=group_sizes,
          out_dtype=preferred_element_type,
          tiling=config.gmm_rhs_transpose_tiling or config.gmm_tiling,
          transpose_rhs=True,
          interpret=self.interpret,  # pytype: disable=attribute-error
          input_buffer_count=config.input_buffer_count,
      )
    elif ragged_dot_dimension_numbers == DRHS_RAGGED_DOT_DIM_NUMS:  # drhs
      lhs_trans = jax.tree.map(lambda x: x.mT, lhs)
      # here, handle fast-path special cases that arise in backwards gmm
      if isinstance(lhs_trans, QuantizedArray) and isinstance(rhs, jax.Array):
        if lhs_trans.scales.shape[0] == 1:
          # STRATEGY 1: full-channel quantization along the reduction dimension
          # here, apply lhs scales to rhs and compute with lhs quant values
          # lhs_trans = quant[k, m], scale[1, m] and rhs/dout = float[m, n]
          rhs *= lhs_trans.scales.mT
          lhs_trans = lhs_trans.values
        else:
          lhs_trans = lhs_trans.recompose()
          lhs_trans = maybe_quantize(lhs_trans, (1, lhs_trans.shape[1]))
      else:
        lhs_trans = maybe_quantize(lhs_trans, (1, lhs_trans.shape[1]))
        rhs = maybe_quantize(rhs, (rhs.shape[0], 1))

      out = backend.tgmm(
          lhs_trans,
          rhs,
          group_sizes=group_sizes,
          out_dtype=preferred_element_type,
          tiling=config.tgmm_tiling or config.gmm_tiling,
          interpret=self.interpret,  # pytype: disable=attribute-error
          input_buffer_count=config.input_buffer_count,
      )
    else:
      raise NotImplementedError(
          UNSUPPORTED_DIMENSIONS_MSG.format(ragged_dot_dimension_numbers)
      )
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    lhs, rhs = ba.arguments["lhs"], ba.arguments["rhs"]

    # this is generally an incorrect assumption, but ok for a heuristic
    is_quantized = (isinstance(lhs, QuantizedArray)
                    or isinstance(rhs, QuantizedArray))

    ragged_dot_dimension_numbers = ba.arguments.get(
        "ragged_dot_dimension_numbers", DEFAULT_RAGGED_DOT_DIM_NUMS
    )
    default_config = Config()
    if ragged_dot_dimension_numbers == DEFAULT_RAGGED_DOT_DIM_NUMS:
      (m, k), (g, _, n) = lhs.shape, rhs.shape
      lut_key = (m, k, n, g, is_quantized)
      if lut_key in GMM_TILING_TUNED_LUT:
        gmm_tiling, input_buffer_count = GMM_TILING_TUNED_LUT[lut_key]
        return Config(gmm_tiling=gmm_tiling,
                      input_buffer_count=input_buffer_count)
      return default_config
    elif ragged_dot_dimension_numbers == DLHS_RAGGED_DOT_DIM_NUMS:
      grad = lhs
      (m, n), (g, k, _) = grad.shape, rhs.shape  # lhs is out
      lut_key = (m, k, n, g, is_quantized)
      if lut_key in GMM_RHS_TRANSPOSE_TILING_TUNED_LUT:
        gmm_rhs_transpose_tiling, input_buffer_count = (
            GMM_RHS_TRANSPOSE_TILING_TUNED_LUT[lut_key]
        )
        return Config(
            gmm_rhs_transpose_tiling=gmm_rhs_transpose_tiling,
            input_buffer_count=input_buffer_count,
        )
      return default_config
    elif ragged_dot_dimension_numbers == DRHS_RAGGED_DOT_DIM_NUMS:
      group_sizes = ba.arguments["group_sizes"]
      grad = rhs
      if isinstance(group_sizes, base.GroupSizes):
        group_sizes = jnp.array(group_sizes)
      (m, k), (_, n), g = lhs.shape, grad.shape, group_sizes.shape[0]
      lut_key = (m, k, n, g, is_quantized)
      if lut_key in TGMM_TILING_TUNED_LUT:
        tgmm_tiling, input_buffer_count = TGMM_TILING_TUNED_LUT[lut_key]
        return Config(tgmm_tiling=tgmm_tiling,
                      input_buffer_count=input_buffer_count)
      return default_config
    else:
      raise NotImplementedError(
          UNSUPPORTED_DIMENSIONS_MSG.format(ragged_dot_dimension_numbers)
      )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    lhs, rhs = ba.arguments["lhs"], ba.arguments["rhs"]

    (m, k), (g, _, n) = lhs.shape, rhs.shape
    tile_range = [128]  # TODO: Add more configs.
    # Based on some empirical TPU tiling performance. Create a reasonable
    # tiling search space.
    tile_m_range = range(128, 1024, 128)
    tile_k_range = range(128, k, 128)
    tile_n_range = range(128, int(n / 2), 128)
    return set(
        Config(gmm_tiling=(tile_m, tile_k, tile_n))
        for tile_m, tile_k, tile_n in itertools.product(
            tile_m_range, tile_k_range, tile_n_range
        )
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and mosaic_tpu.tpu_generation() >= 5
