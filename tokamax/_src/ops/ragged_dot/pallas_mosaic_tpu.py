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
import functools
import itertools
import types
from typing import ClassVar

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import pydantic
import qwix
from tokamax._src import precision as precision_lib
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as backend
from typing_extensions import override


# Tiling on TPU technically needs to be a multiple of 128, but it's possible to
# request a "full tile" of an array equal to the full axis size and that doesn't
# require a multiple of 128. In the same case, tiles < 128 are allowed.
TilingTuple = tuple[
    pydantic.PositiveInt,  # tile_m
    pydantic.PositiveInt,  # tile_k
    pydantic.PositiveInt,  # tile_n
]
InputBufferCount = pydantic.PositiveInt

QArray = base.QArray
AsQArray = base.AsQArray

def _is_fp8(dtype) -> bool:
  """Returns True if dtype is an FP8 type (1-byte floating point)."""
  dtype = jnp.dtype(dtype)
  return jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize == 1


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class FP8Residuals:
  """Pre-quantized FP8 inputs saved during forward for backward use.

  Attributes:
    lhs_for_drhs: lhs columnwise-quantized as FP8, used for drhs (tgmm).
    rhs_for_dlhs: rhs columnwise-quantized as FP8, used for dlhs.
    activation_residuals: activation residuals (pre-activation output) for
      use in the VJP when activation is not None.
  """
  lhs_for_drhs: QArray | None
  rhs_for_dlhs: QArray | None
  activation_residuals: jax.Array | None


Residuals = FP8Residuals | types.NoneType


def _group_sizes_to_indices(gs: jax.Array, *, m: int) -> jax.Array:
  gsc = jnp.concat([jnp.zeros((1,), gs.dtype), jnp.cumsum(gs)])
  s, e = gsc[:-1], gsc[1:]
  iota, inc = jnp.arange(m), jnp.arange(gs.size)
  mask = (iota[None, :] >= s[:, None]) & (iota[None, :] < e[:, None])
  return jnp.sum(inc[:, None] * mask, axis=0)


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Pallas Mosaic TPU Ragged Dot config."""

  tile_m: pydantic.PositiveInt = 128
  tile_k: pydantic.PositiveInt = 128
  tile_n: pydantic.PositiveInt = 128
  input_buffer_count: InputBufferCount = 2
  combine_scopes: bool = False


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


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
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
          qdtype=qdtype,
          interpret=self.interpret,
      )(
          *args, **kw
      )
      object.__setattr__(
          self,
          "vjp",
          functools.partial(base.vjp, dlhs_ragged_dot=fn, drhs_ragged_dot=fn),
      )

  @override
  def _fwd(
      self,
      lhs: jax.Array | QArray | AsQArray,
      rhs: jax.Array | QArray | AsQArray,
      *,
      group_sizes: jax.Array | base.GroupSizes,
      ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
      precision: base.CanonicalPrecision,
      preferred_element_type: jax.typing.DTypeLike | None,
      return_residuals: bool = False,
      config: Config,
      activation: base.ActivationFunction | None = None,
  ) -> tuple[jax.Array, base.Residuals]:
    # TODO: Support more ragged_dot_dimension_numbers
    # configurations.

    lhs, rhs = map(quantization.as_array_or_qarray, (lhs, rhs))
    fp8_residuals = None

    if any(  # pallas mosaic TPU requires all non-expert dimensions to be >= 128
        size < 128 for size in tuple(lhs.shape[-2:]) + tuple(rhs.shape[-2:])
    ):
      raise NotImplementedError(
          f"RaggedDot inputs must be >= 128, but {lhs.shape=}, {rhs.shape=}"
      )

    def maybe_quantize(x, tile_shape):
      if isinstance(x, QArray) or self.qdtype is None:
        return x
      tiled_axes = {i: d for i, d in enumerate(tile_shape)}
      return qwix.quantize(x, self.qdtype, tiled_axes=tiled_axes)

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

      # Pre-quantize columnwise for backward if FP8 block-wise
      if return_residuals and isinstance(lhs, QArray) and _is_fp8(lhs.qvalue.dtype):
        lhs_raw = quantization.as_array(lhs)
        rhs_raw = quantization.as_array(rhs)
        # Infer block size from lhs scale shape
        lhs_scale_k = lhs.scale.shape[1]
        block_size = lhs.qvalue.shape[1] // max(1, lhs_scale_k)
        fp8_dtype = lhs.qvalue.dtype
        # For drhs (tgmm): lhs needs block-wise quantization along M axis
        # (axis 0 is the reduction axis in tgmm: lhs.T @ dout)
        lhs_for_drhs = qwix.quantize(
            lhs_raw, fp8_dtype,
            tiled_axes={0: block_size, 1: lhs_raw.shape[1]},
        )
        # For dlhs: rhs needs quantization along a different axis
        rhs_for_dlhs = qwix.quantize(
            rhs_raw, fp8_dtype,
            tiled_axes={0: 1, 1: 1, 2: block_size},
        )
        fp8_residuals = FP8Residuals(
            lhs_for_drhs=lhs_for_drhs,
            rhs_for_dlhs=rhs_for_dlhs,
            activation_residuals=None,  # filled later
        )

      out = backend.gmm(
          lhs,
          rhs,
          group_sizes=group_sizes,
          precision=precision,
          out_dtype=preferred_element_type,
          tiling=(config.tile_m, config.tile_k, config.tile_n),
          interpret=self.interpret,  # pytype: disable=attribute-error
          input_buffer_count=config.input_buffer_count,
          activation=activation if not return_residuals else None,
      )
    elif ragged_dot_dimension_numbers == DLHS_RAGGED_DOT_DIM_NUMS:  # dlhs
      # here, handle fast-path special cases that arise in backwards gmm
      if isinstance(lhs, jax.Array) and isinstance(rhs, QArray):
        if _is_fp8(rhs.qvalue.dtype):
          # FP8 QArray from forward residuals — use directly
          lhs = maybe_quantize(lhs, (1, lhs.shape[1]))
        elif rhs.scale.shape[1] == 1:
          # STRATEGY 1: full-channel quantization along the reduction dimension
          # here, apply rhs scales to lhs and compute with rhs quant values
          indices = _group_sizes_to_indices(group_sizes, m=lhs.shape[0])
          lhs *= jnp.take_along_axis(rhs.scale[:, 0, :], indices[:, None], 0)
          rhs = rhs.qvalue
          lhs = maybe_quantize(lhs, (1, lhs.shape[1]))
        else:
          rhs = maybe_quantize(qwix.dequantize(rhs), (1, 1, rhs.shape[2]))
      else:
        lhs = maybe_quantize(lhs, (1, lhs.shape[1]))
        rhs = maybe_quantize(rhs, (1, 1, rhs.shape[2]))
      out = backend.gmm(
          lhs,
          rhs,
          group_sizes=group_sizes,
          precision=precision,
          out_dtype=preferred_element_type,
          tiling=(config.tile_m, config.tile_k, config.tile_n),
          transpose_rhs=True,
          interpret=self.interpret,  # pytype: disable=attribute-error
          input_buffer_count=config.input_buffer_count,
          activation=activation if not return_residuals else None,
      )
    elif ragged_dot_dimension_numbers == DRHS_RAGGED_DOT_DIM_NUMS:  # drhs
      lhs_trans = jax.tree.map(lambda x: x.mT, lhs)
      # here, handle fast-path special cases that arise in backwards gmm
      if isinstance(lhs_trans, QArray) and isinstance(rhs, jax.Array):
        if _is_fp8(lhs_trans.qvalue.dtype):
          # FP8 QArray from forward residuals — use directly
          rhs = maybe_quantize(rhs, (rhs.shape[0], 1))
        elif lhs_trans.scale.shape[0] == 1:
          # STRATEGY 1: full-channel quantization along the reduction dimension
          # here, apply lhs scales to rhs and compute with lhs quant values
          # lhs_trans = quant[k, m], scale[1, m] and rhs/dout = float[m, n]
          rhs *= lhs_trans.scale.mT
          lhs_trans = lhs_trans.qvalue
        else:
          lhs_trans = qwix.dequantize(lhs_trans)
          lhs_trans = maybe_quantize(lhs_trans, (1, lhs_trans.shape[1]))
      else:
        lhs_trans = maybe_quantize(lhs_trans, (1, lhs_trans.shape[1]))
        rhs = maybe_quantize(rhs, (rhs.shape[0], 1))

      out = backend.tgmm(
          lhs_trans,
          rhs,
          group_sizes=group_sizes,
          precision=precision,
          out_dtype=preferred_element_type,
          tiling=(config.tile_m, config.tile_k, config.tile_n),
          interpret=self.interpret,  # pytype: disable=attribute-error
          input_buffer_count=config.input_buffer_count,
          activation=activation if not return_residuals else None,
          combine_scopes=config.combine_scopes,
      )
    else:
      raise NotImplementedError(
          UNSUPPORTED_DIMENSIONS_MSG.format(ragged_dot_dimension_numbers)
      )

    if fp8_residuals is not None:
      if activation is not None and return_residuals:
        fp8_residuals = dataclasses.replace(fp8_residuals, activation_residuals=out)
        out = activation(out)
      return out, fp8_residuals
    else:
      residuals = out
      if activation is not None and return_residuals:
        out = activation(out)
      return out, residuals if return_residuals else None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config()

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    lhs, rhs = ba.args

    dims = ba.arguments.get(
        "ragged_dot_dimension_numbers", DEFAULT_RAGGED_DOT_DIM_NUMS
    )
    if dims == DEFAULT_RAGGED_DOT_DIM_NUMS:
      (m, k), (_, _, n) = lhs.shape, rhs.shape
    elif dims == DLHS_RAGGED_DOT_DIM_NUMS:
      (m, n), (_, k, _) = lhs.shape, rhs.shape
    elif dims == DRHS_RAGGED_DOT_DIM_NUMS:
      (m, k), (_, n) = lhs.shape, rhs.shape
    else:
      raise NotImplementedError(UNSUPPORTED_DIMENSIONS_MSG.format(dims))

    k_ = ((k + 128 - 1) // 128) * 128  # round up to nearest multiple of 128
    n_ = ((n + 128 - 1) // 128) * 128

    # Based on some empirical TPU tiling performance. Create a reasonable
    # tiling search space. We limit the search space max to 1024 to ensure
    # reasonable compilation time. From our experiments, we found that there
    # is no need on current TPU generations to search for larger tiles for m.
    tile_m_range = [
        64 * (2**i)
        for i in range(8)
        if 64 * (2**i) <= m and 64 * (2**i) <= 1024
    ]

    tile_k_range = set(
        [
            128 * (2**i) for i in range(8) if 128 * (2**i) <= k_
        ]  # upwards powers of 2
        + [k_ // (2**i) for i in range(6)]  # downwards divisors of k_
        + [k]  # full tile
    )

    tile_n_range = set(
        [
            128 * (2**i) for i in range(8) if 128 * (2**i) <= n_
        ]  # upwards powers of 2
        + [n_ // (2**i) for i in range(6)]  # downwards divisors of n_
        + [n]  # full tile
    )
    input_buffer_count_range = [2, 3, 4]
    return set(
        Config(
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            input_buffer_count=input_buffer_count,
        )
        for tile_m, tile_k, tile_n, input_buffer_count in itertools.product(
            tile_m_range, tile_k_range, tile_n_range, input_buffer_count_range
        )
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
