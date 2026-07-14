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
"""Ragged dot Pallas-Mosaic-TPU implementation."""

import dataclasses
import functools
import itertools
from typing import Any, ClassVar
import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
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
Residuals = None


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
      manual_axis_type: jax.sharding.ManualAxisType | None = None,
      group_offset: jax.Array | None = None,
      rhs_scale: jax.Array | None = None,
      rhs_bias: jax.Array | None = None,
      maybe_quantize_lhs: bool = False,
      zero_initialize: bool = True,
      fuse_gateup_activation: str | None = None,
      lhs_quantization_dtype: jax.typing.DTypeLike | None = None,
      rhs_quantization_dtype: jax.typing.DTypeLike | None = None,
  ) -> tuple[jax.Array, base.Residuals]:
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
          "The Pallas-Mosaic-TPU-v1 implementation does not support"
          " group_offset, rhs_scale, rhs_bias, maybe_quantize_lhs,"
          " zero_initialize, fuse_gateup_activation, lhs_quantization_dtype,"
          " or rhs_quantization_dtype."
      )

    # TODO: Support more ragged_dot_dimension_numbers
    # configurations.

    lhs, rhs = map(quantization.as_array_or_qarray, (lhs, rhs))

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
      out = backend.gmm(
          lhs,  # [m, k]
          rhs,  # [g, k, n]
          group_sizes=group_sizes,
          precision=precision,
          out_dtype=preferred_element_type,
          tiling=(config.tile_m, config.tile_k, config.tile_n),
          interpret=self.interpret,  # pytype: disable=attribute-error
          input_buffer_count=config.input_buffer_count,
          activation=activation if not return_residuals else None,
          manual_axis_type=manual_axis_type,
      )
    elif ragged_dot_dimension_numbers == DLHS_RAGGED_DOT_DIM_NUMS:  # dlhs
      # here, handle fast-path special cases that arise in backwards gmm
      if isinstance(lhs, jax.Array) and isinstance(rhs, QArray):
        if rhs.scale.shape[1] == 1:
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
          lhs,  # [m, n]
          rhs,  # [g, k, n]
          group_sizes=group_sizes,
          precision=precision,
          out_dtype=preferred_element_type,
          tiling=(config.tile_m, config.tile_k, config.tile_n),
          transpose_rhs=True,
          interpret=self.interpret,  # pytype: disable=attribute-error
          input_buffer_count=config.input_buffer_count,
          activation=activation if not return_residuals else None,
          manual_axis_type=manual_axis_type,
      )
    elif ragged_dot_dimension_numbers == DRHS_RAGGED_DOT_DIM_NUMS:  # drhs
      lhs_trans = jax.tree.map(lambda x: x.mT, lhs)  # [m, k] -> [k, m]
      # here, handle fast-path special cases that arise in backwards gmm
      if isinstance(lhs_trans, QArray) and isinstance(rhs, jax.Array):
        if lhs_trans.scale.shape[0] == 1:
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
          lhs_trans,  # [k, m]
          rhs,  # [m, n]
          group_sizes=group_sizes,
          precision=precision,
          out_dtype=preferred_element_type,
          tiling=(config.tile_m, config.tile_k, config.tile_n),
          interpret=self.interpret,  # pytype: disable=attribute-error
          input_buffer_count=config.input_buffer_count,
          activation=activation if not return_residuals else None,
          combine_scopes=config.combine_scopes,
          manual_axis_type=manual_axis_type,
      )
    else:
      raise NotImplementedError(
          UNSUPPORTED_DIMENSIONS_MSG.format(ragged_dot_dimension_numbers)
      )

    residuals = out
    if activation is not None and return_residuals:
      out = activation(out)

    return out, residuals if return_residuals else None

  def _fit_within_tpu_vmem(
      self,
      input_tiles: list[tuple[int, int, Any]],
      output_tile: tuple[int, int, Any],
      input_buffer_count: int,
      utilize_ratio: float = 0.75,
  ) -> bool:
    """Returns whether the given tiling fits within TPU VMEM."""
    total_size = 0
    for tile in input_tiles:
      tile_size = np.prod(tile[:-1])
      dtype = tile[-1]
      # For Quantized types, we assume that the upper bound on the tile size is
      # bfloat16. That is, the QArray tile size (qvalue + scale) should be less
      # than or equal to the bfloat16 tile size.
      num_bytes = max(jnp.dtype(dtype).itemsize, 2)
      tile_size *= num_bytes
      total_size += tile_size * input_buffer_count

    # Now do the same for the output tile.
    output_tile_size = np.prod(output_tile[:-1])
    dtype = output_tile[-1]
    # For Quantized types, we assume that the upper bound on the tile size is
    # bfloat16. That is, the QArray tile size (qvalue + scale) should be less
    # than or equal to the bfloat16 tile size.
    num_bytes = max(jnp.dtype(dtype).itemsize, 2)
    output_tile_size *= num_bytes
    # To account for accumulation buffer and prefetch, otherwise we see OOM
    # errors.
    total_size += 3 * output_tile_size

    # Get the TPU scoped VMEM size. Default to half of the VMEM capacity.
    tpu_info = pltpu.get_tpu_info()
    scoped_vmem_capacity = tpu_info.vmem_capacity_bytes / 2
    if tpu_info.generation == 6:
      scoped_vmem_capacity /= 2
    elif tpu_info.generation == 5:
      scoped_vmem_capacity /= 4
    return total_size <= scoped_vmem_capacity * utilize_ratio

  def _deflate_tile(self, current_tile: int) -> int:
    if current_tile <= 128:
      return 128
    new_tile = current_tile // 2
    # Ensure partial tiles are always multiples of 128
    return max((new_tile // 128) * 128, 128)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    if self.qdtype is not None:
      return Config()
    # Currently the heuristics are only implemented for TPU7x.
    if pltpu.get_tpu_info().generation < 7:
      return Config()
    lhs, rhs = ba.args

    dims = ba.arguments.get(
        "ragged_dot_dimension_numbers", DEFAULT_RAGGED_DOT_DIM_NUMS
    )
    input_buffer_count = 2

    # Extract the dimensions of the input arrays. Here m, k, n, are the
    # dimensions of the forward gmm. This is different from the tgmm case which
    # we will handle in _tgmm_heuristics_config.
    if dims == DEFAULT_RAGGED_DOT_DIM_NUMS:
      (m, k), (_, _, n) = lhs.shape, rhs.shape
    elif dims == DLHS_RAGGED_DOT_DIM_NUMS:
      (m, n), (_, k, _) = lhs.shape, rhs.shape
    elif dims == DRHS_RAGGED_DOT_DIM_NUMS:
      (m, k), (_, n) = lhs.shape, rhs.shape
      return self._tgmm_heuristics_config(ba, m, n, k, input_buffer_count)
    else:
      raise ValueError(UNSUPPORTED_DIMENSIONS_MSG.format(dims))

    # Main heuristic: start with the largest possible tile size and "deflate" it
    # until it fits within VMEM.

    # Default m to 1024.
    tile_m = min(m, 1024)
    tile_n = n
    tile_k = k
    # Deflate m, min size for m is 128.
    while (
        not self._fit_within_tpu_vmem(
            [(tile_m, tile_k, lhs.dtype), (tile_k, tile_n, rhs.dtype)],
            (tile_m, tile_n, lhs.dtype),
            input_buffer_count,
        )
        and tile_m > 128
    ):
      tile_m = self._deflate_tile(tile_m)

    # Deflate n, min size for n is 128.
    while (
        not self._fit_within_tpu_vmem(
            [(tile_m, tile_k, lhs.dtype), (tile_k, tile_n, rhs.dtype)],
            (tile_m, tile_n, lhs.dtype),
            input_buffer_count,
        )
        and tile_n > 128
    ):
      tile_n = self._deflate_tile(tile_n)

    # Last priority: deflate k (reduction dimension), min size for k is 128.
    while (
        not self._fit_within_tpu_vmem(
            [(tile_m, tile_k, lhs.dtype), (tile_k, tile_n, rhs.dtype)],
            (tile_m, tile_n, lhs.dtype),
            input_buffer_count,
        )
        and tile_k > 128
    ):
      tile_k = self._deflate_tile(tile_k)

    return Config(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        input_buffer_count=input_buffer_count,
    )

  def _tgmm_heuristics_config(
      self, ba: op.BoundArguments, m, n, k, input_buffer_count
  ) -> Config:
    """Heuristics config for tgmm."""
    lhs, rhs = ba.args
    # tgmm is lhs [m, k] @ rhs [m, n] so it is similar to gmm but transposed.
    # the input buffer count is still 2.
    # Reduction dimension for TGMM is m.
    tile_k = k
    tile_n = n
    tile_m = min(m, 1024)  # Cap to 1024 because ragged dimension.

    # Deflate k, min size for k is 128
    while (
        not self._fit_within_tpu_vmem(
            [(tile_m, tile_k, lhs.dtype), (tile_m, tile_n, rhs.dtype)],
            (tile_k, tile_n, lhs.dtype),
            input_buffer_count,
        )
        and tile_k > 128
    ):
      tile_k = self._deflate_tile(tile_k)

    # Deflate n, min size for n is 128
    while (
        not self._fit_within_tpu_vmem(
            [(tile_m, tile_k, lhs.dtype), (tile_m, tile_n, rhs.dtype)],
            (tile_k, tile_n, lhs.dtype),
            input_buffer_count,
        )
        and tile_n > 128
    ):
      tile_n = self._deflate_tile(tile_n)

    # Last priority: deflate m (reduction dimension), min size for m is 128
    while (
        not self._fit_within_tpu_vmem(
            [(tile_m, tile_k, lhs.dtype), (tile_m, tile_n, rhs.dtype)],
            (tile_k, tile_n, lhs.dtype),
            input_buffer_count,
        )
        and tile_m > 128
    ):
      tile_m = self._deflate_tile(tile_m)

    return Config(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        input_buffer_count=input_buffer_count,
        combine_scopes=False,
    )

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
        128 * (2**i)
        for i in range(8)
        if 128 * (2**i) <= m and 128 * (2**i) <= 1024
    ]

    tile_k_range = set(
        [
            128 * (2**i) for i in range(8) if 128 * (2**i) <= k_
        ]  # upwards powers of 2
        + [
            k_ // (2**i) for i in range(6) if k_ // (2**i) >= 128
        ]  # downwards divisors of k_
        + [k]  # full tile
    )

    tile_n_range = set(
        [
            128 * (2**i) for i in range(8) if 128 * (2**i) <= n_
        ]  # upwards powers of 2
        + [
            n_ // (2**i) for i in range(6) if n_ // (2**i) >= 128
        ]  # downwards divisors of n_
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
