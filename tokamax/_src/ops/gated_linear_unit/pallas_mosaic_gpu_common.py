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
"""Common definitions for Pallas-Mosaic-GPU Gated Linear Unit."""

import enum
import math
from typing import Annotated, Any, Final

import jax
import jax.numpy as jnp
import pydantic
from tokamax._src import precision as precision_lib

DotAlgorithmPreset = jax.lax.DotAlgorithmPreset

SUPPORTED_DTYPES: Final[frozenset[jnp.dtype]] = frozenset({
    jnp.dtype(jnp.float16),
    jnp.dtype(jnp.bfloat16),
    jnp.dtype(jnp.float32),
})


def unbatched_weights_x_shape(x: Any, weights: Any) -> tuple[Any, ...]:
  """Returns effective `x` shape after collapsing unbatched-weights vmap axes."""
  x_vmap_axes = getattr(x, "vmap_axes", ())
  if not x_vmap_axes:
    return x.shape
  w_leaves = jax.tree.leaves(weights)
  shape = list(x.shape)
  for i in range(len(x_vmap_axes) - 1, -1, -1):
    x_axis = x_vmap_axes[i]
    w_batched = any(
        i < len(w_axes := getattr(w, "vmap_axes", ()))
        and w_axes[i] is not None
        for w in w_leaves
    )
    if x_axis is not None and not w_batched:
      shape.insert(*x_axis)
  return tuple(shape)


def get_mnk(ba: Any) -> tuple[int, int, int]:
  """Extracts collapsed `(m, n, k)` dimensions from `BoundArguments`."""
  x = ba.arguments["x"]
  weights = ba.arguments["weights"]
  x_shape = unbatched_weights_x_shape(x, weights)
  w_shapes = tuple(
      getattr(w, "vmap_shape", w.shape) for w in jax.tree.leaves(weights)
  )
  for s in (x_shape, *w_shapes):
    if not all(isinstance(d, int) for d in s):
      raise NotImplementedError(
          "Mosaic GPU GLU requires static integer dimensions, got"
          f" {x_shape=} and {w_shapes=}."
      )
  m = math.prod(x_shape[:-1])
  k = x_shape[-1]
  n = weights[0].shape[-1] if isinstance(weights, tuple) else weights.shape[-1]
  return m, n, k


def resolve_dot_precision(
    x_dtype: jax.typing.DTypeLike,
    weights_dtype: jax.typing.DTypeLike,
    precision: jax.lax.PrecisionLike,
) -> tuple[DotAlgorithmPreset, jnp.dtype, int]:
  """Resolves `(preset, in_dtype, k_mult)` for Mosaic GPU Tensor Core MMA."""
  lhs_dtype = jnp.dtype(x_dtype)
  rhs_dtype = jnp.dtype(weights_dtype)
  if lhs_dtype != rhs_dtype:
    raise NotImplementedError(
        "Mosaic GPU GLU requires matching input dtypes, got"
        f" {lhs_dtype} vs {rhs_dtype}."
    )
  if lhs_dtype not in SUPPORTED_DTYPES:
    raise NotImplementedError(
        f"Mosaic GPU GLU only supports {SUPPORTED_DTYPES}, got {lhs_dtype}."
    )
  preset = precision_lib.to_dot_algorithm_preset(
      lhs_dtype, rhs_dtype, precision
  )
  if preset not in precision_lib.SUPPORTED_PRECISIONS:
    raise NotImplementedError(f"{precision=} ({preset=}) is not supported.")

  match preset:
    case DotAlgorithmPreset.F16_F16_F32:
      return preset, jnp.dtype(jnp.float16), 1
    case DotAlgorithmPreset.BF16_BF16_F32:
      return preset, jnp.dtype(jnp.bfloat16), 1
    case (
        DotAlgorithmPreset.BF16_BF16_F32_X3
        | DotAlgorithmPreset.TF32_TF32_F32
    ):
      return preset, jnp.dtype(jnp.bfloat16), 3
    case DotAlgorithmPreset.BF16_BF16_F32_X6:
      return preset, jnp.dtype(jnp.bfloat16), 6
    case (
        DotAlgorithmPreset.BF16_BF16_F32_X9
        | DotAlgorithmPreset.TF32_TF32_F32_X3
        | DotAlgorithmPreset.F32_F32_F32
    ):
      return preset, jnp.dtype(jnp.bfloat16), 9
    case _:
      raise NotImplementedError(f"{preset=} is not supported.")


def get_problem_spec(
    ba: Any,
) -> tuple[int, int, int, jnp.dtype, jnp.dtype, bool]:
  """Extracts `(m, n, k_eff, in_dtype, store_dtype, return_residuals)`."""
  m, n, k = get_mnk(ba)
  x = ba.arguments["x"]
  weights = ba.arguments["weights"]
  x_dtype = jnp.dtype(x.dtype)
  if isinstance(weights, tuple):
    if jnp.dtype(weights[0].dtype) != jnp.dtype(weights[1].dtype):
      raise NotImplementedError(
          "Mosaic GPU GLU requires matching gate and projection weight dtypes,"
          f" got {weights[0].dtype} vs {weights[1].dtype}."
      )
    w_dtype = jnp.dtype(weights[0].dtype)
  else:
    w_dtype = jnp.dtype(weights.dtype)
  precision = ba.arguments.get("precision")
  return_residuals = bool(ba.arguments.get("return_residuals", False))
  _, in_dtype, k_mult = resolve_dot_precision(x_dtype, w_dtype, precision)
  return m, n, k * k_mult, in_dtype, x_dtype, return_residuals


def _truncate_f32_to_bf16_bits(x_f32: jax.Array) -> jax.Array:
  u32 = jax.lax.bitcast_convert_type(x_f32, jnp.uint32)
  return jax.lax.bitcast_convert_type(u32 & jnp.uint32(0xFFFF0000), jnp.float32)


def _split_2x_bf16(x: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Splits `x` into `(hi, lo)` `bfloat16` components (XLA `Split2xToBF16`)."""
  x_f32 = x.astype(jnp.float32)
  hi_f32 = _truncate_f32_to_bf16_bits(x_f32)
  lo_f32 = jnp.where(jnp.isfinite(x_f32), x_f32 - hi_f32, jnp.float32(0.0))
  return hi_f32.astype(jnp.bfloat16), lo_f32.astype(jnp.bfloat16)


def _split_3x_bf16(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Splits `x` into `(hi, mid, lo)` `bfloat16` parts (XLA `Split3xToBF16`)."""
  x_f32 = x.astype(jnp.float32)
  hi_f32 = _truncate_f32_to_bf16_bits(x_f32)
  mid_rem = jnp.where(jnp.isfinite(x_f32), x_f32 - hi_f32, jnp.float32(0.0))
  mid_f32 = _truncate_f32_to_bf16_bits(mid_rem)
  lo_f32 = mid_rem - mid_f32
  return (
      hi_f32.astype(jnp.bfloat16),
      mid_f32.astype(jnp.bfloat16),
      lo_f32.astype(jnp.bfloat16),
  )


def _map_rhs_weights(weights, fn):
  if isinstance(weights, tuple):
    return (fn(weights[0], -2), fn(weights[1], -2))
  return fn(weights, -3)


def prepare_dot_inputs(
    x: jax.Array,
    weights: jax.Array | tuple[jax.Array, jax.Array],
    preset: DotAlgorithmPreset,
) -> tuple[jax.Array, jax.Array | tuple[jax.Array, jax.Array]]:
  """Prepares 16-bit Tensor Core `(x, weights)` operands for `preset`.

  Preserves both fused `(..., K, 2, N)` and unfused `((..., K, N), (..., K, N))`
  weight representations without introducing XLA slices or joins on single-pass
  16-bit presets. For multi-pass `float32` / `BF16_X{3,6,9}` presets, splits
  operands into `bfloat16` components (matching XLA's
  `dot_algorithm_rewriter.cc`) and concatenates partial products along `K` in
  ascending magnitude order.
  """
  match preset:
    case DotAlgorithmPreset.F16_F16_F32:
      if jnp.dtype(x.dtype) == jnp.float16:
        return x, weights
      return (
          x.astype(jnp.float16),
          jax.tree.map(lambda w: w.astype(jnp.float16), weights),
      )
    case DotAlgorithmPreset.BF16_BF16_F32:
      if jnp.dtype(x.dtype) == jnp.bfloat16:
        return x, weights
      return (
          x.astype(jnp.bfloat16),
          jax.tree.map(lambda w: w.astype(jnp.bfloat16), weights),
      )
    case (
        DotAlgorithmPreset.BF16_BF16_F32_X3
        | DotAlgorithmPreset.TF32_TF32_F32
    ):
      xh, xl = _split_2x_bf16(x)

      def _rhs_2x(w: jax.Array, k_axis: int) -> jax.Array:
        wh, wl = _split_2x_bf16(w)
        return jnp.concatenate([wh, wl, wh], axis=k_axis)

      return (
          jnp.concatenate([xl, xh, xh], axis=-1),
          _map_rhs_weights(weights, _rhs_2x),
      )
    case DotAlgorithmPreset.BF16_BF16_F32_X6:
      xh, xm, xl = _split_3x_bf16(x)

      def _rhs_3x_6(w: jax.Array, k_axis: int) -> jax.Array:
        wh, wm, wl = _split_3x_bf16(w)
        return jnp.concatenate([wm, wl, wh, wm, wh, wh], axis=k_axis)

      return (
          jnp.concatenate([xm, xh, xl, xh, xm, xh], axis=-1),
          _map_rhs_weights(weights, _rhs_3x_6),
      )
    case (
        DotAlgorithmPreset.BF16_BF16_F32_X9
        | DotAlgorithmPreset.TF32_TF32_F32_X3
        | DotAlgorithmPreset.F32_F32_F32
    ):
      xh, xm, xl = _split_3x_bf16(x)

      def _rhs_3x_9(w: jax.Array, k_axis: int) -> jax.Array:
        wh, wm, wl = _split_3x_bf16(w)
        return jnp.concatenate(
            [wl, wm, wl, wm, wl, wh, wm, wh, wh], axis=k_axis
        )

      return (
          jnp.concatenate([xl, xl, xm, xm, xh, xl, xh, xm, xh], axis=-1),
          _map_rhs_weights(weights, _rhs_3x_9),
      )
    case _:
      raise NotImplementedError(f"{preset=} is not supported.")


class MatmulDimension(enum.IntEnum):
  M = 0
  N = 1

  def __str__(self):
    return self.name

  def __repr__(self):
    return f"<MatmulDimension.{self.name}>"


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Configuration for the Pallas-Mosaic-GPU Gated Linear Unit.

  Attributes:
    tile_m: M dimension of the tile.
    tile_n: N dimension of the tile.
    tile_k: K dimension of the tile.
    num_stages: Maximum number of concurrent steps in the pipeline.
    epi_tile_n: Epilogue tile size in the N dimension.
    epi_tile_m: Epilogue tile size in the M dimension.
    grid_minor_dim: The minor dimension for the grid tiling.
    grid_tile_width: The width of the grid tile.
    wg_dimension: The dimension along which workgroups are distributed.
    cluster_size_m: The size of the cluster in the M dimension.
    cluster_size_n: The size of the cluster in the N dimension.
  """

  tile_m: Annotated[int, pydantic.Field(ge=64, multiple_of=64)] = 128
  tile_n: Annotated[int, pydantic.Field(ge=64, multiple_of=64)] = 128
  tile_k: pydantic.PositiveInt = 128
  num_stages: pydantic.PositiveInt = 4
  epi_tile_n: pydantic.PositiveInt | None = 64
  epi_tile_m: pydantic.PositiveInt | None = 64
  grid_minor_dim: MatmulDimension = MatmulDimension.N
  grid_tile_width: pydantic.PositiveInt = 1
  wg_dimension: MatmulDimension = MatmulDimension.N
  cluster_size_m: Annotated[int, pydantic.Field(ge=1, le=2)] = 1
  cluster_size_n: Annotated[int, pydantic.Field(ge=1, le=2)] = 1

  def __post_init__(self):
    if self.cluster_size_m > 1 and self.cluster_size_n > 1:
      raise ValueError(
          "At least one of cluster_size_m or cluster_size_n must be 1."
      )
    if self.cluster_size_m not in (1, 2) or self.cluster_size_n not in (1, 2):
      raise ValueError("Cluster sizes must be 1 or 2.")
    if self.epi_tile_n is not None and self.tile_n % self.epi_tile_n != 0:
      raise ValueError(
          f"{self.tile_n=} must be divisible by {self.epi_tile_n=}"
      )
    if self.epi_tile_m is not None and self.tile_m % self.epi_tile_m != 0:
      raise ValueError(
          f"{self.tile_m=} must be divisible by {self.epi_tile_m=}"
      )
