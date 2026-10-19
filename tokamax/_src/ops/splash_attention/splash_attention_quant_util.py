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

"""Quantization utilities for Splash Attention emulated quantization."""

import enum
import importlib
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp


def _get_fast_fake_micro_quantize_no_reshape():
  return None


def fast_fake_micro_quantize_no_reshape(*args, **kwargs):
  """Dispatches to fast_fake_micro_quantize_no_reshape if available."""
  fn = _get_fast_fake_micro_quantize_no_reshape()
  if fn is None:
    raise NotImplementedError(
        "FAKE_MICRO quantization is not available in this environment."
    )
  return fn(*args, **kwargs)


FP8_MAX = 448.0


class QuantizationStrategy(enum.IntEnum):
  UNQUANTIZED = enum.auto()
  CAST = enum.auto()
  PER_TILE = enum.auto()
  PER_AXIS = enum.auto()
  FAKE_MICRO = enum.auto()


def _resolve_dtype(dtype: str | jax.typing.DTypeLike) -> Any:
  """Resolves string or type specification to a concrete jax/numpy dtype."""
  if isinstance(dtype, str):
    if dtype in ("float8_e4m3fn", "fp8_e4m3fn", "e4m3fn"):
      return jnp.float8_e4m3fn
    if dtype in ("float8_e5m2", "fp8_e5m2", "e5m2"):
      return jnp.float8_e5m2
    return jnp.dtype(dtype)
  return jnp.dtype(dtype)


def apply_quantization(
    x: jax.Array,
    strategy: QuantizationStrategy,
    contract_axis: int,
    ablate_scale: bool = False,
    use_ceil_scale: bool = False,
    dtype: str | jax.typing.DTypeLike = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array | None]:
  """Applies the selected quantization strategy to an input tensor."""
  dtype = _resolve_dtype(dtype)
  if strategy == QuantizationStrategy.UNQUANTIZED:
    return x, None

  if strategy == QuantizationStrategy.CAST:
    # NOMUTANTS(AOR): [-448, -447] all round to -448.0 in float8_e4m3fn.
    clipped = jnp.clip(x, -FP8_MAX, FP8_MAX)
    return clipped.astype(jnp.float8_e4m3fn), None

  elif strategy == QuantizationStrategy.PER_TILE:
    amax = jnp.max(jnp.abs(x).astype(jnp.float32))
    amax = jnp.maximum(amax, 1e-12)
    scale = FP8_MAX / amax
    return (x * scale).astype(jnp.float8_e4m3fn), amax / FP8_MAX

  elif strategy == QuantizationStrategy.PER_AXIS:
    amax = jnp.max(jnp.abs(x), axis=contract_axis, keepdims=True)
    amax = jnp.maximum(amax, 1e-12)
    scale = FP8_MAX / amax
    scaled_x = (x * scale).astype(jnp.float8_e4m3fn)
    return scaled_x, (amax / FP8_MAX).astype(x.dtype)

  elif strategy == QuantizationStrategy.FAKE_MICRO:
    return fast_fake_micro_quantize_no_reshape(
        x,
        contract_axis=contract_axis,
        dtype=dtype,
        ablate_scale=ablate_scale,
        use_ceil_scale=use_ceil_scale,
    )

  else:
    raise ValueError(f"Unknown quantization strategy {strategy}")


def _dequantize_out(
    out: jax.Array,
    lhs_scale: jax.Array | None,
    rhs_scale: jax.Array | None,
    dimension_numbers: tuple[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[tuple[int, ...], tuple[int, ...]],
    ],
    lhs_ndim: int,
    rhs_ndim: int,
) -> jax.Array:
  """Dequantizes out after lax.dot_general using lhs and rhs scales."""
  if lhs_scale is None and rhs_scale is None:
    return out

  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = (
      dimension_numbers
  )

  # Fast-path for scalar scales (e.g. PER_TILE).
  if (
      lhs_scale is not None
      and rhs_scale is not None
      and lhs_scale.size == 1
      and rhs_scale.size == 1
  ):
    return out * (lhs_scale * rhs_scale)

  if lhs_scale is not None:
    if lhs_scale.size == 1:
      out = out * lhs_scale
    else:
      rhs_non_c = [
          j
          for j in range(rhs_ndim)
          if j not in rhs_contracting and j not in rhs_batch
      ]
      squeezed = jnp.squeeze(lhs_scale, axis=lhs_contracting)
      target_shape = squeezed.shape + (1,) * len(rhs_non_c)
      out = out * squeezed.reshape(target_shape)

  if rhs_scale is not None:
    if rhs_scale.size == 1:
      out = out * rhs_scale
    else:
      lhs_non_c = [
          i
          for i in range(lhs_ndim)
          if i not in lhs_contracting and i not in lhs_batch
      ]
      squeezed = jnp.squeeze(rhs_scale, axis=rhs_contracting)
      prefix = squeezed.shape[: len(rhs_batch)]
      suffix = squeezed.shape[len(rhs_batch) :]
      target_shape = prefix + (1,) * len(lhs_non_c) + suffix
      out = out * squeezed.reshape(target_shape)

  return out


def dot_general_quantized(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: tuple[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[tuple[int, ...], tuple[int, ...]],
    ],
    lhs_strategy: QuantizationStrategy,
    rhs_strategy: QuantizationStrategy,
    use_fp8: bool,
    preferred_element_type: Any = jnp.float32,
    ablate_scale: bool = False,
    use_ceil_scale: bool = False,
    lhs_dtype: str | jax.typing.DTypeLike = jnp.float8_e4m3fn,
    rhs_dtype: str | jax.typing.DTypeLike = jnp.float8_e4m3fn,
) -> jax.Array:
  """Wrapper around lax.dot_general with scaled quantization."""
  lhs_dtype = _resolve_dtype(lhs_dtype)
  rhs_dtype = _resolve_dtype(rhs_dtype)
  lhs_c_axis = dimension_numbers[0][0][0]
  rhs_c_axis = dimension_numbers[0][1][0]

  lhs_q, lhs_scale = apply_quantization(
      lhs,
      lhs_strategy,
      lhs_c_axis,
      ablate_scale=ablate_scale,
      use_ceil_scale=use_ceil_scale,
      dtype=lhs_dtype,
  )
  rhs_q, rhs_scale = apply_quantization(
      rhs,
      rhs_strategy,
      rhs_c_axis,
      ablate_scale=ablate_scale,
      use_ceil_scale=use_ceil_scale,
      dtype=rhs_dtype,
  )

  if use_fp8:
    if lhs_q.dtype != jnp.float8_e4m3fn or rhs_q.dtype != jnp.float8_e4m3fn:
      raise ValueError(
          "TRUE fp8 matmul requires float8_e4m3fn inputs. Got lhs:"
          f" {lhs_q.dtype}, rhs: {rhs_q.dtype}"
      )
    lhs_compute = lhs_q
    rhs_compute = rhs_q
  else:
    lhs_compute = lhs_q.astype(jnp.bfloat16)
    rhs_compute = rhs_q.astype(jnp.bfloat16)

  out = lax.dot_general(
      lhs_compute,
      rhs_compute,
      dimension_numbers,
      preferred_element_type=preferred_element_type,
  )

  return _dequantize_out(
      out, lhs_scale, rhs_scale, dimension_numbers, lhs.ndim, rhs.ndim
  )
