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
"""Precision classes and utilities."""

import logging
from typing import Final

import jax
import jax.numpy as jnp


DotAlgorithm = jax.lax.DotAlgorithm
DotAlgorithmPreset = jax.lax.DotAlgorithmPreset
Precision = jax.lax.Precision
PrecisionLike = jax.lax.PrecisionLike
DTypeLike = jax.typing.DTypeLike


SUPPORTED_PRECISIONS: Final[tuple[DotAlgorithmPreset, ...]] = (
    DotAlgorithmPreset.F16_F16_F32,
    DotAlgorithmPreset.BF16_BF16_F32,
    DotAlgorithmPreset.BF16_BF16_F32_X3,
    DotAlgorithmPreset.BF16_BF16_F32_X6,
    DotAlgorithmPreset.BF16_BF16_F32_X9,
    DotAlgorithmPreset.F32_F32_F32,
    DotAlgorithmPreset.TF32_TF32_F32,
    DotAlgorithmPreset.TF32_TF32_F32_X3,
)

_F32_DOT_PRECISION_MAP: Final[dict[str, dict[Precision, str]]] = dict(
    tpu={
        Precision.DEFAULT: "BF16_BF16_F32",
        Precision.HIGH: "BF16_BF16_F32_X3",
        Precision.HIGHEST: "BF16_BF16_F32_X6",
    },
    gpu_old={
        Precision.DEFAULT: "F32_F32_F32",
        Precision.HIGH: "F32_F32_F32",
        Precision.HIGHEST: "F32_F32_F32",
    },
    gpu={
        Precision.DEFAULT: "TF32_TF32_F32",
        Precision.HIGH: "TF32_TF32_F32",
        Precision.HIGHEST: "F32_F32_F32",
    },
    cpu={
        Precision.DEFAULT: "F32_F32_F32",
        Precision.HIGH: "F32_F32_F32",
        Precision.HIGHEST: "F32_F32_F32",
    },
)


CanonicalPrecision = (
    tuple[Precision, Precision] | DotAlgorithm | DotAlgorithmPreset
)


def canonicalize_precision(precision: PrecisionLike) -> CanonicalPrecision:
  """Canonicalizes a `PrecisionLike`."""

  if precision is None:
    if (precision := jax.config.jax_default_matmul_precision) is None:
      return Precision.DEFAULT, Precision.DEFAULT
  if isinstance(precision, str):
    if precision in DotAlgorithmPreset.__members__:
      return DotAlgorithmPreset[precision]
    try:
      # jax.lax.Precision supports aliases like 'fastest' for
      # jax.lax.Precision.DEFAULT. Can only tell whether a string is a valid
      # alias by trying the constructor.
      return (Precision(precision),) * 2
    except ValueError:
      raise ValueError(  # pylint: disable=raise-missing-from
          f"Unsupported enum value: {precision}. Must be refer to either"
          " a `jax.lax.DotAlgorithmPreset` or a `jax.lax.Precision` enum."
      )
  if isinstance(precision, Precision):
    return precision, precision
  if isinstance(precision, (DotAlgorithm, DotAlgorithmPreset)):
    return precision
  if isinstance(precision, (list, tuple)) and len(precision) == 2:
    p0, p1 = precision
    if isinstance(p0, Precision) and isinstance(p1, Precision):
      return precision
    if isinstance(p0, str) and isinstance(p1, str):
      return Precision(p0), Precision(p1)
  raise ValueError(f"Invalid precision: {precision}")


def to_dot_algorithm_preset(
    a_dtype: DTypeLike, b_dtype: DTypeLike, precision: PrecisionLike
) -> DotAlgorithmPreset:
  """Converts a `PrecisionLike` to a `DotAlgorithmPreset`."""

  precision = canonicalize_precision(precision)

  if isinstance(precision, DotAlgorithmPreset):
    return precision
  if isinstance(precision, DotAlgorithm):
    raise NotImplementedError("`DotAlgorithm` is not supported.")
  if precision[0] != precision[1]:
    raise NotImplementedError("Mismatched `Precision`s not supported.")

  precision = precision[0]

  if a_dtype != b_dtype:
    # TODO: Support this case.
    raise ValueError("Cannot infer precision if operand types differ.")
  dtype = jnp.dtype(a_dtype)

  backend = jax.default_backend()
  if backend == "gpu":
    device = jax.devices()[0]
    compute_capability = getattr(device, "compute_capability", None)
    if compute_capability is None:
      logging.warning(
          "Unknown GPU compute capability when determining dot precision"
          " preset; assuming compute_capability >= 8.0"
      )
    elif float(compute_capability) < 8.0:
      backend = "gpu_old"

  match dtype:
    case jnp.float16:
      if backend == "tpu":
        match precision:
          case Precision.DEFAULT | None:
            return DotAlgorithmPreset.BF16_BF16_F32
          case Precision.HIGH:
            return DotAlgorithmPreset.BF16_BF16_F32_X3
          case Precision.HIGHEST:
            return DotAlgorithmPreset.BF16_BF16_F32_X6
          case _:
            raise ValueError(f"Unexpected precision {precision}")
      else:
        return DotAlgorithmPreset.F16_F16_F32
    case jnp.bfloat16:
      if backend == "gpu_old":
        return DotAlgorithmPreset.F32_F32_F32
      else:
        return DotAlgorithmPreset.BF16_BF16_F32
    case jnp.float32:
      new_precision = _F32_DOT_PRECISION_MAP[backend][precision]
      return DotAlgorithmPreset[new_precision]
    case jnp.float64:
      return DotAlgorithmPreset.F64_F64_F64
    case _:
      raise ValueError(f"Unsupported dtype: {dtype}")


def is_default(
    a_dtype: DTypeLike, b_dtype: DTypeLike, precision: PrecisionLike
) -> bool:
  """Returns whether the given precision is equivalent to `DEFAULT`."""
  if precision in (Precision.DEFAULT, DotAlgorithmPreset.DEFAULT):
    return True
  try:
    default = to_dot_algorithm_preset(a_dtype, b_dtype, Precision.DEFAULT)
    return to_dot_algorithm_preset(a_dtype, b_dtype, precision) == default
  except (ValueError, NotImplementedError):
    return False


def precision_input_dtype(precision: DotAlgorithmPreset) -> jnp.dtype:
  """Returns the input dtype for the given precision."""
  dtypes = precision.supported_lhs_types
  if dtypes is None:
    raise ValueError(f"Could not obtain input dtype for {precision=}.")
  return jnp.dtype(dtypes[0])
