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

SUPPORTED_PRECISIONS: Final[tuple[jax.lax.DotAlgorithmPreset, ...]] = (
    jax.lax.DotAlgorithmPreset.F16_F16_F32,
    jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
    jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3,
    jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6,
    jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X9,
    jax.lax.DotAlgorithmPreset.F32_F32_F32,
    jax.lax.DotAlgorithmPreset.TF32_TF32_F32,
    jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3,
)

_F32_DOT_PRECISION_MAP: Final[dict[str, dict[jax.lax.Precision, str]]] = dict(
    tpu={
        jax.lax.Precision.DEFAULT: "BF16_BF16_F32",
        jax.lax.Precision.HIGH: "BF16_BF16_F32_X3",
        jax.lax.Precision.HIGHEST: "BF16_BF16_F32_X6",
    },
    gpu_old={
        jax.lax.Precision.DEFAULT: "F32_F32_F32",
        jax.lax.Precision.HIGH: "F32_F32_F32",
        jax.lax.Precision.HIGHEST: "F32_F32_F32",
    },
    gpu={
        jax.lax.Precision.DEFAULT: "TF32_TF32_F32",
        jax.lax.Precision.HIGH: "TF32_TF32_F32",
        jax.lax.Precision.HIGHEST: "F32_F32_F32",
    },
    cpu={
        jax.lax.Precision.DEFAULT: "F32_F32_F32",
        jax.lax.Precision.HIGH: "F32_F32_F32",
        jax.lax.Precision.HIGHEST: "F32_F32_F32",
    },
)


def _canonicalize_precision(
    precision: jax.lax.PrecisionLike,
) -> jax.lax.DotAlgorithmPreset | jax.lax.Precision:
  """Converts a `str` to a `DotAlgorithmPreset` or `Precision`."""

  if precision is None:
    precision = jax.config.jax_default_matmul_precision
    precision = precision or jax.lax.Precision.DEFAULT

  if isinstance(precision, tuple):
    if len(precision) != 2:
      raise ValueError(f"Expected 2 elements in tuple, got {len(precision)}.")
    if precision[0] != precision[1]:
      raise NotImplementedError("Only identical precision pairs are supported.")
    precision = precision[0]

  if isinstance(precision, jax.lax.DotAlgorithm):
    raise NotImplementedError("`DotAlgorithm` is not yet supported.")

  if isinstance(precision, (jax.lax.DotAlgorithmPreset, jax.lax.Precision)):
    return precision
  elif isinstance(precision, str):
    if precision in jax.lax.DotAlgorithmPreset.__members__:
      return jax.lax.DotAlgorithmPreset[precision]
    try:
      # jax.lax.Precision supports aliases like 'fastest' for
      # jax.lax.Precision.DEFAULT. Can only tell whether a string is a valid
      # alias by trying the constructor.
      return jax.lax.Precision(precision)
    except ValueError:
      raise ValueError(  # pylint: disable=raise-missing-from
          f"Unsupported enum value: {precision}. Must be refer to either"
          " a `jax.lax.DotAlgorithmPreset` or a `jax.lax.Precision` enum."
      )
  else:
    raise ValueError(f"Invalid precision: {precision}")


def to_dot_algorithm_preset(
    a_dtype: jax.typing.DTypeLike,
    b_dtype: jax.typing.DTypeLike,
    precision: jax.lax.PrecisionLike,
) -> jax.lax.DotAlgorithmPreset:
  """Converts a `PrecisionLike` to a `DotAlgorithmPreset`."""

  precision = _canonicalize_precision(precision)

  if isinstance(precision, jax.lax.DotAlgorithmPreset):
    return precision

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
          case jax.lax.Precision.DEFAULT | None:
            return jax.lax.DotAlgorithmPreset.BF16_BF16_F32
          case jax.lax.Precision.HIGH:
            return jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3
          case jax.lax.Precision.HIGHEST:
            return jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6
          case _:
            raise ValueError(f"Unexpected precision {precision}")
      else:
        return jax.lax.DotAlgorithmPreset.F16_F16_F32
    case jnp.bfloat16:
      if backend == "gpu_old":
        return jax.lax.DotAlgorithmPreset.F32_F32_F32
      else:
        return jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    case jnp.float32:
      new_precision = _F32_DOT_PRECISION_MAP[backend][precision]
      return jax.lax.DotAlgorithmPreset[new_precision]
    case _:
      raise ValueError(f"Unsupported dtype: {dtype}")


def precision_input_dtype(precision: jax.lax.DotAlgorithmPreset) -> jnp.dtype:
  """Returns the input dtype for the given precision."""
  dtypes = precision.supported_lhs_types
  if dtypes is None:
    raise ValueError(f"Could not obtain input dtype for {precision=}.")
  return jnp.dtype(dtypes[0])
