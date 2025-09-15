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
"""Mosaic-TPU utils."""

import re
from typing import Final
import jax
import jax.numpy as jnp


# TODO: Add tests for this file.

_SUPPORTED_TPU_GENERATIONS: Final[dict[str, int]] = {
    "TPU v4": 4,
    "TPU v5 lite": 5,
    "TPU v5": 5,
    "TPU v5e": 5,
    "TPU v5p": 5,
    "TPU v6 lite": 6,
    "TPU7x": 7,
}


def tpu_generation() -> int:
  """Generation number of the currently attached TPU."""
  device_kind = jax.devices()[0].device_kind
  try:
    return _SUPPORTED_TPU_GENERATIONS[device_kind]
  except KeyError as e:
    raise ValueError(f"{device_kind} is not a supported TPU device") from e


def has_mosaic_tpu_support() -> bool:
  """Checks if Mosaic TPU is supported on the attached TPU."""
  return "TPU" in jax.devices()[0].device_kind and tpu_generation() >= 4


def supports_bfloat16_matmul() -> bool:
  """Checks TPU generation to determine if bfloat16 matmul is supported."""
  return "TPU" in jax.devices()[0].device_kind and tpu_generation() >= 4


def assert_is_supported_dtype(dtype: jnp.dtype) -> None:
  if dtype not in (jnp.bfloat16, jnp.float32):
    raise ValueError(f"Expected bfloat16 or float32 array but got {dtype}.")


def select_input_dtype(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.dtype:
  """A type to which both input should be adapted to before dot product."""
  # bf16xbf16 matmul is only supported since TPUv4 generation. In case of mixed
  # input precision, we need to convert bf16 argument to fp32 beforehand.
  if not has_mosaic_tpu_support():
    raise ValueError("Mosaic TPU is not supported on this platform.")
  if lhs.dtype == rhs.dtype == jnp.bfloat16:
    return jnp.bfloat16
  else:
    return jnp.float32
