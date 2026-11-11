# Copyright 2026 Google LLC
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
"""Utility functions for Batched Multi-Head Latent Attention (MLA)."""

from collections.abc import Sequence
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def align_to(a: Any, b: int) -> Any:
  """Returns 'a' aligned up to the nearest multiple of 'b'."""
  return pl.cdiv(a, b) * b


def get_tpu_num_lanes() -> int:
  """Returns the number of physical lanes on the current TPU (typically 128)."""
  return pltpu.get_tpu_info().num_lanes


def align_span(start: Any, size: Any, align: int) -> tuple[Any, Any, Any]:
  """Returns (aligned_start, lead, aligned_size) for [start, start + size).

  Grows the span outward to multiples of `align` on both ends; `lead` is how
  far `aligned_start` moved down from `start`.
  """
  lead = start % align
  return start - lead, lead, align_to(lead + size, align)


def broadcast_minor(src: jax.Array, shape: Sequence[int]) -> jax.Array:
  """Broadcasts 'src' to 'shape' in the minor dimension."""
  if src.shape == shape:
    return src
  assert src.shape[:-1] == shape[:-1]
  if src.shape[-1] == 1:
    return jnp.broadcast_to(src, shape)
  num_lanes = pltpu.get_tpu_info().num_lanes
  assert src.shape[-1] % num_lanes == 0
  target_minor = align_to(shape[-1], src.shape[-1])
  reps = (1,) * (src.ndim - 1) + (target_minor // src.shape[-1],)
  broadcasted = jnp.tile(src, reps)
  return broadcasted[..., : shape[-1]]


def get_dtype_packing(dtype: jax.typing.DTypeLike) -> int:
  """Returns number of packed elements per 32-bit word."""
  return 32 // jax.dtypes.itemsize_bits(dtype)

