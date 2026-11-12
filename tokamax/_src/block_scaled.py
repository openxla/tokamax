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
"""Hooks for block-scaled matmuls on TPU.

Block-scaled matmuls are not available in open-source builds, where
`is_available()` returns False and the other hooks raise `NotImplementedError`.
"""

import types

import jax
import jax.numpy as jnp

FP8_DTYPES = (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2))
DTYPES = FP8_DTYPES


def is_available() -> bool:
  """Returns whether block-scaled matmul support is available in this build."""
  return False


def _layout() -> types.ModuleType:
  """Returns the block-scale layout module."""
  raise NotImplementedError(
      "Block-scaled matmul is not available in this build."
  )


def check_contracting_size(
    k: int, dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn
) -> None:
  """Raises `ValueError` if `k` is not a valid contracting dimension size."""
  _layout().check_contracting_size(k, dtype)


def check_tile_k(
    tile_k: int,
    k: int,
    dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
) -> None:
  """Raises `ValueError` if `tile_k` is not a valid contracting tile size."""
  _layout().check_tile_k(tile_k, k, dtype)


def scale_rows(
    k: int,
    *,
    dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
) -> int:
  """Returns the scale axis size for contracting dimension `k`."""
  return _layout().compact_scale_rows(k, dtype)


def kernel_scale_rows(
    k: int,
    *,
    dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
) -> int:
  """Returns the kernel scale axis size after `prepare_scales`."""
  return _layout().padded_rhs_scale_rows(k, dtype)


def activation_scale_vmem_bytes(
    tile_m: int,
    tile_k: int,
    dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
) -> int:
  """Returns scratch VMEM bytes for one activation tile's scales."""
  return _layout().lhs_scale_vmem_bytes(tile_m, tile_k, dtype)


def prepare_scales(
    scale: jax.Array,
    k: int,
    *,
    dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
) -> jax.Array:
  """Prepares `[..., rows, n]` weight scales for kernel execution."""
  return _layout().prepare_rhs_scales(scale, k, dtype)


def quantize(
    x: jax.Array,
    dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `[..., k, n]` weights into quantized data and `e8m0` scales."""
  return _layout().quantize_rhs(x, dtype)


def matmul(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    matmul_dtype: jax.typing.DTypeLike,
    acc_dtype: jax.typing.DTypeLike = jnp.float32,
) -> jax.Array:
  """Computes a 2D block-scaled matmul with dynamic activation quantization."""
  del x, w, w_scale, matmul_dtype, acc_dtype
  raise NotImplementedError(
      "Block-scaled matmul is not available in this build."
  )
