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


def get_kv_cache_shape(total_num_pages, page_size, kv_dim, kv_dtype):
  """KV cache shape for v3's SEQ_ALONG_LANE layout.

  v1/v2 store tokens on sublanes as
  `[pages, page_size // packing, packing, kv_dim]`. v3 puts tokens on the lane
  axis instead, so the cache is the 3D transpose `[pages, kv_dim, page_size]`
  and the packing factor no longer appears in the shape -- it is implied by the
  dtype. Every DMA in `bref_override` slices this as `[p, :, ds(off, sz)]`,
  which is why `page_size` has to be the minormost dimension.

  The signature matches `v1.kernel.get_kv_cache_shape` so the runner can swap
  between them; `kv_dtype` is unused here and kept only for that reason.
  """
  del kv_dtype
  # Mirrors `MlaConfigs.kv_dim_align`.
  return (
      total_num_pages,
      align_to(kv_dim, pltpu.get_tpu_info().num_lanes),
      page_size,
  )


def transpose_kv_cache_to_seq_along_lane(
    cache_kv: jax.Array,
    kv_packing: int | None = None,
) -> jax.Array:
  """Transposes KV cache to 3D transposed format [pages, kv_dim, page_size]."""
  if cache_kv.ndim == 4:
    total_num_pages, page_size_per_packing, packing, kv_dim = cache_kv.shape
    if kv_packing is not None:
      assert kv_packing == packing
    page_size = page_size_per_packing * packing
    flat_tokens = cache_kv.reshape((total_num_pages, page_size, kv_dim))
  else:
    total_num_pages, page_size, kv_dim = cache_kv.shape
    flat_tokens = cache_kv

  return flat_tokens.transpose((0, 2, 1))


def transpose_kv_cache_from_seq_along_lane(
    transposed_cache_kv: jax.Array,
    kv_packing: int | None = None,
) -> jax.Array:
  """Transposes 3D transposed KV cache [pages, kv_dim, page_size]

  back to untransposed 4D KV cache [pages, page_size // P, P, kv_dim].
  """
  if transposed_cache_kv.ndim == 4:
    total_num_pages, kv_sublanes, packing, page_size = transposed_cache_kv.shape
    if kv_packing is not None:
      assert kv_packing == packing
    kv_dim = kv_sublanes * packing
    flat_channels = transposed_cache_kv.reshape(
        (total_num_pages, kv_dim, page_size)
    )
  else:
    total_num_pages, kv_dim, page_size = transposed_cache_kv.shape
    flat_channels = transposed_cache_kv
    if kv_packing is None:
      kv_packing = get_dtype_packing(transposed_cache_kv.dtype)
    packing = kv_packing

  untransposed_2d = flat_channels.transpose((0, 2, 1))
  return untransposed_2d.reshape(
      (total_num_pages, page_size // packing, packing, kv_dim)
  )
