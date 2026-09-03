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
"""Utility functions for MLA kernel."""

import math
import jax
import jax.numpy as jnp
import numpy as np


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def align_to(a, b):
  return ((a + b - 1) // b) * b


def get_dtype_bitwidth(dtype):
  return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
  bits = get_dtype_bitwidth(dtype)
  return 32 // bits


def generate_mla_inputs(
    seq_lens,  # List[(q_len, kv_len)]
    num_heads,
    lkv_dim,
    r_dim,
    page_size,
    q_dtype,
    kv_dtype,
    num_pages,
    rng=None,
    with_kv_cache=True,
):
  """Generates inputs for the MLA kernel.

  Args:
    seq_lens: List of (q_len, kv_len) for each sequence.
    num_heads: Number of attention heads.
    lkv_dim: Dimension of the linear KV part.
    r_dim: Dimension of the rotary embedding part.
    page_size: Size of each page in the KV cache.
    q_dtype: Data type for queries.
    kv_dtype: Data type for keys and values.
    num_pages: Total number of pages in the cache.
    rng: Optional numpy random number generator.
    with_kv_cache: Whether to generate KV cache.

  Returns:
    A tuple containing:
      - ql_nope: Query linear part without positional encoding.
      - q_pe: Query positional encoding part.
      - new_kv_c: New KV cache data.
      - new_k_pe: New Key positional encoding.
      - cache_kv: The existing KV cache.
      - kv_lens: Array of KV lengths for each sequence.
      - page_indices: Indices mapping sequence pages to cache pages.
      - cu_q_lens: Cumulative query lengths.
      - distribution: Mode distribution (e.g., prefill, decode, mixed).
  """
  if rng is None:
    rng = np.random.default_rng(1234)

  def gen_random(shape, dtype) -> jax.Array:
    return jnp.array(rng.random(size=shape, dtype=np.float32)).astype(dtype)

  padded_r_dim = align_to(r_dim, 128)
  padded_lkv_dim = align_to(lkv_dim, 128)
  padded_kv_dim = padded_lkv_dim + padded_r_dim
  packing = get_dtype_packing(kv_dtype)
  q_lens = [s[0] for s in seq_lens]
  kv_lens_list = [s[1] for s in seq_lens]
  total_q_len = sum(q_lens)
  cu_q_lens_list = [0]
  for q_len in q_lens:
    cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)

  max_kv_len = max(kv_lens_list) if kv_lens_list else 0
  max_q_len = max(q_lens) if q_lens else 0
  pages_per_seq = cdiv(max_kv_len + max_q_len, page_size)

  page_indices_list = []
  page_count = 0
  seq_pages = []
  for kv_len in kv_lens_list:
    num_seq_pages = cdiv(kv_len, page_size)
    indices = list(range(page_count, page_count + num_seq_pages))
    seq_pages.append(indices)
    page_count += num_seq_pages

  total_num_pages = max(num_pages, len(seq_lens) * pages_per_seq)

  extra_page_idx = page_count
  for indices in seq_pages:
    needed = pages_per_seq - len(indices)
    extra_indices = []
    for _ in range(needed):
      if extra_page_idx < total_num_pages:
        extra_indices.append(extra_page_idx)
        extra_page_idx += 1
      else:
        extra_indices.append(-1)
    page_indices_list.extend(indices + extra_indices)

  ql_nope = gen_random((total_q_len, num_heads, lkv_dim), q_dtype)
  q_pe = gen_random((total_q_len, num_heads, r_dim), q_dtype)
  new_kv_c = gen_random((total_q_len, lkv_dim), kv_dtype)
  new_k_pe = gen_random((total_q_len, r_dim), kv_dtype)

  if with_kv_cache:
    cache_kv = gen_random(
        (total_num_pages, page_size // packing, packing, padded_kv_dim),
        kv_dtype,
    )
  else:
    cache_kv = None
  kv_lens = jnp.array(kv_lens_list, dtype=jnp.int32)
  page_indices = jnp.array(page_indices_list, dtype=jnp.int32)
  cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)

  # Find the number of decode sequences at the beginning of the batch.
  num_decode_seqs = 0
  for s in seq_lens:
    if s[0] == 1:
      num_decode_seqs += 1
    else:
      break
  distribution = jnp.array(
      [num_decode_seqs, num_decode_seqs, len(seq_lens)], dtype=jnp.int32
  )

  return (
      ql_nope,
      q_pe,
      new_kv_c,
      new_k_pe,
      cache_kv,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
  )


def unsigned_cdiv(a, b):
  exponent = int(math.log2(b))
  if b == int(math.pow(2, exponent)):
    return (a + b - 1) >> exponent
  return (a + b - 1) // b


def unsigned_align_to(a, b):
  exponent = int(math.log2(b))
  if b == int(math.pow(2, exponent)):
    return (a + b - 1) & (-int(b))
  return unsigned_cdiv(a, b) * b


def static_validate_inputs(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_blocks: tuple[int, int, int] | None = None,
    num_queries_per_blocks: tuple[int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    decode_batch_size: int = 1,
    transpose_kv_cache: bool = False,
    debug_mode: bool = False,
):
  """Validate inputs to the MLA RPA implementation statically."""
  if len(ql_nope.shape) != 3:
    raise ValueError(f"Expected 3D array for {ql_nope.shape=}")
  if len(q_pe.shape) != 3:
    raise ValueError(f"Expected 3D array for {q_pe.shape=}")
  if len(new_kv_c.shape) != 2:
    raise ValueError(f"Expected 2D array for {new_kv_c.shape=}")
  if len(new_k_pe.shape) != 2:
    raise ValueError(f"Expected 2D array for {new_k_pe.shape=}")

  if ql_nope.shape[:2] != q_pe.shape[:2]:
    raise ValueError(
        f"Expected {ql_nope.shape[:2]=} to be equal to {q_pe.shape[:2]=}"
    )
  if ql_nope.shape[0] != new_kv_c.shape[0]:
    raise ValueError(
        f"Expected {ql_nope.shape[0]=} to be equal to {new_kv_c.shape[0]=}"
    )
  if new_kv_c.shape[0] != new_k_pe.shape[0]:
    raise ValueError(
        f"Expected {new_kv_c.shape[0]=} to be equal to {new_k_pe.shape[0]=}"
    )
  if ql_nope.shape[2] != new_kv_c.shape[1]:
    raise ValueError(
        f"Expected {ql_nope.shape[2]=} to be equal to {new_kv_c.shape[1]=}"
    )
  if q_pe.shape[2] != new_k_pe.shape[1]:
    raise ValueError(
        f"Expected {q_pe.shape[2]=} to be equal to {new_k_pe.shape[1]=}"
    )

  actual_lkv_dim = ql_nope.shape[2]
  actual_r_dim = q_pe.shape[2]
  lkv_dim = unsigned_align_to(actual_lkv_dim, 128)
  r_dim = unsigned_align_to(actual_r_dim, 128)

  if not transpose_kv_cache:
    (
        _,
        _,
        kv_packing,
        kv_dim,
    ) = cache_kv.shape
    if kv_packing != get_dtype_packing(cache_kv.dtype):
      raise ValueError(f"{kv_packing=} does not match with {cache_kv.dtype=}")
  else:
    (
        _,
        kv_dim,
        _,
    ) = cache_kv.shape

  if lkv_dim + r_dim != kv_dim:
    raise ValueError(f"Expected {lkv_dim=} + {r_dim=} to be equal to {kv_dim=}")

  if not (cache_kv.dtype == new_kv_c.dtype):
    raise ValueError(
        f"Expected {cache_kv.dtype=} to be equal to {new_kv_c.dtype=}."
    )
  if not (cache_kv.dtype == new_k_pe.dtype):
    raise ValueError(
        f"Expected {cache_kv.dtype=} to be equal to {new_k_pe.dtype=}."
    )

  if not jnp.issubdtype(cache_kv.dtype, jnp.floating):
    raise ValueError(f"Expected {cache_kv.dtype=} to be a floating point.")

  if not (
      jnp.int32
      == kv_lens.dtype
      == page_indices.dtype
      == cu_q_lens.dtype
      == distribution.dtype
  ):
    raise ValueError(
        f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}, {distribution.dtype=}"
    )

  if not (
      len(kv_lens.shape) == len(page_indices.shape) == len(cu_q_lens.shape) == 1
  ):
    raise ValueError(
        f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
        f" {cu_q_lens.shape=}"
    )

  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  if num_page_indices % max_num_seqs != 0:
    raise ValueError(
        f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)."
    )
  if distribution.shape != (3,):
    raise ValueError(f"Expected {distribution.shape=} to be (3,).")

  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if chunk_prefill_size is not None and chunk_prefill_size <= 0:
    raise ValueError(f"{chunk_prefill_size=} must be positive.")
  if num_kv_pages_per_blocks is not None:
    for num_kv_pages_per_block in num_kv_pages_per_blocks:
      if num_kv_pages_per_block <= 0:
        raise ValueError(f"{num_kv_pages_per_block=} must be positive.")
  if num_queries_per_blocks is not None:
    for num_queries_per_block in num_queries_per_blocks:
      if num_queries_per_block <= 0:
        raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")

  del sm_scale
  del mask_value
  del q_scale
  del k_scale
  del v_scale
  del decode_batch_size
  del debug_mode
