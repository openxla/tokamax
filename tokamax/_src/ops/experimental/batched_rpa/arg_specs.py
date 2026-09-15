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
"""Benchmark and autotuning argument specifications for Batched RPA."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.autotuning import arg_spec


class HashableNPArray(np.ndarray):
  """Hashable numpy array for use as an ArgSpec argument."""

  def __new__(cls, input_array):
    return np.asarray(input_array).view(cls)

  def __hash__(self):
    return hash((self.tobytes(), self.shape, self.dtype))


def _cdiv(a: int, b: int) -> int:
  return (a + b - 1) // b


def _make_rpa_spec(
    name: str,
    *,
    seq_lens: Sequence[tuple[int, int]],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    num_pages: int,
    q_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    kv_cache_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    project: str = "qwen3.5",
    tags: tuple[arg_spec.Tag, ...] = ("primary", "ci_tests"),
) -> arg_spec.ArgSpec:
  """Creates an ArgSpec for Batched RPA with concrete metadata arrays."""
  head_dim_aligned = _cdiv(head_dim, 128) * 128

  q_lens = [s[0] for s in seq_lens]
  kv_lens_list = [s[1] for s in seq_lens]
  total_q_len = sum(q_lens)
  max_num_seqs = len(seq_lens)

  # Build cumulative query lengths.
  cu_q_lens_list = [0]
  for q_len in q_lens:
    cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)

  # Build page table with valid page indices.
  max_kv_len = max(kv_lens_list) if kv_lens_list else 0
  pages_per_seq = _cdiv(max_kv_len, page_size)

  page_indices_list = []
  page_count = 0
  for kv_len in kv_lens_list:
    num_seq_pages = _cdiv(kv_len, page_size)
    indices = list(range(page_count, page_count + num_seq_pages))
    page_indices_list.extend(indices + [0] * (pages_per_seq - num_seq_pages))
    page_count += num_seq_pages

  total_num_pages = max(num_pages, page_count)

  # Build distribution: [num_decode, num_decode, total_seqs].
  num_decode_seqs = 0
  for s in seq_lens:
    if s[0] == 1:
      num_decode_seqs += 1
    else:
      break
  distribution_list = [num_decode_seqs, num_decode_seqs, len(seq_lens)]

  # Tensor shapes.
  queries = jax.ShapeDtypeStruct(
      (total_q_len, num_q_heads, head_dim), q_dtype
  )
  keys = jax.ShapeDtypeStruct(
      (total_q_len, num_kv_heads, head_dim), kv_cache_dtype
  )
  values = jax.ShapeDtypeStruct(
      (total_q_len, num_kv_heads, head_dim), kv_cache_dtype
  )
  kv_cache = jax.ShapeDtypeStruct(
      (total_num_pages, page_size, num_kv_heads * 2, head_dim_aligned),
      kv_cache_dtype,
  )

  # Concrete integer metadata arrays (critical for valid kernel execution).
  kv_lens = HashableNPArray(np.array(kv_lens_list, dtype=np.int32))
  page_indices = HashableNPArray(np.array(page_indices_list, dtype=np.int32))
  cu_q_lens = HashableNPArray(np.array(cu_q_lens_list, dtype=np.int32))
  distribution = HashableNPArray(np.array(distribution_list, dtype=np.int32))

  return arg_spec.ArgSpec(
      args=dict(
          queries=queries,
          keys=keys,
          values=values,
          kv_cache=kv_cache,
          kv_lens=kv_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
      ),
      project=project,
      name=name,
      tags=tags,
  )


ARG_SPECS = (
    # Decode workloads.
    _make_rpa_spec(
        "qwen35_decode_bs16",
        seq_lens=[(1, 4096)] * 16,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=256,
        num_pages=512,
        tags=("primary", "ci_tests"),
    ),
    _make_rpa_spec(
        "qwen35_decode_bs64",
        seq_lens=[(1, 4096)] * 64,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=256,
        num_pages=2048,
        tags=("primary",),
    ),
    _make_rpa_spec(
        "qwen35_decode_bs256",
        seq_lens=[(1, 4096)] * 256,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=256,
        num_pages=8192,
        tags=("primary",),
    ),
    # Prefill workloads.
    _make_rpa_spec(
        "qwen35_prefill_q4096",
        seq_lens=[(4096, 4096)],
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=256,
        num_pages=256,
        tags=("primary",),
    ),
    # Mixed workload (decode + prefill).
    _make_rpa_spec(
        "qwen35_mixed_d8_p1",
        seq_lens=[(1, 4096)] * 8 + [(2048, 2048)],
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=256,
        num_pages=1024,
        tags=("primary",),
    ),
)
