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

from typing import Literal

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
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    num_pages: int,
    mode: Literal["decode", "prefill", "mixed"] = "decode",
    seq_len: int = 4096,
    num_seqs: int = 1,
    num_decode_seqs: int = 0,
    decode_seq_len: int = 4096,
    num_prefill_seqs: int = 0,
    prefill_seq_len: int = 2048,
    q_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    kv_cache_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    project: str = "qwen3.5",
    tags: tuple[arg_spec.Tag, ...] = ("primary", "ci_tests"),
) -> arg_spec.ArgSpec:
  """Creates an ArgSpec for Batched RPA with concrete metadata arrays.

  Generates inputs and metadata arrays from sequence length and mode,
  following the approach in batched RPA microbenchmarks.
  """
  head_dim_aligned = _cdiv(head_dim, 128) * 128

  if mode == "decode":
    total_q_len = num_seqs
    cu_q_lens_list = list(range(num_seqs + 1))
    kv_lens_list = [seq_len] * num_seqs
    distribution_list = [num_seqs, num_seqs, num_seqs]
    pages_per_seq = _cdiv(seq_len, page_size)
    page_indices_list = list(range(num_seqs * pages_per_seq))
    page_count = num_seqs * pages_per_seq
  elif mode == "prefill":
    total_q_len = num_seqs * seq_len
    cu_q_lens_list = [i * seq_len for i in range(num_seqs + 1)]
    kv_lens_list = [seq_len] * num_seqs
    distribution_list = [0, 0, num_seqs]
    pages_per_seq = _cdiv(seq_len, page_size)
    page_indices_list = list(range(num_seqs * pages_per_seq))
    page_count = num_seqs * pages_per_seq
  elif mode == "mixed":
    total_q_len = num_decode_seqs + num_prefill_seqs * prefill_seq_len
    cu_q_lens_list = list(range(num_decode_seqs + 1))
    for i in range(1, num_prefill_seqs + 1):
      cu_q_lens_list.append(num_decode_seqs + i * prefill_seq_len)
    kv_lens_list = [decode_seq_len] * num_decode_seqs + [
        prefill_seq_len
    ] * num_prefill_seqs
    total_seqs = num_decode_seqs + num_prefill_seqs
    distribution_list = [num_decode_seqs, num_decode_seqs, total_seqs]
    decode_pages = _cdiv(decode_seq_len, page_size)
    prefill_pages = _cdiv(prefill_seq_len, page_size)
    pages_per_seq = max(decode_pages, prefill_pages)
    page_indices_list = []
    page_count = 0
    for seq_pages in [decode_pages] * num_decode_seqs + [
        prefill_pages
    ] * num_prefill_seqs:
      indices = list(range(page_count, page_count + seq_pages))
      page_indices_list.extend(indices + [0] * (pages_per_seq - seq_pages))
      page_count += seq_pages
  else:
    raise ValueError(f"Unknown mode: {mode}")

  total_num_pages = max(num_pages, page_count)

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
        mode="decode",
        num_seqs=16,
        seq_len=4096,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=256,
        num_pages=512,
        tags=("primary", "ci_tests"),
    ),
    _make_rpa_spec(
        "qwen35_decode_bs64",
        mode="decode",
        num_seqs=64,
        seq_len=4096,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=256,
        num_pages=2048,
        tags=("primary",),
    ),
    _make_rpa_spec(
        "qwen35_decode_bs256",
        mode="decode",
        num_seqs=256,
        seq_len=4096,
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
        mode="prefill",
        num_seqs=1,
        seq_len=4096,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=256,
        num_pages=256,
        tags=("primary",),
    ),
    # Mixed workload (decode + prefill).
    # TODO rgxu: How do we express input token q_len=1024 and final
    # KV_len=8192 in this approach? We need to simulate a chunked prefill
    # scenario.
    _make_rpa_spec(
        "qwen35_mixed_d8_p1",
        mode="mixed",
        num_decode_seqs=8,
        decode_seq_len=4096,
        num_prefill_seqs=1,
        prefill_seq_len=2048,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=256,
        num_pages=1024,
        tags=("primary",),
    ),
)
