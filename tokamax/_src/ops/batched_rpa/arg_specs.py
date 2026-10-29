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
from typing import Any, Literal

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
    prefill_seq_lens: Sequence[int] | None = None,
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
    # `prefill_seq_lens`, if given, sets per-sequence prefill lengths and
    # overrides `num_prefill_seqs` and `prefill_seq_len`.
    if prefill_seq_lens is None:
      prefill_lens = [prefill_seq_len] * num_prefill_seqs
    else:
      prefill_lens = list(prefill_seq_lens)
    total_q_len = num_decode_seqs + sum(prefill_lens)
    cu_q_lens_list = list(range(num_decode_seqs + 1))
    for prefill_len in prefill_lens:
      cu_q_lens_list.append(cu_q_lens_list[-1] + prefill_len)
    kv_lens_list = [decode_seq_len] * num_decode_seqs + prefill_lens
    total_seqs = num_decode_seqs + len(prefill_lens)
    distribution_list = [num_decode_seqs, num_decode_seqs, total_seqs]
    seq_pages_list = [_cdiv(decode_seq_len, page_size)] * num_decode_seqs + [
        _cdiv(prefill_len, page_size) for prefill_len in prefill_lens
    ]
    pages_per_seq = max(seq_pages_list)
    page_indices_list = []
    page_count = 0
    for seq_pages in seq_pages_list:
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
  kv_packing = 32 // jax.dtypes.itemsize_bits(kv_cache_dtype)
  kv_cache = jax.ShapeDtypeStruct(
      (
          total_num_pages,
          page_size,
          _cdiv(num_kv_heads * 2, kv_packing),
          kv_packing,
          head_dim_aligned,
      ),
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


# Qwen3.8-Max full-attention ("Gated GQA") layers: 64 query heads, 4 KV heads,
# head_dim 256. These are 23 of the model's 92 layers; the other 69 are GDN.
# Shapes are per device (one TensorCore) with heads sharded over TP, so
# KV heads are replicated when TP > 4.
_QWEN3_8_MAX_NUM_Q_HEADS = 64
_QWEN3_8_MAX_NUM_KV_HEADS = 4
_QWEN3_8_MAX_HEAD_DIM = 256
_QWEN3_8_MAX_PAGE_SIZE = 256


def _qwen3_8_max_model_args(tp: int) -> dict[str, Any]:
  """Returns the per-device Qwen3.8-Max `_make_rpa_spec` args for TP `tp`."""
  return dict(
      num_q_heads=_QWEN3_8_MAX_NUM_Q_HEADS // tp,
      num_kv_heads=max(_QWEN3_8_MAX_NUM_KV_HEADS // tp, 1),
      head_dim=_QWEN3_8_MAX_HEAD_DIM,
      page_size=_QWEN3_8_MAX_PAGE_SIZE,
      num_pages=0,  # Sized to fit the workload.
      project="qwen3_8_max",
  )


def _qwen3_8_max_prefill(
    tp: int,
    num_seqs: int,
    seq_len: int = 8192,
    tags: tuple[arg_spec.Tag, ...] = (),
) -> arg_spec.ArgSpec:
  return _make_rpa_spec(
      f"tp{tp}_b{num_seqs}_prefill",
      mode="prefill",
      num_seqs=num_seqs,
      seq_len=seq_len,
      tags=tags,
      **_qwen3_8_max_model_args(tp),
  )


def _qwen3_8_max_decode(
    tp: int,
    num_seqs: int,
    context_len: int = 8192,
    tags: tuple[arg_spec.Tag, ...] = (),
) -> arg_spec.ArgSpec:
  return _make_rpa_spec(
      f"tp{tp}_b{num_seqs}_decode",
      mode="decode",
      num_seqs=num_seqs,
      seq_len=context_len,
      tags=tags,
      **_qwen3_8_max_model_args(tp),
  )


def _qwen3_8_max_mixed(
    tp: int,
    num_decode_seqs: int,
    prefill_seq_lens: Sequence[int],
    context_len: int = 8192,
    tags: tuple[arg_spec.Tag, ...] = (),
) -> arg_spec.ArgSpec:
  num_tokens = num_decode_seqs + sum(prefill_seq_lens)
  return _make_rpa_spec(
      f"tp{tp}_mixed_b{num_decode_seqs}_t{num_tokens}",
      mode="mixed",
      num_decode_seqs=num_decode_seqs,
      decode_seq_len=context_len,
      prefill_seq_lens=prefill_seq_lens,
      tags=tags,
      **_qwen3_8_max_model_args(tp),
  )


# Decode sequences attend to an 8192-token context unless noted. Mixed batches
# pair one-token decode steps with fresh prefill chunks and are named by their
# total token count.
_QWEN3_8_MAX_ARG_SPECS = (
    # TP2: 32 query heads, 2 KV heads.
    _qwen3_8_max_prefill(tp=2, num_seqs=1, tags=("primary",)),
    _qwen3_8_max_decode(tp=2, num_seqs=48, tags=("primary",)),
    _qwen3_8_max_decode(tp=2, num_seqs=80, context_len=5120),
    _qwen3_8_max_decode(tp=2, num_seqs=93),
    _qwen3_8_max_decode(tp=2, num_seqs=120),
    _qwen3_8_max_mixed(
        tp=2,
        num_decode_seqs=48,
        prefill_seq_lens=[8192 - 48],
        tags=("primary",),
    ),
    # TP4: 16 query heads, 1 KV head.
    _qwen3_8_max_prefill(tp=4, num_seqs=1),
    _qwen3_8_max_decode(tp=4, num_seqs=111),
    _qwen3_8_max_decode(tp=4, num_seqs=188),
    _qwen3_8_max_decode(tp=4, num_seqs=243),
    _qwen3_8_max_decode(tp=4, num_seqs=332),
    _qwen3_8_max_mixed(
        tp=4, num_decode_seqs=188, prefill_seq_lens=[8192 - 188]
    ),
    # TP4 serving workload from a Qwen3.8-Max 4-layer serving benchmark
    # (8 x 8k prefill, then 512-sequence decode at ~8.5k).
    _qwen3_8_max_prefill(tp=4, num_seqs=8),
    _qwen3_8_max_decode(tp=4, num_seqs=512, context_len=8500),
    # TP8: 8 query heads, 1 KV head (replicated 2x).
    _qwen3_8_max_prefill(tp=8, num_seqs=2),
    _qwen3_8_max_decode(tp=8, num_seqs=178),
    _qwen3_8_max_decode(tp=8, num_seqs=328),
    _qwen3_8_max_decode(tp=8, num_seqs=528),
    _qwen3_8_max_mixed(
        tp=8, num_decode_seqs=328, prefill_seq_lens=[8192 - 328]
    ),
    _qwen3_8_max_mixed(
        tp=8, num_decode_seqs=328, prefill_seq_lens=[8192, 8192 - 328]
    ),
    # TP16: 4 query heads, 1 KV head (replicated 4x).
    _qwen3_8_max_prefill(tp=16, num_seqs=4),
    _qwen3_8_max_decode(tp=16, num_seqs=448),
    _qwen3_8_max_decode(tp=16, num_seqs=704),
    # TP32: 2 query heads, 1 KV head (replicated 8x).
    _qwen3_8_max_prefill(tp=32, num_seqs=7),
    _qwen3_8_max_decode(tp=32, num_seqs=257),
    _qwen3_8_max_decode(tp=32, num_seqs=512),
    _qwen3_8_max_decode(tp=32, num_seqs=800),
    _qwen3_8_max_mixed(
        tp=32, num_decode_seqs=512, prefill_seq_lens=[8192 - 512]
    ),
    _qwen3_8_max_mixed(
        tp=32,
        num_decode_seqs=512,
        prefill_seq_lens=[8192] * 6 + [8192 - 512],
    ),
    # TP64: 1 query head, 1 KV head (replicated 16x).
    _qwen3_8_max_decode(tp=64, num_seqs=896),
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
    *_QWEN3_8_MAX_ARG_SPECS,
)
