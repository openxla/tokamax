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
    prefill_kv_lens: Sequence[int] | None = None,
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
    # overrides `num_prefill_seqs` and `prefill_seq_len`. `prefill_kv_lens`, if
    # given, sets each prefill sequence's final KV length (cached prefix plus
    # the new chunk), so chunked prefill attends to more KV than it has queries.
    if prefill_seq_lens is None:
      prefill_lens = [prefill_seq_len] * num_prefill_seqs
    else:
      prefill_lens = list(prefill_seq_lens)
    if prefill_kv_lens is None:
      prefill_kvs = prefill_lens
    else:
      prefill_kvs = list(prefill_kv_lens)
      if len(prefill_kvs) != len(prefill_lens) or any(
          kv < q for q, kv in zip(prefill_lens, prefill_kvs)
      ):
        raise ValueError(
            f"Invalid {prefill_kv_lens=} for prefill lengths {prefill_lens}."
        )
    total_q_len = num_decode_seqs + sum(prefill_lens)
    cu_q_lens_list = list(range(num_decode_seqs + 1))
    for prefill_len in prefill_lens:
      cu_q_lens_list.append(cu_q_lens_list[-1] + prefill_len)
    kv_lens_list = [decode_seq_len] * num_decode_seqs + prefill_kvs
    total_seqs = num_decode_seqs + len(prefill_lens)
    distribution_list = [num_decode_seqs, num_decode_seqs, total_seqs]
    seq_pages_list = [_cdiv(decode_seq_len, page_size)] * num_decode_seqs + [
        _cdiv(prefill_kv, page_size) for prefill_kv in prefill_kvs
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
      # Batched RPA is inference-only and ignores `return_residuals`, so skip
      # the derived residual and VJP variants.
      tags=tuple(dict.fromkeys((*tags, "forward_only"))),
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


# Qwen3.5-397B full-attention layers: 32 query heads, 2 KV heads, head_dim 256.
# Shapes are per device with heads sharded over TP, so KV heads are replicated
# when TP > 2. The serving workloads use DP8 attention (TP1 heads per device),
# so each device holds ceil(concurrency / 8) sequences.
_QWEN3_5_NUM_Q_HEADS = 32
_QWEN3_5_NUM_KV_HEADS = 2
_QWEN3_5_HEAD_DIM = 256
_QWEN3_5_PAGE_SIZE = 256
_QWEN3_5_DP = 8
# Primary specs: the nightly benchmark set, one per workload type.
_QWEN3_5_PRIMARY = frozenset({
    "decode_bs16_tp1",
    "decode_bs64_tp1",
    "decode_bs256_tp1",
    "prefill_q4096_tp1",
    "isl8k_mixed_c64_dp8tp1",
})
_QWEN3_5_CI_TESTS = frozenset({"decode_bs16_tp1"})


def _qwen3_5_tags(name: str) -> tuple[arg_spec.Tag, ...]:
  if name in _QWEN3_5_CI_TESTS:
    return ("primary", "ci_tests")
  if name in _QWEN3_5_PRIMARY:
    return ("primary",)
  return ()


def _qwen3_5_model_args(tp: int) -> dict[str, Any]:
  """Returns the per-device Qwen3.5-397B `_make_rpa_spec` args for TP `tp`."""
  return dict(
      num_q_heads=_QWEN3_5_NUM_Q_HEADS // tp,
      num_kv_heads=max(_QWEN3_5_NUM_KV_HEADS // tp, 1),
      head_dim=_QWEN3_5_HEAD_DIM,
      page_size=_QWEN3_5_PAGE_SIZE,
      num_pages=0,  # Sized to fit the workload.
      project="qwen3_5_397b",
  )


def _qwen3_5_decode(
    name: str, *, num_seqs: int, context_len: int, tp: int = 1
) -> arg_spec.ArgSpec:
  return _make_rpa_spec(
      name,
      mode="decode",
      num_seqs=num_seqs,
      seq_len=context_len,
      tags=_qwen3_5_tags(name),
      **_qwen3_5_model_args(tp),
  )


def _qwen3_5_prefill(name: str, *, seq_len: int, tp: int) -> arg_spec.ArgSpec:
  return _make_rpa_spec(
      name,
      mode="prefill",
      num_seqs=1,
      seq_len=seq_len,
      tags=_qwen3_5_tags(name),
      **_qwen3_5_model_args(tp),
  )


def _qwen3_5_chunked_prefill(
    name: str,
    *,
    num_seqs: int,
    chunk_len: int,
    context_len: int,
    num_decode_seqs: int = 0,
    decode_context_len: int = 0,
    tp: int = 1,
) -> arg_spec.ArgSpec:
  """Last prefill chunk of `chunk_len` tokens on a `context_len` KV context."""
  return _make_rpa_spec(
      name,
      mode="mixed",
      num_decode_seqs=num_decode_seqs,
      decode_seq_len=decode_context_len,
      prefill_seq_lens=[chunk_len] * num_seqs,
      prefill_kv_lens=[context_len] * num_seqs,
      tags=_qwen3_5_tags(name),
      **_qwen3_5_model_args(tp),
  )


def _qwen3_5_isl8k_specs(concurrency: int) -> tuple[arg_spec.ArgSpec, ...]:
  """8k-input / 1k-output serving at `concurrency`, with DP8 attention."""
  num_seqs = _cdiv(concurrency, _QWEN3_5_DP)
  label = f"c{concurrency}_dp8tp1"
  return (
      # Mixed step: decode sequences at 8k + 1k context, plus one sequence's
      # last 1k prefill chunk on its 8k prompt.
      _qwen3_5_chunked_prefill(
          f"isl8k_mixed_{label}",
          num_seqs=1,
          chunk_len=1024,
          context_len=8192,
          num_decode_seqs=max(num_seqs - 1, 1),
          decode_context_len=9216,
      ),
      # Pure decode (TPOT).
      _qwen3_5_decode(
          f"isl8k_decode_{label}", num_seqs=num_seqs, context_len=9216
      ),
      # Last prefill chunk (TTFT).
      _qwen3_5_chunked_prefill(
          f"isl8k_prefill_{label}",
          num_seqs=num_seqs,
          chunk_len=1024,
          context_len=8192,
      ),
  )


# Long-context batches are limited to these KV tokens per device, which bounds
# the benchmark's HBM use.
_QWEN3_5_LONGCTX_DECODE_KV_TOKENS = 2 * 1024 * 1024
_QWEN3_5_LONGCTX_CHUNK_KV_TOKENS = 1024 * 1024


def _qwen3_5_longctx_specs(
    context_len: int, concurrency: int
) -> tuple[arg_spec.ArgSpec, ...]:
  """Multi-turn long context: decode and a 2k-token turn, with DP8 attention."""
  num_seqs = concurrency // _QWEN3_5_DP
  label = f"c{concurrency}_ctx{context_len // 1024}k_dp8tp1"
  specs = []
  if num_seqs * context_len <= _QWEN3_5_LONGCTX_DECODE_KV_TOKENS:
    specs.append(
        _qwen3_5_decode(
            f"decode_{label}", num_seqs=num_seqs, context_len=context_len
        )
    )
  if num_seqs * context_len <= _QWEN3_5_LONGCTX_CHUNK_KV_TOKENS:
    specs.append(
        _qwen3_5_chunked_prefill(
            f"last_chunk_{label}",
            num_seqs=num_seqs,
            chunk_len=2048,
            context_len=context_len,
        )
    )
  return tuple(specs)


_QWEN3_5_397B_ARG_SPECS = (
    # Decode at a 4k context.
    *(
        _qwen3_5_decode(
            f"decode_bs{num_seqs}_tp{tp}",
            num_seqs=num_seqs,
            context_len=4096,
            tp=tp,
        )
        for num_seqs in (16, 64, 256, 512)
        for tp in (1, 4, 8)
    ),
    # Full-prompt prefill.
    *(
        _qwen3_5_prefill(f"prefill_q{seq_len}_tp{tp}", seq_len=seq_len, tp=tp)
        for seq_len in (1024, 2048, 4096, 8192)
        for tp in (1, 4, 8)
    ),
    # 8k-input / 1k-output serving.
    *(
        spec
        for concurrency in (4, 8, 16, 32, 64, 128, 256)
        for spec in _qwen3_5_isl8k_specs(concurrency)
    ),
    # Last 1k prefill chunk on a 4k context.
    *(
        _qwen3_5_chunked_prefill(
            f"last_chunk_c{num_seqs}_tp{tp}",
            num_seqs=num_seqs,
            chunk_len=1024,
            context_len=4096,
            tp=tp,
        )
        for num_seqs in (16, 64, 256, 512)
        for tp in (1, 4, 8)
    ),
    # Long context (64k to 256k).
    *(
        spec
        for context_len in (65536, 131072, 262144)
        for concurrency in (16, 64, 256, 512)
        for spec in _qwen3_5_longctx_specs(context_len, concurrency)
    ),
)


ARG_SPECS = (
    *_QWEN3_5_397B_ARG_SPECS,
    *_QWEN3_8_MAX_ARG_SPECS,
)
