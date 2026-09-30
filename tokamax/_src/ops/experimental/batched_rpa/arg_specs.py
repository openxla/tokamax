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
"""Benchmark and autotuning argument specifications for Batched RPA.

Covers representative model architectures and execution scenarios:
- Qwen 3.5 (397B, 35B) configurations across TP8, TP4, TP1 (DP8). 397B uses
  the HF full-attention config (32 Q heads, 2 KV heads, head_dim 256).
- 8k/1k serving: ISL=8192, OSL=1024, concurrencies 4..256, DP8 attention.
- Long context: decode and ~2k delta prefill over 64k-256k context.
- Prefill, decode, and mixed continuous batching scenarios.
- Chunked delta prefill attending over a large cached context prefix.
"""

from collections.abc import Sequence
from typing import Any, Final, Literal

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.autotuning import arg_spec


class HashableNPArray(np.ndarray):
  """Numpy array that can be retained as a concrete argument in an ArgSpec."""

  def __new__(cls, input_array: Any) -> "HashableNPArray":
    return np.asarray(input_array).view(cls)

  def __hash__(self) -> int:
    return hash((self.tobytes(), self.shape, self.dtype))

  def __eq__(self, other: Any) -> bool:
    return (
        isinstance(other, np.ndarray)
        and self.shape == other.shape
        and self.dtype == other.dtype
        and bool(np.array_equal(self, other))
    )


def _cdiv(a: int, b: int) -> int:
  return (a + b - 1) // b


def make_rpa_spec(
    name: str,
    *,
    project: str = "qwen3_5_397b",
    num_q_heads: int = 4,
    num_kv_heads: int = 1,
    head_dim: int = 128,
    page_size: int = 256,
    num_pages: int | None = None,
    mode: Literal["decode", "prefill", "chunked_prefill", "mixed"] = "decode",
    num_seqs: int = 1,
    seq_len: int = 4096,
    chunk_size: int = 1024,
    num_decode_seqs: int | None = None,
    decode_seq_len: int | None = None,
    num_prefill_seqs: int | None = None,
    prefill_seq_len: int | None = None,
    query_lens: Sequence[int] | None = None,
    kv_lens: Sequence[int] | int | None = None,
    distribution: Sequence[int] | None = None,
    q_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    kv_cache_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    tags: tuple[arg_spec.Tag, ...] = ("primary", "forward_only"),
) -> arg_spec.ArgSpec:
  """Creates an ArgSpec for Batched RPA with concrete metadata arrays."""
  head_dim_aligned = _cdiv(head_dim, 128) * 128

  if query_lens is not None:
    num_seqs = len(query_lens)
    q_lens_list = list(query_lens)
    if kv_lens is None:
      kv_lens_list = [seq_len] * num_seqs
    elif isinstance(kv_lens, int):
      kv_lens_list = [kv_lens] * num_seqs
    else:
      kv_lens_list = list(kv_lens)
    if distribution is not None:
      distribution_list = list(distribution)
    else:
      num_decode = sum(1 for q in q_lens_list if q == 1)
      distribution_list = [num_decode, num_decode, num_seqs]
  elif mode == "decode":
    q_lens_list = [1] * num_seqs
    kv_lens_list = [seq_len] * num_seqs
    distribution_list = [num_seqs, num_seqs, num_seqs]
  elif mode == "prefill":
    q_lens_list = [seq_len] * num_seqs
    kv_lens_list = [seq_len] * num_seqs
    distribution_list = [0, 0, num_seqs]
  elif mode == "chunked_prefill":
    q_lens_list = [chunk_size] * num_seqs
    kv_lens_list = [seq_len] * num_seqs
    distribution_list = [0, 0, num_seqs]
  elif mode == "mixed":
    n_dec = num_decode_seqs if num_decode_seqs is not None else max(1, num_seqs - 1)
    n_pref = num_prefill_seqs if num_prefill_seqs is not None else 1
    total_seqs = n_dec + n_pref
    d_kv_len = decode_seq_len if decode_seq_len is not None else seq_len
    p_kv_len = (
        prefill_seq_len
        if prefill_seq_len is not None
        else (seq_len - 1024 if seq_len >= 9216 else seq_len)
    )
    q_lens_list = [1] * n_dec + [chunk_size] * n_pref
    kv_lens_list = [d_kv_len] * n_dec + [p_kv_len] * n_pref
    distribution_list = [n_dec, n_dec, total_seqs]
    num_seqs = total_seqs
  else:
    raise ValueError(f"Unknown mode: {mode}")

  total_q_len = sum(q_lens_list)
  cu_q_lens_list = list(np.cumsum([0] + q_lens_list, dtype=np.int32))

  max_kv_len = max(kv_lens_list) if kv_lens_list else seq_len
  pages_per_seq = _cdiv(max_kv_len, page_size)
  total_pages_needed = num_seqs * pages_per_seq
  page_indices_list = list(range(total_pages_needed))
  total_num_pages = max(num_pages or 0, total_pages_needed)

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
  # 5D HEAD_ALONG_SUBLANE layout:
  # [pages, page_size, cdiv(num_kv_heads * 2, packing), packing, head_dim].
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

  # Concrete integer metadata arrays.
  kv_lens_arr = HashableNPArray(np.array(kv_lens_list, dtype=np.int32))
  page_indices_arr = HashableNPArray(np.array(page_indices_list, dtype=np.int32))
  cu_q_lens_arr = HashableNPArray(np.array(cu_q_lens_list, dtype=np.int32))
  distribution_arr = HashableNPArray(np.array(distribution_list, dtype=np.int32))

  return arg_spec.ArgSpec(
      args=dict(
          queries=queries,
          keys=keys,
          values=values,
          kv_cache=kv_cache,
          kv_lens=kv_lens_arr,
          page_indices=page_indices_arr,
          cu_q_lens=cu_q_lens_arr,
          distribution=distribution_arr,
      ),
      project=project,
      name=name,
      tags=tags,
  )


_make_rpa_spec = make_rpa_spec

# Qwen 3.5 397B full-attention layers (HF config.json: num_attention_heads=32,
# num_key_value_heads=2, head_dim=256). KV heads replicate when TP > 2.
_QWEN35_397B_Q_HEADS = 32
_QWEN35_397B_KV_HEADS = 2
_QWEN35_397B_HEAD_DIM = 256
# DP attention width of the serving deployments modeled below.
_DP_SIZE = 8


def _qwen35_397b_spec(
    name: str,
    *,
    tp: int = 1,
    tags: tuple[arg_spec.Tag, ...] = ("primary", "forward_only"),
    **kwargs,
) -> arg_spec.ArgSpec:
  """Qwen 3.5 397B spec with per-chip heads for the given TP degree."""
  return make_rpa_spec(
      name,
      num_q_heads=_QWEN35_397B_Q_HEADS // tp,
      num_kv_heads=max(1, _QWEN35_397B_KV_HEADS // tp),
      head_dim=_QWEN35_397B_HEAD_DIM,
      tags=tags,
      **kwargs,
  )


def _serving_isl8k_specs() -> list[arg_spec.ArgSpec]:
  """ISL=8192 / OSL=1024 serving workloads, Co in 4..256, DP8 attention.

  Each chip serves ceil(Co / 8) sequences with full (TP1) heads.
    Mixed continuous batching: (n - 1) decode (q=1, kv=9216) +
      1 last prefill chunk (q=1024, kv=8192).
    Pure decode (TPOT): n decode sequences (q=1, kv=9216).
    Last prefill chunk (TTFT): n prefill sequences (q=1024, kv=8192).
  """
  specs = []
  for co in (4, 8, 16, 32, 64, 128, 256):
    n = _cdiv(co, _DP_SIZE)
    specs += [
        _qwen35_397b_spec(
            f"qwen35_isl8k_mixed_c{co}_dp8tp1",
            mode="mixed",
            num_seqs=n,
            seq_len=9216,
            decode_seq_len=9216,
            prefill_seq_len=8192,
            chunk_size=1024,
        ),
        _qwen35_397b_spec(
            f"qwen35_isl8k_decode_c{co}_dp8tp1",
            mode="decode",
            num_seqs=n,
            seq_len=9216,
        ),
        _qwen35_397b_spec(
            f"qwen35_isl8k_prefill_c{co}_dp8tp1",
            mode="chunked_prefill",
            num_seqs=n,
            seq_len=8192,
            chunk_size=1024,
        ),
    ]
  return specs


# Long-context multi-turn workloads: 64k-256k context, each turn appends a ~2k
# delta. DP8 attention. Single-layer KV cache budget per chip, by mode: the
# prefill path halts with a DMA BoundsCheck once the KV cache reaches 2 GiB
# (likely an int32 byte-offset overflow; some block sizes already halt at
# exactly 2 GiB), while decode is verified at 4 GiB.
_LONGCTX_DELTA = 2048
_LONGCTX_DECODE_KV_BUDGET_BYTES = 4 * 1024**3
_LONGCTX_PREFILL_KV_BUDGET_BYTES = 2 * 1024**3


def _longctx_specs() -> list[arg_spec.ArgSpec]:
  """Long-context decode and last-chunk delta prefill workloads."""
  kv_bytes_per_token = _QWEN35_397B_KV_HEADS * 2 * _QWEN35_397B_HEAD_DIM * 2
  specs = []
  for ctx in (65536, 131072, 262144):
    for co in (16, 64, 256, 512):
      n = co // _DP_SIZE
      kv_bytes = n * ctx * kv_bytes_per_token
      label = f"c{co}_ctx{ctx // 1024}k_dp8tp1"
      if kv_bytes <= _LONGCTX_DECODE_KV_BUDGET_BYTES:
        specs.append(
            _qwen35_397b_spec(
                f"qwen35_longctx_decode_{label}",
                mode="decode",
                num_seqs=n,
                seq_len=ctx,
            )
        )
      if kv_bytes < _LONGCTX_PREFILL_KV_BUDGET_BYTES:
        specs.append(
            _qwen35_397b_spec(
                f"qwen35_longctx_last_chunk_{label}",
                mode="chunked_prefill",
                num_seqs=n,
                seq_len=ctx,
                chunk_size=_LONGCTX_DELTA,
            )
        )
  return specs


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    # ==========================================================================
    # 1. Qwen 3.5 397B Serving Workloads (ISL=8192, OSL=1024)
    # ==========================================================================
    *_serving_isl8k_specs(),
    # ==========================================================================
    # 2. Plain Qwen 3.5 397B Standard Scaling Workloads
    # ==========================================================================
    *(
        _qwen35_397b_spec(
            f"qwen35_decode_bs{bs}",
            mode="decode",
            num_seqs=bs,
            seq_len=4096,
            tags=("primary", "forward_only")
            + (("ci_tests",) if bs == 16 else ()),
        )
        for bs in (16, 64, 256, 512)
    ),
    *(
        _qwen35_397b_spec(
            f"qwen35_decode_bs{bs}_tp{tp}", tp=tp, mode="decode", num_seqs=bs, seq_len=4096
        )
        for bs in (64, 256)
        for tp in (4, 8)
    ),
    # Prefill Prompt Length Sweeps
    *(
        _qwen35_397b_spec(f"qwen35_prefill_q{q}", mode="prefill", num_seqs=1, seq_len=q)
        for q in (1024, 2048, 4096, 8192)
    ),
    # ==========================================================================
    # 3. Chunked Delta Prefill over a 4k Cached Prefix
    # ==========================================================================
    *(
        _qwen35_397b_spec(
            f"qwen35_chunked_prefill_c{co}" + (f"_tp{tp}" if tp > 1 else ""),
            tp=tp,
            mode="chunked_prefill",
            num_seqs=co,
            chunk_size=1024,
            seq_len=5120,
        )
        # c256 at TP1 is omitted: its KV cache (256 x 5120 tokens, ~2.5 GiB)
        # exceeds the 2 GiB prefill-path limit, and the resulting int32 CHECK
        # in Mosaic lowering crashes the autotuner process.
        for co, tp in ((16, 1), (64, 1), (64, 4), (64, 8))
    ),
    # ==========================================================================
    # 3b. Long-Context Workloads (64k-256k context)
    # ==========================================================================
    *_longctx_specs(),
    # ==========================================================================
    # 4. Qwen 3.5 35B and 397B TP8 Long-Context Decode Workloads
    # ==========================================================================
    make_rpa_spec(
        "qwen35_35b_tp8_decode_64_ctx8192",
        project="qwen3_5_35b",
        mode="decode",
        num_seqs=64,
        seq_len=8192,
        num_q_heads=2,
        num_kv_heads=1,
        tags=("primary", "forward_only"),
    ),
    make_rpa_spec(
        "qwen35_35b_tp8_prefill_2048",
        project="qwen3_5_35b",
        mode="prefill",
        num_seqs=1,
        seq_len=2048,
        num_q_heads=2,
        num_kv_heads=1,
        tags=("primary", "forward_only"),
    ),
    make_rpa_spec(
        "qwen35_tp8_decode_64_ctx8192",
        mode="decode",
        num_seqs=64,
        seq_len=8192,
        num_q_heads=4,
        num_kv_heads=1,
        head_dim=_QWEN35_397B_HEAD_DIM,
        tags=("primary", "forward_only"),
    ),
)
