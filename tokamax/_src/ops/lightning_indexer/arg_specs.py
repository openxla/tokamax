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
"""Lightning Indexer (StreamIndex Top-K) benchmark argument specifications."""

from collections.abc import Sequence
import random
from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.lightning_indexer.kernel import config as kernel_config

ShapeDtype = jax.ShapeDtypeStruct
KVLayout = kernel_config.KVLayout

_KV_PACKING = 4


class _HashableNPArray(np.ndarray):
  """Numpy array that can be retained as a concrete argument in an ArgSpec."""

  def __new__(cls, input_array: Any) -> "_HashableNPArray":
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


def _int32_array(x: Any) -> _HashableNPArray:
  return _HashableNPArray(np.asarray(x, dtype=np.int32))


def _cdiv(a: int, b: int) -> int:
  return (a + b - 1) // b


def _align_to(x: int, a: int) -> int:
  return _cdiv(x, a) * a


def _make_argspec(
    *,
    name: str,
    batch_size: int = 1,
    num_q_heads: int = 64,
    head_dim: int = 128,
    k: int = 1024,
    q_len: int = 1,
    kv_len: int = 1024,
    t_list: Sequence[int] | None = None,
    s_list: Sequence[int] | None = None,
    shuffle_s_list: bool = False,
    page_size: int = 1024,
    shuffle_page_indices: bool = False,
    page_indices: Sequence[int] | np.ndarray | None = None,
    compression_ratio: int = 4,
    kv_layout: KVLayout = KVLayout.SEQ_ALONG_LANE,
    cp_size: int = 1,
    cp_rank: int = 0,
    interleave_size: int = 1,
    return_scores: bool = False,
    uniform_q_len: int | None = None,
    q_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    indexer_weights_dtype: jax.typing.DTypeLike = jnp.float32,
    tags: tuple[arg_spec.Tag, ...] = ("forward_only",),
) -> arg_spec.ArgSpec:
  """Creates an argspec for Lightning Indexer."""
  if batch_size <= 0:
    raise ValueError(f"batch_size must be positive, got {batch_size}.")
  if t_list is None:
    t_list = [q_len] * batch_size
  else:
    t_list = [int(x) for x in t_list]
  if s_list is None:
    s_list = [kv_len] * batch_size
  else:
    s_list = [int(x) for x in s_list]

  if len(t_list) != batch_size or len(s_list) != batch_size:
    raise ValueError(
        f"t_list ({len(t_list)}) and s_list ({len(s_list)}) must match"
        f" batch_size ({batch_size})."
    )
  if shuffle_s_list:
    random.Random(42).shuffle(s_list)

  if (
      compression_ratio < 1
      or (compression_ratio & (compression_ratio - 1)) != 0
  ):
    raise ValueError("compression_ratio must be a power of 2.")
  if page_size <= 0 or page_size % compression_ratio != 0:
    raise ValueError(
        f"page_size ({page_size}) must be a positive multiple of"
        f" compression_ratio ({compression_ratio})."
    )
  if cp_size < 1:
    raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
  if not 0 <= cp_rank < cp_size:
    raise ValueError(f"cp_rank ({cp_rank}) must be in [0, {cp_size}).")
  if cp_size > 1 and interleave_size % compression_ratio != 0:
    raise ValueError(
        f"interleave_size ({interleave_size}) must be a multiple of"
        f" compression_ratio ({compression_ratio})."
    )

  compressed_page_size = page_size // compression_ratio
  if kv_layout == KVLayout.SEQ_ALONG_LANE and compressed_page_size % 128 != 0:
    raise ValueError(
        "SEQ_ALONG_LANE requires compressed_page_size"
        f" ({compressed_page_size}) to be a multiple of 128."
    )
  if (
      kv_layout == KVLayout.HEAD_ALONG_SUBLANE
      and compressed_page_size % _KV_PACKING != 0
  ):
    raise ValueError(
        "HEAD_ALONG_SUBLANE requires compressed_page_size"
        f" ({compressed_page_size}) to be a multiple of {_KV_PACKING}."
    )

  if cp_size == 1:
    required_pages_per_seq = max(1, _cdiv(max(s_list), page_size))
  else:
    interleave_c = interleave_size // compression_ratio
    cycle_c = cp_size * interleave_c
    max_global_c = _cdiv(max(s_list), compression_ratio)
    full_cycles, rem_c = divmod(max_global_c, cycle_c)
    max_local_c = full_cycles * interleave_c + min(rem_c, interleave_c)
    required_pages_per_seq = max(1, _cdiv(max_local_c, compressed_page_size))

  if page_indices is not None:
    if shuffle_page_indices:
      raise ValueError(
          "Cannot specify both page_indices and shuffle_page_indices."
      )
    page_indices_arr = np.asarray(page_indices, dtype=np.int32)
    if page_indices_arr.ndim != 1 or page_indices_arr.shape[0] % batch_size:
      raise ValueError(
          f"page_indices shape {page_indices_arr.shape} must be 1D and"
          f" divisible by batch_size ({batch_size})."
      )
    max_pages_per_seq = page_indices_arr.shape[0] // batch_size
    if max_pages_per_seq < required_pages_per_seq:
      raise ValueError(
          f"page_indices provides {max_pages_per_seq} pages per sequence,"
          f" need at least {required_pages_per_seq}."
      )
    if np.any(page_indices_arr < 0):
      raise ValueError("page_indices entries must be non-negative.")
    total_pages = max(
        batch_size * max_pages_per_seq, int(np.max(page_indices_arr)) + 1
    )
  else:
    max_pages_per_seq = required_pages_per_seq
    total_pages = batch_size * max_pages_per_seq
    if shuffle_page_indices:
      page_indices_arr = (
          np.random.default_rng(42).permutation(total_pages).astype(np.int32)
      )
    else:
      page_indices_arr = np.arange(total_pages, dtype=np.int32)

  num_tokens = sum(t_list)
  padded_head_dim = _align_to(head_dim, 128)

  if kv_layout == KVLayout.SEQ_ALONG_LANE:
    cache_shape = (
        total_pages,
        padded_head_dim // _KV_PACKING + 1,
        _KV_PACKING,
        compressed_page_size,
    )
  else:
    width = _align_to(padded_head_dim + padded_head_dim // 128, 128)
    cache_shape = (
        total_pages,
        compressed_page_size // _KV_PACKING,
        _KV_PACKING,
        width,
    )

  num_decodes = 0
  while num_decodes < batch_size and t_list[num_decodes] == 1:
    num_decodes += 1

  uniform_end = num_decodes
  if uniform_q_len is not None:
    if uniform_q_len <= 0:
      raise ValueError(
          f"uniform_q_len must be positive, got {uniform_q_len}."
      )
    while uniform_end < batch_size and t_list[uniform_end] == uniform_q_len:
      uniform_end += 1

  cu_q_lens = np.concatenate([[0], np.cumsum(t_list)]).astype(np.int32)
  distribution = np.array([num_decodes, uniform_end, batch_size], np.int32)

  args: dict[str, Any] = {
      "q": ShapeDtype((num_tokens, num_q_heads, head_dim), q_dtype),
      "indexer_weights": ShapeDtype(
          (num_tokens, num_q_heads), indexer_weights_dtype
      ),
      "cache_kv": ShapeDtype(cache_shape, jnp.uint8),
      "seq_lens": _int32_array(s_list),
      "page_indices": _int32_array(page_indices_arr),
      "cu_q_lens": _int32_array(cu_q_lens),
      "distribution": _int32_array(distribution),
      "k": k,
      "compression_ratio": compression_ratio,
      "kv_layout": kv_layout,
  }
  if cp_size != 1:
    args["cp_size"] = cp_size
  if cp_rank != 0:
    args["cp_rank"] = cp_rank
  if interleave_size != 1:
    args["interleave_size"] = interleave_size
  if return_scores:
    args["return_scores"] = True
  if uniform_q_len is not None:
    args["uniform_q_len"] = uniform_q_len

  return arg_spec.ArgSpec(
      args=args,
      project="inference",
      name=name,
      tags=tags,
  )


_COMPRESSED_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_argspec(
        name="decode_b1_q1_kv1048576_k1024",
        batch_size=1,
        k=1024,
        q_len=1,
        kv_len=1048576,
        page_size=1024,
    ),
    _make_argspec(
        name="decode_b4_q1_kv1048576_k1024",
        batch_size=4,
        k=1024,
        q_len=1,
        kv_len=1048576,
        page_size=1024,
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv1048576_k1024",
        batch_size=1,
        k=1024,
        q_len=1024,
        kv_len=1048576,
        page_size=1024,
    ),
    _make_argspec(
        name="decode_b256_q1_kv9216_k512",
        batch_size=256,
        k=512,
        q_len=1,
        kv_len=9216,
        page_size=1024,
        tags=("primary", "forward_only"),
    ),
    _make_argspec(
        name="decode_b256_q1_kv9216_k1024",
        batch_size=256,
        k=1024,
        q_len=1,
        kv_len=9216,
        page_size=1024,
        tags=("primary", "forward_only"),
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv8192_k512",
        batch_size=1,
        k=512,
        q_len=1024,
        kv_len=8192,
        page_size=1024,
        tags=("primary", "forward_only"),
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv8192_k1024",
        batch_size=1,
        k=1024,
        q_len=1024,
        kv_len=8192,
        page_size=1024,
        tags=("primary", "forward_only"),
    ),
    _make_argspec(
        name="ragged_prefill_b4_k512",
        batch_size=4,
        k=512,
        t_list=[512, 1024, 256, 768],
        s_list=[2048, 8192, 4096, 6144],
        page_size=1024,
    ),
    _make_argspec(
        name="ragged_decode_b256_k512",
        batch_size=256,
        k=512,
        q_len=1,
        s_list=[1024] * 64 + [3072] * 64 + [6144] * 64 + [9216] * 64,
        page_size=1024,
    ),
    _make_argspec(
        name="ragged_decode_b256_k512_shuffled",
        batch_size=256,
        k=512,
        q_len=1,
        s_list=[1024] * 64 + [3072] * 64 + [6144] * 64 + [9216] * 64,
        shuffle_s_list=True,
        page_size=1024,
    ),
    _make_argspec(
        name="decode_b256_q1_kv9216_k512_has",
        batch_size=256,
        k=512,
        q_len=1,
        kv_len=9216,
        page_size=1024,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _make_argspec(
        name="decode_b256_q1_kv9216_k1024_has",
        batch_size=256,
        k=1024,
        q_len=1,
        kv_len=9216,
        page_size=1024,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv8192_k512_has",
        batch_size=1,
        k=512,
        q_len=1024,
        kv_len=8192,
        page_size=1024,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv8192_k1024_has",
        batch_size=1,
        k=1024,
        q_len=1024,
        kv_len=8192,
        page_size=1024,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv1024_k512",
        batch_size=1,
        k=512,
        q_len=1024,
        kv_len=1024,
        page_size=1024,
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv1024_k1024",
        batch_size=1,
        k=1024,
        q_len=1024,
        kv_len=1024,
        page_size=1024,
    ),
    _make_argspec(
        name="mixed_b9_d8_p1024_kv8192_k512",
        batch_size=9,
        k=512,
        t_list=[1] * 8 + [1024],
        kv_len=8192,
        page_size=1024,
    ),
    _make_argspec(
        name="mixed_b10_d8_p512_kv9216_k1024",
        batch_size=10,
        k=1024,
        t_list=[1] * 8 + [512, 512],
        s_list=[9216] * 8 + [4096, 8192],
        page_size=1024,
    ),
    _make_argspec(
        name="spec_verify_b256_q2_kv8192_k512",
        batch_size=256,
        k=512,
        q_len=2,
        kv_len=8192,
        page_size=1024,
        uniform_q_len=2,
    ),
    _make_argspec(
        name="spec_verify_b256_q4_kv8192_k512",
        batch_size=256,
        k=512,
        q_len=4,
        kv_len=8192,
        page_size=1024,
        uniform_q_len=4,
    ),
    _make_argspec(
        name="spec_verify_b256_q8_kv8192_k512",
        batch_size=256,
        k=512,
        q_len=8,
        kv_len=8192,
        page_size=1024,
        uniform_q_len=8,
    ),
    _make_argspec(
        name="spec_verify_b256_q16_kv8192_k512",
        batch_size=256,
        k=512,
        q_len=16,
        kv_len=8192,
        page_size=1024,
        uniform_q_len=16,
    ),
    _make_argspec(
        name="mixed_spec_verify_b12_q6_kv9216_k512",
        batch_size=12,
        k=512,
        t_list=[1] * 4 + [6] * 4 + [256, 512, 256, 512],
        s_list=[9216] * 8 + [2048, 4096, 6144, 8192],
        page_size=1024,
        uniform_q_len=6,
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv32768_k1024_cp4",
        batch_size=1,
        k=1024,
        q_len=1024,
        kv_len=32768,
        page_size=1024,
        cp_size=4,
        interleave_size=4,
        return_scores=True,
    ),
    _make_argspec(
        name="decode_b64_q1_kv32768_k512_cp4",
        batch_size=64,
        k=512,
        q_len=1,
        kv_len=32768,
        page_size=1024,
        cp_size=4,
        interleave_size=4,
        return_scores=True,
    ),
    _make_argspec(
        name="decode_b64_q1_kv1048576_k1024_cp8",
        batch_size=64,
        k=1024,
        q_len=1,
        kv_len=1048576,
        page_size=1024,
        cp_size=8,
        interleave_size=4,
        return_scores=True,
    ),
    _make_argspec(
        name="decode_b256_q1_kv9216_k512_h16",
        batch_size=256,
        num_q_heads=16,
        k=512,
        q_len=1,
        kv_len=9216,
        page_size=1024,
    ),
    _make_argspec(
        name="decode_b256_q1_kv9216_k512_h8",
        batch_size=256,
        num_q_heads=8,
        k=512,
        q_len=1,
        kv_len=9216,
        page_size=1024,
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv8192_k512_h16",
        batch_size=1,
        num_q_heads=16,
        k=512,
        q_len=1024,
        kv_len=8192,
        page_size=1024,
    ),
    _make_argspec(
        name="decode_b256_q1_kv9216_k512_bf16q",
        batch_size=256,
        k=512,
        q_len=1,
        kv_len=9216,
        page_size=1024,
        q_dtype=jnp.bfloat16,
        indexer_weights_dtype=jnp.bfloat16,
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv8192_k512_bf16q",
        batch_size=1,
        k=512,
        q_len=1024,
        kv_len=8192,
        page_size=1024,
        q_dtype=jnp.bfloat16,
        indexer_weights_dtype=jnp.bfloat16,
    ),
    _make_argspec(
        name="decode_b256_q1_kv9216_k512_p512",
        batch_size=256,
        k=512,
        q_len=1,
        kv_len=9216,
        page_size=512,
    ),
    _make_argspec(
        name="decode_b256_q1_kv9216_k512_fragmented_pages",
        batch_size=256,
        k=512,
        q_len=1,
        kv_len=9216,
        page_size=1024,
        shuffle_page_indices=True,
    ),
)

_UNCOMPRESSED_COMMON = dict(
    num_q_heads=32,
    head_dim=128,
    k=2048,
    compression_ratio=1,
    page_size=1024,
    q_dtype=jnp.float8_e4m3fn,
    indexer_weights_dtype=jnp.bfloat16,
)


def _make_layout_pair(
    *,
    base_name: str,
    batch_size: int,
    q_len: int,
    kv_len: int,
    sal_tags: tuple[arg_spec.Tag, ...] = ("forward_only",),
) -> tuple[arg_spec.ArgSpec, arg_spec.ArgSpec]:
  """Creates HEAD_ALONG_SUBLANE and SEQ_ALONG_LANE specs for a given shape."""
  return (
      _make_argspec(
          name=f"{base_name}_has",
          batch_size=batch_size,
          q_len=q_len,
          kv_len=kv_len,
          kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
          **_UNCOMPRESSED_COMMON,
      ),
      _make_argspec(
          name=f"{base_name}_sal",
          batch_size=batch_size,
          q_len=q_len,
          kv_len=kv_len,
          kv_layout=KVLayout.SEQ_ALONG_LANE,
          tags=sal_tags,
          **_UNCOMPRESSED_COMMON,
      ),
  )


_UNCOMPRESSED_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    *_make_layout_pair(
        base_name="decode_b64_q1_kv8192_k2048",
        batch_size=64,
        q_len=1,
        kv_len=8192,
    ),
    *_make_layout_pair(
        base_name="decode_b64_q1_kv9216_k2048",
        batch_size=64,
        q_len=1,
        kv_len=9216,
        sal_tags=("primary", "forward_only"),
    ),
    *_make_layout_pair(
        base_name="decode_b128_q1_kv8192_k2048",
        batch_size=128,
        q_len=1,
        kv_len=8192,
    ),
    *_make_layout_pair(
        base_name="decode_b128_q1_kv9216_k2048",
        batch_size=128,
        q_len=1,
        kv_len=9216,
        sal_tags=("primary", "forward_only"),
    ),
    *_make_layout_pair(
        base_name="decode_b16_q1_kv262144_k2048",
        batch_size=16,
        q_len=1,
        kv_len=262144,
    ),
    *_make_layout_pair(
        base_name="decode_b32_q1_kv262144_k2048",
        batch_size=32,
        q_len=1,
        kv_len=262144,
    ),
    *_make_layout_pair(
        base_name="prefill_b1_q1024_kv1024_k2048",
        batch_size=1,
        q_len=1024,
        kv_len=1024,
    ),
    *_make_layout_pair(
        base_name="prefill_b1_q1024_kv8192_k2048",
        batch_size=1,
        q_len=1024,
        kv_len=8192,
        sal_tags=("primary", "forward_only"),
    ),
    *_make_layout_pair(
        base_name="prefill_b1_q1024_kv262144_k2048",
        batch_size=1,
        q_len=1024,
        kv_len=262144,
    ),
)

ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _COMPRESSED_SPECS + _UNCOMPRESSED_SPECS
)
