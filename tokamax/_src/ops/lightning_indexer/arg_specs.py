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
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.lightning_indexer.kernel import config as kernel_config

ShapeDtype = jax.ShapeDtypeStruct
KVLayout = kernel_config.KVLayout

_KV_PACKING = 4


class _HashableNPArray(np.ndarray):

  def __new__(cls, input_array):
    return np.asarray(input_array).view(cls)

  def __hash__(self):
    return hash((self.tobytes(), self.shape, self.dtype))


def _int32_array(x) -> _HashableNPArray:
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
    page_size: int = 1024,
    compression_ratio: int = 4,
    kv_layout: KVLayout = KVLayout.SEQ_ALONG_LANE,
    q_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    indexer_weights_dtype: jax.typing.DTypeLike = jnp.float32,
    tags: tuple[arg_spec.Tag, ...] = ("forward_only",),
) -> arg_spec.ArgSpec:
  """Creates an argspec for Lightning Indexer."""
  if t_list is None:
    t_list = [q_len] * batch_size
  else:
    t_list = list(t_list)
  if s_list is None:
    s_list = [kv_len] * batch_size
  else:
    s_list = list(s_list)

  if len(t_list) != batch_size or len(s_list) != batch_size:
    raise ValueError(
        f"t_list ({len(t_list)}) and s_list ({len(s_list)}) must match"
        f" batch_size ({batch_size})."
    )

  max_pages_per_seq = _cdiv(max(s_list), page_size)
  compressed_page_size = page_size // compression_ratio
  num_tokens = sum(t_list)
  total_pages = batch_size * max_pages_per_seq

  if kv_layout == KVLayout.SEQ_ALONG_LANE:
    cache_shape = (
        total_pages,
        (head_dim + _KV_PACKING) // _KV_PACKING,
        _KV_PACKING,
        compressed_page_size,
    )
  else:
    width = _align_to(head_dim + 1, 128)
    cache_shape = (
        total_pages,
        compressed_page_size // _KV_PACKING,
        _KV_PACKING,
        width,
    )

  num_decodes = 0
  while num_decodes < batch_size and t_list[num_decodes] == 1:
    num_decodes += 1

  cu_q_lens = np.concatenate([[0], np.cumsum(t_list)]).astype(np.int32)
  distribution = np.array([num_decodes, num_decodes, batch_size], np.int32)

  return arg_spec.ArgSpec(
      args={
          "q": ShapeDtype((num_tokens, num_q_heads, head_dim), q_dtype),
          "indexer_weights": ShapeDtype(
              (num_tokens, num_q_heads), indexer_weights_dtype
          ),
          "cache_kv": ShapeDtype(cache_shape, jnp.uint8),
          "seq_lens": _int32_array(s_list),
          "page_indices": _int32_array(np.arange(total_pages, dtype=np.int32)),
          "cu_q_lens": _int32_array(cu_q_lens),
          "distribution": _int32_array(distribution),
          "k": k,
          "compression_ratio": compression_ratio,
          "kv_layout": kv_layout,
      },
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
