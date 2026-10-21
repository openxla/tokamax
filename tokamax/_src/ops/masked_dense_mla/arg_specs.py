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
"""Masked-Dense MLA benchmark argument specifications."""

from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct

_NUM_Q_HEADS = 64
_HEAD_DIM = 576
_CSA_TOPK = 2048


class _HashableNPArray(np.ndarray):
  """Hashable numpy array for use as an ArgSpec argument."""

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
    batch_size: int,
    q_len: int,
    kv_len_val: int,
    page_size: int = 1024,
    actual_batch_size: int | None = None,
    num_q_heads: int = _NUM_Q_HEADS,
    head_dim: int = _HEAD_DIM,
    csa_topk: int = _CSA_TOPK,
    q_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    tags: tuple[arg_spec.Tag, ...] = ("forward_only",),
) -> arg_spec.ArgSpec:
  """Makes an argspec for Masked-Dense MLA."""
  if actual_batch_size is None:
    actual_batch_size = batch_size
  if not 1 <= actual_batch_size <= batch_size:
    raise ValueError(
        f"actual_batch_size ({actual_batch_size}) must be in [1, {batch_size}]."
    )

  num_tokens = batch_size * q_len
  cu_q_lens = np.arange(0, num_tokens + 1, q_len, dtype=np.int32)

  pages_per_seq = _align_to(_cdiv(kv_len_val, page_size) + 2, 16)
  total_pages = batch_size * pages_per_seq
  page_indices = np.arange(total_pages, dtype=np.int32)

  kv_lens = np.full((batch_size,), kv_len_val, dtype=np.int32)
  num_decode_seqs = actual_batch_size if q_len == 1 else 0
  distribution = np.array(
      [num_decode_seqs, num_decode_seqs, actual_batch_size], dtype=np.int32
  )

  args = {
      "q": ShapeDtype((num_tokens, num_q_heads, head_dim), q_dtype),
      "cache_kv_nope": ShapeDtype((total_pages, page_size, 4, 128), jnp.uint8),
      "cache_kv_rope": ShapeDtype(
          (total_pages, page_size // 4, 4, 128), jnp.uint8
      ),
      "kv_lens": _int32_array(kv_lens),
      "topk_indices": numerics.RangedArrayInitializer(
          (num_tokens, csa_topk), jnp.int32, 0, kv_len_val
      ),
      "page_indices": _int32_array(page_indices),
      "cu_q_lens": _int32_array(cu_q_lens),
      "distribution": _int32_array(distribution),
      "sm_scale": 1.0,
      "k_scale": 1.0,
  }

  return arg_spec.ArgSpec(
      args=args,
      project="inference",
      name=name,
      tags=tags,
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_argspec(
        name="decode_b64_q1_kv8192_p1024",
        batch_size=64,
        q_len=1,
        kv_len_val=8192,
        page_size=1024,
    ),
    _make_argspec(
        name="decode_b128_q1_kv8192_p1024",
        batch_size=128,
        q_len=1,
        kv_len_val=8192,
        page_size=1024,
    ),
    _make_argspec(
        name="decode_b64_q1_kv9216_p1024",
        batch_size=64,
        q_len=1,
        kv_len_val=9216,
        page_size=1024,
        tags=("primary", "forward_only"),
    ),
    _make_argspec(
        name="decode_b128_q1_kv9216_p1024",
        batch_size=128,
        q_len=1,
        kv_len_val=9216,
        page_size=1024,
        tags=("primary", "forward_only"),
    ),
    _make_argspec(
        name="decode_b1_q1_kv262144_p1024",
        batch_size=1,
        q_len=1,
        kv_len_val=262144,
        page_size=1024,
    ),
    _make_argspec(
        name="decode_b16_q1_kv262144_p1024",
        batch_size=16,
        q_len=1,
        kv_len_val=262144,
        page_size=1024,
    ),
    _make_argspec(
        name="decode_b32_q1_kv262144_p1024",
        batch_size=32,
        q_len=1,
        kv_len_val=262144,
        page_size=1024,
    ),
    _make_argspec(
        name="decode_b16_act8_q1_kv131072_p1024",
        batch_size=16,
        actual_batch_size=8,
        q_len=1,
        kv_len_val=131072,
        page_size=1024,
    ),
    _make_argspec(
        name="decode_b64_act40_q1_kv131072_p1024",
        batch_size=64,
        actual_batch_size=40,
        q_len=1,
        kv_len_val=131072,
        page_size=1024,
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv1024_p1024",
        batch_size=1,
        q_len=1024,
        kv_len_val=1024,
        page_size=1024,
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv8192_p1024",
        batch_size=1,
        q_len=1024,
        kv_len_val=8192,
        page_size=1024,
        tags=("primary", "forward_only"),
    ),
    _make_argspec(
        name="prefill_b1_q2048_kv2048_p1024",
        batch_size=1,
        q_len=2048,
        kv_len_val=2048,
        page_size=1024,
    ),
    _make_argspec(
        name="prefill_b1_q2048_kv8192_p1024",
        batch_size=1,
        q_len=2048,
        kv_len_val=8192,
        page_size=1024,
        tags=("primary", "forward_only"),
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv1024_p256",
        batch_size=1,
        q_len=1024,
        kv_len_val=1024,
        page_size=256,
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv8192_p256",
        batch_size=1,
        q_len=1024,
        kv_len_val=8192,
        page_size=256,
    ),
    _make_argspec(
        name="prefill_b1_q1024_kv262144_p1024",
        batch_size=1,
        q_len=1024,
        kv_len_val=262144,
        page_size=1024,
    ),
)
