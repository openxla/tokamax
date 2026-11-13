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
"""Sparse MLA benchmark argument specifications."""

from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct

_HEAD_DIM = 512
_COMPRESSION_RATIO = 4


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


def _make_argspec(
    *,
    kv_len_val: int,
    batch_size: int = 1,
    q_len: int = 1,
    page_size: int = 1024,
    csa_topk: int = 2048,
    num_q_heads: int = 64,
    actual_batch_size: int | None = None,
    head_dim: int = _HEAD_DIM,
    q_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    primary: bool = False,
    name: str | None = None,
) -> arg_spec.ArgSpec:
  """Makes a single argspec for Sparse MLA."""
  if name is None:
    mode = "decode" if q_len == 1 else "prefill"
    act_str = (
        f"_act{actual_batch_size}" if actual_batch_size is not None else ""
    )
    page_str = f"_p{page_size}" if page_size != 1024 else ""
    name = (
        f"{mode}_b{batch_size}{act_str}_q{q_len}_kv{kv_len_val}"
        f"{page_str}_topk{csa_topk}_h{num_q_heads}"
    )
  tags: tuple[arg_spec.Tag, ...] = (
      ("primary", "forward_only") if primary else ("forward_only",)
  )
  compressed_page_size = page_size // _COMPRESSION_RATIO
  kv_len_after_compression = kv_len_val // _COMPRESSION_RATIO
  num_tokens = batch_size * q_len
  cu_q_lens = np.arange(0, num_tokens + 1, q_len, dtype=np.int32)

  pages_per_seq = _cdiv(kv_len_after_compression, compressed_page_size) + 2
  total_pages = batch_size * pages_per_seq
  page_indices = np.arange(total_pages, dtype=np.int32)

  effective_batch_size = (
      batch_size if actual_batch_size is None else actual_batch_size
  )
  num_decode_seqs = effective_batch_size if q_len == 1 else 0
  distribution = np.array(
      [num_decode_seqs, num_decode_seqs, effective_batch_size], dtype=np.int32
  )

  args = {
      "q": ShapeDtype((num_tokens, num_q_heads, head_dim), q_dtype),
      "cache_kv_nope": ShapeDtype(
          (total_pages, compressed_page_size, 128), jnp.int32
      ),
      "cache_kv_rope": ShapeDtype(
          (total_pages, compressed_page_size // 4, 128), jnp.int32
      ),
      "topk_indices": numerics.RangedArrayInitializer(
          (num_tokens, csa_topk), jnp.int32, 0, kv_len_after_compression
      ),
      "page_indices": _int32_array(page_indices),
      "cu_q_lens": _int32_array(cu_q_lens),
      "distribution": _int32_array(distribution),
      "attention_sinks": ShapeDtype((num_q_heads,), jnp.float32),
      "swa_accumution": ShapeDtype(
          (num_tokens, num_q_heads, head_dim), q_dtype
      ),
      "swa_l": ShapeDtype((num_tokens, num_q_heads), jnp.float32),
      "swa_m": ShapeDtype((num_tokens, num_q_heads), jnp.float32),
      "sm_scale": 1.0,
  }

  return arg_spec.ArgSpec(
      args=args,
      project="inference",
      name=name,
      tags=tags,
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    # DeepSeek-V4 shapes
    _make_argspec(
        batch_size=256,
        kv_len_val=9216,
        csa_topk=1024,
        num_q_heads=128,
        primary=True,
    ),
    _make_argspec(
        q_len=1024, kv_len_val=1024, csa_topk=1024, num_q_heads=128
    ),
    _make_argspec(
        q_len=1024,
        kv_len_val=8192,
        csa_topk=1024,
        num_q_heads=128,
        primary=True,
    ),
    _make_argspec(
        batch_size=64, kv_len_val=1048576, csa_topk=1024, num_q_heads=128
    ),
    _make_argspec(
        q_len=1024, kv_len_val=1048576, csa_topk=1024, num_q_heads=128
    ),
    _make_argspec(
        batch_size=256, kv_len_val=9216, csa_topk=512, primary=True
    ),
    _make_argspec(q_len=1024, kv_len_val=1024, csa_topk=512),
    _make_argspec(q_len=1024, kv_len_val=8192, csa_topk=512, primary=True),
    # GLM-5.2 decode shapes
    _make_argspec(batch_size=64, kv_len_val=8192),
    _make_argspec(batch_size=128, kv_len_val=8192),
    _make_argspec(batch_size=64, kv_len_val=9216),
    _make_argspec(batch_size=128, kv_len_val=9216, primary=True),
    _make_argspec(batch_size=1, kv_len_val=262144),
    _make_argspec(batch_size=16, kv_len_val=262144),
    _make_argspec(batch_size=32, kv_len_val=262144),
    _make_argspec(batch_size=16, actual_batch_size=8, kv_len_val=131072),
    _make_argspec(batch_size=64, actual_batch_size=40, kv_len_val=131072),
    # GLM-5.2 prefill shapes
    _make_argspec(q_len=1024, kv_len_val=1024),
    _make_argspec(q_len=1024, kv_len_val=8192, primary=True),
    _make_argspec(q_len=2048, kv_len_val=2048),
    _make_argspec(q_len=2048, kv_len_val=8192),
    _make_argspec(q_len=1024, kv_len_val=1024, page_size=256),
    _make_argspec(q_len=1024, kv_len_val=8192, page_size=256),
    _make_argspec(q_len=1024, kv_len_val=131072),
    _make_argspec(q_len=4096, kv_len_val=131072),
    _make_argspec(q_len=1024, kv_len_val=262144),
)
