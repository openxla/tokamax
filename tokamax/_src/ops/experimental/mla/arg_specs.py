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
"""Multi-Head Latent Attention benchmark argument specifications."""

from collections.abc import Sequence
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.experimental.mla import utils


class HashableNPArray(np.ndarray):

  def __new__(cls, input_array):
    return np.asarray(input_array).view(cls)

  def __hash__(self):
    return hash((self.tobytes(), self.shape, self.dtype))


def _make_mla_spec(
    name: str,
    seq_lens: Sequence[tuple[int, int]],
    num_heads: int,
    lkv_dim: int,
    r_dim: int,
    page_size: int,
    q_dtype: jax.typing.DTypeLike,
    kv_dtype: jax.typing.DTypeLike,
    num_pages: int,
) -> arg_spec.ArgSpec:
  """Generates an argument specification for MLA."""
  padded_r_dim = utils.align_to(r_dim, 128)
  padded_lkv_dim = utils.align_to(lkv_dim, 128)
  padded_kv_dim = padded_lkv_dim + padded_r_dim
  packing = utils.get_dtype_packing(kv_dtype)
  q_lens = [s[0] for s in seq_lens]
  kv_lens_list = [s[1] for s in seq_lens]
  total_q_len = sum(q_lens)
  cu_q_lens_list = [0]
  for q_len in q_lens:
    cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)

  max_kv_len = max(kv_lens_list) if kv_lens_list else 0
  pages_per_seq = utils.cdiv(max_kv_len, page_size)

  page_indices_list = []
  page_count = 0
  for kv_len in kv_lens_list:
    num_seq_pages = utils.cdiv(kv_len, page_size)
    indices = list(range(page_count, page_count + num_seq_pages))
    page_indices_list.extend(indices + [-1] * (pages_per_seq - num_seq_pages))
    page_count += num_seq_pages

  total_num_pages = max(num_pages, page_count)

  num_decode_seqs = 0
  for s in seq_lens:
    if s[0] == 1:
      num_decode_seqs += 1
    else:
      break
  distribution_list = [num_decode_seqs, num_decode_seqs, len(seq_lens)]

  ql_nope = jax.ShapeDtypeStruct((total_q_len, num_heads, lkv_dim), q_dtype)
  q_pe = jax.ShapeDtypeStruct((total_q_len, num_heads, r_dim), q_dtype)
  new_kv_c = jax.ShapeDtypeStruct((total_q_len, lkv_dim), kv_dtype)
  new_k_pe = jax.ShapeDtypeStruct((total_q_len, r_dim), kv_dtype)
  cache_kv = jax.ShapeDtypeStruct(
      (total_num_pages, page_size // packing, packing, padded_kv_dim), kv_dtype
  )

  kv_lens = HashableNPArray(np.array(kv_lens_list, dtype=np.int32))
  page_indices = HashableNPArray(np.array(page_indices_list, dtype=np.int32))
  cu_q_lens = HashableNPArray(np.array(cu_q_lens_list, dtype=np.int32))
  distribution = HashableNPArray(np.array(distribution_list, dtype=np.int32))

  return arg_spec.ArgSpec(
      args=dict(
          ql_nope=ql_nope,
          q_pe=q_pe,
          new_kv_c=new_kv_c,
          new_k_pe=new_k_pe,
          cache_kv=cache_kv,
          kv_lens=kv_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
      ),
      project='deepseek_v3',
      name=name,
      tags=('primary', 'forward_only', 'ci_tests'),
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_mla_spec(
        name='decode_bf16',
        seq_lens=[(1, 8192)] * 3,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        num_pages=128,
    ),
    _make_mla_spec(
        name='prefill_bf16',
        seq_lens=[(2048, 2048)] * 2,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        num_pages=128,
    ),
    _make_mla_spec(
        name='decode_f8',
        seq_lens=[(1, 8192)] * 128,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
    ),
)
