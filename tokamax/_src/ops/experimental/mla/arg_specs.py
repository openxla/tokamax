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
from tokamax._src.ops.experimental.mla import base
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


# Token-major (v1) layout, page 256.
V1_ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
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


# ------------------------------------------------------------------------------
# MLA v3 (`layout=HEAD_MAJOR_SEQ_ALONG_LANE`): DeepSeek serving, per device.
# ------------------------------------------------------------------------------

# Fixed per-device serving shapes: 128 heads, 32 sequence slots, 1024-token
# pages (14 per slot, for a 13824-token max model length), fp8 queries and KV.
_V3_NUM_HEADS = 128
_V3_LKV_DIM = 512
_V3_R_DIM = 64
_V3_PAGE_SIZE = 1024
_V3_MAX_NUM_SEQS = 32
_V3_PAGES_PER_SEQ = 14
_V3_DTYPE = jnp.float8_e4m3fn


def _make_mla_v3_spec(
    name: str,
    num_tokens: int,
    seq_lens: Sequence[tuple[int, int]],
    distribution: tuple[int, int, int],
) -> arg_spec.ArgSpec:
  """A per-device MLA v3 call as DeepSeek serving makes it.

  Args:
    name: The spec name.
    num_tokens: The padded token bucket.
    seq_lens: `(q_len, kv_len)` of the real sequences; `kv_len` includes the new
      tokens. The remaining slots up to `_V3_MAX_NUM_SEQS` are padding.
    distribution: `[d, p, n]`; vLLM always writes `p == d`.

  Returns:
    The arg spec.
  """
  padding = [(0, 0)] * (_V3_MAX_NUM_SEQS - len(seq_lens))
  kv_lens_list, cu_q_lens_list, page_indices_list = [], [0], []
  page_count = 0
  for q_len, kv_len in list(seq_lens) + padding:
    kv_lens_list.append(kv_len)
    cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)
    num_seq_pages = utils.cdiv(kv_len, _V3_PAGE_SIZE)
    page_indices_list.extend(
        list(range(page_count, page_count + num_seq_pages))
        + [-1] * (_V3_PAGES_PER_SEQ - num_seq_pages)
    )
    page_count += num_seq_pages
  assert cu_q_lens_list[-1] <= num_tokens, name

  kv_dim = utils.align_to(_V3_LKV_DIM, 128) + utils.align_to(_V3_R_DIM, 128)
  sds = jax.ShapeDtypeStruct
  return arg_spec.ArgSpec(
      args=dict(
          ql_nope=sds((_V3_NUM_HEADS, num_tokens, _V3_LKV_DIM), _V3_DTYPE),
          q_pe=sds((num_tokens, _V3_NUM_HEADS, _V3_R_DIM), _V3_DTYPE),
          new_kv_c=sds((num_tokens, _V3_LKV_DIM), _V3_DTYPE),
          new_k_pe=sds((num_tokens, _V3_R_DIM), _V3_DTYPE),
          # The whole per-device pool; the page count is normalized out of the
          # autotuning key anyway.
          cache_kv=sds(
              (_V3_MAX_NUM_SEQS * _V3_PAGES_PER_SEQ, kv_dim, _V3_PAGE_SIZE),
              _V3_DTYPE,
          ),
          kv_lens=HashableNPArray(np.array(kv_lens_list, dtype=np.int32)),
          page_indices=HashableNPArray(
              np.array(page_indices_list, dtype=np.int32)
          ),
          cu_q_lens=HashableNPArray(np.array(cu_q_lens_list, dtype=np.int32)),
          distribution=HashableNPArray(np.array(distribution, dtype=np.int32)),
          sm_scale=192**-0.5,
          q_scale=1.0,
          k_scale=1.0,
          v_scale=1.0,
          layout=base.MlaLayout.HEAD_MAJOR_SEQ_ALONG_LANE,
      ),
      project='deepseek_v3',
      name=name,
      tags=('forward_only',),
  )


def _decode(kv_lens: Sequence[int]) -> list[tuple[int, int]]:
  return [(1, kv_len) for kv_len in kv_lens]


# Representative DeepSeek serving batches (random-range ratio 0.5): pure decode
# at three concurrencies, and a mixed batch admitting a new prompt.
V3_ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    # Concurrency 8 and 32.
    _make_mla_v3_spec(
        'mla_v3_decode_rr05_t4',
        num_tokens=4,
        seq_lens=_decode([3073, 7520]),
        distribution=(2, 2, 2),
    ),
    # Concurrency 64.
    _make_mla_v3_spec(
        'mla_v3_decode_rr05_t8',
        num_tokens=8,
        seq_lens=_decode([1537, 2561, 3585, 4609, 5650, 6837, 8296, 10417]),
        distribution=(8, 8, 8),
    ),
    # Concurrency 256.
    _make_mla_v3_spec(
        'mla_v3_decode_rr05_t32',
        num_tokens=32,
        seq_lens=_decode([
            1153,
            1409,
            1665,
            1921,
            2177,
            2433,
            2689,
            2945,
            3201,
            3457,
            3713,
            3969,
            4225,
            4481,
            4737,
            4993,
            5250,
            5514,
            5788,
            6072,
            6368,
            6677,
            7001,
            7342,
            7704,
            8092,
            8510,
            8969,
            9482,
            10075,
            10805,
            11865,
        ]),
        distribution=(32, 32, 32),
    ),
    # 31 decodes plus a new prompt's 768-token first chunk.
    _make_mla_v3_spec(
        'mla_v3_mixed_rr05_t1024',
        num_tokens=1024,
        seq_lens=_decode([
            1157,
            1421,
            1686,
            1950,
            2214,
            2478,
            2743,
            3007,
            3271,
            3535,
            3800,
            4064,
            4328,
            4592,
            4857,
            5121,
            5390,
            5668,
            5956,
            6257,
            6571,
            6900,
            7247,
            7615,
            8008,
            8433,
            8899,
            9420,
            10023,
            10765,
            11842,
        ])
        + [(768, 768)],
        distribution=(31, 31, 32),
    ),
)
ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = V1_ARG_SPECS + V3_ARG_SPECS
