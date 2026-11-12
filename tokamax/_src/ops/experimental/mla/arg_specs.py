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


# ------------------------------------------------------------------------------
# MLA: DeepSeek serving, per device.
# ------------------------------------------------------------------------------

# Fixed per-device serving shapes: 128 heads, 32 sequence slots, 1024-token
# pages (14 per slot, for a 13824-token max model length), fp8 queries and KV.
_NUM_HEADS = 128
_LKV_DIM = 512
_R_DIM = 64
_PAGE_SIZE = 1024
_MAX_NUM_SEQS = 32
_PAGES_PER_SEQ = 14
_DTYPE = jnp.float8_e4m3fn


def _make_mla_spec(
    name: str,
    num_tokens: int,
    seq_lens: Sequence[tuple[int, int]],
    distribution: tuple[int, int, int],
) -> arg_spec.ArgSpec:
  """A per-device MLA call as DeepSeek serving makes it.

  Args:
    name: The spec name.
    num_tokens: The padded token bucket.
    seq_lens: `(q_len, kv_len)` of the real sequences; `kv_len` includes the new
      tokens. The remaining slots up to `_MAX_NUM_SEQS` are padding.
    distribution: `[d, p, n]`; vLLM always writes `p == d`.

  Returns:
    The arg spec.
  """
  padding = [(0, 0)] * (_MAX_NUM_SEQS - len(seq_lens))
  kv_lens_list, cu_q_lens_list, page_indices_list = [], [0], []
  page_count = 0
  for q_len, kv_len in list(seq_lens) + padding:
    kv_lens_list.append(kv_len)
    cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)
    num_seq_pages = utils.cdiv(kv_len, _PAGE_SIZE)
    page_indices_list.extend(
        list(range(page_count, page_count + num_seq_pages))
        + [-1] * (_PAGES_PER_SEQ - num_seq_pages)
    )
    page_count += num_seq_pages
  assert cu_q_lens_list[-1] <= num_tokens, name

  kv_dim = utils.align_to(_LKV_DIM, 128) + utils.align_to(_R_DIM, 128)
  sds = jax.ShapeDtypeStruct
  return arg_spec.ArgSpec(
      args=dict(
          ql_nope=sds((_NUM_HEADS, num_tokens, _LKV_DIM), _DTYPE),
          q_pe=sds((num_tokens, _NUM_HEADS, _R_DIM), _DTYPE),
          new_kv_c=sds((num_tokens, _LKV_DIM), _DTYPE),
          new_k_pe=sds((num_tokens, _R_DIM), _DTYPE),
          # The whole per-device pool; the page count is normalized out of the
          # autotuning key anyway.
          cache_kv=sds(
              (_MAX_NUM_SEQS * _PAGES_PER_SEQ, kv_dim, _PAGE_SIZE),
              _DTYPE,
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
      ),
      project='deepseek_v3',
      name=name,
      tags=('primary', 'forward_only'),
  )


def _decode(kv_lens: Sequence[int]) -> list[tuple[int, int]]:
  return [(1, kv_len) for kv_len in kv_lens]


# Representative DeepSeek serving batches (random-range ratio 0.5): pure decode
# at three concurrencies, and a mixed batch admitting a new prompt.
ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    # Concurrency 8 and 32.
    _make_mla_spec(
        'mla_decode_rr05_t4',
        num_tokens=4,
        seq_lens=_decode([3073, 7520]),
        distribution=(2, 2, 2),
    ),
    # Concurrency 64.
    _make_mla_spec(
        'mla_decode_rr05_t8',
        num_tokens=8,
        seq_lens=_decode([1537, 2561, 3585, 4609, 5650, 6837, 8296, 10417]),
        distribution=(8, 8, 8),
    ),
    # Concurrency 256.
    _make_mla_spec(
        'mla_decode_rr05_t32',
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
    _make_mla_spec(
        'mla_mixed_rr05_t1024',
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
