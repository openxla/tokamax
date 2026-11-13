# Copyright 2026 Google LLC. All Rights Reserved.
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
"""Exactness tests for the fused TensorCore streaming Top-K.

Compares the multi-KV-block streaming path (per-lane candidate lists and
staged evictions) against the single-KV-block direct path (full sort) bit for
bit: ordered indices and returned score bits, on random, heavily tied and
all-tied inputs.
"""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.lightning_indexer.kernel import tc_streamindex_topk as tck

_PAGE = 256
_D = 128
_H = 64


def _make(mode, num_seqs, q_len, n, seed):
  rng = np.random.default_rng(seed)
  num_tokens = num_seqs * q_len
  pages_per_seq = n // _PAGE
  num_pages = num_seqs * pages_per_seq
  if mode == "random":
    q = rng.standard_normal((num_tokens, _H, _D))
    keys = rng.standard_normal((num_pages, _PAGE, _D))
    w = rng.uniform(-1.5, 1.5, (num_tokens, _H))
  elif mode == "ties":
    q = np.tile(rng.standard_normal((1, _H, _D)), (num_tokens, 1, 1))
    pool = rng.standard_normal((3, _D))
    keys = pool[rng.integers(0, 3, (num_pages, _PAGE))]
    w = np.ones((num_tokens, _H))
  else:  # "zero": every score is exactly zero.
    q = rng.standard_normal((num_tokens, _H, _D))
    keys = rng.standard_normal((num_pages, _PAGE, _D))
    w = np.zeros((num_tokens, _H))
  q = jnp.asarray(q, jnp.bfloat16).astype(jnp.float8_e4m3fn)
  keys_u8 = jax.lax.bitcast_convert_type(
      jnp.asarray(keys, jnp.bfloat16).astype(jnp.float8_e4m3fn), jnp.uint8
  )
  scales = jnp.full((num_pages, _PAGE, 1), 127, jnp.uint8)
  record = jnp.concatenate([keys_u8, scales], axis=-1)
  record = jnp.pad(record, ((0, 0), (0, 0), (0, 256 - record.shape[-1])))
  cache_kv = record.reshape(num_pages, _PAGE // 4, 4, 256)
  perm = rng.permutation(num_pages).astype(np.int32)
  page_indices = jnp.asarray(perm)
  seq_lens = jnp.full((num_seqs,), n, jnp.int32)
  cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32) * q_len
  if q_len == 1:
    dist = jnp.array([num_seqs, num_seqs, num_seqs], jnp.int32)
  else:
    dist = jnp.array([0, 0, num_seqs], jnp.int32)
  w = jnp.asarray(w, jnp.float32)
  return q, w, cache_kv, seq_lens, page_indices, cu_q_lens, dist


def _cases():
  cases = []
  for mode, k in itertools.product(
      ("random", "ties", "zero"), (64, 256, 300, 512, 1024, 2048)
  ):
    # (name, mode, num_seqs, q_len, n, bkv_p, bq, dec_bs, k)
    cases.append((f"decode_b1_{mode}_k{k}", mode, 1, 1, 8192, 2, 8, 1, k))
    cases.append((f"decode_b8_{mode}_k{k}", mode, 8, 1, 4096, 1, 8, 8, k))
    cases.append((f"prefill_{mode}_k{k}", mode, 1, 64, 4096, 2, 32, 1, k))
  return cases


class StreamTopkExactTest(parameterized.TestCase):
  """Streaming vs. direct-path bit-exactness."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 7:
      self.skipTest("Requires TPU v7 or newer.")

  @parameterized.named_parameters(*_cases())
  def test_stream_matches_direct(
      self, mode, num_seqs, q_len, n, bkv_p, bq, dec_bs, k
  ):
    args = _make(mode, num_seqs, q_len, n, seed=k + n)
    kw = dict(
        k=k,
        compression_ratio=1,
        num_queries_per_block=bq,
        decode_req_batch_size=dec_bs,
        return_scores=True,
    )
    s_idx, s_sc = tck.tc_streamindex_topk(
        *args, num_kv_pages_per_block=bkv_p, **kw
    )
    d_idx, d_sc = tck.tc_streamindex_topk(
        *args, num_kv_pages_per_block=n // _PAGE, **kw
    )
    s_idx, s_sc, d_idx, d_sc = map(np.asarray, (s_idx, s_sc, d_idx, d_sc))
    np.testing.assert_array_equal(s_idx, d_idx)
    np.testing.assert_array_equal(s_sc, d_sc)
    # Sanity: descending scores, ties by ascending index, valid count.
    sc = s_sc.view(np.float32)
    for row in range(min(4, s_idx.shape[0])):
      valid = s_idx[row] >= 0
      v_sc, v_idx = sc[row][valid], s_idx[row][valid]
      order = np.lexsort((v_idx, -v_sc))
      np.testing.assert_array_equal(order, np.arange(len(order)))


if __name__ == "__main__":
  absltest.main()
