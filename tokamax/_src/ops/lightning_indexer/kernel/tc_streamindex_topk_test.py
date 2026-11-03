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


def _make(mode, num_seqs, q_len, n, seed, num_heads=_H):
  rng = np.random.default_rng(seed)
  num_tokens = num_seqs * q_len
  pages_per_seq = n // _PAGE
  num_pages = num_seqs * pages_per_seq
  if mode == "random":
    q = rng.standard_normal((num_tokens, num_heads, _D))
    keys = rng.standard_normal((num_pages, _PAGE, _D))
    w = rng.uniform(-1.5, 1.5, (num_tokens, num_heads))
  elif mode == "ties":
    q = np.tile(rng.standard_normal((1, num_heads, _D)), (num_tokens, 1, 1))
    pool = rng.standard_normal((3, _D))
    keys = pool[rng.integers(0, 3, (num_pages, _PAGE))]
    w = np.ones((num_tokens, num_heads))
  else:  # "zero": every score is exactly zero.
    q = rng.standard_normal((num_tokens, num_heads, _D))
    keys = rng.standard_normal((num_pages, _PAGE, _D))
    w = np.zeros((num_tokens, num_heads))
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
  for mode in ("random", "ties", "zero"):
    # 64 tiles per row with 16 output blocks: the direct path prunes lanes,
    # in a plain and in a fused prefill pair loop.
    cases.append((f"pruned_prefill_{mode}", mode, 1, 64, 8192, 2, 32, 1, 2048))
    cases.append(
        (f"pruned_fused_prefill_{mode}", mode, 1, 256, 8192, 2, 256, 1, 2048)
    )
  return cases


def _uniform_cases():
  cases = []
  for mode, q_len in itertools.product(
      ("random", "ties", "zero"), (2, 4, 8, 16)
  ):
    cases.append((f"uniform_q{q_len}_{mode}", mode, q_len))
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

  @parameterized.product(
      mode=("random", "ties", "zero"), k=(1024, 2048), q_len=(1, 64)
  )
  def test_long_kv_select_matches_stream(self, mode, k, q_len):
    # 32-64 KV blocks of 1024 keys, so `select` ranks per-block lane maxima
    # first (including the residual top-up and heavy-tie fallbacks).
    num_seqs, n = (8, 32768) if q_len == 1 else (1, 65536)
    args = _make(mode, num_seqs, q_len, n, seed=500 + k + q_len)
    kw = dict(
        k=k,
        compression_ratio=1,
        num_kv_pages_per_block=4,
        num_queries_per_block=32,
        decode_req_batch_size=num_seqs,
        return_scores=True,
    )
    s_idx, s_sc = tck.tc_streamindex_topk(
        *args, topk_impl=tck.TOPK_IMPL_SELECT, **kw
    )
    r_idx, r_sc = tck.tc_streamindex_topk(
        *args, topk_impl=tck.TOPK_IMPL_STREAM, **kw
    )
    np.testing.assert_array_equal(np.asarray(s_idx), np.asarray(r_idx))
    np.testing.assert_array_equal(np.asarray(s_sc), np.asarray(r_sc))

  @parameterized.named_parameters(*_uniform_cases())
  def test_uniform_multi_seq_prefill_matches_unbatched(self, mode, q_len):
    n = 2048
    num_seqs = 16
    q, w, cache_kv, seq_lens, page_indices, cu_q_lens, _ = _make(
        mode, num_seqs, q_len, n, seed=100 + q_len
    )
    dist_uni = jnp.array([0, num_seqs, num_seqs], jnp.int32)
    dist_mix = jnp.array([0, 0, num_seqs], jnp.int32)
    kw = dict(
        k=128,
        compression_ratio=1,
        num_kv_pages_per_block=n // _PAGE,
        decode_req_batch_size=8,
        return_scores=True,
    )
    u_idx, u_sc = tck.tc_streamindex_topk(
        q,
        w,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        dist_uni,
        num_queries_per_block=32,
        uniform_q_len=q_len,
        **kw,
    )
    m_idx, m_sc = tck.tc_streamindex_topk(
        q,
        w,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        dist_mix,
        num_queries_per_block=8 if q_len < 8 else 16,
        uniform_q_len=None,
        **kw,
    )
    np.testing.assert_array_equal(np.asarray(u_idx), np.asarray(m_idx))
    np.testing.assert_array_equal(np.asarray(u_sc), np.asarray(m_sc))

  def test_uniform_q_len_with_mixed_distribution_matches_unbatched(self):
    n = 2048
    cases = (
        # 2 decodes + 2 q=2 seqs padded to 4*2=8 tokens.
        (4, 2, [0, 1, 2, 4, 6], [2, 4, 4]),
        # 1 decode + 2 q=2 seqs + 1 mixed q=3 seq = 8 tokens.
        (4, 2, [0, 1, 3, 5, 8], [1, 3, 4]),
        # 4 decodes + 12 q=2 seqs padded to 16*2=32 tokens (cand_sb=16 > 1).
        (
            16,
            2,
            [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
            [4, 16, 16],
        ),
        # 2 decodes + 12 q=4 seqs + 2 mixed q=8,6 seqs = 64 tokens (cand_sb=8).
        (
            16,
            4,
            [0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 58, 64],
            [2, 14, 16],
        ),
    )
    for idx_case, (num_seqs, q_len, cu_list, dist_list) in enumerate(cases):
      q, w, cache_kv, seq_lens, page_indices, _, _ = _make(
          "random", num_seqs, q_len, n, seed=300 + idx_case
      )
      cu_q_lens = jnp.array(cu_list, jnp.int32)
      dist = jnp.array(dist_list, jnp.int32)
      dist_mix = jnp.array(
          [dist_list[0], dist_list[0], dist_list[2]], jnp.int32
      )
      kw = dict(
          k=128,
          compression_ratio=1,
          num_kv_pages_per_block=n // _PAGE,
          decode_req_batch_size=8,
          return_scores=True,
      )
      u_idx, u_sc = tck.tc_streamindex_topk(
          q,
          w,
          cache_kv,
          seq_lens,
          page_indices,
          cu_q_lens,
          dist,
          num_queries_per_block=(8, 32, 8),
          uniform_q_len=q_len,
          **kw,
      )
      m_idx, m_sc = tck.tc_streamindex_topk(
          q,
          w,
          cache_kv,
          seq_lens,
          page_indices,
          cu_q_lens,
          dist_mix,
          num_queries_per_block=8,
          uniform_q_len=None,
          **kw,
      )
      valid_toks = cu_list[dist_list[2]]
      np.testing.assert_array_equal(
          np.asarray(u_idx)[:valid_toks], np.asarray(m_idx)[:valid_toks]
      )
      np.testing.assert_array_equal(
          np.asarray(u_sc)[:valid_toks], np.asarray(m_sc)[:valid_toks]
      )

  @parameterized.product(num_heads=(4, 8), q_len=(2, 4))
  def test_uniform_small_heads_prefill_disabled_matches_unbatched(
      self, num_heads, q_len
  ):
    n = 2048
    num_seqs = 16
    cfg = tck._BlockedPrefillConfig.create(
        bq_sz=q_len,
        bkv_sz=n,
        num_q_heads=num_heads,
        q_dtype=jnp.float8_e4m3fn,
        seq_batch_size=8,
    )
    if num_heads == 4:
      self.assertFalse(cfg.enabled)
    q, w, cache_kv, seq_lens, page_indices, cu_q_lens, _ = _make(
        "random",
        num_seqs,
        q_len,
        n,
        seed=400 + num_heads * 10 + q_len,
        num_heads=num_heads,
    )
    dist_uni = jnp.array([0, num_seqs, num_seqs], jnp.int32)
    dist_mix = jnp.array([0, 0, num_seqs], jnp.int32)
    kw = dict(
        k=128,
        compression_ratio=1,
        num_kv_pages_per_block=n // _PAGE,
        decode_req_batch_size=8,
        return_scores=True,
    )
    u_idx, u_sc = tck.tc_streamindex_topk(
        q,
        w,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        dist_uni,
        num_queries_per_block=32,
        uniform_q_len=q_len,
        **kw,
    )
    m_idx, m_sc = tck.tc_streamindex_topk(
        q,
        w,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        dist_mix,
        num_queries_per_block=8,
        uniform_q_len=None,
        **kw,
    )
    np.testing.assert_array_equal(np.asarray(u_idx), np.asarray(m_idx))
    np.testing.assert_array_equal(np.asarray(u_sc), np.asarray(m_sc))

  @parameterized.named_parameters(
      # (name, q_lens, seq_lens, dist, pages_per_seq, k, cp_size)
      (
          "ragged_k256",
          [128, 256, 64, 192, 0],
          [2048, 4096, 8192, 6000, 300],
          [0, 0, 5],
          8,
          256,
          1,
      ),
      (
          "ragged_k1024_cp2",
          [128, 256, 64, 192, 0],
          [2048, 4096, 8192, 6000, 300],
          [0, 0, 5],
          8,
          1024,
          2,
      ),
      (
          "decode_and_prefill_k512",
          [1, 1, 1, 128, 64, 192],
          [9216, 9216, 5000, 8192, 2048, 4000],
          [3, 3, 6],
          9,
          512,
          1,
      ),
  )
  def test_kv_bucket_passes_match_stream(
      self, q_lens, seq_lens, dist, pages_per_seq, k, cp_size
  ):
    """KV-length bucket passes match the multi-KV-block streaming path."""
    rng = np.random.default_rng(pages_per_seq * 1000 + k + cp_size)
    num_seqs = len(q_lens)
    num_tokens = sum(q_lens)
    num_pages = num_seqs * pages_per_seq
    q = jnp.asarray(
        rng.standard_normal((num_tokens, _H, _D)), jnp.bfloat16
    ).astype(jnp.float8_e4m3fn)
    w = jnp.asarray(rng.uniform(-1.5, 1.5, (num_tokens, _H)), jnp.float32)
    keys = rng.standard_normal((num_pages, _PAGE, _D))
    keys_u8 = jax.lax.bitcast_convert_type(
        jnp.asarray(keys, jnp.bfloat16).astype(jnp.float8_e4m3fn), jnp.uint8
    )
    scales = jnp.full((num_pages, _PAGE, 1), 127, jnp.uint8)
    record = jnp.concatenate([keys_u8, scales], axis=-1)
    record = jnp.pad(record, ((0, 0), (0, 0), (0, 256 - record.shape[-1])))
    cache_kv = record.reshape(num_pages, _PAGE // 4, 4, 256)
    page_indices = jnp.asarray(rng.permutation(num_pages), jnp.int32)
    cu_q_lens = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
    args = (
        q,
        w,
        cache_kv,
        jnp.asarray(seq_lens, jnp.int32),
        page_indices,
        cu_q_lens,
        jnp.asarray(dist, jnp.int32),
    )
    kw = dict(
        k=k,
        compression_ratio=4,
        num_queries_per_block=64,
        decode_req_batch_size=2,
        return_scores=True,
        cp_size=cp_size,
        cp_rank=cp_size - 1,
        interleave_size=512,
    )
    b_idx, b_sc = tck.tc_streamindex_topk(
        *args, num_kv_pages_per_block=pages_per_seq, **kw
    )
    s_idx, s_sc = tck.tc_streamindex_topk(
        *args, num_kv_pages_per_block=1, topk_impl="stream", **kw
    )
    np.testing.assert_array_equal(np.asarray(b_idx), np.asarray(s_idx))
    np.testing.assert_array_equal(np.asarray(b_sc), np.asarray(s_sc))


if __name__ == "__main__":
  absltest.main()
