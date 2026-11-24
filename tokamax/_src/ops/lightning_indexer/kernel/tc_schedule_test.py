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
"""Tests the in-kernel pass schedule of the fused TensorCore kernel.

`tc_streamindex_topk.build_schedule` must give every pipeline step the same
`(batch_tile_idx, bq_idx, cur_bkv, is_last_bkv)` as the XLA reference
`metadata.compute_metadata(..., qblock_schedule=True)`. The kernel also writes
the empty rows of inactive sequences and padding tokens itself.
"""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.lightning_indexer.kernel import metadata
from tokamax._src.ops.lightning_indexer.kernel import tc_streamindex_topk as tck

_BOUNDS: tuple[tck.SeqBound, ...] = (None, (0, 1), (1, 1), (2, 1))


@dataclasses.dataclass(frozen=True)
class _ScheduleCase:
  """Randomized schedule configuration shared with the XLA reference."""

  seq_lens: np.ndarray
  cu_q_lens: np.ndarray
  dist: np.ndarray
  num_tokens: int
  static_q_len: int | None
  seq_batch_size: int
  bq_sz: int
  compression_ratio: int
  page_size: int
  pages_per_seq: int
  bkv_p: int
  chunk_tokens: int | None
  cp_size: int
  cp_rank: int
  interleave_c: int
  start_bound: tck.SeqBound
  end_bound: tck.SeqBound


def _bound_value(dist: np.ndarray, bound: tck.SeqBound) -> int:
  if bound is None:
    return 0
  idx, align = bound
  return int(dist[idx]) // align * align


def _kernel_schedule(
    seq_lens,
    cu_q_lens,
    dist,
    cp_rank: int,
    *,
    q_cap: int,
    table_cap: int,
    chunk_start: int | None,
    **kw,
):
  """Runs `build_schedule` in a Pallas kernel; returns its SMEM arrays."""

  def kernel(seq_ref, cu_ref, dist_ref, cp_ref, *outs):
    sched = tck.ScheduleRef(
        info=outs[0],
        batch_tile_idx=outs[1],
        bq_idx=outs[2],
        q_start=outs[3],
        q_end=outs[4],
        step_qb=outs[5] if table_cap else None,
        chunk_start=0 if chunk_start is None else chunk_start,
    )
    tck.build_schedule(sched, seq_ref, cu_ref, dist_ref, cp_ref[0], **kw)

  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  shapes = [3] + [q_cap] * 4 + ([table_cap] if table_cap else [])
  return pl.pallas_call(
      kernel,
      in_specs=[smem] * 4,
      out_specs=[smem] * len(shapes),
      out_shape=[jax.ShapeDtypeStruct((n,), jnp.int32) for n in shapes],
  )(seq_lens, cu_q_lens, dist, jnp.asarray([cp_rank], jnp.int32))


def _steps_from_kernel(
    outs, table_cap: int
) -> list[tuple[int, int, int, bool]]:
  """Per-step `(bt, bq, cur_bkv, is_last)` decoded from the kernel schedule."""
  outs = [np.asarray(o) for o in outs]
  info, bt, bq, q_start, q_end = outs[:5]
  num_steps = int(info[2])
  rows: list[tuple[int, int, int, bool]] = []
  for s in range(num_steps):
    if table_cap:
      qb = int(outs[5][s])
    else:
      qb = int(np.searchsorted(q_end, s, side="right"))
    assert q_start[qb] <= s < q_end[qb], (s, qb)
    rows.append((
        int(bt[qb]),
        int(bq[qb]),
        int(s - q_start[qb]),
        bool(s + 1 >= q_end[qb]),
    ))
  return rows


def _steps_from_reference(
    meta, chunk_idx: int
) -> list[tuple[int, int, int, bool]]:
  """Per-step rows of chunk `chunk_idx` from the XLA qblock schedule."""
  num_steps = np.asarray(meta.num_steps)
  q_end = np.asarray(meta.bkv_idx)
  bt = np.asarray(meta.batch_tile_idx)
  bq = np.asarray(meta.bq_idx)
  offset = int(num_steps[:chunk_idx].sum())
  rows: list[tuple[int, int, int, bool]] = []
  for s in range(int(num_steps[chunk_idx])):
    g = offset + s
    qb = int(np.searchsorted(q_end, g, side="right"))
    q_start = int(q_end[qb - 1]) if qb > 0 else 0
    rows.append(
        (int(bt[qb]), int(bq[qb]), int(g - q_start), bool(g + 1 >= q_end[qb]))
    )
  return rows


def _random_case(seed: int) -> _ScheduleCase:
  rng = np.random.default_rng(seed)
  num_seqs = int(rng.choice([1, 2, 3, 5, 8, 13, 16, 24, 33]))
  kind = rng.choice(["decode", "uniform", "mixed"])
  if kind == "decode":
    q_lens = np.ones(num_seqs, np.int64)
    static_q_len = 1
    seq_batch_size = int(rng.choice([1, 2, 4, 8, 16]))
  elif kind == "uniform":
    static_q_len = int(rng.choice([2, 5, 8, 33]))
    q_lens = np.full(num_seqs, static_q_len)
    seq_batch_size = 1
  else:
    static_q_len = None
    q_lens = rng.integers(0, 70, num_seqs)
    seq_batch_size = 1
  compression_ratio = int(rng.choice([1, 4]))
  page_size = int(rng.choice([128, 256]))
  pages_per_seq = int(rng.integers(1, 9))
  bkv_p = int(rng.integers(1, pages_per_seq + 1))
  max_kv = pages_per_seq * page_size * compression_ratio
  seq_lens = np.maximum(q_lens, rng.integers(0, max_kv + 1, num_seqs))
  if rng.random() < 0.2:
    seq_lens[rng.integers(0, num_seqs)] = 0
  cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
  num_tokens = max(1, int(cu_q_lens[-1]) + int(rng.integers(0, 9)))
  if rng.random() < 0.5 and num_tokens >= 4:
    divisors = [d for d in (1, 2, 4) if num_tokens % d == 0]
    chunk_tokens = num_tokens // int(rng.choice(divisors))
  else:
    chunk_tokens = None
  cuts = np.sort(rng.integers(0, num_seqs + 2, 3))
  if rng.random() < 0.15:
    cuts = rng.integers(-1, num_seqs + 3, 3)
  dist = cuts.astype(np.int32)
  cp_size = int(rng.choice([1, 1, 2, 4]))
  cp_rank = int(rng.integers(0, cp_size))
  interleave_c = int(rng.choice([1, 64, 128]))
  end_bound = (
      (0, seq_batch_size)
      if seq_batch_size > 1 and rng.random() < 0.5
      else _BOUNDS[int(rng.integers(0, 4))]
  )
  return _ScheduleCase(
      seq_lens=seq_lens.astype(np.int32),
      cu_q_lens=cu_q_lens,
      dist=dist,
      num_tokens=num_tokens,
      static_q_len=static_q_len,
      seq_batch_size=seq_batch_size,
      bq_sz=1 if static_q_len == 1 else int(rng.choice([1, 8, 16, 32])),
      compression_ratio=compression_ratio,
      page_size=page_size,
      pages_per_seq=pages_per_seq,
      bkv_p=bkv_p,
      chunk_tokens=chunk_tokens,
      cp_size=cp_size,
      cp_rank=cp_rank,
      interleave_c=interleave_c,
      start_bound=_BOUNDS[int(rng.integers(0, 4))],
      end_bound=end_bound,
  )


class _TpuV7TestCase(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 7:
      self.skipTest("Requires TPU v7 or newer.")


class BuildScheduleTest(_TpuV7TestCase):

  @parameterized.parameters(*range(48))
  def test_matches_xla_schedule(self, seed):
    case = _random_case(seed)
    num_tokens = case.num_tokens
    chunk_tokens = case.chunk_tokens
    num_chunks = 1 if chunk_tokens is None else num_tokens // chunk_tokens
    eff_chunk_tokens = chunk_tokens if num_chunks > 1 else None
    chunk_starts = (
        tuple(m * chunk_tokens for m in range(num_chunks))
        if eff_chunk_tokens is not None and chunk_tokens is not None
        else (None,)
    )
    bkv_sz = case.bkv_p * case.page_size
    num_kv_blocks = tck.cdiv(case.pages_per_seq, case.bkv_p)
    seq_lens = jnp.asarray(case.seq_lens)
    cu_q_lens = jnp.asarray(case.cu_q_lens)
    dist = jnp.asarray(case.dist)
    want_start = _bound_value(case.dist, case.start_bound)
    want_end = _bound_value(case.dist, case.end_bound)
    meta = metadata.compute_metadata(
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        start_seq_idx=jnp.int32(want_start),
        end_seq_idx=jnp.int32(want_end),
        bq_sz=case.bq_sz,
        bkv_sz=bkv_sz,
        pages_per_seq=case.pages_per_seq,
        page_size=case.page_size,
        max_num_tokens=num_tokens,
        compression_ratio=case.compression_ratio,
        static_q_len=case.static_q_len,
        seq_batch_size=case.seq_batch_size,
        chunk_token_start=chunk_starts,
        chunk_tokens=eff_chunk_tokens or num_tokens,
        cp_rank=case.cp_rank,
        cp_size=case.cp_size,
        interleave_c=case.interleave_c,
        qblock_schedule=True,
    )
    q_cap, full_table_cap = tck.schedule_capacity(
        max_num_seqs=case.seq_lens.shape[0],
        max_num_tokens=num_tokens,
        seq_batch_size=case.seq_batch_size,
        bq_sz=case.bq_sz,
        static_q_len=case.static_q_len,
        chunk_tokens=eff_chunk_tokens,
        num_kv_blocks=num_kv_blocks,
    )
    for table in (True, False):
      table_cap = full_table_cap if table else 0
      for m in range(num_chunks):
        outs = _kernel_schedule(
            seq_lens,
            cu_q_lens,
            dist,
            case.cp_rank,
            q_cap=q_cap,
            table_cap=table_cap,
            chunk_start=chunk_starts[m],
            start_bound=case.start_bound,
            end_bound=case.end_bound,
            seq_batch_size=case.seq_batch_size,
            bq_sz=case.bq_sz,
            bkv_sz=bkv_sz,
            num_kv_blocks=num_kv_blocks,
            compression_ratio=case.compression_ratio,
            static_q_len=case.static_q_len,
            chunk_tokens=eff_chunk_tokens,
            cp_size=case.cp_size,
            interleave_c=case.interleave_c,
        )
        got = _steps_from_kernel(outs, table_cap)
        want = _steps_from_reference(meta, m)
        self.assertEqual(got, want, msg=f"{case=} {m=} {table=}")
        info = np.asarray(outs[0])
        self.assertEqual(int(info[0]), want_start)
        self.assertEqual(int(info[1]), want_end)


def _has_cache(num_pages: int, page_size: int, rng) -> jax.Array:
  keys = rng.standard_normal((num_pages, page_size, 128))
  keys_u8 = jax.lax.bitcast_convert_type(
      jnp.asarray(keys, jnp.bfloat16).astype(jnp.float8_e4m3fn), jnp.uint8
  )
  scales = jnp.full((num_pages, page_size, 1), 127, jnp.uint8)
  record = jnp.concatenate([keys_u8, scales], axis=-1)
  record = jnp.pad(record, ((0, 0), (0, 0), (0, 256 - record.shape[-1])))
  return record.reshape(num_pages, page_size // 4, 4, 256)


def _sal_cache(num_pages: int, page_size: int, rng) -> jax.Array:
  keys = rng.standard_normal((num_pages, 128, page_size))
  keys_u8 = jax.lax.bitcast_convert_type(
      jnp.asarray(keys, jnp.bfloat16).astype(jnp.float8_e4m3fn), jnp.uint8
  ).reshape(num_pages, 128 // 4, 4, page_size)
  scales = jnp.full((num_pages, 1, 4, page_size), 127, jnp.uint8)
  return jnp.concatenate([keys_u8, scales], axis=1)


class InactiveRowsTest(_TpuV7TestCase):
  """Rows of inactive sequences and padding tokens come back empty."""

  @parameterized.product(
      q_lens=((1, 1, 1, 1), (5, 3)),
      contiguous=(True, False),
      return_scores=(True, False),
  )
  def test_inactive_rows(self, q_lens, contiguous, return_scores):
    rng = np.random.default_rng(len(q_lens) + 2 * contiguous)
    page_size, pages_per_seq, h = 256, 2, 8
    num_active = len(q_lens)
    num_seqs = num_active + 2
    num_pages = num_seqs * pages_per_seq
    cache_kv = _has_cache(num_pages, page_size, rng)
    perm = np.arange(num_pages) if contiguous else rng.permutation(num_pages)
    page_indices = jnp.asarray(perm, jnp.int32)
    active_tokens = sum(q_lens)
    num_tokens = active_tokens + 5
    q = jnp.asarray(
        rng.standard_normal((num_tokens, h, 128)), jnp.bfloat16
    ).astype(jnp.float8_e4m3fn)
    w = jnp.asarray(rng.uniform(0.1, 1.5, (num_tokens, h)), jnp.float32)
    seq_lens = np.full((num_seqs,), pages_per_seq * page_size, np.int32)
    cu = np.concatenate([[0], np.cumsum(q_lens), [active_tokens] * 2])
    decode = all(t == 1 for t in q_lens)
    dist = [num_active] * 3 if decode else [0, 0, num_active]
    kw = dict(
        k=64,
        compression_ratio=1,
        num_kv_pages_per_block=1,
        num_queries_per_block=4,
        decode_req_batch_size=2,
        return_scores=return_scores,
    )
    got = tck.tc_streamindex_topk(
        q,
        w,
        cache_kv,
        jnp.asarray(seq_lens),
        page_indices,
        jnp.asarray(cu, jnp.int32),
        jnp.asarray(dist, jnp.int32),
        **kw,
    )
    want = tck.tc_streamindex_topk(
        q[:active_tokens],
        w[:active_tokens],
        cache_kv,
        jnp.asarray(seq_lens[:num_active]),
        page_indices[: num_active * pages_per_seq],
        jnp.asarray(cu[: num_active + 1], jnp.int32),
        jnp.asarray([min(d, num_active) for d in dist], jnp.int32),
        **kw,
    )
    got_tuple = got if return_scores else (got,)
    want_tuple = want if return_scores else (want,)
    empties = (-1, tck.BITS_NEG_INF) if return_scores else (-1,)
    for g, wt, empty in zip(got_tuple, want_tuple, empties, strict=True):
      g_arr = np.asarray(g)
      np.testing.assert_array_equal(g_arr[:active_tokens], np.asarray(wt))
      np.testing.assert_array_equal(g_arr[active_tokens:], empty)

  @parameterized.product(
      case=(
          ("trailing_mixed", (2, 0), False),
          ("middle_mixed", (1, 0, 1), False),
          ("trailing_decode", (1, 0), True),
      ),
      return_scores=(True, False),
  )
  def test_zero_len_seq_with_bq_sz_one(self, case, return_scores):
    _, q_lens, is_decode = case
    rng = np.random.default_rng(len(q_lens) + 10 * is_decode)
    page_size, pages_per_seq, h = 256, 2, 8
    num_seqs = len(q_lens)
    active_tokens = sum(q_lens)
    cache_kv = _has_cache(num_seqs * pages_per_seq, page_size, rng)
    page_indices = np.arange(num_seqs * pages_per_seq, dtype=np.int32)
    q = jnp.asarray(
        rng.standard_normal((active_tokens, h, 128)), jnp.bfloat16
    ).astype(jnp.float8_e4m3fn)
    w = jnp.asarray(rng.uniform(0.1, 1.5, (active_tokens, h)), jnp.float32)
    seq_lens = np.array(
        [pages_per_seq * page_size if ql > 0 else 0 for ql in q_lens],
        dtype=np.int32,
    )
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    dist = [num_seqs] * 3 if is_decode else [0, 0, num_seqs]
    kw = dict(
        k=64,
        compression_ratio=1,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        decode_req_batch_size=1,
        return_scores=return_scores,
    )
    got = tck.tc_streamindex_topk(
        q,
        w,
        cache_kv,
        jnp.asarray(seq_lens),
        jnp.asarray(page_indices),
        jnp.asarray(cu),
        jnp.asarray(dist, jnp.int32),
        **kw,
    )
    nz = [i for i, ql in enumerate(q_lens) if ql > 0]
    nz_q_lens = [q_lens[i] for i in nz]
    nz_pages = np.concatenate(
        [page_indices[i * pages_per_seq : (i + 1) * pages_per_seq] for i in nz]
    )
    nz_cu = np.concatenate([[0], np.cumsum(nz_q_lens)]).astype(np.int32)
    nz_dist = [len(nz)] * 3 if is_decode else [0, 0, len(nz)]
    want = tck.tc_streamindex_topk(
        q,
        w,
        cache_kv,
        jnp.asarray(seq_lens[nz]),
        jnp.asarray(nz_pages),
        jnp.asarray(nz_cu),
        jnp.asarray(nz_dist, jnp.int32),
        **kw,
    )
    got_tuple = got if return_scores else (got,)
    want_tuple = want if return_scores else (want,)
    for g, wt in zip(got_tuple, want_tuple, strict=True):
      np.testing.assert_array_equal(np.asarray(g), np.asarray(wt))

  @parameterized.product(
      kv_layout=(
          tck.KVLayout.HEAD_ALONG_SUBLANE,
          tck.KVLayout.SEQ_ALONG_LANE,
      ),
      bkv_p=(1, 2),
  )
  def test_negative_page_indices_clamped(self, kv_layout, bkv_p):
    rng = np.random.default_rng(42 + bkv_p)
    page_size, h = 256, 8
    num_seqs = 2
    num_pages = 3
    if kv_layout == tck.KVLayout.SEQ_ALONG_LANE:
      cache_kv = _sal_cache(num_pages, page_size, rng)
    else:
      cache_kv = _has_cache(num_pages, page_size, rng)
    q = jnp.asarray(
        rng.standard_normal((num_seqs, h, 128)), jnp.bfloat16
    ).astype(jnp.float8_e4m3fn)
    w = jnp.asarray(rng.uniform(0.1, 1.5, (num_seqs, h)), jnp.float32)
    # Seq 1 uses only 1 page; its padded second slot holds sentinel -1.
    seq_lens = jnp.array([2 * page_size, page_size], jnp.int32)
    cu = jnp.array([0, 1, 2], jnp.int32)
    dist = jnp.array([2, 2, 2], jnp.int32)
    kw = dict(
        k=64,
        compression_ratio=1,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=1,
        decode_req_batch_size=2,
        kv_layout=kv_layout,
        return_scores=True,
    )
    got_idx, got_sc = tck.tc_streamindex_topk(
        q,
        w,
        cache_kv,
        seq_lens,
        jnp.array([0, 1, 2, -1], jnp.int32),
        cu,
        dist,
        **kw,
    )
    want_idx, want_sc = tck.tc_streamindex_topk(
        q,
        w,
        cache_kv,
        seq_lens,
        jnp.array([0, 1, 2, 0], jnp.int32),
        cu,
        dist,
        **kw,
    )
    np.testing.assert_array_equal(np.asarray(got_idx), np.asarray(want_idx))
    np.testing.assert_array_equal(np.asarray(got_sc), np.asarray(want_sc))


if __name__ == "__main__":
  absltest.main()
