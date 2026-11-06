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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.lightning_indexer.kernel import metadata
from tokamax._src.ops.lightning_indexer.kernel import tc_streamindex_topk as tck

_BOUNDS = (None, (0, 1), (1, 1), (2, 1))


def _bound_value(dist, bound):
  if bound is None:
    return 0
  idx, align = bound
  return int(dist[idx]) // align * align


def _kernel_schedule(
    seq_lens,
    cu_q_lens,
    dist,
    cp_rank,
    *,
    q_cap,
    table_cap,
    chunk_start,
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


def _steps_from_kernel(outs, table_cap):
  """Per-step `(bt, bq, cur_bkv, is_last)` decoded from the kernel schedule."""
  outs = [np.asarray(o) for o in outs]
  info, bt, bq, q_start, q_end = outs[:5]
  num_steps = int(info[2])
  rows = []
  for s in range(num_steps):
    if table_cap:
      qb = int(outs[5][s])
    else:
      qb = int(np.searchsorted(q_end, s, side="right"))
    assert q_start[qb] <= s < q_end[qb], (s, qb)
    rows.append((bt[qb], bq[qb], s - q_start[qb], s + 1 >= q_end[qb]))
  return rows


def _steps_from_reference(meta, chunk_idx):
  """Per-step rows of chunk `chunk_idx` from the XLA qblock schedule."""
  num_steps = np.asarray(meta.num_steps)
  q_end = np.asarray(meta.bkv_idx)
  bt = np.asarray(meta.batch_tile_idx)
  bq = np.asarray(meta.bq_idx)
  offset = int(num_steps[:chunk_idx].sum())
  rows = []
  for s in range(int(num_steps[chunk_idx])):
    g = offset + s
    qb = int(np.searchsorted(q_end, g, side="right"))
    q_start = int(q_end[qb - 1]) if qb > 0 else 0
    rows.append((bt[qb], bq[qb], g - q_start, g + 1 >= q_end[qb]))
  return rows


def _random_case(seed, kind=None):
  rng = np.random.default_rng(seed)
  num_seqs = int(rng.choice([1, 2, 3, 5, 8, 13, 16, 24, 33]))
  drawn_kind = rng.choice(["decode", "uniform", "mixed"])
  kind = drawn_kind if kind is None else kind
  if kind == "decode":
    q_lens = np.ones(num_seqs, np.int64)
    static_q_len = 1
    sb = int(rng.choice([1, 2, 4, 8, 16]))
  elif kind == "uniform":
    static_q_len = int(rng.choice([2, 5, 8, 33]))
    q_lens = np.full(num_seqs, static_q_len)
    sb = 1
  else:
    static_q_len = None
    q_lens = rng.integers(0, 70, num_seqs)
    sb = 1
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
  return dict(
      seq_lens=seq_lens.astype(np.int32),
      cu_q_lens=cu_q_lens,
      dist=dist,
      num_tokens=num_tokens,
      static_q_len=static_q_len,
      sb=sb,
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
      end_bound=(
          (0, sb) if sb > 1 and rng.random() < 0.5
          else _BOUNDS[int(rng.integers(0, 4))]
      ),
  )


class BuildScheduleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 7:
      self.skipTest("Requires TPU v7 or newer.")

  @parameterized.parameters(*range(48))
  def test_matches_xla_schedule(self, seed):
    c = _random_case(seed)
    num_tokens = c["num_tokens"]
    chunk_tokens = c["chunk_tokens"]
    num_chunks = 1 if chunk_tokens is None else num_tokens // chunk_tokens
    chunked = num_chunks > 1
    chunk_starts = (
        tuple(m * chunk_tokens for m in range(num_chunks))
        if chunked
        else (None,)
    )
    bkv_sz = c["bkv_p"] * c["page_size"]
    num_kv_blocks = tck.cdiv(c["pages_per_seq"], c["bkv_p"])
    meta = metadata.compute_metadata(
        seq_lens=jnp.asarray(c["seq_lens"]),
        cu_q_lens=jnp.asarray(c["cu_q_lens"]),
        start_seq_idx=jnp.int32(_bound_value(c["dist"], c["start_bound"])),
        end_seq_idx=jnp.int32(_bound_value(c["dist"], c["end_bound"])),
        bq_sz=c["bq_sz"],
        bkv_sz=bkv_sz,
        pages_per_seq=c["pages_per_seq"],
        page_size=c["page_size"],
        max_num_tokens=num_tokens,
        compression_ratio=c["compression_ratio"],
        static_q_len=c["static_q_len"],
        seq_batch_size=c["sb"],
        chunk_token_start=chunk_starts,
        chunk_tokens=chunk_tokens if chunked else num_tokens,
        cp_rank=c["cp_rank"],
        cp_size=c["cp_size"],
        interleave_c=c["interleave_c"],
        qblock_schedule=True,
    )
    for table in (True, False):
      for m in range(num_chunks):
        q_cap, table_cap = tck.schedule_capacity(
            max_num_seqs=c["seq_lens"].shape[0],
            max_num_tokens=num_tokens,
            seq_batch_size=c["sb"],
            bq_sz=c["bq_sz"],
            static_q_len=c["static_q_len"],
            chunk_tokens=chunk_tokens if chunked else None,
            num_kv_blocks=num_kv_blocks,
        )
        if not table:
          table_cap = 0
        outs = _kernel_schedule(
            jnp.asarray(c["seq_lens"]),
            jnp.asarray(c["cu_q_lens"]),
            jnp.asarray(c["dist"]),
            c["cp_rank"],
            q_cap=q_cap,
            table_cap=table_cap,
            chunk_start=chunk_starts[m],
            start_bound=c["start_bound"],
            end_bound=c["end_bound"],
            seq_batch_size=c["sb"],
            bq_sz=c["bq_sz"],
            bkv_sz=bkv_sz,
            num_kv_blocks=num_kv_blocks,
            compression_ratio=c["compression_ratio"],
            static_q_len=c["static_q_len"],
            chunk_tokens=chunk_tokens if chunked else None,
            cp_size=c["cp_size"],
            interleave_c=c["interleave_c"],
        )
        got = _steps_from_kernel(outs, table_cap)
        want = _steps_from_reference(meta, m)
        self.assertEqual(
            [tuple(map(int, r)) for r in got],
            [tuple(map(int, r)) for r in want],
            msg=f"{c=} {m=} {table=}",
        )
        info = np.asarray(outs[0])
        self.assertEqual(
            int(info[0]), _bound_value(c["dist"], c["start_bound"])
        )
        self.assertEqual(int(info[1]), _bound_value(c["dist"], c["end_bound"]))

  @parameterized.parameters(*range(24))
  def test_vis_range_buckets_partition_schedule(self, seed):
    """KV-length bucket passes split one pass's query blocks exactly."""
    c = _random_case(1000 + seed, kind="mixed")
    page_size = c["page_size"]
    pages_per_seq = c["pages_per_seq"]
    num_tokens = c["num_tokens"]
    chunk_tokens = c["chunk_tokens"]
    num_chunks = 1 if chunk_tokens is None else num_tokens // chunk_tokens
    chunked = num_chunks > 1
    bucket_pages = sorted(
        {p for p in (pages_per_seq // 4, pages_per_seq // 2) if p >= 1}
        - {pages_per_seq}
    )

    def _blocks(chunk_start, table, bkv_p, vis_range=None):
      num_kv_blocks = tck.cdiv(pages_per_seq, bkv_p)
      if vis_range is not None and vis_range[1] is not None:
        num_kv_blocks = 1
      q_cap, table_cap = tck.schedule_capacity(
          max_num_seqs=c["seq_lens"].shape[0],
          max_num_tokens=num_tokens,
          seq_batch_size=1,
          bq_sz=c["bq_sz"],
          static_q_len=None,
          chunk_tokens=chunk_tokens if chunked else None,
          num_kv_blocks=num_kv_blocks,
      )
      outs = _kernel_schedule(
          jnp.asarray(c["seq_lens"]),
          jnp.asarray(c["cu_q_lens"]),
          jnp.asarray(c["dist"]),
          c["cp_rank"],
          q_cap=q_cap,
          table_cap=table_cap if table else 0,
          chunk_start=chunk_start,
          start_bound=c["start_bound"],
          end_bound=c["end_bound"],
          seq_batch_size=1,
          bq_sz=c["bq_sz"],
          bkv_sz=bkv_p * page_size,
          num_kv_blocks=num_kv_blocks,
          compression_ratio=c["compression_ratio"],
          static_q_len=None,
          chunk_tokens=chunk_tokens if chunked else None,
          cp_size=c["cp_size"],
          interleave_c=c["interleave_c"],
          vis_range=vis_range,
      )
      blocks = {}
      for bt, bq, _, _ in _steps_from_kernel(outs, table_cap if table else 0):
        key = (int(bt), int(bq))
        blocks[key] = blocks.get(key, 0) + 1
      return blocks

    for table in (True, False):
      for m in range(num_chunks):
        chunk_start = m * chunk_tokens if chunked else None
        full = _blocks(chunk_start, table, pages_per_seq)
        order = list(full)
        # A query block fits in `p` pages iff it has one KV block of `p`.
        fits = {
            p: {b for b, n in _blocks(chunk_start, table, p).items() if n == 1}
            for p in bucket_pages
        }
        merged = {}
        lo, lo_fits = -1, set()
        for p in bucket_pages + [None]:
          if p is None:
            vis_range = (lo, None)
            got = _blocks(chunk_start, table, pages_per_seq, vis_range)
            want = set(full) - lo_fits
          else:
            vis_range = (lo, p * page_size)
            got = _blocks(chunk_start, table, p, vis_range)
            want = fits[p] - lo_fits
            lo, lo_fits = p * page_size, fits[p]
          msg = f"{c=} {m=} {table=} {vis_range=}"
          self.assertEqual(set(got), want, msg=msg)
          self.assertEqual(list(got), sorted(got, key=order.index), msg=msg)
          for key, n in got.items():
            self.assertEqual(n, full[key] if p is None else 1, msg=msg)
          merged.update(got)
        self.assertEqual(merged, full, msg=f"{c=} {m=} {table=}")


def _has_cache(num_pages, page_size, rng):
  keys = rng.standard_normal((num_pages, page_size, 128))
  keys_u8 = jax.lax.bitcast_convert_type(
      jnp.asarray(keys, jnp.bfloat16).astype(jnp.float8_e4m3fn), jnp.uint8
  )
  scales = jnp.full((num_pages, page_size, 1), 127, jnp.uint8)
  record = jnp.concatenate([keys_u8, scales], axis=-1)
  record = jnp.pad(record, ((0, 0), (0, 0), (0, 256 - record.shape[-1])))
  return record.reshape(num_pages, page_size // 4, 4, 256)


class InactiveRowsTest(parameterized.TestCase):
  """Rows of inactive sequences and padding tokens come back empty."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 7:
      self.skipTest("Requires TPU v7 or newer.")

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
    got = got if return_scores else (got,)
    want = want if return_scores else (want,)
    for g, wt, empty in zip(got, want, (-1, tck.BITS_NEG_INF)):
      g = np.asarray(g)
      np.testing.assert_array_equal(g[:active_tokens], np.asarray(wt))
      np.testing.assert_array_equal(g[active_tokens:], empty)


if __name__ == "__main__":
  absltest.main()
