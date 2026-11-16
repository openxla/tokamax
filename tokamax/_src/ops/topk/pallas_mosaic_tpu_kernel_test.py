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

"""Tests for the SparseCore top-k kernel."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.topk import pallas_mosaic_tpu_kernel as sc_module
from tokamax._src.ops.topk.pallas_mosaic_tpu_kernel import (
    BLOCK_UNROLL,
    LANES,
    MAX_DOUBLE_BUF_WORDS,
    MAX_SLICE_WORDS,
    _align_to,
    _block_chunks,
    _pick_partition,
    _sc_topk_direct,
    sparsecore_topk,
)

N_SHORT = 10240  # --max-model-len 9216
N_128K = 133120  # --max-model-len 132096
N_1M = 1048576  # --max-model-len 1048576

SEED = 20260904


def _topk_n(max_model_len: int, page_size: int = 1024, bkv_p: int = 2) -> int:
  pages_per_seq = -(-max_model_len // page_size)
  return _align_to(pages_per_seq, bkv_p) * page_size // 128 * 128


def _reference_scores(
    scores: np.ndarray, k: int, row_lengths: np.ndarray
) -> np.ndarray:
  """Exact top-k scores per row, descending, -inf suffix-padded."""
  out = np.full((scores.shape[0], k), -np.inf, np.float32)
  for i in range(scores.shape[0]):
    row = scores[i, : int(row_lengths[i])]
    eligible = row[row > -np.inf]
    vals = np.sort(eligible)[::-1][:k]
    out[i, : vals.size] = vals
  return out


def _mismatch(
    scores: np.ndarray, idx: np.ndarray, k: int, row_lengths: np.ndarray
) -> str | None:
  """Returns None if idx is a valid exact top-k, else a failure reason."""
  b, n = scores.shape
  ref = _reference_scores(scores, k, row_lengths)
  got = np.full((b, k), -np.inf, np.float32)

  for i in range(b):
    sel = idx[i]
    live = sel[sel >= 0]
    n_live = live.size
    if not (np.all(idx[i, n_live:] < 0) and np.all(idx[i, :n_live] >= 0)):
      return f"row {i}: -1 padding is not a suffix"
    if n_live and live.max() >= n:
      return f"row {i}: index {live.max()} out of range for n={n}"
    if np.unique(live).size != n_live:
      return f"row {i}: duplicate indices"
    if n_live and live.max() >= 0:
      beyond = live[live >= int(row_lengths[i])]
      if beyond.size:
        return (
            f"row {i}: index {beyond[0]} at or beyond "
            f"row_length {int(row_lengths[i])}"
        )
    vals = np.sort(scores[i, live])[::-1] if n_live else np.zeros(0)
    got[i, : vals.size] = vals

  n_ref = np.isfinite(ref).sum(axis=1)
  n_got = np.isfinite(got).sum(axis=1)
  if not (n_ref == n_got).all():
    bad = int(np.flatnonzero(n_ref != n_got)[0])
    return (
        f"row {bad}: returned {n_got[bad]} valid entries, expected {n_ref[bad]}"
    )

  finite = np.isfinite(ref) & np.isfinite(got)
  if finite.any():
    diff = np.abs(ref[finite] - got[finite]).max()
    if diff != 0.0:
      return f"max score difference {diff!r}, expected exactly 0.0"
  return None


def _scores(kind: str, b: int, n: int, rng: np.random.Generator) -> np.ndarray:
  """Generates test score matrices."""
  if kind == "strict_desc":
    # Strictly descending negative values to exercise monotone_key on negative floats.
    return np.repeat(-np.arange(n, dtype=np.float32)[None, :], b, axis=0)
  if kind == "random":
    return rng.standard_normal((b, n), dtype=np.float32)
  if kind == "heavy_ties":
    # 17 distinct values across the row to force ties at partition boundaries.
    return rng.integers(0, 17, size=(b, n), dtype=np.int8).astype(np.float32)
  if kind == "all_neg_inf":
    return np.full((b, n), -np.inf, np.float32)
  raise ValueError(kind)


def _run(scores: np.ndarray, k: int, row_lengths: np.ndarray) -> np.ndarray:
  return np.asarray(
      jax.block_until_ready(
          sparsecore_topk(jnp.asarray(scores), k, jnp.asarray(row_lengths))
      )
  )


def _block_walk_lengths(n: int, p: int, blk: int) -> list[int]:
  """Row lengths at every edge the block-padded slice walks care about.

  Covers empty and one-chunk rows, partial and whole tail chunks, lengths one
  chunk either side of each block boundary of the first slice and of the last
  block of a slice, and, for p > 1, the edges of every slice's live part
  (slices start at multiples of the padded slice length, so the last ones hold
  less of the row or none of it).

  Args:
    n: row width.
    p: subcores per row.
    blk: chunks per block of the slice walks.

  Returns:
    Sorted distinct lengths in [0, n].
  """
  block = LANES * blk
  raw = n // p
  slice_len = _align_to(raw, block)
  edges = {0, 1, LANES - 1, LANES, LANES + 1, n - 1, n}
  for e in (block, 2 * block, (raw // block) * block, slice_len):
    edges |= {e - LANES, e - 1, e, e + 1, e + LANES}
  for s in range(1, p):
    e = s * slice_len
    edges |= {e - 1, e, e + 1, e + LANES - 1}
  return sorted(x for x in edges if 0 <= x <= n)


def _block_walk_rows(
    rng: np.random.Generator, b: int, n: int, k: int, lengths: list[int]
) -> np.ndarray:
  """Score rows for the block-padded walks; row r gets ``lengths[r]``.

  Live scores are mostly negative, so a pad word left at its HBM value (bits
  0, i.e. +0.0) would outrank them, and every score at or past the row length
  is 1e30, so a walk that reads past the live chunks selects it. Some rows
  have fewer finite keys than k (the rest -inf) or many ties at the k-th
  score, so the tie and -inf handling also run over the padded tail.
  """
  scores = (rng.standard_normal((b, n)) - 4.0).astype(np.float32)
  for r, length in enumerate(lengths):
    kind = r % 4
    if kind == 1:
      # Fewer finite keys than k: the threshold falls in the -inf bucket.
      keep = rng.random(n) < 0.5 * k / max(n, 1)
      scores[r, ~keep] = -np.inf
    elif kind == 2:
      # Heavy ties: most of the row shares one score just under the top.
      scores[r, rng.random(n) < 0.7] = -1.5
      scores[r, rng.integers(0, n, size=k // 4)] = -0.5
    scores[r, length:] = 1e30
  return scores


def _assert_returned_score_bits(
    scores: np.ndarray,
    idxs: jax.Array | np.ndarray,
    score_bits: jax.Array,
):
  """The returned scores are the winners' exact f32 bits, -inf for padding.

  No winner may carry a -inf key (a padded tail word leaking out), and every
  slot is either a live index or the -1 suffix pad.
  """
  idxs = np.asarray(idxs)
  bits = np.asarray(score_bits)
  neg_inf_bits = np.array(-np.inf, np.float32).view(np.int32)
  assert np.all(idxs >= -1), f"Invalid sentinel index: {idxs[idxs < -1]}"
  assert not np.any(
      bits[idxs >= 0] == neg_inf_bits
  ), "A selected index returned a -inf key"
  assert np.all(
      bits[idxs < 0] == neg_inf_bits
  ), "A -1 pad slot returned a non -inf key"
  rows = np.arange(scores.shape[0])[:, None]
  want = np.where(
      idxs >= 0,
      scores.view(np.int32)[rows, np.maximum(idxs, 0)],
      neg_inf_bits,
  )
  np.testing.assert_array_equal(bits, want)


# (name, b, n, k, expected_p) covering single-stage (P=1) and two-stage paths.
SHAPES = (
    ("p1_b64_n10240", 64, N_SHORT, 2048, 1),
    ("p2_b16_n10240", 16, N_SHORT, 2048, 2),
    ("p8_b16_n133120", 16, N_128K, 2048, 8),
    ("p8_b64_n133120", 64, N_128K, 2048, 8),
    ("p8_b16_n133120_k512", 16, N_128K, 512, 8),
    ("p8_b1024_n133120", 1024, N_128K, 2048, 8),
)


class SparseCoreTopKTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 7:
      self.skipTest(
          "SparseCore TopK Pallas TPU kernel requires TPU v7 or newer."
      )

  def test_shape_derivation_matches_max_model_len(self):
    """The 128K shape is derived, not hardcoded, so a page-size change shows."""
    assert _topk_n(9216) == N_SHORT
    assert _topk_n(132096) == N_128K
    assert _topk_n(1048576, page_size=256) == N_1M

  @parameterized.parameters((64, 2048), (128, 2048))
  def test_exact_topk_1m_single_stage_partition(self, b, k):
    """Verifies single-stage P=16 VMEM_SHARED execution at N=1M (64K slices)."""
    p = _pick_partition(b, N_1M, k)
    assert p == 16, f"Expected p=16 for {b=} {k=} at N_1M, got {p=}"
    assert N_1M // p <= MAX_SLICE_WORDS
    rng = np.random.default_rng(SEED)
    scores = _scores("random", b, N_1M, rng)
    row_lengths = np.full(b, N_1M, np.int32)
    assert (
        _mismatch(scores, _run(scores, k, row_lengths), k, row_lengths) is None
    )

  @parameterized.named_parameters(*SHAPES)
  def test_partition_factor_is_what_these_tests_assume(
      self, b, n, k, expected_p
  ):
    """Verifies _pick_partition returns the expected partition factor P."""
    assert _pick_partition(b, n, k) == expected_p

  @parameterized.named_parameters(
      (f"{name}_{kind}", b, n, k, kind)
      for name, b, n, k, _ in SHAPES
      for kind in ("strict_desc", "random", "heavy_ties")
  )
  def test_exact_topk_full_rows(self, b, n, k, kind):
    rng = np.random.default_rng(SEED)
    scores = _scores(kind, b, n, rng)
    row_lengths = np.full(b, n, np.int32)
    assert (
        _mismatch(scores, _run(scores, k, row_lengths), k, row_lengths) is None
    )

  @parameterized.named_parameters(*SHAPES)
  def test_exact_topk_ragged_rows_straddle_partition_boundaries(
      self, b, n, k, expected_p
  ):
    """Verifies exact top-k when ragged row_lengths straddle partition boundaries."""
    n_p = n // expected_p
    rng = np.random.default_rng(SEED)
    scores = _scores("random", b, n, rng)
    edges = sorted({
        min(max(e, 0), n)
        for e in (
            0,
            1,
            LANES,
            k,
            n_p - 1,
            n_p,
            n_p + 1,
            2 * n_p,
            2 * n_p + 17,
            3 * n_p,
            n - 1,
            n,
        )
    })
    row_lengths = np.array([edges[i % len(edges)] for i in range(b)], np.int32)
    assert (
        _mismatch(scores, _run(scores, k, row_lengths), k, row_lengths) is None
    )

  @parameterized.named_parameters(
      (name, b, n, k) for name, b, n, k, _ in SHAPES
  )
  def test_all_neg_inf_rows_return_no_selection(self, b, n, k):
    """-inf is never eligible, so every output slot must be -1."""
    rng = np.random.default_rng(SEED)
    scores = _scores("all_neg_inf", b, n, rng)
    row_lengths = np.full(b, n, np.int32)
    idx = _run(scores, k, row_lengths)
    assert (idx == -1).all(), f"expected all -1, got {np.unique(idx)[:8]}"
    assert _mismatch(scores, idx, k, row_lengths) is None

  def test_topk_concentrated_in_the_last_partition(self):
    """Verifies two-stage top-k when all top-k elements sit in the final slice."""
    b, n, k, p = 16, N_128K, 2048, 8
    assert _pick_partition(b, n, k) == p
    rng = np.random.default_rng(SEED)
    scores = np.full((b, n), -1e30, np.float32)
    scores[:, -(k + 64) :] = rng.standard_normal((b, k + 64)).astype(np.float32)
    row_lengths = np.full(b, n, np.int32)
    assert (
        _mismatch(scores, _run(scores, k, row_lengths), k, row_lengths) is None
    )

  @parameterized.parameters("worst_index", "drop_one", "duplicate", "negative")
  def test_comparison_detects_a_wrong_index(self, corruption):
    """Verifies _mismatch rejects corrupted top-k outputs."""
    b, n, k = 16, N_128K, 2048
    rng = np.random.default_rng(SEED)
    scores = _scores("random", b, n, rng)
    row_lengths = np.full(b, n, np.int32)
    idx = _run(scores, k, row_lengths)
    assert _mismatch(scores, idx, k, row_lengths) is None, "baseline must pass"

    bad = idx.copy()
    if corruption == "worst_index":
      bad[0, 0] = int(np.argmin(scores[0, : int(row_lengths[0])]))
    elif corruption == "drop_one":
      bad[0, 0] = -1
    elif corruption == "duplicate":
      bad[0, 0] = int(bad[0, 1])
    elif corruption == "negative":
      bad[0, 0] = n + 5

    assert (
        _mismatch(scores, bad, k, row_lengths) is not None
    ), f"corruption {corruption!r} was not detected"

  @parameterized.parameters(
      (32, 2048, 256),  # single_stage
      (4, 8192, 256),  # two_stage
  )
  def test_return_scores(self, b: int, n: int, k: int):
    """Verifies return_scores=True returns matching scores and -inf for pad."""
    np.random.seed(800 + b + n + k)
    scores = np.random.randn(b, n).astype(np.float32)
    row_lengths = np.random.randint(k, n + 1, size=(b,), dtype=np.int32)

    actual_idxs, actual_scores_bits = sparsecore_topk(
        jnp.array(scores),
        k,
        row_lengths=jnp.array(row_lengths),
        return_scores=True,
    )
    assert _mismatch(scores, actual_idxs, k, row_lengths) is None

    actual_scores = np.asarray(
        jax.lax.bitcast_convert_type(actual_scores_bits, jnp.float32)
    )
    for r in range(b):
      for i in range(k):
        idx = int(actual_idxs[r, i])
        score_val = actual_scores[r, i]
        if idx >= 0:
          np.testing.assert_allclose(score_val, scores[r, idx], rtol=1e-5)
        else:
          assert score_val == -np.inf

  def test_block_chunks_keeps_padding_and_buffering_bounds(self):
    """The slice block never pads past 1/8 of the slice or a buffer limit.

    Every slice length a row can have up to MAX_SLICE_WORDS is checked: the
    padded slice must stay within 9/8 of the 16-aligned length, must not
    cross MAX_DOUBLE_BUF_WORDS (that would silently drop the slice to single
    buffering) or MAX_SLICE_WORDS (tile memory), and the block never exceeds
    BLOCK_UNROLL.
    """
    for raw in range(1, MAX_SLICE_WORDS + 1):
      blk = _block_chunks(raw)
      self.assertGreaterEqual(blk, 1, msg=f"raw={raw}")
      self.assertEqual(blk & (blk - 1), 0, msg=f"raw={raw}: blk={blk}")
      self.assertLessEqual(blk, BLOCK_UNROLL, msg=f"raw={raw}")
      base = _align_to(raw, LANES)
      padded = _align_to(raw, LANES * blk)
      if blk > 1:
        self.assertLessEqual(8 * padded, 9 * base, msg=f"raw={raw}")
      for limit in (MAX_DOUBLE_BUF_WORDS, MAX_SLICE_WORDS):
        if base <= limit:
          self.assertLessEqual(padded, limit, msg=f"raw={raw}")

  @parameterized.named_parameters(
      # (b, n, k, p, blk): p and blk are asserted so each case keeps reaching
      # the walk it is named for. Ragged rows fill the tail of the slice up to
      # a block in tile memory in every case with blk > 1; all of those but
      # the DeepSeek-V4 row (2,304 keys, a whole number of blocks) also pad
      # the slice in HBM.
      ("p1_blk8_compact", 32, 2112, 128, 1, 8),
      ("p1_blk8_walk", 32, 2112, 512, 1, 8),
      ("p1_blk8_short_compact", 32, 1040, 128, 1, 8),
      ("p1_blk8_ds_v4_row_two_waves", 64, 2304, 512, 1, 8),
      ("p1_blk8_long_compact", 32, 16400, 2048, 1, 8),
      ("p1_blk8_long_walk_k4096", 32, 16400, 4096, 1, 8),
      ("p1_blk4", 32, 176, 16, 1, 4),
      ("p1_blk2", 32, 144, 16, 1, 2),
      ("p1_unblocked", 32, 48, 16, 1, 1),
      ("p1_single_chunk", 32, 16, 16, 1, 1),
      ("p4_blk8", 8, 4112, 256, 4, 8),
      ("p8_blk8", 4, 16400, 1024, 8, 8),
      ("p16_blk8_compact", 1, 36512, 256, 16, 8),
      ("p16_blk8_long_compact", 2, 262656, 2048, 16, 8),
      # n > 16 * MAX_DOUBLE_BUF_WORDS: p comes from the slice ceiling, the
      # 37,501-word slices are single-buffered and not LANES aligned, and the
      # last slice holds the HBM pad.
      ("p16_blk8_single_buffered_unaligned", 1, 600016, 2048, 16, 8),
  )
  def test_block_padded_slice_walks(
      self, b: int, n: int, k: int, p: int, blk: int
  ):
    """Whole-block walks over a -inf padded slice tail are exact.

    The level-0 histogram, compaction, refinement and emit walks run over
    whole blocks of ``blk`` chunks, the slice is rounded up to a whole block
    and its tail is overwritten with -inf in tile memory. Row lengths sit on
    every chunk, block and slice edge, rows are mostly negative with 1e30 past
    their length, and half the calls also return scores, which must be the
    winners' exact bits.
    """
    self.assertEqual(_pick_partition(b, n, k), p)
    self.assertEqual(_block_chunks(n // p), blk)

    rng = np.random.default_rng(n + k + p)
    lengths = _block_walk_lengths(n, p, blk)
    # At most 16 calls of b rows each: at small b (where p > 1) that is fewer
    # rows than edges, so take them evenly from the sorted list (it always
    # keeps 0 and n).
    budget = 16 * b
    if len(lengths) > budget:
      pick = np.linspace(0, len(lengths) - 1, budget).round().astype(int)
      lengths = [lengths[i] for i in sorted(set(pick))]
    calls = [lengths[i : i + b] for i in range(0, len(lengths), b)]
    for call, batch in enumerate(calls):
      batch = (batch * b)[:b] if len(batch) < b else batch
      scores = _block_walk_rows(rng, b, n, k, batch)
      row_lengths = np.array(batch, np.int32)
      return_scores = call % 2 == 1
      out = sparsecore_topk(
          jnp.array(scores),
          k,
          row_lengths=jnp.array(row_lengths),
          return_scores=return_scores,
      )
      idxs = np.asarray(out[0] if return_scores else out)
      self.assertTrue(np.all(idxs >= -1), f"Invalid index: {idxs[idxs < -1]}")
      self.assertIsNone(_mismatch(scores, idxs, k, row_lengths))
      if return_scores:
        _assert_returned_score_bits(scores, idxs, out[1])

  @parameterized.named_parameters(
      # (b, n, k, p): p is forced, so n / p need not be LANES aligned and the
      # last subcore's slice holds part of the HBM pad, or none of the row.
      ("p8_unaligned_slices", 2, 2128, 128, 8),
      ("p16_unaligned_slices", 2, 4112, 256, 16),
      ("p16_long_unaligned_slices", 2, 262160, 2048, 16),
  )
  def test_forced_partition_unaligned_slices(
      self, b: int, n: int, k: int, p: int
  ):
    """Each subcore's slice is exact when p splits n into unaligned slices.

    Row lengths end in the last slice, at the edge of the second-to-last one
    (so the last subcore gets an empty slice), and inside the first slice.
    """
    self.assertNotEqual((n // p) % LANES, 0)
    rng = np.random.default_rng(n + k + p)
    raw = -(-n // p)
    slice_len = _align_to(raw, LANES * _block_chunks(raw))
    lengths = [
        n,
        n - 1,
        (p - 1) * slice_len,
        (p - 1) * slice_len + 1,
        k + 7,
        slice_len - 1,
        LANES + 3,
        0,
    ]
    for i in range(0, len(lengths), b):
      batch = lengths[i : i + b]
      scores = _block_walk_rows(rng, b, n, k, batch)
      row_lengths = np.array(batch, np.int32)
      idxs, keys = _sc_topk_direct(
          jnp.array(scores),
          k,
          jnp.array(row_lengths),
          p=p,
          return_scores=True,
      )
      self.assertIsNone(_mismatch(scores, np.asarray(idxs), k, row_lengths))
      _assert_returned_score_bits(scores, idxs, keys)


TWO_STAGE = (16, N_128K, 2048)
SINGLE_STAGE = (64, N_SHORT, 2048)


def _stage_group_ids(b, n, k, **kwargs) -> list[int | None]:
  """Returns the scheduling_group_id received by each _sc_topk_direct call."""
  seen: list[int | None] = []

  def _stub(
      scores,
      k_,
      row_lengths,
      *,
      local_idxs=None,
      p=1,
      write_empty=True,
      return_scores=True,
      scheduling_group_id=None,
  ):
    del row_lengths, local_idxs, p, write_empty
    seen.append(scheduling_group_id)
    out = jnp.zeros((scores.shape[0], k_), jnp.int32)
    return (out, out) if return_scores else out

  with mock.patch.object(sc_module, "_sc_topk_direct", _stub):
    sc_module.sparsecore_topk.__wrapped__(jnp.zeros((b, n), jnp.float32), k, **kwargs)  # pyrefly: ignore[missing-attribute]
  return seen


class SparseCoreTopKSchedulingGroupTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 7:
      self.skipTest(
          "SparseCore TopK Pallas TPU kernel requires TPU v7 or newer."
      )

  @parameterized.named_parameters(
      (
          "p8_both",
          TWO_STAGE,
          dict(scheduling_group_id=3, stage2_scheduling_group_id=4),
          [3],
      ),
      ("p8_stage1_only", TWO_STAGE, dict(scheduling_group_id=3), [3]),
      ("p8_stage2_only", TWO_STAGE, dict(stage2_scheduling_group_id=4), [None]),
      ("p8_neither", TWO_STAGE, {}, [None]),
      (
          "p1_both",
          SINGLE_STAGE,
          dict(scheduling_group_id=3, stage2_scheduling_group_id=4),
          [3],
      ),
      (
          "p1_stage2_only",
          SINGLE_STAGE,
          dict(stage2_scheduling_group_id=4),
          [None],
      ),
  )
  def test_each_stage_gets_the_group_id_it_was_given(
      self, shape, kwargs, expected
  ):
    b, n, k = shape
    assert _pick_partition(b, n, k) == (8 if shape == TWO_STAGE else 1)
    assert _stage_group_ids(b, n, k, **kwargs) == expected

  def test_scheduling_group_ids_reach_the_lowered_hlo(self):
    """Verifies scheduling group IDs appear as frontend attributes in lowered HLO."""
    b, n, k = TWO_STAGE
    assert _pick_partition(b, n, k) == 8
    scores = jnp.zeros((b, n), jnp.float32)
    row_lengths = jnp.full((b,), n, jnp.int32)

    def _text(**kwargs):
      return sparsecore_topk.lower(scores, k, row_lengths, **kwargs).as_text()

    both = _text(scheduling_group_id=3, stage2_scheduling_group_id=4)
    assert '_scheduling_group_id = "3"' in both

    # Nothing is annotated unless it was asked for.
    stage1_only = _text(scheduling_group_id=3)
    assert '_scheduling_group_id = "3"' in stage1_only
    assert "_scheduling_group_id" not in _text()

  @parameterized.parameters("random", "heavy_ties")
  def test_scheduling_group_ids_do_not_change_the_selection(self, kind):
    """A scheduling hint that changed the answer would be a correctness bug."""
    b, n, k = TWO_STAGE
    rng = np.random.default_rng(SEED)
    scores = _scores(kind, b, n, rng)
    row_lengths = np.full(b, n, np.int32)

    plain = _run(scores, k, row_lengths)
    grouped = np.asarray(
        jax.block_until_ready(
            sparsecore_topk(
                jnp.asarray(scores),
                k,
                jnp.asarray(row_lengths),
                scheduling_group_id=3,
                stage2_scheduling_group_id=4,
            )
        )
    )

    assert _mismatch(scores, grouped, k, row_lengths) is None
    # Indices may legitimately differ on ties, the selected scores may not.
    np.testing.assert_array_equal(
        np.sort(
            np.take_along_axis(scores, np.maximum(plain, 0), axis=1), axis=1
        ),
        np.sort(
            np.take_along_axis(scores, np.maximum(grouped, 0), axis=1), axis=1
        ),
    )


if __name__ == "__main__":
  absltest.main()
