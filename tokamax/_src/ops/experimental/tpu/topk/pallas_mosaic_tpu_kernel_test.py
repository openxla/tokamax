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
from tokamax._src.ops.experimental.tpu.topk import pallas_mosaic_tpu_kernel as sc_module
from tokamax._src.ops.experimental.tpu.topk.pallas_mosaic_tpu_kernel import (
    LANES,
    MAX_SLICE_WORDS,
    _align_to,
    _pick_partition,
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
  def test_exact_topk_when_stage_two_needs_its_own_partition(self, b, k):
    """Verifies recursive stage-2 partitioning when p * k_p > MAX_SLICE_WORDS."""
    p = _pick_partition(b, N_1M, k)
    assert (
        p * min(k, N_1M // p) > MAX_SLICE_WORDS
    ), f"{b=} {k=} no longer reaches the recursive stage-2 path (p={p})"
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


TWO_STAGE = (16, N_128K, 2048)
SINGLE_STAGE = (64, N_SHORT, 2048)


def _stage_group_ids(b, n, k, **kwargs) -> list[int | None]:
  """Returns the scheduling_group_id received by each _sc_topk_direct call."""
  seen: list[int | None] = []

  def _stub(
      scores, k_, row_lengths, *, write_empty=True, scheduling_group_id=None
  ):
    del row_lengths, write_empty
    seen.append(scheduling_group_id)
    out = jnp.zeros((scores.shape[0], k_), jnp.int32)
    return out, out

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
          "both",
          TWO_STAGE,
          dict(scheduling_group_id=3, stage2_scheduling_group_id=4),
          [3, 4],
      ),
      ("stage1_only", TWO_STAGE, dict(scheduling_group_id=3), [3, None]),
      ("stage2_only", TWO_STAGE, dict(stage2_scheduling_group_id=4), [None, 4]),
      ("neither", TWO_STAGE, {}, [None, None]),
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
    assert _pick_partition(b, n, k) == (8 if len(expected) == 2 else 1)
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
    assert '_scheduling_group_id = "4"' in both

    # Nothing is annotated unless it was asked for.
    stage1_only = _text(scheduling_group_id=3)
    assert '_scheduling_group_id = "3"' in stage1_only
    assert '_scheduling_group_id = "4"' not in stage1_only
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
