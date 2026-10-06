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
"""Shared correctness tests for TopK implementations."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.topk import base


def assert_topk_matches_reference(
    scores: np.ndarray,
    actual_indices: np.ndarray | jax.Array,
    k: int,
    row_lengths: np.ndarray | None = None,
    actual_scores_bits: np.ndarray | jax.Array | None = None,
) -> None:
  """Verifies that reference output matches base.TopK output."""
  b, n = scores.shape
  idx = np.asarray(actual_indices, dtype=np.int32)
  assert idx.shape == (b, k), f"Expected shape {(b, k)}, got {idx.shape}"

  if row_lengths is None:
    eff_lengths = np.full((b,), n, dtype=np.int32)
  else:
    eff_lengths = np.asarray(row_lengths, dtype=np.int32)

  base_op = base.TopK()
  ref_indices, ref_scores_bits = base_op(
      jnp.asarray(scores, dtype=jnp.float32),
      k,
      None if row_lengths is None else jnp.asarray(eff_lengths),
      return_scores=True,
  )
  ref_idx = np.asarray(ref_indices, dtype=np.int32)
  ref_scores = np.asarray(
      jax.lax.bitcast_convert_type(ref_scores_bits, jnp.float32)
  )

  for i in range(b):
    sel = idx[i]
    live = sel[sel >= 0]
    n_live = live.size
    assert np.all(sel[:n_live] >= 0) and np.all(
        sel[n_live:] == -1
    ), f"row {i}: -1 padding is not a suffix: {sel}"
    if n_live:
      assert live.max() < int(
          eff_lengths[i]
      ), f"row {i}: index {live.max()} >= row_length {int(eff_lengths[i])}"
      assert (
          np.unique(live).size == n_live
      ), f"row {i}: duplicate indices in {live}"

    ref_live = ref_idx[i][ref_idx[i] >= 0]
    assert (
        n_live == ref_live.size
    ), f"row {i}: got {n_live} valid indices, expected {ref_live.size}"
    if n_live:
      got_vals = np.sort(scores[i, live])[::-1]
      want_vals = ref_scores[i, :n_live]
      np.testing.assert_array_equal(got_vals, want_vals)

  if actual_scores_bits is not None:
    got_scores = np.asarray(
        jax.lax.bitcast_convert_type(
            jnp.asarray(actual_scores_bits, dtype=jnp.int32), jnp.float32
        )
    )
    assert got_scores.shape == (b, k)
    for i in range(b):
      for j in range(k):
        col = int(idx[i, j])
        if col >= 0:
          assert got_scores[i, j] == scores[i, col]
        else:
          assert got_scores[i, j] == -np.inf


class TopKTestBase(parameterized.TestCase):
  """Correctness suite shared by all TopK implementations."""

  def __init__(self, *args, topk_fn):
    super().__init__(*args)
    self._topk_fn = topk_fn

  @parameterized.named_parameters(
      dict(testcase_name="single_stage_b32_n2048_k16", b=32, n=2048, k=16),
      dict(testcase_name="single_stage_b32_n2048_k256", b=32, n=2048, k=256),
      dict(testcase_name="two_stage_b4_n8192_k256", b=4, n=8192, k=256),
      dict(testcase_name="two_stage_b16_n10240_k512", b=16, n=10240, k=512),
  )
  def test_topk_full_rows(self, b: int, n: int, k: int):
    rng = np.random.default_rng(42 + b + n + k)
    scores = rng.standard_normal((b, n), dtype=np.float32)
    actual = self._topk_fn(jnp.asarray(scores), k)
    assert_topk_matches_reference(scores, actual, k)

  @parameterized.named_parameters(
      dict(testcase_name="single_stage_ragged", b=32, n=2048, k=64),
      dict(testcase_name="two_stage_ragged", b=4, n=8192, k=64),
  )
  def test_topk_ragged_row_lengths(self, b: int, n: int, k: int):
    rng = np.random.default_rng(100 + b + n + k)
    scores = rng.standard_normal((b, n), dtype=np.float32)
    # Include empty rows (0), partial rows (< k), and longer rows.
    row_lengths = np.array(
        [0, k // 2, k, n // 2] + list(rng.integers(0, n + 1, size=(b - 4,))),
        dtype=np.int32,
    )
    actual = self._topk_fn(
        jnp.asarray(scores), k, row_lengths=jnp.asarray(row_lengths)
    )
    assert_topk_matches_reference(scores, actual, k, row_lengths=row_lengths)

  def test_topk_neg_inf_and_ties(self):
    b, n, k = 16, 2048, 32
    rng = np.random.default_rng(7)
    scores = rng.integers(-5, 5, size=(b, n), dtype=np.int8).astype(np.float32)
    scores[0, :] = -np.inf
    scores[1, 10:] = -np.inf
    actual = self._topk_fn(jnp.asarray(scores), k)
    assert_topk_matches_reference(scores, actual, k)

  @parameterized.parameters(
      (32, 2048, 64),
      (4, 8192, 64),
  )
  def test_topk_return_scores(self, b: int, n: int, k: int):
    rng = np.random.default_rng(200 + b + n + k)
    scores = rng.standard_normal((b, n), dtype=np.float32)
    row_lengths = rng.integers(k // 2, n + 1, size=(b,), dtype=np.int32)
    actual_idx, actual_scores_bits = self._topk_fn(
        jnp.asarray(scores),
        k,
        row_lengths=jnp.asarray(row_lengths),
        return_scores=True,
    )
    assert_topk_matches_reference(
        scores,
        actual_idx,
        k,
        row_lengths=row_lengths,
        actual_scores_bits=actual_scores_bits,
    )

  def test_topk_int32_bitcast_scores(self):
    b, n, k = 16, 2048, 32
    rng = np.random.default_rng(99)
    scores = rng.standard_normal((b, n), dtype=np.float32)
    scores_i32 = jax.lax.bitcast_convert_type(jnp.asarray(scores), jnp.int32)
    actual = self._topk_fn(scores_i32, k)
    assert_topk_matches_reference(scores, actual, k)
