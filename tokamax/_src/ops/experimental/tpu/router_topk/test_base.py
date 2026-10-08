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
"""Shared correctness tests for MoE router top-k implementations.

The cases are ported from upstream vllm-torchtpu
`tests/kernels/test_router_topk.py`. Upstream compares the kernel with the torch
reference (`layers/core/rowmax_topk.py`), here ported as `reference`, and with
`torch.topk`, here `jax.lax.top_k`. All comparisons are bit-exact, as upstream.
"""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.router_topk import reference

# Qwen3.5 routes each token over 512 experts with top-k 10.
EXPERTS = 512
TOPK = 10


def make_scores(rows: int, experts: int = EXPERTS, seed: int | None = None):
  """Uniform `[0, 1)` float32 scores, as upstream draws them."""
  rng = np.random.default_rng(rows if seed is None else seed)
  return rng.random((rows, experts), dtype=np.float32)


def reference_topk(scores, k: int = TOPK) -> tuple[np.ndarray, np.ndarray]:
  """`reference.rowmax_topk` on the float32 scores, as NumPy arrays."""
  w, i = reference.rowmax_topk(jnp.asarray(scores, jnp.float32), k)
  return np.asarray(w), np.asarray(i)


class RouterTopKTestBase(parameterized.TestCase):
  """Correctness suite shared by all MoE router top-k implementations.

  Subclasses pass the implementation under test as `topk_fn`, called as
  `topk_fn(scores, k)` and returning `(weights, indices)`.
  """

  def __init__(self, *args, topk_fn):
    super().__init__(*args)
    self._topk_fn = topk_fn

  def _run(self, scores, k: int = TOPK) -> tuple[np.ndarray, np.ndarray]:
    w, i = self._topk_fn(jnp.asarray(scores), k)
    rows = scores.shape[0]
    self.assertEqual(w.shape, (rows, k))
    self.assertEqual(i.shape, (rows, k))
    self.assertEqual(w.dtype, jnp.float32)
    self.assertEqual(i.dtype, jnp.int32)
    return np.asarray(w), np.asarray(i)

  def _both(self, scores, k: int = TOPK):
    return self._run(scores, k), reference_topk(scores, k)

  @parameterized.parameters(8, 64, 256, 512, 1024, 1536)
  def test_matches_reference(self, rows):
    (kw, ki), (rw, ri) = self._both(make_scores(rows))
    np.testing.assert_array_equal(kw, rw)
    np.testing.assert_array_equal(ki, ri)

  @parameterized.parameters(17, 33, 250, 1000, 1040, 4384)
  def test_partial_last_block_does_not_leak(self, rows):
    """Row counts that are not a multiple of the block stay exact."""
    (kw, ki), (rw, ri) = self._both(make_scores(rows))
    np.testing.assert_array_equal(kw, rw)
    np.testing.assert_array_equal(ki, ri)
    # A leak from the surplus region shows up as an impossible expert or a
    # sentinel weight first.
    self.assertTrue(((ki >= 0) & (ki < EXPERTS)).all())
    self.assertTrue(np.isfinite(kw).all())

  def test_matches_lax_top_k_on_real_routing_distribution(self):
    rng = np.random.default_rng(99)
    logits = jnp.asarray(
        rng.standard_normal((1024, EXPERTS), dtype=np.float32) * 2.0
    ).astype(jnp.bfloat16)
    scores = np.asarray(jax.nn.softmax(logits.astype(jnp.float32), axis=-1))
    kw, ki = self._run(scores)
    ref_v, _ = jax.lax.top_k(jnp.asarray(scores), TOPK)
    np.testing.assert_array_equal(kw, ref_v)
    gathered = np.take_along_axis(scores, ki, axis=1)
    np.testing.assert_array_equal(gathered, ref_v)

  def test_all_nan_row_selects_k_distinct_experts(self):
    """An all-NaN row selects `TOPK` distinct experts, as a sort does.

    An all-NaN row is the flush sentinel in every column; masking with that
    same sentinel would keep every column tied with the row max, so every pass
    would return column 0 and the row would select one expert `TOPK` times.
    Upstream measured that against the sort path at 512 experts / top-k 10: 1
    distinct expert where `lax.sort` takes 10.
    """
    scores = np.full((3, EXPERTS), np.nan, dtype=np.float32)
    kw, ki = self._run(scores)
    self.assertTrue(np.isnan(kw).all())
    for row in ki:
      self.assertEqual(sorted(row.tolist()), list(range(TOPK)))
    iota = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    _, sort_i = jax.lax.sort(
        (-jnp.asarray(scores), iota), dimension=1, is_stable=False, num_keys=1
    )
    for got, ref in zip(ki, np.asarray(sort_i)[:, :TOPK]):
      self.assertLen(set(got.tolist()), TOPK)
      self.assertLen(set(ref.tolist()), TOPK)

  def test_partial_nan_row_selects_k_distinct_finite_experts(self):
    rng = np.random.default_rng(5)
    scores = rng.random((2, EXPERTS), dtype=np.float32)
    scores[:, :5] = np.nan
    kw, ki = self._run(scores)
    self.assertTrue(np.isfinite(kw).all())
    for row in ki:
      self.assertLen(set(row.tolist()), TOPK)
      self.assertGreaterEqual(min(row.tolist()), 5)

  @parameterized.named_parameters(
      ("nan", np.nan),
      ("neg_inf", -np.inf),
      ("neg_flt_max", np.finfo(np.float32).min),
  )
  def test_row_at_or_below_the_sentinel_selects_k_distinct_experts(self, bad):
    """The same property, for all three spellings.

    `NEG` is above `-inf` and above `-FLT_MAX`, so without the clamp a masked
    column would outrank a genuine one and the row would collapse onto expert
    0 `TOPK` times. The reference is asserted alongside, because fixing one and
    not the other leaves the two agreeing with each other and wrong together.
    """
    scores = np.full((3, EXPERTS), bad, dtype=np.float32)
    kw, ki = self._run(scores)
    self.assertTrue(np.isnan(kw).all())
    for row in ki:
      self.assertEqual(sorted(row.tolist()), list(range(TOPK)))
    tw, ti = reference_topk(scores)
    np.testing.assert_array_equal(ki, ti)
    self.assertTrue(np.isnan(tw).all())

  def test_ties_and_nan_match_the_reference(self):
    scores = np.zeros((4, EXPERTS), dtype=np.float32)
    scores[0] = 1.0 / EXPERTS  # padded token: all tied
    scores[1, 200:212] = 4.0  # tie across the boundary
    scores[2] = np.nan  # fully poisoned row
    scores[3, :3] = np.nan  # partially poisoned
    scores[3, 300:310] = np.linspace(1.0, 0.1, 10)
    (kw, ki), (rw, ri) = self._both(scores)
    np.testing.assert_array_equal(ki, ri)
    self.assertTrue(np.isnan(kw[2]).all())
    self.assertTrue(np.isnan(rw[2]).all())
    keep = [0, 1, 3]
    np.testing.assert_array_equal(kw[keep], rw[keep])
    # Ties resolve to the lowest expert id.
    np.testing.assert_array_equal(ki[0], np.arange(TOPK))
    np.testing.assert_array_equal(ki[1], np.arange(200, 200 + TOPK))

  # Not upstream: other router geometries, e.g. DeepSeek-V3 (256 experts, top-k
  # 8), plus expert counts that are not a multiple of 128 and the `k` extremes.
  @parameterized.parameters(
      (40, 256, 8),
      (300, 384, 6),
      (64, 128, 8),
      (24, 160, 6),
      (33, 100, 7),
      (16, 64, 1),
      (8, 16, 16),
      (520, 1024, 10),
  )
  def test_routing_geometries(self, rows, experts, k):
    scores = make_scores(rows, experts)
    (kw, ki), (rw, ri) = self._both(scores, k)
    np.testing.assert_array_equal(kw, rw)
    np.testing.assert_array_equal(ki, ri)
    ref_v, _ = jax.lax.top_k(jnp.asarray(scores), k)
    np.testing.assert_array_equal(kw, ref_v)
    np.testing.assert_array_equal(np.take_along_axis(scores, ki, 1), ref_v)

  # Not upstream: bfloat16 and float16 scores are cast to float32 exactly, so
  # they select as their float32 values do.
  @parameterized.parameters(jnp.bfloat16, jnp.float16)
  def test_half_precision_scores(self, dtype):
    rng = np.random.default_rng(7)
    logits = jnp.asarray(rng.standard_normal((200, EXPERTS), dtype=np.float32))
    scores = jax.nn.softmax(logits, axis=-1).astype(dtype)
    kw, ki = self._run(scores)
    rw, ri = reference_topk(scores.astype(jnp.float32))
    np.testing.assert_array_equal(kw, rw)
    np.testing.assert_array_equal(ki, ri)
