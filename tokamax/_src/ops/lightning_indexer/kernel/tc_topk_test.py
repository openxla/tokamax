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
"""Tests for TensorCore exact Top-K, DCP merge, and non-negative compaction."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.lightning_indexer.kernel import metadata
from tokamax._src.ops.lightning_indexer.kernel import tc_topk


def _reference_topk(
    scores: np.ndarray,
    k: int,
    row_lengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Exact `(score, -index)` Top-K reference matching `jax.lax.top_k`."""
  num_rows, num_cols = scores.shape
  if row_lengths is None:
    eff = np.full((num_rows,), num_cols, dtype=np.int32)
  else:
    eff = np.clip(row_lengths, 0, num_cols)

  masked = np.where(
      np.arange(num_cols)[None, :] < eff[:, None],
      scores.astype(np.float32),
      -np.inf,
  )
  if num_cols < k:
    masked = np.pad(
        masked, ((0, 0), (0, k - num_cols)), constant_values=-np.inf
    )

  top_vals, top_idxs = lax.top_k(jnp.asarray(masked), k)
  top_vals_np = np.asarray(top_vals)
  top_idxs_np = np.asarray(top_idxs, dtype=np.int32)

  # Valid columns `< eff[r]` remain valid even if their value was `-inf`, while
  # masked/padded columns `>= eff[r]` or beyond `k` become `-1` / `-inf`.
  valid_slot = np.arange(k)[None, :] < np.minimum(eff[:, None], k)
  top_idxs_np = np.where(valid_slot, top_idxs_np, -1).astype(np.int32)
  top_vals_np = np.where(valid_slot, top_vals_np, -np.inf).astype(np.float32)
  top_bits_np = top_vals_np.view(np.int32)
  return top_idxs_np, top_bits_np


class TcTopKTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("TensorCore Pallas kernels require TPU backend.")

  def test_f32_ord_bijection_monotonic_and_invertible(self):
    vals = np.array(
        [
            -np.inf,
            -3.4e38,
            -100.5,
            -1.0,
            -1e-38,
            -1.4e-45,
            -0.0,
            +0.0,
            1.4e-45,
            1e-38,
            1.0,
            100.5,
            3.4e38,
            np.inf,
        ],
        dtype=np.float32,
    )
    mask = jnp.ones_like(vals, dtype=jnp.bool_)
    ords = np.asarray(tc_topk.f32_to_ord_i32(jnp.asarray(vals), mask))

    # Every valid value (including -inf) strictly exceeds EMPTY_ORD_SENTINEL.
    self.assertTrue(np.all(ords > tc_topk.EMPTY_ORD_SENTINEL))
    # Matches IEEE-754 totalOrder / `jax.lax.top_k`:
    # ord(-0.0) == -1 < ord(+0.0) == 0.
    self.assertEqual(int(ords[6]), -1)
    self.assertEqual(int(ords[7]), 0)
    # Strictly monotonic across all values (including -0.0 < +0.0).
    for i in range(len(ords) - 1):
      self.assertLess(int(ords[i]), int(ords[i + 1]))

    # Exact bit-for-bit round-trip recovery (preserving -0.0 vs +0.0).
    rec_bits = np.asarray(tc_topk.ord_i32_to_f32_bits(jnp.asarray(ords), mask))
    np.testing.assert_array_equal(rec_bits, vals.view(np.int32))

  @parameterized.product(
      k=[1, 5, 16, 64, 128, 300, 512, 1024, 2048],
      num_cols=[64, 128, 1024, 2304, 16384],
  )
  def test_exact_topk_shapes_and_k_values(self, k, num_cols):
    rng = np.random.default_rng(1000 + k * 31 + num_cols)
    num_rows = 11  # Intentionally non-multiple of 8.
    scores = rng.standard_normal((num_rows, num_cols), dtype=np.float32)
    row_lengths = rng.integers(
        0, num_cols + 1, size=(num_rows,), dtype=np.int32
    )
    row_lengths[0] = 0
    row_lengths[1] = min(num_cols, max(1, k // 2))
    row_lengths[2] = num_cols

    got_idx, got_bits = tc_topk.tensorcore_topk(
        jnp.asarray(scores),
        k,
        row_lengths=jnp.asarray(row_lengths),
        return_scores=True,
    )
    want_idx, want_bits = _reference_topk(scores, k, row_lengths)

    np.testing.assert_array_equal(np.asarray(got_idx), want_idx)
    np.testing.assert_array_equal(np.asarray(got_bits), want_bits)

  @parameterized.named_parameters(
      ("long_64k_k1024", 4, 65536, 1024),
      ("long_262k_k1024", 4, 262144, 1024),
      ("long_64k_k2048", 4, 65536, 2048),
  )
  def test_long_context_streaming(self, num_rows, num_cols, k):
    rng = np.random.default_rng(42 + num_cols + k)
    scores = rng.standard_normal((num_rows, num_cols), dtype=np.float32)
    got_idx = np.asarray(tc_topk.tensorcore_topk(jnp.asarray(scores), k))
    want_idx, _ = _reference_topk(scores, k)
    np.testing.assert_array_equal(got_idx, want_idx)

  @parameterized.named_parameters(
      ("strictly_ascending", "ascending"),
      ("strictly_descending", "descending"),
      ("all_zero_ties", "zeros"),
      ("all_negative", "negative"),
      ("signed_zeros", "signed_zeros"),
      ("single_lane_periodic_spikes", "lane0_spikes"),
  )
  def test_adversarial_distributions(self, mode):
    num_rows, num_cols, k = 9, 8192, 256
    rng = np.random.default_rng(777)
    if mode == "ascending":
      base = np.arange(num_cols, dtype=np.float32)[None, :]
      scores = base + np.arange(num_rows, dtype=np.float32)[:, None] * 0.1
    elif mode == "descending":
      base = np.arange(num_cols, 0, -1, dtype=np.float32)[None, :]
      scores = base + np.arange(num_rows, dtype=np.float32)[:, None] * 0.1
    elif mode == "zeros":
      scores = np.zeros((num_rows, num_cols), dtype=np.float32)
    elif mode == "negative":
      scores = (
          -np.abs(rng.standard_normal((num_rows, num_cols), np.float32)) - 1.0
      )
    elif mode == "signed_zeros":
      scores = np.where(
          rng.random((num_rows, num_cols)) < 0.5,
          np.float32(-0.0),
          np.float32(0.0),
      )
    elif mode == "lane0_spikes":
      # Every hot spike lands on lane 0 (`col % 128 == 0`), maximally stressing
      # staging bucket collisions and secondary 64-lane XOR probing.
      scores = np.full((num_rows, num_cols), -100.0, dtype=np.float32)
      spike_cols = np.arange(0, num_cols, 128)
      scores[:, spike_cols] = rng.uniform(
          1.0, 1000.0, (num_rows, len(spike_cols))
      ).astype(np.float32)
    else:
      raise ValueError(mode)

    got_idx, got_bits = tc_topk.tensorcore_topk(
        jnp.asarray(scores), k, return_scores=True, tile_cols=128
    )
    want_idx, want_bits = _reference_topk(scores, k)
    np.testing.assert_array_equal(np.asarray(got_idx), want_idx)
    np.testing.assert_array_equal(np.asarray(got_bits), want_bits)

  @parameterized.named_parameters(
      ("random_r4", 4, "random"),
      ("ties_r4", 4, "zeros"),
      ("duplicates_r4", 4, "duplicates"),
      ("lane0_overflow_r4", 4, "lane0_overflow"),
      ("random_r16", 16, "random"),
      ("lane0_overflow_r16", 16, "lane0_overflow"),
  )
  def test_lane_bitonic_4r_to_r_with_overflow_fixup(self, r_blocks, mode):
    num_rows = 8
    m_blocks = 4 * r_blocks
    num_cols = m_blocks * 128
    k = r_blocks * 128
    rng = np.random.default_rng(31415 + r_blocks)
    if mode == "random":
      scores = rng.standard_normal((num_rows, num_cols), dtype=np.float32)
    elif mode == "zeros":
      scores = np.zeros((num_rows, num_cols), dtype=np.float32)
    elif mode == "duplicates":
      scores = rng.integers(0, 5, size=(num_rows, num_cols)).astype(np.float32)
    elif mode == "lane0_overflow":
      scores = rng.standard_normal((num_rows, num_cols), dtype=np.float32)
      # Force all 4*R entries in lane 0 into the global Top-K so > 2*R entries
      # come from a single lane and the loser fixup path must recover them.
      scores[:, np.arange(0, num_cols, 128)] += 1000.0
    else:
      raise ValueError(mode)

    @jax.jit
    def _run(scores_arr):
      ords = []
      idxs = []
      lanes = lax.broadcasted_iota(jnp.int32, (8, 128), 1)
      valid = jnp.ones((8, 128), dtype=jnp.bool_)
      for m in range(m_blocks):
        s_tile = scores_arr[:, m * 128 : (m + 1) * 128]
        ords.append(tc_topk.f32_to_ord_i32(s_tile, valid))
        idxs.append(jnp.int32(m * 128) + lanes)
      c0_o, c0_i = tuple(ords[:r_blocks]), tuple(idxs[:r_blocks])
      c1_o, c1_i = (
          tuple(ords[r_blocks : 2 * r_blocks]),
          tuple(idxs[r_blocks : 2 * r_blocks]),
      )
      c2_o, c2_i = (
          tuple(ords[2 * r_blocks : 3 * r_blocks]),
          tuple(idxs[2 * r_blocks : 3 * r_blocks]),
      )
      c3_o, c3_i = tuple(ords[3 * r_blocks :]), tuple(idxs[3 * r_blocks :])
      w_o, w_i, bot_max = tc_topk.reduce_4r_to_2r_lane_bitonic(
          c0_o,
          c0_i,
          c1_o,
          c1_i,
          c2_o,
          c2_i,
          c3_o,
          c3_i,
          ordered_inputs=True,
      )
      top_o, top_i = tc_topk.bitonic_topk_2r_to_r(
          w_o[:r_blocks],
          w_i[:r_blocks],
          w_o[r_blocks:],
          w_i[r_blocks:],
          descending=True,
          bitonic_lanes=True,
      )
      tau_x = tc_topk.exclusive_threshold(
          tc_topk.extract_row_kth_threshold(top_o, k)
      )
      l_o, l_i = tc_topk.reduce_4r_to_2r_lane_losers(
          c0_o,
          c0_i,
          c1_o,
          c1_i,
          c2_o,
          c2_i,
          c3_o,
          c3_i,
          ordered_inputs=True,
      )
      cand_o, cand_i = tc_topk.bitonic_topk_2r_to_r(
          l_o[:r_blocks],
          l_i[:r_blocks],
          l_o[r_blocks:],
          l_i[r_blocks:],
          descending=False,
          bitonic_lanes=True,
      )
      _, fix_i = tc_topk.bitonic_topk_merge_blocks(
          top_o, top_i, cand_o, cand_i, descending=True
      )
      need_fix = jnp.any(bot_max > tau_x)
      out_i = [
          jnp.where(need_fix, f_i, t_i) for f_i, t_i in zip(fix_i, top_i)
      ]
      fin_i = tc_topk.lane_major_to_block_major(tuple(out_i))
      return jnp.concatenate(fin_i, axis=1), need_fix

    got_idx, need_fix = _run(jnp.asarray(scores))
    want_idx, _ = _reference_topk(scores, k)
    if mode in ("lane0_overflow", "zeros", "duplicates"):
      self.assertTrue(bool(need_fix))
    elif r_blocks >= 16:
      self.assertFalse(bool(need_fix))
    np.testing.assert_array_equal(np.asarray(got_idx), want_idx)

  @parameterized.product(
      shape=[(1, 16), (16, 128), (33, 256), (9, 1024), (8, 2048)],
      keep_rate=[0.0, 0.1, 0.5, 0.9, 1.0],
  )
  def test_pack_nonnegative_tc_preserves_order(self, shape, keep_rate):
    num_rows, width = shape
    rng = np.random.default_rng(2026 + num_rows * 17 + width)
    vals = rng.integers(0, 1 << 20, size=(num_rows, width), dtype=np.int32)
    vals[rng.random((num_rows, width)) >= keep_rate] = -1

    # XLA reference pack preserving column relative order.
    keep = vals >= 0
    slot = np.cumsum(keep, axis=1) - keep
    want = np.full((num_rows, width + 1), -1, dtype=np.int32)
    rows = np.arange(num_rows)[:, None]
    want[rows, np.where(keep, slot, width)] = np.where(keep, vals, -1)
    want = want[:, :width]

    got = np.asarray(tc_topk.pack_nonnegative_tc(jnp.asarray(vals)))
    np.testing.assert_array_equal(got, want)

  @parameterized.product(
      dcp_size=[2, 4, 8],
      k=[16, 512, 2048],
      inputs_sorted=[False, True],
      all_ties=[False, True],
  )
  def test_dcp_merge_and_compact_tc_exact_partition(
      self, dcp_size, k, inputs_sorted, all_ties
  ):
    rng = np.random.default_rng(900 + dcp_size * 13 + k)
    num_rows = 10
    span = max(dcp_size * k, 4096)
    interleave_c = 4

    if all_ties:
      scores = np.zeros((num_rows, span), dtype=np.float32)
    else:
      scores = (
          rng.permutation(num_rows * span)
          .reshape(num_rows, span)
          .astype(np.float32)
      )

    owner = (np.arange(span) // interleave_c) % dcp_size
    local_idxs = np.full((dcp_size, num_rows, k), -1, np.int32)
    local_scores = np.full((dcp_size, num_rows, k), -np.inf, np.float32)

    for r in range(dcp_size):
      owned = np.flatnonzero(owner == r)
      for t in range(num_rows):
        # Stable argsort on -scores breaks ties by smaller global index.
        order = np.argsort(-scores[t, owned], kind="stable")[:k]
        best = owned[order]
        if not inputs_sorted:
          perm = rng.permutation(len(best))
          best = best[perm]
        local_idxs[r, t, : len(best)] = np.asarray(
            metadata.cp_global_to_local(
                jnp.asarray(best), dcp_size, interleave_c
            )
        )
        local_scores[r, t, : len(best)] = scores[t, best]

    gathered_bits = np.concatenate(list(local_scores), axis=1).view(np.int32)
    per_rank_out = [
        np.asarray(
            tc_topk.dcp_merge_and_compact_tc(
                jnp.asarray(gathered_bits),
                jnp.asarray(local_idxs[r]),
                k=k,
                dcp_rank=jnp.int32(r),
                dcp_size=dcp_size,
                inputs_sorted=inputs_sorted,
            )
        )
        for r in range(dcp_size)
    ]

    for t in range(num_rows):
      reassembled = []
      for r in range(dcp_size):
        row = per_rank_out[r][t]
        kept = row[row >= 0]
        # Verify -1 padding is strictly a suffix.
        self.assertTrue(np.all(row[len(kept) :] == -1))
        for loc in kept.tolist():
          g = int(metadata.cp_local_to_global(loc, r, dcp_size, interleave_c))
          self.assertEqual((g // interleave_c) % dcp_size, r)
          reassembled.append(g)

      # Exact count k, multiplicity 1 across all ranks.
      self.assertLen(reassembled, min(k, span))
      self.assertLen(set(reassembled), len(reassembled))
      if not all_ties:
        want = set(np.argsort(-scores[t], kind="stable")[:k].tolist())
        self.assertEqual(set(reassembled), want)


if __name__ == "__main__":
  absltest.main()
