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
      num_blocks=[1, 2, 4, 8],
      descending=[True, False],
  )
  def test_lane_major_bitonic_sort_and_transpose(self, num_blocks, descending):
    rng = np.random.default_rng(101 + num_blocks * 7 + int(descending))
    num_cols = num_blocks * 128
    # Mix random values with exact ties to exercise tie-breaking on indices.
    scores = rng.integers(-4, 5, size=(8, num_cols)).astype(np.float32)
    scores[0] = rng.standard_normal(num_cols).astype(np.float32)

    @jax.jit
    def _run(scores_arr):
      lanes = lax.broadcasted_iota(jnp.int32, (8, 128), 1)
      valid = jnp.ones((8, 128), dtype=jnp.bool_)
      ords = tuple(
          tc_topk.f32_to_ord_i32(scores_arr[:, r * 128 : (r + 1) * 128], valid)
          for r in range(num_blocks)
      )
      idxs = tuple(jnp.int32(r * 128) + lanes for r in range(num_blocks))
      lm_o, lm_i = tc_topk.bitonic_sort_blocks(
          ords, idxs, descending=descending
      )
      bm_o = tc_topk.lane_major_to_block_major(lm_o)
      bm_i = tc_topk.lane_major_to_block_major(lm_i)
      return lm_i, jnp.concatenate(bm_o, axis=1), jnp.concatenate(bm_i, axis=1)

    lm_i, bm_o, bm_i = _run(jnp.asarray(scores))
    want_idx, want_bits = _reference_topk(scores, num_cols)
    if not descending:
      want_idx = want_idx[:, ::-1]
      want_bits = want_bits[:, ::-1]

    # Verify lane-major layout (`pos = lane * num_blocks + r`) before transpose.
    lm_stacked = np.stack([np.asarray(blk) for blk in lm_i], axis=-1)
    np.testing.assert_array_equal(lm_stacked.reshape(8, num_cols), want_idx)
    # Verify block-major layout (`pos = r * 128 + lane`) after transpose.
    np.testing.assert_array_equal(np.asarray(bm_i), want_idx)
    got_bits = tc_topk.ord_i32_to_f32_bits(bm_o, jnp.ones_like(bm_o, jnp.bool_))
    np.testing.assert_array_equal(np.asarray(got_bits), want_bits)

  @parameterized.product(
      num_blocks=[1, 2, 4],
      desc_a=[True, False],
      desc_b=[True, False],
  )
  def test_bitonic_sort_blocks_pair_matches_single(
      self, num_blocks, desc_a, desc_b
  ):
    rng = np.random.default_rng(
        211 + num_blocks * 11 + 2 * int(desc_a) + int(desc_b)
    )
    num_cols = num_blocks * 128
    scores_a = rng.integers(-3, 4, size=(8, num_cols)).astype(np.float32)
    scores_b = rng.standard_normal((8, num_cols), dtype=np.float32)

    @jax.jit
    def _run(sa, sb):
      lanes = lax.broadcasted_iota(jnp.int32, (8, 128), 1)
      valid = jnp.ones((8, 128), dtype=jnp.bool_)
      a_o = tuple(
          tc_topk.f32_to_ord_i32(sa[:, r * 128 : (r + 1) * 128], valid)
          for r in range(num_blocks)
      )
      b_o = tuple(
          tc_topk.f32_to_ord_i32(sb[:, r * 128 : (r + 1) * 128], valid)
          for r in range(num_blocks)
      )
      idxs = tuple(jnp.int32(r * 128) + lanes for r in range(num_blocks))
      got_a, got_b = tc_topk.bitonic_sort_blocks_pair(
          a_o, idxs, b_o, idxs, desc_a=desc_a, desc_b=desc_b
      )
      want_a = tc_topk.bitonic_sort_blocks(a_o, idxs, descending=desc_a)
      want_b = tc_topk.bitonic_sort_blocks(b_o, idxs, descending=desc_b)
      return got_a, got_b, want_a, want_b

    (ga_o, ga_i), (gb_o, gb_i), (wa_o, wa_i), (wb_o, wb_i) = _run(
        jnp.asarray(scores_a), jnp.asarray(scores_b)
    )
    for got, want in zip(
        ga_o + ga_i + gb_o + gb_i, wa_o + wa_i + wb_o + wb_i, strict=True
    ):
      np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

  @parameterized.product(
      num_blocks=[1, 2, 4],
      arity=[2, 4],
      descending=[True, False],
  )
  def test_bitonic_topk_2r_and_4r_to_r(self, num_blocks, arity, descending):
    rng = np.random.default_rng(
        307 + num_blocks * 13 + arity * 3 + int(descending)
    )
    in_blocks = arity * num_blocks
    num_cols = in_blocks * 128
    k = num_blocks * 128
    scores = rng.standard_normal((8, num_cols), dtype=np.float32)
    scores[1] = rng.integers(-2, 3, size=(num_cols,)).astype(np.float32)

    @jax.jit
    def _run(scores_arr):
      lanes = lax.broadcasted_iota(jnp.int32, (8, 128), 1)
      valid = jnp.ones((8, 128), dtype=jnp.bool_)
      ords = tuple(
          tc_topk.f32_to_ord_i32(scores_arr[:, m * 128 : (m + 1) * 128], valid)
          for m in range(in_blocks)
      )
      idxs = tuple(jnp.int32(m * 128) + lanes for m in range(in_blocks))
      if arity == 2:
        top_o, top_i = tc_topk.bitonic_topk_2r_to_r(
            ords[:num_blocks],
            idxs[:num_blocks],
            ords[num_blocks:],
            idxs[num_blocks:],
            descending=descending,
        )
      else:
        top_o, top_i = tc_topk.bitonic_topk_4r_to_r(
            ords, idxs, descending=descending
        )
      bm_o = tc_topk.lane_major_to_block_major(top_o)
      bm_i = tc_topk.lane_major_to_block_major(top_i)
      return jnp.concatenate(bm_o, axis=1), jnp.concatenate(bm_i, axis=1)

    got_ord, got_idx = _run(jnp.asarray(scores))
    got_bits = tc_topk.ord_i32_to_f32_bits(
        got_ord, jnp.ones_like(got_ord, jnp.bool_)
    )
    want_idx, want_bits = _reference_topk(scores, k)
    if not descending:
      want_idx = want_idx[:, ::-1]
      want_bits = want_bits[:, ::-1]
    np.testing.assert_array_equal(np.asarray(got_idx), want_idx)
    np.testing.assert_array_equal(np.asarray(got_bits), want_bits)

  @parameterized.named_parameters(
      ("exact_m4_desc", 4, 4, True),
      ("divisor_m1_of_4_asc", 1, 4, False),
      ("divisor_m2_of_4_desc", 2, 4, True),
      ("non_divisor_m3_of_4_asc", 3, 4, False),
      ("non_divisor_m3_of_4_desc", 3, 4, True),
  )
  def test_sort_blocks_padded(self, m_blocks, num_blocks, descending):
    rng = np.random.default_rng(401 + m_blocks * 19 + int(descending))
    scores = rng.standard_normal((8, m_blocks * 128), dtype=np.float32)

    @jax.jit
    def _run(scores_arr):
      lanes = lax.broadcasted_iota(jnp.int32, (8, 128), 1)
      valid = jnp.ones((8, 128), dtype=jnp.bool_)
      ords = [
          tc_topk.f32_to_ord_i32(scores_arr[:, m * 128 : (m + 1) * 128], valid)
          for m in range(m_blocks)
      ]
      idxs = [jnp.int32(m * 128) + lanes for m in range(m_blocks)]
      lm_o, lm_i = tc_topk.sort_blocks_padded(
          ords, idxs, num_blocks, descending=descending
      )
      bm_i = tc_topk.lane_major_to_block_major(lm_i)
      return jnp.concatenate(bm_i, axis=1), lm_o

    got_idx, _ = _run(jnp.asarray(scores))
    want_idx, _ = _reference_topk(scores, num_blocks * 128)
    if not descending:
      want_idx = want_idx[:, ::-1]
    np.testing.assert_array_equal(np.asarray(got_idx), want_idx)

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
