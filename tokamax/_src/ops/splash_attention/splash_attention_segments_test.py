# Copyright 2025 The Tokamax Authors.
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

"""Tests for runtime `segment_ids` block pruning in Splash attention."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.splash_attention import base
from tokamax._src.ops.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib
from tokamax._src.ops.splash_attention import splash_attention_mask_info as mask_info_lib


def _ids_from_lengths(lengths: tuple[int, ...]) -> np.ndarray:
  """Builds packed `segment_ids` from a list of document lengths."""
  return np.concatenate(
      [np.full((n,), i, np.int32) for i, n in enumerate(lengths)]
  )


def _segment_ids(lengths: tuple[int, ...]) -> base.SegmentIds:
  ids = jnp.asarray(_ids_from_lengths(lengths))
  return base.SegmentIds(ids, ids)


def _tile_properties(
    q_ids: np.ndarray, kv_ids: np.ndarray, bq: int, bkv: int
) -> np.ndarray:
  """Brute-force `[q_blocks, kv_blocks]` of "this tile shares a segment id"."""
  q_blocks, kv_blocks = len(q_ids) // bq, len(kv_ids) // bkv
  contributes = np.zeros((q_blocks, kv_blocks), bool)
  for i in range(q_blocks):
    qs = q_ids[i * bq : (i + 1) * bq]
    for j in range(kv_blocks):
      kvs = kv_ids[j * bkv : (j + 1) * bkv]
      contributes[i, j] = bool((qs[:, None] == kvs[None, :]).any())
  return contributes


def _causal_mask_info(
    q_seq_len: int,
    kv_seq_len: int,
    bq: int,
    bkv: int,
    *,
    is_dkv: bool,
    dynamic: bool,
):
  """Builds a causal `MaskInfo`, dense (`dynamic=False`) or compacted."""
  mask = mask_lib.CausalMask(shape=(q_seq_len, kv_seq_len))
  process = (
      mask_info_lib.process_mask_dkv if is_dkv else mask_info_lib.process_mask
  )
  mask_info, _ = process(mask, (bq, bkv), return_dynamic_grid=dynamic)
  return jax.tree_util.tree_map(jnp.asarray, mask_info)


def _num_active(mask_info) -> int:
  num_active_blocks = mask_info.num_active_blocks
  assert num_active_blocks is not None
  return int(num_active_blocks[0])


def _scheduled_tiles(mask_info) -> dict[tuple[int, int], int]:
  """Returns `{(row, col): block_mask}` for the active prefix of a block list."""
  n = _num_active(mask_info)
  rows = np.asarray(mask_info.active_rows)[:n]
  cols = np.asarray(mask_info.active_cols)[:n]
  block_mask = np.asarray(mask_info.block_mask)[:n]
  tiles = {}
  for r, c, m in zip(rows.tolist(), cols.tolist(), block_mask.tolist()):
    assert (r, c) not in tiles, f"duplicate block ({r}, {c})"
    tiles[(r, c)] = m
  return tiles


def _unpruned_refine(mask_info, segment_ids, **kwargs):
  """Stand-in for `_refine_mask_info_with_segments` that prunes nothing."""
  del segment_ids, kwargs
  bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)
  return mask_info, bounds_start, bounds_end


class SegmentBlockOverlapTest(parameterized.TestCase):
  """`_segment_block_overlap` must never miss a contributing tile."""

  @parameterized.named_parameters(
      ("aligned", (512, 512, 512, 512), 512, 512),
      ("straddling", (300, 500, 700, 548), 512, 512),
      ("single_segment", (2048,), 512, 512),
      ("many_short", tuple([128] * 16), 512, 512),
      ("rectangular_blocks", (768, 1280), 256, 512),
  )
  def test_matches_bruteforce(self, lengths, bq, bkv):
    seq_len = sum(lengths)
    ids = _ids_from_lengths(lengths)
    contributes = _tile_properties(ids, ids, bq, bkv)
    q_blocks, kv_blocks = seq_len // bq, seq_len // bkv

    seg_any = splash._segment_block_overlap(  # pylint: disable=protected-access
        _segment_ids(lengths),
        jnp.arange(q_blocks)[:, None],
        jnp.arange(kv_blocks)[None, :],
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        bq=bq,
        bkv=bkv,
    )
    np.testing.assert_array_equal(contributes & ~np.asarray(seg_any), False)

  def test_unsorted_ids_are_still_sound(self):
    rng = np.random.default_rng(0)
    seq_len, bq, bkv = 512, 128, 128
    ids = rng.integers(0, 5, size=(seq_len,)).astype(np.int32)
    contributes = _tile_properties(ids, ids, bq, bkv)

    seg_any = splash._segment_block_overlap(  # pylint: disable=protected-access
        base.SegmentIds(jnp.asarray(ids), jnp.asarray(ids)),
        jnp.arange(seq_len // bq)[:, None],
        jnp.arange(seq_len // bkv)[None, :],
        q_blocks=seq_len // bq,
        kv_blocks=seq_len // bkv,
        bq=bq,
        bkv=bkv,
    )
    np.testing.assert_array_equal(contributes & ~np.asarray(seg_any), False)

  def test_vmap_takes_the_union_over_the_batch(self):
    bq = bkv = 512
    seq_len = 1024
    ids = jnp.stack([
        jnp.asarray(_ids_from_lengths((512, 512))),
        jnp.asarray(_ids_from_lengths((1024,))),
    ])

    def overlap(segment_ids):
      return splash._segment_block_overlap(  # pylint: disable=protected-access
          segment_ids,
          jnp.arange(seq_len // bq)[:, None],
          jnp.arange(seq_len // bkv)[None, :],
          q_blocks=seq_len // bq,
          kv_blocks=seq_len // bkv,
          bq=bq,
          bkv=bkv,
      )

    seg_any = jax.vmap(overlap)(base.SegmentIds(ids, ids))
    self.assertEqual(seg_any.shape, (2, 2, 2))
    np.testing.assert_array_equal(
        np.asarray(seg_any[0]), np.asarray(seg_any[1])
    )
    np.testing.assert_array_equal(np.asarray(seg_any), True)

  def test_bitmask_and_interval_prune_non_monotonic_and_large_ids(self):
    bq = bkv = 128
    blocks = [
        np.concatenate([np.zeros(64, np.int32), np.full(64, 5, np.int32)]),
        np.full(128, 2, np.int32),
        np.full(128, 1, np.int32),
        np.full(128, 257, np.int32),
        -np.arange(1, 129, dtype=np.int32),
        -np.arange(129, 257, dtype=np.int32),
    ]
    ids = jnp.asarray(np.concatenate(blocks))
    n_blocks = len(blocks)
    seg_ids = base.SegmentIds(ids, ids)

    dense = np.asarray(
        splash._segment_block_overlap(  # pylint: disable=protected-access
            seg_ids,
            q_blocks=n_blocks,
            kv_blocks=n_blocks,
            bq=bq,
            bkv=bkv,
        )
    )
    indexed = np.asarray(
        splash._segment_block_overlap(  # pylint: disable=protected-access
            seg_ids,
            jnp.arange(n_blocks)[:, None],
            jnp.arange(n_blocks)[None, :],
            q_blocks=n_blocks,
            kv_blocks=n_blocks,
            bq=bq,
            bkv=bkv,
        )
    )
    np.testing.assert_array_equal(dense, indexed)
    expected = _tile_properties(np.asarray(ids), np.asarray(ids), bq, bkv)
    np.testing.assert_array_equal(dense, expected)


class RefineMaskInfoWithSegmentsTest(parameterized.TestCase):
  """Schedule tests for `_refine_mask_info_with_segments` (fwd and dkv)."""

  def _refine(self, mask_info, segment_ids, *, seq_len, bq, bkv, is_dkv=False):
    return splash._refine_mask_info_with_segments(  # pylint: disable=protected-access
        mask_info,
        segment_ids,
        q_blocks=seq_len // bq,
        kv_blocks=seq_len // bkv,
        bq=bq,
        bkv=bkv,
        is_dkv=is_dkv,
    )

  @parameterized.named_parameters(
      ("fwd_dense_aligned", (512, 512, 512, 512), 512, 512, False, False),
      ("fwd_dense_straddling", (300, 500, 700, 548), 512, 512, False, False),
      ("fwd_compacted_aligned", (512, 512, 512, 512), 512, 512, True, False),
      ("fwd_compacted_straddling", (300, 500, 700, 548), 512, 512, True, False),
      ("dkv_dense_aligned", (512, 512, 512, 512), 512, 512, False, True),
      ("dkv_compacted_straddling", (300, 500, 700, 548), 512, 512, True, True),
  )
  def test_schedule_is_sound_and_well_formed(
      self, lengths, bq, bkv, dynamic, is_dkv
  ):
    seq_len = sum(lengths)
    ids = _ids_from_lengths(lengths)
    contributes = _tile_properties(ids, ids, bq, bkv)
    q_blocks, kv_blocks = seq_len // bq, seq_len // bkv
    causal = np.tril(np.ones((seq_len, seq_len), bool))
    static_live = np.zeros((q_blocks, kv_blocks), bool)
    for i in range(q_blocks):
      for j in range(kv_blocks):
        static_live[i, j] = causal[
            i * bq : (i + 1) * bq, j * bkv : (j + 1) * bkv
        ].any()

    mask_info = _causal_mask_info(
        seq_len, seq_len, bq, bkv, is_dkv=is_dkv, dynamic=dynamic
    )
    refined, bounds_start, bounds_end = self._refine(
        mask_info,
        _segment_ids(lengths),
        seq_len=seq_len,
        bq=bq,
        bkv=bkv,
        is_dkv=is_dkv,
    )
    tiles = _scheduled_tiles(refined)

    for i in range(q_blocks):
      for j in range(kv_blocks):
        coord = (j, i) if is_dkv else (i, j)
        if static_live[i, j] and contributes[i, j]:
          self.assertIn(coord, tiles, f"dropped contributing block {coord}")
          self.assertNotEqual(tiles[coord], 0, f"zeroed block {coord}")

    num_rows = kv_blocks if is_dkv else q_blocks
    self.assertCountEqual({r for r, _ in tiles}, range(num_rows))

    n = _num_active(refined)
    rows = np.asarray(refined.active_rows)[:n]
    self.assertTrue((np.diff(rows) >= 0).all(), f"rows not sorted: {rows}")
    np.testing.assert_array_equal(
        np.asarray(bounds_start)[:n].astype(bool),
        np.concatenate([[True], rows[1:] != rows[:-1]]),
    )
    np.testing.assert_array_equal(
        np.asarray(bounds_end)[:n].astype(bool),
        np.concatenate([rows[1:] != rows[:-1], [True]]),
    )

  @parameterized.named_parameters(
      ("fwd_dense", False, False),
      ("fwd_compacted", True, False),
      ("dkv_dense", False, True),
      ("dkv_compacted", True, True),
  )
  def test_packing_actually_prunes(self, dynamic, is_dkv):
    bq = bkv = 512
    lengths = (512, 512, 512, 512)
    seq_len = sum(lengths)
    mask_info = _causal_mask_info(
        seq_len, seq_len, bq, bkv, is_dkv=is_dkv, dynamic=dynamic
    )
    refined, _, _ = self._refine(
        mask_info,
        _segment_ids(lengths),
        seq_len=seq_len,
        bq=bq,
        bkv=bkv,
        is_dkv=is_dkv,
    )
    self.assertEqual(_num_active(refined), 4)

  def test_row_with_no_matching_segment_is_still_scheduled(self):
    bq = bkv = 128
    seq_len = 512
    q_ids = np.concatenate([np.zeros(256, np.int32), np.full(256, 9, np.int32)])
    kv_ids = np.zeros(seq_len, np.int32)
    segment_ids = base.SegmentIds(jnp.asarray(q_ids), jnp.asarray(kv_ids))

    mask_info = _causal_mask_info(
        seq_len, seq_len, bq, bkv, is_dkv=False, dynamic=False
    )
    refined, _, _ = self._refine(
        mask_info, segment_ids, seq_len=seq_len, bq=bq, bkv=bkv
    )
    tiles = _scheduled_tiles(refined)
    self.assertCountEqual({i for i, _ in tiles}, range(seq_len // bq))
    for i in (2, 3):
      row = {c: m for (r, c), m in tiles.items() if r == i}
      self.assertLen(row, 1)
      self.assertEqual(next(iter(row.values())), 0)

  def test_fully_empty_schedule_collapses_to_zero_blocks(self):
    bq = bkv = 128
    seq_len = 256
    mask_info = _causal_mask_info(
        seq_len, seq_len, bq, bkv, is_dkv=False, dynamic=False
    )
    refined, _, _ = self._refine(
        mask_info,
        base.SegmentIds(
            jnp.zeros((seq_len,), jnp.int32), jnp.ones((seq_len,), jnp.int32)
        ),
        seq_len=seq_len,
        bq=bq,
        bkv=bkv,
    )
    self.assertEqual(_num_active(refined), 0)

  def test_dkv_always_keeps_a_block_per_row(self):
    bq = bkv = 128
    seq_len = 256
    mask_info = _causal_mask_info(
        seq_len, seq_len, bq, bkv, is_dkv=True, dynamic=True
    )
    refined, _, _ = self._refine(
        mask_info,
        base.SegmentIds(
            jnp.zeros((seq_len,), jnp.int32), jnp.ones((seq_len,), jnp.int32)
        ),
        seq_len=seq_len,
        bq=bq,
        bkv=bkv,
        is_dkv=True,
    )
    tiles = _scheduled_tiles(refined)
    self.assertCountEqual({r for r, _ in tiles}, range(seq_len // bkv))
    self.assertTrue(all(m == 0 for m in tiles.values()))

  @parameterized.named_parameters(
      ("fwd_dense", False, False),
      ("fwd_compacted", True, False),
      ("dkv_dense", False, True),
      ("dkv_compacted", True, True),
  )
  def test_lowers_without_gather(self, dynamic, is_dkv):
    # pylint: disable=protected-access
    # TPU7x offloads large gathers to SparseCore, which fails to lower them for
    # schedule-sized index arrays (T >= 16K), so refine must avoid gathers.
    bq = bkv = 128
    lengths = (300, 500, 700, 548)
    seq_len = sum(lengths)
    mask_info = _causal_mask_info(
        seq_len, seq_len, bq, bkv, is_dkv=is_dkv, dynamic=dynamic
    )
    hlo = (
        jax.jit(
            lambda s: self._refine(
                mask_info, s, seq_len=seq_len, bq=bq, bkv=bkv, is_dkv=is_dkv
            )
        )
        .lower(_segment_ids(lengths))
        .as_text()
    )
    self.assertNotIn("gather", hlo)

    overlap_hlo = (
        jax.jit(
            lambda s: splash._segment_block_overlap(
                s,
                None if not dynamic else jnp.arange(seq_len // bq),
                None if not dynamic else jnp.arange(seq_len // bkv),
                q_blocks=seq_len // bq,
                kv_blocks=seq_len // bkv,
                bq=bq,
                bkv=bkv,
            )
        )
        .lower(_segment_ids(lengths))
        .as_text()
    )
    self.assertNotIn("gather", overlap_hlo)
    self.assertNotIn("scatter", overlap_hlo)

  def test_dense_refine_downcasts_indices_and_chunks_within_smem_budget(self):
    # pylint: disable=protected-access
    bq = bkv = 128
    seq_len = 512
    full_mask = mask_lib.FullMask(_shape=(seq_len, seq_len))
    mask_info, _ = mask_info_lib.process_mask(
        full_mask, (bq, bkv), return_dynamic_grid=False
    )
    mask_info = jax.tree_util.tree_map(jnp.asarray, mask_info)
    refined, _, _ = splash._refine_mask_info_with_segments(
        mask_info,
        _segment_ids((128, 128, 128, 128)),
        q_blocks=seq_len // bq,
        kv_blocks=seq_len // bkv,
        bq=bq,
        bkv=bkv,
        is_dkv=False,
    )
    assert refined.active_rows is not None and refined.active_cols is not None
    self.assertEqual(refined.active_rows.dtype, jnp.int16)
    self.assertEqual(refined.active_cols.dtype, jnp.int16)
    self.assertEqual(_num_active(refined), 4)

    empty_with_sinks, _, _ = splash._refine_mask_info_with_segments(
        mask_info,
        base.SegmentIds(
            jnp.zeros((seq_len,), jnp.int32), jnp.ones((seq_len,), jnp.int32)
        ),
        q_blocks=seq_len // bq,
        kv_blocks=seq_len // bkv,
        bq=bq,
        bkv=bkv,
        is_dkv=False,
        keep_empty_rows=True,
    )
    tiles = _scheduled_tiles(empty_with_sinks)
    self.assertCountEqual({r for r, _ in tiles}, range(seq_len // bq))
    self.assertTrue(all(m == 0 for m in tiles.values()))

    q_blocks = kv_blocks = 471
    entry_bytes = splash._refined_entry_bytes(mask_info, q_blocks, kv_blocks)
    self.assertEqual(entry_bytes, 13)
    chunk = splash._q_blocks_per_chunk(
        mask_info, q_blocks, kv_blocks, is_dkv=False
    )
    self.assertLess(chunk, q_blocks)
    self.assertLessEqual(
        chunk * kv_blocks * entry_bytes, splash._MAX_SCALAR_PREFETCH_BYTES
    )

  def test_padded_compacted_shards_keep_empty_rows(self):
    # pylint: disable=protected-access
    bq = bkv = 128
    shard_seq_len = 256
    q_blocks = kv_blocks = shard_seq_len // bq
    causal_mask = mask_lib.CausalMask(
        shape=(2 * shard_seq_len, 2 * shard_seq_len)
    )
    sharded_info, _ = mask_info_lib.process_mask(
        causal_mask,
        (bq, bkv),
        q_seq_shards=2,
        kv_seq_shards=2,
        return_dynamic_grid=True,
    )

    def _shard(q_shard: int, kv_shard: int) -> mask_info_lib.MaskInfo:
      idx = q_shard * 2 + kv_shard

      def _slice(x):
        if x is None:
          return None
        arr = jnp.asarray(x)
        step = arr.shape[0] // 4
        return arr[idx * step : (idx + 1) * step]

      return mask_info_lib.MaskInfo(
          mask_next=_slice(sharded_info.mask_next),
          active_rows=_slice(sharded_info.active_rows),
          active_cols=_slice(sharded_info.active_cols),
          block_mask=_slice(sharded_info.block_mask),
          num_active_blocks=_slice(sharded_info.num_active_blocks),
          partial_mask_blocks=sharded_info.partial_mask_blocks,
          q_sequence=None,
      )

    disjoint_ids = base.SegmentIds(
        jnp.zeros((shard_seq_len,), jnp.int32),
        jnp.ones((shard_seq_len,), jnp.int32),
    )

    # Above-diagonal causal shard: `num_active_blocks == [0]`, `-1`-padded.
    upper_right_refined, _, _ = splash._refine_mask_info_with_segments(
        _shard(0, 1),
        _segment_ids((128, 128)),
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        bq=bq,
        bkv=bkv,
        is_dkv=False,
        keep_empty_rows=True,
    )
    self.assertEqual(_num_active(upper_right_refined), 0)

    # Diagonal causal shard: `0 < num_active_blocks < m_size`, `-1`-padded.
    diag_pruned, _, _ = splash._refine_mask_info_with_segments(
        _shard(0, 0),
        disjoint_ids,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        bq=bq,
        bkv=bkv,
        is_dkv=False,
        keep_empty_rows=True,
    )
    diag_tiles = _scheduled_tiles(diag_pruned)
    self.assertCountEqual({r for r, _ in diag_tiles}, range(q_blocks))
    self.assertTrue(all(m == 0 for m in diag_tiles.values()))

    diag_active, _, _ = splash._refine_mask_info_with_segments(
        _shard(0, 0),
        _segment_ids((128, 128)),
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        bq=bq,
        bkv=bkv,
        is_dkv=False,
        keep_empty_rows=True,
    )
    self.assertEqual(_num_active(diag_active), 2)
    self.assertCountEqual(_scheduled_tiles(diag_active), [(0, 0), (1, 1)])

    # Partially populated schedule missing row 0 collapses to 0 when pruned.
    missing_row0 = mask_info_lib.MaskInfo(
        mask_next=None,
        active_rows=jnp.array([1, -1, -1, -1], dtype=jnp.int8),
        active_cols=jnp.array([0, -1, -1, -1], dtype=jnp.int8),
        block_mask=jnp.array([2, -1, -1, -1], dtype=jnp.int8),
        num_active_blocks=jnp.array([1], dtype=jnp.int32),
        partial_mask_blocks=None,
        q_sequence=None,
    )
    missing_row0_refined, _, _ = splash._refine_mask_info_with_segments(
        missing_row0,
        disjoint_ids,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        bq=bq,
        bkv=bkv,
        is_dkv=False,
        keep_empty_rows=True,
    )
    self.assertEqual(_num_active(missing_row0_refined), 0)


class SplashAttentionSegmentsNumericsTest(parameterized.TestCase):
  """End-to-end TPU numerics tests for `segment_ids` block pruning."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("End-to-end Pallas Splash kernel requires TPU.")

  @parameterized.named_parameters(
      ("causal_aligned_mha", True, (256, 256, 256, 256), 2, 2, False),
      ("causal_straddling_mha", True, (180, 332, 200, 312), 2, 2, False),
      ("causal_aligned_gqa", True, (256, 256, 256, 256), 4, 2, False),
      ("causal_short_1step_gqa", True, (64, 64), 4, 2, False),
      ("causal_short_2step_gqa", True, (128, 128), 4, 2, False),
      ("causal_short_3step_gqa", True, (128, 128, 128), 4, 2, False),
      ("causal_short_1step_mqa", True, (64, 64), 4, 1, True),
      ("causal_short_2step_mqa", True, (128, 128), 4, 1, True),
      ("full_aligned_mha", False, (256, 256, 256, 256), 2, 2, False),
      ("full_straddling_gqa", False, (180, 332, 200, 312), 4, 2, False),
  )
  def test_fwd_and_bwd_match_reference(
      self, is_causal, lengths, num_q_heads, num_kv_heads, is_mqa
  ):
    seq_len = sum(lengths)
    head_dim = 128
    bq = bkv = 128
    kv_shape = (
        (seq_len, head_dim) if is_mqa else (num_kv_heads, seq_len, head_dim)
    )
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(
        k1, (num_q_heads, seq_len, head_dim), jnp.bfloat16
    ) / np.sqrt(head_dim).astype(jnp.bfloat16)
    k = jax.random.normal(k2, kv_shape, jnp.bfloat16) * jnp.bfloat16(0.5)
    v = jax.random.normal(k3, kv_shape, jnp.bfloat16) * jnp.bfloat16(0.5)
    do = jax.random.normal(
        k4, (num_q_heads, seq_len, head_dim), jnp.bfloat16
    ) * jnp.bfloat16(0.5)
    segment_ids = _segment_ids(lengths)

    mask = (
        mask_lib.CausalMask(shape=(seq_len, seq_len))
        if is_causal
        else mask_lib.FullMask(_shape=(seq_len, seq_len))
    )
    config = splash.SplashConfig(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        block_q_dkv=bq,
        block_kv_dkv=bkv,
        block_kv_dkv_compute=bkv,
    )
    make_attn = (
        splash.make_splash_mqa_single_device
        if is_mqa
        else splash.make_splash_mha_single_device
    )
    attn = make_attn(mask, config=config)
    o, attn_vjp = jax.vjp(attn, q, k, v, segment_ids)
    dq, dk, dv, _ = attn_vjp(do)

    dense_mask = jnp.asarray(mask[:, :])
    q32, k32, v32, do32 = jax.tree.map(
        lambda x: x.astype(jnp.float32), (q, k, v, do)
    )
    o_ref, stats_ref = base.attention_reference(
        q32,
        k32,
        v32,
        dense_mask,
        segment_ids,
        None,
        is_mqa=is_mqa,
        save_residuals=True,
    )
    dq_ref, dk_ref, dv_ref, _ = base.attention_reference_vjp(
        do32,
        q32,
        k32,
        v32,
        dense_mask,
        segment_ids,
        None,
        o_ref,
        stats_ref["logsumexp"],
        is_mqa=is_mqa,
    )

    np.testing.assert_allclose(o, o_ref, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(dq, dq_ref, atol=5e-2, rtol=5e-2)
    np.testing.assert_allclose(dk, dk_ref, atol=5e-2, rtol=5e-2)
    np.testing.assert_allclose(dv, dv_ref, atol=5e-2, rtol=5e-2)

  def test_orphan_segment_empty_rows_write_zero_without_nan(self):
    # First half [0:256] has segment id 0 in both q and kv.
    # Second half [256:512] has segment id 9 in q and segment id 7 in kv, so
    # blocks 2 and 3 in both fwd (q rows) and dkv (kv rows) have NO active
    # tiles and rely on the retained `block_mask == 0` dummy step to zero-init
    # and write their outputs.
    seq_len, head_dim, bq, bkv = 512, 128, 128, 128
    k1, k2, k3, k4 = jax.random.split(jax.random.key(1), 4)
    q = jax.random.normal(k1, (2, seq_len, head_dim), jnp.bfloat16) / np.sqrt(
        head_dim
    ).astype(jnp.bfloat16)
    k = jax.random.normal(k2, (2, seq_len, head_dim), jnp.bfloat16)
    v = jax.random.normal(k3, (2, seq_len, head_dim), jnp.bfloat16)
    # Zero upstream gradient on the orphan q rows so only the well-defined
    # active half contributes to gradients, while still testing that the kernel
    # writes clean zeros (not uninitialized HBM garbage) on empty q/kv rows.
    do_active = jax.random.normal(k4, (2, 256, head_dim), jnp.bfloat16)
    do = jnp.concatenate([do_active, jnp.zeros_like(do_active)], axis=1)

    q_ids = jnp.concatenate(
        [jnp.zeros(256, jnp.int32), np.full(256, 9, jnp.int32)]
    )
    kv_ids = jnp.concatenate(
        [jnp.zeros(256, jnp.int32), np.full(256, 7, jnp.int32)]
    )
    segment_ids = base.SegmentIds(q_ids, kv_ids)

    mask = mask_lib.CausalMask(shape=(seq_len, seq_len))
    config = splash.SplashConfig(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        block_q_dkv=bq,
        block_kv_dkv=bkv,
        block_kv_dkv_compute=bkv,
    )
    attn = splash.make_splash_mha_single_device(mask, config=config)
    o, attn_vjp = jax.vjp(attn, q, k, v, segment_ids)
    dq, dk, dv, _ = attn_vjp(do)

    # Empty q rows in fwd must be zero-initialized (not NaN or garbage), and
    # empty kv rows in bwd must have exact 0 gradients.
    self.assertFalse(bool(jnp.any(jnp.isnan(o))))
    np.testing.assert_array_equal(np.asarray(o[:, 256:, :]), 0.0)
    np.testing.assert_array_equal(np.asarray(dk[:, 256:, :]), 0.0)
    np.testing.assert_array_equal(np.asarray(dv[:, 256:, :]), 0.0)

    # Active half [0:256] must match a standalone 256-token reference.
    dense_half = jnp.tril(jnp.ones((256, 256), dtype=bool))
    seg_half = base.SegmentIds(
        jnp.zeros(256, jnp.int32), jnp.zeros(256, jnp.int32)
    )
    q_h, k_h, v_h, do_h = jax.tree.map(
        lambda x: x[:, :256, :].astype(jnp.float32), (q, k, v, do)
    )
    o_ref, stats_ref = base.attention_reference(
        q_h,
        k_h,
        v_h,
        dense_half,
        seg_half,
        None,
        is_mqa=False,
        save_residuals=True,
    )
    dq_ref, dk_ref, dv_ref, _ = base.attention_reference_vjp(
        do_h,
        q_h,
        k_h,
        v_h,
        dense_half,
        seg_half,
        None,
        o_ref,
        stats_ref["logsumexp"],
        is_mqa=False,
    )
    np.testing.assert_allclose(o[:, :256, :], o_ref, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(dq[:, :256, :], dq_ref, atol=8e-2, rtol=8e-2)
    np.testing.assert_allclose(dk[:, :256, :], dk_ref, atol=8e-2, rtol=8e-2)
    np.testing.assert_allclose(dv[:, :256, :], dv_ref, atol=8e-2, rtol=8e-2)

  def test_vmap_over_heterogeneous_segment_packings(self):
    seq_len, head_dim, bq, bkv = 512, 128, 128, 128
    k1, k2, k3, k4 = jax.random.split(jax.random.key(2), 4)
    q = jax.random.normal(
        k1, (2, 2, seq_len, head_dim), jnp.bfloat16
    ) / np.sqrt(head_dim).astype(jnp.bfloat16)
    k = jax.random.normal(k2, (2, 2, seq_len, head_dim), jnp.bfloat16)
    v = jax.random.normal(k3, (2, 2, seq_len, head_dim), jnp.bfloat16)
    do = jax.random.normal(k4, (2, 2, seq_len, head_dim), jnp.bfloat16)
    ids = jnp.stack([
        jnp.asarray(_ids_from_lengths((128, 128, 128, 128))),
        jnp.asarray(_ids_from_lengths((512,))),
    ])
    segment_ids = base.SegmentIds(ids, ids)

    mask = mask_lib.CausalMask(shape=(seq_len, seq_len))
    config = splash.SplashConfig(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        block_q_dkv=bq,
        block_kv_dkv=bkv,
        block_kv_dkv_compute=bkv,
    )
    attn = jax.vmap(splash.make_splash_mha_single_device(mask, config=config))
    o, attn_vjp = jax.vjp(attn, q, k, v, segment_ids)
    dq, dk, dv, _ = attn_vjp(do)

    dense_mask = jnp.asarray(mask[:, :])
    for b in range(2):
      seg_b = base.SegmentIds(ids[b], ids[b])
      q_b, k_b, v_b, do_b = jax.tree.map(
          lambda x, idx=b: x[idx].astype(jnp.float32), (q, k, v, do)
      )
      o_ref, stats_ref = base.attention_reference(
          q_b,
          k_b,
          v_b,
          dense_mask,
          seg_b,
          None,
          is_mqa=False,
          save_residuals=True,
      )
      dq_ref, dk_ref, dv_ref, _ = base.attention_reference_vjp(
          do_b,
          q_b,
          k_b,
          v_b,
          dense_mask,
          seg_b,
          None,
          o_ref,
          stats_ref["logsumexp"],
          is_mqa=False,
      )
      np.testing.assert_allclose(o[b], o_ref, atol=2e-2, rtol=2e-2)
      np.testing.assert_allclose(dq[b], dq_ref, atol=8e-2, rtol=8e-2)
      np.testing.assert_allclose(dk[b], dk_ref, atol=8e-2, rtol=8e-2)
      np.testing.assert_allclose(dv[b], dv_ref, atol=8e-2, rtol=8e-2)

  @parameterized.named_parameters(
      ("static_dkv_grid", None),
      ("dynamic_dkv_grid", 3),
  )
  def test_dropout_matches_the_unpruned_schedule(self, dq_reduction_steps):
    # Dropout bits are keyed on absolute canonical-block coordinates, so
    # skipping whole tiles must not change which surviving weights are dropped.
    lengths = (180, 332, 200, 312)
    seq_len, head_dim, bq, bkv = sum(lengths), 128, 128, 128
    k1, k2, k3, k4 = jax.random.split(jax.random.key(3), 4)
    q = jax.random.normal(k1, (2, seq_len, head_dim), jnp.bfloat16) / np.sqrt(
        head_dim
    ).astype(jnp.bfloat16)
    k = jax.random.normal(k2, (2, seq_len, head_dim), jnp.bfloat16)
    v = jax.random.normal(k3, (2, seq_len, head_dim), jnp.bfloat16)
    do = jax.random.normal(k4, (2, seq_len, head_dim), jnp.bfloat16)
    segment_ids = _segment_ids(lengths)
    prng_key = jax.random.key(7)

    mask = mask_lib.CausalMask(shape=(seq_len, seq_len))
    config = splash.SplashConfig(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        block_q_dkv=bq,
        block_kv_dkv=bkv,
        block_kv_dkv_compute=bkv,
        dropout_rate=0.25,
        dropout_block_q=bq,
        dropout_block_kv=bkv,
        dq_reduction_steps=dq_reduction_steps,
    )
    attn = splash.make_splash_mha_single_device(mask, config=config)

    def fwd_and_bwd():
      jax.clear_caches()  # `_splash_attention` is jitted; force a retrace.
      o, attn_vjp = jax.vjp(
          lambda *qkv: attn(*qkv, segment_ids, prng_key=prng_key), q, k, v
      )
      return (o, *attn_vjp(do))

    o, dq, dk, dv = fwd_and_bwd()
    with mock.patch.object(
        splash, "_refine_mask_info_with_segments", _unpruned_refine
    ):
      o_ref, dq_ref, dk_ref, dv_ref = fwd_and_bwd()
    jax.clear_caches()

    # Skipped tiles only ever add exact zeros, so any dropout bit that moved
    # would show up here.
    np.testing.assert_array_equal(o, o_ref)
    np.testing.assert_array_equal(dk, dk_ref)
    np.testing.assert_array_equal(dv, dv_ref)
    # A pruned dkv grid is dynamic, and a dynamic grid always accumulates dq in
    # three slots of q's dtype (bf16 here) picked by grid step; pruning
    # renumbers the steps, so dq only matches up to bf16 rounding.
    np.testing.assert_allclose(dq, dq_ref, atol=8e-2, rtol=8e-2)

  @parameterized.named_parameters(
      ("full_mha", False, 2, 2, False, False, False),
      ("full_gqa_sinks", False, 4, 2, False, True, False),
      ("full_mqa", False, 4, 1, True, False, False),
      ("full_vmap", False, 2, 2, False, False, True),
      ("causal_gqa_dkv_chunked", True, 4, 2, False, False, False),
  )
  def test_q_chunked_fwd_and_bwd_match_reference(
      self, is_causal, num_q_heads, num_kv_heads, is_mqa, use_sinks, use_vmap
  ):
    seq_len, head_dim, bq, bkv = 640, 128, 128, 128
    batch_prefix = (2,) if use_vmap else ()
    kv_shape = (
        (*batch_prefix, seq_len, head_dim)
        if is_mqa
        else (*batch_prefix, num_kv_heads, seq_len, head_dim)
    )
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(11), 5)
    q = jax.random.normal(
        k1, (*batch_prefix, num_q_heads, seq_len, head_dim), jnp.bfloat16
    ) / np.sqrt(head_dim).astype(jnp.bfloat16)
    k = jax.random.normal(k2, kv_shape, jnp.bfloat16) * jnp.bfloat16(0.5)
    v = jax.random.normal(k3, kv_shape, jnp.bfloat16) * jnp.bfloat16(0.5)
    do = jax.random.normal(
        k4, (*batch_prefix, num_q_heads, seq_len, head_dim), jnp.bfloat16
    ) * jnp.bfloat16(0.5)
    sinks = (
        jax.random.normal(k5, (*batch_prefix, num_q_heads), jnp.float32)
        if use_sinks
        else None
    )
    base_ids = np.concatenate([
        np.zeros(60, np.int32),
        np.full(140, 5, np.int32),
        np.full(200, 2, np.int32),
        np.full(176, 5, np.int32),
        np.zeros(64, np.int32),
    ])
    if use_vmap:
      alt_ids = _ids_from_lengths((320, 320))
      ids = jnp.stack([jnp.asarray(base_ids), jnp.asarray(alt_ids)])
    else:
      ids = jnp.asarray(base_ids)
    segment_ids = base.SegmentIds(ids, ids)

    mask = (
        mask_lib.CausalMask(shape=(seq_len, seq_len))
        if is_causal
        else mask_lib.FullMask(_shape=(seq_len, seq_len))
    )
    config = splash.SplashConfig(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        block_q_dkv=bq,
        block_kv_dkv=bkv,
        block_kv_dkv_compute=bkv,
    )
    make_attn = (
        splash.make_splash_mqa_single_device
        if is_mqa
        else splash.make_splash_mha_single_device
    )
    # 5 q blocks with a 2-block SMEM budget forces uneven 2 + 2 + 1 chunking.
    with mock.patch.object(splash, "_MAX_SCALAR_PREFETCH_BYTES", 2 * 5 * 13):
      jax.clear_caches()
      attn = make_attn(mask, config=config)
      if use_vmap:
        attn = jax.vmap(attn)
      o, attn_vjp = jax.vjp(
          lambda q_, k_, v_, s_: attn(q_, k_, v_, segment_ids, sinks=s_),
          q,
          k,
          v,
          sinks,
      )
      dq, dk, dv, dsinks = attn_vjp(do)
    jax.clear_caches()

    dense_mask = jnp.asarray(mask[:, :])
    if use_vmap:
      for b in range(2):
        seg_b = base.SegmentIds(ids[b], ids[b])
        q_b, k_b, v_b, do_b = jax.tree.map(
            lambda x, idx=b: x[idx].astype(jnp.float32), (q, k, v, do)
        )
        o_ref, stats_ref = base.attention_reference(
            q_b,
            k_b,
            v_b,
            dense_mask,
            seg_b,
            None,
            is_mqa=is_mqa,
            save_residuals=True,
        )
        dq_ref, dk_ref, dv_ref, _ = base.attention_reference_vjp(
            do_b,
            q_b,
            k_b,
            v_b,
            dense_mask,
            seg_b,
            None,
            o_ref,
            stats_ref["logsumexp"],
            is_mqa=is_mqa,
        )
        np.testing.assert_allclose(o[b], o_ref, atol=2e-2, rtol=2e-2)
        np.testing.assert_allclose(dq[b], dq_ref, atol=6e-2, rtol=6e-2)
        np.testing.assert_allclose(dk[b], dk_ref, atol=6e-2, rtol=6e-2)
        np.testing.assert_allclose(dv[b], dv_ref, atol=6e-2, rtol=6e-2)
    else:
      q32, k32, v32, do32 = jax.tree.map(
          lambda x: x.astype(jnp.float32), (q, k, v, do)
      )
      o_ref, stats_ref = base.attention_reference(
          q32,
          k32,
          v32,
          dense_mask,
          segment_ids,
          sinks,
          is_mqa=is_mqa,
          save_residuals=True,
      )
      dq_ref, dk_ref, dv_ref, dsinks_ref = base.attention_reference_vjp(
          do32,
          q32,
          k32,
          v32,
          dense_mask,
          segment_ids,
          sinks,
          o_ref,
          stats_ref["logsumexp"],
          is_mqa=is_mqa,
      )
      np.testing.assert_allclose(o, o_ref, atol=2e-2, rtol=2e-2)
      np.testing.assert_allclose(dq, dq_ref, atol=6e-2, rtol=6e-2)
      np.testing.assert_allclose(dk, dk_ref, atol=6e-2, rtol=6e-2)
      np.testing.assert_allclose(dv, dv_ref, atol=6e-2, rtol=6e-2)
      if use_sinks:
        np.testing.assert_allclose(dsinks, dsinks_ref, atol=5e-2, rtol=5e-2)

  def test_q_chunked_dropout_matches_unchunked(self):
    lengths = (180, 200, 260)
    seq_len, head_dim, bq, bkv = sum(lengths), 128, 128, 128
    k1, k2, k3, k4 = jax.random.split(jax.random.key(17), 4)
    q = jax.random.normal(k1, (2, seq_len, head_dim), jnp.bfloat16) / np.sqrt(
        head_dim
    ).astype(jnp.bfloat16)
    k = jax.random.normal(k2, (2, seq_len, head_dim), jnp.bfloat16)
    v = jax.random.normal(k3, (2, seq_len, head_dim), jnp.bfloat16)
    do = jax.random.normal(k4, (2, seq_len, head_dim), jnp.bfloat16)
    segment_ids = _segment_ids(lengths)
    prng_key = jax.random.key(23)

    mask = mask_lib.FullMask(_shape=(seq_len, seq_len))
    config = splash.SplashConfig(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        block_q_dkv=bq,
        block_kv_dkv=bkv,
        block_kv_dkv_compute=bkv,
        dropout_rate=0.25,
        dropout_block_q=bq,
        dropout_block_kv=bkv,
    )

    def run(max_bytes):
      with mock.patch.object(splash, "_MAX_SCALAR_PREFETCH_BYTES", max_bytes):
        jax.clear_caches()
        attn = splash.make_splash_mha_single_device(mask, config=config)
        o, attn_vjp = jax.vjp(
            lambda *qkv: attn(*qkv, segment_ids, prng_key=prng_key), q, k, v
        )
        return (o, *attn_vjp(do))

    o_chunked, dq_chunked, dk_chunked, dv_chunked = run(2 * 5 * 13)
    o_full, dq_full, dk_full, dv_full = run(splash._MAX_SCALAR_PREFETCH_BYTES)
    jax.clear_caches()

    np.testing.assert_array_equal(o_chunked, o_full)
    np.testing.assert_array_equal(dk_chunked, dk_full)
    np.testing.assert_array_equal(dv_chunked, dv_full)
    np.testing.assert_allclose(dq_chunked, dq_full, atol=6e-2, rtol=6e-2)

  def test_q_chunked_empty_q_chunk_residuals_and_sinks(self):
    seq_len, head_dim, bq, bkv = 512, 128, 128, 128
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(31), 5)
    q = jax.random.normal(k1, (2, seq_len, head_dim), jnp.bfloat16) / np.sqrt(
        head_dim
    ).astype(jnp.bfloat16)
    k = jax.random.normal(
        k2, (2, seq_len, head_dim), jnp.bfloat16
    ) * jnp.bfloat16(0.5)
    v = jax.random.normal(
        k3, (2, seq_len, head_dim), jnp.bfloat16
    ) * jnp.bfloat16(0.5)
    do = jax.random.normal(
        k4, (2, seq_len, head_dim), jnp.bfloat16
    ) * jnp.bfloat16(0.5)
    sinks = jax.random.normal(k5, (2,), jnp.float32)

    kv_ids = jnp.asarray(_ids_from_lengths((160, 160, 192)))
    # Trailing 2 q blocks use an orphan segment id absent from kv_ids so the
    # second 2-block q chunk has zero active tiles.
    q_ids = jnp.concatenate(
        [kv_ids[:256], jnp.full((256,), 99, dtype=jnp.int32)]
    )
    segment_ids = base.SegmentIds(q_ids, kv_ids)
    mask = mask_lib.FullMask(_shape=(seq_len, seq_len))

    config_orphan = splash.SplashConfig(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        block_q_dkv=bq,
        block_kv_dkv=bkv,
        block_kv_dkv_compute=bkv,
        max_logit_const=0.5,
    )

    def run_orphan(max_bytes):
      with mock.patch.object(splash, "_MAX_SCALAR_PREFETCH_BYTES", max_bytes):
        jax.clear_caches()
        attn = splash.make_splash_mha_single_device(
            mask, config=config_orphan, save_residuals=True
        )
        (o, stats), attn_vjp = jax.vjp(
            lambda *qkv: attn(*qkv, segment_ids), q, k, v
        )
        zero_stats = jax.tree.map(jnp.zeros_like, stats)
        dq, dk, dv = attn_vjp((do, zero_stats))
        return o, stats, dq, dk, dv

    o_c, stats_c, dq_c, dk_c, dv_c = run_orphan(2 * 4 * 13)
    o_u, stats_u, dq_u, dk_u, dv_u = run_orphan(
        splash._MAX_SCALAR_PREFETCH_BYTES
    )
    jax.clear_caches()

    np.testing.assert_array_equal(o_c, o_u)
    np.testing.assert_array_equal(o_c[:, 256:, :], 0.0)
    np.testing.assert_array_equal(stats_c["logsumexp"], stats_u["logsumexp"])
    self.assertTrue(
        np.all(np.isneginf(np.asarray(stats_c["logsumexp"][:, 256:])))
    )
    np.testing.assert_allclose(
        stats_c["max_logits"], stats_u["max_logits"], atol=1e-6
    )
    np.testing.assert_allclose(stats_c["max_logits"][:, 256:], 0.5, atol=1e-5)
    np.testing.assert_array_equal(dq_c[:, 256:, :], 0.0)
    np.testing.assert_allclose(dq_c, dq_u, atol=6e-2, rtol=6e-2)
    np.testing.assert_array_equal(dk_c, dk_u)
    np.testing.assert_array_equal(dv_c, dv_u)

    config_sinks = splash.SplashConfig(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        block_q_dkv=bq,
        block_kv_dkv=bkv,
        block_kv_dkv_compute=bkv,
        fuse_reciprocal=False,
    )
    with mock.patch.object(splash, "_MAX_SCALAR_PREFETCH_BYTES", 2 * 4 * 13):
      jax.clear_caches()
      attn_s = splash.make_splash_mha_single_device(
          mask, config=config_sinks, save_residuals=True
      )
      (o_s, stats_s), vjp_s = jax.vjp(
          lambda q_, k_, v_, s_: attn_s(q_, k_, v_, segment_ids, sinks=s_),
          q,
          k,
          v,
          sinks,
      )
      dq_s, dk_s, dv_s, dsinks_s = vjp_s(
          (do, jax.tree.map(jnp.zeros_like, stats_s))
      )
    jax.clear_caches()

    dense_mask = jnp.ones((seq_len, seq_len), dtype=bool)
    q32, k32, v32, do32 = jax.tree.map(
        lambda x: x.astype(jnp.float32), (q, k, v, do)
    )
    o_ref, stats_ref = base.attention_reference(
        q32,
        k32,
        v32,
        dense_mask,
        segment_ids,
        sinks,
        is_mqa=False,
        save_residuals=True,
    )
    dq_ref, dk_ref, dv_ref, dsinks_ref = base.attention_reference_vjp(
        do32,
        q32,
        k32,
        v32,
        dense_mask,
        segment_ids,
        sinks,
        o_ref,
        stats_ref["logsumexp"],
        is_mqa=False,
    )
    np.testing.assert_allclose(o_s, o_ref, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(
        stats_s["logsumexp"], stats_ref["logsumexp"], atol=2e-2, rtol=2e-2
    )
    np.testing.assert_allclose(
        stats_s["max_logits"], stats_ref["max_logits"], atol=2e-2, rtol=2e-2
    )
    np.testing.assert_allclose(dq_s, dq_ref, atol=6e-2, rtol=6e-2)
    np.testing.assert_allclose(dk_s, dk_ref, atol=6e-2, rtol=6e-2)
    np.testing.assert_allclose(dv_s, dv_ref, atol=6e-2, rtol=6e-2)
    np.testing.assert_allclose(dsinks_s, dsinks_ref, atol=5e-2, rtol=5e-2)

  def test_q_chunked_dense_partial_mask_blocks_matches_reference(self):
    seq_len, head_dim, bq, bkv = 640, 128, 128, 128
    k1, k2, k3, k4 = jax.random.split(jax.random.key(37), 4)
    q = jax.random.normal(k1, (2, seq_len, head_dim), jnp.bfloat16) / np.sqrt(
        head_dim
    ).astype(jnp.bfloat16)
    k = jax.random.normal(
        k2, (2, seq_len, head_dim), jnp.bfloat16
    ) * jnp.bfloat16(0.5)
    v = jax.random.normal(
        k3, (2, seq_len, head_dim), jnp.bfloat16
    ) * jnp.bfloat16(0.5)
    do = jax.random.normal(
        k4, (2, seq_len, head_dim), jnp.bfloat16
    ) * jnp.bfloat16(0.5)
    segment_ids = _segment_ids((180, 220, 240))

    # Sub-block band offset yields empty (0), partial (1), and full (2) tiles
    # with non-None `mask_next` and `partial_mask_blocks` (no `mask_function`).
    np_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool), k=32)
    mask = mask_lib.NumpyMask(np_mask)
    config = splash.SplashConfig(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        block_q_dkv=bq,
        block_kv_dkv=bkv,
        block_kv_dkv_compute=bkv,
        dq_reduction_steps=None,
    )
    fwd_info, mask_fn = mask_info_lib.process_mask(
        mask, (bq, bkv), return_dynamic_grid=False
    )
    dkv_info, _ = mask_info_lib.process_mask_dkv(
        mask, (bq, bkv), return_dynamic_grid=False
    )
    self.assertIsNone(mask_fn)
    self.assertIsNone(fwd_info.active_rows)
    self.assertIsNotNone(fwd_info.mask_next)
    self.assertIsNotNone(fwd_info.partial_mask_blocks)
    self.assertIsNone(dkv_info.active_rows)
    self.assertIsNotNone(dkv_info.mask_next)
    self.assertIsNotNone(dkv_info.partial_mask_blocks)

    fwd_entry_bytes = splash._refined_entry_bytes(fwd_info, 5, 5)
    self.assertEqual(fwd_entry_bytes, 14)

    fwd_info = jax.tree.map(jnp.asarray, fwd_info)
    dkv_info = jax.tree.map(jnp.asarray, dkv_info)
    kernel = splash.SplashAttentionKernel(
        fwd_info,
        dkv_info,
        config=config,
        is_mqa=False,
        save_residuals=False,
        mask_value=base.DEFAULT_MASK_VALUE,
        mask_function=None,
        fwd_mask_sparsity=1.0,
        dkv_mask_sparsity=1.0,
    )

    with mock.patch.object(
        splash, "_MAX_SCALAR_PREFETCH_BYTES", 2 * 5 * fwd_entry_bytes
    ):
      jax.clear_caches()
      o, vjp_fn = jax.vjp(lambda *qkv: kernel(*qkv, segment_ids), q, k, v)
      dq, dk, dv = vjp_fn(do)
    jax.clear_caches()

    dense_mask = jnp.asarray(np_mask)
    q32, k32, v32, do32 = jax.tree.map(
        lambda x: x.astype(jnp.float32), (q, k, v, do)
    )
    o_ref, stats_ref = base.attention_reference(
        q32,
        k32,
        v32,
        dense_mask,
        segment_ids,
        None,
        is_mqa=False,
        save_residuals=True,
    )
    dq_ref, dk_ref, dv_ref, _ = base.attention_reference_vjp(
        do32,
        q32,
        k32,
        v32,
        dense_mask,
        segment_ids,
        None,
        o_ref,
        stats_ref["logsumexp"],
        is_mqa=False,
    )
    assert isinstance(o, jax.Array)
    np.testing.assert_allclose(o, o_ref, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(dq, dq_ref, atol=6e-2, rtol=6e-2)
    np.testing.assert_allclose(dk, dk_ref, atol=6e-2, rtol=6e-2)
    np.testing.assert_allclose(dv, dv_ref, atol=6e-2, rtol=6e-2)

  def test_128k_packed_full_attention_fits_smem_and_matches_segment_ref(self):
    num_docs, doc_len = 256, 512
    seq_len = num_docs * doc_len
    bq = bkv = 512
    head_dim = 128
    k1, k2, k3, k4 = jax.random.split(jax.random.key(29), 4)
    scale = jnp.bfloat16(0.5)
    q = jax.random.normal(k1, (2, seq_len, head_dim), jnp.bfloat16) / np.sqrt(
        head_dim
    ).astype(jnp.bfloat16)
    k = jax.random.normal(k2, (2, seq_len, head_dim), jnp.bfloat16) * scale
    v = jax.random.normal(k3, (2, seq_len, head_dim), jnp.bfloat16) * scale
    do = jax.random.normal(k4, (2, seq_len, head_dim), jnp.bfloat16) * scale
    segment_ids = _segment_ids((doc_len,) * num_docs)

    mask = mask_lib.FullMask(_shape=(seq_len, seq_len))
    config = splash.SplashConfig(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv,
        block_q_dkv=bq,
        block_kv_dkv=bkv,
        block_kv_dkv_compute=bkv,
    )
    attn = splash.make_splash_mha_single_device(mask, config=config)
    o, attn_vjp = jax.vjp(attn, q, k, v, segment_ids)
    dq, dk, dv, _ = attn_vjp(do)

    dense_doc_mask = jnp.ones((doc_len, doc_len), dtype=bool)
    for doc_idx in (0, 127, 255):
      s0, s1 = doc_idx * doc_len, (doc_idx + 1) * doc_len
      q_d, k_d, v_d, do_d = jax.tree.map(
          lambda x, a=s0, b=s1: x[:, a:b, :].astype(jnp.float32), (q, k, v, do)
      )
      o_ref, stats_ref = base.attention_reference(
          q_d,
          k_d,
          v_d,
          dense_doc_mask,
          None,
          None,
          is_mqa=False,
          save_residuals=True,
      )
      dq_ref, dk_ref, dv_ref, _ = base.attention_reference_vjp(
          do_d,
          q_d,
          k_d,
          v_d,
          dense_doc_mask,
          None,
          None,
          o_ref,
          stats_ref["logsumexp"],
          is_mqa=False,
      )
      np.testing.assert_allclose(o[:, s0:s1, :], o_ref, atol=2e-2, rtol=2e-2)
      np.testing.assert_allclose(dq[:, s0:s1, :], dq_ref, atol=6e-2, rtol=6e-2)
      np.testing.assert_allclose(dk[:, s0:s1, :], dk_ref, atol=6e-2, rtol=6e-2)
      np.testing.assert_allclose(dv[:, s0:s1, :], dv_ref, atol=6e-2, rtol=6e-2)


if __name__ == "__main__":
  absltest.main()
