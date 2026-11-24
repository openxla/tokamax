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

  def test_sign_partitioned_overlap_prunes_unaligned_negative_padding(self):
    # 448 consecutive negative padding IDs (`-(pos + 1)`) fill all 256 modulo
    # buckets inside a single block; sign-partitioning isolates them so the
    # mixed boundary block still prunes against earlier positive documents.
    bq = bkv = 512
    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max
    blocks = [
        np.full(512, 1, np.int32),
        np.full(512, 2, np.int32),
        np.full(512, 256, np.int32),
        np.zeros(512, np.int32),
        np.full(512, -256, np.int32),
        # Mixed boundary block: 64 tokens of doc 15 + 448 unique negative IDs.
        np.concatenate([
            np.full(64, 15, np.int32),
            -(np.arange(211008, 211456, dtype=np.int32) + 1),
        ]),
        # Adjacent pure-padding block with strictly lower negative IDs.
        -(np.arange(211456, 211968, dtype=np.int32) + 1),
        # Earlier block sharing doc 15 only.
        np.full(512, 15, np.int32),
        # Non-monotonic negative block {-1, -5} vs disjoint {-2}.
        np.concatenate([
            np.full(256, -1, np.int32),
            np.full(256, -5, np.int32),
        ]),
        np.full(512, -2, np.int32),
        # int32 extremes across empty positive/negative classes.
        np.full(512, int32_min, np.int32),
        np.full(512, int32_max, np.int32),
    ]
    ids_np = np.concatenate(blocks)
    ids = jnp.asarray(ids_np)
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
    expected = _tile_properties(ids_np, ids_np, bq, bkv)
    np.testing.assert_array_equal(dense, expected)

    # Confirm the unaligned mixed block with 448 negative IDs matches only
    # itself and the doc-15 block (pruned against doc 1, 2, 256, 0, pure-pad).
    self.assertEqual(np.flatnonzero(dense[5]).tolist(), [5, 7])
    # Cross-sign mod-256 identical residues {0, 256, -256} must not collide.
    self.assertFalse(dense[2, 3])
    self.assertFalse(dense[2, 4])
    self.assertFalse(dense[3, 4])
    # Verify inside vmap across heterogeneous batch elements (0 vs neg vs pos).
    ids_b = jnp.stack([
        jnp.concatenate([
            np.full(512, 1, np.int32),
            np.full(512, 2, np.int32),
        ]),
        jnp.concatenate([
            np.concatenate([
                np.full(64, 15, np.int32),
                -(np.arange(1, 449, dtype=np.int32)),
            ]),
            np.zeros(512, np.int32),
        ]),
    ])
    vmap_dense = jax.vmap(
        lambda s: splash._segment_block_overlap(  # pylint: disable=protected-access
            base.SegmentIds(s, s),
            q_blocks=2,
            kv_blocks=2,
            bq=512,
            bkv=512,
        )
    )(ids_b)
    np.testing.assert_array_equal(
        np.asarray(vmap_dense[0]), np.array([[True, False], [False, True]])
    )
    vmap_indexed = jax.vmap(
        lambda s: splash._segment_block_overlap(  # pylint: disable=protected-access
            base.SegmentIds(s, s),
            jnp.arange(2)[:, None],
            jnp.arange(2)[None, :],
            q_blocks=2,
            kv_blocks=2,
            bq=512,
            bkv=512,
        )
    )(ids_b)
    np.testing.assert_array_equal(
        np.asarray(vmap_indexed[0]), np.array([[True, False], [False, True]])
    )

    # Also verify unsigned dtype handling (neg class is empty).
    u_ids = jnp.concatenate([
        jnp.zeros(512, jnp.uint32),
        jnp.full(512, 7, jnp.uint32),
    ])
    u_dense = np.asarray(
        splash._segment_block_overlap(  # pylint: disable=protected-access
            base.SegmentIds(u_ids, u_ids),
            q_blocks=2,
            kv_blocks=2,
            bq=512,
            bkv=512,
        )
    )
    np.testing.assert_array_equal(
        u_dense, np.array([[True, False], [False, True]])
    )


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

  def test_dense_refine_downcasts_indices(self):
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
      ("causal_aligned_mha", True, (256, 256, 256, 256), 2, 2, False, None),
      ("causal_straddling_mha", True, (180, 332, 200, 312), 2, 2, False, None),
      ("causal_aligned_gqa", True, (256, 256, 256, 256), 4, 2, False, None),
      ("causal_aligned_gqa_dq1", True, (256, 256, 256, 256), 4, 2, False, 1),
      ("causal_short_1step_gqa", True, (64, 64), 4, 2, False, None),
      ("causal_short_1step_gqa_dq1", True, (64, 64), 4, 2, False, 1),
      ("causal_short_2step_gqa", True, (128, 128), 4, 2, False, None),
      ("causal_short_3step_gqa", True, (128, 128, 128), 4, 2, False, None),
      ("causal_short_1step_mqa", True, (64, 64), 4, 1, True, None),
      ("causal_short_2step_mqa", True, (128, 128), 4, 1, True, None),
      ("full_aligned_mha", False, (256, 256, 256, 256), 2, 2, False, None),
      ("full_straddling_gqa", False, (180, 332, 200, 312), 4, 2, False, None),
  )
  def test_fwd_and_bwd_match_reference(
      self,
      is_causal,
      lengths,
      num_q_heads,
      num_kv_heads,
      is_mqa,
      dq_reduction_steps,
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
        dq_reduction_steps=dq_reduction_steps,
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
      ("static_dkv_grid_mha", None, 2, 2),
      ("dynamic_dkv_grid_mha", 3, 2, 2),
      ("static_dkv_grid_gqa", None, 4, 2),
      ("dynamic_dkv_grid_gqa", 3, 4, 2),
  )
  def test_dropout_matches_the_unpruned_schedule(
      self, dq_reduction_steps, num_q_heads, num_kv_heads
  ):
    # Dropout bits are keyed on absolute canonical-block coordinates, so
    # skipping whole tiles must not change which surviving weights are dropped.
    lengths = (180, 332, 200, 312)
    seq_len, head_dim, bq, bkv = sum(lengths), 128, 128, 128
    k1, k2, k3, k4 = jax.random.split(jax.random.key(3), 4)
    q = jax.random.normal(
        k1, (num_q_heads, seq_len, head_dim), jnp.bfloat16
    ) / np.sqrt(head_dim).astype(jnp.bfloat16)
    k = jax.random.normal(k2, (num_kv_heads, seq_len, head_dim), jnp.bfloat16)
    v = jax.random.normal(k3, (num_kv_heads, seq_len, head_dim), jnp.bfloat16)
    do = jax.random.normal(k4, (num_q_heads, seq_len, head_dim), jnp.bfloat16)
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
    # would show up here. Static GQA sums across q blocks before q heads, while
    # the dynamic GQA grid sums across q heads inside each q block.
    np.testing.assert_array_equal(o, o_ref)
    if dq_reduction_steps is None and num_q_heads != num_kv_heads:
      np.testing.assert_allclose(dk, dk_ref, atol=1e-2, rtol=1e-2)
      np.testing.assert_allclose(dv, dv_ref, atol=1e-2, rtol=1e-2)
    else:
      np.testing.assert_array_equal(dk, dk_ref)
      np.testing.assert_array_equal(dv, dv_ref)
    # A pruned dkv grid is dynamic, and a dynamic grid always accumulates dq in
    # three slots of q's dtype (bf16 here) picked by grid step; pruning
    # renumbers the steps, so dq only matches up to bf16 rounding.
    np.testing.assert_allclose(dq, dq_ref, atol=8e-2, rtol=8e-2)

  @parameterized.named_parameters(
      ("gqa", 4, 2, False),
      ("mqa", 4, 1, True),
  )
  def test_large_dkv_block_gqa_fits_in_vmem(
      self, num_q_heads, num_kv_heads, is_mqa
  ):
    if jax.devices()[0].device_kind != "TPU7x":
      self.skipTest("Requires 64 MiB VMEM on TPU7x.")
    seq_len, head_dim = 8192, 512
    lengths = (2048, 2048, 4096)
    kv_shape = (
        (seq_len, head_dim) if is_mqa else (num_kv_heads, seq_len, head_dim)
    )
    k1, k2, k3, k4 = jax.random.split(jax.random.key(11), 4)
    q = jax.random.normal(
        k1, (num_q_heads, seq_len, head_dim), jnp.bfloat16
    ) / np.sqrt(head_dim).astype(jnp.bfloat16)
    k = jax.random.normal(k2, kv_shape, jnp.bfloat16) * jnp.bfloat16(0.5)
    v = jax.random.normal(k3, kv_shape, jnp.bfloat16) * jnp.bfloat16(0.5)
    do = jax.random.normal(
        k4, (num_q_heads, seq_len, head_dim), jnp.bfloat16
    ) * jnp.bfloat16(0.5)
    segment_ids = _segment_ids(lengths)

    mask = mask_lib.CausalMask(shape=(seq_len, seq_len))
    config = splash.SplashConfig(
        block_q=512,
        block_kv=512,
        block_kv_compute=512,
        block_q_dkv=1024,
        block_kv_dkv=2048,
        block_kv_dkv_compute=1024,
        q_layout=splash.QKVLayout.SEQ_MINOR,
        k_layout=splash.QKVLayout.SEQ_MINOR,
        v_layout=splash.QKVLayout.HEAD_DIM_MINOR,
        use_experimental_scheduler=True,
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


if __name__ == "__main__":
  absltest.main()
