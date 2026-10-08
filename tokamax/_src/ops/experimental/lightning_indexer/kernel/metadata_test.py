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
"""Tests for batch_tile_idx, bq_idx, and bkv_idx metadata."""

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.lightning_indexer.kernel import metadata


class MetadataTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "standard_decode",
          256,
          16,
          40704,
      ),
      (
          "batched_decode_bs4",
          256,
          64,
          36608,
      ),
      (
          "prefill_chunk",
          1,
          64,
          42240,
      ),
      (
          "ragged_prefill_batch",
          64,
          32,
          41472,
      ),
      (
          "small_batch",
          4,
          8,
          42240,
      ),
      (
          "long_context_64k",
          128,
          128,
          36736,
      ),
  )
  def test_generate_max_steps(
      self,
      num_seqs: int,
      max_pages_per_seq: int,
      expected_max_steps: int,
  ):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    max_steps = metadata.generate_max_steps(
        num_seqs=num_seqs,
        max_pages_per_seq=max_pages_per_seq,
    )
    self.assertEqual(max_steps, expected_max_steps)
    self.assertGreater(max_steps, 0)
    self.assertEqual(max_steps % 128, 0)

  def test_one_sequence_num_steps_one(self):
    """Test metadata for 1 sequence with 1 tile."""
    seq_lens = jnp.array([128], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1], dtype=jnp.int32)
    start_seq_idx = 0
    end_seq_idx = 1
    bq_sz = 16
    bkv_sz = 128
    compression_ratio = 1
    static_q_len = 1
    seq_batch_size = 1

    meta = metadata.compute_batched_seq_metadata(
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        start_seq_idx=start_seq_idx,
        end_seq_idx=end_seq_idx,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        pages_per_seq=1,
        page_size=128,
        max_num_tokens=1,
        compression_ratio=compression_ratio,
        static_q_len=static_q_len,
        seq_batch_size=seq_batch_size,
    )

    self.assertEqual(int(meta.num_steps[0]), 1)

  def test_compute_batched_seq_metadata_decode(self):
    seq_lens = jnp.array([128, 256, 512, 64], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    start_seq_idx = 0
    end_seq_idx = 4
    bq_sz = 16
    bkv_sz = 64
    compression_ratio = 2
    static_q_len = 1
    seq_batch_size = 2

    meta = metadata.compute_batched_seq_metadata(
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        start_seq_idx=start_seq_idx,
        end_seq_idx=end_seq_idx,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        pages_per_seq=8,
        page_size=32,
        max_num_tokens=4,
        compression_ratio=compression_ratio,
        static_q_len=static_q_len,
        seq_batch_size=seq_batch_size,
    )

    # Check start_seq_idx is batch_tile_idx per tile
    np.testing.assert_array_equal(
        np.array(meta.start_seq_idx)[:6], [0, 0, 2, 2, 2, 2]
    )

    # For tile 0 (seq 0..1): max(64, 128) = 128 -> nbkv=2 (bkv 0..1)
    # For tile 1 (seq 2..3): max(256, 32) = 256 -> nbkv=4 (bkv 0..3)
    # Total tiles = 2 + 4 = 6
    self.assertEqual(int(meta.num_steps[0]), 6)
    np.testing.assert_array_equal(
        np.array(meta.batch_tile_idx)[:6], [0, 0, 2, 2, 2, 2]
    )
    np.testing.assert_array_equal(np.array(meta.bq_idx)[:6], [0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(
        np.array(meta.bkv_idx)[:6], [0, 1, 0, 1, 2, 3]
    )

  def test_compute_per_seq_metadata_prefill(self):
    seq_lens = jnp.array([128, 256], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 64, 256], dtype=jnp.int32)
    start_seq_idx = 0
    end_seq_idx = 2
    bq_sz = 32
    bkv_sz = 64
    compression_ratio = 1

    meta = metadata.compute_metadata(
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        start_seq_idx=start_seq_idx,
        end_seq_idx=end_seq_idx,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        pages_per_seq=4,
        page_size=64,
        max_num_tokens=256,
        compression_ratio=compression_ratio,
        static_q_len=None,
    )

    np.testing.assert_array_equal(
        np.array(meta.start_seq_idx)[:4], [0, 0, 0, 0]
    )

    # seq 0: q_len=64 -> nbq=2 (bq 0..1), kv_len=128 -> nbkv=2 (bkv 0..1) => 4 tiles
    # seq 1: q_len=192 -> nbq=6 (bq 0..5), kv_len=256 -> nbkv=4 (bkv 0..3)
    #   => 24 tiles
    # Total tiles = 4 + 24 = 28
    self.assertEqual(int(meta.num_steps[0]), 28)
    # Check seq 0 tiles (first 4)
    np.testing.assert_array_equal(
        np.array(meta.batch_tile_idx)[:4], [0, 0, 0, 0]
    )
    np.testing.assert_array_equal(np.array(meta.bq_idx)[:4], [0, 0, 1, 1])
    np.testing.assert_array_equal(np.array(meta.bkv_idx)[:4], [0, 1, 0, 1])

  def test_mixed_config_and_print_schedule(self):
    """Test config with B=13 from streamindex_topk_test.py."""
    S_list = [128, 64, 32, 256, 128, 128, 64, 32, 256, 128, 32, 64, 256]
    seq_lens = jnp.array(S_list, dtype=jnp.int32)
    cu_q_lens = jnp.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 266, 268, 276], dtype=jnp.int32
    )
    bq_sz = 64
    bkv_p = 2
    page_size = 64
    bkv_sz = page_size * bkv_p  # 128
    comp_ratio = 1

    print("\n" + "=" * 60)
    print("TEST CASE 11 (B=13) METADATA & GENERATED SCHEDULE")
    print("=" * 60)

    # Pass 1: Decode Batched (seq 0 to 8 with seq_batch_size = 4)
    meta_decode_batch = metadata.compute_metadata(
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        start_seq_idx=0,
        end_seq_idx=8,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        pages_per_seq=4,
        page_size=page_size,
        max_num_tokens=276,
        compression_ratio=comp_ratio,
        static_q_len=1,
        seq_batch_size=4,
    )

    # Pass 2: Decode Remainder (seq 8 to 10 with seq_batch_size = 1)
    meta_decode_rem = metadata.compute_metadata(
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        start_seq_idx=8,
        end_seq_idx=10,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        pages_per_seq=4,
        page_size=page_size,
        max_num_tokens=276,
        compression_ratio=comp_ratio,
        static_q_len=1,
        seq_batch_size=1,
    )

    # Pass 3: Mixed / Prefill (seq 10 to 13 with seq_batch_size = 1)
    meta_mixed = metadata.compute_metadata(
        seq_lens=seq_lens,
        cu_q_lens=cu_q_lens,
        start_seq_idx=10,
        end_seq_idx=13,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        pages_per_seq=4,
        page_size=page_size,
        max_num_tokens=276,
        compression_ratio=comp_ratio,
        static_q_len=None,
        seq_batch_size=1,
    )
    # ============================================================
    # META_DECODE_BATCH (Batched Decode: seqs 0..7)
    # ============================================================
    # Pipeline Step (all_steps):              [   0,     1,     2   ]
    # Owner Tile:                             |--- Tile 0 ----|  |T 2|
    # Covered Sequences:                      |-- seqs 0..3 --|  |4..7|
    # ------------------------------------------------------------
    # Tile's Starting Step:                       0      0      2
    # Local Step in Tile (tile_step_id):          0      1      0
    # ------------------------------------------------------------
    # Query Block Index (bq_idx):                 0      0      0
    # KV Block Index (bkv_idx):                   0      1      0
    # ============================================================
    # --- GENERATED TILE SCHEDULE ---
    # Tile 00: BatchedDecode (seqs 0..3) [start_seq=0] | bq_idx=0 | bkv_idx=0
    # Tile 01: BatchedDecode (seqs 0..3) [start_seq=0] | bq_idx=0 | bkv_idx=1
    # Tile 02: BatchedDecode (seqs 4..7) [start_seq=4] | bq_idx=0 | bkv_idx=0
    # Tile 03: DecodeRemainder (seq 8) [start_seq=8]  | bq_idx=0 | bkv_idx=0
    # Tile 04: DecodeRemainder (seq 8) [start_seq=8]  | bq_idx=0 | bkv_idx=1
    # Tile 05: DecodeRemainder (seq 9) [start_seq=9]  | bq_idx=0 | bkv_idx=0
    # Tile 06: Prefill/Mixed (seq 10) [start_seq=10]  | bq_idx=0 | bkv_idx=0
    # Tile 07: Prefill/Mixed (seq 10) [start_seq=10]  | bq_idx=1 | bkv_idx=0
    # Tile 08: Prefill/Mixed (seq 10) [start_seq=10]  | bq_idx=2 | bkv_idx=0
    # Tile 09: Prefill/Mixed (seq 10) [start_seq=10]  | bq_idx=3 | bkv_idx=0
    # Tile 10: Prefill/Mixed (seq 11) [start_seq=11]  | bq_idx=0 | bkv_idx=0
    # Tile 11: Prefill/Mixed (seq 12) [start_seq=12]  | bq_idx=0 | bkv_idx=0
    # Tile 12: Prefill/Mixed (seq 12) [start_seq=12]  | bq_idx=0 | bkv_idx=1
    # ============================================================
    self.assertEqual(int(meta_decode_batch.num_steps[0]), 3)
    self.assertEqual(int(meta_decode_rem.num_steps[0]), 3)
    self.assertEqual(int(meta_mixed.num_steps[0]), 7)

    # Full Tile Schedule Summary
    print("\n--- GENERATED TILE SCHEDULE ---")
    schedule = []
    tile_count = 0

    for meta, label in [
        (meta_decode_batch, "BatchedDecode"),
        (meta_decode_rem, "DecodeRemainder"),
        (meta_mixed, "Prefill/Mixed"),
    ]:
      num_t = int(meta.num_steps[0])
      for p in range(num_t):
        s_idx = int(meta.batch_tile_idx[p])
        start_s = int(meta.start_seq_idx[p])
        bq = int(meta.bq_idx[p])
        bkv = int(meta.bkv_idx[p])
        seq_str = (
            f"seqs {s_idx}..{s_idx + 3}"
            if "Batched" in label
            else f"seq {s_idx}"
        )
        schedule.append((
            tile_count,
            f"{label} ({seq_str}) [start_seq={start_s}]",
            bq,
            bkv,
        ))
        tile_count += 1

    for tile_id, phase, bq, bkv in schedule:
      print(f"Tile {tile_id:02d}: {phase:<38s} | bq_idx={bq} | bkv_idx={bkv}")

    print("=" * 60 + "\n")


class ChunkedMetadataTest(parameterized.TestCase):
  """`chunk_token_start` as a tuple of chunk starts.

  A tuple builds ONE schedule holding every chunk back to back, and the
  kernel runs chunk `m` by jumping to `sum(num_steps[:m])` and executing
  `num_steps[m]` steps from there. Three things have to hold for that to
  work, and each has a test below: the per-chunk step counts are right, the
  steps sitting at that offset are the ones chunk `m` would have got on its
  own, and the schedule arrays are long enough to hold every chunk.

  None of this builds a kernel, so it is a CPU test.
  """

  # Four sequences of 256 query tokens each over a 1024-token flat axis. A
  # 256-token chunk therefore owns exactly one whole sequence, and every
  # chunk cut lands on a sequence boundary and a bq block boundary at once --
  # which is what makes the "same work, just split up" assertion below exact.
  CHUNK_SEQ_LENS = (512, 1024, 768, 256)  # kv lengths, nbkv = 4, 8, 6, 2
  CHUNK_CU_Q_LENS = (0, 256, 512, 768, 1024)
  CHUNK_BQ_SZ = 64  # 256 query tokens -> 4 bq blocks per sequence
  CHUNK_BKV_SZ = 128
  CHUNK_PAGE_SIZE = 64
  CHUNK_PAGES_PER_SEQ = 16
  CHUNK_MAX_NUM_TOKENS = 1024
  # 4 bq blocks x (4 + 8 + 6 + 2) kv blocks.
  CHUNK_UNCHUNKED_STEPS = 80

  def _chunk_meta(self, chunk_token_start, chunk_tokens=256, **overrides):
    """One prefill pass over the four-sequence config above."""
    kwargs = dict(
        seq_lens=jnp.array(self.CHUNK_SEQ_LENS, dtype=jnp.int32),
        cu_q_lens=jnp.array(self.CHUNK_CU_Q_LENS, dtype=jnp.int32),
        start_seq_idx=0,
        end_seq_idx=len(self.CHUNK_SEQ_LENS),
        bq_sz=self.CHUNK_BQ_SZ,
        bkv_sz=self.CHUNK_BKV_SZ,
        pages_per_seq=self.CHUNK_PAGES_PER_SEQ,
        page_size=self.CHUNK_PAGE_SIZE,
        max_num_tokens=self.CHUNK_MAX_NUM_TOKENS,
        static_q_len=None,
        seq_batch_size=1,
        chunk_token_start=chunk_token_start,
        chunk_tokens=chunk_tokens,
    )
    kwargs.update(overrides)
    return metadata.compute_metadata(**kwargs)

  def _expected_num_steps(self, starts, chunk_tokens, cu_q_lens=None):
    """Steps per chunk, straight from the definition, in numpy.

    Chunk `m` gives sequence `s` the tokens in
    `[chunk_start, chunk_start + chunk_tokens) & [q_start, q_end)`, lays bq
    blocks out from the low end of that intersection, and pairs each with
    every kv block of the sequence.
    """
    cu_q_lens = np.array(
        cu_q_lens if cu_q_lens is not None else self.CHUNK_CU_Q_LENS
    )
    kv_lens = np.array(self.CHUNK_SEQ_LENS)
    num_bkv = np.maximum(1, -(-kv_lens // self.CHUNK_BKV_SZ))
    out = []
    for start in starts:
      steps = 0
      for s in range(len(kv_lens)):
        lo = np.clip(start, cu_q_lens[s], cu_q_lens[s + 1])
        hi = np.clip(start + chunk_tokens, cu_q_lens[s], cu_q_lens[s + 1])
        steps += -(-(hi - lo) // self.CHUNK_BQ_SZ) * num_bkv[s]
      out.append(int(steps))
    return out

  def test_num_steps_holds_one_entry_per_chunk(self):
    """Shape is the caller's contract: `num_steps[m]` is chunk `m`.

    Unchunked keeps the pre-tuple shape of (1,) so callers indexing
    `num_steps[0]` are unaffected.
    """
    self.assertEqual(
        self._chunk_meta(None, chunk_tokens=None).num_steps.shape, (1,)
    )
    self.assertEqual(self._chunk_meta(0).num_steps.shape, (1,))
    self.assertEqual(self._chunk_meta((0, 256, 512, 768)).num_steps.shape, (4,))

  @parameterized.named_parameters(("first_chunk", 0), ("middle_chunk", 512))
  def test_scalar_chunk_start_matches_the_one_tuple(self, start):
    """A bare int and a 1-tuple of it are the same single-chunk schedule."""
    scalar = self._chunk_meta(start)
    tupled = self._chunk_meta((start,))
    for name in ("num_steps", "batch_tile_idx", "bq_idx", "bkv_idx"):
      np.testing.assert_array_equal(
          np.array(getattr(scalar, name)),
          np.array(getattr(tupled, name)),
          err_msg=name,
      )

  def test_each_chunk_slice_matches_its_standalone_schedule(self):
    """The property the kernel's `step_offset` rests on.

    Chunk `m` starts at `sum(num_steps[:m])`, and the steps from there must
    be exactly the schedule that chunk builds when it is the only one.
    """
    starts = (0, 256, 512, 768)
    combined = self._chunk_meta(starts)

    offset = 0
    for m, start in enumerate(starts):
      solo = self._chunk_meta(start)
      n = int(solo.num_steps[0])
      self.assertEqual(int(combined.num_steps[m]), n, f"chunk {m} step count")
      for name in ("batch_tile_idx", "bq_idx", "bkv_idx"):
        np.testing.assert_array_equal(
            np.array(getattr(combined, name))[offset : offset + n],
            np.array(getattr(solo, name))[:n],
            err_msg=f"chunk {m} at step offset {offset}: {name}",
        )
      offset += n

  def test_aligned_chunks_are_the_unchunked_schedule_cut_up(self):
    """Chunking that lands on block boundaries adds and drops no work."""
    plain = self._chunk_meta(None, chunk_tokens=None)
    combined = self._chunk_meta((0, 256, 512, 768))

    total = int(np.sum(np.array(combined.num_steps)))
    self.assertEqual(int(plain.num_steps[0]), self.CHUNK_UNCHUNKED_STEPS)
    self.assertEqual(total, self.CHUNK_UNCHUNKED_STEPS)
    for name in ("batch_tile_idx", "bq_idx", "bkv_idx"):
      np.testing.assert_array_equal(
          np.array(getattr(combined, name))[:total],
          np.array(getattr(plain, name))[:total],
          err_msg=name,
      )

  @parameterized.named_parameters(
      ("two_chunks", 512),
      ("four_chunks", 256),
      ("eight_chunks", 128),
      ("sixteen_chunks", 64),
  )
  def test_whole_sweep_of_chunks_fits_and_keeps_every_step(self, chunk_tokens):
    """`max_steps` has to scale with the chunk count, not the chunk.

    All the chunks live in one set of arrays now, so a bound sized for a
    single chunk would silently truncate the last ones.
    """
    starts = tuple(range(0, self.CHUNK_MAX_NUM_TOKENS, chunk_tokens))
    meta_ = self._chunk_meta(starts, chunk_tokens=chunk_tokens)

    num_steps = np.array(meta_.num_steps)
    self.assertEqual(num_steps.shape, (len(starts),))
    np.testing.assert_array_equal(
        num_steps, self._expected_num_steps(starts, chunk_tokens)
    )
    # Cuts are aligned here, so the split is exact however fine it gets.
    self.assertEqual(int(num_steps.sum()), self.CHUNK_UNCHUNKED_STEPS)
    self.assertLessEqual(int(num_steps.sum()), meta_.bq_idx.shape[0])

  def test_chunks_cutting_mid_sequence_recompute_only_the_seam_block(self):
    """Ragged q lens: a cut inside a bq block costs that block twice.

    Sequence boundaries at 100 and 512 put every 256-token cut inside a
    sequence, which is the case the flat-axis layout has to keep straight.
    """
    cu_q_lens = (0, 100, 512, 1024, 1024)
    starts = (0, 256, 512, 768)
    meta_ = self._chunk_meta(
        starts, cu_q_lens=jnp.array(cu_q_lens, dtype=jnp.int32)
    )

    num_steps = np.array(meta_.num_steps)
    np.testing.assert_array_equal(
        num_steps, self._expected_num_steps(starts, 256, cu_q_lens=cu_q_lens)
    )
    # Seam duplication only ever adds work, and never more than the arrays
    # can hold.
    self.assertGreaterEqual(int(num_steps.sum()), 1)
    self.assertLessEqual(int(num_steps.sum()), meta_.bq_idx.shape[0])

  def test_empty_pass_returns_a_zero_per_chunk(self):
    """The empty branch has to agree with the schedule branch on shape."""
    meta_ = self._chunk_meta((0, 256, 512), start_seq_idx=0, end_seq_idx=0)
    np.testing.assert_array_equal(np.array(meta_.num_steps), [0, 0, 0])


if __name__ == "__main__":
  absltest.main()
