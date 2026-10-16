# Copyright 2026 Rabdos AI
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
"""Tests for fused GDN schedule metadata."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import metadata as fused_metadata

_D_K = 128
_D_V = 128


class GDNScheduleTest(parameterized.TestCase):
  """Test suite for dynamic prefill schedule construction."""

  def _schedule(
      self, decodes, lengths, chunk_size=128, heads=(8, 24), capacity=None
  ):
    """Construct a schedule fixture and return materialized metadata for
    assertions.

    Args:
      decodes: Number of one-token requests before the prefill requests.
      lengths: Submitted token lengths of the prefill requests.
      chunk_size: Number of token rows in one prefill tile.
      heads: Pair (number of Q/K heads, number of value heads) for the fixture.
      capacity: Tile-record capacity, or None to use the planner bound.

    Returns:
      Schedule validity, materialized metadata, and request token boundaries.
    """
    starts = np.cumsum([0] + [1] * decodes + list(lengths), dtype=np.int32)
    requests, tokens = len(starts) - 1, int(starts[-1])
    n_kq, n_v = heads
    dtype = jnp.dtype(jnp.float32)
    cfg = config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=dtype,
            act_out=dtype,
            compute=dtype,
            recurrent_state=dtype,
            conv_state=dtype,
        ),
        batch_size=tokens,
        dim_size=(2 * n_kq + n_v) * _D_K,
        kernel_size=4,
        tile_size=chunk_size,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=_D_K,
        v_head_dim=_D_V,
    )
    if capacity is None:
      capacity = fused_metadata.prefill_schedule_capacity(
          requests, tokens, chunk_size, n_kq, n_v
      )

    @jax.jit
    def build():
      """Return schedule validity and metadata built with temporary JAX refs."""
      refs = jax.tree.map(
          jax.new_ref, fused_metadata.metadata_template(cfg, requests, capacity)
      )
      valid = fused_metadata.build_smem_schedule(
          jnp.asarray(starts),
          jnp.arange(1, requests + 1, dtype=jnp.int32),
          jnp.array([decodes, decodes, requests], dtype=jnp.int32),
          jnp.asarray(np.diff(starts) + 1024),
          refs,
          jax.new_ref(jnp.zeros(requests + 1, jnp.int32)),
          jax.new_ref(jnp.zeros(requests + 2, jnp.int32)),
          cfg=cfg,
      )
      return valid, jax.tree.map(lambda ref: ref[...], refs)

    valid, metadata = build()
    return bool(valid), metadata, starts

  @parameterized.named_parameters(
      ("long_unaligned", 1, (4096,), 128, (8, 24), True),
      ("long_aligned", 8, (4096,), 128, (8, 24), True),
      ("ragged_requests", 3, (4096, 4103), 128, (8, 24), True),
      ("largest_shift", 7, (4096,), 128, (8, 24), True),
      ("small_tile", 1, (4096,), 64, (8, 24), True),
      ("large_tile", 1, (4096,), 1024, (8, 24), True),
      ("below_threshold", 1, (4095,), 128, (8, 24), False),
      ("other_heads", 1, (4096,), 128, (4, 12), False),
  )
  def test_tiles_cover_requests(
      self, decodes, lengths, chunk_size, heads, align
  ):
    """Check coverage, ownership, first/last flags, and expected alignment."""
    valid, metadata, starts = self._schedule(
        decodes, lengths, chunk_size, heads
    )
    self.assertTrue(valid)
    records = [
        metadata.get_record(i, 0) for i in range(int(metadata.num_tiles))
    ]
    for owner in range(decodes, len(starts) - 1):
      tiles = [r for r in records if int(r.s_idx) == owner]
      self.assertNotEmpty(tiles)
      cursor = int(starts[owner])
      if align:
        self.assertBetween(int(tiles[0].r_size), chunk_size - 7, chunk_size)
      if not align or cursor % 8 == 0:
        self.assertEqual(int(tiles[0].r_size), chunk_size)
      for i, tile in enumerate(tiles):
        self.assertEqual(int(tile.r_base), cursor)
        self.assertBetween(int(tile.r_size), 1, chunk_size)
        self.assertEqual(bool(tile.is_first_tile), i == 0)
        self.assertEqual(bool(tile.is_last_tile), i == len(tiles) - 1)
        if align and i:
          self.assertEqual(int(tile.r_base) % 8, 0)
        cursor += int(tile.r_size)
      self.assertEqual(cursor, int(starts[owner + 1]))
    self.assertEqual(
        {int(r.s_idx) for r in records}, set(range(decodes, len(starts) - 1))
    )

  def test_alignment_leaves_a_one_token_tail(self):
    """Check the long unaligned case leaves the expected one-token tail."""
    valid, metadata, _ = self._schedule(1, (4096,))
    self.assertTrue(valid)
    self.assertEqual(int(metadata.num_tiles), 33)
    first, last = metadata.get_record(0, 0), metadata.get_record(32, 0)
    self.assertEqual((int(first.r_base), int(first.r_size)), (1, 127))
    self.assertEqual((int(last.r_base), int(last.r_size)), (4096, 1))

  def test_insufficient_capacity_rejects_before_materializing(self):
    """Check insufficient record capacity rejects the schedule before
    materializing tiles.
    """
    valid, metadata, _ = self._schedule(1, (4096,), capacity=32)
    self.assertFalse(valid)
    self.assertEqual(int(metadata.num_tiles), 0)


if __name__ == "__main__":
  absltest.main()
