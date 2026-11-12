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
"""Tests of `BoardTopology` and its static gateway tables."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib

BoardTopology = topology_lib.BoardTopology

# (ep, board_size): mirror gateways serve every source of each remote board
# once, so S = nb - 1.
MIRROR_CASES = (
    (8, 4),
    (16, 4),
    (32, 4),
    (16, 8),
    (32, 8),
    (64, 8),
    (128, 8),
)


def _rotated_gateways(ep, bs):
  """Gateway of s on board b: its local index shifted by 1 + b mod (bs - 1).

  The shift is never 0 mod bs, so no gateway is the mirror one, and it is
  the same for every source of one board, so each remote board's devices
  still serve the sources of one board once each.

  Args:
    ep: Devices.
    bs: Board size.

  Returns:
    The [ep][nb] gateway map.
  """
  nb = ep // bs
  return [
      [
          -1 if b == s // bs else b * bs + (s + 1 + b % (bs - 1)) % bs
          for b in range(nb)
      ]
      for s in range(ep)
  ]


def _single_gateways(ep, bs):
  """Every source lands on local device 0 of each remote board."""
  nb = ep // bs
  return [
      [-1 if b == s // bs else b * bs for b in range(nb)] for s in range(ep)
  ]


class TopologyTablesTest(parameterized.TestCase):

  def _check_consistent(self, topo, tab):
    """served / sigma / sigma_at_gw / n_served agree with the gateway map."""
    ep, nb, bs = topo.num_devices, topo.num_boards, topo.board_size
    self.assertEqual(
        (tab.num_devices, tab.num_boards, tab.board_size), (ep, nb, bs)
    )
    np.testing.assert_array_equal(tab.board, np.arange(ep) // bs)
    for s in range(ep):
      for b in range(nb):
        if b == s // bs:
          self.assertEqual(tab.gw[s, b], -1)
          self.assertEqual(tab.sigma_at_gw[s, b], -1)
          continue
        g = topo.gateway(s, b)
        self.assertEqual(tab.gw[s, b], g, (s, b))
        self.assertEqual(g // bs, b)
        sigma = tab.sigma_at_gw[s, b]
        self.assertEqual(tab.sigma[s, g], sigma)
        self.assertEqual(tab.served[g, sigma], s)
    for g in range(ep):
      lst = [int(x) for x in tab.served[g] if x >= 0]
      self.assertLen(lst, tab.n_served[g])
      np.testing.assert_array_equal(tab.served[g, len(lst) :], -1)
      self.assertEqual(
          sorted(lst), [s for s in range(ep) if tab.sigma[s, g] >= 0]
      )
      for rank, s in enumerate(lst):
        self.assertEqual(tab.sigma[s, g], rank)
        self.assertNotEqual(s // bs, g // bs)
      # Served order is the expected arrival order: sources issue their stage
      # runs to boards board(s) + 1, board(s) + 2, ..., ties by source.
      keys = [((g // bs - s // bs) % nb, s) for s in lst]
      self.assertEqual(keys, sorted(keys))
    self.assertEqual(tab.num_slots, max(1, int(tab.n_served.max())))
    self.assertEqual(tab.max_served, tab.num_slots)
    # Every (source, remote board) pair has exactly one gateway slot.
    self.assertEqual(int(tab.n_served.sum()), ep * (nb - 1))

  @parameterized.named_parameters(
      (f"ep{ep}_bs{bs}", ep, bs) for ep, bs in MIRROR_CASES
  )
  def test_mirror_tables(self, ep, bs):
    topo = BoardTopology.uniform(ep, bs)
    tab = topo.tables()
    nb = ep // bs
    self.assertEqual(tab.num_slots, nb - 1, tab.describe())
    self._check_consistent(topo, tab)
    for s in range(ep):
      for b in range(nb):
        if b != s // bs:
          self.assertEqual(tab.gw[s, b], b * bs + s % bs)
    np.testing.assert_array_equal(tab.n_served, nb - 1)

  @parameterized.named_parameters(
      ("ep8_bs4", 8, 4),
      ("ep16_bs4", 16, 4),
      ("ep16_bs8", 16, 8),
      ("ep32_bs8", 32, 8),
  )
  def test_rotated_gateways(self, ep, bs):
    topo = BoardTopology.uniform(ep, bs, gateways=_rotated_gateways(ep, bs))
    tab = topo.tables()
    self._check_consistent(topo, tab)
    nb = ep // bs
    # A permutation of each board's devices: still S = nb - 1, never mirror.
    self.assertEqual(tab.num_slots, nb - 1)
    for s in range(ep):
      for b in range(nb):
        if b != s // bs:
          self.assertNotEqual(tab.gw[s, b], b * bs + s % bs)

  @parameterized.named_parameters(
      ("ep8_bs4", 8, 4),
      ("ep16_bs4", 16, 4),
      ("ep32_bs8", 32, 8),
  )
  def test_single_gateway_per_board(self, ep, bs):
    topo = BoardTopology.uniform(ep, bs, gateways=_single_gateways(ep, bs))
    tab = topo.tables()
    self._check_consistent(topo, tab)
    nb = ep // bs
    self.assertEqual(tab.num_slots, (nb - 1) * bs)
    for g in range(ep):
      self.assertEqual(tab.n_served[g], (nb - 1) * bs if g % bs == 0 else 0)

  def test_served_order(self):
    # ep = 16 as 4 boards of 4, mirror: gateway 4 (board 1) serves sources
    # 0, 12 and 8 in arrival order: board 1 is the first board source 0
    # (board 0) issues to, the second for source 12 (board 3) and the third
    # for source 8 (board 2).
    tab = BoardTopology.uniform(16, 4).tables()
    self.assertEqual(tab.served[4].tolist(), [0, 12, 8])
    self.assertEqual(tab.served[0].tolist(), [12, 8, 4])

  def test_explicit_device_groups(self):
    topo = BoardTopology(((0, 1, 2, 3), (4, 5, 6, 7)))
    self.assertEqual(
        (topo.num_devices, topo.board_size, topo.num_boards), (8, 4, 2)
    )
    self.assertEqual(topo, BoardTopology.uniform(8, 4))
    # Lists are normalized to tuples (hashable, usable as a static argument).
    as_lists: Any = [[0, 1, 2, 3], [4, 5, 6, 7]]
    self.assertEqual(BoardTopology(as_lists), topo)
    self.assertEqual(hash(BoardTopology(as_lists)), hash(topo))

  def test_tables_cached_and_hash_by_identity(self):
    a = BoardTopology.uniform(16, 8).tables()
    b = BoardTopology.uniform(16, 8).tables()
    self.assertIs(a, b)  # Cached per topology.
    c = topology_lib.TopologyTables(**{
        f: getattr(a, f)
        for f in (
            "num_devices",
            "board_size",
            "num_boards",
            "num_slots",
            "board",
            "gw",
            "sigma",
            "sigma_at_gw",
            "served",
            "n_served",
        )
    })
    self.assertNotEqual(a, c)  # eq=False: usable as a jit static argument.
    self.assertEqual(a, a)
    self.assertEqual(hash(a), hash(a))

  @parameterized.named_parameters(
      ("one_board", ((0, 1, 2, 3),), "at least 2"),
      ("no_boards", (), "at least 2"),
      ("unequal", ((0, 1, 2, 3), (4, 5, 6, 7, 8, 9, 10, 11)), "equal sizes"),
      ("size2", ((0, 1), (2, 3)), "not supported"),
      ("size6", (tuple(range(6)), tuple(range(6, 12))), "not supported"),
      ("size16", (tuple(range(16)), tuple(range(16, 32))), "not supported"),
      ("non_contiguous", ((0, 2, 4, 6), (1, 3, 5, 7)), "contiguous"),
      ("permuted", ((1, 0, 2, 3), (4, 5, 6, 7)), "contiguous"),
      ("swapped_boards", ((4, 5, 6, 7), (0, 1, 2, 3)), "contiguous"),
      ("offset", ((1, 2, 3, 4), (5, 6, 7, 8)), "contiguous"),
  )
  def test_invalid_device_groups(self, groups, msg):
    with self.assertRaisesRegex(ValueError, msg):
      BoardTopology(groups)

  def test_uniform_rejects_non_multiple(self):
    with self.assertRaisesRegex(ValueError, "multiple"):
      BoardTopology.uniform(12, 8)
    with self.assertRaisesRegex(ValueError, "multiple"):
      BoardTopology.uniform(8, 0)
    with self.assertRaisesRegex(ValueError, "not supported"):
      BoardTopology.uniform(12, 6)
    with self.assertRaisesRegex(ValueError, "at least 2"):
      BoardTopology.uniform(8, 8)

  def test_invalid_gateways(self):
    ep, bs = 8, 4
    good = _rotated_gateways(ep, bs)
    BoardTopology.uniform(ep, bs, gateways=good)
    # Wrong shape.
    with self.assertRaisesRegex(ValueError, "gateways must be"):
      BoardTopology.uniform(ep, bs, gateways=good[:-1])
    with self.assertRaisesRegex(ValueError, "gateways must be"):
      BoardTopology.uniform(ep, bs, gateways=[row + [4] for row in good])
    # A gateway off its board.
    bad = [list(row) for row in good]
    bad[0][1] = 2  # Device 2 is on board 0, not board 1.
    with self.assertRaisesRegex(ValueError, "not on board 1"):
      BoardTopology.uniform(ep, bs, gateways=bad)
    bad = [list(row) for row in good]
    bad[5][0] = 8  # Out of range.
    with self.assertRaisesRegex(ValueError, "not on board 0"):
      BoardTopology.uniform(ep, bs, gateways=bad)
    # Entries for a source's own board are ignored.
    ignored = [list(row) for row in good]
    ignored[0][0] = 123
    tab = BoardTopology.uniform(ep, bs, gateways=ignored).tables()
    self.assertEqual(tab.gw[0, 0], -1)

  def test_supported_board_sizes(self):
    self.assertEqual(topology_lib.SUPPORTED_BOARD_SIZES, (4, 8))
    self.assertLessEqual(
        max(topology_lib.SUPPORTED_BOARD_SIZES), topology_lib.MAX_BOARD_SIZE
    )

  def test_describe(self):
    tab = BoardTopology.uniform(32, 8).tables()
    self.assertEqual(
        tab.describe(),
        "ep=32, board_size=8, nb=4, S=3, served per gateway [3]",
    )


if __name__ == "__main__":
  absltest.main()
