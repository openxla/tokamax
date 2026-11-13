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
"""Tests for layer/mode_registry.py: mode names and their resolution."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib
from tokamax._src.ops.experimental.dedup_moe.layer import mode_registry

BoardTopology = topology_lib.BoardTopology


class ModeNameTest(parameterized.TestCase):

  def test_modes_cover_every_kind_and_wave_count(self):
    want = {
        f"{kind}_w{w}"
        for kind, w in itertools.product(
            ("direct", "dedup"), mode_registry.SUPPORTED_WAVES
        )
    }
    self.assertEqual(set(mode_registry.MODES), want)

  @parameterized.parameters(*sorted(mode_registry.MODES))
  def test_parse_round_trip(self, name):
    spec = mode_registry.parse_mode(name)
    self.assertEqual(spec, mode_registry.MODES[name])
    self.assertEqual(spec.name, name)
    self.assertEqual(
        mode_registry.mode_name(dedup=spec.dedup, waves=spec.waves), name
    )
    self.assertEqual(spec.dedup, name.startswith(mode_registry.DEDUP))
    self.assertIn(f"W={spec.waves}", spec.label)
    self.assertIn("De-Dup" if spec.dedup else "Direct", spec.label)

  @parameterized.parameters(
      "direct_w3",
      "dedup_w16",
      "direct_w0",
      "direct",
      "dedup_w4~ptw0",
      "direct_w4+tc",
      "rules",
      "auto",
      "",
  )
  def test_parse_rejects_unknown_names(self, name):
    with self.assertRaisesRegex(ValueError, "unknown mode"):
      mode_registry.parse_mode(name)

  def test_fallback_mode_is_a_direct_mode(self):
    spec = mode_registry.parse_mode(mode_registry.FALLBACK_MODE)
    self.assertFalse(spec.dedup)
    self.assertIn(mode_registry.FALLBACK_MODE, mode_registry.MODES)


class ResolveModeTest(parameterized.TestCase):

  @parameterized.parameters(*mode_registry.SUPPORTED_WAVES)
  def test_direct_needs_no_topology(self, waves):
    kw = mode_registry.resolve_mode(
        f"direct_w{waves}", ep=16, g_local=8, topology=None
    )
    self.assertEqual(kw, dict(waves=waves, dedup_topology=None))

  def test_direct_ignores_a_topology(self):
    topology = BoardTopology.uniform(16, 8)
    kw = mode_registry.resolve_mode(
        "direct_w4", ep=16, g_local=8, topology=topology
    )
    self.assertEqual(kw, dict(waves=4, dedup_topology=None))

  @parameterized.parameters(
      (16, 8, 2), (16, 8, 4), (16, 4, 8), (8, 4, 1), (32, 8, 4)
  )
  def test_dedup_carries_the_topology(self, ep, board_size, waves):
    topology = BoardTopology.uniform(ep, board_size)
    kw = mode_registry.resolve_mode(
        f"dedup_w{waves}", ep=ep, g_local=8, topology=topology
    )
    self.assertEqual(kw, dict(waves=waves, dedup_topology=topology))
    # The forward-table depth is sized by layer/plan.py, not here.
    self.assertNotIn("dedup_fwd_max_runs", kw)

  @parameterized.parameters("direct_w8", "dedup_w8")
  def test_waves_must_divide_g_local(self, name):
    with self.assertRaisesRegex(mode_registry.SkipCaseError, "does not divide"):
      mode_registry.resolve_mode(
          name, ep=16, g_local=4, topology=BoardTopology.uniform(16, 8)
      )

  def test_dedup_without_topology_skips(self):
    with self.assertRaisesRegex(mode_registry.SkipCaseError, "BoardTopology"):
      mode_registry.resolve_mode("dedup_w4", ep=16, g_local=8, topology=None)

  def test_dedup_topology_must_cover_ep(self):
    with self.assertRaisesRegex(mode_registry.SkipCaseError, "not ep=32"):
      mode_registry.resolve_mode(
          "dedup_w4", ep=32, g_local=8, topology=BoardTopology.uniform(16, 8)
      )

  def test_every_dedup_wave_count_fits_the_forward_tables(self):
    self.assertGreaterEqual(
        dedup_plan_mod.MAX_PTW_WAVES, max(mode_registry.SUPPORTED_WAVES)
    )

  def test_unknown_name_raises_value_error(self):
    with self.assertRaises(ValueError):
      mode_registry.resolve_mode("dedup_w3", ep=16, g_local=8, topology=None)

  @parameterized.parameters(4, 8, 16, 32)
  def test_fallback_mode_resolves_without_topology(self, g_local):
    kw = mode_registry.resolve_mode(
        mode_registry.FALLBACK_MODE, ep=16, g_local=g_local, topology=None
    )
    self.assertIsNone(kw["dedup_topology"])


if __name__ == "__main__":
  absltest.main()
