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
"""Tests for layer/plan.py: the static plan `modular.dedup_moe` runs.

Host-only: the rules config (`derive_config`), t buckets, the static
forward-table depth, policy resolution (`auto`, `rules`, pinned modes) and
the fallback, on uniform board topologies with board sizes 4 and 8.
"""

import dataclasses
import itertools

from absl.testing import absltest
from absl.testing import parameterized
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib
from tokamax._src.ops.experimental.dedup_moe.layer import config as layer_config
from tokamax._src.ops.experimental.dedup_moe.layer import mode_registry
from tokamax._src.ops.experimental.dedup_moe.layer import plan as layer_plan

BoardTopology = topology_lib.BoardTopology
UnifiedRules = layer_config.UnifiedRules

# A large MoE layer: the rules' thresholds are meant for shapes like it.
HIDDEN = 8192
E_TOTAL = 512
TOPK = 10
FFN = 2048


def _topology(ep, board_size):
  return None if board_size is None else BoardTopology.uniform(ep, board_size)


def _cfg(ep, board_size=None, **kw):
  kw.setdefault("num_experts", E_TOTAL)
  return layer_config.DedupMoeConfig(
      ep=ep,
      hidden=HIDDEN,
      top_k=TOPK,
      ffn=FFN,
      topology=_topology(ep, board_size),
      **kw,
  )


def _derive(ep, board_size, t, rules=UnifiedRules(), e_total=E_TOTAL):
  return layer_plan.derive_config(
      _topology(ep, board_size),
      ep,
      e_total // ep,
      t,
      top_k=TOPK,
      hidden=HIDDEN,
      rules=rules,
  )


class DeriveConfigTest(parameterized.TestCase):

  @parameterized.parameters(
      (8, 1024), (16, 8), (16, 128), (32, 256), (64, 1024), (128, 4096)
  )
  def test_no_topology_is_direct(self, ep, t):
    mode, kw, notes = _derive(ep, None, t)
    g_local = E_TOTAL // ep
    waves = 8 if g_local >= UnifiedRules().waves_g_cut else 4
    self.assertEqual(mode, f"direct_w{waves}")
    self.assertEqual(kw, dict(waves=waves, dedup_topology=None))
    self.assertFalse(notes["dedup"])
    self.assertEqual(notes["boards"], 1)
    self.assertEqual(notes["board_fanout"], 1.0)

  # (ep, board size, t) -> De-Dup on? Under uniform routing with K=10:
  # De-Dup needs a remote-board fan-out F >= 2 and t * (1 - 1/F) >= 85.
  @parameterized.named_parameters(
      ("ep16_bs8_t128", 16, 8, 128, True),
      ("ep16_bs8_t512", 16, 8, 512, True),
      ("ep16_bs8_t64", 16, 8, 64, False),  # too few rows saved
      ("ep32_bs8_t128", 32, 8, 128, False),  # too few rows saved
      ("ep32_bs8_t256", 32, 8, 256, True),
      ("ep64_bs8_t1024", 64, 8, 1024, False),  # fan-out below 2
      ("ep16_bs4_t128", 16, 4, 128, False),  # too few rows saved
      ("ep16_bs4_t256", 16, 4, 256, True),
      ("ep32_bs4_t1024", 32, 4, 1024, False),  # fan-out below 2
      ("ep8_bs4_t1024", 8, 4, 1024, True),
      ("ep8_bs4_t64", 8, 4, 64, False),  # too few rows saved
  )
  def test_rules_dedup_decision(self, ep, board_size, t, dedup):
    topology = _topology(ep, board_size)
    mode, kw, notes = _derive(ep, board_size, t)
    g_local = E_TOTAL // ep
    rules = UnifiedRules()
    fanout = layer_plan.expected_board_fanout(ep, TOPK, board_size)
    self.assertEqual(notes["board_fanout"], round(fanout, 3))
    self.assertEqual(notes["boards"], ep // board_size)
    self.assertEqual(notes["dedup"], dedup)
    self.assertEqual(
        dedup,
        fanout >= rules.dedup_min_board_fanout
        and t * (1 - 1 / fanout) >= rules.dedup_min_savings,
    )
    if dedup:
      waves = rules.waves_dedup
      self.assertEqual(mode, f"dedup_w{waves}")
      self.assertEqual(kw["dedup_topology"], topology)
      num_slots = topology.tables().num_slots
      self.assertEqual(notes["num_slots"], num_slots)
      boards = ep // board_size
      self.assertEqual(
          kw["dedup_min_fanout"],
          rules.min_fanout_small
          if boards <= rules.min_fanout_boards_cut
          else rules.min_fanout_large,
      )
      max_runs = kw["dedup_fwd_max_runs"]
      self.assertEqual(
          max_runs,
          layer_plan.static_fwd_max_runs(
              num_slots, waves, t, factor=rules.fwd_runs_factor
          ),
      )
      self.assertGreaterEqual(max_runs, dedup_plan_mod.MAX_FWD_RUNS)
      table_bytes = num_slots * waves * max_runs * dedup_plan_mod.FWD_FIELDS * 4
      self.assertLessEqual(table_bytes, mode_registry.FWD_TAB_SMEM_BUDGET_BYTES)
    else:
      waves = rules.waves_large if g_local >= rules.waves_g_cut else 4
      self.assertEqual(mode, f"direct_w{waves}")
      self.assertIsNone(kw["dedup_topology"])
      self.assertNotIn("dedup_fwd_max_runs", kw)
      self.assertNotIn("dedup_min_fanout", kw)
    self.assertEqual(kw["waves"], waves)
    # The mode name resolves to the same dispatch.
    resolved = mode_registry.resolve_mode(
        mode, ep=ep, g_local=g_local, topology=topology
    )
    self.assertEqual(resolved["waves"], kw["waves"])
    self.assertEqual(resolved["dedup_topology"], kw["dedup_topology"])

  def test_expected_board_fanout(self):
    self.assertAlmostEqual(layer_plan.expected_board_fanout(16, 10, 8), 3.83, 2)
    self.assertAlmostEqual(layer_plan.expected_board_fanout(32, 10, 8), 2.36, 2)
    self.assertAlmostEqual(layer_plan.expected_board_fanout(64, 10, 8), 1.63, 2)
    self.assertAlmostEqual(layer_plan.expected_board_fanout(16, 10, 4), 2.06, 2)
    # Fewer than two boards: no remote board.
    self.assertEqual(layer_plan.expected_board_fanout(8, 10, 8), 1.0)
    self.assertEqual(layer_plan.expected_board_fanout(4, 10, 4), 1.0)
    # More experts per token reach more devices of a board.
    self.assertGreater(
        layer_plan.expected_board_fanout(16, 10, 8),
        layer_plan.expected_board_fanout(16, 4, 8),
    )

  def test_large_static_stage_sends_direct(self):
    # Two boards of 8, t = 512: the static stage is 1 * 512 * 8192 * 2 B.
    rules = UnifiedRules(static_stage_max_mib=1.0)
    mode, kw, notes = _derive(16, 8, 512, rules)
    self.assertEqual(notes["static_stage_mib"], 8.0)
    self.assertIn("dedup_off", notes)
    self.assertFalse(notes["dedup"])
    self.assertIsNone(kw["dedup_topology"])
    self.assertNotIn("dedup_fwd_max_runs", kw)
    self.assertEqual(kw["waves"], 8)  # g_local = 32: direct-send waves.
    self.assertEqual(mode, "direct_w8")
    # The default limit keeps De-Dup on there.
    _, kw, notes = _derive(16, 8, 512)
    self.assertTrue(notes["dedup"])
    self.assertNotIn("dedup_off", notes)

  def test_waves_dedup_rule(self):
    rules = UnifiedRules(waves_dedup=8)
    mode, kw, _ = _derive(16, 8, 128, rules)
    self.assertEqual(mode, "dedup_w8")
    self.assertEqual(kw["waves"], 8)
    self.assertIn("dedup_fwd_max_runs", kw)
    # Eight boards send direct: g_local = 8 < waves_g_cut -> waves_small.
    mode, kw, _ = _derive(64, 8, 128, rules)
    self.assertEqual(mode, "direct_w4")

  def test_dedup_waves_are_capped_by_the_forward_tables(self):
    rules = UnifiedRules(waves_dedup=2 * dedup_plan_mod.MAX_PTW_WAVES)
    _, kw, _ = _derive(16, 8, 128, rules, e_total=16 * 64)
    self.assertLessEqual(kw["waves"], dedup_plan_mod.MAX_PTW_WAVES)
    self.assertEqual(kw["waves"], rules.waves_small)

  @parameterized.parameters((2, 2), (1, 1), (6, 2), (12, 4))
  def test_waves_halve_until_they_divide_g_local(self, g_local, waves):
    mode, kw, _ = _derive(16, None, 128, e_total=16 * g_local)
    self.assertEqual(kw["waves"], waves)
    self.assertEqual(mode, f"direct_w{waves}")

  def test_board_fanout_override(self):
    # A measured per-layer fan-out below the threshold turns De-Dup off.
    _, kw, notes = _derive(16, 8, 512, UnifiedRules(board_fanout=1.5))
    self.assertEqual(notes["board_fanout"], 1.5)
    self.assertIsNone(kw["dedup_topology"])
    # ... and a high one turns it on where the estimate would not.
    _, kw, _ = _derive(32, 8, 128, UnifiedRules(board_fanout=4.0))
    self.assertIsNotNone(kw["dedup_topology"])

  def test_dedup_min_boards(self):
    _, kw, _ = _derive(16, 8, 512, UnifiedRules(dedup_min_boards=4))
    self.assertIsNone(kw["dedup_topology"])
    _, kw, _ = _derive(32, 8, 512, UnifiedRules(dedup_min_boards=4))
    self.assertIsNotNone(kw["dedup_topology"])

  def test_min_fanout_above_the_board_cut(self):
    rules = UnifiedRules(min_fanout_boards_cut=2, board_fanout=4.0)
    _, kw, _ = _derive(32, 8, 512, rules)
    self.assertEqual(kw["dedup_min_fanout"], rules.min_fanout_large)
    _, kw, _ = _derive(16, 8, 512, rules)
    self.assertEqual(kw["dedup_min_fanout"], rules.min_fanout_small)


class StaticScheduleTest(parameterized.TestCase):

  def test_t_bucket(self):
    self.assertEqual(layer_plan.t_bucket(1), 8)
    self.assertEqual(layer_plan.t_bucket(8), 8)
    self.assertEqual(layer_plan.t_bucket(9), 16)
    self.assertEqual(layer_plan.t_bucket(12), 16)
    self.assertEqual(layer_plan.t_bucket(1000), 1024)
    self.assertEqual(layer_plan.t_bucket(1025), 2048)
    self.assertEqual(layer_plan.t_bucket(3000), 4096)
    self.assertEqual(layer_plan.t_bucket(5, (4, 6)), 6)
    self.assertEqual(layer_plan.t_bucket(5, (6, 4)), 6)
    with self.assertRaises(ValueError):
      layer_plan.t_bucket(0)

  def test_bucket_math(self):
    self.assertEqual(layer_plan.bucket_math(16), 0.0)
    self.assertEqual(layer_plan.bucket_math(12), 0.25)
    self.assertEqual(layer_plan.bucket_math(1), 7 / 8)

  def test_static_fwd_max_runs(self):
    # max(MAX_FWD_RUNS, next_pow2(factor * t)), capped by
    # FWD_TAB_SMEM_BUDGET_BYTES / (S * W * FWD_FIELDS * 4 B).
    self.assertEqual(dedup_plan_mod.MAX_FWD_RUNS, 128)
    self.assertEqual(layer_plan.static_fwd_max_runs(2, 4, 8), 128)
    self.assertEqual(layer_plan.static_fwd_max_runs(7, 4, 128), 128)
    self.assertEqual(layer_plan.static_fwd_max_runs(2, 4, 300), 512)
    self.assertEqual(layer_plan.static_fwd_max_runs(2, 8, 1024), 1024)
    self.assertEqual(layer_plan.static_fwd_max_runs(7, 8, 1024), 256)
    self.assertEqual(layer_plan.static_fwd_max_runs(2, 4, 128, factor=2.0), 256)

  @parameterized.parameters(
      *itertools.product((1, 3, 7), (1, 4, 8), (8, 128, 4096))
  )
  def test_static_fwd_max_runs_is_a_power_of_two_in_budget(self, s, w, t):
    runs = layer_plan.static_fwd_max_runs(s, w, t)
    self.assertEqual(runs & (runs - 1), 0)
    self.assertGreaterEqual(runs, dedup_plan_mod.MAX_FWD_RUNS)
    if runs > dedup_plan_mod.MAX_FWD_RUNS:
      self.assertLessEqual(
          s * w * runs * dedup_plan_mod.FWD_FIELDS * 4,
          mode_registry.FWD_TAB_SMEM_BUDGET_BYTES,
      )

  @parameterized.parameters(
      (4, "direct_w4"),
      (8, "direct_w4"),
      (32, "direct_w4"),
      (2, "direct_w2"),
      (6, "direct_w2"),
      (3, "direct_w1"),
      (1, "direct_w1"),
  )
  def test_fallback_mode(self, g_local, want):
    self.assertEqual(layer_plan.fallback_mode(g_local), want)
    self.assertEqual(layer_plan.FALLBACK_MODE, mode_registry.FALLBACK_MODE)


class PolicyTest(parameterized.TestCase):

  def test_auto_is_the_default_policy(self):
    self.assertEqual(_cfg(16, 8).policy, "auto")

  def test_rules_policy(self):
    cfg = _cfg(16, 8, policy="rules")
    self.assertEqual(layer_plan.resolve_policy(cfg, 128), "rules")
    p = layer_plan.plan(cfg, 100)
    self.assertEqual(p.t_bucket, 128)
    self.assertEqual(p.mode, layer_plan.RULES_MODE)
    mode, kw, _ = _derive(16, 8, 128)
    self.assertEqual(p.kwargs, kw)
    self.assertIn(("rules_mode", mode), p.notes)
    self.assertNotIn("policy", dict(p.notes))

  @parameterized.parameters("direct_w4", "dedup_w4", "dedup_w2", "direct_w8")
  def test_pinned_mode_resolves(self, mode):
    cfg = _cfg(16, 8, policy=mode)
    self.assertEqual(layer_plan.resolve_policy(cfg, 128), mode)
    p = layer_plan.plan(cfg, 128)
    self.assertEqual(p.mode, mode)
    self.assertNotIn("fallback_reason", dict(p.notes))
    want = mode_registry.resolve_mode(
        mode, ep=16, g_local=cfg.g_local, topology=cfg.topology
    )
    got = dict(p.kwargs)
    if mode.startswith("dedup"):
      # The plan sizes the forward-table depth statically.
      self.assertEqual(
          got.pop("dedup_fwd_max_runs"),
          layer_plan.static_fwd_max_runs(
              cfg.topology.tables().num_slots,
              want["waves"],
              128,
              factor=cfg.rules.fwd_runs_factor,
          ),
      )
    self.assertEqual(got, want)

  def test_unknown_policy_raises(self):
    with self.assertRaisesRegex(ValueError, "unknown mode"):
      layer_plan.resolve_policy(_cfg(16, 8, policy="table"), 128)

  @parameterized.named_parameters(
      ("dedup_without_topology", None, "dedup_w4", 32),
      ("waves_do_not_divide", 8, "dedup_w8", 4),
      ("direct_waves_do_not_divide", None, "direct_w8", 4),
  )
  def test_pinned_mode_falls_back(self, board_size, mode, g_local):
    cfg = _cfg(16, board_size, policy=mode, num_experts=16 * g_local)
    p = layer_plan.plan(cfg, 64)
    self.assertEqual(p.mode, layer_plan.fallback_mode(g_local))
    self.assertIn("fallback_reason", dict(p.notes))
    self.assertIsNone(p.kwargs["dedup_topology"])
    self.assertNotIn("dedup_fwd_max_runs", p.kwargs)

  # Points no measured table covers: `auto` takes the rules there.
  @parameterized.named_parameters(
      ("no_topology", 16, None, "fp8", 128),
      ("decode_bucket", 32, 8, "fp8", 8),
      ("above_the_largest_bucket", 32, 8, "fp8", 4096),
      ("bf16", 16, 8, "bf16", 512),
      ("board_size_4", 16, 4, "fp8", 256),
      ("eight_boards_of_4", 32, 4, "fp8", 512),
      ("three_boards", 24, 8, "fp8", 256),
  )
  def test_auto_takes_the_rules_where_unmeasured(self, ep, board_size, fmt, t):
    e_total = 24 * 16 if ep == 24 else E_TOTAL
    cfg = _cfg(ep, board_size, weight_format=fmt, num_experts=e_total)
    self.assertIsNone(
        layer_plan.measured_mode(cfg.topology, layer_plan.t_bucket(t), fmt)
    )
    self.assertEqual(layer_plan.resolve_policy(cfg, t), "rules")
    got = layer_plan.plan(cfg, t)
    ref = layer_plan.plan(dataclasses.replace(cfg, policy="rules"), t)
    self.assertEqual(
        (got.t_bucket, got.mode, got.kwargs),
        (ref.t_bucket, ref.mode, ref.kwargs),
    )
    self.assertIn(("policy", "auto->rules"), got.notes)

  def test_custom_gateways_are_never_measured(self):
    ep, bs = 16, 8
    gateways = [
        [(b * bs + (s + 1) % bs) for b in range(ep // bs)] for s in range(ep)
    ]
    topology = BoardTopology.uniform(ep, bs, gateways=gateways)
    for t in layer_config.DEFAULT_T_BUCKETS:
      self.assertIsNone(layer_plan.measured_mode(topology, t, "fp8"))

  def test_every_measured_mode_is_a_mode(self):
    for rows in layer_plan.MEASURED_POLICY.values():
      for _, mode in rows:
        self.assertIn(mode, mode_registry.MODES)

  def test_custom_t_buckets(self):
    cfg = _cfg(16, 8, policy="rules", t_buckets=(4, 12, 24))
    self.assertEqual(layer_plan.plan(cfg, 10).t_bucket, 12)
    self.assertEqual(layer_plan.plan(cfg, 25).t_bucket, 32)


class ConfigTest(absltest.TestCase):

  def test_topology_must_cover_ep(self):
    with self.assertRaisesRegex(ValueError, "topology covers"):
      layer_config.DedupMoeConfig(
          ep=32,
          hidden=HIDDEN,
          num_experts=E_TOTAL,
          top_k=TOPK,
          ffn=FFN,
          topology=BoardTopology.uniform(16, 8),
      )

  def test_experts_must_split_over_ep(self):
    with self.assertRaisesRegex(ValueError, "does not split"):
      layer_config.DedupMoeConfig(
          ep=16, hidden=HIDDEN, num_experts=100, top_k=TOPK, ffn=FFN
      )

  def test_g_local(self):
    self.assertEqual(_cfg(16, 8).g_local, 32)

  def test_placement_must_be_a_permutation(self):
    with self.assertRaises(NotImplementedError):
      layer_config.check_placement([0, 0, 1, 2], 4)
    with self.assertRaises(ValueError):
      layer_config.check_placement([0, 1, 2], 4)
    self.assertIsNone(layer_config.check_placement(range(4), 4))
    self.assertIsNone(layer_config.check_placement(None, 4))
    placement = layer_config.check_placement([1, 0, 3, 2], 4)
    assert placement is not None
    self.assertEqual(list(placement), [1, 0, 3, 2])


if __name__ == "__main__":
  absltest.main()
