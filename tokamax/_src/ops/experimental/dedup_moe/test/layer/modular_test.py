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
"""End-to-end tests of layer/modular.dedup_moe (dual mode, see hw_util).

`dedup_moe` is checked against
  * the dense f32 reference (reference.ep_reference, rel-L2 bound),
  * `monolithic.dedup_ep_moe` called with the plan's keyword arguments (bit
    exact: the modular entry adds nothing but padding and placement),
  * the direct W=4 mode (bit exact: every mode computes the same arithmetic),
for the rules config with De-Dup forced on (board sizes 4 and 8), pinned
modes, an expert placement (canonical routing makes it invisible), bucket
padding (pad rows never change real rows) and the fallback of a pinned mode
that cannot run.

CPU interpret mode by default (small shapes, up to 16 CPU devices); on TPU
v8i the same cases run at hardware-legal shapes and the cases needing more
devices than the host has skip.
"""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib
from tokamax._src.ops.experimental.dedup_moe.layer import modular
from tokamax._src.ops.experimental.dedup_moe.layer import monolithic
from tokamax._src.ops.experimental.dedup_moe.layer import plan as layer_plan
from tokamax._src.ops.experimental.dedup_moe.test.common import hw_util
from tokamax._src.ops.experimental.dedup_moe.test.common import moe_inputs
from tokamax._src.ops.experimental.dedup_moe.test.common import reference

# CPU interpret mode's devices, set in __main__ before the backend starts (on
# TPU this only sizes the unused CPU backend). Cases that need more devices
# than JAX has skip.
MAX_CPU_DEVICES = 16

BoardTopology = topology_lib.BoardTopology
G_LOCAL = 4
TOPK = 4
# The CPU interpret shapes; setUpModule switches to hardware-legal ones on
# TPU (hw_util decides only after absl has parsed the flags).
HIDDEN = 256
INTER = 256
CAPACITY = 16
T_LOCAL = 16
# rel-L2 bound against the f32 reference: fp8 (W8A8) also quantizes the rows,
# the intermediate and the wire.
REL_L2_BOUND = {"bf16": 2e-2, "fp8": 1e-1}
# De-Dup on whatever the routing: the small test shapes are below the rules'
# thresholds.
FORCED_DEDUP = modular.UnifiedRules(
    dedup_min_board_fanout=0.0, dedup_min_savings=0.0
)
GATE_MODE = "direct_w4"


def setUpModule():
  global HIDDEN, INTER, CAPACITY
  hw_util.setup_module("tpu8i")
  HIDDEN = hw_util.pick(HIDDEN, hw_util.HW_HIDDEN)
  INTER = hw_util.pick(INTER, hw_util.HW_INTER)
  CAPACITY = hw_util.check_capacity(hw_util.pick(CAPACITY, hw_util.HW_CAPACITY))


class ModularLayerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_v8i(self)
    hw_util.reset_interpret_state()

  def _setup(self, ep, board_size, g_local=G_LOCAL, fmt="fp8", **cfg_kw):
    mesh = hw_util.make_mesh(ep, self)
    cfg = modular.DedupMoeConfig(
        ep=ep,
        hidden=HIDDEN,
        num_experts=ep * g_local,
        top_k=TOPK,
        ffn=INTER,
        topology=(
            None
            if board_size is None
            else BoardTopology.uniform(ep, board_size)
        ),
        weight_format=fmt,
        capacity=CAPACITY,
        **cfg_kw,
    )
    weights = moe_inputs.random_weights(
        mesh,
        e_total=cfg.num_experts,
        hidden=HIDDEN,
        inter=INTER,
        weight_format=fmt,
        seed=0,
    )
    return mesh, cfg, weights

  def _inputs(self, mesh, cfg, t_local, seed=1):
    return moe_inputs.random_inputs(
        mesh,
        tokens=cfg.ep * t_local,
        hidden=HIDDEN,
        e_total=cfg.num_experts,
        seed=seed,
    )

  def _modular(
      self, mesh, cfg, weights, x, gating, placement=None, topk_ids=None
  ):
    prepared = modular.prepare_weights(
        mesh, *weights, cfg=cfg, placement=placement
    )
    with hw_util.cpu_layer():
      out = hw_util.run(
          lambda x, g, prep, ids: modular.dedup_moe(
              x, g, prep, cfg, topk_ids=ids
          ),
          x,
          gating,
          prepared,
          topk_ids,
      )
      return np.asarray(out, np.float32)

  def _monolithic(self, mesh, cfg, kwargs, weights, x, gating):
    with hw_util.cpu_layer():
      out = hw_util.run(
          lambda x, g, w: monolithic.dedup_ep_moe(
              x,
              w[0],
              w[1],
              w[2],
              w[3],
              g,
              topk=TOPK,
              renormalize=True,
              mesh=mesh,
              capacity=CAPACITY,
              weight_format=arg.WeightFormat(cfg.weight_format),
              **kwargs,
          ),
          x,
          gating,
          weights,
      )
      return np.asarray(out, np.float32)

  def _assert_close(self, got, mesh, weights, x, gating, fmt, ids=None):
    w1, w2, w1s, w2s = weights
    want = reference.ep_reference(
        x, w1, w2, w1s, w2s, gating, topk=TOPK, mesh=mesh, hash_topk_ids=ids
    )
    self.assertFalse(np.isnan(got).any())
    self.assertLess(reference.rel_l2(jnp.asarray(got), want), REL_L2_BOUND[fmt])

  @parameterized.named_parameters(
      ("ep4_fp8", 4, "fp8"),
      ("ep8_bf16", 8, "bf16"),
  )
  def test_direct_matches_reference_and_monolithic(self, ep, fmt):
    mesh, cfg, weights = self._setup(ep, None, fmt=fmt)
    x, gating = self._inputs(mesh, cfg, T_LOCAL)
    p = modular.plan(cfg, T_LOCAL)
    self.assertEqual(p.mode, layer_plan.RULES_MODE)
    self.assertIsNone(p.kwargs["dedup_topology"])
    got = self._modular(mesh, cfg, weights, x, gating)
    self.assertEqual(got.shape, (ep * T_LOCAL, HIDDEN))
    want = self._monolithic(mesh, cfg, p.kwargs, weights, x, gating)
    np.testing.assert_array_equal(got, want)
    self._assert_close(got, mesh, weights, x, gating, fmt)

  @parameterized.named_parameters(
      ("ep16_bs8", 16, 8),
      ("ep16_bs4", 16, 4),
      ("ep8_bs4", 8, 4),
  )
  def test_rules_dedup_bit_exact_vs_monolithic_and_direct(self, ep, board_size):
    mesh, cfg, weights = self._setup(
        ep, board_size, policy="rules", rules=FORCED_DEDUP
    )
    x, gating = self._inputs(mesh, cfg, T_LOCAL)
    p = modular.plan(cfg, T_LOCAL)
    self.assertEqual(p.kwargs["dedup_topology"], cfg.topology)
    self.assertIn(("rules_mode", f"dedup_w{p.kwargs['waves']}"), p.notes)
    got = self._modular(mesh, cfg, weights, x, gating)
    want = self._monolithic(mesh, cfg, p.kwargs, weights, x, gating)
    np.testing.assert_array_equal(got, want)
    # De-Dup is bit-identical to the direct wave dispatch.
    gate = self._modular(
        mesh, dataclasses.replace(cfg, policy=GATE_MODE), weights, x, gating
    )
    np.testing.assert_array_equal(got, gate)
    self._assert_close(got, mesh, weights, x, gating, cfg.weight_format)

  @parameterized.named_parameters(
      ("ep8_direct_w1", 8, None, "direct_w1", 4),
      ("ep8_direct_w8", 8, None, "direct_w8", 8),
      ("ep8_bs4_dedup_w2", 8, 4, "dedup_w2", 4),
      ("ep16_bs8_dedup_w8", 16, 8, "dedup_w8", 8),
  )
  def test_pinned_mode_bit_exact_vs_direct(self, ep, board_size, mode, g_local):
    mesh, cfg, weights = self._setup(
        ep, board_size, g_local=g_local, policy=mode
    )
    p = modular.plan(cfg, T_LOCAL)
    self.assertEqual(p.mode, mode)
    self.assertNotIn("fallback_reason", dict(p.notes))
    x, gating = self._inputs(mesh, cfg, T_LOCAL)
    got = self._modular(mesh, cfg, weights, x, gating)
    want = self._monolithic(mesh, cfg, p.kwargs, weights, x, gating)
    np.testing.assert_array_equal(got, want, err_msg=mode)
    gate = self._modular(
        mesh, dataclasses.replace(cfg, policy=GATE_MODE), weights, x, gating
    )
    np.testing.assert_array_equal(got, gate, err_msg=f"{mode} vs {GATE_MODE}")

  @parameterized.named_parameters(
      ("ep8_bs4_dedup", 8, 4, dict(policy="rules", rules=FORCED_DEDUP)),
      ("ep4_direct", 4, None, dict(policy="auto")),
  )
  def test_placement_round_trip(self, ep, board_size, cfg_kw):
    mesh, cfg, weights = self._setup(ep, board_size, **cfg_kw)
    x, gating = self._inputs(mesh, cfg, T_LOCAL)
    perm = np.random.default_rng(7).permutation(cfg.num_experts)
    perm = perm.astype(np.int32)
    base = self._modular(mesh, cfg, weights, x, gating)
    canon_cfg = dataclasses.replace(cfg, canonical_routing=True)
    canon = self._modular(mesh, canon_cfg, weights, x, gating)
    placed = self._modular(mesh, cfg, weights, x, gating, placement=perm)
    # Token for token: the placement is invisible (canonical routing is on
    # with a placement, so compare with it on and no placement).
    np.testing.assert_array_equal(placed, canon)
    # The layer's own softmax router agrees to about one bf16 ulp.
    np.testing.assert_allclose(placed, base, rtol=2e-2, atol=1e-5)
    # The placement given through the config is the same placement.
    placed_cfg = dataclasses.replace(cfg, placement=tuple(int(i) for i in perm))
    np.testing.assert_array_equal(
        self._modular(mesh, placed_cfg, weights, x, gating), placed
    )

  @parameterized.named_parameters(
      ("ep8_direct", 8, None, dict(policy="auto")),
      ("ep8_bs4_dedup", 8, 4, dict(policy="rules", rules=FORCED_DEDUP)),
  )
  def test_padding_rows_are_invisible(self, ep, board_size, cfg_kw):
    t_real, t_full = 12, 16  # 12 is not a bucket; it runs at 16.
    mesh, cfg, weights = self._setup(ep, board_size, **cfg_kw)
    self.assertEqual(modular.plan(cfg, t_real).t_bucket, t_full)
    x_full, g_full = self._inputs(mesh, cfg, t_full)
    keep = np.arange(ep * t_full).reshape(ep, t_full)[:, :t_real].reshape(-1)
    x = moe_inputs.shard_rows(mesh, np.asarray(x_full)[keep])
    gating = moe_inputs.shard_rows(mesh, np.asarray(g_full)[keep])
    got = self._modular(mesh, cfg, weights, x, gating)
    full = self._modular(mesh, cfg, weights, x_full, g_full)
    self.assertEqual(got.shape, (ep * t_real, HIDDEN))
    np.testing.assert_array_equal(got, full[keep])
    self._assert_close(got, mesh, weights, x, gating, cfg.weight_format)

  def test_padding_with_caller_topk_ids(self):
    ep, t_real = 8, 12
    mesh, cfg, weights = self._setup(ep, 4, policy="rules", rules=FORCED_DEDUP)
    x, gating = self._inputs(mesh, cfg, t_real)
    ids = moe_inputs.expert_ids(
        "uniform", ep * t_real, cfg.num_experts, TOPK, G_LOCAL, seed=3
    )
    ids = moe_inputs.shard_rows(mesh, ids)
    got = self._modular(mesh, cfg, weights, x, gating, topk_ids=ids)
    self.assertEqual(got.shape, (ep * t_real, HIDDEN))
    self._assert_close(
        got, mesh, weights, x, gating, cfg.weight_format, ids=ids
    )

  def test_pinned_mode_that_cannot_run_falls_back(self):
    # A De-Dup mode without a topology runs the fallback (direct) mode.
    mesh, cfg, weights = self._setup(8, None, policy="dedup_w4")
    p = modular.plan(cfg, T_LOCAL)
    self.assertEqual(p.mode, layer_plan.fallback_mode(G_LOCAL))
    self.assertIn("fallback_reason", dict(p.notes))
    x, gating = self._inputs(mesh, cfg, T_LOCAL)
    got = self._modular(mesh, cfg, weights, x, gating)
    want = self._modular(
        mesh, dataclasses.replace(cfg, policy=p.mode), weights, x, gating
    )
    np.testing.assert_array_equal(got, want)

  def test_tokens_must_split_over_ep(self):
    mesh, cfg, weights = self._setup(4, None)
    prepared = modular.prepare_weights(mesh, *weights, cfg=cfg)
    x = jnp.zeros((4 * T_LOCAL + 1, HIDDEN), jnp.bfloat16)
    g = jnp.zeros((4 * T_LOCAL + 1, cfg.num_experts), jnp.float32)
    with self.assertRaisesRegex(ValueError, "do not split"):
      modular.dedup_moe(x, g, prepared, cfg)


if __name__ == "__main__":
  jax.config.update("jax_num_cpu_devices", MAX_CPU_DEVICES)
  absltest.main()
