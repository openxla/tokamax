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
"""Ramp expert waves: wave_dispatch.wave_plan and the exchange it lays out.

* wave_plan assigns the ramp waves of a reference cost model (ported below
  with the integer costs the layer uses). The visit order keeps every wave
  contiguous and ascending, fewest rows first inside a wave, empty experts
  last, and the bounds row maps each visit position to its expert's wave
  the way the kernel reads it (wave_dispatch.wave_bounds_row).
* The fixed-slot exchange (source_plan), replayed in numpy with those
  waves, delivers every needed row exactly once, in the first-hit wave of
  the token on its destination, and the rows each shard waits for per wave
  are exactly what the senders send.
* The layer (CPU interpret mode with the emulated exchange, or TPU v8i
  hardware) is bit-exact with the direct W=4 gate for other wave counts:
  the waves change only when rows land and which expert runs when.
"""

import functools
import itertools
import types
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.dispatch import source_plan as sp
from tokamax._src.ops.experimental.dedup_moe.dispatch import wave_dispatch
from tokamax._src.ops.experimental.dedup_moe.layer.monolithic import dedup_ep_moe
from tokamax._src.ops.experimental.dedup_moe.test.common import hw_util
from tokamax._src.ops.experimental.dedup_moe.test.common import moe_inputs
from tokamax._src.ops.experimental.fused_moe import host as fused_host


def setUpModule():
  hw_util.setup_module("tpu8i")


def _ref_waves(n, waves):
  """The reference ramp wave assignment for one device, with integer costs.

  The model's cost tau + c * n is EXPERT_COST_ROWS + n in units of c here, and
  the ramp cut compares S * |cs - target| in integers, as wave_plan does.

  Args:
    n: [g] rows per local expert.
    waves: W.

  Returns:
    [g] wave of each local expert.
  """
  g = n.size
  n = n.astype(np.int64)
  key = np.where(n > 0, n, 1 << 40) * g + np.arange(g)
  asc = np.argsort(key, kind="stable")
  cost = np.where(n > 0, wave_dispatch.EXPERT_COST_ROWS + n, 0)
  wl = np.full(g, waves - 1, np.int64)
  m = int((n > 0).sum())
  if m <= waves:
    wl[asc[:m]] = np.arange(m)
    return wl
  cs = np.cumsum(cost[asc[:m]])
  share = wave_dispatch.RAMP_RATIO ** np.arange(waves)
  cum = np.cumsum(share)
  cuts, prev = [], 0
  for w in range(waves - 1):
    cand = np.arange(prev + 1, m - (waves - 1 - w) + 1)
    err = np.abs(cs[cand - 1] * share.sum() - cs[-1] * cum[w])
    c = int(cand[np.argmin(err)])
    cuts.append(c)
    prev = c
  wl[asc[:m]] = np.searchsorted(np.array(cuts), np.arange(m), side="right")
  return wl


def _hists(ep, g_local, seed):
  """Per-expert row counts: uniform, skewed, with empty experts and shards."""
  rng = np.random.default_rng(seed)
  out = {
      "uniform": rng.integers(140, 180, (ep, g_local)),
      "skewed": (
          rng.integers(1, 400, (ep, g_local))
          * (rng.random((ep, g_local)) < 0.8)
      ),
      "ties": np.full((ep, g_local), 160),
      "mostly_empty": (
          rng.integers(1, 9, (ep, g_local)) * (rng.random((ep, g_local)) < 0.3)
      ),
  }
  # One shard with no rows at all, one with a single nonempty expert.
  out["mostly_empty"][0] = 0
  out["mostly_empty"][-1] = 0
  out["mostly_empty"][-1, g_local // 2] = 5
  return {k: v.astype(np.int32).reshape(-1) for k, v in out.items()}


_SHAPES = (
    dict(ep=4, g_local=16, waves=4),
    dict(ep=4, g_local=16, waves=8),
    dict(ep=8, g_local=8, waves=4),
    dict(ep=8, g_local=8, waves=8),
    dict(ep=3, g_local=16, waves=2),
    dict(ep=5, g_local=4, waves=2),
    dict(ep=4, g_local=4, waves=4),
)


def _plan_cases():
  for shape in _SHAPES:
    case: dict[str, Any] = dict(shape)
    case["testcase_name"] = (
        f"ep{shape['ep']}_g{shape['g_local']}_w{shape['waves']}"
    )
    yield case


class WavePlanTest(parameterized.TestCase):
  """wave_plan against the reference ramp, on the host."""

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_v8i(self)

  @parameterized.named_parameters(*_plan_cases())
  def test_waves_match_the_model_and_the_kernel_reads_them_back(
      self, ep, g_local, waves
  ):
    for name, hist in _hists(
        ep, g_local, seed=ep * 100 + g_local + waves
    ).items():
      with self.subTest(hist=name):
        with hw_util.host_only():
          visit, wave_of, bounds = (
              np.asarray(a)
              for a in wave_dispatch.wave_plan(
                  jnp.asarray(hist), ep=ep, g_local=g_local, waves=waves
              )
          )
        h = hist.reshape(ep, g_local)
        for d in range(ep):
          n = h[d]
          m = int((n > 0).sum())
          ref = _ref_waves(n, waves)
          np.testing.assert_array_equal(wave_of[d][n > 0], ref[n > 0])
          self.assertTrue((wave_of[d][n == 0] == waves - 1).all())
          # The visit list: a permutation, nonempty experts first.
          self.assertEqual(sorted(visit[d].tolist()), list(range(g_local)))
          self.assertTrue((n[visit[d][:m]] > 0).all())
          self.assertTrue((n[visit[d][m:]] == 0).all())
          # Waves contiguous and ascending; fewest rows first inside a wave.
          vw = wave_of[d][visit[d][:m]]
          vn = n[visit[d][:m]]
          self.assertTrue((np.diff(vw) >= 0).all())
          same = np.diff(vw) == 0
          self.assertTrue((np.diff(vn)[same] >= 0).all())
          # The bounds row, read the kernel's way.
          self.assertEqual(int(bounds[d, -1]), m)
          self.assertTrue((np.diff(bounds[d]) >= 0).all())
          for p in range(m):
            kernel_wave = int((bounds[d, : waves - 1] <= p).sum())
            self.assertEqual(kernel_wave, int(vw[p]), f"shard {d} visit {p}")
          if m >= waves:
            # Every wave has an expert: no empty wave ahead of a full one.
            self.assertTrue(
                (np.diff(np.concatenate([[0], bounds[d]])) > 0).all()
            )

  def test_ramp_grows_the_waves(self):
    # 16 equal experts: ramp cuts 1 : 2 : 4 : 9 experts (shares 1 : 2 : 4 : 8).
    with hw_util.host_only():
      hist = jnp.full((16,), 160, jnp.int32)
      _, _, ramp = wave_dispatch.wave_plan(hist, ep=1, g_local=16, waves=4)
      ramp = np.asarray(ramp)
    np.testing.assert_array_equal(ramp[0], [1, 3, 7, 16])


# -- the exchange with these waves, on the host --------------------------------

_EXCHANGE_SHAPES = (
    dict(
        testcase_name="ep4_g8_w4", ep=4, t_local=16, g_local=8, topk=3, waves=4
    ),
    dict(
        testcase_name="ep4_g8_w2", ep=4, t_local=16, g_local=8, topk=3, waves=2
    ),
    dict(
        testcase_name="ep8_g4_w4", ep=8, t_local=16, g_local=4, topk=4, waves=4
    ),
)


def _exchange_cases():
  for shape, routing in itertools.product(
      _EXCHANGE_SHAPES, moe_inputs.ROUTINGS
  ):
    case = dict(shape)
    case["testcase_name"] = f"{shape['testcase_name']}_{routing}"
    case.update(routing=routing)
    yield case


class WaveExchangeTest(parameterized.TestCase):
  """The fixed-slot exchange with ramp waves, replayed in numpy."""

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_v8i(self)

  @parameterized.named_parameters(*_exchange_cases())
  def test_every_needed_row_lands_once_in_its_first_wave(
      self, routing, ep, t_local, g_local, topk, waves
  ):
    T = ep * t_local  # pylint: disable=invalid-name
    ids = moe_inputs.expert_ids(
        routing, T, ep * g_local, topk, g_local, seed=ep + waves + t_local
    )
    send_alloc = fused_host.align_up(t_local * min(topk, ep), fused_host.ROWBLK)
    with hw_util.host_only():
      hist = jnp.asarray(
          np.bincount(ids.reshape(-1), minlength=ep * g_local), jnp.int32
      )
      _, wave_of, _ = wave_dispatch.wave_plan(
          hist, ep=ep, g_local=g_local, waves=waves
      )
      kw = dict(
          ep=ep, t_local=t_local, g_local=g_local, waves=waves, wave_of=wave_of
      )
      pairs = jnp.asarray(ids)
      plans = [sp.source_plan(pairs, jnp.int32(s), **kw) for s in range(ep)]
      tabs, perms, rows_on_dest = [], [], []
      for s in range(ep):
        recv = sp.receiver_counts(pairs, jnp.int32(s), **kw)
        tabs.append(
            np.asarray(
                sp.dispatch_table(plans[s], jnp.int32(s), recv, t_local=t_local)
            )
        )
        perms.append(np.asarray(sp.send_perm(plans[s], send_alloc=send_alloc)))
        rows_on_dest.append(np.asarray(plans[s].row_on_dest))
      wave_np = np.asarray(wave_of)
    landed = np.full((ep, T), -1)
    landed_wave = np.full((ep, T), -1)
    for src in range(ep):
      tab, perm = tabs[src], perms[src]
      for d in range(ep):
        for w in range(waves):
          off, n, dst = tab[d, w], tab[ep + d, w], tab[2 * ep + d, w]
          self.assertTrue(
              (landed[d, dst : dst + n] == -1).all(), "rows overlap"
          )
          landed[d, dst : dst + n] = src * t_local + perm[off : off + n]
          landed_wave[d, dst : dst + n] = w
    dest = ids // g_local
    for d in range(ep):
      # The kernel waits for exactly what the senders send, wave by wave.
      for w in range(waves):
        self.assertEqual(tabs[d][3 * ep, w], sum(tb[ep + d, w] for tb in tabs))
      needed = np.flatnonzero((dest == d).any(axis=1))
      self.assertEqual(int((landed[d] >= 0).sum()), needed.size)
      for t in needed:
        s, tl = divmod(int(t), t_local)
        row = rows_on_dest[s][d, tl]
        self.assertEqual(landed[d, row], t)
        first = min(
            int(wave_np[d, e % g_local]) for e in ids[t] if e // g_local == d
        )
        self.assertEqual(landed_wave[d, row], first)


# -- the layer -----------------------------------------------------------------

# 8 CPU devices in interpret mode; on hardware 4 TPU v8i devices (one host).
CPU_DEVICES = 8
G_LOCAL = 8
TOPK = 4


@functools.cache
def _shapes() -> types.SimpleNamespace:
  """The layer shapes (hw_util.pick needs JAX, so not at import time)."""
  ep = hw_util.pick(8, 4)
  return types.SimpleNamespace(
      ep=ep,
      e_total=ep * G_LOCAL,
      hidden=hw_util.pick(512, hw_util.HW_HIDDEN),
      inter=hw_util.pick(512, hw_util.HW_INTER),
      capacity=hw_util.pick(32, hw_util.HW_CAPACITY),
  )


ROWS = 16
# The bit-exactness gate: the direct wave dispatch with this many waves.
GATE_WAVES = 4


def _layer(mesh, x, weights, gating, ids, *, waves):
  """One layer call (direct wave dispatch), as f32 numpy."""

  def fn(x, w1, w2, w1_scale, w2_scale, gating, ids):
    return dedup_ep_moe(
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        gating,
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=_shapes().capacity,
        weight_format=arg.WeightFormat.FP8,
        hash_topk_ids=ids,
        waves=waves,
    )

  with hw_util.cpu_layer():
    return np.asarray(hw_util.run(fn, x, *weights, gating, ids), np.float32)


_LAYER_CASES = (
    dict(testcase_name="w8_uniform", waves=8, routing="uniform"),
    dict(testcase_name="w2_skewed", waves=2, routing="skewed"),
    dict(testcase_name="w8_dominant", waves=8, routing="dominant"),
    dict(testcase_name="w2_worst", waves=2, routing="worst"),
)


class WaveOrderLayerTest(parameterized.TestCase):
  """Bit equality with the W=4 gate: CPU interpret mode or TPU v8i."""

  # The mesh and weights, built by the first test that runs.
  _cache: dict[str, Any] = {}

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_v8i(self)
    if "mesh" not in self._cache:
      mesh = hw_util.make_mesh(_shapes().ep, self)
      self._cache["mesh"] = mesh
      self._cache["weights"] = moe_inputs.random_weights(
          mesh,
          e_total=_shapes().e_total,
          hidden=_shapes().hidden,
          inter=_shapes().inter,
          weight_format="fp8",
          seed=0,
      )
    self.mesh = self._cache["mesh"]
    self.weights = self._cache["weights"]
    hw_util.reset_interpret_state()

  @parameterized.named_parameters(*_LAYER_CASES)
  def test_bit_exact_with_the_w4_gate(self, waves, routing):
    tokens = ROWS * _shapes().ep
    x, gating = moe_inputs.random_inputs(
        self.mesh,
        tokens=tokens,
        hidden=_shapes().hidden,
        e_total=_shapes().e_total,
        seed=waves,
    )
    ids = moe_inputs.shard_rows(
        self.mesh,
        moe_inputs.expert_ids(
            routing, tokens, _shapes().e_total, TOPK, G_LOCAL, 1
        ),
    )
    want = _layer(self.mesh, x, self.weights, gating, ids, waves=GATE_WAVES)
    got = _layer(self.mesh, x, self.weights, gating, ids, waves=waves)
    self.assertTrue(np.isfinite(want).all())
    np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
  jax.config.update("jax_num_cpu_devices", CPU_DEVICES)
  absltest.main()
