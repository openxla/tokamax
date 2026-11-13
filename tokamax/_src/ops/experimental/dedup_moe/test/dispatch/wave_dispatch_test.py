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
"""Wave dispatch tests: the dispatch plan and the layer's wave counts.

  * The plan tests run on the host: the reference dispatch plan
    (wave_dispatch.dispatch_plan) with ramp waves, replayed in numpy.
  * The layer tests run on CPU devices in Pallas TPU interpret mode with the
    split-phase exchange emulated (interpret_util.emulate_wave_exchange), at
    reduced shapes (LAYER_HIDDEN, LAYER_INTER, LAYER_CAPACITY, LAYER_ROWS), or
    on TPU v8i devices with the real kernels (hw_util). The gate is the direct
    wave dispatch with W=4 (dedup_ep_moe(waves=4, dedup_topology=None)):
    every other wave count must match it bit for bit.
  * The rejection test needs no kernel at all (every refusal precedes the
    first kernel), so it runs on CPU devices.

Wave dispatch changes only how the routed rows reach the expert shards and in
which order the kernel visits the experts: the same rows arrive by a pairwise
exchange, in waves, and the kernel runs the same matmuls on them. So the
number of waves never changes the output bits.

The routing is forced through `hash_topk_ids`, so each case pins how the
pairs spread over the shards:

  uniform  random experts: every shard is near its uniform share;
  skewed   7 of 8 tokens on shard 0's experts only;
  worst    every token on experts 0..topk-1, i.e. all of it on shard 0 -- the
           largest exchange and expert slab any routing can produce, which is
           what the no-drop buffer bounds are sized for;
  dominant every shard's local expert 0 takes one token in ep, from every
           source shard: the one-big-expert-per-shard shape a placement
           table leaves.

waves=1 waits for the whole exchange before the first expert; more waves
wait for each group of experts before its first tile, smallest experts first.

The plan tests replay the exchange the dispatch plan describes in numpy: every
row a shard needs lands exactly once, no later than the wave of the first of
its experts on that shard, and the per-wave counts the kernel waits for are
exactly what the senders send.
"""

import functools
import itertools
import types
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.common.arg import WeightFormat
from tokamax._src.ops.experimental.dedup_moe.dispatch import wave_dispatch
from tokamax._src.ops.experimental.dedup_moe.fused_kernel.config import AXIS
from tokamax._src.ops.experimental.dedup_moe.layer.monolithic import dedup_ep_moe
from tokamax._src.ops.experimental.dedup_moe.test.common import hw_util
from tokamax._src.ops.experimental.dedup_moe.test.common import interpret_util
from tokamax._src.ops.experimental.dedup_moe.test.common import moe_inputs
from tokamax._src.ops.experimental.fused_moe import host as fused_host

_expert_ids = moe_inputs.expert_ids

# CPU devices the test process creates (the layer and rejection tests).
CPU_DEVICES = 8
EXPERTS_PER_SHARD = 4
HIDDEN = 1024
INTER = 512
TOPK = 4
CAPACITY = 128
FP8 = jnp.float8_e4m3fn

# The layer tests: 8 CPU devices in interpret mode; on hardware 4 TPU v8i
# devices (one host) at the smallest hardware-legal shapes.


@functools.cache
def _shapes() -> types.SimpleNamespace:
  """The layer shapes (hw_util.pick needs JAX, so not at import time)."""
  ep = hw_util.pick(8, 4)
  return types.SimpleNamespace(
      ep=ep,
      e_total=ep * EXPERTS_PER_SHARD,
      hidden=hw_util.pick(256, hw_util.HW_HIDDEN),
      inter=hw_util.pick(256, hw_util.HW_INTER),
      capacity=hw_util.pick(32, hw_util.HW_CAPACITY),
  )


LAYER_ROWS = (16, 32)
# The bit-exactness gate: the direct wave dispatch with this many waves.
GATE_WAVES = 4


def setUpModule():
  hw_util.setup_module("tpu8i")


# -- the dispatch plan, on the host -------------------------------------------

_PLAN_SHAPES = (
    dict(testcase_name="ep4_w1", ep=4, t_local=16, g_local=4, topk=3, waves=1),
    dict(testcase_name="ep4_w2", ep=4, t_local=16, g_local=4, topk=3, waves=2),
    dict(testcase_name="ep4_w4", ep=4, t_local=16, g_local=4, topk=3, waves=4),
    dict(testcase_name="ep8_w4", ep=8, t_local=16, g_local=4, topk=3, waves=4),
    # More selections than shards: a token reaches at most ep shards.
    dict(
        testcase_name="ep2_topk4", ep=2, t_local=24, g_local=8, topk=4, waves=4
    ),
)


def _plan_cases():
  for shape, routing in itertools.product(
      _PLAN_SHAPES, ("uniform", "skewed", "worst", "dominant")
  ):
    case = dict(shape)
    case["testcase_name"] = f"{shape['testcase_name']}_{routing}"
    case.update(routing=routing)
    yield case


class WaveDispatchPlanTest(parameterized.TestCase):
  """The reference dispatch plan, replayed in numpy on the host."""

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_v8i(self)

  def test_experts_are_visited_fewest_rows_first_and_empties_last(self):
    hist = np.array([[5, 0, 2, 2], [0, 0, 0, 0], [1, 9, 3, 0]])
    with hw_util.host_only():
      pos, order = wave_dispatch.ascending_expert_order(
          jnp.asarray(hist.reshape(-1), jnp.int32), ep=3, g_local=4
      )
    # Ties break by index, as in expert_visit_list.
    want = np.array([[2, 3, 0, 1], [0, 1, 2, 3], [0, 2, 1, 3]])
    np.testing.assert_array_equal(np.asarray(order), want)
    np.testing.assert_array_equal(
        np.take_along_axis(np.asarray(pos), want, axis=1),
        np.broadcast_to(np.arange(4), (3, 4)),
    )

  @parameterized.named_parameters(*_plan_cases())
  def test_plan_delivers_every_needed_row_once_by_its_wave(
      self, routing, ep, t_local, g_local, topk, waves
  ):
    T = ep * t_local  # pylint: disable=invalid-name
    ids = _expert_ids(routing, T, ep * g_local, topk, g_local, seed=ep + waves)
    send_alloc = fused_host.align_up(t_local * min(topk, ep), fused_host.ROWBLK)
    with hw_util.host_only():
      hist = jnp.asarray(
          np.bincount(ids.reshape(-1), minlength=ep * g_local), jnp.int32
      )
      _, wave_of, _ = wave_dispatch.wave_plan(
          hist, ep=ep, g_local=g_local, waves=waves
      )
      wave_of = np.asarray(wave_of)
      plans = [
          tuple(
              np.asarray(a)
              for a in wave_dispatch.dispatch_plan(
                  jnp.asarray(ids),
                  jnp.int32(me),
                  ep=ep,
                  t_local=t_local,
                  g_local=g_local,
                  waves=waves,
                  send_alloc=send_alloc,
                  wave_of=jnp.asarray(wave_of),
              )
          )
          for me in range(ep)
      ]

    # Replay the exchange: every (destination, wave) segment of every sender.
    landed, landed_wave = self._replay(plans, waves, t_local, send_alloc)

    dest = ids // g_local
    for d, (_, tab, posmap) in enumerate(plans):
      # The kernel waits for exactly what the senders send, wave by wave.
      for w in range(waves):
        self.assertEqual(tab[3 * ep, w], sum(p[1][ep + d, w] for p in plans))
      needed = np.flatnonzero((dest == d).any(axis=1))
      # Every needed token exactly once, packed from row 0.
      np.testing.assert_array_equal(np.sort(landed[d, : needed.size]), needed)
      self.assertTrue((landed[d, needed.size :] == -1).all())
      for t in needed:
        row = posmap[t]
        self.assertEqual(landed[d, row], t)
        first = min(
            int(wave_of[d, e % g_local]) for e in ids[t] if e // g_local == d
        )
        self.assertEqual(landed_wave[d, row], first)

  def _replay(self, plans, waves, t_local, send_alloc):
    """Lands every sender's (destination, wave) segments; checks overlaps."""
    ep = len(plans)
    landed = np.full((ep, ep * t_local), -1)
    landed_wave = np.full((ep, ep * t_local), -1)
    for src, (perm, tab, _) in enumerate(plans):
      remote = 0
      for d in range(ep):
        for w in range(waves):
          off, n, dst = tab[d, w], tab[ep + d, w], tab[2 * ep + d, w]
          self.assertLessEqual(off + n, send_alloc)
          self.assertTrue(
              (landed[d, dst : dst + n] == -1).all(), "rows overlap"
          )
          landed[d, dst : dst + n] = src * t_local + perm[off : off + n]
          landed_wave[d, dst : dst + n] = w
          remote += n if d != src else 0
      self.assertEqual(tab[3 * ep + 1, 0], remote)
    return landed, landed_wave


# -- the layer -----------------------------------------------------------------


def _weights(mesh, hidden, inter):
  """fp8 (per output channel) and bf16 expert weights, sharded on experts."""
  e_total = len(mesh.devices.reshape(-1)) * EXPERTS_PER_SHARD
  rng = np.random.default_rng(0)
  w1 = rng.normal(0, hidden**-0.5, (e_total, hidden, 2 * inter)).astype(
      np.float32
  )
  w2 = rng.normal(0, inter**-0.5, (e_total, inter, hidden)).astype(np.float32)
  shard = NamedSharding(mesh, P(AXIS))

  def fp8(w):
    scale = np.maximum(np.abs(w).max(axis=1), 1e-12) / float(jnp.finfo(FP8).max)
    q = jnp.asarray(w / scale[:, None, :]).astype(FP8)
    return jax.device_put(q, shard), jax.device_put(scale, shard)

  (q1, s1), (q2, s2) = fp8(w1), fp8(w2)
  b1, b2 = (
      jax.device_put(jnp.asarray(w, jnp.bfloat16), shard) for w in (w1, w2)
  )
  return {
      WeightFormat.FP8: (q1, q2, s1, s2),
      WeightFormat.BF16: (b1, b2, None, None),
  }


def _inputs(mesh, x, gating):
  shard = NamedSharding(mesh, P(AXIS))
  return (
      jax.device_put(jnp.asarray(x, jnp.bfloat16), shard),
      jax.device_put(np.asarray(gating, np.float32), shard),
  )


def _layer(
    mesh,
    x,
    weights,
    gating,
    *,
    weight_format,
    waves,
    hash_topk_ids=None,
    **kwargs,
):
  """One layer call (direct wave dispatch), as f32 numpy.

  Args:
    mesh: The EP mesh.
    x: [T, hidden] token rows.
    weights: (w1, w2, w1_scale, w2_scale).
    gating: [T, E] router logits.
    weight_format: The WeightFormat.
    waves: W.
    hash_topk_ids: Optional [T, topk] forced expert ids.
    **kwargs: Static dedup_ep_moe arguments (closed over, never traced).

  Returns:
    [T, hidden] f32.
  """
  w1, w2, w1_scale, w2_scale = weights

  def fn(x, w1, w2, w1_scale, w2_scale, gating, hash_topk_ids):
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
        weight_format=weight_format,
        waves=waves,
        hash_topk_ids=hash_topk_ids,
        **kwargs,
    )

  with hw_util.cpu_layer():
    out = hw_util.run(fn, x, w1, w2, w1_scale, w2_scale, gating, hash_topk_ids)
    out = np.asarray(out, np.float32)
  assert out.shape == x.shape
  return out


def _assert_bit_exact_with_gate(
    mesh, x, weights, gating, *, weight_format, waves, **kwargs
):
  """`waves` waves give the same bits as the direct W=4 gate."""
  want = _layer(
      mesh,
      x,
      weights,
      gating,
      weight_format=weight_format,
      waves=GATE_WAVES,
      **kwargs,
  )
  got = _layer(
      mesh,
      x,
      weights,
      gating,
      weight_format=weight_format,
      waves=waves,
      **kwargs,
  )
  assert np.isfinite(want).all()
  np.testing.assert_array_equal(got, want)


# Wave counts that divide EXPERTS_PER_SHARD, other than the gate's.
_WAVES = (
    dict(testcase_name="w1", waves=1),
    dict(testcase_name="w2", waves=2),
)


def _bit_exact_cases():
  for rows, routing, waves in itertools.product(
      LAYER_ROWS, ("uniform", "skewed", "worst", "dominant"), _WAVES
  ):
    case = dict(waves)
    case["testcase_name"] = f"rows{rows}_{routing}_{waves['testcase_name']}"
    case.update(rows_per_shard=rows, routing=routing)
    yield case


class WaveDispatchLayerTest(parameterized.TestCase):
  """Wave counts against the W=4 gate: CPU interpret mode or TPU v8i."""

  # The mesh and weights, built by the first test that runs.
  _cache: dict[str, Any] = {}

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_v8i(self)
    if "mesh" not in self._cache:
      mesh = hw_util.make_mesh(_shapes().ep, self)
      self._cache["mesh"] = mesh
      self._cache["weights"] = _weights(mesh, _shapes().hidden, _shapes().inter)
    self.mesh = self._cache["mesh"]
    self.weights = self._cache["weights"]
    hw_util.reset_interpret_state()

  @parameterized.named_parameters(*_bit_exact_cases())
  def test_bit_exact_with_the_w4_gate(self, rows_per_shard, routing, waves):
    tokens = rows_per_shard * _shapes().ep
    rng = np.random.default_rng(rows_per_shard)
    x, gating = _inputs(
        self.mesh,
        rng.normal(size=(tokens, _shapes().hidden)),
        rng.normal(size=(tokens, _shapes().e_total)),
    )
    ids = _expert_ids(
        routing, tokens, _shapes().e_total, TOPK, EXPERTS_PER_SHARD, seed=1
    )
    _assert_bit_exact_with_gate(
        self.mesh,
        x,
        self.weights[WeightFormat.FP8],
        gating,
        weight_format=WeightFormat.FP8,
        waves=waves,
        hash_topk_ids=jax.device_put(ids, NamedSharding(self.mesh, P(AXIS))),
    )

  def test_unquantized_rows_are_bit_exact(self):
    """bf16 weights put bf16 rows, with no row scale, on the wire."""
    tokens = LAYER_ROWS[0] * _shapes().ep
    rng = np.random.default_rng(2)
    x, gating = _inputs(
        self.mesh,
        rng.normal(size=(tokens, _shapes().hidden)),
        rng.normal(size=(tokens, _shapes().e_total)),
    )
    _assert_bit_exact_with_gate(
        self.mesh,
        x,
        self.weights[WeightFormat.BF16],
        gating,
        weight_format=WeightFormat.BF16,
        waves=2,
    )

  def test_token_replica_groups_are_bit_exact(self):
    """Replicated rows: each group member routes and sends a disjoint half."""
    groups = tuple((d, d + 1) for d in range(0, _shapes().ep, 2))
    rows = 2 * LAYER_ROWS[0]
    rng = np.random.default_rng(3)
    x_g = rng.normal(size=(len(groups), rows, _shapes().hidden))
    gating_g = rng.normal(size=(len(groups), rows, _shapes().e_total))
    owner = {}
    for i, group in enumerate(groups):
      for d in group:
        owner[d] = i
    x, gating = _inputs(
        self.mesh,
        np.concatenate([x_g[owner[d]] for d in range(_shapes().ep)]),
        np.concatenate([gating_g[owner[d]] for d in range(_shapes().ep)]),
    )
    _assert_bit_exact_with_gate(
        self.mesh,
        x,
        self.weights[WeightFormat.FP8],
        gating,
        weight_format=WeightFormat.FP8,
        waves=2,
        token_replica_groups=groups,
    )


class WaveDispatchRejectionTest(absltest.TestCase):
  """Configurations the layer refuses before running any kernel."""

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_v8i(self)

  def _args(self, ep):
    mesh = Mesh(np.asarray(interpret_util.cpu_devices(ep)), axis_names=(AXIS,))
    weights = _weights(mesh, HIDDEN, INTER)
    tokens = 64 * ep
    rng = np.random.default_rng(4)
    x, gating = _inputs(
        mesh,
        rng.normal(size=(tokens, HIDDEN)),
        rng.normal(size=(tokens, ep * EXPERTS_PER_SHARD)),
    )
    common = dict(
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
    )
    return x, weights[WeightFormat.FP8], gating, common

  def test_rejects_configurations_it_does_not_serve(self):
    if len(jax.devices("cpu")) < CPU_DEVICES:
      self.skipTest(f"needs {CPU_DEVICES} CPU devices")
    x, (w1, w2, s1, s2), gating, common = self._args(CPU_DEVICES)
    for waves in (0, 3, 8):
      with self.assertRaisesRegex(ValueError, "divisor"):
        dedup_ep_moe(x, w1, w2, s1, s2, gating, waves=waves, **common)
    # The wave cuts ride in the dispatch table's padding rows: ep >= 3.
    x, (w1, w2, s1, s2), gating, common = self._args(2)
    with self.assertRaisesRegex(ValueError, "ep >= 3"):
      dedup_ep_moe(x, w1, w2, s1, s2, gating, waves=2, **common)


if __name__ == "__main__":
  jax.config.update("jax_num_cpu_devices", CPU_DEVICES)
  absltest.main()
