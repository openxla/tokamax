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
"""Bucket padding in layer/modular.dedup_moe must be invisible to real rows.

dedup_moe rounds t_local up to its t bucket and appends pad rows (zero
activations, logits that route each pad row to its own device's experts,
modular._pad_expert_ids). A real token's output must not depend on whether
the rest of its bucket is pad rows or other real tokens: every step of the
layer is per row (per-row fp8 activation scale, per-row intermediate and wire
scales, per-token top-k and combine), so the real rows of a padded run are
bit-identical to the same rows run inside a full bucket.

The test runs, for one routing draw, the padded layer and the full-bucket
layer under several policies (auto, the direct W=4 mode, the rules config
and, with a board topology, the rules config with De-Dup forced on) plus the
padding written out by hand, and checks every pairwise identity and the f32
reference. On failure the log table ("PADPROBE" lines) says which run is
off. CPU interpret mode by default; on TPU v8i the same checks run on the
host's devices (hw_util).
"""

import itertools

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib
from tokamax._src.ops.experimental.dedup_moe.layer import modular
from tokamax._src.ops.experimental.dedup_moe.test.common import hw_util
from tokamax._src.ops.experimental.dedup_moe.test.common import moe_inputs
from tokamax._src.ops.experimental.dedup_moe.test.common import reference

# CPU interpret mode: two boards of 4 devices, set in __main__ before the
# backend starts (on TPU this only sizes the unused CPU backend).
CPU_EP = 8

BoardTopology = topology_lib.BoardTopology
G_LOCAL = 4
TOPK = 4
# The CPU interpret shapes; setUpModule switches to hardware-legal ones on
# TPU (hw_util decides only after absl has parsed the flags).
HIDDEN = 256
INTER = 256
CAPACITY = 16
T_FULL = 16
REL_L2_BOUND = {"bf16": 2e-2, "fp8": 1e-1}
GATE = "direct_w4"
FORCED_DEDUP = modular.UnifiedRules(
    dedup_min_board_fanout=0.0, dedup_min_savings=0.0
)


def _padded_t_reals(t_reals: tuple[int, ...]) -> tuple[int, ...]:
  """Returns `t_reals`; each must pad up to the T_FULL bucket.

  A t_real that is itself a bucket pads nothing, so it would not test padding.

  Args:
    t_reals: The real per-device token counts of the cases.

  Raises:
    ValueError: If a t_real does not round up to T_FULL or is a bucket itself.
  """
  for t in t_reals:
    bucket = min(b for b in modular.DEFAULT_T_BUCKETS if b >= t)
    if t == bucket or bucket != T_FULL:
      raise ValueError(
          f"t_real={t} must pad up to T_FULL={T_FULL}; its bucket is {bucket}"
      )
  return t_reals


# Every case runs on hardware; CPU interpret mode runs only `fp8_t12`.
_HW_T_REALS = _padded_t_reals((12, 15))
_CASES = [
    dict(testcase_name=f"{fmt}_t{t}", fmt=fmt, t_real=t)
    for fmt, t in itertools.product(("fp8", "bf16"), _HW_T_REALS)
]
_CPU_CASE = ("fp8", 12)


def setUpModule():
  global HIDDEN, INTER, CAPACITY
  hw_util.setup_module("tpu8i")
  HIDDEN = hw_util.pick(HIDDEN, hw_util.HW_HIDDEN)
  INTER = hw_util.pick(INTER, hw_util.HW_INTER)
  CAPACITY = hw_util.check_capacity(hw_util.pick(CAPACITY, hw_util.HW_CAPACITY))


def _layout():
  """(ep, board size or None): the CPU layout, or what the TPU host holds."""
  if not hw_util.on_hardware():
    return CPU_EP, 4
  n = hw_util.num_devices()
  for ep, board_size in ((16, 8), (8, 4)):
    if n >= ep:
      return ep, board_size
  return n, None


def _maxdiff(a, b):
  a = np.asarray(a, np.float32)
  b = np.asarray(b, np.float32)
  return float(np.max(np.abs(a - b)))


class PaddingTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.ep, board_size = _layout()
    cls.topology = (
        None
        if board_size is None
        else BoardTopology.uniform(cls.ep, board_size)
    )
    cls.board = cls.ep if board_size is None else board_size

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_supported(self)
    hw_util.reset_interpret_state()
    self.mesh = hw_util.make_mesh(self.ep, self)

  def _cfg(self, fmt, policy, **kw):
    return modular.DedupMoeConfig(
        ep=self.ep,
        hidden=HIDDEN,
        num_experts=self.ep * G_LOCAL,
        top_k=TOPK,
        ffn=INTER,
        topology=self.topology,
        capacity=CAPACITY,
        weight_format=fmt,
        policy=policy,
        **kw,
    )

  def _run(self, cfg, weights, x, gating):
    prepared = modular.prepare_weights(self.mesh, *weights, cfg=cfg)
    with hw_util.cpu_layer():
      out = hw_util.run(
          lambda x, g, prep: modular.dedup_moe(x, g, prep, cfg),
          x,
          gating,
          prepared,
      )
      return np.asarray(out, np.float32)

  def _manual_pad(self, x_full, g_full, t_real):
    """x / gating at T_FULL with rows t_real.. replaced by dedup_moe's pads."""
    ep = self.ep
    x = np.asarray(x_full, np.float32).reshape(ep, T_FULL, -1).copy()
    g = np.asarray(g_full, np.float32).reshape(ep, T_FULL, -1).copy()
    n_pad = T_FULL - t_real
    for me in range(ep):
      ids = np.asarray(
          modular._pad_expert_ids(  # pylint: disable=protected-access
              jnp.int32(me), n_pad, g_local=G_LOCAL, board=self.board, topk=TOPK
          )
      )
      x[me, t_real:] = 0.0
      g[me, t_real:] = modular._PAD_LOGIT  # pylint: disable=protected-access
      g[me, np.arange(t_real, T_FULL)[:, None], ids] = 0.0
    x = jnp.asarray(x.reshape(ep * T_FULL, -1), jnp.bfloat16)
    g = jnp.asarray(g.reshape(ep * T_FULL, -1), jnp.float32)
    return (
        moe_inputs.shard_rows(self.mesh, x),
        moe_inputs.shard_rows(self.mesh, g),
    )

  @parameterized.named_parameters(*_CASES)
  def test_padding_rows_are_invisible(self, fmt, t_real):
    if not hw_util.on_hardware() and (fmt, t_real) != _CPU_CASE:
      self.skipTest("hardware-only case (CPU interpret mode runs fp8_t12)")
    ep = self.ep
    weights = moe_inputs.random_weights(
        self.mesh,
        e_total=ep * G_LOCAL,
        hidden=HIDDEN,
        inter=INTER,
        weight_format=fmt,
        seed=0,
    )
    x_full, g_full = moe_inputs.random_inputs(
        self.mesh,
        tokens=ep * T_FULL,
        hidden=HIDDEN,
        e_total=ep * G_LOCAL,
        seed=1,
    )
    keep = np.arange(ep * T_FULL).reshape(ep, T_FULL)[:, :t_real].reshape(-1)
    x = moe_inputs.shard_rows(self.mesh, np.asarray(x_full)[keep])
    g = moe_inputs.shard_rows(self.mesh, np.asarray(g_full)[keep])
    x_man, g_man = self._manual_pad(x_full, g_full, t_real)

    cfgs = {
        "auto": self._cfg(fmt, "auto"),
        "gate": self._cfg(fmt, GATE),
        "rules": self._cfg(fmt, "rules"),
    }
    if self.topology is not None:
      cfgs["dedup"] = self._cfg(fmt, "rules", rules=FORCED_DEDUP)
    for name, cfg in cfgs.items():
      p = modular.plan(cfg, t_real)
      self.assertEqual(p.t_bucket, T_FULL)
      logging.info(
          "PADPROBE plan %s %s: mode=%s kwargs=%s",
          fmt,
          name,
          p.mode,
          dict(p.kwargs),
      )
    if "dedup" in cfgs:
      p = modular.plan(cfgs["dedup"], t_real)
      self.assertIsNotNone(p.kwargs["dedup_topology"])
    runs = {}
    for name, cfg in cfgs.items():
      runs[f"full_{name}"] = self._run(cfg, weights, x_full, g_full)[keep]
      runs[f"pad_{name}"] = self._run(cfg, weights, x, g)
    runs["pad_auto_again"] = self._run(cfgs["auto"], weights, x, g)
    runs["manual_auto"] = self._run(cfgs["auto"], weights, x_man, g_man)[keep]
    runs["manual_gate"] = self._run(cfgs["gate"], weights, x_man, g_man)[keep]

    w1, w2, w1s, w2s = weights
    ref_full = np.asarray(
        reference.ep_reference(
            x_full, w1, w2, w1s, w2s, g_full, topk=TOPK, mesh=self.mesh
        )
    )[keep]
    ref_pad = np.asarray(
        reference.ep_reference(
            x, w1, w2, w1s, w2s, g, topk=TOPK, mesh=self.mesh
        )
    )
    np.testing.assert_allclose(ref_pad, ref_full, rtol=1e-4, atol=1e-5)
    for name, out in runs.items():
      logging.info(
          "PADPROBE %s t=%d %-16s rel-L2 vs f32 %.5f",
          fmt,
          t_real,
          name,
          reference.rel_l2(out, ref_pad),
      )
    pairs = [
        ("pad_auto", "pad_auto_again"),
        ("pad_auto", "full_auto"),
        ("pad_gate", "full_gate"),
        ("pad_rules", "full_rules"),
        ("pad_auto", "pad_gate"),
        ("full_auto", "full_gate"),
        ("pad_rules", "pad_gate"),
        ("manual_auto", "pad_auto"),
        ("manual_gate", "pad_gate"),
        ("manual_gate", "full_gate"),
    ]
    if "dedup" in cfgs:
      pairs += [("pad_dedup", "full_dedup"), ("pad_dedup", "pad_gate")]
    for a, b in pairs:
      logging.info(
          "PADPROBE %s t=%d %s vs %s: equal=%s maxdiff=%.4g",
          fmt,
          t_real,
          a,
          b,
          np.array_equal(runs[a], runs[b]),
          _maxdiff(runs[a], runs[b]),
      )
    for name, out in runs.items():
      with self.subTest(close=name):
        self.assertLess(reference.rel_l2(out, ref_pad), REL_L2_BOUND[fmt])
    for a, b in pairs:
      with self.subTest(pair=f"{a}=={b}"):
        np.testing.assert_array_equal(runs[a], runs[b])


def _jaxprs(jaxpr):
  """`jaxpr` and every jaxpr nested in its equations' params."""
  yield jaxpr
  for eqn in jaxpr.eqns:
    for v in eqn.params.values():
      for sub in v if isinstance(v, (list, tuple)) else (v,):
        inner = getattr(sub, "jaxpr", sub)
        if hasattr(inner, "eqns"):
          yield from _jaxprs(inner)


class RowquantBarrierTest(parameterized.TestCase):
  """Guard for a TPU-only padding fix (CPU has no such fusion to observe).

  On TPU, XLA could fuse dedup_moe's pad-row concatenation (same program)
  into the layer's fp8 row quantization and change its bf16 rounding, so
  padded real rows differed from the same rows in a full bucket.
  monolithic.dedup_ep_moe passes every quantized input through an
  optimization barrier before the row quantization; this checks the traced
  program keeps it (and that unquantized bf16 inputs need none).
  """

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_supported(self)

  @parameterized.parameters(("fp8", True), ("bf16", False))
  def test_quantized_rows_pass_an_optimization_barrier(self, fmt, want):
    ep, board_size = _layout()
    mesh = hw_util.make_mesh(ep, self)
    cfg = modular.DedupMoeConfig(
        ep=ep,
        hidden=HIDDEN,
        num_experts=ep * G_LOCAL,
        top_k=TOPK,
        ffn=INTER,
        topology=(
            None
            if board_size is None
            else BoardTopology.uniform(ep, board_size)
        ),
        capacity=CAPACITY,
        weight_format=fmt,
        policy=GATE,
    )
    weights = moe_inputs.random_weights(
        mesh,
        e_total=ep * G_LOCAL,
        hidden=HIDDEN,
        inter=INTER,
        weight_format=fmt,
        seed=0,
    )
    prepared = modular.prepare_weights(mesh, *weights, cfg=cfg)
    t_real = 12
    x, g = moe_inputs.random_inputs(
        mesh, tokens=ep * t_real, hidden=HIDDEN, e_total=ep * G_LOCAL, seed=1
    )
    t_bucket = modular.plan(cfg, t_real).t_bucket
    with hw_util.cpu_layer():
      closed = jax.make_jaxpr(
          lambda x, g, prep: modular.dedup_moe(x, g, prep, cfg)
      )(x, g, prepared)
    # The layer's shard_map body: its first input is the token block x_l.
    found = False
    for jp in _jaxprs(closed.jaxpr):
      if not jp.invars or len(jp.invars) < 4:
        continue
      x_l = jp.invars[0]
      if getattr(x_l.aval, "shape", None) != (t_bucket, HIDDEN):
        continue
      for eqn in jp.eqns:
        if eqn.primitive.name == "optimization_barrier" and any(
            v is x_l for v in eqn.invars
        ):
          found = True
    self.assertEqual(found, want)


if __name__ == "__main__":
  jax.config.update("jax_num_cpu_devices", CPU_EP)
  absltest.main()
