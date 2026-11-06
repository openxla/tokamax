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
"""Correctness tests of the De-Dup MoE layer, adapted from routing_moe.

Tokamax routing_moe checks its kernels with layer-vs-reference tests
(renormalize, activations, biases, sparse experts), isolated FFN tests (idle
shard) and combine tests (NaN-poisoned top-k reduce, K=1, f32 output,
all-zero weights, per-shard sums). This file checks the same things here:

  LayerCorrectnessTest    The model-facing layer (layer/modular.dedup_moe)
                          for the `auto` and `rules` policies, the direct
                          W=4 mode and (with a board topology) the rules
                          config with De-Dup forced on, against the f32
                          reference (reference.ep_reference, rel-L2: fp8 <
                          0.1, bf16 < 0.02), every policy bit exact with the
                          direct mode; over t_local (padding to t buckets
                          included), fp8 / bf16 and board sizes 4 and 8.
                          Sparse experts (by router and by ids), an idle
                          device and an idle board through the direct mode
                          and a forced De-Dup config.
  MonolithicOptionsTest   monolithic.dedup_ep_moe with renormalize=False,
                          swigluoai and silu_and_mul_with_clamp, w1 / w2
                          biases, sigmoid and sqrtsoftplus scoring,
                          routed_scaling_factor and score_bias, against the
                          extended f32 reference.
  CombineTest             io_relayout/output_reduce.combine_arrivals (the
                          Mosaic kernel where t_local % 8 == 0, else the XLA
                          epilogue) with every unreferenced arrival row and
                          scale NaN-poisoned, K = 1..10, bf16 and f32
                          output, fp8 and bf16 wire, all-zero weights and
                          per-shard sums vs dense.
  ExpertFfnTest           The isolated expert FFN (fused_moe.kernel's
                          expert_ffn_fp8 / expert_ffn_bf16 plus the tile
                          epilogue wire_row_and_scale) against the f32
                          per-expert reference.

Modes (hw_util): by default the kernels run on CPU devices in Pallas TPU
interpret mode at small shapes (16 devices as two boards of 8 or four boards
of 4, D = F = 256, E = 64, K = 4); on TPU v8i they run on the host's devices
at large shapes (D = 8192, F = 2048, K = 10, 32 experts per device; on TPU v7x
D = 4096, F = 1024, which fits its VMEM) and the full t_local sweep, with
De-Dup only where the host holds two boards.
"""

import itertools
from typing import Any

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib
from tokamax._src.ops.experimental.dedup_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.dedup_moe.io_relayout import output_reduce
from tokamax._src.ops.experimental.dedup_moe.layer import modular
from tokamax._src.ops.experimental.dedup_moe.layer import monolithic
from tokamax._src.ops.experimental.dedup_moe.test.common import hw_util
from tokamax._src.ops.experimental.dedup_moe.test.common import moe_inputs
from tokamax._src.ops.experimental.dedup_moe.test.common import reference
from tokamax._src.ops.experimental.fused_moe import kernel as fused_kernel

# CPU interpret mode: 16 devices, set in __main__ before the backend starts (on
# TPU this only sizes the unused CPU backend).
CPU_EP = 16

BoardTopology = topology_lib.BoardTopology
FP8 = arg.FP8
HIDDEN_LANE_BLOCK = output_reduce.HIDDEN_LANE_BLOCK

# Shapes and sweeps per mode. The mode is only known once absl has parsed the
# flags (hw_util), so the parameter lists below carry the cases of both modes
# (a layout or `cpu_` / `hw_` prefix) and a test skips the other mode's cases;
# setUpModule sets the shape constants for the mode that runs.
# Two boards of 8 devices and four boards of 4.
CPU_LAYOUTS = ("bs8", "bs4")
# The layout the TPU host holds (_hw_layout).
HW_LAYOUTS = ("host",)
# CPU: 4 pads to the 8 bucket, 12 to 16 (not buckets).
T_LOCALS = {"cpu": (4, 12), "hw": (1, 4, 16, 64, 128, 256, 512, 100)}
# CPU: 16 / 48 run the Mosaic combine kernel (single parity); 256 the
# pipelined kernel; 20 (t_local % 8 != 0) the XLA epilogue.
COMBINE_T_K = {
    "cpu": ((16, 1), (48, 4), (20, 2), (256, 10)),
    "hw": tuple(itertools.product((48, 128, 300, 512), (1, 2, 4, 10))),
}

# The CPU values; setUpModule switches them to the hardware ones on TPU.
ON_HW = False
HIDDEN, INTER, TOPK = 256, 256, 4
G_LOCAL = 4
CAPACITY = 16
EDGE_T = 16
MONO_T = 16
COMBINE_HIDDEN = 256
# CPU: two 512-column chunks of the intermediate.
FFN_HIDDEN, FFN_INTER = 512, 1024
# The hardware expert shape on TPU v7x, whose VMEM cannot hold one 8192x2048
# bf16 expert (setUpModule).
V7X_HIDDEN, V7X_INTER = 4096, 1024
# The monolithic option matrix: one board on CPU.
MONO_EP_CPU = 8
FMTS = ("fp8", "bf16")
GATE_MODE = "direct_w4"
GATE_WAVES = 4
# rel-L2 bound against the f32 reference. bf16 rounds the intermediate and
# the output; fp8 also quantizes the weights per channel, the rows, the
# intermediate and the wire.
REL_L2_BOUND = {"bf16": 2e-2, "fp8": 1e-1}
# De-Dup on whatever the routing: the small CPU shapes are below the rules'
# thresholds.
FORCED_DEDUP = modular.UnifiedRules(
    dedup_min_board_fanout=0.0, dedup_min_savings=0.0
)


def setUpModule():
  global ON_HW, HIDDEN, INTER, TOPK, G_LOCAL, CAPACITY, EDGE_T, MONO_T
  global COMBINE_HIDDEN, FFN_HIDDEN, FFN_INTER
  hw_util.setup_module("tpu8i")
  ON_HW = hw_util.on_hardware()
  if ON_HW:
    HIDDEN, INTER, TOPK = 8192, 2048, 10
    if not hw_util.on_v8i():
      # One bf16 expert of 8192x2048 weights alone (96MiB) exceeds TPU v7x's
      # VMEM; keep the large-shape cases at an expert size that fits.
      HIDDEN, INTER = V7X_HIDDEN, V7X_INTER
    G_LOCAL = 32
    CAPACITY = 128
    EDGE_T = 128
    MONO_T = 64
    COMBINE_HIDDEN = HIDDEN
    FFN_HIDDEN, FFN_INTER = HIDDEN, INTER


def _skip_other_mode(test, mode: str) -> None:
  """Skips a case of the other mode (`cpu` or `hw`)."""
  if (mode == "hw") != ON_HW:
    test.skipTest(f"{mode} case; running in {hw_util.mode_name()} mode")


def _hw_layout() -> tuple[int, int | None]:
  """(ep, board size or None) on the TPU host: all devices, up to 64."""
  n = min(hw_util.num_devices(), 64)
  if n >= 16:
    return n - n % 8, 8
  if n >= 8:
    return n - n % 4, 4
  return n, None


def _layout(name: str) -> tuple[int, int | None]:
  if name == "host":
    return _hw_layout()
  return CPU_EP, {"bs8": 8, "bs4": 4}[name]


def _layout_mode(name: str) -> str:
  return "hw" if name in HW_LAYOUTS else "cpu"


def _assert_close(test, got, want, fmt, what):
  """routing_moe's numerics check: finite, then rel-L2 vs the f32 oracle."""
  got = np.asarray(got, np.float32)
  want = np.asarray(want, np.float32)
  test.assertEqual(got.shape, want.shape, what)
  test.assertFalse(np.isnan(got).any(), f"{what}: NaN in the output")
  test.assertTrue(np.isfinite(got).all(), f"{what}: non-finite output")
  err = reference.rel_l2(got, want)
  logging.info(
      "[%s] %s %s: rel-L2 %.4g, max-abs-rel %.4g vs f32 reference",
      hw_util.mode_name(),
      what,
      fmt,
      err,
      reference.max_abs_rel(got, want),
  )
  test.assertLess(err, REL_L2_BOUND[fmt], what)


# ------------------------------------------------------------------------------
# The model-facing layer.
# ------------------------------------------------------------------------------


def _layer_cases():
  for layout in CPU_LAYOUTS + HW_LAYOUTS:
    for t, fmt in itertools.product(T_LOCALS[_layout_mode(layout)], FMTS):
      yield dict(
          testcase_name=f"{layout}_t{t}_{fmt}",
          layout=layout,
          t_local=t,
          fmt=fmt,
      )


def _edge_cases():
  routings = (
      # routing_moe gen_moe_inputs: gating U[0, 1), 40% of experts at 0.
      dict(name="sparse_router", routing="router", ratio=0.6),
      # The same experts forced through caller ids.
      dict(name="sparse_ids", routing="sparse:0.6"),
      # gmm_test skip_last_shard: device ep - 1 receives no rows.
      dict(name="idle_device", routing="idle_device"),
      # The last 8 devices (one board of 8 or two of 4) receive no rows.
      dict(name="idle_board", routing="idle_board"),
  )
  for layout, r in itertools.product(CPU_LAYOUTS + HW_LAYOUTS, routings):
    r = dict(r)
    yield dict(testcase_name=f"{layout}_{r.pop('name')}", layout=layout, **r)


class LayerCorrectnessTest(parameterized.TestCase):
  """modular.dedup_moe vs the f32 reference, policies vs the direct mode."""

  _weights = {}

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_supported(self)
    hw_util.reset_interpret_state()

  def _setup(self, layout):
    _skip_other_mode(self, _layout_mode(layout))
    self.ep, board_size = _layout(layout)
    self.mesh = hw_util.make_mesh(self.ep, self)
    self.topology = (
        None
        if board_size is None
        else BoardTopology.uniform(self.ep, board_size)
    )
    self.e_total = self.ep * G_LOCAL

  def weights(self, fmt):
    key = (self.ep, fmt)
    if key not in self._weights:
      self._weights[key] = moe_inputs.random_weights(
          self.mesh,
          e_total=self.e_total,
          hidden=HIDDEN,
          inter=INTER,
          weight_format=fmt,
          seed=0,
      )
    return self._weights[key]

  def _cfg(self, fmt, policy, **kw):
    return modular.DedupMoeConfig(
        ep=self.ep,
        hidden=HIDDEN,
        num_experts=self.e_total,
        top_k=TOPK,
        ffn=INTER,
        topology=self.topology,
        capacity=CAPACITY,
        weight_format=fmt,
        policy=policy,
        **kw,
    )

  def _policies(self):
    """(name, cfg kwargs) of the configs checked against the direct mode."""
    policies: list[tuple[str, dict[str, Any]]] = [
        ("auto", dict(policy="auto")),
        ("rules", dict(policy="rules")),
    ]
    if self.topology is not None:
      policies.append(("dedup", dict(policy="rules", rules=FORCED_DEDUP)))
    return policies

  def _inputs(self, t_local, seed=1, ratio=1.0):
    return moe_inputs.random_inputs(
        self.mesh,
        tokens=self.ep * t_local,
        hidden=HIDDEN,
        e_total=self.e_total,
        seed=seed,
        active_expert_ratio=ratio,
    )

  def _ids(self, routing, t_local, seed=3):
    if routing == "router":
      return None
    ids = moe_inputs.expert_ids(
        routing, self.ep * t_local, self.e_total, TOPK, G_LOCAL, seed=seed
    )
    return moe_inputs.shard_rows(self.mesh, ids)

  def _layer(self, cfg, x, gating, ids=None):
    """One dedup_moe call (jitted on TPU)."""
    t_local = x.shape[0] // cfg.ep
    p = modular.plan(cfg, t_local)
    what = f"policy={cfg.policy} mode={p.mode} t={t_local}"
    if cfg.rules is FORCED_DEDUP:
      self.assertIsNotNone(p.kwargs["dedup_topology"], what)
    prepared = modular.prepare_weights(
        self.mesh, *self.weights(cfg.weight_format), cfg=cfg
    )
    with hw_util.cpu_layer():
      out = hw_util.run(
          lambda x, g, prep, ids: modular.dedup_moe(
              x, g, prep, cfg, topk_ids=ids
          ),
          x,
          gating,
          prepared,
          ids,
      )
      out = np.asarray(out, np.float32)
    logging.info("[%s] ran %s", hw_util.mode_name(), what)
    self.assertEqual(out.shape, x.shape)
    return out

  def _reference(self, fmt, x, gating, ids=None):
    w1, w2, w1s, w2s = self.weights(fmt)
    return reference.ep_reference(
        x,
        w1,
        w2,
        w1s,
        w2s,
        gating,
        topk=TOPK,
        mesh=self.mesh,
        hash_topk_ids=ids,
    )

  @parameterized.named_parameters(*_layer_cases())
  def test_policies_match_reference_and_direct(self, layout, t_local, fmt):
    self._setup(layout)
    x, gating = self._inputs(t_local, seed=t_local + 1)
    want = self._reference(fmt, x, gating)
    gate = self._layer(self._cfg(fmt, GATE_MODE), x, gating)
    _assert_close(self, gate, want, fmt, f"direct t={t_local}")
    for name, kw in self._policies():
      with self.subTest(policy=name):
        got = self._layer(self._cfg(fmt, **kw), x, gating)
        np.testing.assert_array_equal(got, gate, err_msg=name)
        _assert_close(self, got, want, fmt, f"{name} t={t_local}")

  @parameterized.named_parameters(*_edge_cases())
  def test_sparse_and_idle_experts(self, layout, routing, ratio=1.0):
    self._setup(layout)
    if routing == "idle_board" and self.ep < 16:
      self.skipTest(f"idle_board needs two boards of 8 devices; ep={self.ep}")
    fmt = "fp8"
    t_local = EDGE_T
    x, gating = self._inputs(t_local, seed=11, ratio=ratio)
    ids = self._ids(routing, t_local)
    if ids is not None:
      # The routing really leaves the experts / devices it names empty.
      used = np.zeros(self.e_total, bool)
      used[np.asarray(ids).reshape(-1)] = True
      pool = moe_inputs.expert_pool(routing, self.e_total, G_LOCAL, seed=3)
      self.assertFalse(used[np.setdiff1d(np.arange(self.e_total), pool)].any())
    want = self._reference(fmt, x, gating, ids)
    gate = self._layer(self._cfg(fmt, GATE_MODE), x, gating, ids)
    _assert_close(self, gate, want, fmt, f"direct {routing}")
    policies = [p for p in self._policies() if p[0] == "dedup"]
    if ON_HW:
      policies.append(("auto", dict(policy="auto")))
    for name, kw in policies:
      with self.subTest(policy=name):
        got = self._layer(self._cfg(fmt, **kw), x, gating, ids)
        np.testing.assert_array_equal(got, gate, err_msg=name)
        _assert_close(self, got, want, fmt, f"{name} {routing}")


# ------------------------------------------------------------------------------
# The monolithic layer's option matrix.
# ------------------------------------------------------------------------------

_OPTION_CASES = (
    dict(testcase_name="renormalize_false", renormalize=False),
    dict(testcase_name="swigluoai", act_fn="swigluoai", x_scale=4.0),
    dict(
        testcase_name="silu_and_mul_with_clamp",
        act_fn="silu_and_mul_with_clamp",
        x_scale=6.0,
    ),
    dict(testcase_name="bias", bias=True),
    dict(testcase_name="bias_bf16", bias=True, fmt="bf16"),
    dict(
        testcase_name="swigluoai_bias_bf16",
        act_fn="swigluoai",
        bias=True,
        fmt="bf16",
        x_scale=4.0,
    ),
    dict(testcase_name="sigmoid", scoring_fn="sigmoid"),
    dict(testcase_name="sqrtsoftplus", scoring_fn="sqrtsoftplus"),
    dict(testcase_name="routed_scaling_factor", routed_scaling_factor=2.5),
    dict(
        testcase_name="score_bias_sigmoid",
        scoring_fn="sigmoid",
        score_bias=True,
    ),
)


class MonolithicOptionsTest(parameterized.TestCase):
  """dedup_ep_moe options vs the extended f32 reference."""

  _weights = {}

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_supported(self)
    hw_util.reset_interpret_state()
    self.ep = _hw_layout()[0] if ON_HW else MONO_EP_CPU
    self.mesh = hw_util.make_mesh(self.ep, self)
    self.e_total = self.ep * G_LOCAL

  def weights(self, fmt):
    key = (self.ep, fmt)
    if key not in self._weights:
      self._weights[key] = moe_inputs.random_weights(
          self.mesh,
          e_total=self.e_total,
          hidden=HIDDEN,
          inter=INTER,
          weight_format=fmt,
          seed=0,
      )
    return self._weights[key]

  @parameterized.named_parameters(*_OPTION_CASES)
  def test_option_matches_reference(
      self,
      fmt="fp8",
      renormalize=True,
      act_fn="silu",
      bias=False,
      scoring_fn="softmax",
      routed_scaling_factor=1.0,
      score_bias=False,
      x_scale=1.0,
  ):
    x, gating = moe_inputs.random_inputs(
        self.mesh,
        tokens=self.ep * MONO_T,
        hidden=HIDDEN,
        e_total=self.e_total,
        seed=31,
        x_scale=x_scale,
    )
    w1b = w2b = sb = None
    if bias:
      w1b, w2b = moe_inputs.random_biases(
          self.mesh, e_total=self.e_total, hidden=HIDDEN, inter=INTER, seed=5
      )
    if score_bias:
      sb = jax.random.normal(jax.random.key(7), (self.e_total,), jnp.float32)
      sb = sb * 0.1
    opts = dict(
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        scoring_fn=scoring_fn,
    )
    weights = self.weights(fmt)
    with hw_util.cpu_layer():
      out = hw_util.run(
          lambda x, g, w, b1, b2, sb_: monolithic.dedup_ep_moe(
              x,
              w[0],
              w[1],
              w[2],
              w[3],
              g,
              b1,
              b2,
              topk=TOPK,
              mesh=self.mesh,
              capacity=CAPACITY,
              weight_format=arg.WeightFormat(fmt),
              act_fn=act_fn,
              waves=GATE_WAVES,
              score_bias=sb_,
              **opts,
          ),
          x,
          gating,
          weights,
          w1b,
          w2b,
          sb,
      )
      got = np.asarray(out, np.float32)
    w1, w2, w1s, w2s = weights
    want = reference.ep_reference(
        x,
        w1,
        w2,
        w1s,
        w2s,
        gating,
        topk=TOPK,
        mesh=self.mesh,
        act_fn=act_fn,
        w1_bias=w1b,
        w2_bias=w2b,
        score_bias=sb,
        **opts,
    )
    _assert_close(self, got, want, fmt, self._testMethodName)


# ------------------------------------------------------------------------------
# The combine.
# ------------------------------------------------------------------------------


def ref_topk_reduce(rows2d, weights, out_dtype=jnp.bfloat16):
  """routing_moe's top-k reduce reference: rows2d[T * K, H] (token, pick)."""
  num_tokens, topk = weights.shape
  w = weights.astype(jnp.float32).reshape(-1, 1)
  x = jnp.where(w != 0, rows2d.astype(jnp.float32), 0.0) * w
  return x.reshape(num_tokens, topk, -1).sum(axis=1).astype(out_dtype)


# The mirror's lane count (a pow2 tile height) in the synthetic arrivals.
_MIRROR_TILE = 16
# Unreferenced arrival rows (NaN-poisoned) beyond the referenced ones.
_SPARE_ROWS = 24


def _combine_inputs(t_local, topk, hidden, wire, *, seed, num_experts=64):
  """Synthetic arrivals for one shard's combine.

  Args:
    t_local: Tokens.
    topk: Selections per token.
    hidden: Row width.
    wire: `fp8` or `bf16` wire rows.
    seed: PRNG seed.
    num_experts: Experts the synthetic router picks from.

  Returns:
    dict with arrivals [R, hidden / 128, 128] in the wire dtype, scales
    [mirror_sublanes, tile] f32 (None on a bf16 wire), pos / mirror_pos
    [t, K] i32, weights [t, K] f32, ids [t, K] i32, the clean dequantized
    rows [t * K, hidden] f32 in (token, pick) order, and the referenced-row
    mask [R].
  """
  rng = np.random.default_rng(seed)
  k1, k2 = jax.random.split(jax.random.key(seed))
  gating = jax.random.uniform(k1, (t_local, num_experts), jnp.float32)
  w, ids = jax.lax.top_k(jax.nn.softmax(gating, axis=-1), topk)
  w = w / w.sum(axis=-1, keepdims=True)
  m = t_local * topk
  recv_rows = m + _SPARE_ROWS
  pos = rng.permutation(recv_rows)[:m].reshape(t_local, topk).astype(np.int32)
  referenced = np.zeros(recv_rows, bool)
  referenced[pos.reshape(-1)] = True
  rows = np.asarray(
      jax.random.normal(k2, (recv_rows, hidden), jnp.float32) / 10
  )
  lanes = HIDDEN_LANE_BLOCK
  sublanes = -(-recv_rows // _MIRROR_TILE)
  if wire == "fp8":
    q, s = input_pack.rowquant_fp8(jnp.asarray(rows, jnp.bfloat16))
    arrivals = np.asarray(q)
    flat_scale = np.zeros(sublanes * _MIRROR_TILE, np.float32)
    flat_scale[:recv_rows] = np.asarray(s)[:, 0]
    clean = arrivals.astype(np.float32) * flat_scale[:recv_rows, None]
    # Scales of unreferenced rows (and the mirror's tail) are poisoned too.
    live = np.zeros(flat_scale.size, bool)
    live[:recv_rows] = referenced
    scales = np.where(live, flat_scale, np.nan).astype(np.float32)
    scales = scales.reshape(sublanes, _MIRROR_TILE)
    clean_scales = flat_scale.reshape(sublanes, _MIRROR_TILE)
    nan = np.asarray(jnp.asarray(np.nan, FP8))
  else:
    arrivals = np.asarray(jnp.asarray(rows, jnp.bfloat16))
    clean = arrivals.astype(np.float32)
    scales = clean_scales = None
    nan = np.asarray(jnp.asarray(np.nan, jnp.bfloat16))
  poisoned = np.where(referenced[:, None], arrivals, nan)
  shape = (recv_rows, hidden // lanes, lanes)
  return dict(
      arrivals=jnp.asarray(poisoned.reshape(shape)),
      clean_arrivals=jnp.asarray(arrivals.reshape(shape)),
      scales=None if scales is None else jnp.asarray(scales),
      clean_scales=None if clean_scales is None else jnp.asarray(clean_scales),
      pos=jnp.asarray(pos),
      # The mirror holds one scale per arrival row at the row's own index.
      mirror_pos=jnp.asarray(pos),
      weights=w.astype(jnp.float32),
      ids=ids.astype(jnp.int32),
      rows2d=jnp.asarray(clean[pos.reshape(-1)]),
      referenced=referenced,
  )


def _combine(inp, weights, out_dtype, *, clean=False):
  key = "clean_" if clean else ""
  with hw_util.interpret_pallas():
    out = output_reduce.combine_arrivals(
        inp[key + "arrivals"],
        inp[key + "scales"],
        inp["pos"],
        inp["mirror_pos"],
        weights,
        out_dtype,
    )
    return jax.block_until_ready(out)


def _combine_cases():
  for mode in ("cpu", "hw"):
    for (t, k), wire, out in itertools.product(
        COMBINE_T_K[mode], ("fp8", "bf16"), ("bfloat16", "float32")
    ):
      yield dict(
          testcase_name=f"{mode}_t{t}_k{k}_{wire}_{out}",
          mode=mode,
          t_local=t,
          topk=k,
          wire=wire,
          out_dtype=out,
      )


class CombineTest(parameterized.TestCase):
  """The arrival combine (routing_moe's output reduce tests, adapted)."""

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_supported(self)
    hw_util.reset_interpret_state()

  @parameterized.named_parameters(*_combine_cases())
  def test_nan_poisoned_unused_rows(self, mode, t_local, topk, wire, out_dtype):
    _skip_other_mode(self, mode)
    out_dtype = jnp.dtype(out_dtype)
    inp = _combine_inputs(t_local, topk, COMBINE_HIDDEN, wire, seed=42)
    path = "kernel" if output_reduce.combine_step_tokens(t_local) else "xla"
    expected = ref_topk_reduce(inp["rows2d"], inp["weights"], out_dtype)
    actual = _combine(inp, inp["weights"], out_dtype)
    self.assertEqual(actual.shape, (t_local, COMBINE_HIDDEN))
    self.assertEqual(actual.dtype, out_dtype)
    self.assertFalse(bool(jnp.isnan(actual).any()), f"NaN read ({path})")
    rtol = 2e-2
    atol = rtol * max(float(jnp.abs(expected).max()), 1e-3)
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        atol=atol,
        rtol=rtol,
        err_msg=path,
    )

  @parameterized.parameters("fp8", "bf16")
  def test_all_zero_weights_give_zeros(self, wire):
    t_local, topk = (256, 10) if ON_HW else (48, 4)
    inp = _combine_inputs(t_local, topk, COMBINE_HIDDEN, wire, seed=7)
    zeros = jnp.zeros((t_local, topk), jnp.float32)
    out = _combine(inp, zeros, jnp.bfloat16, clean=True)
    np.testing.assert_array_equal(
        np.asarray(out, np.float32), np.zeros((t_local, COMBINE_HIDDEN))
    )

  def test_shards_sum_to_dense(self):
    """Per-shard partial combines sum to the dense combine.

    Each EP shard's partial combine keeps only the picks whose expert it
    owns; the partials sum to the dense combine.
    """
    t_local, topk, ep, e_total = (256, 10, 8, 512) if ON_HW else (48, 4, 8, 64)
    local = e_total // ep
    inp = _combine_inputs(
        t_local, topk, COMBINE_HIDDEN, "fp8", seed=99, num_experts=e_total
    )
    shard_sum = jnp.zeros((t_local, COMBINE_HIDDEN), jnp.float32)
    for shard in range(ep):
      own = (inp["ids"] >= shard * local) & (inp["ids"] < (shard + 1) * local)
      w_shard = jnp.where(own, inp["weights"], 0.0)
      out = _combine(inp, w_shard, jnp.float32)
      self.assertFalse(bool(jnp.isnan(out).any()))
      shard_sum = shard_sum + out
    dense = ref_topk_reduce(inp["rows2d"], inp["weights"], jnp.float32)
    np.testing.assert_allclose(
        np.asarray(shard_sum), np.asarray(dense), rtol=2e-2, atol=2e-2
    )


# ------------------------------------------------------------------------------
# The isolated expert FFN.
# ------------------------------------------------------------------------------


def _ffn_cases():
  for fmt, act, bias, m in itertools.product(
      FMTS, reference.ACT_FNS, (False, True), (16, 128)
  ):
    yield dict(
        testcase_name=f"{fmt}_{act}_{'bias' if bias else 'nobias'}_m{m}",
        fmt=fmt,
        act_fn=act,
        bias=bias,
        m=m,
    )


def _expert_operands(fmt, m, act_fn, bias, seed):
  """One expert's rows, weights (+ scales) and biases, as the kernel takes them."""
  ks = jax.random.split(jax.random.key(seed), 5)
  x_scale = 4.0 if act_fn != "silu" else 1.0
  x = (jax.random.normal(ks[0], (m, FFN_HIDDEN), jnp.float32) * x_scale).astype(
      jnp.bfloat16
  )
  w1 = jax.random.normal(ks[1], (FFN_HIDDEN, 2 * FFN_INTER), jnp.float32)
  w1 = w1 * FFN_HIDDEN**-0.5
  w2 = jax.random.normal(ks[2], (FFN_INTER, FFN_HIDDEN), jnp.float32)
  w2 = w2 * FFN_INTER**-0.5
  b1 = b2 = None
  if bias:
    b1 = jax.random.normal(ks[3], (1, 2 * FFN_INTER), jnp.float32) * 0.1
    b2 = jax.random.normal(ks[4], (1, FFN_HIDDEN), jnp.float32) * 0.1
  if fmt == "bf16":
    return (
        x,
        w1.astype(jnp.bfloat16),
        w2.astype(jnp.bfloat16),
        None,
        None,
        b1,
        b2,
    )

  def quantize(w):
    scale = jnp.maximum(jnp.abs(w).max(axis=0, keepdims=True), 1e-12) / float(
        jnp.finfo(FP8).max
    )
    return (w / scale).astype(FP8), scale

  (q1, s1), (q2, s2) = quantize(w1), quantize(w2)
  return x, q1, q2, s1, s2, b1, b2


def _kernel_ffn(fmt, x, w1, w2, s1, s2, b1, b2, act_fn, wire):
  """The fused kernel's expert FFN plus the tile epilogue, on values."""

  def fn(x, w1, w2, s1, s2, b1, b2):
    if fmt == "fp8":
      act_q, act_scale = input_pack.rowquant_fp8(x)
      acc2, mid_scale = fused_kernel.expert_ffn_fp8(
          act_q, act_scale, w1, w2, s1, act_fn=act_fn, w1b=b1
      )
    else:
      acc2, mid_scale = fused_kernel.expert_ffn_bf16(
          x, w1, w2, act_fn=act_fn, w1b=b1
      )
    rows, row_scale = fused_kernel.wire_row_and_scale(
        acc2,
        mid_scale,
        w2s=s2,
        w2b=b2,
        wire_dtype=FP8 if wire == "fp8" else jnp.bfloat16,
    )
    return rows.astype(jnp.float32) * row_scale

  return jax.jit(fn)(x, w1, w2, s1, s2, b1, b2)


class ExpertFfnTest(parameterized.TestCase):
  """The expert FFN (+ epilogue) vs the f32 per-expert reference."""

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_supported(self)

  @parameterized.named_parameters(*_ffn_cases())
  def test_expert_ffn_matches_reference(self, fmt, act_fn, bias, m):
    x, w1, w2, s1, s2, b1, b2 = _expert_operands(fmt, m, act_fn, bias, seed=m)
    w1f = w1.astype(jnp.float32) * (1.0 if s1 is None else s1)
    w2f = w2.astype(jnp.float32) * (1.0 if s2 is None else s2)
    want = reference.expert_ffn(x, w1f, w2f, act_fn=act_fn, b1=b1, b2=b2)
    # The layer's wire is fp8 for fp8 weights and bf16 for bf16 weights.
    for wire in {"fp8": ("fp8", "bf16"), "bf16": ("bf16",)}[fmt]:
      with self.subTest(wire=wire):
        got = _kernel_ffn(fmt, x, w1, w2, s1, s2, b1, b2, act_fn, wire)
        _assert_close(self, got, want, fmt, f"ffn {act_fn} wire={wire}")


if __name__ == "__main__":
  jax.config.update("jax_num_cpu_devices", CPU_EP)
  absltest.main()
