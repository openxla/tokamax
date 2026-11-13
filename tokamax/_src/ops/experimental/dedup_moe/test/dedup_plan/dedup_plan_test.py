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
"""Golden tests of the on-device De-Dup plan (dedup_plan.py).

References:
  * A small numpy reference of the De-Dup plan (`_ref_source`): stage
    selection (tokens needed by >= min_fanout devices of a remote board),
    the stage order (stage wave, then the inverse Gray code of the target
    mask, then token) and the forward runs (maximal stretches of stage rows
    of one wave that one target needs);
  * `wave_dispatch.dispatch_plan`, the reference direct wave plan: with
    staging off the plan's direct part is exactly that plan.
The exchange is emulated in numpy from the plan arrays alone (direct runs,
stage runs, gateway forwards), checking that no row is written twice, that
forwards only read their own (slot, wave) stage group, and that every needed
(source, token) lands exactly once at `row_on_dest`.

The routing is drawn in numpy (`_generate_routing`: uniform or Zipf-skewed
expert popularity, distinct experts per token) and the routing waves come
from `phase1_ref.wave_of_from_rows_by_dest` (equal-width waves); the plan
functions take any `wave_of`.
"""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dp
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib
from tokamax._src.ops.experimental.dedup_moe.dispatch import wave_dispatch
from tokamax._src.ops.experimental.dedup_moe.test.common import phase1_ref
from tokamax._src.ops.experimental.fused_moe import host as fused_host

import pytest

GATEWAY_MAPS = ("mirror", "rotate", "single")


@dataclasses.dataclass(frozen=True)
class Cfg:
  """One routing + topology configuration."""

  name: str
  ep: int
  board_size: int
  num_experts: int
  top_k: int
  t: int
  waves: int
  zipf: float = 0.0
  min_fanout: int = 2
  seed: int = 0
  gateways: str = "mirror"


GOLDEN = (
    # Two boards of 4.
    Cfg("ep8_bs4_t32_w1", 8, 4, 64, 4, 32, 1),
    Cfg("ep8_bs4_t64_w2_zipf", 8, 4, 64, 6, 64, 2, zipf=1.2, seed=1),
    Cfg("ep8_bs4_t16_w4_mf1", 8, 4, 64, 4, 16, 4, min_fanout=1, seed=2),
    # Two boards of 8.
    Cfg("ep16_bs8_t32_w1", 16, 8, 64, 4, 32, 1),
    Cfg("ep16_bs8_t32_w2_zipf", 16, 8, 64, 4, 32, 2, zipf=1.2),
    Cfg("ep16_bs8_t128_w4_zipf", 16, 8, 64, 10, 128, 4, zipf=1.2, seed=1),
    Cfg("ep16_bs8_t16_w2", 16, 8, 64, 4, 16, 2, seed=2),
    Cfg("ep16_bs8_t8_w1_mf1", 16, 8, 64, 4, 8, 1, min_fanout=1, seed=3),
    Cfg("ep16_bs8_t2048_w2_zipf", 16, 8, 64, 4, 2048, 2, zipf=1.2, seed=4),
    # Four boards of 4.
    Cfg("ep16_bs4_t32_w2", 16, 4, 64, 4, 32, 2, seed=5),
    Cfg("ep16_bs4_t128_w4_zipf", 16, 4, 64, 10, 128, 4, zipf=1.2, seed=6),
    # Four boards of 8.
    Cfg("ep32_bs8_t32_w2", 32, 8, 64, 4, 32, 2, seed=5),
    Cfg("ep32_bs8_t128_w2_zipf", 32, 8, 64, 10, 128, 2, zipf=1.2, seed=6),
    Cfg("ep32_bs8_t16_w1", 32, 8, 512, 10, 16, 1, seed=7),
    Cfg("ep32_bs8_t128_w4_zipf", 32, 8, 512, 10, 128, 4, zipf=1.2, seed=8),
    Cfg("ep32_bs8_t64_w4", 32, 8, 512, 8, 64, 4, seed=9),
    Cfg("ep32_bs8_t64_w8_zipf", 32, 8, 512, 10, 64, 8, zipf=1.2, seed=11),
    # Eight boards of 4 and eight boards of 8.
    Cfg("ep32_bs4_t32_w2_zipf", 32, 4, 256, 8, 32, 2, zipf=1.2, seed=12),
    Cfg("ep64_bs8_t8_w2", 64, 8, 512, 10, 8, 2, seed=10),
    # Custom (non-mirror) gateway maps.
    Cfg(
        "ep8_bs4_rotate_t32_w2", 8, 4, 64, 4, 32, 2, seed=13, gateways="rotate"
    ),
    Cfg(
        "ep16_bs8_rotate_t64_w2_zipf",
        16,
        8,
        64,
        8,
        64,
        2,
        zipf=1.2,
        seed=14,
        gateways="rotate",
    ),
    Cfg(
        "ep16_bs4_single_t32_w2",
        16,
        4,
        64,
        6,
        32,
        2,
        seed=15,
        gateways="single",
    ),
    Cfg(
        "ep32_bs8_single_t32_w4_zipf",
        32,
        8,
        512,
        10,
        32,
        4,
        zipf=1.2,
        seed=16,
        gateways="single",
    ),
)
_BY_NAME = {c.name: c for c in GOLDEN}


def _np(tree):
  return jax.tree_util.tree_map(np.asarray, tree)


def _gateway_map(kind: str, ep: int, bs: int) -> list[list[int]] | None:
  """The [ep][nb] gateway map of a named kind (None = mirror)."""
  nb = ep // bs
  if kind == "mirror":
    return None
  if kind == "rotate":
    local = lambda s, b: (s + 1 + b % (bs - 1)) % bs
  elif kind == "single":
    local = lambda s, b: 0
  else:
    raise ValueError(kind)
  return [
      [-1 if b == s // bs else b * bs + local(s, b) for b in range(nb)]
      for s in range(ep)
  ]


def _topology(cfg: Cfg) -> topology_lib.BoardTopology:
  return topology_lib.BoardTopology.uniform(
      cfg.ep,
      cfg.board_size,
      gateways=_gateway_map(cfg.gateways, cfg.ep, cfg.board_size),
  )


def _generate_routing(
    ep: int, t: int, num_experts: int, top_k: int, *, seed: int, zipf: float
) -> np.ndarray:
  """[ep, t, K] expert ids, distinct per token.

  Expert popularity is uniform (zipf = 0) or Zipf(zipf) over a random
  permutation of the experts; each token draws top_k distinct experts without
  replacement (Gumbel top-k).

  Args:
    ep: Devices.
    t: Tokens per device.
    num_experts: E.
    top_k: K.
    seed: RNG seed.
    zipf: Zipf exponent, 0 for uniform.

  Returns:
    The expert ids.
  """
  rng = np.random.default_rng(seed)
  if zipf > 0:
    weights = 1.0 / np.arange(1, num_experts + 1) ** zipf
    weights = weights[rng.permutation(num_experts)]
  else:
    weights = np.ones(num_experts)
  logits = np.log(weights / weights.sum())
  keys = logits + rng.gumbel(size=(ep * t, num_experts))
  ids = np.argsort(-keys, axis=1)[:, :top_k].astype(np.int32)
  return ids.reshape(ep, t, top_k)


def _ref_inverse_gray(code: np.ndarray) -> np.ndarray:
  """Position of each code in the reflected Gray sequence (bitwise loop)."""
  code = np.asarray(code, np.int64)
  out = np.zeros_like(code)
  shifted = code.copy()
  while np.any(shifted):
    out ^= shifted
    shifted >>= 1
  return out


@dataclasses.dataclass
class RefStage:
  """Reference stage rows of one (source, remote board) pair, stage order."""

  tokens: np.ndarray  # [n] staged tokens.
  codes: np.ndarray  # [n] target masks.
  waves: np.ndarray  # [n] stage waves (min fw over the board).
  runs: list[list[tuple[int, int, int, int]]]  # per wave.


@dataclasses.dataclass
class RefSource:
  """Reference De-Dup plan of one source."""

  direct: np.ndarray  # [ep, t] bool, sent directly.
  n_dir: np.ndarray  # [ep]
  boards: dict[int, RefStage]


def _ref_forward_runs(
    st: RefStage, *, s: int, b: int, bs: int, t: int, n_dir, num_waves: int
) -> list[list[tuple[int, int, int, int]]]:
  """Forward runs per wave: maximal stretches one target needs."""
  bits = (st.codes[:, None] >> np.arange(bs)[None, :]) & 1  # [n, bs]
  prefix = np.cumsum(bits, axis=0) - bits
  runs = []
  for wave in range(num_waves):
    rows = np.flatnonzero(st.waves == wave)
    ents = []
    if rows.size:
      lo, hi = int(rows[0]), int(rows[-1]) + 1
      assert hi - lo == rows.size  # Waves are contiguous in stage order.
      for r in range(lo, hi):
        for j in range(bs):
          if bits[r, j] and (r == lo or not bits[r - 1, j]):
            n = 1
            while r + n < hi and bits[r + n, j]:
              n += 1
            d = b * bs + j
            ents.append((r, n, d, s * t + int(n_dir[d]) + int(prefix[r, j])))
    runs.append(ents)
  return runs


def _ref_source(
    fw_s: np.ndarray, s: int, *, bs: int, num_waves: int, min_fanout: int
) -> RefSource:
  """The reference De-Dup plan of source s from its first waves fw_s [ep, t]."""
  ep, t = fw_s.shape
  nb = ep // bs
  need = fw_s < num_waves
  direct = need.copy()
  boards = {}
  for b in range(nb):
    if b == s // bs:
      continue
    lo, hi = b * bs, (b + 1) * bs
    pop = need[lo:hi].sum(axis=0)
    staged = np.flatnonzero(pop >= max(1, min_fanout))
    if not staged.size:
      continue
    direct[lo:hi, staged] = False
    codes = (
        need[lo:hi][:, staged].astype(np.int64) << np.arange(bs)[:, None]
    ).sum(axis=0)
    waves = fw_s[lo:hi][:, staged].min(axis=0)
    order = np.lexsort((staged, _ref_inverse_gray(codes), waves))
    boards[b] = RefStage(
        tokens=staged[order], codes=codes[order], waves=waves[order], runs=[]
    )
  n_dir = direct.sum(axis=1)
  for b, st in boards.items():
    st.runs = _ref_forward_runs(
        st, s=s, b=b, bs=bs, t=t, n_dir=n_dir, num_waves=num_waves
    )
  return RefSource(direct=direct, n_dir=n_dir, boards=boards)


@dataclasses.dataclass
class Env:
  """Everything a golden check needs for one configuration."""

  cfg: Cfg
  topo: topology_lib.BoardTopology
  tables: topology_lib.TopologyTables
  experts: np.ndarray  # [ep, t, K]
  mask: np.ndarray  # [src, t, dst] bool
  pairs_g: jax.Array  # [T, K]
  wave_of: jax.Array  # [ep, g]
  fw_all: np.ndarray  # [ep (source), ep (dest), t]
  plans: list[dp.SourcePlan]
  gws: list[dp.GatewayPlan]
  rcs: list[dp.ReceiverCounts]
  fwd_schedule: dp.FwdSchedule = dp.FwdSchedule()

  @property
  def ep(self):
    return self.tables.num_devices

  @property
  def nb(self):
    return self.tables.num_boards

  @property
  def bs(self):
    return self.tables.board_size

  def ref(self) -> list[RefSource]:
    return [
        _ref_source(
            self.fw_all[s],
            s,
            bs=self.bs,
            num_waves=self.cfg.waves,
            min_fanout=self.cfg.min_fanout,
        )
        for s in range(self.ep)
    ]


def _rows_by_dest(experts: np.ndarray, num_experts: int) -> np.ndarray:
  ep, t, k = experts.shape
  out = np.zeros((ep, num_experts), np.int32)
  np.add.at(out, (np.repeat(np.arange(ep), t * k), experts.reshape(-1)), 1)
  return out


def _phase1_fn(*, t, waves):
  return jax.jit(
      lambda pairs, wave_of, s: phase1_ref.phase1_inputs(
          pairs, wave_of, s, t=t, num_waves=waves
      ).fw
  )


def build_env(cfg: Cfg, fwd_schedule: dp.FwdSchedule = dp.FwdSchedule()) -> Env:
  """Routing, the per-source fw rows and every device's three plan views."""
  topo = _topology(cfg)
  tables = topo.tables()
  ep, t, w, bs = cfg.ep, cfg.t, cfg.waves, cfg.board_size
  g = cfg.num_experts // ep
  experts = _generate_routing(
      ep, t, cfg.num_experts, cfg.top_k, seed=cfg.seed, zipf=cfg.zipf
  )
  dest = experts // g  # [ep, t, K]
  mask = (dest[..., None] == np.arange(ep)).any(axis=2)  # [src, t, dst]
  pairs_g = jnp.asarray(experts.reshape(ep * t, cfg.top_k))
  wave_of = phase1_ref.wave_of_from_rows_by_dest(
      jnp.asarray(_rows_by_dest(experts, cfg.num_experts)), g=g, num_waves=w
  )
  p1 = _phase1_fn(t=t, waves=w)
  fw_all = np.stack(
      [np.asarray(p1(pairs_g, wave_of, jnp.int32(s))) for s in range(ep)]
  )
  mf = cfg.min_fanout
  plan_fn = jax.jit(
      lambda fw, s: dp.dedup_plan(
          fw,
          s,
          tables=tables,
          num_waves=w,
          min_fanout=mf,
          schedule=fwd_schedule,
      )
  )
  gw_fn = jax.jit(
      lambda fws, gw: dp.gateway_plan(
          fws,
          gw,
          tables=tables,
          num_waves=w,
          min_fanout=mf,
          schedule=fwd_schedule,
      )
  )
  rc_fn = jax.jit(
      lambda fwb, me: dp.receiver_counts(
          fwb,
          me,
          tables=tables,
          num_waves=w,
          min_fanout=mf,
          schedule=fwd_schedule,
      )
  )
  plans = [_np(plan_fn(fw_all[s], jnp.int32(s))) for s in range(ep)]
  gws = []
  for gw in range(ep):
    b = gw // bs
    src = np.maximum(tables.served[gw], 0)
    fws = fw_all[src][:, bs * b : bs * (b + 1)]
    gws.append(_np(gw_fn(fws, jnp.int32(gw))))
  rcs = []
  for me in range(ep):
    b = me // bs
    rcs.append(_np(rc_fn(fw_all[:, bs * b : bs * (b + 1)], jnp.int32(me))))
  return Env(
      cfg=cfg,
      topo=topo,
      tables=tables,
      experts=experts,
      mask=mask,
      pairs_g=pairs_g,
      wave_of=wave_of,
      fw_all=fw_all,
      plans=plans,
      gws=gws,
      rcs=rcs,
      fwd_schedule=fwd_schedule,
  )


class HelperTest(parameterized.TestCase):

  def test_inverse_gray(self):
    codes = np.arange(256, dtype=np.int64)
    np.testing.assert_array_equal(
        np.asarray(dp.inverse_gray(jnp.asarray(codes, jnp.int32))),
        _ref_inverse_gray(codes),
    )
    # The reference inverts the reflected Gray code g(i) = i ^ (i >> 1).
    np.testing.assert_array_equal(
        _ref_inverse_gray(codes ^ (codes >> 1)), codes
    )

  @parameterized.parameters(1, 5, 128, 1024, 1025, 1500, 2048)
  def test_exclusive_prefix(self, n):
    rng = np.random.default_rng(n)
    x = rng.random((3, 2, n)) < 0.5
    got = np.asarray(jax.jit(dp.exclusive_prefix)(jnp.asarray(x)))
    exp = np.cumsum(x, axis=-1) - x
    np.testing.assert_array_equal(got, exp)

  def test_exclusive_prefix_all_ones(self):
    x = np.ones((1, 4096), np.int32)
    got = np.asarray(jax.jit(dp.exclusive_prefix)(jnp.asarray(x)))
    np.testing.assert_array_equal(got[0], np.arange(4096))

  @parameterized.parameters(1, 2, 7, 64, 100)
  def test_suffix_min(self, n):
    rng = np.random.default_rng(n)
    x = rng.integers(0, 1000, (2, n)).astype(np.int32)
    got = np.asarray(dp.suffix_min(jnp.asarray(x), 1 << 30))
    exp = np.minimum.accumulate(x[:, ::-1], axis=1)[:, ::-1]
    np.testing.assert_array_equal(got, exp)

  def test_generate_routing(self):
    for zipf in (0.0, 1.2):
      ids = _generate_routing(4, 64, 32, 6, seed=0, zipf=zipf)
      self.assertEqual(ids.shape, (4, 64, 6))
      self.assertTrue(np.all((ids >= 0) & (ids < 32)))
      srt = np.sort(ids, axis=-1)
      self.assertFalse(np.any(srt[..., 1:] == srt[..., :-1]))  # Distinct.


class PlanChecks(parameterized.TestCase):
  """The golden checks, shared by the test classes below (no tests here)."""

  def _check_phase1_rows(self, env):
    """fw < W is exactly the routing's destination mask."""
    w = env.cfg.waves
    for s in range(env.ep):
      np.testing.assert_array_equal(env.fw_all[s] < w, env.mask[s].T)
      self.assertTrue(np.all(env.fw_all[s] <= w))

  def _check_stage_vs_reference(self, env, ref):
    """Stage selection, direct tokens and stage order vs the reference."""
    t, w, bs = env.cfg.t, env.cfg.waves, env.bs
    gray_sh, wave_sh = dp.key_shifts(t)
    for s in range(env.ep):
      p, rs = env.plans[s], ref[s]
      dir_tok = p.oh_dir.any(axis=1)
      np.testing.assert_array_equal(dir_tok, rs.direct, err_msg=f"s={s}")
      np.testing.assert_array_equal(p.n_dir, rs.n_dir)
      np.testing.assert_array_equal(p.cnt_dir.sum(axis=1), p.n_dir)
      np.testing.assert_array_equal(p.cnt_dir, p.oh_dir.sum(axis=2))
      self.assertFalse(np.any(p.oh_dir.sum(axis=1) > 1))
      for b in range(env.nb):
        n = int(p.n_stage[b])
        np.testing.assert_array_equal(p.mask_sorted[b, n:], 0)
        np.testing.assert_array_equal(
            p.stage_pos[b][p.stage_perm[b]], np.arange(t)
        )
        bits = (p.mask_sorted[b][None, :] >> np.arange(bs)[:, None]) & 1
        np.testing.assert_array_equal(
            p.fprefix[b], np.cumsum(bits, axis=1) - bits
        )
        if b not in rs.boards:
          self.assertEqual(n, 0, f"s={s} b={b}")
          self.assertFalse(np.any(p.stage[b]))
          np.testing.assert_array_equal(p.w_stage[b], w)
          continue
        st = rs.boards[b]
        self.assertEqual(n, st.tokens.size, f"s={s} b={b}")
        np.testing.assert_array_equal(
            p.stage_perm[b, :n], st.tokens, err_msg=f"s={s} b={b}"
        )
        np.testing.assert_array_equal(p.mask_sorted[b, :n], st.codes)
        np.testing.assert_array_equal(
            np.flatnonzero(p.stage[b]), np.sort(st.tokens)
        )
        np.testing.assert_array_equal(p.mask8[b][st.tokens], st.codes)
        np.testing.assert_array_equal(p.w_stage[b][st.tokens], st.waves)
        np.testing.assert_array_equal(p.w_stage[b][~p.stage[b]], w)
        np.testing.assert_array_equal(
            p.cnt_stage[b], np.bincount(st.waves, minlength=w)[:w]
        )
        tok = np.arange(t)
        key = np.where(
            p.stage[b],
            (p.w_stage[b] << wave_sh)
            | (_ref_inverse_gray(p.mask8[b]) << gray_sh)
            | tok,
            dp.NOT_STAGED | tok,
        )
        np.testing.assert_array_equal(p.stage_key[b], key)
      self.assertEqual(int(p.n_stage[s // bs]), 0)

  def _check_forward_runs_vs_reference(self, env, ref):
    """Forward runs per (b, w) group = the reference's maximal runs."""
    w = env.cfg.waves
    for s in range(env.ep):
      p, rs = env.plans[s], ref[s]
      for b in range(env.nb):
        if b not in rs.boards:
          np.testing.assert_array_equal(p.fwd_cnt[b], 0)
          np.testing.assert_array_equal(p.fwd_runs[b], 0)
          continue
        for wave in range(w):
          exp = rs.boards[b].runs[wave]
          cnt = int(p.fwd_cnt[b, wave])
          # D5: Gray order bounds the runs of one group.
          self.assertLessEqual(cnt, dp.MAX_FWD_RUNS)
          got = [
              tuple(int(v) for v in row) for row in p.fwd_runs[b, wave, :cnt]
          ]
          self.assertEqual(got, exp, f"s={s} b={b} w={wave}")
          np.testing.assert_array_equal(p.fwd_runs[b, wave, cnt:], 0)

  def _check_gateway_vs_reference(self, env, ref):
    """Gateway expected stage rows and absolute forward runs vs reference."""
    t, w, bs = env.cfg.t, env.cfg.waves, env.bs
    n_groups = 0
    for gw in range(env.ep):
      gp = env.gws[gw]
      bg = gw // bs
      for sigma in range(env.tables.num_slots):
        src = int(env.tables.served[gw, sigma])
        if src < 0 or bg not in ref[src].boards:
          np.testing.assert_array_equal(gp.stage_rows[sigma], 0)
          np.testing.assert_array_equal(gp.fwd_cnt[sigma], 0)
          continue
        st = ref[src].boards[bg]
        n_groups += 1
        np.testing.assert_array_equal(
            gp.stage_rows[sigma], np.bincount(st.waves, minlength=w)[:w]
        )
        for wave in range(w):
          cnt = int(gp.fwd_cnt[sigma, wave])
          got = [
              (int(r) - sigma * t, int(n), int(d), int(dr))
              for r, n, d, dr in gp.fwd_tab[sigma, wave, :cnt]
          ]
          self.assertEqual(
              got, st.runs[wave], f"gw={gw} sigma={sigma} w={wave}"
          )
    # Every non-empty gateway slot is a staged (source, board) pair.
    self.assertEqual(n_groups, sum(len(r.boards) for r in ref))
    self.assertEqual(
        n_groups, int(sum((g.stage_rows.sum(1) > 0).sum() for g in env.gws))
    )

  def _check_exchange(self, env):
    """numpy emulation of the exchange from the plan arrays alone."""
    ep, nb, t, w, bs = env.ep, env.nb, env.cfg.t, env.cfg.waves, env.bs
    num_slots = env.tables.num_slots
    recv = np.full((ep, ep * t, 2), -1, np.int64)
    recv_wave = np.full((ep, ep * t), -1, np.int64)
    stage = np.full((ep, num_slots * t, 2), -1, np.int64)
    owner = np.full((ep, num_slots * t, 2), -1, np.int64)

    def write(buf, dev, row, payload, what):
      n = len(payload)
      self.assertTrue(
          0 <= row and row + n <= buf.shape[1], f"{what} OOB dev={dev}"
      )
      self.assertFalse(
          np.any(buf[dev, row : row + n, 0] >= 0), f"{what} rewrite dev={dev}"
      )
      buf[dev, row : row + n] = payload

    for s in range(ep):
      p = env.plans[s]
      total = int(p.send_total)
      self.assertEqual(total, int(p.cnt_dir.sum()) + int(p.cnt_stage.sum()))
      self.assertEqual(int(p.remote_rows), total - int(p.n_dir[s]))
      valid = p.send_row >= 0
      rows = p.send_row[valid]
      self.assertEqual(sorted(rows.tolist()), list(range(total)))
      np.testing.assert_array_equal(p.perm[rows], np.nonzero(valid)[1])
      send = np.stack([np.full(total, s), p.perm[:total]], axis=1)
      for d in range(ep):
        for wave in range(w):
          n = int(p.cnt_dir[d, wave])
          src_row, dst_row = int(p.send_off[d, wave]), int(p.dst_off[d, wave])
          self.assertEqual(dst_row, s * t + int(p.slot_off[d, wave]))
          if not n:
            continue
          self.assertTrue(s * t <= dst_row and dst_row + n <= (s + 1) * t)
          payload = send[src_row : src_row + n]
          np.testing.assert_array_equal(
              payload[:, 1], np.flatnonzero(p.oh_dir[d, wave])
          )
          write(recv, d, dst_row, payload, "direct")
          recv_wave[d, dst_row : dst_row + n] = wave
      for b in range(nb):
        if b == s // bs:
          np.testing.assert_array_equal(p.cnt_stage[b], 0)
          self.assertEqual((int(p.gw[b]), int(p.sigma[b])), (-1, -1))
          continue
        gw, sigma = int(p.gw[b]), int(p.sigma[b])
        self.assertEqual(gw, env.tables.gw[s, b])
        self.assertEqual(sigma, env.tables.sigma_at_gw[s, b])
        off = 0
        for wave in range(w):
          n = int(p.cnt_stage[b, wave])
          src_row = int(p.stage_send_off[b, wave])
          dst_row = int(p.stage_dst[b, wave])
          self.assertEqual(dst_row, sigma * t + off)
          if n:
            payload = send[src_row : src_row + n]
            np.testing.assert_array_equal(
                payload[:, 1], p.stage_perm[b, off : off + n]
            )
            write(stage, gw, dst_row, payload, "stage")
            owner[gw, dst_row : dst_row + n] = (sigma, wave)
          off += n

    forwarded = np.zeros((ep, num_slots * t), bool)
    for gw in range(ep):
      gp = env.gws[gw]
      np.testing.assert_array_equal(gp.src, env.tables.served[gw])
      for sigma in range(num_slots):
        for wave in range(w):
          cnt = int(gp.fwd_cnt[sigma, wave])
          self.assertLessEqual(cnt, dp.MAX_FWD_RUNS)
          for i in range(cnt):
            r, n, d, dr = (int(v) for v in gp.fwd_tab[sigma, wave, i])
            self.assertGreater(n, 0)
            self.assertEqual(d // bs, gw // bs)
            np.testing.assert_array_equal(
                owner[gw, r : r + n],
                np.tile([sigma, wave], (n, 1)),
                err_msg=f"gw={gw} sigma={sigma} w={wave} reads outside group",
            )
            write(recv, d, dr, stage[gw, r : r + n], "forward")
            recv_wave[d, dr : dr + n] = wave
            forwarded[gw, r : r + n] = True
          np.testing.assert_array_equal(gp.fwd_tab[sigma, wave, cnt:], 0)
    np.testing.assert_array_equal(forwarded, stage[..., 0] >= 0)

    for d in range(ep):
      valid = recv[d, :, 0] >= 0
      got = recv[d][valid]
      exp = np.argwhere(env.mask[:, :, d])
      self.assertLen(got, len(exp))
      np.testing.assert_array_equal(np.unique(got, axis=0), exp)
      np.testing.assert_array_equal(
          np.bincount(recv_wave[d][valid], minlength=w)[:w],
          env.rcs[d].recv_cnt,
          err_msg=f"per-wave arrivals d={d}",
      )
      for s in range(ep):
        slot_waves = recv_wave[d, s * t : (s + 1) * t]
        np.testing.assert_array_equal(
            np.bincount(slot_waves[slot_waves >= 0], minlength=w)[:w],
            env.plans[s].recv_cnt[d],
        )
    return recv

  def _check_row_on_dest(self, env, recv):
    """row_on_dest: dense, unique, inside slot s, holds (s, t)."""
    t = env.cfg.t
    for s in range(env.ep):
      p = env.plans[s]
      need = env.mask[s].T
      np.testing.assert_array_equal(p.row_on_dest >= 0, need)
      for d in range(env.ep):
        rows = p.row_on_dest[d][need[d]]
        n = int(p.n_dir[d] + p.n_fwd[d])
        self.assertEqual(sorted(rows.tolist()), list(range(s * t, s * t + n)))
        np.testing.assert_array_equal(recv[d, rows, 0], s)
        np.testing.assert_array_equal(recv[d, rows, 1], np.flatnonzero(need[d]))

  def _check_inter_board_rows(self, env, ref):
    """Rows per (source board, destination board) pair vs the reference."""
    nb, bs = env.nb, env.bs
    ours = np.zeros((nb, nb), np.int64)
    want = np.zeros((nb, nb), np.int64)
    naive = np.zeros((nb, nb), np.int64)
    for s in range(env.ep):
      p, rs = env.plans[s], ref[s]
      for b in range(nb):
        if b == s // bs:
          continue
        devs = slice(bs * b, bs * (b + 1))
        ours[s // bs, b] += int(p.n_dir[devs].sum()) + int(p.n_stage[b])
        want[s // bs, b] += int(rs.n_dir[devs].sum())
        if b in rs.boards:
          want[s // bs, b] += rs.boards[b].tokens.size
        naive[s // bs, b] += int(env.mask[s][:, devs].sum())
    np.testing.assert_array_equal(ours, want)
    # De-Dup never sends more inter-board rows than the direct send.
    self.assertTrue(np.all(ours <= naive))
    if env.cfg.min_fanout == 2 and any(r.boards for r in ref):
      self.assertLess(int(ours.sum()), int(naive.sum()))

  def _check_views_agree(self, env):
    """Sender, gateway and receiver views agree."""
    bs = env.bs
    for gw in range(env.ep):
      gp = env.gws[gw]
      bg = gw // bs
      self.assertTrue(np.all(gp.fwd_cnt <= env.fwd_schedule.max_runs))
      np.testing.assert_array_equal(gp.fwd_slot_cnt.sum(axis=-1), gp.fwd_cnt)
      for sigma in range(env.tables.num_slots):
        src = int(env.tables.served[gw, sigma])
        if src < 0:
          np.testing.assert_array_equal(gp.stage_rows[sigma], 0)
          np.testing.assert_array_equal(gp.fwd_cnt[sigma], 0)
          np.testing.assert_array_equal(gp.fwd_tab[sigma], 0)
          continue
        p = env.plans[src]
        np.testing.assert_array_equal(gp.stage_rows[sigma], p.cnt_stage[bg])
        np.testing.assert_array_equal(gp.fwd_runs[sigma], p.fwd_runs[bg])
        np.testing.assert_array_equal(gp.fwd_cnt[sigma], p.fwd_cnt[bg])
        np.testing.assert_array_equal(
            gp.fwd_slot_cnt[sigma], p.fwd_slot_cnt[bg]
        )
      rows = gp.fwd_tab[..., dp.FWD_ROWS]
      dst = gp.fwd_tab[..., dp.FWD_DST] & dp.FWD_DST_MASK
      np.testing.assert_array_equal(
          gp.fwd_local, np.where(dst == gw, rows, 0).sum(axis=(0, 2))
      )
      np.testing.assert_array_equal(
          gp.fwd_remote + gp.fwd_local, rows.sum(axis=(0, 2))
      )
    for d in range(env.ep):
      rc = env.rcs[d]
      plans = env.plans
      exp = np.stack([p.recv_cnt[d] for p in plans])
      np.testing.assert_array_equal(rc.cnt_from, exp)
      np.testing.assert_array_equal(rc.recv_cnt, exp.sum(axis=0))
      np.testing.assert_array_equal(
          rc.rows_from, [p.n_dir[d] + p.n_fwd[d] for p in plans]
      )
      np.testing.assert_array_equal(rc.n_fwd_from, [p.n_fwd[d] for p in plans])

  def _check_kernel_operands(self, env):
    """stage_tab / fwd_tab packing and the De-Dup disp_tab layout."""
    ep, nb, w = env.ep, env.nb, env.cfg.waves
    num_slots = env.tables.num_slots
    layout = dp.stage_tab_layout(nb, num_slots)
    for me in (0, ep - 1):
      p, gp, rc = env.plans[me], env.gws[me], env.rcs[me]
      ops = _np(dp.pack_smem_operands(p, gp))
      self.assertEqual(ops.stage_tab.shape, (2 * nb + num_slots + 2, w))
      np.testing.assert_array_equal(
          ops.stage_tab[layout["stage_send_off"]], p.stage_send_off
      )
      np.testing.assert_array_equal(
          ops.stage_tab[layout["stage_rows"]], p.cnt_stage
      )
      np.testing.assert_array_equal(
          ops.stage_tab[layout["expected_stage_rows"]], gp.stage_rows
      )
      np.testing.assert_array_equal(
          ops.stage_tab[layout["fwd_remote"]][0], gp.fwd_remote
      )
      np.testing.assert_array_equal(
          ops.stage_tab[layout["fwd_local"]][0], gp.fwd_local
      )
      np.testing.assert_array_equal(ops.fwd_tab, gp.fwd_tab)
      np.testing.assert_array_equal(ops.fwd_cnt, gp.fwd_slot_cnt)
      # D4: no early slot, every run waits for its own stage run.
      self.assertEqual(ops.fwd_cnt.shape, (num_slots, w, dp.FWD_SLOTS))
      np.testing.assert_array_equal(ops.fwd_cnt[..., 0], 0)
      np.testing.assert_array_equal(ops.fwd_cnt[..., 1], gp.fwd_cnt)
      tab = np.asarray(dp.dedup_disp_tab(p, rc.recv_cnt))
      self.assertEqual(tab.shape, (4 * ep, w))
      np.testing.assert_array_equal(tab[:ep], p.send_off)
      np.testing.assert_array_equal(tab[ep : 2 * ep], p.cnt_dir)
      np.testing.assert_array_equal(tab[2 * ep : 3 * ep], p.dst_off)
      np.testing.assert_array_equal(tab[3 * ep], rc.recv_cnt)
      self.assertEqual(int(tab[3 * ep + 1, 0]), int(p.remote_rows))
      np.testing.assert_array_equal(tab[3 * ep + 1, 1:], 0)
      np.testing.assert_array_equal(tab[3 * ep + 2 :], 0)


class DedupPlanGoldenTest(PlanChecks):

  @parameterized.named_parameters((c.name, c) for c in GOLDEN)
  def test_golden(self, cfg):
    env = build_env(cfg)
    ref = env.ref()
    self._check_phase1_rows(env)
    self._check_stage_vs_reference(env, ref)
    self._check_forward_runs_vs_reference(env, ref)
    self._check_gateway_vs_reference(env, ref)
    recv = self._check_exchange(env)
    self._check_row_on_dest(env, recv)
    self._check_inter_board_rows(env, ref)
    self._check_views_agree(env)
    self._check_kernel_operands(env)


# Configurations the per-target-wave schedule is tested on (max_runs = bs t,
# the bound every (row, target) pair being its own run gives).
PTW_CASES = (
    "ep8_bs4_t64_w2_zipf",
    "ep16_bs8_t128_w4_zipf",
    "ep16_bs4_t128_w4_zipf",
    "ep32_bs8_t64_w4",
    "ep64_bs8_t8_w2",
    "ep32_bs8_t64_w8_zipf",
    "ep16_bs8_rotate_t64_w2_zipf",
    "ep16_bs4_single_t32_w2",
)


def _ptw_schedule(cfg: Cfg, max_runs: int | None = None) -> dp.FwdSchedule:
  return dp.FwdSchedule(
      per_target_wave=True,
      max_runs=cfg.board_size * cfg.t if max_runs is None else max_runs,
  )


class FwdScheduleTest(PlanChecks):
  """Per-target-wave forwarding, emulated as the kernels run.

  Senders send every (board, wave) stage run on its own gateway semaphore
  (`tc_sender.ragged_a2a_start_waves`); every gateway then forwards wave w
  for w = 0..W-1: ready slot 0 of each served source (rows that crossed in
  an earlier stage wave), then per source wait for the wave-w stage run
  (byte-exact) and issue slot 1. Every run lands on its target's signal-wave
  semaphore.
  """

  @parameterized.named_parameters((c, c) for c in PTW_CASES)
  def test_per_target_wave(self, cfg_name):
    cfg = _BY_NAME[cfg_name]
    fwd_schedule = _ptw_schedule(cfg)
    env = build_env(cfg, fwd_schedule)
    bs = env.bs
    self._check_views_agree(env)
    recv = self._check_scheduled_exchange(env)
    self._check_row_on_dest(env, recv)
    # The receiver counts forwarded rows in the target's own first wave.
    for d in range(env.ep):
      b = d // bs
      fwd_waves = []
      for s in range(env.ep):
        p = env.plans[s]
        if b == s // bs:
          continue
        sel = p.stage[b] & (env.fw_all[s, d] < cfg.waves)
        fwd_waves.append(env.fw_all[s, d][sel])
      got = np.concatenate(fwd_waves) if fwd_waves else np.zeros(0, int)
      np.testing.assert_array_equal(
          np.bincount(got, minlength=cfg.waves)[: cfg.waves],
          np.asarray(env.rcs[d].cnt_from).sum(axis=0)
          - self._direct_waves(env, d),
      )

  @parameterized.named_parameters(
      ("ep16_bs8_w4", "ep16_bs8_t128_w4_zipf"),
      ("ep32_bs8_w8", "ep32_bs8_t64_w8_zipf"),
      ("ep16_bs4_w4", "ep16_bs4_t128_w4_zipf"),
      ("ep32_bs8_w4", "ep32_bs8_t128_w4_zipf"),
  )
  @pytest.mark.long
  def test_fwd_runs_overflow_fallback(self, cfg_name):
    """Forward runs over `max_runs` per (slot, wave) demote stage tokens to direct."""
    cfg = _BY_NAME[cfg_name]
    base_sched = _ptw_schedule(cfg)
    base_env = build_env(cfg, base_sched)
    bs = base_env.bs
    peak_runs = max(int(np.max(g.fwd_cnt)) for g in base_env.gws)
    self.assertGreater(peak_runs, 8)

    # `gateway_forward_runs_peak` reports uncapped run counts.
    probe_sched = dataclasses.replace(base_sched, max_runs=1)
    for gw in (0, base_env.ep - 1):
      fws = np.zeros((base_env.tables.num_slots, bs, cfg.t), dtype=np.int32)
      b = gw // bs
      for sigma in range(base_env.tables.num_slots):
        src = int(base_env.tables.served[gw, sigma])
        if src >= 0:
          fws[sigma] = base_env.fw_all[src, bs * b : bs * (b + 1)]
      got_peak = np.asarray(
          dp.gateway_forward_runs_peak(
              jnp.asarray(fws),
              gw,
              tables=base_env.tables,
              num_waves=cfg.waves,
              min_fanout=cfg.min_fanout,
              schedule=probe_sched,
          )
      )
      np.testing.assert_array_equal(got_peak, base_env.gws[gw].fwd_cnt)

    # 1. Exact fit (`max_runs = peak_runs`): bit-exact with `base_env`.
    exact_sched = dataclasses.replace(base_sched, max_runs=peak_runs)
    exact_env = build_env(cfg, exact_sched)
    self._check_views_agree(exact_env)
    for s in range(exact_env.ep):
      np.testing.assert_array_equal(
          exact_env.plans[s].stage, base_env.plans[s].stage
      )
      np.testing.assert_array_equal(
          exact_env.plans[s].fwd_cnt, base_env.plans[s].fwd_cnt
      )

    # 2. Forced overflow (`max_runs < peak_runs`): both partial slot trim
    # (`max_runs >= bs`) and full slot demotion (`max_runs < bs`) stay within
    # `max_runs`, keep all 3 views consistent, and deliver the exact exchange.
    for cap_runs in (max(8, peak_runs // 2), bs // 2):
      if cap_runs >= peak_runs:
        continue
      cap_sched = dataclasses.replace(base_sched, max_runs=cap_runs)
      cap_env = build_env(cfg, cap_sched)
      self._check_views_agree(cap_env)
      for s in range(cap_env.ep):
        self.assertTrue(np.all(cap_env.plans[s].fwd_cnt <= cap_runs))
      for g in cap_env.gws:
        self.assertTrue(np.all(g.fwd_cnt <= cap_runs))
      recv = self._check_scheduled_exchange(cap_env)
      self._check_row_on_dest(cap_env, recv)

  def _direct_waves(self, env, d):
    w = env.cfg.waves
    out = np.zeros(w, np.int64)
    for s in range(env.ep):
      out += env.plans[s].cnt_dir[d]
    return out

  def _check_scheduled_exchange(self, env):
    ep, nb, t, w, bs = env.ep, env.nb, env.cfg.t, env.cfg.waves, env.bs
    sched = env.fwd_schedule
    num_slots = env.tables.num_slots
    recv = np.full((ep, ep * t, 2), -1, np.int64)
    recv_wave = np.full((ep, ep * t), -1, np.int64)
    stage = np.full((ep, num_slots * t, 2), -1, np.int64)
    stage_w = np.full((ep, num_slots * t), -1, np.int64)
    sems = {}

    def write(buf, dev, row, payload, what):
      n = len(payload)
      self.assertTrue(
          0 <= row and row + n <= buf.shape[1], f"{what} OOB dev={dev}"
      )
      self.assertFalse(
          np.any(buf[dev, row : row + n, 0] >= 0), f"{what} rewrite dev={dev}"
      )
      buf[dev, row : row + n] = payload

    # Senders: direct runs, then one stage run per (board, wave) on its sem.
    for s in range(ep):
      p = env.plans[s]
      total = int(p.send_total)
      send = np.stack([np.full(total, s), p.perm[:total]], axis=1)
      for d in range(ep):
        for wave in range(w):
          n = int(p.cnt_dir[d, wave])
          if n:
            dst_row = int(p.dst_off[d, wave])
            src_row = int(p.send_off[d, wave])
            write(recv, d, dst_row, send[src_row : src_row + n], "direct")
            recv_wave[d, dst_row : dst_row + n] = wave
      for b in range(nb):
        if b == s // bs:
          continue
        gw, sigma = int(p.gw[b]), int(p.sigma[b])
        for wave in range(w):
          n = int(p.cnt_stage[b, wave])
          if not n:
            continue
          src0 = int(p.stage_send_off[b, wave])
          dst0 = int(p.stage_dst[b, wave])
          write(stage, gw, dst0, send[src0 : src0 + n], "stage")
          stage_w[gw, dst0 : dst0 + n] = wave
          key = (gw, sigma, wave)
          sems[key] = sems.get(key, 0) + n

    # Gateways: forward wave w, w = 0..W-1.
    read = np.zeros((ep, num_slots * t), bool)
    for gwd in range(ep):
      gp = env.gws[gwd]
      cnt = np.asarray(gp.fwd_slot_cnt)
      self.assertEqual(cnt.shape[-1], dp.FWD_SLOTS)
      waited = set()

      def issue(sigma, wave, slot, gp=gp, cnt=cnt, gwd=gwd, waited=None):
        assert waited is not None
        first = int(cnt[sigma, wave, :slot].sum())
        for i in range(first, first + int(cnt[sigma, wave, slot])):
          r, n, d_field, dr = (int(v) for v in gp.fwd_tab[sigma, wave, i])
          d = d_field & dp.FWD_DST_MASK
          tw = wave + (d_field >> dp.FWD_DST_SHIFT)
          msg = f"gw={gwd} sigma={sigma} w={wave} slot={slot} run={i}"
          self.assertGreater(n, 0, msg)
          self.assertEqual(d // bs, gwd // bs, msg)
          self.assertLessEqual(wave, tw, msg + ": issued after its wave")
          self.assertTrue(
              sigma * t <= r and r + n <= (sigma + 1) * t, msg + ": slot"
          )
          rows = slice(r, r + n)
          self.assertTrue(np.all(stage[gwd, rows, 0] >= 0), msg + ": unlanded")
          sw = stage_w[gwd, rows]
          for stage_wave in set(sw.tolist()):
            self.assertIn(
                (sigma, stage_wave), waited, msg + ": read before wait"
            )
          if slot == 0:
            self.assertTrue(np.all(sw < wave), msg)
          else:
            np.testing.assert_array_equal(sw, wave, err_msg=msg)
          src_s, toks = stage[gwd, rows, 0], stage[gwd, rows, 1]
          if sched.per_target_wave:
            np.testing.assert_array_equal(
                env.fw_all[src_s, d, toks], tw, err_msg=msg
            )
            np.testing.assert_array_equal(np.maximum(sw, tw), wave, err_msg=msg)
          else:
            self.assertEqual(tw, wave, msg)
          write(recv, d, dr, stage[gwd, rows], "forward")
          recv_wave[d, dr : dr + n] = tw
          read[gwd, rows] = True

      for wave in range(w):
        for sigma in range(num_slots):
          issue(sigma, wave, 0, waited=waited)
        for sigma in range(num_slots):
          n = int(gp.stage_rows[sigma, wave])
          if n:
            self.assertEqual(
                sems.pop((gwd, sigma, wave), 0),
                n,
                f"stage wait gw={gwd} sigma={sigma} w={wave}",
            )
            waited.add((sigma, wave))
          issue(sigma, wave, 1, waited=waited)
    self.assertEqual(sems, {}, "stage semaphores left with counts")
    np.testing.assert_array_equal(read, stage[..., 0] >= 0)

    for d in range(ep):
      valid = recv[d, :, 0] >= 0
      got = recv[d][valid]
      exp = np.argwhere(env.mask[:, :, d])
      self.assertLen(got, len(exp))
      np.testing.assert_array_equal(np.unique(got, axis=0), exp)
      np.testing.assert_array_equal(
          np.bincount(recv_wave[d][valid], minlength=w)[:w],
          env.rcs[d].recv_cnt,
          err_msg=f"per-wave arrivals d={d}",
      )
      for s in range(ep):
        slot_waves = recv_wave[d, s * t : (s + 1) * t]
        np.testing.assert_array_equal(
            np.bincount(slot_waves[slot_waves >= 0], minlength=w)[:w],
            env.plans[s].recv_cnt[d],
        )
    return recv


class DirectPlanFidelityTest(parameterized.TestCase):
  """With staging off (min_fanout = 9) the direct part is the reference plan."""

  @parameterized.named_parameters(
      ("ep16_bs8_w2", 16, 8, 64, 4, 32, 2, 0.0),
      ("ep16_bs8_w2_zipf", 16, 8, 64, 4, 32, 2, 1.2),
      ("ep16_bs4_w2_zipf", 16, 4, 64, 6, 32, 2, 1.2),
      ("ep8_bs4_w4", 8, 4, 64, 4, 16, 4, 0.0),
      ("ep32_bs8_w2_zipf", 32, 8, 64, 10, 16, 2, 1.2),
      ("ep32_bs8_w4", 32, 8, 512, 8, 16, 4, 0.0),
  )
  def test_matches_direct_dispatch_plan(
      self, ep, board_size, num_experts, top_k, t, waves, zipf
  ):
    cfg = Cfg(
        "direct",
        ep,
        board_size,
        num_experts,
        top_k,
        t,
        waves,
        zipf=zipf,
        min_fanout=9,
        seed=11,
    )
    env = build_env(cfg)
    g = num_experts // ep
    ref_fn = jax.jit(
        lambda pairs, me, wave_of: wave_dispatch.dispatch_plan(
            pairs,
            me,
            ep=ep,
            t_local=t,
            g_local=g,
            waves=waves,
            send_alloc=fused_host.align_up(
                t * min(top_k, ep), fused_host.ROWBLK
            ),
            wave_of=wave_of,
        )
    )
    for me in range(ep):
      perm_k, tab_k, _ = _np(ref_fn(env.pairs_g, jnp.int32(me), env.wave_of))
      p, rc = env.plans[me], env.rcs[me]
      np.testing.assert_array_equal(p.n_stage, 0)
      np.testing.assert_array_equal(p.send_off, tab_k[:ep])
      np.testing.assert_array_equal(p.cnt_dir, tab_k[ep : 2 * ep])
      np.testing.assert_array_equal(rc.recv_cnt, tab_k[3 * ep])
      self.assertEqual(int(p.remote_rows), int(tab_k[3 * ep + 1, 0]))
      n = int(p.send_total)
      np.testing.assert_array_equal(p.perm[:n], perm_k[:n])
      tab = np.asarray(dp.dedup_disp_tab(p, rc.recv_cnt))
      np.testing.assert_array_equal(tab[: 2 * ep], tab_k[: 2 * ep])
      np.testing.assert_array_equal(tab[3 * ep :], tab_k[3 * ep :])


class ViewInputsTest(parameterized.TestCase):
  """phase1_ref's destination-block views equal slices of the full rows."""

  @parameterized.named_parameters(
      ("ep16_bs8_w2", 16, 8, 64, 4, 16, 2, "mirror"),
      ("ep16_bs4_w2", 16, 4, 64, 4, 16, 2, "mirror"),
      ("ep32_bs8_w4", 32, 8, 512, 10, 16, 4, "mirror"),
      ("ep16_bs4_single_w2", 16, 4, 64, 4, 16, 2, "single"),
  )
  def test_views(self, ep, board_size, num_experts, top_k, t, waves, gateways):
    cfg = Cfg(
        "views",
        ep,
        board_size,
        num_experts,
        top_k,
        t,
        waves,
        seed=12,
        gateways=gateways,
    )
    env = build_env(cfg)
    tables, w, bs = env.tables, waves, board_size
    served_fn = jax.jit(
        lambda pairs, wave_of, gw: phase1_ref.served_fw(
            pairs, wave_of, gw, tables=tables, t=t, num_waves=w
        )
    )
    board_fn = jax.jit(
        lambda pairs, wave_of, me: phase1_ref.board_fw(
            pairs, wave_of, me, t=t, num_waves=w, board_size=bs
        )
    )
    block_fn = jax.jit(
        lambda pairs, wave_of, s, base: phase1_ref.phase1_inputs(
            pairs,
            wave_of,
            s,
            t=t,
            num_waves=w,
            dest_base=base,
            num_dests=bs,
        ).fw
    )
    args = (env.pairs_g, env.wave_of)
    for dev in range(env.ep):
      b = dev // bs
      got = np.asarray(served_fn(*args, jnp.int32(dev)))
      for sigma in range(tables.num_slots):
        src = int(tables.served[dev, sigma])
        if src >= 0:
          np.testing.assert_array_equal(
              got[sigma], env.fw_all[src][bs * b : bs * (b + 1)]
          )
      np.testing.assert_array_equal(
          np.asarray(board_fn(*args, jnp.int32(dev))),
          env.fw_all[:, bs * b : bs * (b + 1)],
      )
      s = (3 * dev + 1) % env.ep
      np.testing.assert_array_equal(
          np.asarray(block_fn(*args, jnp.int32(s), jnp.int32(bs * b))),
          env.fw_all[s][bs * b : bs * (b + 1)],
      )


class StructureTest(parameterized.TestCase):

  @parameterized.named_parameters(("bs8", 16, 8), ("bs4", 8, 4))
  @pytest.mark.long
  def test_all_codes_give_gray_bound_runs(self, ep, bs):
    """D5: every nonzero bs-bit code in one (b, w) group -> 2^(bs-1) runs."""
    tables = topology_lib.BoardTopology.uniform(ep, bs).tables()
    w = 2
    n_codes = (1 << bs) - 1
    t = (1 << bs) * w
    fw = np.full((ep, t), w, np.int32)
    for wave in range(w):
      for code in range(1, 1 << bs):
        for j in range(bs):
          if (code >> j) & 1:
            fw[bs + j, (1 << bs) * wave + code - 1] = wave
    p = _np(
        dp.dedup_plan(
            jnp.asarray(fw), 0, tables=tables, num_waves=w, min_fanout=1
        )
    )
    runs_bound = 1 << (bs - 1)
    np.testing.assert_array_equal(p.fwd_cnt[1], [runs_bound] * w)
    np.testing.assert_array_equal(p.cnt_stage[1], [n_codes] * w)
    self.assertLessEqual(runs_bound, dp.MAX_FWD_RUNS)
    # The runs are the reference's Gray-order runs of the group.
    rs = _ref_source(fw, 0, bs=bs, num_waves=w, min_fanout=1)
    for wave in range(w):
      self.assertLen(rs.boards[1].runs[wave], runs_bound)
      got = [
          tuple(int(v) for v in row)
          for row in p.fwd_runs[1, wave, : int(p.fwd_cnt[1, wave])]
      ]
      self.assertEqual(got, rs.boards[1].runs[wave])
    # The gateway view of the same source agrees.
    gw, sigma = int(tables.gw[0, 1]), int(tables.sigma_at_gw[0, 1])
    fws = np.full((tables.num_slots, bs, t), w, np.int32)
    fws[sigma] = fw[bs : 2 * bs]
    gp = _np(
        dp.gateway_plan(
            jnp.asarray(fws), gw, tables=tables, num_waves=w, min_fanout=1
        )
    )
    np.testing.assert_array_equal(gp.fwd_cnt[sigma], [runs_bound] * w)
    np.testing.assert_array_equal(gp.fwd_runs[sigma], p.fwd_runs[1])

  @parameterized.named_parameters(
      ("ep16_bs8_w2", 16, 8, 64, 4, 16, 2),
      ("ep16_bs4_w4", 16, 4, 64, 4, 32, 4),
      ("ep32_bs8_w4", 32, 8, 512, 10, 32, 4),
      ("ep16_bs8_t2048", 16, 8, 64, 4, 2048, 2),
  )
  def test_no_scatter_or_gather(
      self, ep, board_size, num_experts, top_k, t, waves
  ):
    """The builders are scatter- and gather-free."""
    tables = topology_lib.BoardTopology.uniform(ep, board_size).tables()
    w, bs = waves, board_size
    g = num_experts // ep
    pairs = jnp.zeros((ep * t, top_k), jnp.int32)
    rows_by_dest = jnp.ones((ep, num_experts), jnp.int32)
    me = jnp.int32(3)
    fw = jnp.zeros((ep, t), jnp.int32)
    fws = jnp.zeros((tables.num_slots, bs, t), jnp.int32)
    fwb = jnp.zeros((ep, bs, t), jnp.int32)
    wave_of = phase1_ref.wave_of_from_rows_by_dest(
        rows_by_dest, g=g, num_waves=w
    )
    ptw = dp.FwdSchedule(per_target_wave=True)

    def plan(f, s, schedule=dp.FwdSchedule()):
      return dp.dedup_plan(f, s, tables=tables, num_waves=w, schedule=schedule)

    def gw_plan(f, s, schedule=dp.FwdSchedule()):
      return dp.gateway_plan(
          f, s, tables=tables, num_waves=w, schedule=schedule
      )

    def counts(f, s, schedule=dp.FwdSchedule()):
      return dp.receiver_counts(
          f, s, tables=tables, num_waves=w, schedule=schedule
      )

    p1kw = dict(t=t, num_waves=w)
    fns = {
        "wave_of_from_rows_by_dest": (
            lambda r: phase1_ref.wave_of_from_rows_by_dest(r, g=g, num_waves=w),
            (rows_by_dest,),
        ),
        "phase1_inputs": (
            lambda p, s: phase1_ref.phase1_inputs(p, wave_of, s, **p1kw),
            (pairs, me),
        ),
        "served_fw": (
            lambda p, s: phase1_ref.served_fw(
                p, wave_of, s, tables=tables, **p1kw
            ),
            (pairs, me),
        ),
        "board_fw": (
            lambda p, s: phase1_ref.board_fw(
                p, wave_of, s, board_size=bs, **p1kw
            ),
            (pairs, me),
        ),
        "dedup_plan": (plan, (fw, me)),
        "gateway_plan": (gw_plan, (fws, me)),
        "receiver_counts": (counts, (fwb, me)),
        "pack_smem_operands+dedup_disp_tab": (
            lambda f, fs, fb, s: (
                dp.pack_smem_operands(plan(f, s), gw_plan(fs, s)),
                dp.dedup_disp_tab(plan(f, s), counts(fb, s).recv_cnt),
            ),
            (fw, fws, fwb, me),
        ),
        "per_target_wave_builders": (
            lambda f, fs, fb, s: (
                dp.pack_smem_operands(plan(f, s, ptw), gw_plan(fs, s, ptw)),
                counts(fb, s, ptw),
            ),
            (fw, fws, fwb, me),
        ),
    }
    for name, (fn, args) in fns.items():
      jaxpr = jax.make_jaxpr(fn)(*args)
      self.assertEqual(dp.forbidden_primitives(jaxpr), [], name)

  def test_forbidden_primitives_detects_gather(self):
    jaxpr = jax.make_jaxpr(lambda x, i: x[i])(jnp.zeros(8), jnp.zeros(3, int))
    self.assertNotEmpty(dp.forbidden_primitives(jaxpr))
    jaxpr = jax.make_jaxpr(
        lambda x, i, pred: jax.lax.cond(
            pred, lambda: x.at[i].set(1.0), lambda: x
        )
    )(jnp.zeros(8), jnp.int32(2), jnp.bool_(True))
    self.assertNotEmpty(dp.forbidden_primitives(jaxpr))

  def test_static_checks(self):
    tables = topology_lib.BoardTopology.uniform(16, 8).tables()
    with self.assertRaises(ValueError):
      dp.dedup_plan(jnp.zeros((8, 8), jnp.int32), 0, tables=tables, num_waves=1)
    with self.assertRaises(ValueError):
      dp.dedup_plan(
          jnp.zeros((16, 8), jnp.int32), 0, tables=tables, num_waves=2048
      )
    with self.assertRaises(ValueError):
      dp.gateway_plan(
          jnp.zeros((tables.num_slots, 4, 8), jnp.int32),
          0,
          tables=tables,
          num_waves=1,
      )
    with self.assertRaises(ValueError):
      dp.receiver_counts(
          jnp.zeros((16, 4, 8), jnp.int32), 0, tables=tables, num_waves=1
      )
    with self.assertRaises(ValueError):
      dp.FwdSchedule(max_runs=0).check(4)
    dp.FwdSchedule(per_target_wave=True).check(dp.MAX_PTW_WAVES)
    with self.assertRaises(ValueError):
      dp.FwdSchedule(per_target_wave=True).check(dp.MAX_PTW_WAVES + 1)


if __name__ == "__main__":
  absltest.main()
