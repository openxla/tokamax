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
"""CPU tests of source_plan, the sender lists and tg_assemble.

The waves are the layer's ramp waves (wave_dispatch.wave_plan).

  * source_plan(me) reproduces the reference plan (wave_dispatch.dispatch_plan)
    for me: send offsets and rows (disp_tab rows 0..2ep-1), the packed send
    order (perm), the rows each shard receives per wave (row 3ep, recomputed
    by receiver_counts) and the remote total (row 3ep + 1). Only the
    destination offsets change (fixed per-source slots).
  * The fixed-slot exchange the plans describe, replayed in numpy, delivers
    every needed (token, destination) row exactly once, inside its source's
    slot, at row_on_dest, in the token's first-hit wave.
  * The lists carry, behind their header, the receive row of every pair in
    the destination's slab order (and the fp8 scale bits as a second plane).
  * tg_assemble (interpret mode, both impls, one and two words) builds the
    same table as the reference token-gather scatter run with the fixed-slot
    receive rows, and every live slab row names the same token as the table
    built from the reference plan's receive rows does.
"""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from jax import lax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.dispatch import source_plan as sp
from tokamax._src.ops.experimental.dedup_moe.dispatch import tg_assemble
from tokamax._src.ops.experimental.dedup_moe.dispatch import wave_dispatch
from tokamax._src.ops.experimental.dedup_moe.test.common import hw_util
from tokamax._src.ops.experimental.dedup_moe.test.common import moe_inputs
from tokamax._src.ops.experimental.fused_moe import host

ROWBLK = host.ROWBLK
align_up = host.align_up

# wave_dispatch_test's plan shapes, plus a t above RANK_FULL_MAX (blocked
# ranks) and a routing with repeated experts per token.
_PLAN_SHAPES = (
    dict(testcase_name="ep4_w1", ep=4, t_local=16, g_local=4, topk=3, waves=1),
    dict(testcase_name="ep4_w2", ep=4, t_local=16, g_local=4, topk=3, waves=2),
    dict(testcase_name="ep4_w4", ep=4, t_local=16, g_local=4, topk=3, waves=4),
    dict(testcase_name="ep8_w4", ep=8, t_local=16, g_local=4, topk=3, waves=4),
    dict(
        testcase_name="ep2_topk4", ep=2, t_local=24, g_local=8, topk=4, waves=4
    ),
    dict(
        testcase_name="ep2_t1536",
        ep=2,
        t_local=1536,
        g_local=4,
        topk=2,
        waves=2,
    ),
)
_ROUTINGS = ("uniform", "skewed", "worst", "dominant", "dup")


def setUpModule():
  hw_util.setup_module("tpu8i")


def _expert_ids(routing, tokens, e_total, topk, g_local, seed):
  if routing != "dup":
    return moe_inputs.expert_ids(routing, tokens, e_total, topk, g_local, seed)
  # Hash-routed ids may repeat an expert within a token: every repeat is a
  # pair of its own, so one token can put topk entries on one list.
  rng = np.random.default_rng(seed)
  ids = rng.integers(0, e_total, (tokens, topk)).astype(np.int32)
  ids[::3] = ids[::3, :1]
  return ids


def _cases(shapes=_PLAN_SHAPES, routings=_ROUTINGS):
  for shape, routing in itertools.product(shapes, routings):
    case = dict(shape)
    case["testcase_name"] = f"{shape['testcase_name']}_{routing}"
    case.update(routing=routing)
    yield case


def _token_gather(slab_row, posmap, *, topk, ragged_stride):
  """The reference token-gather table for receive rows `posmap`.

  Scatters each routed pair's receive row onto the slab row it computes on
  (pairs of other shards fall outside the slab and are dropped).

  Args:
    slab_row: [T * K] slab row of every pair on this shard.
    posmap: [T] receive row of every token.
    topk: K.
    ragged_stride: Slab rows.

  Returns:
    [ragged_stride] int32 token-gather table.
  """
  n_tokens = posmap.shape[0]
  return jnp.clip(
      jnp.zeros((ragged_stride,), jnp.int32)
      .at[slab_row]
      .add(jnp.repeat(posmap, topk), mode="drop"),
      0,
      n_tokens - 1,
  )


class _Setup:
  """Routing, wave layout and every source's plan for one case."""

  def __init__(self, routing, ep, t_local, g_local, topk, waves):
    self.ep, self.t, self.g, self.topk, self.waves = (
        ep,
        t_local,
        g_local,
        topk,
        waves,
    )
    self.T = ep * t_local  # pylint: disable=invalid-name
    self.E = ep * g_local  # pylint: disable=invalid-name
    self.ids = _expert_ids(
        routing, self.T, self.E, topk, g_local, seed=ep + waves + t_local
    )
    hist = jnp.asarray(
        np.bincount(self.ids.reshape(-1), minlength=self.E), jnp.int32
    )
    _, self.wave_of, _ = wave_dispatch.wave_plan(
        hist, ep=ep, g_local=g_local, waves=waves
    )
    self.kw = dict(
        ep=ep,
        t_local=t_local,
        g_local=g_local,
        waves=waves,
        wave_of=self.wave_of,
    )
    self.send_alloc = align_up(t_local * min(topk, ep), ROWBLK)
    pairs = jnp.asarray(self.ids)
    self.plans = [
        sp.source_plan(pairs, jnp.int32(s), **self.kw) for s in range(ep)
    ]
    self.hist_by_src = np.stack([
        np.bincount(
            self.ids[s * t_local : (s + 1) * t_local].reshape(-1),
            minlength=self.E,
        )
        for s in range(ep)
    ]).astype(np.int32)

  def np_plan(self, s):
    return {k: np.asarray(v) for k, v in self.plans[s]._asdict().items()}

  def owed_wave(self, e):
    d, local = divmod(int(e), self.g)
    return int(self.wave_of[d, local])


class SourcePlanTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_supported(self)
    # Plan logic runs on the host CPU.
    self.enter_context(hw_util.host_only())

  @parameterized.named_parameters(*_cases())
  def test_source_plan_reproduces_the_reference_dispatch_plan(
      self, routing, ep, t_local, g_local, topk, waves
  ):
    c = _Setup(routing, ep, t_local, g_local, topk, waves)
    pairs = jnp.asarray(c.ids)
    for me in range(ep):
      perm, tab, _ = (
          np.asarray(a)
          for a in wave_dispatch.dispatch_plan(
              pairs, jnp.int32(me), send_alloc=c.send_alloc, **c.kw
          )
      )
      plan = c.np_plan(me)
      np.testing.assert_array_equal(plan["send_off"], tab[:ep])
      np.testing.assert_array_equal(plan["send_cnt"], tab[ep : 2 * ep])
      total = int(plan["send_cnt"].sum())
      np.testing.assert_array_equal(
          np.asarray(sp.send_perm(c.plans[me], send_alloc=c.send_alloc))[
              :total
          ],
          perm[:total],
      )
      recv = sp.receiver_counts(pairs, jnp.int32(me), **c.kw)
      np.testing.assert_array_equal(np.asarray(recv), tab[3 * ep])
      ours = np.asarray(
          sp.dispatch_table(c.plans[me], jnp.int32(me), recv, t_local=t_local)
      )
      self.assertEqual(ours.shape, tab.shape)
      np.testing.assert_array_equal(ours[: 2 * ep], tab[: 2 * ep])
      np.testing.assert_array_equal(ours[3 * ep :], tab[3 * ep :])
      np.testing.assert_array_equal(
          ours[2 * ep : 3 * ep], me * t_local + plan["slot_off"]
      )
      self.assertEqual(
          int(sp.remote_rows(c.plans[me], jnp.int32(me))), tab[3 * ep + 1, 0]
      )

  @parameterized.named_parameters(*_cases())
  def test_fixed_slots_deliver_every_needed_row_once_by_its_wave(
      self, routing, ep, t_local, g_local, topk, waves
  ):
    c = _Setup(routing, ep, t_local, g_local, topk, waves)
    pairs = jnp.asarray(c.ids)
    T = c.T  # pylint: disable=invalid-name
    tabs, perms = [], []
    for s in range(ep):
      recv = sp.receiver_counts(pairs, jnp.int32(s), **c.kw)
      tabs.append(
          np.asarray(
              sp.dispatch_table(c.plans[s], jnp.int32(s), recv, t_local=t_local)
          )
      )
      perms.append(
          np.asarray(sp.send_perm(c.plans[s], send_alloc=c.send_alloc))
      )
    landed = np.full((ep, T), -1)
    landed_wave = np.full((ep, T), -1)
    for src in range(ep):
      self._land_source(c, src, tabs[src], perms[src], landed, landed_wave)
    dest = c.ids // g_local
    for d in range(ep):
      for w in range(waves):
        self.assertEqual(tabs[d][3 * ep, w], sum(tb[ep + d, w] for tb in tabs))
      needed = set(np.flatnonzero((dest == d).any(axis=1)).tolist())
      self.assertLen(needed, int((landed[d] >= 0).sum()))
      for t in range(T):
        self._check_token(c, d, t, t in needed, landed, landed_wave)

  def _land_source(self, c, src, tab, perm, landed, landed_wave):
    """Lands source `src`'s runs per its dispatch table; checks the slots."""
    ep, t_local = c.ep, c.t
    remote = 0
    for d in range(ep):
      for w in range(c.waves):
        off, n, dst = tab[d, w], tab[ep + d, w], tab[2 * ep + d, w]
        self.assertLessEqual(off + n, c.send_alloc)
        # Inside the source's own slot on d.
        self.assertGreaterEqual(dst, src * t_local)
        self.assertLessEqual(dst + n, (src + 1) * t_local)
        self.assertTrue((landed[d, dst : dst + n] == -1).all(), "overlap")
        landed[d, dst : dst + n] = src * t_local + perm[off : off + n]
        landed_wave[d, dst : dst + n] = w
        remote += n if d != src else 0
    self.assertEqual(tab[3 * ep + 1, 0], remote)

  def _check_token(self, c, d, t, needed, landed, landed_wave):
    """Token t landed on d exactly where and in the wave the plan says."""
    s, tl = divmod(t, c.t)
    plan = c.np_plan(s)
    if not needed:
      self.assertFalse(plan["need"][d, tl])
      self.assertEqual(plan["row_on_dest"][d, tl], 0)
      return
    row = plan["row_on_dest"][d, tl]
    self.assertEqual(landed[d, row], t)
    first = min(c.owed_wave(e) for e in c.ids[t] if e // c.g == d)
    self.assertEqual(landed_wave[d, row], first)
    self.assertEqual(plan["fw"][d, tl], first)

  @parameterized.parameters(8, 16, 1024, 1032, 1536, 4096)
  def test_exclusive_ranks(self, t):
    rng = np.random.default_rng(t)
    x = (rng.random((5, t)) < 0.3).astype(np.int32)
    want = np.cumsum(x, axis=1) - x
    np.testing.assert_array_equal(
        np.asarray(sp.exclusive_ranks(jnp.asarray(x))), want
    )


_LIST_SHAPES = (
    dict(testcase_name="ep4_w2", ep=4, t_local=16, g_local=4, topk=3, waves=2),
    dict(testcase_name="ep8_w4", ep=8, t_local=16, g_local=4, topk=4, waves=4),
    # Lists longer than one 128-word row; the busiest list is t * K entries.
    dict(
        testcase_name="ep2_t256", ep=2, t_local=256, g_local=2, topk=2, waves=2
    ),
)


def _list_cases():
  for case in _cases(_LIST_SHAPES, ("uniform", "worst", "dup")):
    for nw in (1, 2):
      out = dict(case)
      out["testcase_name"] += f"_nw{nw}"
      out["nw"] = nw
      yield out


def _scale_bits(t_local, s):
  return (
      np.random.default_rng(100 + s)
      .integers(-(2**31), 2**31 - 1, t_local, dtype=np.int64)
      .astype(np.int32)
  )


def _send_lists(c, s, nw):
  bits = jnp.asarray(_scale_bits(c.t, s)) if nw == 2 else None
  return np.asarray(
      sp.build_send_lists(
          jnp.asarray(c.ids),
          c.plans[s],
          jnp.int32(s),
          jnp.asarray(c.hist_by_src[s]),
          ep=c.ep,
          t_local=c.t,
          g_local=c.g,
          scale_bits=bits,
      )
  )


class SendListsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_supported(self)
    self.enter_context(hw_util.host_only())

  @parameterized.named_parameters(*_list_cases())
  def test_lists_carry_slab_ordered_receive_rows(
      self, routing, ep, t_local, g_local, topk, waves, nw
  ):
    c = _Setup(routing, ep, t_local, g_local, topk, waves)
    lmax = sp.list_capacity(t_local, topk)
    words = sp.list_row_words(t_local, topk, nw)
    for s in range(ep):
      lists = _send_lists(c, s, nw)
      self.assertEqual(lists.shape, (ep * words // 128, 128))
      blocks = lists.reshape(ep, words)
      plan = c.np_plan(s)
      bits = _scale_bits(t_local, s)
      mine = c.ids[s * t_local : (s + 1) * t_local]
      for d in range(ep):
        rows, sbits = _expected_entries(plan, bits, mine, d, g_local)
        n = len(rows)
        self.assertLessEqual(n, lmax)
        hdr, body = blocks[d, :128], blocks[d, 128:]
        want_hdr = np.zeros(128, np.int64)
        want_hdr[sp.HDR_N] = n
        want_hdr[sp.HDR_SRC] = s
        want_hdr[sp.HDR_WAVES] = waves
        want_hdr[sp.HDR_NW] = nw
        want_hdr[sp.HDR_SEND_OFF : sp.HDR_SEND_OFF + waves] = plan["send_off"][
            d
        ]
        want_hdr[sp.HDR_CNT_DIR : sp.HDR_CNT_DIR + waves] = plan["cnt_dir"][d]
        np.testing.assert_array_equal(hdr, want_hdr)
        np.testing.assert_array_equal(body[:n], rows)
        stride = align_up(n, 128)
        np.testing.assert_array_equal(body[n:stride], 0)
        if nw == 2:
          np.testing.assert_array_equal(body[stride : stride + n], sbits)
          np.testing.assert_array_equal(body[stride + n :], 0)
        else:
          np.testing.assert_array_equal(body[stride:], 0)
        self.assertEqual(
            int(sp.list_dma_rows(jnp.int32(n), nw)), 1 + nw * stride // 128
        )


def _expected_entries(plan, bits, mine, d, g_local):
  """A list's (receive rows, scale bits): local expert, then pair order."""
  rows, sbits = [], []
  t_local, topk = mine.shape
  for gl in range(g_local):
    hit_tokens = [
        tl
        for tl, k in itertools.product(range(t_local), range(topk))
        if mine[tl, k] == d * g_local + gl
    ]
    rows += [plan["row_on_dest"][d, tl] for tl in hit_tokens]
    sbits += [bits[tl] for tl in hit_tokens]
  return rows, sbits


_GARBAGE = 0x7EADBEEF


def _landed(c, me, nw):
  """tg_land on `me` after every source's list DMA.

  Rows past what a source sends hold garbage, which tg_assemble must never
  read.

  Args:
    c: The case setup.
    me: The receiving shard.
    nw: Planes per entry.

  Returns:
    [ep * list_rows, 128] int32.
  """
  rows = sp.list_rows(c.t, c.topk, nw)
  land = np.full((c.ep * rows, 128), _GARBAGE, np.int32)
  for s in range(c.ep):
    blk = _send_lists(c, s, nw).reshape(c.ep, rows, 128)[me]
    n = int(c.hist_by_src[s].reshape(c.ep, c.g)[me].sum())
    m = 1 + nw * align_up(n, 128) // 128
    land[s * rows : s * rows + m] = blk[:m]
  return land


def _routing_tables(c, me, *, capacity, stride):
  K = c.topk  # pylint: disable=invalid-name
  block = host.routing_block(c.t, K)
  expert_blocks = jnp.asarray(c.ids.reshape(-1, block))
  per_src = c.t * K // block
  local_blocks = expert_blocks[me * per_src : (me + 1) * per_src]
  gathered = host.GatheredPairs(
      expert_blocks=expert_blocks,
      local_blocks=local_blocks,
      block_hist=host.pair_block_hist(local_blocks, c.E),
      rows_by_dest=jnp.asarray(c.hist_by_src),
      n_tokens=c.T,
      topk=K,
  )
  return (
      host.build_routing_tables_sharded(
          None,
          jnp.int32(me),
          e_total=c.E,
          ep=c.ep,
          t_local=c.t,
          block=block,
          tile_m=capacity,
          shard_stride=stride,
          gathered=gathered,
      ),
      gathered,
  )


_TG_SHAPES = (
    dict(testcase_name="ep8_t16", ep=8, t_local=16, g_local=4, topk=4, waves=4),
    dict(testcase_name="ep4_t64", ep=4, t_local=64, g_local=4, topk=3, waves=2),
    # Runs spanning several 8-row tiles.
    dict(
        testcase_name="ep2_t512", ep=2, t_local=512, g_local=2, topk=4, waves=2
    ),
)


def _tg_cases():
  for case in _cases(_TG_SHAPES, ("uniform", "skewed", "worst", "dup")):
    for impl, nw in itertools.product(tg_assemble.IMPLS, (1, 2)):
      if impl == "smem" and case["t_local"] > 64:
        continue
      out = dict(case)
      out["testcase_name"] += f"_{impl}_nw{nw}"
      out.update(impl=impl, nw=nw)
      yield out


class TgAssembleTest(parameterized.TestCase):
  """tg_assemble on one device in CPU interpret mode."""

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_supported(self)
    hw_util.reset_interpret_state()

  @parameterized.named_parameters(*_tg_cases())
  def test_tg_assemble_matches_the_scatter_and_names_the_reference_tokens(
      self, routing, ep, t_local, g_local, topk, waves, impl, nw
  ):
    c = _Setup(routing, ep, t_local, g_local, topk, waves)
    T = c.T  # pylint: disable=invalid-name
    capacity = 32
    stride = host.ragged_stride_bound(T, topk, c.E, capacity)
    tg_rows = tg_assemble.gather_rows(stride, capacity)
    window = host.token_gather_window_rows(capacity)
    scale_rows = host.act_scale_slab_rows(stride) if nw == 2 else None
    # The fixed-slot receive row of every global token on each shard.
    posmap_slot = np.stack([
        np.concatenate([c.np_plan(s)["row_on_dest"][d] for s in range(ep)])
        for d in range(ep)
    ])
    all_bits = np.concatenate([_scale_bits(t_local, s) for s in range(ep)])
    for me in sorted({0, ep - 1}):
      routing_t, gathered = _routing_tables(
          c, me, capacity=capacity, stride=stride
      )
      _, slab_base = host.shard_expert_slabs(
          routing_t, jnp.int32(me), e_total=c.E, ep=ep
      )
      run_start = lax.dynamic_slice(
          routing_t.run_start_aligned, (me * g_local, 0), (g_local, ep)
      )
      runs, meta = tg_assemble.assemble_runs(
          gathered.rows_by_dest,
          jnp.int32(me),
          ep=ep,
          g_local=g_local,
          t_local=t_local,
          topk=topk,
          nw=nw,
          slab_base=slab_base,
          run_start=run_start,
      )
      land = _landed(c, me, nw)
      with hw_util.interpret_pallas():
        tg, scale = tg_assemble.tg_assemble(
            jnp.asarray(land),
            runs,
            meta,
            None,
            ep=ep,
            nw=nw,
            n_tokens=T,
            tg_rows=tg_rows,
            scale_rows=scale_rows,
            impl=impl,
        )
        tg = np.asarray(tg)
        scale = None if scale is None else np.asarray(scale)
      self.assertEqual(tg.shape, (tg_rows * 128,))
      slab_row = host.local_slab_rows(
          routing_t, jnp.int32(me), shard_stride=stride
      )
      want = np.asarray(
          _token_gather(
              slab_row,
              jnp.asarray(posmap_slot[me]),
              topk=topk,
              ragged_stride=stride,
          )
      )
      np.testing.assert_array_equal(tg[:stride], want)
      np.testing.assert_array_equal(tg[stride:], 0)
      self.assertGreaterEqual(tg.size, stride + window)
      # Same token on every live slab row as the reference plan's own table
      # (its receive rows mapped back through its own posmap).
      _, _, posmap_ref = wave_dispatch.dispatch_plan(
          jnp.asarray(c.ids),
          jnp.int32(me),
          send_alloc=c.send_alloc,
          **c.kw,
      )
      posmap_ref = np.asarray(posmap_ref)
      tg_ref = np.asarray(
          _token_gather(
              slab_row,
              jnp.asarray(posmap_ref),
              topk=topk,
              ragged_stride=stride,
          )
      )
      on_me = (c.ids // g_local) == me
      needed = np.flatnonzero(on_me.any(axis=1))
      token_of_ref_row = {int(posmap_ref[t]): int(t) for t in needed}
      token_of_slot_row = {int(posmap_slot[me][t]): int(t) for t in needed}
      live = np.asarray(slab_row)[on_me.reshape(-1)]
      self.assertEqual(live.size, int(on_me.sum()))
      for r in live:
        self.assertEqual(
            token_of_slot_row[int(tg[r])], token_of_ref_row[int(tg_ref[r])]
        )
      dead = np.setdiff1d(np.arange(stride), live)
      np.testing.assert_array_equal(tg[dead], 0)
      if nw == 2:
        assert scale is not None and scale_rows is not None
        ref_scale = np.asarray(
            jnp.zeros((scale_rows * 128,), jnp.int32)
            .at[slab_row]
            .add(jnp.asarray(np.repeat(all_bits, topk)), mode="drop")
        )
        self.assertEqual(scale.shape, (scale_rows, 128))
        flat = scale.reshape(-1)
        # The reference scatter also piles other shards' pairs onto element
        # `stride`, which nothing reads; the slab itself must match.
        np.testing.assert_array_equal(flat[:stride], ref_scale[:stride])
        np.testing.assert_array_equal(flat[stride:], 0)


if __name__ == "__main__":
  absltest.main()
