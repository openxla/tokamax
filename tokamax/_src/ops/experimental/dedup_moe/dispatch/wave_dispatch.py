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
"""Wave dispatch: the token exchange of the De-Dup MoE layer, in expert waves.

Instead of an activation all-gather, which delivers every token to every
shard, each shard sends every (token, destination shard) pair with a hit
once, through the split-phase Pallas ragged all-to-all in
tc_sender.py.

Every shard orders its local experts fewest rows first, from the per-expert
counts the routing exchange already all-gathers, so senders and receivers
agree on the order without exchanging it, and cuts that order into W groups
("waves") whose modeled compute grows 1 : 2 : 4 : ... (wave_plan). The
exchange is laid out and issued wave by wave, and the fused kernel waits for
an expert's wave only before that expert's first tile: the smallest experts
start soonest and the rest of the exchange lands under their matmuls. The
routing tables are built while the exchange is in flight.

With De-Dup (dedup_tables), a token needed by several devices of a remote
board is sent once to that board's gateway, which forwards it inside the
fused kernel (dedup_plan).

Every buffer is sized for the worst case (all tokens on one shard), so no
routing drops a token.
"""

from typing import NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.dispatch import source_plan
from tokamax._src.ops.experimental.dedup_moe.dispatch import tc_sender as dispatch_kernels
from tokamax._src.ops.experimental.fused_moe import host as fused_host

ROWBLK = fused_host.ROWBLK
align_up = fused_host.align_up

# Rows in the first copy of each destination's share of a wave; the rest of
# the share follows in one copy (tc_sender._start_waves_kernel).
CHUNK_ROWS = 8


class WaveDispatch(NamedTuple):
  """An exchange in flight and the plan it was laid out by.

  fut: tc_sender.ragged_a2a_start_waves's (send_sem, recv_sems,
    send, recv, list_send_sem, tg_sem, tg_lists, tg_land[, stage_sems,
    stage]). The fused kernel reads its rows from `recv` and waits on
    `recv_sems`; `tg_assemble` reads `tg_land` and waits on `tg_sem`;
    `finish` drains `send_sem` and `list_send_sem`.
  disp_tab [4 * ep, waves] i32: the dispatch plan (see tc_sender);
    row wave_bounds_row(ep) carries this shard's wave cuts.
  visit_order [ep, g_local] i32: every shard's local experts in visit
    order (waves ascending, fewest rows first), empty experts last.
  list_tab [ep + 1] i32: token-gather list DMA row counts per destination
    plus the remote total at index `ep`.
  stage_send_tab [5 * nb, waves] i32 | None: the sender's De-Dup stage DMA
    table.
  smem_ops SmemOperands | None: the gateway's stage_tab, fwd_tab and
    fwd_cnt for the fused kernel (De-Dup; materialize_smem_ops).
  dplan SourcePlan | None: the sender's De-Dup dispatch plan arrays.
  fwd_schedule FwdSchedule | None: the De-Dup forward schedule.
  wave_of [ep, g_local] i32: the wave of every shard's local experts.
  """

  fut: tuple[jax.Array, ...]
  disp_tab: jax.Array
  visit_order: jax.Array
  list_tab: jax.Array | None = None
  stage_send_tab: jax.Array | None = None
  smem_ops: dedup_plan_mod.SmemOperands | None = None
  dplan: dedup_plan_mod.SourcePlan | None = None
  fwd_schedule: dedup_plan_mod.FwdSchedule | None = None
  wave_of: jax.Array | None = None

  @property
  def send_buf(self):
    return self.fut[2]

  @property
  def tg_sem(self):
    return self.fut[5] if len(self.fut) > 4 else None

  @property
  def tg_land(self):
    return self.fut[7] if len(self.fut) > 4 else None

  @property
  def stage_sems(self):
    return self.fut[8] if len(self.fut) > 8 else None

  @property
  def stage(self):
    return self.fut[9] if len(self.fut) > 8 else None


def ascending_expert_order(hist, *, ep, g_local):
  """Orders each shard's local experts fewest rows first, empties last.

  Args:
    hist: [ep * g_local] i32 rows per global expert, identical on every shard.
    ep: Expert-parallel width.
    g_local: Experts per shard.

  Returns:
    (pos, order): pos [ep, g_local] i32 is the visit position of local expert e
    on shard d; order [ep, g_local] i32 is the local expert at visit position p,
    so order[me] is a drop-in visit list. Ties break by ascending index.
  """
  h = hist.reshape(ep, g_local).astype(jnp.int32)
  idx = jnp.arange(g_local, dtype=jnp.int32)
  # 2**24 > any row count; 2**24 * g_local + g_local stays below 2**31.
  key = jnp.where(h > 0, jnp.minimum(h, 2**24 - 1), 2**24) * g_local + idx
  pos = jnp.sum((key[:, None, :] < key[:, :, None]).astype(jnp.int32), axis=2)
  order = jnp.sum(
      jnp.where(pos[:, None, :] == idx[None, :, None], idx[None, None, :], 0),
      axis=2,
  )
  return pos, order


# Ramp waves: each shard's experts, fewest rows first, cut into W waves of
# VARIABLE size so the waves' modeled compute grows RAMP_RATIO-fold per wave
# (1 : 2 : 4 : ...), at least one expert per wave: a small first wave is a
# short pipeline fill. The fused kernel reads each wave's end position from
# row `wave_bounds_row(ep)` of disp_tab. An expert's modeled compute is
# tau + c * rows (a fixed cost per visited expert plus a cost per row),
# here in units of c as the integer EXPERT_COST_ROWS + rows, so every shard
# derives the same waves bit for bit (the cost model's ramp wave
# assignment). The waves change
# only when rows land and which expert is computed when, never what is
# computed.
RAMP_RATIO = 2
EXPERT_COST_ROWS = 129  # round(tau / c) of the fitted cost model.


def wave_bounds_row(ep):
  """Returns the disp_tab row that carries the wave bounds.

  The disp_tab row that carries the wave bounds.

  disp_tab rows 3 * ep + 2 .. 4 * ep - 1 are zero padding
  (source_plan.dispatch_table, dedup_plan.dedup_disp_tab), so the bounds
  need no kernel operand of their own; hence ep >= 3.

  Args:
    ep: Expert-parallel width (>= 3).

  Returns:
    3 * ep + 2, a zero-padding row of the dispatch table.
  """
  return 3 * ep + 2


def _rank_by(keys, *, g_local):
  """Per-shard rank of each local expert under lexicographic `keys`.

  Ties break by ascending expert index. A compare-reduce over [ep, g, g]
  rather than a sort.

  Args:
    keys: [ep, g_local] i32 arrays, most significant first.
    g_local: Experts per shard.

  Returns:
    (pos, order) like ascending_expert_order.
  """
  idx = jnp.arange(g_local, dtype=jnp.int32)
  # before[d, e, e2]: e2 precedes e.
  before = idx[None, None, :] < idx[None, :, None]
  for k in reversed(keys):
    a, b = k[:, None, :], k[:, :, None]
    before = (a < b) | ((a == b) & before)
  pos = jnp.sum(before.astype(jnp.int32), axis=2)
  order = jnp.sum(
      jnp.where(pos[:, None, :] == idx[None, :, None], idx[None, None, :], 0),
      axis=2,
  )
  return pos, order


def wave_plan(
    hist,
    *,
    ep,
    g_local,
    waves,
    ramp_ratio=RAMP_RATIO,
    expert_cost_rows=EXPERT_COST_ROWS,
):
  """Computes every shard's visit order, expert waves and wave bounds (ramp waves).

  Args:
    hist: [ep * g_local] i32 rows per global expert, identical on every shard.
    ep: Expert-parallel width.
    g_local: Experts per shard.
    waves: W.
    ramp_ratio: Growth of each wave's modeled compute.
    expert_cost_rows: Fixed per-expert cost, in rows.

  Returns:
    (visit, wave_of, bounds): visit [ep, g_local] i32 local expert at each visit
    position (waves contiguous and ascending, fewest rows first inside a wave,
    empty experts last); wave_of [ep, g_local] i32 each local expert's wave (W -
    1 for empty experts); bounds [ep, W] i32 nonempty experts in waves 0..w.
    With at most W nonempty experts, each gets its own wave.
  """
  h = hist.reshape(ep, g_local).astype(jnp.int32)
  nonempty = h > 0
  m = jnp.sum(nonempty.astype(jnp.int32), axis=1)  # [ep]
  pos, asc = ascending_expert_order(hist, ep=ep, g_local=g_local)
  w_iota = jnp.arange(waves, dtype=jnp.int32)
  g_iota = jnp.arange(g_local, dtype=jnp.int32)
  cost = jnp.where(nonempty, expert_cost_rows + h, 0)  # [ep, g] i32
  cost_asc = jnp.take_along_axis(cost, asc, axis=1)
  cs = jnp.cumsum(cost_asc, axis=1)  # cs[:, c - 1]: the first c experts
  total = cs[:, -1]
  share = [ramp_ratio**w for w in range(waves)]
  share_sum = sum(share)
  cum_share = [sum(share[: w + 1]) for w in range(waves)]
  big = jnp.int32(2**30)
  cand = g_iota + 1  # experts in waves 0..w
  prev = jnp.zeros((ep,), jnp.int32)
  cuts = []
  for w in range(waves - 1):
    hi = m - (waves - 1 - w)
    valid = (cand[None, :] >= prev[:, None] + 1) & (
        cand[None, :] <= hi[:, None]
    )
    # |cs * S - total * cum_share[w]| = S * |cs - target_w|, exact.
    err = jnp.abs(cs * share_sum - (total * cum_share[w])[:, None])
    c = jnp.argmin(jnp.where(valid, err, big), axis=1).astype(jnp.int32) + 1
    cuts.append(c)
    prev = c
  if cuts:
    cut = jnp.stack(cuts, axis=1)  # [ep, W - 1]
    wave_asc = jnp.sum(
        (cut[:, None, :] <= g_iota[None, :, None]).astype(jnp.int32), axis=2
    )
  else:
    wave_asc = jnp.zeros((ep, g_local), jnp.int32)
  wave_of = jnp.take_along_axis(wave_asc, pos, axis=1)
  few = (m <= waves)[:, None]
  wave_of = jnp.where(few, jnp.minimum(pos, waves - 1), wave_of)
  wave_of = jnp.where(nonempty, wave_of, waves - 1).astype(jnp.int32)
  _, visit = _rank_by([jnp.where(nonempty, wave_of, waves), h], g_local=g_local)
  cnt = jnp.sum(
      (
          (wave_of[:, :, None] == w_iota[None, None, :]) & nonempty[:, :, None]
      ).astype(jnp.int32),
      axis=1,
  )
  return visit, wave_of, jnp.cumsum(cnt, axis=1).astype(jnp.int32)


def dispatch_plan(
    pairs_g,
    me,
    *,
    ep,
    t_local,
    g_local,
    waves,
    send_alloc,
    wave_of,
):
  """Reference dispatch plan for shard `me` (tests only).

  The original dense wave plan, kept as the oracle source_plan is checked
  against (source_plan_test, dedup_plan_test); the layer builds its plan
  with source_plan.

  Every (token, dest) pair with a hit is sent ONCE, in its first-hit wave.
  The receive buffer on dest d is laid out by (wave, source), tokens
  ascending within each segment.

  Args:
    pairs_g: [n_tokens, topk] i32 global expert ids in mesh order (dest = id //
      g_local).
    me: This shard's index.
    ep: Expert-parallel width.
    t_local: Tokens per shard.
    g_local: Experts per shard.
    waves: W.
    send_alloc: Rows of the send buffer.
    wave_of: [ep, g_local] i32 wave of each shard's local experts.

  Returns:
    (perm [send_alloc] i32: local token row of each packed send row;
    tab [4 * ep, waves] i32: send offset / send rows / dest offset per
    (d, w), rows received per wave (row 3 ep), total rows sent to other
    shards (row 3 ep + 1, col 0); posmap [n_tokens] i32: receive-buffer row of
    global token t on `me`, 0 where `me` does not need t).
  """
  n_tokens, topk = pairs_g.shape
  # The token axis stays minor in every array below. With topk minor, a
  # [n_tokens, topk] int32 array fills topk of a vreg's 128 lanes, and the
  # relayouts and elementwise ops on it cost more than the plan's arithmetic.
  pk = pairs_g.T.astype(jnp.int32)  # [topk, n_tokens]
  dest = pk // g_local
  table = wave_of.astype(jnp.int32)  # [ep, g_local]
  # The wave of every pair is table[dest, local expert]. Not jnp.take (an
  # XLA TPU gather over every pair is slow), nor a compare over all E
  # experts. A one-hot of the local expert
  # times the table on the MXU gives each pair its wave on every shard, and
  # a compare over the ep shards keeps its own. The entries are below
  # 2 * waves, so bf16 holds them exactly.
  g_iota = jnp.arange(g_local, dtype=jnp.int32)
  local = (pk - dest * g_local).reshape(1, topk * n_tokens)
  onehot = (local == g_iota[:, None]).astype(
      jnp.bfloat16
  )  # [g_local, topk*n_tokens]
  by_dest = jnp.dot(
      table.astype(jnp.bfloat16), onehot, preferred_element_type=jnp.float32
  )  # [ep, topk*n_tokens]
  d_iota = jnp.arange(ep, dtype=jnp.int32)
  wave = (
      jnp.sum(
          jnp.where(
              dest.reshape(1, topk * n_tokens) == d_iota[:, None], by_dest, 0.0
          ),
          axis=0,
      )
      .astype(jnp.int32)
      .reshape(topk, n_tokens)
  )
  # First-hit wave of each token on each shard; `waves` = no hit.
  fw = jnp.min(
      jnp.where(
          dest[:, None, :] == d_iota[None, :, None], wave[:, None, :], waves
      ),
      axis=0,
  )  # [dest, n_tokens]
  w_iota = jnp.arange(waves, dtype=jnp.int32)
  oh = (
      (fw[:, None, :] == w_iota[None, :, None])
      .astype(jnp.int32)
      .reshape(ep, waves, ep, t_local)
  )  # [dest, w, src, t]
  cnt = oh.sum(axis=3)  # [dest, w, src]
  # Receive layout on every dest: segments in (w, src) order.
  seg = cnt.reshape(ep, waves * ep)
  base = (jnp.cumsum(seg, axis=1) - seg).reshape(ep, waves, ep)  # [d, w, s]
  dst_off = lax.dynamic_index_in_dim(base, me, axis=2, keepdims=False)  # [d, w]
  send_cnt = lax.dynamic_index_in_dim(cnt, me, axis=2, keepdims=False)  # [d, w]
  flat = send_cnt.reshape(-1)
  send_off = (jnp.cumsum(flat) - flat).reshape(ep, waves)
  recv_cnt = lax.dynamic_index_in_dim(cnt, me, axis=0, keepdims=False).sum(1)
  remote = (
      send_cnt.sum()
      - lax.dynamic_index_in_dim(send_cnt, me, axis=0, keepdims=False).sum()
  )
  col = jnp.arange(waves, dtype=jnp.int32)
  extra = jnp.concatenate(
      [
          recv_cnt.astype(jnp.int32)[None, :],
          jnp.where(col == 0, remote, 0)[None, :].astype(jnp.int32),
          jnp.zeros((ep - 2, waves), jnp.int32),
      ],
      axis=0,
  )
  tab = jnp.concatenate([send_off, send_cnt, dst_off, extra], axis=0)
  # Exclusive prefix counts along the token axis as a matmul with a
  # strictly upper triangular ones matrix (0/1 in bf16, exact f32 sums),
  # which is faster than XLA's reduce-window cumsum on TPU.
  tri = jnp.triu(jnp.ones((t_local, t_local), jnp.bfloat16), k=1)

  def ranks(x):  # [rows, t_local] 0/1 -> exclusive prefix count per row
    return jnp.dot(
        x.astype(jnp.bfloat16), tri, preferred_element_type=jnp.float32
    ).astype(jnp.int32)

  # Sender: packed rows ordered (d, w, t).
  my = lax.dynamic_index_in_dim(oh, me, axis=2, keepdims=False)  # [d, w, t]
  pos = send_off[:, :, None] + ranks(my.reshape(ep * waves, t_local)).reshape(
      ep, waves, t_local
  )
  pos_dt = jnp.sum(my * pos, axis=1)  # [d, t]
  valid = my.sum(axis=1) > 0
  t_idx = jnp.broadcast_to(
      jnp.arange(t_local, dtype=jnp.int32)[None, :], (ep, t_local)
  )
  # Valid send positions are unique and dense in [0, total), so sorting the
  # (d, t) candidates by position puts each slot's token in place; the
  # slots past the total are never sent.
  _, perm = lax.sort_key_val(
      jnp.where(valid, pos_dt, jnp.int32(2**30)).reshape(-1), t_idx.reshape(-1)
  )
  perm = perm[:send_alloc]
  # Receiver: where each needed token landed on me.
  mine = lax.dynamic_index_in_dim(oh, me, axis=0, keepdims=False)  # [w, s, t]
  base_me = lax.dynamic_index_in_dim(base, me, axis=0, keepdims=False)  # [w, s]
  posr = base_me[:, :, None] + ranks(mine.reshape(waves * ep, t_local)).reshape(
      waves, ep, t_local
  )
  posmap = jnp.sum(mine * posr, axis=0).reshape(n_tokens)
  return perm, tab, posmap


def _uninit_hbm(shape, dtype):
  """An HBM buffer with no defined contents, at no cost.

  A Pallas call whose body never writes its output hands XLA a fresh,
  unfilled buffer. The ragged all-to-all only defines the rows it lands and
  the kernel reads only those, so a zero fill (which grows with T and EP)
  would be pure waste.

  The XLA buffer is also pinned to HBM (the out_specs memory space only
  types the kernel's ref). Unpinned, memory-space assignment may place the
  buffer in VMEM and then copy its undefined contents to HBM for the start
  kernel.

  Args:
    shape: Buffer shape.
    dtype: Buffer element type.

  Returns:
    The uninitialized HBM buffer.
  """
  return pl.pallas_call(
      lambda o_ref: None,
      out_shape=pltpu.HBM(shape, dtype),
      out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
      name="uninit_recv",
  )()


def _fwd_schedule(fwd_max_runs):
  """The De-Dup forward schedule: per target wave, lookahead 0."""
  return dedup_plan_mod.FwdSchedule(per_target_wave=True, max_runs=fwd_max_runs)


def start(
    rows,
    pairs,
    rows_by_dest,
    me,
    *,
    ep,
    g_local,
    waves,
    scale_bits=None,
    dedup_tables=None,
    min_fanout=2,
    fwd_max_runs=dedup_plan_mod.MAX_FWD_RUNS,
):
  """Plan shard `me`'s exchange and issue it; returns a WaveDispatch.

  rows [t_local, lane blocks, 128]: this shard's token rows as they go on
    the wire (fp8 where the weight format quantizes activations).
  pairs [T, topk] i32: every shard's selected expert ids, in mesh order.
  rows_by_dest [ep, ep * g_local] i32: the all-gathered per-expert pair
    counts, identical on every shard.
  scale_bits [t_local] i32 | None: this shard's activation-scale bits when
    rows are fp8 (carried as plane 1 of the token-gather lists).
  dedup_tables TopologyTables | None: when supplied (De-Dup), stages rows
    needed by >= `min_fanout` devices of a remote board once to that board's
    gateway, which forwards them inside the fused kernel. Each forwarded
    row counts in its target's own first wave (per-target-wave
    forwarding), and the gateway's forward tables hold `fwd_max_runs` runs
    per (served source, wave); the caller builds them after the start
    kernel (materialize_smem_ops) so they overlap the dispatch DMAs.

  The token-gather lists are built without negative-index wraps
  (source_plan.build_send_lists) and every exchange buffer (send, recv,
  lists, landing, stage) is pinned to HBM, so memory-space assignment adds
  no VMEM round trips around the start and drain kernels.

  Args:
    rows: [t_local, lane blocks, 128] this shard's token rows as they go on the
      wire.
    pairs: [T, topk] i32 every shard's selected expert ids, in mesh order.
    rows_by_dest: [ep, ep * g_local] i32 all-gathered per-expert pair counts.
    me: This shard's index.
    ep: Expert-parallel width.
    g_local: Experts per shard.
    waves: W.
    scale_bits: Optional [t_local] i32 activation-scale bits when rows are fp8.
    dedup_tables: TopologyTables to enable De-Dup, or None.
    min_fanout: Devices of a remote board that must need a token for it to be
      staged.
    fwd_max_runs: Gateway forward runs per (served source, wave).

  Returns:
    A WaveDispatch.

  Raises:
    ValueError: If ep < 3.
  """
  if ep < 3:
    raise ValueError(f"wave dispatch needs ep >= 3; got ep={ep}")
  t_local = rows.shape[0]
  n_tokens, topk = pairs.shape
  hist = rows_by_dest.sum(axis=0)
  order, wave_of, wave_bounds = wave_plan(
      hist, ep=ep, g_local=g_local, waves=waves
  )
  plan_p1 = source_plan.source_plan(
      pairs,
      me,
      ep=ep,
      t_local=t_local,
      g_local=g_local,
      waves=waves,
      wave_of=wave_of,
  )
  # Receiver per-wave counts on `me`. Under De-Dup's per-target-wave
  # forwarding every row `me` needs (direct or gateway-forwarded) counts in
  # `me`'s own first-hit wave, so the counts are the direct send's.
  recv_cnt = source_plan.receiver_counts(
      pairs,
      me,
      ep=ep,
      t_local=t_local,
      g_local=g_local,
      waves=waves,
      wave_of=wave_of,
  )
  # A token is sent to at most min(topk, ep) shards (and under De-Dup a
  # staged remote board replaces >= min_fanout direct rows with 1 stage row,
  # so total direct + stage rows is still <= t_local * min(topk, ep)).
  send_alloc = align_up(t_local * min(topk, ep), ROWBLK)
  if dedup_tables is None:
    fwd_schedule = None
    plan = plan_p1
    disp_tab = source_plan.dispatch_table(plan, me, recv_cnt, t_local=t_local)
    perm = source_plan.send_perm(plan, send_alloc=send_alloc)
    stage_send_tab = None
    dplan = None
    stage_buf = None
    num_boards = 0
    num_slots = 0
    board_size = 0
  else:
    fwd_schedule = _fwd_schedule(fwd_max_runs)
    num_boards = dedup_tables.num_boards
    num_slots = dedup_tables.num_slots
    board_size = dedup_tables.board_size
    dplan = dedup_plan_mod.dedup_plan(
        plan_p1.fw,
        me,
        tables=dedup_tables,
        num_waves=waves,
        min_fanout=min_fanout,
        need=plan_p1.need,
        schedule=fwd_schedule,
        compute_fwd_runs=False,
    )
    disp_tab = dedup_plan_mod.dedup_disp_tab(dplan, recv_cnt)
    perm = dplan.perm[:send_alloc]
    stage_send_tab = jnp.concatenate(
        [
            dplan.stage_send_off,
            dplan.cnt_stage,
            dplan.stage_dst,
            jnp.broadcast_to(dplan.gw[:, None], (num_boards, waves)),
            jnp.broadcast_to(dplan.sigma[:, None], (num_boards, waves)),
        ],
        axis=0,
    ).astype(jnp.int32)
    stage_buf = _uninit_hbm(
        (align_up(num_slots * t_local, ROWBLK), *rows.shape[1:]), rows.dtype
    )
    plan = plan_p1._replace(
        cnt_dir=dplan.cnt_dir,
        slot_off=dplan.slot_off,
        row_on_dest=jnp.maximum(dplan.row_on_dest, 0),
        send_off=dplan.send_off,
        send_cnt=dplan.cnt_dir,
    )
  # The rows from wave_bounds_row(ep) on are zero padding; the fused
  # kernel reads its wave cuts here.
  disp_tab = disp_tab.at[wave_bounds_row(ep)].set(
      lax.dynamic_index_in_dim(wave_bounds, me, axis=0, keepdims=False)
  )
  hist_me = lax.dynamic_index_in_dim(rows_by_dest, me, axis=0, keepdims=False)
  nw = 1 if scale_bits is None else 2
  tg_lists = source_plan.build_send_lists(
      pairs,
      plan,
      me,
      hist_me,
      ep=ep,
      t_local=t_local,
      g_local=g_local,
      scale_bits=scale_bits,
  )
  n_per_dest = source_plan.list_lengths(hist_me, ep=ep, g_local=g_local)
  dma_rows = source_plan.list_dma_rows(n_per_dest, nw)
  remote_list_rows = dma_rows.sum() - lax.dynamic_index_in_dim(
      dma_rows, me, axis=0, keepdims=False
  )
  list_tab = jnp.concatenate([dma_rows, remote_list_rows[None]]).astype(
      jnp.int32
  )
  send = _uninit_hbm((send_alloc, *rows.shape[1:]), rows.dtype)
  # A shard receives each token at most once (in fixed per-source slots).
  recv = _uninit_hbm((align_up(n_tokens, ROWBLK), *rows.shape[1:]), rows.dtype)
  tg_land = _uninit_hbm(tg_lists.shape, tg_lists.dtype)
  fut = dispatch_kernels.ragged_a2a_start_waves(
      send,
      recv,
      disp_tab,
      me,
      ep=ep,
      waves=waves,
      chunk_rows=CHUNK_ROWS,
      rows=rows,
      perm=perm,
      list_tab=list_tab,
      tg_lists=tg_lists,
      tg_land=tg_land,
      stage_send_tab=stage_send_tab,
      stage=stage_buf,
      num_boards=num_boards,
      num_slots=num_slots,
      board_size=board_size,
  )
  return WaveDispatch(
      fut,
      disp_tab,
      order,
      list_tab,
      stage_send_tab,
      None,
      dplan,
      fwd_schedule,
      wave_of,
  )


def materialize_smem_ops(
    wd: WaveDispatch,
    pairs: jax.Array,
    me,
    *,
    t_local: int,
    waves: int,
    dedup_tables,
    min_fanout: int = 2,
) -> WaveDispatch:
  """Builds the gateway SMEM operands (`stage_tab`, `fwd_tab`, `fwd_cnt`).

  Args:
    wd: The WaveDispatch returned by `start`.
    pairs: [T, topk] i32 expert ids in mesh order.
    me: This shard's index.
    t_local: Tokens per shard.
    waves: W.
    dedup_tables: TopologyTables, or None without De-Dup.
    min_fanout: As in `start`.

  Returns:
    `wd` with `smem_ops` filled in, or `wd` unchanged without De-Dup or when
    already built.
  """
  if dedup_tables is None or wd.smem_ops is not None:
    return wd
  assert wd.wave_of is not None
  assert wd.fwd_schedule is not None
  assert wd.dplan is not None
  fw_board = dedup_plan_mod.served_fw(
      pairs,
      wd.wave_of,
      me,
      tables=dedup_tables,
      t=t_local,
      num_waves=waves,
  )
  gplan = dedup_plan_mod.gateway_plan(
      fw_board,
      me,
      tables=dedup_tables,
      num_waves=waves,
      min_fanout=min_fanout,
      schedule=wd.fwd_schedule,
  )
  smem_ops = dedup_plan_mod.pack_smem_operands(wd.dplan, gplan)
  return wd._replace(smem_ops=smem_ops)


def forward_runs_needed(
    pairs,
    *,
    ep,
    g_local,
    waves,
    t_local,
    dedup_tables,
    min_fanout=2,
):
  """The most forward runs any (gateway, served source, wave) needs.

  Per-target-wave runs have no static bound (dedup_plan.FwdSchedule), so a
  caller that knows its routing sizes the layer's dedup_fwd_max_runs with
  this before building the layer: every gateway's own `gateway_plan` (same
  stage selection, stage order and run split as `start`), for all ep
  gateways.

  Args:
    pairs: [T, topk] i32 global expert ids in mesh order, as `start` takes.
    ep: Expert-parallel width.
    g_local: Experts per shard.
    waves: W.
    t_local: Tokens per shard.
    dedup_tables: The De-Dup topology tables.
    min_fanout: As `start`.

  Returns:
    i32 scalar.
  """
  pairs = jnp.asarray(pairs, jnp.int32)
  hist = jnp.bincount(pairs.reshape(-1), length=ep * g_local)
  _, wave_of, _ = wave_plan(hist, ep=ep, g_local=g_local, waves=waves)
  schedule = _fwd_schedule(dedup_plan_mod.MAX_FWD_RUNS)

  def one(gw):
    fws = dedup_plan_mod.served_fw(
        pairs, wave_of, gw, tables=dedup_tables, t=t_local, num_waves=waves
    )
    fwd_cnt = dedup_plan_mod.gateway_forward_runs_peak(
        fws,
        gw,
        tables=dedup_tables,
        num_waves=waves,
        min_fanout=min_fanout,
        schedule=schedule,
    )
    return jnp.max(fwd_cnt)

  return jnp.max(lax.map(one, jnp.arange(ep, dtype=jnp.int32)))


def finish(wd, arrivals, *, ep):
  """Drain this shard's outgoing copies; returns `arrivals` (aliased).

  Called after the fused kernel, which gave the copies the whole kernel to
  finish. Threading the kernel's arrivals through orders the drain after
  the kernel and keeps it from being dead-code-eliminated
  (tc_sender.ragged_a2a_send_done), and keeps them in HBM.

  Args:
    wd: The WaveDispatch returned by `start`.
    arrivals: The fused kernel's arrivals.
    ep: Expert-parallel width.

  Returns:
    `arrivals`, aliased.
  """
  send_sem, _, send, _, list_send_sem, _, tg_lists, *_ = wd.fut
  return dispatch_kernels.ragged_a2a_send_done(
      send_sem,
      send,
      wd.disp_tab,
      arrivals,
      ep=ep,
      list_tab=wd.list_tab,
      tg_lists=tg_lists,
      list_send_sem=list_send_sem,
  )
