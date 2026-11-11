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
"""Sender-local wave dispatch plan and token-gather lists.

The reference dispatch plan (`wave_dispatch.dispatch_plan`) evaluates the
expert-wave plan for every
global token on every shard (O(K * ep * T) for the first-hit waves and an
[ep, W, ep, t] one-hot) and lays the receive buffer out (wave, source)-major,
which needs a prefix over all sources and a `posmap [T]` per shard.

This module evaluates the same plan for ONE source s over its own t tokens
(convention D1 of dedup_plan.py) and lays every receive buffer out in fixed
per-source slots (D2):
rows [s * t, (s + 1) * t) of every destination's receive buffer belong to
source s, (wave, token) ordered inside the slot. Every receive row is then
known to the source alone, so the source also tells each destination where its
rows landed: one int32 list per (source -> destination), already in the
destination's slab order, behind a 128-word header.
`tg_assemble.py` turns the landed lists into the fused kernel's token-gather
table. Which (token, destination) pairs are sent, and in which wave, is
exactly the reference plan's; only where they land changes.

Everything here is pure jnp built from compares, MXU one-hot products, matmul
ranks, one sort and dynamic slices: no XLA scatter and no element gather.
"""

from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_moe import host as fused_host

align_up = fused_host.align_up

LANES = 128

# The header block in front of every (source -> destination) list. One
# block goes to EVERY destination, even with no entries, so a receiver always
# waits for a static ep * 512 B of headers plus the bodies.
HDR_WORDS = LANES
MAX_WAVES = 16
HDR_N = 0  # entries in the list, n(s -> d)
HDR_SRC = 1  # the source shard s
HDR_WAVES = 2  # W
HDR_NW = 3  # int32 words per entry (planes in the body)
HDR_SEND_OFF = 8  # [MAX_WAVES] s's send offset of (d, w)
HDR_CNT_DIR = HDR_SEND_OFF + MAX_WAVES  # [MAX_WAVES] rows of (d, w) in slot s
HDR_N_FWD = (
    HDR_CNT_DIR + MAX_WAVES
)  # [MAX_WAVES] De-Dup: forwarded rows (unused)
HDR_STAGE_SEND_OFF = HDR_N_FWD + MAX_WAVES  # [MAX_WAVES] De-Dup (unused)
HDR_CNT_STAGE = HDR_STAGE_SEND_OFF + MAX_WAVES  # [MAX_WAVES] De-Dup (unused)
assert HDR_CNT_STAGE + MAX_WAVES <= HDR_WORDS

# Exclusive ranks along t use one [t, t] strictly upper triangular matmul up
# to this many tokens and RANK_BLOCK-wide blocks above it (a full tri is t^2
# bf16: 32 MiB at t = 4096).
RANK_FULL_MAX = 1024
RANK_BLOCK = 512


def list_capacity(t_local, topk):
  """Returns the most entries one (source -> destination) list can hold.

  Lmax: the most entries one (source -> destination) list can hold.

  Every pair of the source can compute on one destination. That is t * K
  and not t * min(K, g): hash-routed ids may repeat an expert within a
  token, and each repeat is a pair of its own (dropless).

  Args:
    t_local: Tokens per shard.
    topk: Experts per token.

  Returns:
    t_local * topk rounded up to whole 128-word rows.
  """
  return align_up(t_local * topk, LANES)


def list_row_words(t_local, topk, nw):
  """Returns the words of one source's row of the list buffers.

  Args:
    t_local: Tokens per shard.
    topk: Experts per token.
    nw: Planes per entry (1, or 2 with scale bits).

  Returns:
    The header words plus `nw` planes of `list_capacity` words.
  """
  return HDR_WORDS + nw * list_capacity(t_local, topk)


def list_rows(t_local, topk, nw):
  """Returns one source's row of the list buffers in [*, 128] rows.

  Args:
    t_local: Tokens per shard.
    topk: Experts per token.
    nw: Planes per entry.

  Returns:
    list_row_words(...) // 128.
  """
  return list_row_words(t_local, topk, nw) // LANES


def list_dma_rows(n, nw):
  """Returns the [*, 128] rows one list DMA moves.

  Args:
    n: Entries in the list.
    nw: Planes per entry.

  Returns:
    One header row plus `nw` planes of ceil(n / 128) rows.
  """
  return 1 + nw * ((n + (LANES - 1)) // LANES)


class SourcePlan(NamedTuple):
  """Source s's half of the expert-wave dispatch plan.

  dest [K, t] i32: destination shard of every pair (pk // g).
  wave_pair [K, t] i32: the wave of every pair on its destination.
  fw [ep, t] i32: first-hit wave of each token on each shard (W: no hit).
  need [ep, t] bool: fw < W.
  oh_dir [ep, W, t] i32: fw == w, the rows s sends directly in wave w.
  cnt_dir [ep, W] i32: rows per (d, w).
  slot_off [ep, W] i32: where (d, w)'s rows start inside slot s on d.
  rank_dir [ep, W, t] i32: exclusive rank of each token within (d, w).
  row_on_dest [ep, t] i32: the receive row of token t on d,
    s * t + slot_off[d, fw] + rank (0 where d does not need t).
  send_off [ep, W] i32: where (d, w)'s rows start in s's send buffer.
  send_cnt [ep, W] i32: rows s sends per (d, w) (= cnt_dir without De-Dup).
  """

  dest: jax.Array
  wave_pair: jax.Array
  fw: jax.Array
  need: jax.Array
  oh_dir: jax.Array
  cnt_dir: jax.Array
  slot_off: jax.Array
  rank_dir: jax.Array
  row_on_dest: jax.Array
  send_off: jax.Array
  send_cnt: jax.Array


def exclusive_ranks(x):
  """Returns the exclusive prefix count of a 0/1 array along its last axis.

  [rows, t] 0/1 -> exclusive prefix count along t, i32.

  A strictly upper triangular ones matmul (0/1 in bf16, exact f32 sums), as
  the reference dispatch_plan; above RANK_FULL_MAX tokens the tri is RANK_BLOCK
  wide and the block totals are prefixed separately.

  Args:
    x: [rows, t] 0/1 values.

  Returns:
    [rows, t] i32 exclusive prefix counts.
  """
  rows, t = x.shape
  if t <= RANK_FULL_MAX:
    tri = jnp.triu(jnp.ones((t, t), jnp.bfloat16), k=1)
    return jnp.dot(
        x.astype(jnp.bfloat16), tri, preferred_element_type=jnp.float32
    ).astype(jnp.int32)
  nb = -(-t // RANK_BLOCK)
  tp = nb * RANK_BLOCK
  xp = jnp.pad(x.astype(jnp.int32), ((0, 0), (0, tp - t))).reshape(
      rows, nb, RANK_BLOCK
  )
  tri = jnp.triu(jnp.ones((RANK_BLOCK, RANK_BLOCK), jnp.bfloat16), k=1)
  within = jnp.einsum(
      "rnb,bc->rnc",
      xp.astype(jnp.bfloat16),
      tri,
      preferred_element_type=jnp.float32,
  ).astype(jnp.int32)
  tot = xp.sum(axis=2)
  before = jnp.cumsum(tot, axis=1) - tot
  return (within + before[:, :, None]).reshape(rows, tp)[:, :t]


def source_plan(
    pairs_g,
    s,
    *,
    ep,
    t_local,
    g_local,
    waves,
    wave_of,
    staged=None,
):
  """Builds source s's dispatch plan: per-pair waves and D2 receive rows.

  Args:
    pairs_g: [T, K] i32 global expert ids in mesh order.
    s: Source shard (int or traced scalar).
    ep: Expert-parallel width.
    t_local: Tokens per shard.
    g_local: Experts per shard.
    waves: W.
    wave_of: [ep, g] i32 wave of every shard's local experts
      (wave_dispatch.wave_plan).
    staged: Optional [ep, t] bool De-Dup mask of (dest, token) pairs taken out
      of the direct rows.

  Returns:
    A SourcePlan; its per-pair wave, first-hit wave, send rows and send offsets
    match the reference dispatch plan.

  Raises:
    ValueError: If waves exceeds MAX_WAVES (the list header width).
  """
  _, topk = pairs_g.shape
  if waves > MAX_WAVES:
    raise ValueError(f"waves={waves} > {MAX_WAVES} (list header width)")
  t = t_local
  s = jnp.asarray(s, jnp.int32)
  # The token axis stays minor, as in the reference plan: [K, t], not [t, K].
  pk = lax.dynamic_slice(pairs_g.astype(jnp.int32), (s * t, 0), (t, topk)).T
  dest = pk // g_local
  table = wave_of.astype(jnp.int32)
  # The MXU lookup (one-hot of the local expert times the table, then a
  # select by destination), over this source's t tokens instead of T.
  g_iota = jnp.arange(g_local, dtype=jnp.int32)
  local = (pk - dest * g_local).reshape(1, topk * t)
  onehot = (local == g_iota[:, None]).astype(jnp.bfloat16)  # [g, K * t]
  by_dest = jnp.dot(
      table.astype(jnp.bfloat16), onehot, preferred_element_type=jnp.float32
  )  # [ep, K * t]
  d_iota = jnp.arange(ep, dtype=jnp.int32)
  wave_pair = (
      jnp.sum(
          jnp.where(dest.reshape(1, topk * t) == d_iota[:, None], by_dest, 0.0),
          axis=0,
      )
      .astype(jnp.int32)
      .reshape(topk, t)
  )
  fw = jnp.min(
      jnp.where(
          dest[:, None, :] == d_iota[None, :, None],
          wave_pair[:, None, :],
          waves,
      ),
      axis=0,
  )  # [ep, t]
  need = fw < waves
  w_iota = jnp.arange(waves, dtype=jnp.int32)
  oh_dir = (fw[:, None, :] == w_iota[None, :, None]).astype(jnp.int32)
  if staged is not None:
    oh_dir = jnp.where(staged[:, None, :], 0, oh_dir)
  cnt_dir = oh_dir.sum(axis=2)  # [ep, W]
  slot_off = jnp.cumsum(cnt_dir, axis=1) - cnt_dir
  rank_dir = exclusive_ranks(oh_dir.reshape(ep * waves, t)).reshape(
      ep, waves, t
  )
  direct = oh_dir.sum(axis=1) > 0
  row_on_dest = jnp.where(
      direct,
      s * t + jnp.sum(oh_dir * (slot_off[:, :, None] + rank_dir), axis=1),
      0,
  )
  flat = cnt_dir.reshape(-1)
  send_off = (jnp.cumsum(flat) - flat).reshape(ep, waves)
  return SourcePlan(
      dest=dest,
      wave_pair=wave_pair,
      fw=fw,
      need=need,
      oh_dir=oh_dir,
      cnt_dir=cnt_dir,
      slot_off=slot_off,
      rank_dir=rank_dir,
      row_on_dest=row_on_dest,
      send_off=send_off,
      send_cnt=cnt_dir,
  )


def send_perm(plan, *, send_alloc):
  """Returns the local token of each packed send row.

  Args:
    plan: The SourcePlan of s = me.
    send_alloc: Rows of the send buffer.

  Returns:
    [send_alloc] i32 tokens in (dest, wave, token) order, the reference plan's
    `perm`.
  """
  ep, _, t = plan.oh_dir.shape
  pos = plan.send_off[:, :, None] + plan.rank_dir
  pos_dt = jnp.sum(plan.oh_dir * pos, axis=1)  # [d, t]
  valid = plan.oh_dir.sum(axis=1) > 0
  t_idx = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32)[None, :], (ep, t))
  _, perm = lax.sort_key_val(
      jnp.where(valid, pos_dt, jnp.int32(2**30)).reshape(-1), t_idx.reshape(-1)
  )
  return perm[:send_alloc]


def remote_rows(plan, me):
  """Returns the rows s = me sends to other shards.

  Args:
    plan: The SourcePlan of s = me.
    me: This shard's index.

  Returns:
    The remote total (disp_tab row 3 ep + 1, column 0).
  """
  mine = lax.dynamic_index_in_dim(plan.send_cnt, me, axis=0, keepdims=False)
  return plan.send_cnt.sum() - mine.sum()


def receiver_counts(pairs_g, me, *, ep, t_local, g_local, waves, wave_of):
  """Returns the rows `me` receives per wave.

  [W] i32: rows `me` receives per wave, sum over s of cnt_dir_s[me].

  Recomputed from the exchanged routing: only table row `me`
  is looked up, through the same kind of one-hot, so the cost is one
  [g, K * T] one-hot (the reference plan also built it) plus O(K * T), against
  its
  [ep, K * T] lookup and [K, ep, T] compare. The headers carry the same
  counts (HDR_CNT_DIR); switching the kernel's row 3 ep to them is 0c work.

  Args:
    pairs_g: [T, K] i32 global expert ids in mesh order.
    me: This shard's index.
    ep: Expert-parallel width.
    t_local: Tokens per shard.
    g_local: Experts per shard.
    waves: W.
    wave_of: [ep, g] i32 wave of every shard's local experts.

  Returns:
    [W] i32: the sum over sources s of cnt_dir_s[me].
  """
  n_tokens, topk = pairs_g.shape
  pk = pairs_g.T.astype(jnp.int32)  # [K, T]
  me = jnp.asarray(me, jnp.int32)
  del ep, t_local
  table = wave_of.astype(jnp.int32)
  row = lax.dynamic_index_in_dim(table, me, axis=0, keepdims=True)  # [1, g]
  first = me * g_local
  g_iota = jnp.arange(g_local, dtype=jnp.int32)
  # Pairs computing elsewhere fall outside [0, g): an all-zero column.
  local = (pk - first).reshape(1, topk * n_tokens)
  onehot = (local == g_iota[:, None]).astype(jnp.bfloat16)  # [g, K * T]
  wave = (
      jnp.dot(
          row.astype(jnp.bfloat16), onehot, preferred_element_type=jnp.float32
      )
      .astype(jnp.int32)
      .reshape(topk, n_tokens)
  )
  on_me = (pk >= first) & (pk < first + g_local)
  fw_me = jnp.min(jnp.where(on_me, wave, waves), axis=0)  # [T]
  w_iota = jnp.arange(waves, dtype=jnp.int32)
  return jnp.sum((fw_me[None, :] == w_iota[:, None]).astype(jnp.int32), axis=1)


def dispatch_table(plan, me, recv_cnt, *, t_local):
  """Builds the [4 * ep, W] dispatch table of shard `me`.

  The [4 * ep, W] disp_tab of dispatch_kernels, from plan (s = me).

  Rows as the reference plan's: send offset, send rows, destination offset, then
  rows
  received per wave (row 3 ep) and the remote total (row 3 ep + 1, col 0).
  Only the destination offset changes: (d, w) lands in slot me on d (D2).
  Built by concatenation, not .at[].set, so there is no XLA scatter.

  Args:
    plan: The SourcePlan of s = me.
    me: This shard's index.
    recv_cnt: [W] rows `me` receives per wave.
    t_local: Tokens per shard.

  Returns:
    [4 * ep, W] i32 disp_tab.

  Raises:
    ValueError: If ep < 2.
  """
  ep, waves = plan.cnt_dir.shape
  if ep < 2:
    raise ValueError("wave dispatch needs ep >= 2")
  dst_off = jnp.asarray(me, jnp.int32) * t_local + plan.slot_off
  col = jnp.arange(waves, dtype=jnp.int32)
  extra = jnp.concatenate(
      [
          recv_cnt.astype(jnp.int32)[None, :],
          jnp.where(col == 0, remote_rows(plan, me), 0)[None, :].astype(
              jnp.int32
          ),
          jnp.zeros((ep - 2, waves), jnp.int32),
      ],
      axis=0,
  )
  return jnp.concatenate([plan.send_off, plan.send_cnt, dst_off, extra], axis=0)


def list_lengths(hist_s, *, ep, g_local):
  """Returns the entries of each (s -> d) list.

  Args:
    hist_s: [ep * g] i32 per-expert pair counts of source s.
    ep: Expert-parallel width.
    g_local: Experts per shard.

  Returns:
    [ep] i32 n(s -> d).
  """
  return hist_s.reshape(ep, g_local).sum(axis=1).astype(jnp.int32)


def list_headers(plan, me, n, *, nw):
  """Returns the header block of each (me -> d) list.

  Args:
    plan: The SourcePlan of s = me.
    me: This shard's index.
    n: [ep] entries per list.
    nw: Planes per entry.

  Returns:
    [ep, 128] i32 header blocks.
  """
  ep, waves = plan.cnt_dir.shape

  def col(v):
    return jnp.broadcast_to(jnp.asarray(v, jnp.int32), (ep,))[:, None]

  pad = jnp.zeros((ep, MAX_WAVES - waves), jnp.int32)
  return jnp.concatenate(
      [
          n.astype(jnp.int32)[:, None],
          col(me),
          col(waves),
          col(nw),
          jnp.zeros((ep, HDR_SEND_OFF - 4), jnp.int32),
          plan.send_off,
          pad,
          plan.cnt_dir,
          pad,
          # De-Dup fields (n_fwd, stage send offsets, stage counts) and spare.
          jnp.zeros((ep, HDR_WORDS - HDR_N_FWD), jnp.int32),
      ],
      axis=1,
  )


def build_send_lists(
    pairs_g,
    plan,
    me,
    hist_me,
    *,
    ep,
    t_local,
    g_local,
    scale_bits=None,
):
  """Builds the token-gather lists shard `me` sends.

  The token-gather lists shard `me` sends, [ep * list_rows, 128] i32.

  Row block d (list_rows rows) is the (me -> d) list: the header, then the
  entries of me's pairs on d's experts in d's slab order -- local expert
  ascending, then pair index (t * K + k), exactly how the run (expert, me)
  is laid out in d's slab -- one plane per word: word 0 is the receive row
  of the pair's token on d (plan.row_on_dest); word 1, with scale_bits [t]
  (fp8 rows), the token's activation-scale bits. Planes are
  align_up(n, 128) apart, so ONE DMA of list_dma_rows(n, nw) rows carries
  the whole list. Words past n in a plane are zero and never read.

  One sort of the t * K pairs keyed by (expert, pair) plus a dynamic slice
  per destination; the receive row rides the sort as an operand, selected
  per pair by a compare over the ep destinations (no gather).
  hist_me [ep * g] i32 is me's per-expert pair count (rows_by_dest[me]).

  The per-destination slices pass allow_negative_indices=False: their
  starts are never negative, and JAX's negative-index wrap stages three
  scalar HLO ops (add, compare, select) per destination that XLA runs one
  by one outside any fusion, a cost that grows with EP. Same bits.

  Args:
    pairs_g: [T, K] i32 global expert ids in mesh order.
    plan: The SourcePlan of s = me.
    me: This shard's index.
    hist_me: [ep * g] i32 me's per-expert pair counts (rows_by_dest[me]).
    ep: Expert-parallel width.
    t_local: Tokens per shard.
    g_local: Experts per shard.
    scale_bits: Optional [t] i32 activation-scale bits (fp8 rows), sent as plane
      1.

  Returns:
    [ep * list_rows, 128] i32 lists, row block d being the (me -> d) list.

  Raises:
    ValueError: If the list sort key would overflow int32.
  """
  _, topk = pairs_g.shape
  t = t_local
  me = jnp.asarray(me, jnp.int32)
  nw = 1 if scale_bits is None else 2
  lmax = list_capacity(t, topk)
  pk = lax.dynamic_slice(pairs_g.astype(jnp.int32), (me * t, 0), (t, topk)).T
  d_iota = jnp.arange(ep, dtype=jnp.int32)
  val = jnp.sum(
      jnp.where(
          plan.dest[:, None, :] == d_iota[None, :, None],
          plan.row_on_dest[None, :, :],
          0,
      ),
      axis=1,
  )  # [K, t]
  pair = (
      jnp.arange(t, dtype=jnp.int32)[None, :] * topk
      + jnp.arange(topk, dtype=jnp.int32)[:, None]
  )
  # Unique per pair, below (E + 1) * t * K. An id outside [0, E) is in no
  # destination's count (n below), so it sorts past every list instead of
  # shifting the segments.
  e_total = ep * g_local
  if (e_total + 1) * t * topk >= 2**31:
    raise ValueError(
        f"list sort key (E + 1) * t * K = {(e_total + 1) * t * topk} "
        "does not fit int32"
    )
  valid = (pk >= 0) & (pk < e_total)
  key = jnp.where(valid, pk, e_total) * (t * topk) + pair
  operands = [key.reshape(-1), val.reshape(-1)]
  if scale_bits is not None:
    sb = scale_bits.astype(jnp.int32)
    if sb.ndim == 1:
      sb = jnp.broadcast_to(sb[None, :], (topk, t))
    elif sb.shape == (t, topk):
      sb = sb.T
    else:
      raise ValueError(
          f"scale_bits must be [t_local] or [t_local, topk], got {sb.shape}"
      )
    operands.append(sb.reshape(-1))
  ordered = lax.sort(tuple(operands), num_keys=1)
  # Padded so a slice never runs off the end (dynamic_slice would clamp
  # its start, shifting the entries).
  planes = [jnp.pad(p, (0, lmax)) for p in ordered[1:]]
  n = list_lengths(hist_me, ep=ep, g_local=g_local)
  start = jnp.cumsum(n) - n
  lane = jnp.arange(lmax, dtype=jnp.int32)
  bodies = []
  for d in range(ep):
    live = lane < n[d]
    segs = [
        jnp.where(
            live,
            lax.dynamic_slice(
                p, (start[d],), (lmax,), allow_negative_indices=False
            ),
            0,
        )
        for p in planes
    ]
    body = segs[0]
    if nw == 2:
      stride = (n[d] + (LANES - 1)) // LANES * LANES
      body = lax.dynamic_update_slice(
          jnp.concatenate([segs[0], jnp.zeros((lmax,), jnp.int32)]),
          segs[1],
          (stride,),
          allow_negative_indices=False,
      )
    bodies.append(body)
  blocks = jnp.concatenate(
      [list_headers(plan, me, n, nw=nw), jnp.stack(bodies)], axis=1
  )  # [ep, list_row_words]
  return blocks.reshape(ep * list_rows(t, topk, nw), LANES)


def slab_row_on_dest(
    local_blocks,
    block_hist,
    run_start_aligned,
    expert_base,
    me,
    *,
    ep,
    g_local,
    shard_stride,
    t_local,
    topk,
):
  """Returns the destination slab row each of shard me's pairs computes on.

  [K, t] i32: the row of its destination's own slab each of shard me's
  pairs computes on (for oracles).

  expert_base[e] - owner(e) * stride + run_start_aligned[e, me] + the
  pair's rank among me's pairs of expert e, the rank being fused_host's
  my_off + my_rank (as `pos` in build_routing_tables_sharded).

  Args:
    local_blocks: Routing blocks of me's pairs.
    block_hist: Per-block expert histogram.
    run_start_aligned: [E, ep] aligned run starts.
    expert_base: [E] slab base of each expert.
    me: This shard's index.
    ep: Expert-parallel width.
    g_local: Experts per shard.
    shard_stride: Slab rows per shard.
    t_local: Tokens per shard.
    topk: Experts per token.

  Returns:
    [K, t] i32 slab rows (for oracles).
  """
  e_total = ep * g_local
  bins = jnp.arange(e_total, dtype=jnp.int32)
  my_off = jnp.cumsum(block_hist, axis=0) - block_hist
  my_run_start = lax.dynamic_slice(
      run_start_aligned, (0, jnp.asarray(me, jnp.int32)), (e_total, 1)
  ).reshape(e_total)
  base = expert_base - (bins // g_local) * shard_stride + my_run_start
  block = local_blocks.shape[1]
  rows = fused_host._lookup_per_pair(  # pylint: disable=protected-access
      my_off + base[None, :], local_blocks, bins
  ) + fused_host._rank_within_block(  # pylint: disable=protected-access
      local_blocks, block
  )
  return rows.reshape(t_local, topk).T
