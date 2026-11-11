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
"""On-device (pure jnp) De-Dup dispatch plan.

A token that several devices of one remote board need crosses to that board
once: the source stages it to the board's gateway, and the gateway forwards it
to each device that needs it. This module builds every array that dispatch
runs on, from each source's first-wave rows `fw` (see `served_fw`). Boards
are contiguous blocks of `board_size` <= 8 devices (`topology.BoardTopology`).

Conventions referenced below as D1-D6:
  * D1: every device evaluates the same plan functions in three views (sender,
    gateway, receiver), so the views agree by construction.
  * D2: receive buffers are laid out in fixed per-source slots.
  * D3: each sender's token-gather list is already in the destination's slab
    order.
  * D4: the default forward schedule issues and counts every forward in its
    stage wave.
  * D5: stage rows are sorted in reflected-Gray order of their (at most
    8-bit) target mask, which bounds the forward runs of one (board, wave)
    group by 128.
  * D6: stage rows land on the gateway at `sigma * t + stage_row`.

Everything is scatter- and gather-free, because XLA scatters and element
gathers run on the TensorCore and are slow on TPU: compares, one-hot sums,
strictly-upper-triangular 0/1 matmuls for prefix counts (bf16 in, f32
accumulate: exact), static shifts, `dynamic_slice` and `lax.sort` with
payload operands. `dedup_plan_test` asserts it on the jaxpr.

Views (D1): every device evaluates the same functions
  * as a sender, `dedup_plan(fw(me), me)`: stage selection, Gray stage order,
    direct-run masking, `row_on_dest`, send layout, `send_row`;
  * as a gateway, `gateway_plan(fw of its served sources, me)`: expected stage
    rows and the forward runs (`fwd_tab`) per (served source sigma, wave);
  * as a receiver, `receiver_counts(fw of every source on its board, me)`:
    its byte-exact per-wave row counts (D4).
The gateway and receiver views run the same stage-selection code on the same
inputs as the sender, so the three views agree by construction (and the
golden tests check it across all devices).

Layouts:
  * recv on d (D2): slot s = rows [s t, (s + 1) t); inside the slot
    [direct rows, (wave, token) order][forwarded rows, stage order].
  * stage on gateway G: slot sigma = rows [sigma t, (sigma + 1) t), the
    source's stage rows for G's board in stage order.
  * send on s: [(d, w) direct runs][(b, w) stage runs].
  * stage order of (s, b) (D5): ascending stage_key =
    (w_stage, inverse_gray(mask8), token); runs of one bit j of the target
    mask inside one (b, w) group are <= 128 for any t.

Forward schedule (`FwdSchedule`): the layer runs per-target-wave forwarding,
which counts every forwarded row in its target's own first wave `fw` (the
"signal wave", carried in the high bits of the run's dst field) and issues it
in wave max(w_stage, fw). The default schedule is D4 exactly (issue and count
every forward in the stage wave); the golden tests check it against a host
reference.
"""

from collections.abc import Sequence
import dataclasses
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib

# The largest board the plan supports; the bit fields below are sized for it.
MAX_BOARD_SIZE = topology_lib.MAX_BOARD_SIZE
# D5: forward runs per (source, board, wave) group in reflected-Gray order.
MAX_FWD_RUNS = 128
# fwd_runs / fwd_tab entry fields.
FWD_STAGE_ROW, FWD_ROWS, FWD_DST, FWD_DST_ROW = 0, 1, 2, 3
FWD_FIELDS = 4
# The dst field holds dst | (signal wave - issue wave) << FWD_DST_SHIFT; the
# difference is 0 under D4, so default tables keep their dst values.
FWD_DST_SHIFT = 16
FWD_DST_MASK = (1 << FWD_DST_SHIFT) - 1
# Ready slots per (source, issue wave): rows that crossed in an earlier stage
# wave (slot 0), then the rows of this wave's stage run (slot 1).
FWD_SLOTS = 2
# Per-target-wave forwarding packs fw of all (<= 8) board targets in one
# int32 payload operand (4 bits per target -> 0..15, so W <= 15).
MAX_PTW_WAVES = 15
# stage_key of a token that is not staged (plus its token index, so that the
# sort is a deterministic permutation): sorts after every staged token.
NOT_STAGED = 1 << 30
_BIG = 1 << 30
# Prefix counts: one t x t triangle up to this t, else 512-blocked triangles
# (a full triangle is t^2 bf16 = 32 MiB at t = 4096).
FULL_TRI_MAX = 1024
TRI_BLOCK = 512


@dataclasses.dataclass(frozen=True)
class FwdSchedule:
  """When a gateway forwards a staged row and in which wave its target counts it.

  Attributes:
    per_target_wave: False is D4 (issue and count every forward in the stage
      wave). True counts a forwarded row in its target's own first wave fw and
      issues it in wave max(w_stage, fw); forward runs then also split where the
      target wave changes. The layer always runs True.
    max_runs: forward-table entries per (served source, issue wave). D5 bounds
      D4's runs by 128; per-target-wave runs have no such bound, so callers size
      this statically (`layer.plan.static_fwd_max_runs`) or from the routing
      (`wave_dispatch.forward_runs_needed`). If any `(slot, issue_wave)` exceeds
      `max_runs`, `_stage_select` demotes excess staged tokens on that slot to
      direct sends so `fwd_cnt <= max_runs` always holds.
  """

  per_target_wave: bool = False
  max_runs: int = MAX_FWD_RUNS

  def check(self, num_waves: int) -> None:
    """Validates the schedule for a wave count.

    Args:
      num_waves: W.

    Raises:
      ValueError: If max_runs < 1, or per_target_wave is set with more than
        MAX_PTW_WAVES waves.
    """
    if self.max_runs < 1:
      raise ValueError(f"max_runs={self.max_runs} < 1.")
    if self.per_target_wave and num_waves > MAX_PTW_WAVES:
      raise ValueError(
          f"per_target_wave packs fw in 4 bits; num_waves={num_waves} >"
          f" {MAX_PTW_WAVES}."
      )


def key_shifts(t: int) -> tuple[int, int]:
  """Returns the bit offsets (gray, wave) of `stage_key`.

  Args:
    t: Tokens per device.

  Returns:
    (gray shift, wave shift): 12 and 20 for t <= 4096.
  """
  tok_bits = max(12, (t - 1).bit_length())
  return tok_bits, tok_bits + 8


def run_key_shifts(t: int) -> tuple[int, int, int]:
  """Bit offsets (slot, pad, issue wave) of the forward-run compaction key.

  Below `slot`: stage_row * MAX_BOARD_SIZE + j (< 8 t); then 3 ready-slot bits
  (FWD_SLOTS
  uses 1), the pad bit, and the issue wave on top.

  Args:
    t: Tokens per device.

  Returns:
    (slot_sh, pad_sh, wave_sh).
  """
  tok_bits, _ = key_shifts(t)
  slot_sh = tok_bits + 3
  pad_sh = slot_sh + 3
  return slot_sh, pad_sh, pad_sh + 1


def _check_static(t: int, num_waves: int) -> None:
  _, wave_sh = key_shifts(t)
  if num_waves < 1 or (num_waves << wave_sh) > NOT_STAGED:
    raise ValueError(f"num_waves={num_waves} does not fit stage_key at t={t}.")
  if (num_waves << run_key_shifts(t)[2]) > _BIG:
    raise ValueError(f"num_waves={num_waves} does not fit the run sort key.")
  if t > 1 << 15:
    raise ValueError(f"t={t} > 32768 overflows the packed fprefix sort.")


# ----------------------------------------------------------------------------
# Scatter/gather-free helpers.
# ----------------------------------------------------------------------------


def inverse_gray(code: jax.Array) -> jax.Array:
  """Returns the position of an (at most) 8-bit code in the reflected Gray sequence.

  Args:
    code: Integer array of 8-bit codes.

  Returns:
    Integer array of positions.
  """
  rank = code
  for shift in (1, 2, 4):
    rank = rank ^ (rank >> shift)
  return rank


def _strict_upper(n: int) -> jax.Array:
  i = lax.broadcasted_iota(jnp.int32, (n, n), 0)
  j = lax.broadcasted_iota(jnp.int32, (n, n), 1)
  return (i < j).astype(jnp.bfloat16)


def exclusive_prefix(x: jax.Array) -> jax.Array:
  """Exclusive prefix counts of a 0/1 array along its last axis (int32).

  A strictly upper triangular 0/1 matmul (as in `wave_dispatch.dispatch_plan`),
  exact
  with bf16 inputs and f32 accumulation. Above `FULL_TRI_MAX` it is blocked:
  512 x 512 triangles inside each block plus an int32 cumsum of the (at most
  t / 512) block totals.

  Args:
    x: [..., n] bool or 0/1 integer array.

  Returns:
    [..., n] int32, out[..., i] = sum(x[..., :i]).
  """
  *lead, n = x.shape
  rows = int(np.prod(lead)) if lead else 1
  xb = x.reshape(rows, n).astype(jnp.bfloat16)
  if n <= FULL_TRI_MAX:
    out = jnp.dot(xb, _strict_upper(n), preferred_element_type=jnp.float32)
    return out.astype(jnp.int32).reshape(*lead, n)
  blk = TRI_BLOCK
  npad = -(-n // blk) * blk
  if npad != n:
    xb = jnp.pad(xb, ((0, 0), (0, npad - n)))
  nblk = npad // blk
  xr = xb.reshape(rows, nblk, blk)
  within = jnp.einsum(
      "rkb,bc->rkc", xr, _strict_upper(blk), preferred_element_type=jnp.float32
  ).astype(jnp.int32)
  tot = jnp.sum(xr.astype(jnp.int32), axis=-1)  # [rows, nblk]
  off = jnp.cumsum(tot, axis=-1) - tot
  out = (within + off[:, :, None]).reshape(rows, npad)
  return lax.slice_in_dim(out, 0, n, axis=1).reshape(*lead, n)


def suffix_min(x: jax.Array, fill: int) -> jax.Array:
  """Returns the suffix minimum along the last axis.

  Args:
    x: Integer array.
    fill: Value shifted in past the end.

  Returns:
    out[..., r] = min(x[..., r:]), by log-step doubling with static shifts.
  """
  n = x.shape[-1]
  k = 1
  while k < n:
    tail = jnp.full(x.shape[:-1] + (k,), fill, x.dtype)
    x = jnp.minimum(
        x,
        jnp.concatenate([lax.slice_in_dim(x, k, n, axis=-1), tail], axis=-1),
    )
    k *= 2
  return x


def _shift(x: jax.Array, k: int, fill) -> jax.Array:
  """k = 1: out[r] = x[r - 1]; k = -1: out[r] = x[r + 1] (last axis)."""
  n = x.shape[-1]
  pad = jnp.full(x.shape[:-1] + (1,), fill, x.dtype)
  if k == 1:
    return jnp.concatenate([pad, lax.slice_in_dim(x, 0, n - 1, axis=-1)], -1)
  return jnp.concatenate([lax.slice_in_dim(x, 1, n, axis=-1), pad], -1)


def _exclusive_cumsum(x: jax.Array, axis: int) -> jax.Array:
  """Tiny exclusive cumsums ([ep, W]-sized tables)."""
  return jnp.cumsum(x, axis=axis) - x


def _per_device(x: jax.Array, board_size: int) -> jax.Array:
  """[nb, t] per-board row -> [ep, t] (row bs b + j = row b), no gather."""
  nb, t = x.shape
  return jnp.broadcast_to(x[:, None, :], (nb, board_size, t)).reshape(
      nb * board_size, t
  )


def _count_by_wave(sel: jax.Array, wave: jax.Array, num_waves: int):
  """[..., t] bool selection, [..., t] wave -> [..., W] counts per wave."""
  w_iota = jnp.arange(num_waves, dtype=jnp.int32)
  hit = sel[..., None, :] & (wave[..., None, :] == w_iota[:, None])
  return jnp.sum(hit.astype(jnp.int32), axis=-1)


# ----------------------------------------------------------------------------
# The shared stage core: a batch of (source, remote board) pairs, each given
# the first waves of the board's devices.
# ----------------------------------------------------------------------------


class _StageSelect(NamedTuple):
  mask8: jax.Array  # [B, t] int32
  stage: jax.Array  # [B, t] bool
  w_stage: jax.Array  # [B, t] int32, W where not staged


def _stage_select(
    fw_b,
    need_b,
    stage_ok,
    *,
    num_waves,
    min_fanout,
    schedule: FwdSchedule | None = None,
):
  """mask8, stage (popcount >= min_fanout on a remote board), w_stage.

  Args:
    fw_b: [B, bs, t] first waves of the board's bs devices per token.
    need_b: [B, bs, t] bool, fw_b < W.
    stage_ok: [B] bool, the pair's board is remote.
    num_waves: W.
    min_fanout: Stage tokens needed by >= this many devices of the board.
    schedule: Optional `FwdSchedule`; if any `(slot, issue_wave)` forward-run
      count would exceed `schedule.max_runs`, excess staged tokens on that slot
      fall back to direct send so `fwd_cnt <= schedule.max_runs` always holds.

  Returns:
    The stage selection.
  """
  board_size = fw_b.shape[1]
  bit = (1 << jnp.arange(board_size, dtype=jnp.int32))[None, :, None]
  need_i = need_b.astype(jnp.int32)
  mask8 = jnp.sum(need_i * bit, axis=1)
  pop = jnp.sum(need_i, axis=1)
  # `DA/schedule.py:654`: sub.sum(axis=1) >= max(1, min_fanout).
  stage = (pop >= max(1, min_fanout)) & stage_ok[:, None]
  w_stage = jnp.where(stage, jnp.min(fw_b, axis=1), num_waves)
  if (
      schedule is not None
      and schedule.max_runs < board_size * fw_b.shape[2]
      and (schedule.per_target_wave or schedule.max_runs < MAX_FWD_RUNS)
  ):
    cand_order = _stage_order(
        _StageSelect(mask8=mask8, stage=stage, w_stage=w_stage),
        num_waves=num_waves,
        board_size=board_size,
        fw_b=fw_b if schedule.per_target_wave else None,
        compute_fprefix=False,
    )
    max_fwd_cnt = jnp.max(
        _fwd_cnt_from_order(cand_order, num_waves=num_waves, schedule=schedule),
        axis=1,
    )
    safe_rows = schedule.max_runs // board_size
    keep_runs = (max_fwd_cnt <= schedule.max_runs)[:, None] | (
        exclusive_prefix(stage) < safe_rows
    )
    stage = stage & keep_runs
    w_stage = jnp.where(stage, jnp.min(fw_b, axis=1), num_waves)
  return _StageSelect(mask8=mask8, stage=stage, w_stage=w_stage)


class _StageOrder(NamedTuple):
  stage_key: jax.Array  # [B, t]
  stage_perm: jax.Array  # [B, t]
  mask_sorted: jax.Array  # [B, t]
  w_sorted: jax.Array  # [B, t], W past the staged rows
  cnt_stage: jax.Array  # [B, W]
  bits: jax.Array  # [B, bs, t] bit j of mask_sorted
  fprefix: jax.Array | None = None  # [B, bs, t]
  fw_sorted: jax.Array | None = None  # [B, bs, t] fw of target j, stage order


def _stage_order(
    sel: _StageSelect,
    *,
    num_waves,
    board_size,
    fw_b=None,
    compute_fprefix: bool = True,
) -> _StageOrder:
  """One lax.sort per board over (stage_key, mask8, token), num_keys=1.

  Args:
    sel: The stage selection.
    num_waves: W.
    board_size: Devices per board (bs).
    fw_b: Optional [B, bs, t] first waves of the board's devices; when given
      (per-target-wave forwarding) they ride the sort as one payload operand, 4
      bits per target, and come back as `fw_sorted`.
    compute_fprefix: Whether to compute `fprefix`.

  Returns:
    The stage order.
  """
  b, t = sel.mask8.shape
  gray_sh, wave_sh = key_shifts(t)
  tok = lax.broadcasted_iota(jnp.int32, (b, t), 1)
  cnt_stage = _count_by_wave(sel.stage, sel.w_stage, num_waves)
  key = jnp.where(
      sel.stage,
      (sel.w_stage << wave_sh) | (inverse_gray(sel.mask8) << gray_sh) | tok,
      NOT_STAGED | tok,
  )
  operands = [key, jnp.where(sel.stage, sel.mask8, 0), tok]
  fw_shift = 4 * jnp.arange(board_size, dtype=jnp.int32)[None, :, None]
  if fw_b is not None:
    operands.append(
        jnp.sum(
            jnp.minimum(fw_b.astype(jnp.int32), MAX_PTW_WAVES) << fw_shift,
            axis=1,
        )
    )
  out = lax.sort(tuple(operands), dimension=1, num_keys=1)
  key_sorted, mask_sorted, stage_perm = out[:3]
  w_sorted = jnp.where(
      key_sorted < NOT_STAGED, key_sorted >> wave_sh, num_waves
  )
  bits = (
      mask_sorted[:, None, :]
      >> jnp.arange(board_size, dtype=jnp.int32)[None, :, None]
  ) & 1
  return _StageOrder(
      stage_key=key,
      stage_perm=stage_perm,
      mask_sorted=mask_sorted,
      w_sorted=w_sorted,
      cnt_stage=cnt_stage,
      bits=bits,
      fprefix=exclusive_prefix(bits) if compute_fprefix else None,
      fw_sorted=(
          None
          if fw_b is None
          else (out[3][:, None, :] >> fw_shift) & MAX_PTW_WAVES
      ),
  )


def _fwd_cnt_from_order(
    order: _StageOrder,
    *,
    num_waves: int,
    schedule: FwdSchedule = FwdSchedule(),
) -> jax.Array:
  """Forward-run counts `[B, W]` per `(batch, issue_wave)` (sort-free)."""
  bits = order.bits
  b = bits.shape[0]
  on = bits > 0
  ws = order.w_sorted
  ws3 = jnp.broadcast_to(ws[:, None, :], on.shape)
  brk_first = (_shift(ws, 1, -1) != ws)[:, None, :]
  if schedule.per_target_wave:
    if order.fw_sorted is None:
      raise ValueError("per_target_wave needs the stage order's fw_sorted.")
    tw = order.fw_sorted
    brk_first = brk_first | (_shift(tw, 1, -1) != tw)
    iw = jnp.maximum(ws3, tw)
  else:
    iw = ws3
  is_start = on & (~_shift(on, 1, False) | brk_first)
  return _count_by_wave(is_start.reshape(b, -1), iw.reshape(b, -1), num_waves)


def _forward_runs(
    order: _StageOrder,
    n_dir_b,
    src,
    board,
    *,
    num_waves,
    schedule: FwdSchedule = FwdSchedule(),
):
  """Forward runs per (b, issue wave): (stage_row, rows, dst, dst_row).

  A run is a maximal stretch of stage-order rows of one stage wave in which
  target j (device bs b + j) needs every row -- under per_target_wave also
  first needs every row in the same wave tw, its signal wave -- and it is
  one forward DMA. Its issue wave iw is the stage wave (D4) or max(w_stage,
  tw), and its ready slot is 0 when its rows crossed in a stage wave before
  iw, else 1 (the gateway issues slot 1 after waiting for the stage run of
  wave iw). Runs are
  compacted into a fixed [W, max_runs] layout by one sort: every run start
  gets the key (iw, slot, stage_row, j) (DA's (stage_row, dst) entry order
  inside a slot), and each issue wave gets max_runs - fwd_cnt[w] pad entries
  keyed right after its runs, so the first W * max_runs sorted entries are
  exactly the padded table. Under the default schedule every run has slot 1
  and the table is D4's.

  Args:
    order: The stage order (with `fw_sorted` under per_target_wave).
    n_dir_b: [B, bs] direct rows from the source to each target of the board.
    src: [B] source device of each batch row.
    board: [B] board of each batch row.
    num_waves: W.
    schedule: The forward schedule.

  Returns:
    (fwd_runs [B, W, max_runs, 4] int32: slot-relative stage rows, rows,
    dst | (tw - iw) << FWD_DST_SHIFT and absolute dst rows, zero padded;
    fwd_cnt [B, W] int32 runs per issue wave -- the table is only valid
    where fwd_cnt <= max_runs; fwd_slot_cnt [B, W, FWD_SLOTS] int32 runs per
    (issue wave, ready slot)).
  """
  schedule.check(num_waves)
  bits = order.bits
  b, board_size, t = bits.shape
  w = num_waves
  max_runs = schedule.max_runs
  slots = FWD_SLOTS
  slot_sh, pad_sh, wave_sh = run_key_shifts(t)
  if max_runs > 1 << pad_sh:
    raise ValueError(f"max_runs={max_runs} does not fit the run sort key.")
  on = bits > 0
  ws = order.w_sorted
  ws3 = jnp.broadcast_to(ws[:, None, :], on.shape)
  r = lax.broadcasted_iota(jnp.int32, on.shape, 2)
  j = lax.broadcasted_iota(jnp.int32, on.shape, 1)
  brk_first = (_shift(ws, 1, -1) != ws)[:, None, :]
  brk_last = (_shift(ws, -1, -1) != ws)[:, None, :]
  if schedule.per_target_wave:
    if order.fw_sorted is None:
      raise ValueError("per_target_wave needs the stage order's fw_sorted.")
    tw = order.fw_sorted
    brk_first = brk_first | (_shift(tw, 1, -1) != tw)
    brk_last = brk_last | (_shift(tw, -1, -1) != tw)
    iw = jnp.maximum(ws3, tw)
  else:
    tw = iw = ws3
  is_start = on & (~_shift(on, 1, False) | brk_first)
  is_end = on & (~_shift(on, -1, False) | brk_last)
  run_len = suffix_min(jnp.where(is_end, r, _BIG), _BIG) - r + 1
  # Ready slot from the run's last row, relative to its stage-wave segment
  # (runs never cross a segment, so `chunk` is 0 for every run start).
  w_iota = jnp.arange(w, dtype=jnp.int32)[None, :, None]
  cnt = order.cnt_stage[:, :, None]
  seg_lo = jnp.sum(jnp.where(w_iota < ws[:, None, :], cnt, 0), axis=1)
  seg_n = jnp.sum(jnp.where(w_iota == ws[:, None, :], cnt, 0), axis=1)
  wave_off = _exclusive_cumsum(order.cnt_stage, axis=1)
  wave_lo = jnp.sum(
      jnp.where(w_iota == ws[:, None, :], wave_off[:, :, None], 0), axis=1
  )
  stage_r = r - seg_lo[:, None, :] + wave_lo[:, None, :]
  last = jnp.where(is_start, r + run_len - 1 - seg_lo[:, None, :], 0)
  chunk = last // jnp.maximum(seg_n, 1)[:, None, :]
  slot = jnp.where(ws3 < iw, 0, 1 + chunk)
  fwd_slot_cnt = _count_by_wave(
      is_start.reshape(b, -1), (iw * slots + slot).reshape(b, -1), w * slots
  ).reshape(b, w, slots)
  fwd_cnt = jnp.sum(fwd_slot_cnt, axis=2)
  dst = (board[:, None, None] * board_size + j) | ((tw - iw) << FWD_DST_SHIFT)
  dst_row = src[:, None, None] * t + n_dir_b[:, :, None] + order.fprefix
  real_key = jnp.where(
      is_start,
      (iw << wave_sh) | (slot << slot_sh) | (r * MAX_BOARD_SIZE + j),
      _BIG,
  ).reshape(b, board_size * t)
  pad_shape = (b, w, max_runs)
  pad_i = lax.broadcasted_iota(jnp.int32, pad_shape, 2)
  pad_w = lax.broadcasted_iota(jnp.int32, pad_shape, 1)
  pad_key = jnp.where(
      pad_i >= fwd_cnt[:, :, None],
      (pad_w << wave_sh) | (1 << pad_sh) | pad_i,
      _BIG,
  ).reshape(b, w * max_runs)
  pad_zero = jnp.zeros((b, w * max_runs), jnp.int32)

  def cand(x):
    flat = jnp.where(is_start, x, 0).reshape(b, board_size * t)
    return jnp.concatenate([flat, pad_zero], axis=1)

  out = lax.sort(
      (
          jnp.concatenate([real_key, pad_key], axis=1),
          cand(stage_r),
          cand(run_len),
          cand(dst),
          cand(dst_row),
      ),
      dimension=1,
      num_keys=1,
  )
  fields = [
      lax.slice_in_dim(x, 0, w * max_runs, axis=1).reshape(pad_shape)
      for x in out[1:]
  ]
  return jnp.stack(fields, axis=-1), fwd_cnt, fwd_slot_cnt


def _to_token_order(stage_perm, fprefix):
  """Inverse permutation and fprefix in token order: one sort back."""
  b, t = stage_perm.shape
  board_size = fprefix.shape[1]
  pos = lax.broadcasted_iota(jnp.int32, (b, t), 1)
  fp = [
      lax.index_in_dim(fprefix, j, axis=1, keepdims=False)
      for j in range(board_size)
  ]
  # Two 16-bit prefixes per sort payload (t <= 32768, see _check_static).
  packed = [
      fp[2 * i] | ((fp[2 * i + 1] << 16) if 2 * i + 1 < board_size else 0)
      for i in range((board_size + 1) // 2)
  ]
  out = lax.sort((stage_perm, pos, *packed), dimension=1, num_keys=1)
  unpacked = []
  for p in out[2:]:
    unpacked += [p & 0xFFFF, p >> 16]
  return out[1], jnp.stack(unpacked[:board_size], axis=1)


# ----------------------------------------------------------------------------
# Sender view.
# ----------------------------------------------------------------------------


class SourcePlan(NamedTuple):
  """De-Dup plan of source s (sender view). int32 unless noted.

  De-Dup rows, one row per board b (b = board(s) never
  stages):
    mask8 [nb, t]: sum_j need[bs b + j, t] 2^j.
    stage [nb, t] bool: popcount(mask8) >= min_fanout and b != board(s).
    w_stage [nb, t]: min_{d in b} fw[d, t] for staged tokens, W otherwise:
      the wave the stage run (and every forward of it) travels in (D4).
    stage_key [nb, t]: w_stage 2^20 + inverse_gray(mask8) 2^12 + t (t <=
      4096; see `key_shifts`); non-staged: 2^30 + t.
    stage_perm [nb, t]: token at stage-order position r; the first
      n_stage[b] positions are the staged tokens.
    mask_sorted [nb, t]: mask8 of stage_perm, 0 past n_stage[b].
    stage_pos [nb, t]: inverse of stage_perm (stage-order position of t).
    cnt_stage [nb, W]: staged rows per (b, w).
    n_stage [nb]: staged rows per board.
    fprefix [nb, bs, t]: exclusive count of bit j of mask_sorted along stage
      order.
    fwd_runs [nb, W, max_runs, 4]: forward runs of the (b, issue wave)
      group as the gateway executes them (`_forward_runs`): (stage_row,
      rows, dst, dst_row). stage_row is slot-relative (stage-order position;
      absolute = sigma t + stage_row on the gateway, D6); dst = bs b + j, plus
      (signal wave - issue wave) << FWD_DST_SHIFT under per_target_wave;
      dst_row is absolute on dst.
    fwd_cnt [nb, W]: valid entries of fwd_runs (<= 128 under D4, D5).
    fwd_slot_cnt [nb, W, FWD_SLOTS]: the same per ready slot.
  Direct-send rows after De-Dup masking:
    oh_dir [ep, W, t] bool: direct rows, fw == w with staged (d, t) masked.
    cnt_dir [ep, W]: direct rows per (d, w) (= the direct send's send_cnt).
    n_dir [ep]: direct rows s -> d.
    slot_off [ep, W]: offset of the (d, w) direct run inside slot s on d.
    n_fwd [ep]: forwarded rows s -> d (they follow the direct rows, D2).
    recv_cnt [ep, W]: s's share of each target's wave count: direct rows with
      fw = w plus forwarded rows with w_stage = w (D4), or with fw = w under
      per_target_wave.
    row_on_dest [ep, t]: row of (s, t) in d's recv buffer, -1 if not needed:
      direct s t + slot_off[d, fw] + rank; forwarded s t + n_dir[d] +
      fprefix (in [s t, (s + 1) t), D2).
  Send side (send buffer [(d, w) direct runs][(b, w) stage runs]):
    send_off [ep, W]: direct run (d, w) in s's send buffer (rows cnt_dir).
    dst_off [ep, W]: s t + slot_off: direct run (d, w) in d's recv buffer.
    stage_send_off [nb, W]: stage run (b, w) in s's send buffer.
    stage_dst [nb, W]: stage run (b, w) in gw[b]'s stage buffer:
      sigma[b] t + sum_{w' < w} cnt_stage[b, w'].
    gw [nb]: gateway of s on board b (-1 for board(s)).
    sigma [nb]: s's stage slot at gw[b] (-1 for board(s)).
    send_row [ep + nb, t]: send-buffer row of t for direct dest d (rows
      [0, ep)) or stage board b (rows ep + b); -1 if not sent there.
    perm [(ep + nb) t]: token of each send row; the first send_total are
      valid (the start kernel packs rows[perm[:send_total]]).
    send_total []: rows in s's send buffer (direct, local included, + stage).
    remote_rows []: rows s sends to other devices (disp_tab row 3ep + 1).
  """

  mask8: jax.Array
  stage: jax.Array
  w_stage: jax.Array
  stage_key: jax.Array
  stage_perm: jax.Array
  mask_sorted: jax.Array
  stage_pos: jax.Array
  cnt_stage: jax.Array
  n_stage: jax.Array
  fprefix: jax.Array
  fwd_runs: jax.Array
  fwd_cnt: jax.Array
  fwd_slot_cnt: jax.Array
  oh_dir: jax.Array
  cnt_dir: jax.Array
  n_dir: jax.Array
  slot_off: jax.Array
  n_fwd: jax.Array
  recv_cnt: jax.Array
  row_on_dest: jax.Array
  send_off: jax.Array
  dst_off: jax.Array
  stage_send_off: jax.Array
  stage_dst: jax.Array
  gw: jax.Array
  sigma: jax.Array
  send_row: jax.Array
  perm: jax.Array
  send_total: jax.Array
  remote_rows: jax.Array


def dedup_plan(
    fw: jax.Array,
    s: jax.Array | int,
    *,
    tables: topology_lib.TopologyTables,
    num_waves: int,
    min_fanout: int = 2,
    need: jax.Array | None = None,
    schedule: FwdSchedule = FwdSchedule(),
    compute_fwd_runs: bool = True,
) -> SourcePlan:
  """The sender view of source s: its De-Dup rows and plan arrays.

  Args:
    fw: [ep, t] int32 first wave each device needs each of s's tokens; W = not
      needed.
    s: Source device (Python int or traced int32 scalar, e.g. `me`).
    tables: Static topology tables (`topology.build_topology_tables`).
    num_waves: W.
    min_fanout: Stage tokens needed by >= this many devices of a remote board.
    need: Optional [ep, t] bool (= fw < W).
    schedule: The forward schedule (default D4).
    compute_fwd_runs: Whether to build the sender's copy of its forward runs
      (`fwd_runs`, `fwd_cnt`, `fwd_slot_cnt`; empty otherwise). The layer does
      not: each gateway builds its own (`gateway_plan`). The golden tests check
        the sender's copy against the host oracle.

  Returns:
    The plan.
  """
  ep, t = fw.shape
  if ep != tables.num_devices:
    raise ValueError(f"fw has {ep} devices, tables {tables.num_devices}.")
  nb = tables.num_boards
  bs = tables.board_size
  w = num_waves
  _check_static(t, w)
  schedule.check(w)
  s = jnp.asarray(s, jnp.int32)
  fw = fw.astype(jnp.int32)
  need = (fw < w) if need is None else need.astype(bool)
  boards = jnp.arange(nb, dtype=jnp.int32)
  sel = _stage_select(
      fw.reshape(nb, bs, t),
      need.reshape(nb, bs, t),
      boards != s // bs,
      num_waves=w,
      min_fanout=min_fanout,
      schedule=schedule,
  )

  # Direct rows: the direct send's rows with staged (d, t) masked out.
  stage_dt = _per_device(sel.stage, bs)
  staged_dt = stage_dt & need
  dir_need = need & ~stage_dt
  w_iota = jnp.arange(w, dtype=jnp.int32)
  oh_dir = (fw[:, None, :] == w_iota[None, :, None]) & dir_need[:, None, :]
  cnt_dir = jnp.sum(oh_dir.astype(jnp.int32), axis=2)
  n_dir = jnp.sum(cnt_dir, axis=1)
  slot_off = _exclusive_cumsum(cnt_dir, axis=1)
  rank_dir = exclusive_prefix(oh_dir)
  in_slot = jnp.sum(
      jnp.where(oh_dir, slot_off[:, :, None] + rank_dir, 0), axis=1
  )
  flat = cnt_dir.reshape(ep * w)
  send_off = _exclusive_cumsum(flat, axis=0).reshape(ep, w)
  in_send = jnp.sum(
      jnp.where(oh_dir, send_off[:, :, None] + rank_dir, 0), axis=1
  )
  n_direct = jnp.sum(flat)

  # Forwarded rows: counted in the target's wave w_stage (D4), or in its own
  # first wave fw under per_target_wave.
  ptw = schedule.per_target_wave
  fwd_wave = _count_by_wave(
      staged_dt, fw if ptw else _per_device(sel.w_stage, bs), w
  )
  n_fwd = jnp.sum(fwd_wave, axis=1)

  order = _stage_order(
      sel,
      num_waves=w,
      board_size=bs,
      fw_b=fw.reshape(nb, bs, t) if ptw else None,
  )
  assert order.fprefix is not None
  if compute_fwd_runs:
    fwd_runs, fwd_cnt, fwd_slot_cnt = _forward_runs(
        order,
        n_dir.reshape(nb, bs),
        jnp.broadcast_to(s, (nb,)),
        boards,
        num_waves=w,
        schedule=schedule,
    )
  else:
    empty = jnp.zeros((0,), jnp.int32)
    fwd_runs = fwd_cnt = fwd_slot_cnt = empty
  stage_pos, fprefix_tok = _to_token_order(order.stage_perm, order.fprefix)

  row_on_dest = jnp.where(
      dir_need,
      s * t + in_slot,
      jnp.where(
          staged_dt, s * t + n_dir[:, None] + fprefix_tok.reshape(ep, t), -1
      ),
  )

  cs = order.cnt_stage.reshape(nb * w)
  stage_send_off = (n_direct + _exclusive_cumsum(cs, axis=0)).reshape(nb, w)
  send_total = n_direct + jnp.sum(cs)
  remote_rows = send_total - lax.dynamic_index_in_dim(n_dir, s, keepdims=False)
  gw = lax.dynamic_index_in_dim(jnp.asarray(tables.gw), s, keepdims=False)
  sigma = lax.dynamic_index_in_dim(
      jnp.asarray(tables.sigma_at_gw), s, keepdims=False
  )
  stage_dst = jnp.maximum(sigma, 0)[:, None] * t + _exclusive_cumsum(
      order.cnt_stage, axis=1
  )
  dir_row = jnp.where(dir_need, in_send, -1)
  stage_row_b = jnp.where(
      sel.stage,
      lax.slice_in_dim(stage_send_off, 0, 1, axis=1) + stage_pos,
      -1,
  )
  send_row = jnp.concatenate([dir_row, stage_row_b], axis=0)
  # When a token is staged for board `b` (`sel.stage[b, tok]` is True), none of
  # `b`'s devices have `dir_need[bs*b + j, tok]` True (`dir_need = need &
  # ~stage_dt`). Folding `stage_row_b` into `j = 0` of each board shrinks the
  # sort domain from `(ep + nb) * t` to `ep * t` (a power of 2 for
  # power-of-2 ep and t, avoiding a 2x bitonic-sort padding jump) with
  # identical `perm[:send_total]`.
  j0_stage = (jnp.arange(bs, dtype=jnp.int32)[None, :, None] == 0) & sel.stage[
      :, None, :
  ]
  send_row_ep = jnp.where(
      j0_stage,
      stage_row_b[:, None, :],
      dir_row.reshape(nb, bs, t),
  ).reshape(ep, t)
  tok = lax.broadcasted_iota(jnp.int32, (ep, t), 1)
  _, perm_ep = lax.sort(
      (
          jnp.where(send_row_ep >= 0, send_row_ep, _BIG).reshape(-1),
          tok.reshape(-1),
      ),
      dimension=0,
      num_keys=1,
  )
  perm = jnp.pad(perm_ep, (0, nb * t))
  return SourcePlan(
      mask8=sel.mask8,
      stage=sel.stage,
      w_stage=sel.w_stage,
      stage_key=order.stage_key,
      stage_perm=order.stage_perm,
      mask_sorted=order.mask_sorted,
      stage_pos=stage_pos,
      cnt_stage=order.cnt_stage,
      n_stage=jnp.sum(order.cnt_stage, axis=1),
      fprefix=order.fprefix,
      fwd_runs=fwd_runs,
      fwd_cnt=fwd_cnt,
      fwd_slot_cnt=fwd_slot_cnt,
      oh_dir=oh_dir,
      cnt_dir=cnt_dir,
      n_dir=n_dir,
      slot_off=slot_off,
      n_fwd=n_fwd,
      recv_cnt=cnt_dir + fwd_wave,
      row_on_dest=row_on_dest,
      send_off=send_off,
      dst_off=s * t + slot_off,
      stage_send_off=stage_send_off,
      stage_dst=stage_dst,
      gw=gw,
      sigma=sigma,
      send_row=send_row,
      perm=perm,
      send_total=send_total,
      remote_rows=remote_rows,
  )


# ----------------------------------------------------------------------------
# Gateway view.
# ----------------------------------------------------------------------------


class GatewayPlan(NamedTuple):
  """Forward plan of gateway G (gateway view). int32.

  Attributes:
    src: [S] source device of stage slot sigma, -1 for unused slots.
    stage_rows: [S, W] stage rows G receives from slot sigma in wave w (the
      byte-exact `stage_sems[sigma, w]` wait).
    fwd_runs: [S, W, max_runs, 4] forward runs per (sigma, issue wave) with
      slot-relative stage rows (`_forward_runs`).
    fwd_tab: [S, W, max_runs, 4] the same with absolute stage rows (sigma t +
      stage_row): the kernel operand. Entries past fwd_cnt are 0.
    fwd_cnt: [S, W] valid runs per (sigma, issue wave); the table is only valid
      where fwd_cnt <= max_runs (always under D4: <= 128).
    fwd_slot_cnt: [S, W, FWD_SLOTS] the same per ready slot (kernel operand).
    fwd_remote: [W] rows G forwards to other devices per issue wave (its
      `fwd_send_sem` total is the sum).
    fwd_local: [W] rows G copies stage -> its own recv per issue wave.
  """

  src: jax.Array
  stage_rows: jax.Array
  fwd_runs: jax.Array
  fwd_tab: jax.Array
  fwd_cnt: jax.Array
  fwd_slot_cnt: jax.Array
  fwd_remote: jax.Array
  fwd_local: jax.Array


def gateway_plan(
    fw_served: jax.Array,
    gateway: jax.Array | int,
    *,
    tables: topology_lib.TopologyTables,
    num_waves: int,
    min_fanout: int = 2,
    schedule: FwdSchedule = FwdSchedule(),
) -> GatewayPlan:
  """The gateway view of device G.

  Args:
    fw_served: [S, bs, t] int32: for each stage slot sigma, `fw` of source
      `tables.served[G, sigma]` restricted to the bs devices of board(G)
      (`served_fw`). Rows of unused slots are ignored.
    gateway: G (Python int or traced int32 scalar, e.g. `me`).
    tables: Static topology tables.
    num_waves: W.
    min_fanout: As `dedup_plan`.
    schedule: The forward schedule (default D4).

  Returns:
    The gateway plan.
  """
  num_slots, devs, t = fw_served.shape
  bs = tables.board_size
  if num_slots != tables.num_slots or devs != bs:
    raise ValueError(
        f"fw_served {fw_served.shape} != [S={tables.num_slots}, {bs}, t]."
    )
  w = num_waves
  _check_static(t, w)
  schedule.check(w)
  g = jnp.asarray(gateway, jnp.int32)
  served = lax.dynamic_index_in_dim(
      jnp.asarray(tables.served), g, keepdims=False
  )
  src = jnp.maximum(served, 0)
  fw_served = fw_served.astype(jnp.int32)
  need_b = fw_served < w
  sel = _stage_select(
      fw_served,
      need_b,
      served >= 0,
      num_waves=w,
      min_fanout=min_fanout,
      schedule=schedule,
  )
  n_dir_b = jnp.sum((need_b & ~sel.stage[:, None, :]).astype(jnp.int32), axis=2)
  order = _stage_order(
      sel,
      num_waves=w,
      board_size=bs,
      fw_b=fw_served if schedule.per_target_wave else None,
  )
  fwd_runs, fwd_cnt, fwd_slot_cnt = _forward_runs(
      order,
      n_dir_b,
      src,
      jnp.broadcast_to(g // bs, (num_slots,)),
      num_waves=w,
      schedule=schedule,
  )
  rows = lax.index_in_dim(fwd_runs, FWD_ROWS, axis=3, keepdims=False)
  dst = lax.index_in_dim(fwd_runs, FWD_DST, axis=3, keepdims=False)
  shape4 = fwd_runs.shape
  slot = lax.broadcasted_iota(jnp.int32, shape4, 0)
  field = lax.broadcasted_iota(jnp.int32, shape4, 3)
  fwd_tab = fwd_runs + jnp.where(
      (field == FWD_STAGE_ROW) & (rows[..., None] > 0), slot * t, 0
  )
  local = (dst & FWD_DST_MASK) == g
  return GatewayPlan(
      src=served,
      stage_rows=order.cnt_stage,
      fwd_runs=fwd_runs,
      fwd_tab=fwd_tab,
      fwd_cnt=fwd_cnt,
      fwd_slot_cnt=fwd_slot_cnt,
      fwd_remote=jnp.sum(jnp.where(local, 0, rows), axis=(0, 2)),
      fwd_local=jnp.sum(jnp.where(local, rows, 0), axis=(0, 2)),
  )


def gateway_forward_runs_peak(
    fw_served: jax.Array,
    gateway: jax.Array | int,
    *,
    tables: topology_lib.TopologyTables,
    num_waves: int,
    min_fanout: int = 2,
    schedule: FwdSchedule = FwdSchedule(),
) -> jax.Array:
  """Returns the uncapped forward-run counts on a gateway.

  Args:
    fw_served: [S, bs, t] i32 fw of each served source on the gateway's board.
    gateway: Gateway device G.
    tables: Static topology tables.
    num_waves: W.
    min_fanout: Staging threshold.
    schedule: The forward schedule.

  Returns:
    [S, W] i32 run counts, without the `max_runs` cap.

  Raises:
    ValueError: If `fw_served` is not [S, bs, t] for these tables, or the
      schedule is invalid for `num_waves`.
  """
  num_slots, devs, t = fw_served.shape
  bs = tables.board_size
  if num_slots != tables.num_slots or devs != bs:
    raise ValueError(
        f"fw_served {fw_served.shape} != [S={tables.num_slots}, {bs}, t]."
    )
  w = num_waves
  _check_static(t, w)
  schedule.check(w)
  g = jnp.asarray(gateway, jnp.int32)
  served = lax.dynamic_index_in_dim(
      jnp.asarray(tables.served), g, keepdims=False
  )
  fw_served = fw_served.astype(jnp.int32)
  need_b = fw_served < w
  sel = _stage_select(
      fw_served,
      need_b,
      served >= 0,
      num_waves=w,
      min_fanout=min_fanout,
      schedule=None,
  )
  order = _stage_order(
      sel,
      num_waves=w,
      board_size=bs,
      fw_b=fw_served if schedule.per_target_wave else None,
      compute_fprefix=False,
  )
  return _fwd_cnt_from_order(order, num_waves=w, schedule=schedule)


# ----------------------------------------------------------------------------
# Receiver view and kernel operands.
# ----------------------------------------------------------------------------


class ReceiverCounts(NamedTuple):
  """Rows device `me` receives (receiver view). int32.

  Attributes:
    recv_cnt: [W] rows landing in me's recv buffer per wave (disp_tab row 3ep):
      direct rows with fw = w plus forwarded rows with w_stage = w (D4), or with
      fw = w under per_target_wave.
    cnt_from: [ep, W] the same per source.
    rows_from: [ep] n(s -> me) = direct + forwarded rows in slot s.
    n_fwd_from: [ep] forwarded rows of slot s (they follow the direct ones).
  """

  recv_cnt: jax.Array
  cnt_from: jax.Array
  rows_from: jax.Array
  n_fwd_from: jax.Array


def receiver_counts(
    fw_board: jax.Array,
    me: jax.Array | int,
    *,
    tables: topology_lib.TopologyTables,
    num_waves: int,
    min_fanout: int = 2,
    schedule: FwdSchedule = FwdSchedule(),
) -> ReceiverCounts:
  """Per-wave receive counts of `me`, recomputed from every source's fw.

  Args:
    fw_board: [ep, bs, t] int32: `fw` of every source restricted to the bs
      devices of board(me).
    me: Receiving device (Python int or traced int32 scalar).
    tables: Static topology tables.
    num_waves: W.
    min_fanout: As `dedup_plan`.
    schedule: The forward schedule (only per_target_wave matters here).

  Returns:
    The counts.
  """
  ep, devs, _ = fw_board.shape
  bs = tables.board_size
  if ep != tables.num_devices or devs != bs:
    raise ValueError(f"fw_board {fw_board.shape} != [ep, {bs}, t].")
  w = num_waves
  schedule.check(w)
  me = jnp.asarray(me, jnp.int32)
  fw_board = fw_board.astype(jnp.int32)
  src_board = jnp.arange(ep, dtype=jnp.int32) // bs
  sel = _stage_select(
      fw_board,
      fw_board < w,
      src_board != me // bs,
      num_waves=w,
      min_fanout=min_fanout,
      schedule=schedule,
  )
  fw_me = lax.dynamic_index_in_dim(fw_board, me % bs, axis=1, keepdims=False)
  need_me = fw_me < w
  cnt_fwd = _count_by_wave(
      need_me & sel.stage,
      fw_me if schedule.per_target_wave else sel.w_stage,
      w,
  )
  if schedule.per_target_wave:
    cnt_from = _count_by_wave(need_me, fw_me, w)
  else:
    cnt_dir = _count_by_wave(need_me & ~sel.stage, fw_me, w)
    cnt_from = cnt_dir + cnt_fwd
  return ReceiverCounts(
      recv_cnt=jnp.sum(cnt_from, axis=0),
      cnt_from=cnt_from,
      rows_from=jnp.sum(cnt_from, axis=1),
      n_fwd_from=jnp.sum(cnt_fwd, axis=1),
  )


class SmemOperands(NamedTuple):
  """The De-Dup kernel operands of one device.

  Attributes:
    stage_tab: [2 nb + S + 2, W] int32; see `stage_tab_layout`.
    fwd_tab: [S, W, max_runs, 4] int32 (stage_row, rows, dst, dst_row), absolute
      stage rows, per (sigma, issue wave) in ready-slot order; entries past the
      counts are 0.
    fwd_cnt: [S, W, FWD_SLOTS] int32 runs per (sigma, issue wave, ready slot).
  """

  stage_tab: jax.Array
  fwd_tab: jax.Array
  fwd_cnt: jax.Array


def stage_tab_layout(num_boards: int, num_slots: int) -> dict[str, slice]:
  """Returns the row blocks of `stage_tab`.

  Args:
    num_boards: nb.
    num_slots: S.

  Returns:
    {block name: row slice}.
  """
  nb, s = num_boards, num_slots
  layout = {
      "stage_send_off": slice(0, nb),  # (b, w) stage run in my send buffer.
      "stage_rows": slice(nb, 2 * nb),  # rows of my (b, w) stage run.
      "expected_stage_rows": slice(2 * nb, 2 * nb + s),  # (sigma, w) as gw.
      "fwd_remote": slice(2 * nb + s, 2 * nb + s + 1),  # remote fwd rows / w.
      "fwd_local": slice(2 * nb + s + 1, 2 * nb + s + 2),  # local fwd rows / w.
  }
  return layout


def pack_smem_operands(
    src_plan: SourcePlan, gw_plan: GatewayPlan
) -> SmemOperands:
  """Packs the sender and gateway plans into the kernel's SMEM operands.

  Args:
    src_plan: `dedup_plan(fw(me), me)`.
    gw_plan: `gateway_plan(..., me)`.

  Returns:
    The SmemOperands (`stage_tab`, `fwd_tab`, `fwd_cnt`).
  """
  rows = [
      src_plan.stage_send_off,
      src_plan.cnt_stage,
      gw_plan.stage_rows,
      gw_plan.fwd_remote[None, :],
      gw_plan.fwd_local[None, :],
  ]
  stage_tab = jnp.concatenate(rows, axis=0)
  return SmemOperands(
      stage_tab=stage_tab,
      fwd_tab=gw_plan.fwd_tab,
      fwd_cnt=gw_plan.fwd_slot_cnt,
  )


def dedup_disp_tab(src_plan: SourcePlan, recv_cnt: jax.Array) -> jax.Array:
  """The wave dispatch's `disp_tab [4 ep, W]` with De-Dup.

  Rows: [0, ep) send offset, [ep, 2ep) send rows (direct runs only), [2ep,
  3ep) dst offset me t + slot_off (D2), row 3ep the receiver's per-wave count
  (`receiver_counts(...).recv_cnt`), row 3ep + 1 col 0 the remote total
  including the stage rows (so the send drain stays one byte-exact wait).
  Built with concatenates, not `.at[].set`, so there is no XLA scatter.

  Args:
    src_plan: `dedup_plan(fw(me), me)`.
    recv_cnt: [W] me's receive count per wave.

  Returns:
    [4 ep, W] int32.
  """
  ep, w = src_plan.send_off.shape
  col0 = jnp.arange(w, dtype=jnp.int32) == 0
  extra = jnp.concatenate(
      [
          recv_cnt.astype(jnp.int32)[None, :],
          jnp.where(col0, src_plan.remote_rows, 0)[None, :],
          jnp.zeros((ep - 2, w), jnp.int32),
      ],
      axis=0,
  )
  return jnp.concatenate(
      [src_plan.send_off, src_plan.cnt_dir, src_plan.dst_off, extra], axis=0
  )


class DirectSendRows(NamedTuple):
  """The direct-send rows of a block of (token, destination) pairs.

  Attributes:
    dest: [K, n] int32 destination device of each (k, token) pair: pk // g.
    wave_pair: [K, n] int32 wave of each pair on its destination (only valid for
      pairs whose dest is in the requested destination block).
    fw: [nd, n] int32 first wave each destination needs the token in; W = not
      needed. nd is the destination block size.
    need: [nd, n] bool, fw < W.
  """

  dest: jax.Array
  wave_pair: jax.Array
  fw: jax.Array
  need: jax.Array


def direct_send_rows(
    pk: jax.Array,
    wave_of: jax.Array,
    *,
    num_waves: int,
    dest_base: jax.Array | int,
    num_dests: int,
) -> DirectSendRows:
  """Direct-send rows of the pairs `pk` for one block of destinations.

  The same first-wave rows the direct wave dispatch builds
  (`wave_dispatch.dispatch_plan`), restricted to `num_dests` consecutive
  destination devices from `dest_base`, so the cost is O(K * num_dests * n)
  rather than O(K * ep * n). The wave of each pair is looked up with a
  one-hot matmul, which keeps the builder scatter- and gather-free.

  Args:
    pk: [K, n] int32 global expert ids of n tokens (one or more sources).
    wave_of: [ep, g] int32 wave of each device's local experts.
    num_waves: W.
    dest_base: First destination device of the block (traced is fine).
    num_dests: Block size (static).

  Returns:
    DirectSendRows with dest/wave_pair [K, n] and fw/need [num_dests, n].
  """
  ep, g = wave_of.shape
  k, n = pk.shape
  w = num_waves
  dest = pk // g  # [K, n]
  table = wave_of.astype(jnp.int32)
  base = jnp.asarray(dest_base, jnp.int32)
  if num_dests != ep:
    table = lax.dynamic_slice_in_dim(table, base, num_dests, axis=0)
  d_iota = base + jnp.arange(num_dests, dtype=jnp.int32)
  # The one-hot MXU lookup of `dispatch_plan`, on n pairs per k instead of T.
  g_iota = jnp.arange(g, dtype=jnp.int32)
  local = (pk - dest * g).reshape(1, k * n)
  onehot = (local == g_iota[:, None]).astype(jnp.bfloat16)  # [g, K*n]
  by_dest = jnp.dot(
      table.astype(jnp.bfloat16), onehot, preferred_element_type=jnp.float32
  )  # [num_dests, K*n]
  wave = (
      jnp.sum(
          jnp.where(dest.reshape(1, k * n) == d_iota[:, None], by_dest, 0.0),
          axis=0,
      )
      .astype(jnp.int32)
      .reshape(k, n)
  )
  fw = jnp.min(
      jnp.where(dest[:, None, :] == d_iota[None, :, None], wave[:, None, :], w),
      axis=0,
  )  # [num_dests, n]
  return DirectSendRows(dest=dest, wave_pair=wave, fw=fw, need=fw < w)


def served_fw(
    pairs_g: jax.Array,
    wave_of: jax.Array,
    gateway: jax.Array | int,
    *,
    tables: topology_lib.TopologyTables,
    t: int,
    num_waves: int,
) -> jax.Array:
  """The gateway plan's input: fw of each served source on the gateway's board.

  Costs O(S * K * bs * t): only the t tokens of each of the S sources the
  gateway serves, and only the bs destinations of its own board.

  Args:
    pairs_g: [T, K] int32 global expert ids of every token in mesh order.
    wave_of: [ep, g] int32 wave of each device's local experts.
    gateway: Gateway device G (Python int or traced int32 scalar).
    tables: Static topology tables.
    t: Tokens per device.
    num_waves: W.

  Returns:
    [S, bs, t] int32: row sigma = fw of source `tables.served[G, sigma]` for
    the devices of board(G) (source 0's for unused slots; ignored).
  """
  g = jnp.asarray(gateway, jnp.int32)
  served = lax.dynamic_index_in_dim(
      jnp.asarray(tables.served), g, keepdims=False
  )
  bs = tables.board_size
  base = (g // bs) * bs
  s = tables.num_slots
  _, k = pairs_g.shape
  pairs_i32 = pairs_g.astype(jnp.int32)
  src_s = jnp.maximum(served, 0)
  slices = [
      lax.dynamic_slice(
          pairs_i32,
          (lax.index_in_dim(src_s, sigma, keepdims=False) * t, 0),
          (t, k),
      )
      for sigma in range(s)
  ]
  pk = jnp.concatenate(slices, axis=0).T
  rows = direct_send_rows(
      pk,
      wave_of,
      num_waves=num_waves,
      dest_base=base,
      num_dests=bs,
  )
  return jnp.transpose(rows.fw.reshape(bs, s, t), (1, 0, 2))


def _sub_jaxprs(eqn):
  """Yields the jaxprs nested in an equation's parameters."""
  for v in eqn.params.values():
    for sub in v if isinstance(v, Sequence) else (v,):
      inner = getattr(sub, "jaxpr", sub)
      if hasattr(inner, "eqns"):
        yield inner


def forbidden_primitives(closed_jaxpr) -> list[str]:
  """Returns the scatter and gather primitives anywhere in a jaxpr.

  Args:
    closed_jaxpr: A jaxpr or closed jaxpr.

  Returns:
    Primitive names, in visit order.
  """
  found = []
  stack = [getattr(closed_jaxpr, "jaxpr", closed_jaxpr)]
  while stack:
    jaxpr = stack.pop()
    nested = []
    for eqn in jaxpr.eqns:
      name = eqn.primitive.name
      if "scatter" in name or "gather" in name:
        found.append(name)
      nested.extend(_sub_jaxprs(eqn))
    stack.extend(reversed(nested))
  return found
