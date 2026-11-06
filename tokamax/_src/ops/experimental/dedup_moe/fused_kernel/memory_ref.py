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

"""Module that encapsulates memory references, buffer allocations, and DMA logic.

This module isolates all VMEM/SMEM buffer declarations, weight/scale DMA
helpers,
token-gather window streaming, gateway De-Dup forward DMAs, and result return
push/transport DMAs from the stateless compute and pipeline orchestration.

De-Dup stage and gateway forward buffer layout (`stage_hbm` -> `lhs_hbm`)
-------------------------------------------------------------------------
Under two-hop De-Dup dispatch (`dedup_slots > 0`), each source device sends
each unique token at most once per remote board over the inter-board links
into a gateway device's `stage_hbm` buffer, partitioned by source slot
`sigma` and issue wave `w`:

```
  stage_hbm [S * stage_stride, lane_blocks, 128]:
  +-------------------------------------------------------------------+
  | slot sigma = 0:  | wave 0 runs | wave 1 runs | ... | wave W-1     |
  +-------------------------------------------------------------------+
  | slot sigma = 1:  | wave 0 runs | wave 1 runs | ... | wave W-1     |
  +-------------------------------------------------------------------+
  | ...                                                               |
  +-------------------------------------------------------------------+
  | slot sigma = S-1:| wave 0 runs | wave 1 runs | ... | wave W-1     |
  +-------------------------------------------------------------------+
           |
           |  Gateway forward (`forward_staged_wave_body`)
           |  * Slot 0: ready immediately (rows arrived in earlier stage waves)
           |  * Wait `stage_sems[sigma * W + w]` (byte-exact stage arrival)
           |  * Slot 1: newly arrived stage rows for wave `w`
           |  * Each run `r` in `fwd_tab_sm` specifies:
           |      (STAGE_ROW, ROWS, DST = target_dev | (wave_off << 16),
           DST_ROW)
           v
  lhs_hbm [ep * recv_stride, lane_blocks, 128] on target device `d`
  (signals `wave_sems[w + wave_off]` on `d` over the intra-board links)
```

Fixed-slot receive buffer (`lhs_hbm`) and return landing buffer (`recv_hbm`)
----------------------------------------------------------------------------
1. **Receive buffer (`lhs_hbm`)**: Indexed by `token_gather` streamed through
   two alternating SMEM windows (`token_gather_sm0`, `token_gather_sm1`):

```
  token_gather_hbm (aligned 128-int32 DMA windows + 1-tile overfetch runway):
  | window 0 (tile t) | overfetch || window 1 (tile t+1) | overfetch |
           |
           +---> `lhs_issue_tile`: gathers `ROWBLK`-aligned rows from `lhs_hbm`
                 into `lhs_vm[parity, 0..live_rows, lane_blocks, 128]`.
```

2. **Return landing buffer (`recv_hbm` and `rscl_hbm`)**: Computed expert
   outputs leave `out_vm[parity]` either directly (`push_from_vmem`, small
   buckets) or via `contrib_hbm` (`commit_tile` + progressive `push_run`, large
   buckets) and land at their aligned destination offset `push_dst_sm[e, d]`:

```
  out_vm[parity]  --(ship_tile or commit_tile + push_run)-->  recv_hbm[dst_row]
  oscl_vm[parity] --(sublane scale mirror DMA)------------->  rscl_hbm[dst_sub]
```
"""

import functools
from typing import Any

from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import config
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import plan as plan_lib
from tokamax._src.ops.experimental.fused_moe import host as fused_host
from tokamax._src.ops.experimental.fused_moe import kernel as fused_kernel

# Small DMA helpers shared with `fused_moe.kernel`.
# pylint: disable=protected-access
slot_copy = fused_kernel._slot_copy
rows_wait = fused_kernel._rows_wait
and_nonempty = fused_kernel._and_nonempty
# pylint: enable=protected-access


def all_pairs_barrier(ep):
  """Barrier over every shard: transport peers are not only neighbours.

  Same as `fused_moe.kernel._all_pairs_barrier`, but addresses the peers by
  their position on the layer's single mesh axis.

  Args:
    ep: Expert-parallel width.
  """
  barrier_sem = pltpu.get_barrier_semaphore()
  for i in range(ep):
    pl.semaphore_signal(
        barrier_sem,
        inc=1,
        device_id=(jnp.int32(i),),
    )
  pl.semaphore_wait(barrier_sem, ep)

  @functools.partial(pl.run_scoped, second=pltpu.SemaphoreType.REGULAR)
  def _(second):
    for i in range(ep):
      pl.semaphore_signal(
          second,
          inc=1,
          device_id=(jnp.int32(i),),
      )
    pl.semaphore_wait(second, ep)


def rows_wait_send(sem, ref, rows):
  """Block until `rows` rows' worth of outgoing remote DMAs on `sem` have sent.

  A gateway may forward each staged row to every other device of its board, so
  `rows` can exceed `ref.shape[0]`. Draining in chunks of `ref.shape[0]` keeps
  every slice inside `ref` in interpret mode while waiting the exact total
  byte count on `sem`.

  Args:
    sem: The send semaphore.
    ref: A ref whose leading rows give the row size.
    rows: Rows to wait for.
  """
  cap = jnp.int32(ref.shape[0])
  full = lax.div(rows, cap)
  rem = lax.rem(rows, cap)

  def _body(_, c):
    rows_wait(sem, ref, cap)
    return c

  lax.fori_loop(0, full, _body, jnp.int32(0))

  @pl.when(rem > 0)
  def _():
    rows_wait(sem, ref, rem)


def forward_staged_wave_body(
    w,
    *,
    me,
    stage_tab_sm,
    fwd_tab_sm,
    fwd_cnt_sm,
    stage_sems,
    stage_hbm,
    lhs_hbm,
    wave_sems,
    fwd_send_sem,
    num_slots,
    num_waves,
    num_boards,
    fwd_max_runs,
):
  """De-Dup dispatch: forwards the runs of issue wave `w` within the board.

  The runs go over the intra-board links into `lhs_hbm`
  (dedup_plan.FwdSchedule).

  Ready slot 0 of every served source first (its rows crossed in earlier
  stage waves, already waited), then, source by source, wait that source's
  stage run of wave w byte-exact and issue its slot 1. A run's dst field
  carries its signal wave as an offset above FWD_DST_SHIFT: the copy lands
  on the target's `wave_sems[w + offset]`, the target's own first wave for
  the row (per-target-wave forwarding).

  The three dedup tables arrive flattened to 1-D (row-major; see the
  wrapper's `dedup_smem_args`): Mosaic pads the minor dim of a rank >= 2
  SMEM array to 128 words, which made the [S*W*128, 4] forward table 32x
  its size in SMEM.

  Args:
    w: The issue wave (traced int32).
    me: This device (traced int32).
    stage_tab_sm: Flat [(2 nb + S + 2) * W] stage table (expected stage rows of
      (sigma, w) at row 2 nb + sigma).
    fwd_tab_sm: Flat [S * W * fwd_max_runs * FWD_FIELDS] forward table.
    fwd_cnt_sm: Flat [S * W * FWD_SLOTS] runs per (sigma, w, ready slot).
    stage_sems: [S * W] stage semaphores.
    stage_hbm: The stage buffer.
    lhs_hbm: The receive buffer (local copies land here too).
    wave_sems: [W] receive-wave semaphores.
    fwd_send_sem: Send semaphore of the remote forwards.
    num_slots: S.
    num_waves: W.
    num_boards: nb.
    fwd_max_runs: Forward-table depth per (sigma, w).
  """
  fwd_slots = dedup_plan_mod.FWD_SLOTS

  def issue_slot(sig, slot):
    cnt_base = (sig * num_waves + w) * fwd_slots
    first = jnp.int32(0)
    for k in range(slot):
      first = first + fwd_cnt_sm[cnt_base + k]
    n_runs = fwd_cnt_sm[cnt_base + slot]
    tab_base = (sig * num_waves + w) * fwd_max_runs + first

    def issue_fwd(r, _):
      base = (tab_base + r) * dedup_plan_mod.FWD_FIELDS
      s_row = fwd_tab_sm[base + dedup_plan_mod.FWD_STAGE_ROW]
      m = fwd_tab_sm[base + dedup_plan_mod.FWD_ROWS]
      d_field = fwd_tab_sm[base + dedup_plan_mod.FWD_DST]
      d_row = fwd_tab_sm[base + dedup_plan_mod.FWD_DST_ROW]
      d = d_field & dedup_plan_mod.FWD_DST_MASK
      sem = wave_sems.at[
          w + jnp.right_shift(d_field, dedup_plan_mod.FWD_DST_SHIFT)
      ]
      src = stage_hbm.at[pl.ds(s_row, m)]
      dst = lhs_hbm.at[pl.ds(d_row, m)]
      is_me = d == me

      @pl.when(and_nonempty(is_me, m))
      def _():
        pltpu.make_async_copy(src, dst, sem).start()

      @pl.when(and_nonempty(jnp.logical_not(is_me), m))
      def _():
        pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=dst,
            send_sem=fwd_send_sem,
            recv_sem=sem,
            device_id=(d,),
        ).start()

      return jnp.int32(0)

    lax.fori_loop(0, n_runs, issue_fwd, jnp.int32(0))

  # Step 1: Issue ready slot 0 for all served source slots (rows from
  # earlier waves).
  for sig in range(num_slots):
    issue_slot(sig, 0)
  # Step 2: Wait for each source slot's stage run of wave `w` and issue slot 1.
  for sig in range(num_slots):
    n = stage_tab_sm[(2 * num_boards + sig) * num_waves + w]

    @pl.when(n > 0)
    def _(sig=sig, rows=n):
      rows_wait(stage_sems.at[sig * num_waves + w], lhs_hbm, rows)

    issue_slot(sig, 1)


def build_scratch_shapes(cfg: config.FusedKernelConfig) -> tuple[Any, ...]:
  """Builds the ordered SMEM, VMEM and semaphore scratch shapes for `cfg`.

  Args:
    cfg: The resolved FusedKernelConfig.

  Returns:
    The scratch shapes, in the order the kernel body unpacks them.
  """
  out_parities = fused_host.OUT_PARITIES
  nbuf = cfg.nbuf
  scratch = [
      # Two independent scalar-memory windows keep dynamic slot selection
      # out of the per-token loop. Their total is fixed for the 128-row,
      # four-tile schedule, regardless of request size.
      pltpu.SMEM((cfg.gather_window_rows,), jnp.int32),
      pltpu.SMEM((cfg.gather_window_rows,), jnp.int32),
  ]
  scratch += [
      pltpu.VMEM(shape, dtype)
      for _, shape, dtype in plan_lib.vmem_scratch_arrays(
          cfg.g_local,
          cfg.capacity,
          cfg.hidden,
          cfg.inter,
          nbuf=nbuf,
          weight_format=cfg.weight_format,
          has_w1_bias=cfg.has_w1_bias,
          has_w2_bias=cfg.has_w2_bias,
      )
  ]
  scratch += [
      pltpu.SemaphoreType.DMA,  # token_gather_sem0
      pltpu.SemaphoreType.DMA,  # token_gather_sem1
      # lhs_sems. Parity-deep, not nbuf-deep: every index into it, and into
      # the lhs_vm and ls_vm buffers it guards, is an out-parity.
      pltpu.SemaphoreType.DMA((out_parities,)),
      pltpu.SemaphoreType.DMA((nbuf,)),  # w1_sems
      pltpu.SemaphoreType.DMA((nbuf,)),  # w2_sems
  ]
  if cfg.has_scales:
    # The scale tables travel with the slabs they scale, so they are
    # slot-deep and need their own sems: a wait is per-buffer.
    scratch += [
        pltpu.SemaphoreType.DMA((nbuf,)),  # w1s_sems
        pltpu.SemaphoreType.DMA((nbuf,)),  # w2s_sems
    ]
  scratch.append(pltpu.SemaphoreType.DMA)  # cp_sem
  if cfg.push_from_vmem:
    scratch += [
        pltpu.SemaphoreType.DMA((out_parities,)),  # send_sems
        pltpu.SemaphoreType.DMA,  # recv_sem
        pltpu.SemaphoreType.DMA((out_parities,)),  # send_scl_sems
        pltpu.SemaphoreType.DMA,  # recv_scl_sem
    ]
  else:
    scratch += [
        pltpu.SemaphoreType.DMA((out_parities,)),  # commit_sems
        pltpu.SemaphoreType.DMA,  # send_sem
        pltpu.SemaphoreType.DMA,  # recv_sem
        pltpu.SemaphoreType.DMA((out_parities,)),  # commit_scl_sems
        pltpu.SemaphoreType.DMA,  # send_scl_sem
        pltpu.SemaphoreType.DMA,  # recv_scl_sem
        pltpu.SemaphoreType.DMA,  # mehop_sem
        pltpu.SemaphoreType.DMA,  # mehop_scl_sem
    ]
  if cfg.wait_for_exchange:
    scratch += [
        # disp_tab_sm, flat [4 * ep * W]: a rank-2 SMEM array pads its
        # minor dim to 128 words.
        pltpu.SMEM((4 * cfg.ep * cfg.waves,), jnp.int32),
        pltpu.SMEM((1,), jnp.int32),  # waves_done_sm
    ]
  if cfg.dedup:
    scratch += [
        pltpu.SMEM((1,), jnp.int32),  # fwd_done_sm
        pltpu.SemaphoreType.DMA,  # fwd_send_sem
    ]
  return tuple(scratch)
