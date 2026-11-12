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

"""Main kernel module for orchestrating DMA and computation pipeline.

Separates the stateful prologue, inner per-expert tile pipeline,
progressive/VMEM
return pushes, and epilogue transport drain from the stateless expert FFN
compute (shared with `fused_moe.kernel`) and the memory reference helpers
(`memory_ref.py`).
"""

import functools
from typing import Any

from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import config
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import memory_ref
from tokamax._src.ops.experimental.fused_moe import host as fused_host
from tokamax._src.ops.experimental.fused_moe import kernel as fused_kernel


def make_fused_ep_moe_kernel_body(cfg: config.FusedKernelConfig):
  """Builds the Pallas kernel body for a resolved config.

  Args:
    cfg: The resolved FusedKernelConfig.

  Returns:
    The kernel function `kernel(*refs)` that pl.pallas_call runs.
  """
  g_local = cfg.g_local
  hidden = cfg.hidden
  ep = cfg.ep
  weight_format = cfg.weight_format
  act_fn = cfg.act_fn
  has_w1_bias = cfg.has_w1_bias
  has_w2_bias = cfg.has_w2_bias
  waves = cfg.waves
  dedup_slots = cfg.dedup_slots
  wait_for_exchange = cfg.wait_for_exchange
  dedup = cfg.dedup
  fwd_max_runs = cfg.fwd_max_runs
  nb = cfg.nb
  wave_bounds_base = cfg.wave_bounds_base
  form = cfg.form
  skip_scale_mirror = cfg.skip_scale_mirror
  lane_blocks = cfg.lane_blocks
  has_scales = cfg.has_scales
  has_act_scale = cfg.has_act_scale
  has_ls_vm = cfg.has_ls_vm
  prefetch_distance = cfg.prefetch_distance
  nbuf = cfg.nbuf
  prefetches = cfg.prefetches
  tile_m = cfg.tile_m
  tile_blocks = cfg.tile_blocks
  tile_block_shift = cfg.tile_block_shift
  tile_shift = cfg.tile_shift
  height_rungs = cfg.height_rungs
  push_from_vmem = cfg.push_from_vmem
  progressive_push = cfg.progressive_push
  ls_window_rows = cfg.ls_window_rows
  gather_tiles = cfg.gather_tiles
  gather_payload_rows = cfg.gather_payload_rows
  gather_window_rows = cfg.gather_window_rows

  rowblk = fused_host.ROWBLK
  lanes = fused_host.HIDDEN_LANE_BLOCK
  out_parities = fused_host.OUT_PARITIES
  weight_dma_priority = config.WEIGHT_DMA_PRIORITY

  def kernel(*refs):
    # Step 1: Unpack SMEM scalar-prefetch tables, HBM operands, and
    # VMEM/DMA scratch refs.
    it = iter(refs)
    absent: Any = None

    def take(present: bool = True) -> Any:
      """The next ref when the build has it, else None."""
      return next(it) if present else absent

    # Block-unit transport tables, from shard_transport_tables_in_blocks.
    # Three of that builder's tables are the same array as another it
    # already ships, so each arrives once and the duplicates are derived
    # here instead of being prefetched again:
    #   push_src == contrib_off  -- a push reads its source at the offset
    #     the commit wrote it to;
    #   push_len == commit_len   -- `region_rows` IS `run_rows_aligned`
    #     reshaped, so slicing either at this shard gives one array;
    #   recv_row_off == push_dst << ROWBLK_SHIFT -- `recv_base` is a cumsum
    #     of ALIGNED run lengths, so the row offset is the block offset in
    #     rows.
    commit_start_sm, commit_len_sm, contrib_off_sm, push_dst_sm = (
        next(it) for _ in range(config.N_TRANSPORT_TABLES)
    )
    # Row-unit push tables, from shard_push_tables_in_rows.
    (true_rows_sm,) = (next(it) for _ in range(config.N_PUSH_TABLES))
    # (rows, base) i32 [G] row-unit tables from shard_expert_slabs.
    expert_rows_sm, expert_base_sm = (
        next(it) for _ in range(config.N_SLAB_TABLES)
    )
    # visit_sm[visit_i] is the real expert at compacted step visit_i.
    visit_sm = next(it)
    # The five counts, each still spread over the expert-parallel axis.
    counts_sm = next(it)
    # This shard's expert-parallel index, as data rather than `lax.axis_index`.
    rank_sm = next(it)
    if dedup:
      stage_tab_sm = next(it)
      fwd_tab_sm = next(it)
      fwd_cnt_sm = next(it)
    else:
      stage_tab_sm = fwd_tab_sm = fwd_cnt_sm = absent

    def count(row):
      """Close one count's reduction on the scalar core."""
      total = counts_sm[row, 0]
      for d in range(1, ep):
        total = total + counts_sm[row, d]
      return total

    n_visit = count(fused_host.COUNT_VISITS)
    token_gather_hbm = next(it)
    lhs_hbm = next(it)
    disp_tab_hbm = take(wait_for_exchange)
    ls_hbm = take(has_ls_vm)
    w1_hbm, w2_hbm = (next(it) for _ in range(2))
    w1s_hbm = take(has_scales)
    w2s_hbm = take(has_scales)
    w1b_hbm = take(has_w1_bias)
    w2b_hbm = take(has_w2_bias)
    stage_hbm = take(dedup)
    wave_sems = take(wait_for_exchange)
    stage_sems = take(dedup)
    recv_hbm = next(it)
    rscl_hbm = next(it)
    contrib_hbm = next(it)
    cscl_hbm = next(it)
    token_gather_sm0, token_gather_sm1 = (next(it) for _ in range(2))
    lhs_vm, w1_vm, w2_vm = (next(it) for _ in range(3))
    w1s_vm = take(has_scales)
    w2s_vm = take(has_scales)
    w1b_vm = take(has_w1_bias)
    w2b_vm = take(has_w2_bias)
    ls_vm = take(has_ls_vm)
    out_vm = next(it)
    oscl_vm = next(it)
    token_gather_sem0, token_gather_sem1 = (next(it) for _ in range(2))
    lhs_sems, w1_sems, w2_sems = (next(it) for _ in range(3))
    w1s_sems = take(has_scales)
    w2s_sems = take(has_scales)
    cp_sem = next(it)
    if push_from_vmem:
      send_sems, recv_sem = (next(it) for _ in range(2))
      send_scl_sems, recv_scl_sem = (next(it) for _ in range(2))
      commit_sems = commit_scl_sems = absent
      mehop_sem = mehop_scl_sem = absent
    else:
      commit_sems, send_sem, recv_sem = (next(it) for _ in range(3))
      commit_scl_sems, send_scl_sem, recv_scl_sem = (next(it) for _ in range(3))
      mehop_sem = next(it)
      mehop_scl_sem = next(it)
    disp_tab_sm, waves_done_sm = (
        take(wait_for_exchange),
        take(wait_for_exchange),
    )
    if dedup:
      fwd_done_sm = next(it)
      fwd_send_sem = next(it)
    else:
      fwd_done_sm = fwd_send_sem = absent

    me = rank_sm[0, 0]

    def sync(src, dst):
      """Copy `src` into `dst` and wait for it before returning."""
      copy = pltpu.make_async_copy(src, dst, cp_sem)
      copy.start()
      copy.wait()

    def w1_copy(expert, slot):
      """The gate/up weight slab of `expert`, into weight slot `slot`."""
      return memory_ref.slot_copy(w1_hbm.at[expert], w1_vm, w1_sems, slot)

    def w2_copy(expert, slot):
      """The down weight slab of `expert`, into weight slot `slot`."""
      return memory_ref.slot_copy(w2_hbm.at[expert], w2_vm, w2_sems, slot)

    def w1s_copy(expert, slot):
      return memory_ref.slot_copy(w1s_hbm.at[expert], w1s_vm, w1s_sems, slot)

    def w2s_copy(expert, slot):
      return memory_ref.slot_copy(w2s_hbm.at[expert], w2s_vm, w2s_sems, slot)

    def weight_starts(expert, slot):
      """Start every buffer slot `slot` holds for `expert`."""
      if has_scales:
        w1s_copy(expert, slot).start(priority=weight_dma_priority)
        w2s_copy(expert, slot).start(priority=weight_dma_priority)
      w1_copy(expert, slot).start(priority=weight_dma_priority)
      w2_copy(expert, slot).start(priority=weight_dma_priority)

    def weight_waits(expert, slot):
      """Wait for every buffer slot `slot` holds for `expert`."""
      w1_copy(expert, slot).wait()
      w2_copy(expert, slot).wait()
      if has_scales:
        w1s_copy(expert, slot).wait()
        w2s_copy(expert, slot).wait()

    def prologue():
      """Land the resident tables and start the first weight refills."""
      if has_w1_bias:
        sync(w1b_hbm, w1b_vm)
      if has_w2_bias:
        sync(w2b_hbm, w2b_vm)
      n_prologue = min(prefetch_distance, g_local)
      for b in range(n_prologue):

        @pl.when(jnp.int32(b) < n_visit)
        def _(b=b):
          weight_starts(visit_sm[b], b)

    def forward_staged_wave(w):
      """De-Dup dispatch: forward_staged_wave_body for `w`."""
      assert stage_tab_sm is not None
      assert fwd_tab_sm is not None
      assert fwd_cnt_sm is not None
      assert stage_sems is not None
      memory_ref.forward_staged_wave_body(
          w,
          me=me,
          stage_tab_sm=stage_tab_sm,
          fwd_tab_sm=fwd_tab_sm,
          fwd_cnt_sm=fwd_cnt_sm,
          stage_sems=stage_sems,
          stage_hbm=stage_hbm,
          lhs_hbm=lhs_hbm,
          wave_sems=wave_sems,
          fwd_send_sem=fwd_send_sem,
          num_slots=dedup_slots,
          num_waves=waves,
          num_boards=nb,
          fwd_max_runs=fwd_max_runs,
      )

    def forward_staged_through(last):
      """De-Dup dispatch: forward every wave up to `last` not yet forwarded."""
      assert fwd_done_sm is not None
      first = fwd_done_sm[0]

      def body(w, c):
        forward_staged_wave(w)
        return c

      lax.fori_loop(first, last + 1, body, jnp.int32(0))
      fwd_done_sm[0] = jnp.maximum(first, last + 1)

    def wait_wave(w):
      """Wait for all of wave ``w``'s rows."""
      rows = disp_tab_sm[3 * ep * waves + w]

      @pl.when(rows > 0)
      def _():
        memory_ref.rows_wait(wave_sems.at[w], lhs_hbm, rows)

    def wait_waves_through(last):
      """Wait every wave up to ``last`` not yet waited."""
      if dedup:
        forward_staged_through(last)
      first = waves_done_sm[0]

      def body(w, c):
        wait_wave(w)
        return c

      lax.fori_loop(first, last + 1, body, jnp.int32(0))
      waves_done_sm[0] = jnp.maximum(first, last + 1)

    # Prologue: Load biases, start initial weight prefetches, sync the
    # dispatch table, execute the all-pairs barrier, and kick off the wave 0
    # gateway forwards.
    prologue()
    if wait_for_exchange:
      sync(disp_tab_hbm, disp_tab_sm)
      waves_done_sm[0] = jnp.int32(0)
    if dedup:
      assert fwd_done_sm is not None
      fwd_done_sm[0] = jnp.int32(0)
    memory_ref.all_pairs_barrier(ep)
    if dedup:
      # Kick wave 0's gateway forwards immediately after the barrier so wave 0's
      # forwarded rows are in flight before `prime_expert(visit_sm[0])`
      # waits on `wave_sems[0]`.
      forward_staged_through(jnp.int32(0))

    # ---- ragged tile machinery ----
    def gather_dma_coords(logical_base):
      """Aligned HBM base and in-window offset for a slab row."""
      alignment = jnp.int32(fused_host.TOKEN_GATHER_DMA_ALIGNMENT)
      shift = lax.rem(logical_base, alignment)
      dma_base = pl.multiple_of(
          logical_base - shift, fused_host.TOKEN_GATHER_DMA_ALIGNMENT
      )
      return dma_base, shift

    def issue_gather_window(logical_base, slot):
      """Start one fixed token-index window in SMEM slot ``slot``."""
      dma_base, _ = gather_dma_coords(logical_base)

      @pl.when(slot == 0)
      def _():
        pltpu.make_async_copy(
            token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
            token_gather_sm0,
            token_gather_sem0,
        ).start(priority=0)

      @pl.when(slot == 1)
      def _():
        pltpu.make_async_copy(
            token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
            token_gather_sm1,
            token_gather_sem1,
        ).start(priority=0)

    def wait_gather_window(logical_base, slot):
      """Wait for the fixed token-index window in ``slot``."""
      dma_base, _ = gather_dma_coords(logical_base)

      @pl.when(slot == 0)
      def _():
        pltpu.make_async_copy(
            token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
            token_gather_sm0,
            token_gather_sem0,
        ).wait()

      @pl.when(slot == 1)
      def _():
        pltpu.make_async_copy(
            token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
            token_gather_sm1,
            token_gather_sem1,
        ).wait()

    def lhs_issue_tile(row_base, live_blocks, slot, gather_slot, gather_offset):
      """Issue one tile's activations using a resident index window."""

      def issue_from(gather_smem):
        def issue_block(i, _):
          """Fetch rowblk token rows of block i by their numbers."""
          for r in range(rowblk):
            token = gather_smem[gather_offset + i * rowblk + r]
            pltpu.make_async_copy(
                lhs_hbm.at[token],
                lhs_vm.at[slot, i * rowblk + r],
                lhs_sems.at[slot],
            ).start()
          return _

        lax.fori_loop(0, live_blocks, issue_block, jnp.int32(0))

      @pl.when(gather_slot == 0)
      def _():
        issue_from(token_gather_sm0)

      @pl.when(gather_slot == 1)
      def _():
        issue_from(token_gather_sm1)

      if has_ls_vm:
        pltpu.make_async_copy(
            ls_hbm.at[pl.ds(row_base // lanes, ls_window_rows)],
            ls_vm.at[slot],
            lhs_sems.at[slot],
        ).start()

    def lhs_ready_tile(live_rows, slot):
      """Wait the token rows and the scale window of one tile."""
      pltpu.make_async_copy(
          lhs_vm.at[slot, pl.ds(0, live_rows)],
          lhs_vm.at[slot, pl.ds(0, live_rows)],
          lhs_sems.at[slot],
      ).wait()
      if has_ls_vm:
        pltpu.make_async_copy(
            ls_vm.at[slot], ls_vm.at[slot], lhs_sems.at[slot]
        ).wait()

    def _mirror_base(e):
      """Sublane expert e's tiles start at in the contribution mirror."""
      return jnp.right_shift(expert_base_sm[e], tile_shift) + 2 * e

    def _run_tiles(e, d, region_blocks):
      """A run's mirror sublanes: (source, destination, count)."""
      start = commit_start_sm[e, d]
      lo = jnp.right_shift(start, tile_block_shift)
      hi = jnp.right_shift(start + region_blocks - 1, tile_block_shift)
      dst = jnp.right_shift(push_dst_sm[e, d], tile_block_shift) + 2 * (
          me * g_local + e
      )
      return _mirror_base(e) + lo, dst, hi - lo + 1

    def wait_commits(parity, live_blocks):
      """Wait one tile's commits on `parity`: data rows, then scales."""
      memory_ref.rows_wait(
          commit_sems.at[parity], contrib_hbm, live_blocks * rowblk
      )
      if not skip_scale_mirror:
        memory_ref.rows_wait(commit_scl_sems.at[parity], cscl_hbm, 1)

    def wait_shipped(parity, rows, subs):
      """Wait until the wire has read `parity`'s staging slot."""

      @pl.when(rows > 0)
      def _():
        memory_ref.rows_wait(send_sems.at[parity], contrib_hbm, rows)

      if not skip_scale_mirror:

        @pl.when(subs > 0)
        def _():
          memory_ref.rows_wait(send_scl_sems.at[parity], cscl_hbm, subs)

    def prime_expert(e, lhs_slot, wslot, window_started, visit_i):
      """Prime expert ``e``'s metadata and first activation tile."""
      if wait_for_exchange:
        last = jnp.int32(0)
        for w in range(waves - 1):
          last = last + (visit_i >= disp_tab_sm[wave_bounds_base + w]).astype(
              jnp.int32
          )
        wait_waves_through(last)
      rows = expert_rows_sm[e]
      slab_base = expert_base_sm[e]

      @pl.when(jnp.logical_not(window_started))
      def _():
        issue_gather_window(slab_base, wslot)

      wait_gather_window(slab_base, wslot)
      _, shift = gather_dma_coords(slab_base)
      lhs_issue_tile(
          slab_base, jnp.minimum(rows, tile_m) // rowblk, lhs_slot, wslot, shift
      )

      @pl.when(rows > gather_payload_rows)
      def _():
        issue_gather_window(
            slab_base + gather_payload_rows, jnp.int32(1) - wslot
        )

    def expert_tiles(e, visit_i, carry):
      """One expert step: a fori_loop over its [tile_m, H] tiles."""
      # Step 1: Resolve weight slot, wait for current expert's weights, and
      # issue weight prefetch for `visit_i + prefetch_distance`.
      slot = lax.rem(visit_i, jnp.int32(nbuf))
      if not prefetches:
        weight_starts(e, slot)
      weight_waits(e, slot)

      if prefetches:

        @pl.when(visit_i + prefetch_distance < n_visit)
        def _():
          refill_slot = lax.rem(visit_i + prefetch_distance, jnp.int32(nbuf))
          ahead = visit_sm[visit_i + prefetch_distance]
          weight_starts(ahead, refill_slot)

      rows = expert_rows_sm[e]
      slab_base = expert_base_sm[e]
      n_tiles = -(-rows // tile_m)
      wslot = carry[-1]
      n_groups = -(-n_tiles // gather_tiles)
      next_wslot = lax.rem(wslot + n_groups, jnp.int32(2))

      def commit_tile(parity, t, tile_block_base, live_blocks):
        """Commit the tile's intersection with each (expert, dest) run."""
        for d in range(ep):
          run_start = commit_start_sm[e, d]
          run_len = commit_len_sm[e, d]
          lo = jnp.maximum(run_start, tile_block_base)
          hi = jnp.minimum(run_start + run_len, tile_block_base + live_blocks)
          overlap = jnp.maximum(hi - lo, 0)
          lo = jnp.minimum(lo, hi)
          src_block = lo - tile_block_base
          dst_block = contrib_off_sm[e, d] + (lo - run_start)

          @pl.when(overlap > 0)
          def _(
              src_block=src_block,
              dst_block=dst_block,
              overlap=overlap,
              parity=parity,
          ):
            pltpu.make_async_copy(
                out_vm.at[parity, pl.ds(src_block * rowblk, overlap * rowblk)],
                contrib_hbm.at[pl.ds(dst_block * rowblk, overlap * rowblk)],
                commit_sems.at[parity],
            ).start()

        if not skip_scale_mirror:
          pltpu.make_async_copy(
              oscl_vm.at[parity],
              cscl_hbm.at[pl.ds(_mirror_base(e) + t, 1)],
              commit_scl_sems.at[parity],
          ).start()

      def ship_tile(parity, t, tile_block_base, live_blocks):
        """Ship the tile straight out of `out_vm`, no slab in between."""
        rows_out = jnp.int32(0)
        subs_out = jnp.int32(0)
        for d in range(ep):
          run_start = commit_start_sm[e, d]
          run_len = commit_len_sm[e, d]
          lo = jnp.maximum(run_start, tile_block_base)
          hi = jnp.minimum(run_start + run_len, tile_block_base + live_blocks)
          overlap = jnp.maximum(hi - lo, 0)
          lo = jnp.minimum(lo, hi)
          src_row = (lo - tile_block_base) * rowblk
          into_run = (lo - run_start) * rowblk
          dst_row = push_dst_sm[e, d] * rowblk + into_run
          true_here = jnp.clip(
              true_rows_sm[e, d] - into_run, 0, overlap * rowblk
          )
          is_me = jnp.int32(d) == me
          dst_sub = (
              jnp.right_shift(push_dst_sm[e, d], tile_block_shift)
              + 2 * (me * g_local + e)
              + t
              - jnp.right_shift(run_start, tile_block_shift)
          )

          @pl.when(memory_ref.and_nonempty(jnp.logical_not(is_me), true_here))
          def _(
              src_row=src_row,
              dst_row=dst_row,
              true_here=true_here,
              parity=parity,
              d=d,
          ):
            pltpu.make_async_remote_copy(
                src_ref=out_vm.at[parity, pl.ds(src_row, true_here)],
                dst_ref=recv_hbm.at[pl.ds(dst_row, true_here)],
                send_sem=send_sems.at[parity],
                recv_sem=recv_sem,
                device_id=(jnp.int32(d),),
            ).start()

          @pl.when(memory_ref.and_nonempty(is_me, overlap))
          def _(
              src_row=src_row, dst_row=dst_row, overlap=overlap, parity=parity
          ):
            pltpu.make_async_copy(
                out_vm.at[parity, pl.ds(src_row, overlap * rowblk)],
                recv_hbm.at[pl.ds(dst_row, overlap * rowblk)],
                send_sems.at[parity],
            ).start()

          if not skip_scale_mirror:

            @pl.when(memory_ref.and_nonempty(jnp.logical_not(is_me), overlap))
            def _(dst_sub=dst_sub, parity=parity, d=d):
              pltpu.make_async_remote_copy(
                  src_ref=oscl_vm.at[parity],
                  dst_ref=rscl_hbm.at[pl.ds(dst_sub, 1)],
                  send_sem=send_scl_sems.at[parity],
                  recv_sem=recv_scl_sem,
                  device_id=(jnp.int32(d),),
              ).start()

            @pl.when(memory_ref.and_nonempty(is_me, overlap))
            def _(dst_sub=dst_sub, parity=parity):
              pltpu.make_async_copy(
                  oscl_vm.at[parity],
                  rscl_hbm.at[pl.ds(dst_sub, 1)],
                  send_scl_sems.at[parity],
              ).start()

          rows_out = rows_out + jnp.where(is_me, overlap * rowblk, true_here)
          if not skip_scale_mirror:
            subs_out = subs_out + jnp.where(overlap > 0, 1, 0)
        return rows_out, subs_out

      def tile_body(t, carried):
        """Compute tile t, stage it for the wire and ship it."""
        tile_count = carried[0]
        row_base = slab_base + t * tile_m
        live_rows = jnp.minimum(rows - t * tile_m, tile_m)
        live_blocks = live_rows // rowblk
        tile_block_base = t * tile_blocks
        parity = lax.rem(tile_count, jnp.int32(out_parities))

        # Step 1: Guard `out_vm[parity]` reuse and issue progressive pushes.
        if push_from_vmem:
          wait_shipped(
              parity,
              jnp.where(parity == 0, carried[1], carried[2]),
              jnp.where(parity == 0, carried[3], carried[4]),
          )
        else:
          pending = jnp.where(parity == 0, carried[1], carried[2])

          @pl.when(pending > 0)
          def _():
            wait_commits(parity, pending)

          if progressive_push:

            @pl.when(t >= 2)
            def _():
              push_runs_ending_in(e, t - 2)

        # Step 2: Wait for current tile's LHS rows/scales and prefetch tile t+1.
        lhs_ready_tile(live_rows, parity)
        other_parity = jnp.int32(1) - parity

        def issue_next_tile():
          """Issue tile t+1's activations into the other slot."""
          next_t = t + 1
          next_group = next_t // gather_tiles
          next_within = lax.rem(next_t, jnp.int32(gather_tiles))
          next_group_slot = lax.rem(wslot + next_group, jnp.int32(2))
          next_group_base = slab_base + next_group * gather_payload_rows
          _, next_shift = gather_dma_coords(next_group_base)
          next_rows = jnp.minimum(rows - next_t * tile_m, tile_m)
          lhs_issue_tile(
              slab_base + next_t * tile_m,
              next_rows // rowblk,
              other_parity,
              next_group_slot,
              next_shift + next_within * tile_m,
          )

        @pl.when(t + 1 < n_tiles)
        def _():
          next_t = t + 1
          next_group = next_t // gather_tiles
          next_within = lax.rem(next_t, jnp.int32(gather_tiles))
          next_group_slot = lax.rem(wslot + next_group, jnp.int32(2))
          next_group_base = slab_base + next_group * gather_payload_rows

          @pl.when(next_within == 0)
          def _():
            wait_gather_window(next_group_base, next_group_slot)
            following = next_group + 1

            @pl.when(following * gather_tiles < n_tiles)
            def _():
              issue_gather_window(
                  slab_base + following * gather_payload_rows,
                  jnp.int32(1) - next_group_slot,
              )

          issue_next_tile()

        @pl.when(jnp.logical_and(t + 1 == n_tiles, visit_i + 1 < n_visit))
        def _():
          issue_gather_window(expert_base_sm[visit_sm[visit_i + 1]], next_wslot)

        # Step 3: Run stateless expert FFN compute at the active tail-height
        # rung.
        def compute_tile(height):
          """Run the FFN on the tile's leading `height` rows."""
          full = height == tile_m
          act_rows = (
              lhs_vm[parity] if full else lhs_vm[parity, pl.ds(0, height)]
          ).reshape(height, hidden)
          ls_col = None
          if has_ls_vm:
            ls_col = fused_kernel._tile_row_scales(  # pylint: disable=protected-access
                ls_vm[parity], row_base, height
            )
          act_scales = ls_col if has_act_scale else None
          w1b_row = w1b_vm[pl.ds(e, 1), :] if has_w1_bias else None
          if weight_format == arg.WeightFormat.BF16:
            acc2, mid_scale = fused_kernel.expert_ffn_bf16(
                act_rows,
                w1_vm[slot],
                w2_vm[slot],
                act_fn=act_fn,
                w1b=w1b_row,
            )
          else:
            acc2, mid_scale = fused_kernel.expert_ffn_fp8(
                act_rows,
                act_scales,
                w1_vm[slot],
                w2_vm[slot],
                w1s_vm[pl.ds(slot, 1), :],
                act_fn=act_fn,
                w1b=w1b_row,
            )
          wire_rows, wire_scales = fused_kernel.wire_row_and_scale(
              acc2,
              mid_scale,
              w2s=w2s_vm[pl.ds(slot, 1), :] if has_scales else None,
              w2b=w2b_vm[pl.ds(e, 1), :] if has_w2_bias else None,
              wire_dtype=form.wire_dtype,
              tile_m=height,
          )
          wire_rows_staged = wire_rows.reshape(height, lane_blocks, lanes)
          wire_scales_staged = wire_scales.reshape(1, height)

          def store_parity(static_parity):
            """Stage the tile's rows and scales in one out_vm slot."""
            if full:
              out_vm[static_parity] = wire_rows_staged
              oscl_vm[static_parity] = wire_scales_staged
            else:
              out_vm[static_parity, pl.ds(0, height)] = wire_rows_staged
              oscl_vm[static_parity, :, pl.ds(0, height)] = wire_scales_staged

          @pl.when(parity == 0)
          def _():
            store_parity(0)

          @pl.when(parity == 1)
          def _():
            store_parity(1)

        if len(height_rungs) == 1:
          compute_tile(tile_m)
        else:
          for rung_i, rung in enumerate(height_rungs):
            below = height_rungs[rung_i - 1] if rung_i else 0

            @pl.when(jnp.logical_and(live_rows > below, live_rows <= rung))
            def _(rung=rung):
              compute_tile(rung)

        # Step 4: Ship the output tile directly from VMEM or commit it to
        # `contrib_hbm`.
        if push_from_vmem:
          rows_out, subs_out = ship_tile(
              parity, t, tile_block_base, live_blocks
          )
          return (
              tile_count + 1,
              jnp.where(parity == 0, rows_out, carried[1]),
              jnp.where(parity == 1, rows_out, carried[2]),
              jnp.where(parity == 0, subs_out, carried[3]),
              jnp.where(parity == 1, subs_out, carried[4]),
              wslot,
          )

        commit_tile(parity, t, tile_block_base, live_blocks)

        return (
            tile_count + 1,
            jnp.where(parity == 0, live_blocks, carried[1]),
            jnp.where(parity == 1, live_blocks, carried[2]),
            wslot,
        )

      carry = lax.fori_loop(0, n_tiles, tile_body, carry)

      # Cross-expert lookahead: prime the next visited expert's first
      # activation tile before this expert's commit drain and push.
      tiles_done = carry[0]

      @pl.when(visit_i + 1 < n_visit)
      def _():
        nxt = visit_sm[visit_i + 1]
        prime_expert(
            nxt,
            lax.rem(tiles_done, jnp.int32(out_parities)),
            next_wslot,
            jnp.bool_(True),
            visit_i=visit_i + 1,
        )

      return carry[:-1] + (next_wslot,)

    def drain_commits(carried):
      """Drain both parities' pending commits, then zero the counts."""
      tile_count, pending0, pending1, wslot = carried

      @pl.when(pending0 > 0)
      def _():
        wait_commits(0, pending0)

      @pl.when(pending1 > 0)
      def _():
        wait_commits(1, pending1)

      return (tile_count, jnp.int32(0), jnp.int32(0), wslot)

    def drain_shipped(carried):
      """Wait out whatever the last tile on each parity put on the wire."""
      for p in range(out_parities):
        wait_shipped(p, carried[1 + p], carried[3 + p])

    @pl.when(n_visit > 0)
    def _():
      prime_expert(
          visit_sm[0],
          jnp.int32(0),
          jnp.int32(0),
          jnp.bool_(False),
          visit_i=jnp.int32(0),
      )

    def run_last_tile(e, d):
      """The expert-local tile holding run (e, d)'s last block."""
      return jnp.right_shift(
          commit_start_sm[e, d] + commit_len_sm[e, d] - 1, tile_block_shift
      )

    def push_run(e, d):
      """Push run (e, d): its remote rows and scale mirror, or the own-dest hop."""
      region_blocks = commit_len_sm[e, d]
      src_block = contrib_off_sm[e, d]
      dst_block = push_dst_sm[e, d]
      is_me = jnp.int32(d) == me

      @pl.when(memory_ref.and_nonempty(jnp.logical_not(is_me), region_blocks))
      def _():
        true_rows = true_rows_sm[e, d]

        def push_rows():
          """The rows themselves, at their true length."""
          pltpu.make_async_remote_copy(
              src_ref=contrib_hbm.at[
                  pl.ds(contrib_off_sm[e, d] * rowblk, true_rows)
              ],
              dst_ref=recv_hbm.at[pl.ds(push_dst_sm[e, d] * rowblk, true_rows)],
              send_sem=send_sem,
              recv_sem=recv_sem,
              device_id=(jnp.int32(d),),
          ).start()

        pl.when(true_rows > 0)(push_rows)
        if skip_scale_mirror:
          return
        src_sub, dst_sub, n_sub = _run_tiles(e, d, region_blocks)
        pltpu.make_async_remote_copy(
            src_ref=cscl_hbm.at[pl.ds(src_sub, n_sub)],
            dst_ref=rscl_hbm.at[pl.ds(dst_sub, n_sub)],
            send_sem=send_scl_sem,
            recv_sem=recv_scl_sem,
            device_id=(jnp.int32(d),),
        ).start()

      @pl.when(memory_ref.and_nonempty(is_me, region_blocks))
      def _():
        pltpu.make_async_copy(
            contrib_hbm.at[pl.ds(src_block * rowblk, region_blocks * rowblk)],
            recv_hbm.at[pl.ds(dst_block * rowblk, region_blocks * rowblk)],
            mehop_sem,
        ).start()
        if skip_scale_mirror:
          return
        src_sub, dst_sub, n_sub = _run_tiles(e, d, region_blocks)
        pltpu.make_async_copy(
            cscl_hbm.at[pl.ds(src_sub, n_sub)],
            rscl_hbm.at[pl.ds(dst_sub, n_sub)],
            mehop_scl_sem,
        ).start()

    def push_runs_ending_in(e, tile):
      """Progressive push: expert e's runs whose last tile is `tile`."""
      for d in range(ep):
        pl.when(run_last_tile(e, d) == tile)(functools.partial(push_run, e, d))

    def push_expert(e, first_tile=None):
      """Push expert e's remote regions and hop its own-dest region."""
      for d in range(ep):
        if first_tile is None:
          push_run(e, d)
        else:
          pl.when(run_last_tile(e, d) >= first_tile)(
              functools.partial(push_run, e, d)
          )

    # Body: Main visit loop over active local experts.
    def visit_step(visit_i, carry):
      """One step of the visit list: compute, then drain and push."""
      e = visit_sm[visit_i]
      carry = expert_tiles(e, visit_i, carry)
      if push_from_vmem:
        return carry
      carry = drain_commits(carry)
      if progressive_push:
        n_tiles = -(-expert_rows_sm[e] // tile_m)
        push_expert(e, first_tile=n_tiles - 2)
      else:
        push_expert(e)
      return carry

    carry0 = tuple(jnp.int32(0) for _ in range(6 if push_from_vmem else 4))
    carry_end = lax.fori_loop(0, n_visit, visit_step, carry0)

    # Epilogue: Drain remaining wave waits, gateway forward sends, return
    # pushes, and synchronize all shards before returning.
    def drain_transport():
      """Consume the pushes and the commits the per-tile waits left."""
      if push_from_vmem:
        drain_shipped(carry_end)
        memory_ref.rows_wait(
            recv_sem, recv_hbm, count(fused_host.COUNT_RECV_ROWS)
        )
        if not skip_scale_mirror:
          memory_ref.rows_wait(
              recv_scl_sem, rscl_hbm, count(fused_host.COUNT_RECV_MIRROR)
          )
        return
      memory_ref.rows_wait(
          send_sem, contrib_hbm, count(fused_host.COUNT_SEND_ROWS)
      )
      memory_ref.rows_wait(
          recv_sem, recv_hbm, count(fused_host.COUNT_RECV_ROWS)
      )
      if not skip_scale_mirror:
        memory_ref.rows_wait(
            send_scl_sem, cscl_hbm, count(fused_host.COUNT_SEND_MIRROR)
        )
        memory_ref.rows_wait(
            recv_scl_sem, rscl_hbm, count(fused_host.COUNT_RECV_MIRROR)
        )
      self_blocks = commit_len_sm[0, me]
      for g in range(1, g_local):
        self_blocks = self_blocks + commit_len_sm[g, me]
      memory_ref.rows_wait(mehop_sem, recv_hbm, self_blocks * rowblk)
      if not skip_scale_mirror:
        memory_ref.rows_wait(
            mehop_scl_sem, rscl_hbm, count(fused_host.COUNT_SELF_MIRROR)
        )

    if wait_for_exchange:
      wait_waves_through(jnp.int32(waves - 1))
    if dedup:
      fwd_row = (2 * nb + dedup_slots) * waves
      fwd_remote_total = stage_tab_sm[fwd_row]
      for w in range(1, waves):
        fwd_remote_total = fwd_remote_total + stage_tab_sm[fwd_row + w]

      @pl.when(fwd_remote_total > 0)
      def _():
        memory_ref.rows_wait_send(fwd_send_sem, lhs_hbm, fwd_remote_total)

    drain_transport()
    memory_ref.all_pairs_barrier(ep)

  return kernel
