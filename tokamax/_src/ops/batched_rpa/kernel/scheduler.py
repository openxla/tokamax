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
"""Base scheduler and metadata schedule generation kernel for RPA."""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, ClassVar

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import schedule
from tokamax._src.ops.batched_rpa.kernel import utils


@dataclasses.dataclass(frozen=True)
class BaseScheduler:
  """Base scheduler planning each step's DMA schedules.

  Subclasses can customize scheduling and masking behavior:
    1. Host side (wrapper):
       - `get_kv_lens`: Splits each sequence's KV length into cached vs. new
         tokens and q_offsets.
    2. Schedule kernel:
       - `get_rpa_schedule`: Allocates the schedule shape and DMA struct.
       - `k_loop`, `q_loop`, `seq_loop`: Builds the per-step DMA commands.
    3. RPA kernel:
       - `get_q_start`, `get_kv_start`: Maps block indices to global sequence
         coordinates.
       - `get_kv_interleaved_size`: KV token interleave stride.

  Static and Dynamic parameters for custom schedulers:
    - Static parameters can be passed to __init__ as Scheduler is passed as a
      static parameter to ragged_paged_attention.
    - Dynamic per-device parameters for custom schedulers are passed via
      extra_scalars, and prefetched to SMEM for both the metadata schedule
      kernel and the RPA kernel.
  """

  scope: ClassVar[configs.AttentionScope] = configs.AttentionScope.FULL

  def get_cp_group_size(self) -> int | None:
    return None

  # For masking inside RPA kernel
  def get_kv_lens(
      self,
      kv_lens: jax.Array,
      q_lens: jax.Array,
      page_size: int = 1,
      extra_scalars: Sequence[jax.Array] = (),
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns `(kv_cache_lens, kv_new_lens, q_offsets)` for the kernels."""
    del page_size, extra_scalars
    kv_cache_lens = kv_lens - q_lens
    return kv_cache_lens, q_lens, kv_cache_lens

  def get_rpa_schedule(
      self, cfgs: configs.RpaConfigs, multiplier: int = 1
  ) -> schedule.RpaSchedule:
    """Returns the RpaSchedule shape/dtype struct for this scheduler."""
    dma_kv_new_struct_cls = (
        schedule.HeadAlongSublaneDmaNew
        if cfgs.serve.kv_layout == configs.KVLayout.HEAD_ALONG_SUBLANE
        else schedule.SeqAlongLaneDmaNew
    )
    return schedule.RpaSchedule.create_shape_dtype(
        cfgs,
        dma_kv_new_struct_cls=dma_kv_new_struct_cls,
        multiplier=multiplier,
        kv_interleaved_size=self.get_kv_interleaved_size(cfgs),
    )

  def get_q_start(
      self,
      q_idx: jax.Array,
      q_offset: jax.Array,
      cfgs: configs.RpaConfigs,
  ) -> jax.Array:
    """Returns the global starting sequence index for query block `q_idx`."""
    return q_idx * cfgs.bq_sz + q_offset

  def get_kv_start(
      self,
      k_idx: jax.Array,
      cfgs: configs.RpaConfigs,
      extra_refs: Sequence[jax.Ref] = (),
  ) -> jax.Array:
    """Returns the global starting sequence index for KV block `k_idx`."""
    del extra_refs
    return k_idx * cfgs.bkv_sz

  def get_kv_interleaved_size(
      self, cfgs: configs.RpaConfigs | configs.ServingConfigs
  ) -> int:
    """Returns the page interleave factor across ranks for KV tokens."""
    del cfgs
    return 1

  def page_window_idx(self, s_idx, cfgs: configs.RpaConfigs):
    """Window holding sequence `s_idx`'s page_indices row."""
    return s_idx >> (cfgs.page_window_seqs.bit_length() - 1)

  def start_page_window(
      self,
      win_idx,
      pred,
      page_indices_hbm_ref: jax.Ref,
      page_indices_ref: jax.Ref,
      page_indices_sem: jax.Ref,
      cfgs: configs.RpaConfigs,
  ):
    """Starts the DMA of window `win_idx` into slot `win_idx & 1`

    when `pred`.
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    slot_len = cfgs.page_window_slot_len
    start = win_idx * cfgs.page_window_entries
    if cfgs.page_window_entries % num_lanes:
      start = start // num_lanes * num_lanes
    slot = win_idx & 1

    @pl.when(pred)
    def start_window_copy():
      pltpu.make_async_copy(
          page_indices_hbm_ref.at[
              pl.ds(pl.multiple_of(start, num_lanes), slot_len)
          ],
          page_indices_ref.at[
              pl.ds(pl.multiple_of(slot * slot_len, num_lanes), slot_len)
          ],
          page_indices_sem.at[slot],
      ).start()

  def wait_page_window(
      self,
      win_idx,
      page_indices_ref: jax.Ref,
      page_indices_sem: jax.Ref,
      cfgs: configs.RpaConfigs,
  ):
    """Waits for window `win_idx`; only the semaphore and size matter."""
    win = page_indices_ref.at[pl.ds(0, cfgs.page_window_slot_len)]
    pltpu.make_async_copy(win, win, page_indices_sem.at[win_idx & 1]).wait()

  def page_indices_offset(self, s_idx, cfgs: configs.RpaConfigs):
    """Offset in `page_indices_ref` where sequence `s_idx`'s row starts."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    win_idx = self.page_window_idx(s_idx, cfgs)
    offset = (win_idx & 1) * cfgs.page_window_slot_len
    if cfgs.page_window_entries % num_lanes:
      offset += (win_idx * cfgs.page_window_entries) & (num_lanes - 1)
    return (
        offset
        + (s_idx & (cfgs.page_window_seqs - 1)) * cfgs.serve.pages_per_seq
    )

  # For computing RPASchedule
  @jax.named_scope("k_loop")
  def k_loop(
      self,
      k_idx,
      carry: schedule.LoopCarry,
      *,
      s_idx,
      q_idx,
      q_end,
      q_src,
      q_sz_task,
      kv_cache_len,
      kv_new_len,
      end_k_idx,
      p_offset,
      page_indices_ref: jax.Ref,
      schedule_ref: schedule.RpaSchedule,
      schedule_hbm_ref: schedule.RpaSchedule,
      dma_sem: jax.Ref,
      cfgs: configs.RpaConfigs,
      extra_refs: Sequence[jax.Ref] = (),
  ) -> schedule.LoopCarry:
    del extra_refs
    count = carry.count
    step, target_lane = divmod(count, cfgs.batch_size)

    schedule_ref.s_idx[step, target_lane] = s_idx
    schedule_ref.q_idx[step, target_lane] = q_idx
    schedule_ref.k_idx[step, target_lane] = k_idx

    is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
    schedule_ref.is_last_k[step, target_lane] = is_last_k

    schedule_ref.dma_q[step, target_lane, 0] = q_src
    schedule_ref.dma_q[step, target_lane, 1] = q_sz_task

    kv_len_start = k_idx * cfgs.bkv_sz
    kv_p_start = k_idx * cfgs.bkv_p
    k_len = kv_cache_len + kv_new_len
    kv_left = k_len - kv_len_start
    if cfgs.serve.skip_kv_update:
      kv_left_frm_cache = kv_left
    else:
      kv_left_frm_cache = jnp.maximum(kv_cache_len - kv_len_start, 0)
    for i in range(cfgs.bkv_p_cache):
      dst_vmem = i << cfgs.serve.page_size_log2
      dma_sz = kv_left_frm_cache - dst_vmem
      dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)

      p_idx = jnp.minimum(kv_p_start + i, cfgs.serve.pages_per_seq - 1)
      src_hbm = page_indices_ref[p_offset + p_idx]

      schedule_ref.dma_kv_cache[step, target_lane, i, 0] = src_hbm
      schedule_ref.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
      schedule_ref.dma_kv_cache[step, target_lane, i, 2] = utils.align_to(
          dma_sz, cfgs.kv_token_align
      )

    kv_left_frm_new = kv_left - kv_left_frm_cache
    bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
    new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

    # Writeback logic: each new k block is written back by the first q block
    # that attends to it.
    q_wb = jnp.maximum(0, (kv_len_start - kv_cache_len)) // cfgs.bq_sz
    do_writeback = jnp.where((new_sz > 0) & (q_idx == q_wb), 1, 0)
    schedule_ref.do_writeback[step, target_lane] = do_writeback
    src_hbm = q_end - kv_left_frm_new

    def fill_dma_kv_new(i, dst_vmem, dma_sz):
      dma_entry = schedule_ref.dma_kv_new[step, target_lane, i]
      if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        align = cfgs.kv_token_align
        # Fetch: the new tokens are contiguous, so one DMA moves all of them
        # even though this loop runs once per page. Entry 0 carries that DMA
        # and the other entries stay zero.
        if i == 0:
          src, _, fetch_val = utils.align_span(src_hbm, new_sz, align)
          dma_entry.fetch_hbm[...] = src
          dma_entry.fetch_vmem[...] = utils.align_to(bkv_sz_cache, align)
          fetch_val = jnp.where(new_sz > 0, fetch_val, 0)
        else:
          fetch_val = 0

        # Writeback: one DMA per page, since page_indices scatters them.
        tok_idx = kv_len_start + dst_vmem
        p_idx = jnp.minimum(
            tok_idx >> cfgs.serve.page_size_log2,
            cfgs.serve.pages_per_seq - 1,
        )
        p_off = tok_idx & cfgs.serve.page_size_mask
        dst, lead, wb_val = utils.align_span(p_off, dma_sz, align)
        dma_entry.wb_hbm[...] = (
            page_indices_ref[p_offset + p_idx] << cfgs.serve.page_size_log2
        ) | dst
        dma_entry.wb_vmem[...] = dst_vmem - lead
        dma_entry.set_flags(fetch_val, jnp.where(dma_sz > 0, wb_val, 0))
      else:
        p_idx = jnp.minimum(
            (kv_len_start + dst_vmem) >> cfgs.serve.page_size_log2,
            cfgs.serve.pages_per_seq - 1,
        )
        p_off = (kv_len_start + dst_vmem) & cfgs.serve.page_size_mask
        dst_hbm = (
            page_indices_ref[p_offset + p_idx] << cfgs.serve.page_size_log2
        ) | p_off

        dma_entry.fetch_hbm[...] = src_hbm
        dma_entry.fetch_vmem[...] = dst_vmem
        dma_entry.wb_hbm[...] = dst_hbm
        dma_entry.set_flags(dma_sz, dma_sz)

    if cfgs.bkv_p_new < cfgs.bkv_p:
      # General decode path for any bkv_p_new
      curr_vmem = bkv_sz_cache
      curr_rem = new_sz

      for i in range(cfgs.bkv_p_new):
        slot_start = (curr_vmem // cfgs.serve.page_size) * cfgs.serve.page_size
        slot_end = slot_start + cfgs.serve.page_size

        dma_sz = jnp.minimum(curr_rem, slot_end - curr_vmem)
        dma_sz = jnp.where(curr_rem > 0, dma_sz, 0)

        fill_dma_kv_new(i, curr_vmem, dma_sz)

        curr_vmem += dma_sz
        curr_rem = jnp.maximum(0, curr_rem - dma_sz)
    else:
      iters = max(cfgs.bkv_p, cfgs.bkv_p_new)
      for i in range(iters):
        slot_start = i * cfgs.serve.page_size
        slot_end = slot_start + cfgs.serve.page_size

        dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
        end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
        dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)

        fill_dma_kv_new(i, dst_vmem, dma_sz)

    return schedule.advance_carry(
        carry,
        schedule_ref,
        schedule_hbm_ref,
        dma_sem,
        cfgs=cfgs,
    )

  @jax.named_scope("q_loop")
  def q_loop(
      self,
      q_idx,
      carry: schedule.LoopCarry,
      *,
      s_idx,
      q_start,
      q_end,
      q_offset,
      kv_cache_len,
      kv_new_len,
      num_k,
      p_offset,
      page_indices_ref: jax.Ref,
      schedule_ref: schedule.RpaSchedule,
      schedule_hbm_ref: schedule.RpaSchedule,
      dma_sem: jax.Ref,
      cfgs: configs.RpaConfigs,
      extra_refs: Sequence[jax.Ref] = (),
  ) -> schedule.LoopCarry:
    q_src = q_start + q_idx * cfgs.bq_sz
    q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

    start_k_idx = 0
    if (sliding_window := cfgs.model.sliding_window) is not None:
      sw_start_idx = q_offset + q_idx * cfgs.bq_sz - sliding_window + 1
      start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

    end_k_idx_causal = (
        q_offset + q_idx * cfgs.bq_sz + q_sz_task - 1
    ) // cfgs.bkv_sz + 1
    end_k_idx = jnp.minimum(num_k, end_k_idx_causal)

    if cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
      start_k_idx = jnp.maximum(start_k_idx, kv_cache_len // cfgs.bkv_sz)

    def k_loop_fn(k_idx, c: schedule.LoopCarry) -> schedule.LoopCarry:
      return self.k_loop(
          k_idx,
          c,
          s_idx=s_idx,
          q_idx=q_idx,
          q_end=q_end,
          q_src=q_src,
          q_sz_task=q_sz_task,
          kv_cache_len=kv_cache_len,
          kv_new_len=kv_new_len,
          end_k_idx=end_k_idx,
          p_offset=p_offset,
          page_indices_ref=page_indices_ref,
          schedule_ref=schedule_ref,
          schedule_hbm_ref=schedule_hbm_ref,
          dma_sem=dma_sem,
          cfgs=cfgs,
          extra_refs=extra_refs,
      )

    return jax.lax.fori_loop(start_k_idx, end_k_idx, k_loop_fn, carry)

  @jax.named_scope("seq_loop")
  def seq_loop(
      self,
      cu_q_lens_ref: jax.Ref,
      q_offsets_ref: jax.Ref,
      kv_cache_lens_ref: jax.Ref,
      kv_new_lens_ref: jax.Ref,
      distribution_ref: jax.Ref,
      page_indices_hbm_ref: jax.Ref,
      page_indices_ref: jax.Ref,
      page_indices_sem: jax.Ref,
      schedule_ref: schedule.RpaSchedule,
      schedule_hbm_ref: schedule.RpaSchedule,
      dma_sem: jax.Ref,
      *,
      cfgs: configs.RpaConfigs,
      extra_refs: Sequence[jax.Ref] = (),
  ) -> schedule.LoopCarry:
    start_seq_idx, end_seq_idx = cfgs.seq_range(distribution_ref)
    self.start_page_window(
        self.page_window_idx(start_seq_idx, cfgs),
        start_seq_idx < end_seq_idx,
        page_indices_hbm_ref,
        page_indices_ref,
        page_indices_sem,
        cfgs,
    )

    def seq_loop_fn(s_idx, carry: schedule.LoopCarry) -> schedule.LoopCarry:
      q_start = cu_q_lens_ref[s_idx]
      q_end = cu_q_lens_ref[s_idx + 1]
      q_len = q_end - q_start
      q_offset = q_offsets_ref[s_idx]
      kv_cache_len = kv_cache_lens_ref[s_idx]
      kv_new_len = kv_new_lens_ref[s_idx]
      k_len = kv_cache_len + kv_new_len
      num_q = pl.cdiv(q_len, cfgs.bq_sz)
      num_k = pl.cdiv(k_len, cfgs.bkv_sz)
      win_idx = self.page_window_idx(s_idx, cfgs)
      win_seqs = cfgs.page_window_seqs

      def enter_window():
        self.start_page_window(
            win_idx + 1,
            (win_idx + 1) * win_seqs < end_seq_idx,
            page_indices_hbm_ref,
            page_indices_ref,
            page_indices_sem,
            cfgs,
        )
        self.wait_page_window(win_idx, page_indices_ref, page_indices_sem, cfgs)

      if win_seqs == 1:
        enter_window()
      else:
        first_in_window = (s_idx & (win_seqs - 1)) == 0
        pl.when((s_idx == start_seq_idx) | first_in_window)(enter_window)
      p_offset = self.page_indices_offset(s_idx, cfgs)

      def q_loop_fn(q_idx, c: schedule.LoopCarry) -> schedule.LoopCarry:
        return self.q_loop(
            q_idx,
            c,
            s_idx=s_idx,
            q_start=q_start,
            q_end=q_end,
            q_offset=q_offset,
            kv_cache_len=kv_cache_len,
            kv_new_len=kv_new_len,
            num_k=num_k,
            p_offset=p_offset,
            page_indices_ref=page_indices_ref,
            schedule_ref=schedule_ref,
            schedule_hbm_ref=schedule_hbm_ref,
            dma_sem=dma_sem,
            cfgs=cfgs,
            extra_refs=extra_refs,
        )

      return jax.lax.fori_loop(0, num_q, q_loop_fn, carry)

    init_carry = schedule.LoopCarry(0, 0)
    return jax.lax.fori_loop(
        start_seq_idx, end_seq_idx, seq_loop_fn, init_carry
    )


def rpa_metadata_schedule_kernel(
    # Scalar prefetch.
    cu_q_lens_ref: jax.Ref,
    q_offsets_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    extra_scalars_ref: Sequence[jax.Ref],
    # Inputs.
    page_indices_hbm_ref: jax.Ref,
    # Outputs.
    schedule_hbm_ref: schedule.RpaSchedule,
    # Scratch.
    schedule_ref: schedule.RpaSchedule,
    dma_sem: jax.Ref,
    page_indices_ref: jax.Ref,
    page_indices_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
    scheduler: BaseScheduler,
):
  """Generates RPA metadata schedule and stores it in SMEM and HBM buffers."""
  loop_carry = scheduler.seq_loop(
      cu_q_lens_ref=cu_q_lens_ref,
      q_offsets_ref=q_offsets_ref,
      kv_cache_lens_ref=kv_cache_lens_ref,
      kv_new_lens_ref=kv_new_lens_ref,
      distribution_ref=distribution_ref,
      page_indices_hbm_ref=page_indices_hbm_ref,
      page_indices_ref=page_indices_ref,
      page_indices_sem=page_indices_sem,
      schedule_ref=schedule_ref,
      schedule_hbm_ref=schedule_hbm_ref,
      dma_sem=dma_sem,
      cfgs=cfgs,
      extra_refs=extra_scalars_ref,
  )
  count = loop_carry.count
  hbm_offset = loop_carry.hbm_offset
  steps = pl.cdiv(count, cfgs.batch_size) + hbm_offset
  schedule_ref.actual_steps[0] = steps

  # Mask out unvisited steps and flush final chunk to HBM.
  schedule.flush_to_hbm(
      count,
      schedule_ref,
      schedule_hbm_ref,
      hbm_offset,
      dma_sem,
      cfgs=cfgs,
  )


def generate_rpa_metadata(
    cu_q_lens: jax.Array,
    q_offsets: jax.Array,
    kv_cache_lens: jax.Array,
    kv_new_lens: jax.Array,
    distribution: jax.Array,
    page_indices: jax.Array,
    cfgs: configs.RpaConfigs,
    *,
    scheduler: BaseScheduler,
    extra_scalars: Sequence[jax.Array] = (),
    interpret: bool = False,
) -> schedule.RpaSchedule:
  """Generates RPA metadata schedule using the specified scheduler."""
  if pad := cfgs.page_window_slot_len - cfgs.page_window_entries:
    page_indices = jnp.pad(page_indices, (0, pad))

  schedule_shaped_dtype = scheduler.get_rpa_schedule(cfgs)
  schedule_hbm = scheduler.get_rpa_schedule(
      cfgs, multiplier=cfgs.max_schedule_size_multiplier
  )

  return pl.pallas_call(
      functools.partial(
          rpa_metadata_schedule_kernel,
          cfgs=cfgs,
          scheduler=scheduler,
      ),
      out_shape=schedule_hbm,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=6,
          in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
          out_specs=schedule_hbm.out_specs(),
          scratch_shapes=[
              schedule_shaped_dtype.scratch_shapes(),
              pltpu.SemaphoreType.DMA((1,)),
              pltpu.SMEM((2 * cfgs.page_window_slot_len,), jnp.int32),
              pltpu.SemaphoreType.DMA((2,)),
          ],
      ),
      interpret=interpret,
      name="rpa_metadata_schedule",
  )(
      cu_q_lens,
      q_offsets,
      kv_cache_lens,
      kv_new_lens,
      distribution,
      extra_scalars,
      page_indices,
  )
