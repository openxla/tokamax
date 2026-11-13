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
"""Context Parallel (CP) schedulers for RPA."""

from collections.abc import Sequence
import dataclasses
from typing import ClassVar

import jax
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import schedule
from tokamax._src.ops.batched_rpa.kernel import scheduler
from tokamax._src.ops.batched_rpa.kernel import utils


@dataclasses.dataclass(frozen=True)
class CPScheduler(scheduler.BaseScheduler):
  """Base class for page-striped Context Parallel (CP) schedulers.

  Expects `extra_scalars=(cp_rank,)` where `cp_rank` is an `int32[1]` array
  giving the rank of the current device in the CP group.

  Attributes:
    cp_group_size: Number of devices in the CP group.
  """

  cp_group_size: int

  def get_cp_group_size(self) -> int:
    return self.cp_group_size

  def _get_cp_rank(
      self, extra: Sequence[jax.Array] | Sequence[jax.Ref]
  ) -> jax.Array:
    if not extra:
      raise ValueError(
          f"{type(self).__name__} requires `extra_scalars=(cp_rank,)`."
      )
    return extra[0][0]


@dataclasses.dataclass(frozen=True)
class StripedCacheScheduler(CPScheduler):
  """This scheduler only attends to this rank's local KV cache."""

  scope: ClassVar[configs.AttentionScope] = configs.AttentionScope.CACHE_ONLY

  def get_kv_lens(
      self,
      kv_lens: jax.Array,
      q_lens: jax.Array,
      page_size: int = 1,
      extra_scalars: Sequence[jax.Array] = (),
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    cp_rank = self._get_cp_rank(extra_scalars)
    global_cache_lens = kv_lens - q_lens
    local_cache_lens = utils.cp_local_cache_len(
        global_cache_lens, self.cp_group_size, cp_rank, page_size
    )
    return local_cache_lens, jnp.zeros_like(q_lens), local_cache_lens


@dataclasses.dataclass(frozen=True)
class StripedNewTokensScheduler(CPScheduler):
  """This scheduler writes back only owned pages."""

  scope: ClassVar[configs.AttentionScope] = (
      configs.AttentionScope.NEW_TOKENS_ONLY
  )

  def get_rpa_schedule(
      self, cfgs: configs.RpaConfigs, multiplier: int = 1
  ) -> schedule.RpaSchedule:
    """Returns the RpaSchedule struct with CP-specific DMA struct."""
    dma_kv_new_struct_cls = (
        schedule.HeadAlongSublaneDmaNewCP
        if cfgs.serve.kv_layout == configs.KVLayout.HEAD_ALONG_SUBLANE
        else schedule.SeqAlongLaneDmaNew
    )
    return schedule.RpaSchedule.create_shape_dtype(
        cfgs,
        dma_kv_new_struct_cls=dma_kv_new_struct_cls,
        multiplier=multiplier,
    )

  @jax.named_scope("k_loop_cp")
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
    cp_rank = self._get_cp_rank(extra_refs)
    cp_group_size = self.cp_group_size
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
    kv_left_frm_cache = jnp.maximum(kv_cache_len - kv_len_start, 0)

    for i in range(cfgs.bkv_p_cache):
      dst_vmem = i << cfgs.serve.page_size_log2
      dma_sz = kv_left_frm_cache - dst_vmem
      dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)

      if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        p_idx = kv_p_start + i
        local_page_i = jnp.minimum(
            p_idx // cp_group_size,
            cfgs.serve.pages_per_seq - 1,
        )
        src_hbm_cp = page_indices_ref[p_offset + local_page_i]
        dma_sz = jnp.where(p_idx % cp_group_size == cp_rank, dma_sz, 0)
        schedule_ref.dma_kv_cache[step, target_lane, i, 0] = src_hbm_cp
        schedule_ref.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
        schedule_ref.dma_kv_cache[step, target_lane, i, 2] = utils.align_to(
            dma_sz, cfgs.kv_token_align
        )
      else:
        p_idx = jnp.minimum(kv_p_start + i, cfgs.serve.pages_per_seq - 1)
        src_hbm = page_indices_ref[p_offset + p_idx]
        schedule_ref.dma_kv_cache[step, target_lane, i, 0] = src_hbm
        schedule_ref.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
        schedule_ref.dma_kv_cache[step, target_lane, i, 2] = 0

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

        # Writeback: one DMA per page, since page_indices scatters them, and
        # only for the pages this rank owns.
        tok_idx = kv_len_start + dst_vmem
        p_idx = jnp.minimum(
            tok_idx >> cfgs.serve.page_size_log2,
            cfgs.serve.pages_per_seq * cp_group_size - 1,
        )
        p_off = tok_idx & cfgs.serve.page_size_mask
        dst, lead, wb_val = utils.align_span(p_off, dma_sz, align)
        dma_entry.wb_hbm[...] = (
            page_indices_ref[p_offset + p_idx // cp_group_size]
            << cfgs.serve.page_size_log2
        ) | dst
        dma_entry.wb_vmem[...] = dst_vmem - lead
        dma_entry.set_flags(
            fetch_val,
            jnp.where(
                (dma_sz > 0) & (p_idx % cp_group_size == cp_rank), wb_val, 0
            ),
        )
      else:
        tok_idx = kv_len_start + dst_vmem
        p_idx = jnp.minimum(
            tok_idx >> cfgs.serve.page_size_log2,
            cfgs.serve.pages_per_seq * cp_group_size - 1,
        )
        p_off = tok_idx & cfgs.serve.page_size_mask
        local_slot = p_idx // cp_group_size
        dst_hbm = (
            page_indices_ref[p_offset + local_slot] << cfgs.serve.page_size_log2
        ) | p_off
        wb_val = jnp.where(
            p_idx % cp_group_size == cp_rank, dma_sz, jnp.int32(0)
        )

        dma_entry.fetch_hbm[...] = src_hbm
        dma_entry.fetch_vmem[...] = dst_vmem
        dma_entry.wb_hbm[...] = dst_hbm
        dma_entry.set_flags(dma_sz, wb_val)

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
