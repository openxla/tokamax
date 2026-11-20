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
"""Metadata planner kernel: walks (seq, q_tile, k_tile) into RpaSchedule."""

from collections.abc import Sequence
import dataclasses
import functools
import typing
from typing import Any, ClassVar

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import paging
from tokamax._src.ops.batched_rpa.kernel import schedule
from tokamax._src.ops.batched_rpa.kernel import utils


@dataclasses.dataclass(frozen=True)
class BaseScheduler:
  """Customization hooks for sequence-length splitting and CP page striping.

  Subclasses customize planner data without duplicating the traversal loop:
    - `scope`: Which KV positions are attended (`FULL`, `CACHE_ONLY`,
      `NEW_TOKENS_ONLY`).
    - `get_kv_lens`: Host-side split of `kv_lens` into `(kv_cache_lens,
      kv_new_lens, q_offsets)`.
    - `page_ownership`: Kernel-side `paging.PageOwnership(group_size, rank)`
      describing which logical pages reside on this device.
  """

  scope: ClassVar[configs.AttentionScope] = configs.AttentionScope.FULL

  def get_cp_group_size(self) -> int | None:
    return None

  def page_ownership(
      self, extra_refs: Sequence[jax.Ref] = ()
  ) -> paging.PageOwnership:
    del extra_refs
    return paging.PageOwnership()

  def get_kv_lens(
      self,
      kv_lens: jax.Array,
      q_lens: jax.Array,
      page_size: int = 1,
      extra_scalars: Sequence[jax.Array] = (),
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns `(kv_cache_lens, kv_new_lens, q_offsets)` on host."""
    del page_size, extra_scalars
    kv_cache_lens = kv_lens - q_lens
    return kv_cache_lens, q_lens, kv_cache_lens

  def get_q_start(
      self,
      q_idx: jax.Array,
      q_offset: jax.Array,
      cfgs: configs.RpaConfigs,
  ) -> jax.Array:
    """Returns within-sequence logical token position for query block `q_idx`."""
    return q_idx * cfgs.bq_sz + q_offset

  def get_kv_start(
      self,
      k_idx: jax.Array,
      cfgs: configs.RpaConfigs,
      extra_refs: Sequence[jax.Ref] = (),
  ) -> jax.Array:
    """Returns within-sequence logical token position for KV block `k_idx`."""
    del extra_refs
    return k_idx * cfgs.bkv_sz


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
  """Runs the scalar planner kernel and returns the `RpaSchedule` tape in HBM."""
  if pad := cfgs.page_window_slot_len - cfgs.page_window_entries:
    page_indices = jnp.pad(page_indices, (0, pad))

  schedule_smem = schedule.RpaSchedule.create_shape_dtype(cfgs)
  schedule_hbm = schedule.RpaSchedule.create_shape_dtype(
      cfgs, multiplier=cfgs.max_schedule_size_multiplier
  )
  page_win_smem, page_win_sem = paging.PageWindowBuffer.scratch_shapes(cfgs)

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
              schedule_smem.scratch_shapes(),
              pltpu.SemaphoreType.DMA((1,)),
              page_win_smem,
              page_win_sem,
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


def rpa_metadata_schedule_kernel(
    cu_q_lens_ref: jax.Ref,
    q_offsets_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    extra_scalars_ref: Sequence[jax.Ref],
    page_indices_hbm_ref: jax.Ref,
    schedule_hbm_ref: schedule.RpaSchedule,
    schedule_ref: schedule.RpaSchedule,
    dma_sem: jax.Ref,
    page_indices_ref: jax.Ref,
    page_indices_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
    scheduler: BaseScheduler,
) -> None:
  """Builds `RpaSchedule` chunks in SMEM and flushes them to HBM."""
  page_buf = paging.PageWindowBuffer(
      hbm_ref=page_indices_hbm_ref,
      smem_ref=page_indices_ref,
      sem_ref=page_indices_sem,
      cfgs=cfgs,
      ownership=scheduler.page_ownership(extra_scalars_ref),
  )
  carry = _build_schedule(
      cu_q_lens_ref=cu_q_lens_ref,
      q_offsets_ref=q_offsets_ref,
      kv_cache_lens_ref=kv_cache_lens_ref,
      kv_new_lens_ref=kv_new_lens_ref,
      distribution_ref=distribution_ref,
      page_buf=page_buf,
      schedule_ref=schedule_ref,
      schedule_hbm_ref=schedule_hbm_ref,
      dma_sem=dma_sem,
      scheduler=scheduler,
      extra_refs=extra_scalars_ref,
      cfgs=cfgs,
  )
  schedule_ref.actual_steps[0] = (
      pl.cdiv(carry.count, cfgs.batch_size) + carry.hbm_offset
  )
  schedule_ref.flush_to_hbm(
      carry.count, schedule_hbm_ref, carry.hbm_offset, dma_sem
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LoopCarry:
  """Planner loop state tracking HBM step offset and current SMEM slot count."""

  hbm_offset: jax.Array | int
  count: jax.Array | int


def _build_schedule(
    *,
    cu_q_lens_ref: jax.Ref,
    q_offsets_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    page_buf: paging.PageWindowBuffer,
    schedule_ref: schedule.RpaSchedule,
    schedule_hbm_ref: schedule.RpaSchedule,
    dma_sem: jax.Ref,
    scheduler: BaseScheduler,
    extra_refs: Sequence[jax.Ref],
    cfgs: configs.RpaConfigs,
) -> LoopCarry:
  """Walks active `(seq, q_tile, k_tile)` tuples and writes each tape slot."""
  start_seq_idx, end_seq_idx = cfgs.seq_range(distribution_ref)
  page_buf.start_window(
      page_buf.window_idx(start_seq_idx), start_seq_idx < end_seq_idx
  )

  @jax.named_scope("seq_loop")
  def for_each_seq(s_idx, seq_carry: LoopCarry) -> LoopCarry:
    q_hbm_start = cu_q_lens_ref[s_idx]
    q_hbm_end = cu_q_lens_ref[s_idx + 1]
    q_len = q_hbm_end - q_hbm_start
    q_offset = q_offsets_ref[s_idx]
    kv_cache_len = kv_cache_lens_ref[s_idx]
    kv_new_len = kv_new_lens_ref[s_idx]
    kv_len = kv_cache_len + kv_new_len
    num_q = pl.cdiv(q_len, cfgs.bq_sz)
    num_k = pl.cdiv(kv_len, cfgs.bkv_sz)
    seq_page_offset = page_buf.enter_seq(s_idx, start_seq_idx, end_seq_idx)

    @jax.named_scope("q_loop")
    def for_each_q_tile(q_idx, q_carry: LoopCarry) -> LoopCarry:
      q_hbm_src = q_hbm_start + q_idx * cfgs.bq_sz
      q_tokens = jnp.clip(q_hbm_end - q_hbm_src, 0, cfgs.bq_sz)
      start_k_idx, end_k_idx = _k_tile_range(
          q_idx, q_tokens, q_offset, kv_cache_len, num_k, cfgs=cfgs
      )

      @jax.named_scope("k_loop")
      def for_each_k_tile(k_idx, k_carry: LoopCarry) -> LoopCarry:
        step, lane = divmod(k_carry.count, cfgs.batch_size)
        q_seq_pos = scheduler.get_q_start(q_idx, q_offset, cfgs)
        k_seq_pos = scheduler.get_kv_start(k_idx, cfgs, extra_refs)

        # 1. Split KV tile [k_seq_pos, +bkv_sz) into cached prefix + new tokens.
        kv_left = kv_len - k_seq_pos
        kv_left_cache = (
            kv_left
            if cfgs.serve.skip_kv_update
            else jnp.maximum(kv_cache_len - k_seq_pos, 0)
        )
        kv_left_new = kv_left - kv_left_cache
        num_cached = jnp.minimum(kv_left_cache, cfgs.bkv_sz)
        num_new = jnp.minimum(cfgs.bkv_sz - num_cached, kv_left_new)
        first_wb_q_idx = jnp.maximum(0, k_seq_pos - kv_cache_len) // cfgs.bq_sz
        do_writeback = (num_new > 0) & (q_idx == first_wb_q_idx)
        new_kv_row = q_hbm_end - kv_left_new

        # 2. Write logical tile metadata & Q fetch descriptor.
        meta = schedule_ref.tile_meta[step, lane]
        meta.s_idx = s_idx
        meta.is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
        meta.do_writeback = jnp.where(do_writeback, 1, 0)
        meta.causal_offset = k_seq_pos - q_seq_pos
        meta.k_min = q_offset - k_seq_pos
        meta.k_max = kv_cache_len - k_seq_pos

        q_dma = schedule_ref.dma_q[step, lane]
        q_dma.src_hbm = q_hbm_src
        q_dma.length = q_tokens

        # 3. Write KV cache fetches, new-KV fetch/writebacks, & stitch offsets.
        write_kv_fn = (
            _write_kv_seq_along_lane
            if cfgs.is_seq_along_lane
            else _write_kv_head_along_sublane
        )
        write_kv_fn(
            schedule_ref,
            step=step,
            lane=lane,
            k_idx=k_idx,
            kv_left_cache=kv_left_cache,
            num_cached=num_cached,
            num_new=num_new,
            new_kv_row=new_kv_row,
            seq_page_offset=seq_page_offset,
            page_buf=page_buf,
            cfgs=cfgs,
        )
        return _advance_carry(
            k_carry, schedule_ref, schedule_hbm_ref, dma_sem, cfgs=cfgs
        )

      return jax.lax.fori_loop(start_k_idx, end_k_idx, for_each_k_tile, q_carry)

    return jax.lax.fori_loop(0, num_q, for_each_q_tile, seq_carry)

  return jax.lax.fori_loop(
      start_seq_idx, end_seq_idx, for_each_seq, LoopCarry(0, 0)
  )


def _k_tile_range(
    q_idx: jax.Array,
    q_tokens: jax.Array,
    q_offset: jax.Array,
    kv_cache_len: jax.Array,
    num_k: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
) -> tuple[int | jax.Array, jax.Array]:
  """Returns the active KV block range `[start_k_idx, end_k_idx)` for `q_idx`."""
  start_k_idx: int | jax.Array = 0
  if (sliding_window := cfgs.model.sliding_window) is not None:
    sw_start_idx = q_offset + q_idx * cfgs.bq_sz - sliding_window + 1
    start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

  end_k_idx_causal = (
      q_offset + q_idx * cfgs.bq_sz + q_tokens - 1
  ) // cfgs.bkv_sz + 1
  end_k_idx = jnp.minimum(num_k, end_k_idx_causal)

  if cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
    start_k_idx = jnp.maximum(start_k_idx, kv_cache_len // cfgs.bkv_sz)

  return start_k_idx, end_k_idx


def _advance_carry(
    carry: LoopCarry,
    schedule_ref: schedule.RpaSchedule,
    schedule_hbm_ref: schedule.RpaSchedule,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
) -> LoopCarry:
  """Increments the slot counter and flushes SMEM schedule to HBM when full."""
  new_count = carry.count + 1

  def _flush(c: LoopCarry) -> LoopCarry:
    new_hbm_offset, _ = schedule_ref.flush_to_hbm(
        c.count,
        schedule_hbm_ref,
        c.hbm_offset,
        dma_sem,
    )
    return LoopCarry(new_hbm_offset, 0)

  return jax.lax.cond(
      new_count < schedule.RpaSchedule.max_steps_ub(cfgs) * cfgs.batch_size,
      lambda c: c,
      _flush,
      LoopCarry(carry.hbm_offset, new_count),
  )


def _new_kv_page_spans(
    num_cached: jax.Array,
    num_new: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
) -> list[tuple[jax.Array, jax.Array]]:
  """Splits new KV tokens `[num_cached, +num_new)` into `(vmem_dst, count)` per page."""
  page_size = cfgs.serve.page_size
  spans = []
  if cfgs.bkv_p_new < cfgs.bkv_p:
    curr_vmem = num_cached
    curr_rem = num_new
    for _ in range(cfgs.bkv_p_new):
      slot_start = (curr_vmem // page_size) * page_size
      slot_end = slot_start + page_size
      dma_sz = jnp.minimum(curr_rem, slot_end - curr_vmem)
      dma_sz = jnp.where(curr_rem > 0, dma_sz, 0)
      spans.append((curr_vmem, dma_sz))
      curr_vmem += dma_sz
      curr_rem = jnp.maximum(0, curr_rem - dma_sz)
  else:
    for i in range(cfgs.bkv_p):
      slot_start = i * page_size
      slot_end = slot_start + page_size
      vmem_dst = jnp.maximum(slot_start, num_cached)
      end_in_slot = jnp.minimum(slot_end, num_cached + num_new)
      dma_sz = jnp.maximum(0, end_in_slot - vmem_dst)
      spans.append((vmem_dst, dma_sz))
  return spans


def _write_kv_seq_along_lane(
    schedule_ref: schedule.RpaSchedule,
    *,
    step: Any,
    lane: Any,
    k_idx: Any,
    kv_left_cache: jax.Array,
    num_cached: jax.Array,
    num_new: jax.Array,
    new_kv_row: jax.Array,
    seq_page_offset: Any,
    page_buf: paging.PageWindowBuffer,
    cfgs: configs.RpaConfigs,
) -> None:
  """Writes `SEQ_ALONG_LANE` cache fetches, new-KV fetch, stitch, & writebacks."""
  align = cfgs.kv_token_align
  k_seq_pos = k_idx * cfgs.bkv_sz
  kv_p_start = k_idx * cfgs.bkv_p
  staged_vmem = utils.align_to(num_cached, align)

  # 1. Cached page fetches (kept under NEW_TOKENS_ONLY on owned pages for RMW).
  for i in range(cfgs.bkv_p_cache):
    vmem_dst = i << cfgs.serve.page_size_log2
    dma_sz = jnp.clip(kv_left_cache - vmem_dst, 0, cfgs.serve.page_size)
    logical_p = kv_p_start + i
    if page_buf.ownership.is_striped:
      dma_sz = jnp.where(page_buf.owns(logical_p), dma_sz, 0)
    cache_dma = schedule_ref.dma_kv_cache[step, lane, i]
    cache_dma.src_hbm = page_buf.phys_page(seq_page_offset, logical_p)
    cache_dma.dst_vmem = vmem_dst
    cache_dma.length = utils.align_to(dma_sz, align)

  # 2. Coalesced 128-lane new-token fetch (carried on record 0).
  fetch_src, _, raw_fetch_len = utils.align_span(new_kv_row, num_new, align)
  slot0_fetch_len = jnp.where(num_new > 0, raw_fetch_len, 0)

  # 3. Stitch offsets (roll staged new tokens back to `num_cached`).
  meta = typing.cast(
      schedule.SeqAlongLaneTileMeta, schedule_ref.tile_meta[step, lane]
  )
  meta.stitch_dst = num_cached
  meta.stitch_src = staged_vmem + (new_kv_row % align)
  meta.stitch_len = num_new

  # 4. Per-page RMW writebacks (widened to 128-lane tiles).
  for i, (vmem_dst, dma_sz) in enumerate(
      _new_kv_page_spans(num_cached, num_new, cfgs=cfgs)
  ):
    dma_entry = schedule_ref.dma_kv_new[step, lane, i]
    # Record 0 carries the coalesced fetch; subsequent records carry 0.
    dma_entry.fetch_hbm = fetch_src if i == 0 else 0
    dma_entry.fetch_vmem = staged_vmem if i == 0 else 0
    fetch_len = slot0_fetch_len if i == 0 else 0

    tok_idx = k_seq_pos + vmem_dst
    logical_p = jnp.minimum(
        tok_idx >> cfgs.serve.page_size_log2, page_buf.max_logical_page
    )
    page_offset = tok_idx & cfgs.serve.page_size_mask
    dst, lead, wb_len = utils.align_span(page_offset, dma_sz, align)
    phys_page = page_buf.phys_page(seq_page_offset, logical_p)
    dma_entry.wb_hbm = (phys_page << cfgs.serve.page_size_log2) | dst
    dma_entry.wb_vmem = vmem_dst - lead
    wb_cond = (dma_sz > 0) & page_buf.owns(logical_p)
    dma_entry.set_lens(fetch_len, jnp.where(wb_cond, wb_len, 0))


def _write_kv_head_along_sublane(
    schedule_ref: schedule.RpaSchedule,
    *,
    step: Any,
    lane: Any,
    k_idx: Any,
    kv_left_cache: jax.Array,
    num_cached: jax.Array,
    num_new: jax.Array,
    new_kv_row: jax.Array,
    seq_page_offset: Any,
    page_buf: paging.PageWindowBuffer,
    cfgs: configs.RpaConfigs,
) -> None:
  """Writes `HEAD_ALONG_SUBLANE` cache fetches and token-exact writebacks."""
  k_seq_pos = k_idx * cfgs.bkv_sz
  kv_p_start = k_idx * cfgs.bkv_p

  # 1. Cached page fetches (skipped completely when scope is NEW_TOKENS_ONLY).
  for i in range(cfgs.bkv_p_cache):
    vmem_dst = i << cfgs.serve.page_size_log2
    logical_p = kv_p_start + i
    src_hbm = page_buf.phys_page(seq_page_offset, logical_p)
    if cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
      dma_sz = 0
    else:
      dma_sz = jnp.clip(kv_left_cache - vmem_dst, 0, cfgs.serve.page_size)
      if page_buf.ownership.is_striped:
        dma_sz = jnp.where(page_buf.owns(logical_p), dma_sz, 0)
    cache_dma = schedule_ref.dma_kv_cache[step, lane, i]
    cache_dma.src_hbm = src_hbm
    cache_dma.dst_vmem = vmem_dst
    cache_dma.length = dma_sz

  # 2. Per-page token-exact new-KV fetch & writebacks.
  for i, (vmem_dst, dma_sz) in enumerate(
      _new_kv_page_spans(num_cached, num_new, cfgs=cfgs)
  ):
    dma_entry = schedule_ref.dma_kv_new[step, lane, i]
    tok_idx = k_seq_pos + vmem_dst
    logical_p = jnp.minimum(
        tok_idx >> cfgs.serve.page_size_log2, page_buf.max_logical_page
    )
    page_offset = tok_idx & cfgs.serve.page_size_mask
    phys_page = page_buf.phys_page(seq_page_offset, logical_p)
    dst_hbm = (phys_page << cfgs.serve.page_size_log2) | page_offset
    if page_buf.ownership.is_striped:
      wb_len = jnp.where(page_buf.owns(logical_p), dma_sz, jnp.int32(0))
    else:
      wb_len = dma_sz

    dma_entry.fetch_hbm = new_kv_row
    dma_entry.fetch_vmem = vmem_dst
    dma_entry.wb_hbm = dst_hbm
    dma_entry.wb_vmem = vmem_dst
    dma_entry.set_lens(dma_sz, wb_len)
