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
"""Metadata scheduler kernel and SMEM data structures for Batched MLA.

Precomputes memory addresses, DMA offsets, task assignments, and synchronization
wait counters on TPU SMEM before executing the attention compute pipeline.
"""

import dataclasses
import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import utils

# --- SMEM Descriptor & Struct Abstractions ---


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
  """Maps physical 1D SMEM buffer to logical multi-dimensional indexing."""

  data: Any
  shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create_shape_dtype(cls, shape):
    return cls(
        data=jax.ShapeDtypeStruct((np.prod(shape),), jnp.int32), shape=shape
    )

  def _get_pos(self, indices):
    if not isinstance(indices, tuple):
      indices = (indices,)
    strides = pl.strides_from_shape(self.shape)
    assert len(strides) == len(indices)

    pos = 0
    for stride, idx in zip(strides, indices):
      pos += stride * idx
    return pos

  def __getitem__(self, indices):
    return self.data[self._get_pos(indices)]

  def __setitem__(self, indices, value):
    self.data[self._get_pos(indices)] = value


# `actual_steps` is a single int32 word. It is the only schedule leaf kept in
# SMEM (see `MlaSchedule.in_specs`/`out_specs`): the schedule kernel writes it
# straight into an SMEM output and the attention kernel takes that buffer as an
# SMEM operand, so reading it costs no HBM->SMEM copy. No explicit padding is
# needed -- Mosaic tiles 1D 32-bit memrefs to one lane-row of words (128 on
# v4+, 256 on v3) internally, which is why `MlaConfigs.max_steps_ub` still
# budgets a full tile for it.
ACTUAL_STEPS_WORDS = 1


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MlaSchedule:
  """Schedule pytree containing all precomputed metadata arrays."""

  s_idx: SmemWrapper  # [steps, batch]
  q_idx: SmemWrapper  # [steps, batch]
  k_idx: SmemWrapper  # [steps, batch]
  is_last_k: SmemWrapper  # [steps, batch]
  do_writeback: SmemWrapper  # [steps, batch]
  dma_q: SmemWrapper  # [steps, batch, 2]
  dma_kv_cache: SmemWrapper  # [steps, batch, bkv_p, 2]
  dma_kv_new_fetch: SmemWrapper  # [steps, batch, 3]
  dma_kv_new_wb: SmemWrapper  # [steps, batch, bkv_p_wb, 3]
  total_wait_kv_in: SmemWrapper  # [steps]
  total_wait_kv_out: SmemWrapper  # [steps]
  total_wait_q_in: SmemWrapper  # [steps]
  total_wait_o_out: SmemWrapper  # [steps]
  actual_steps: Any  # [ACTUAL_STEPS_WORDS] == [1], lives in SMEM.

  cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create_shape_dtype(cls, cfgs: configs.MlaConfigs, multiplier: int = 1):
    effective_max_steps = cfgs.max_steps_ub * multiplier

    idx_wrapper = SmemWrapper.create_shape_dtype(
        (effective_max_steps, cfgs.batch_size)
    )
    steps_wrapper = SmemWrapper.create_shape_dtype((effective_max_steps,))

    return cls(
        s_idx=idx_wrapper,
        q_idx=idx_wrapper,
        k_idx=idx_wrapper,
        is_last_k=idx_wrapper,
        do_writeback=idx_wrapper,
        dma_q=SmemWrapper.create_shape_dtype(
            (effective_max_steps, cfgs.batch_size, 2)
        ),
        dma_kv_cache=SmemWrapper.create_shape_dtype(
            (effective_max_steps, cfgs.batch_size, max(1, cfgs.bkv_p), 2)
        ),
        dma_kv_new_fetch=SmemWrapper.create_shape_dtype(
            (effective_max_steps, cfgs.batch_size, 3)
        ),
        dma_kv_new_wb=SmemWrapper.create_shape_dtype(
            (effective_max_steps, cfgs.batch_size, cfgs.bkv_p_wb, 3)
        ),
        total_wait_kv_in=steps_wrapper,
        total_wait_kv_out=steps_wrapper,
        total_wait_q_in=steps_wrapper,
        total_wait_o_out=steps_wrapper,
        actual_steps=jax.ShapeDtypeStruct(
            (ACTUAL_STEPS_WORDS,), jnp.int32
        ),
        cfgs=cfgs,
    )

  def step_leaves(self) -> list[Any]:
    """Leaves indexed by step and stored in HBM: all but `actual_steps`."""
    return jax.tree_util.tree_leaves(
        dataclasses.replace(self, actual_steps=None)
    )

  @property
  def total_hbm_words(self) -> int:
    return sum(int(leaf.shape[0]) for leaf in self.step_leaves())

  def get_dma_kv_cache(
      self,
      step: jax.Array,
      batch_idx: int,
      page_idx: int,
  ) -> tuple[jax.Array, jax.Array]:
    pos = self.dma_kv_cache._get_pos((step, batch_idx, page_idx, 0))
    src_off = self.dma_kv_cache.data[pos]
    sz = self.dma_kv_cache.data[pos + 1]
    return src_off, sz

  def get_dma_kv_new_fetch(
      self,
      step: jax.Array,
      batch_idx: int,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    pos = self.dma_kv_new_fetch._get_pos((step, batch_idx, 0))
    fetch_hbm = self.dma_kv_new_fetch.data[pos]
    fetch_vmem = self.dma_kv_new_fetch.data[pos + 1]
    fetch_sz = self.dma_kv_new_fetch.data[pos + 2]
    return fetch_hbm, fetch_vmem, fetch_sz

  def get_dma_kv_new_wb(
      self,
      step: jax.Array,
      batch_idx: int,
      page_idx: int,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    pos = self.dma_kv_new_wb._get_pos((step, batch_idx, page_idx, 0))
    wb_hbm = self.dma_kv_new_wb.data[pos]
    wb_vmem = self.dma_kv_new_wb.data[pos + 1]
    wb_sz = self.dma_kv_new_wb.data[pos + 2]
    return wb_hbm, wb_vmem, wb_sz

  def get_dma_q(
      self, step: jax.Array, batch_idx: int
  ) -> tuple[jax.Array, jax.Array]:
    pos = self.dma_q._get_pos((step, batch_idx, 0))
    src_hbm = self.dma_q.data[pos]
    sz = self.dma_q.data[pos + 1]
    return src_hbm, sz

  def scratch_shapes(self):
    return jax.tree.map(lambda x: pltpu.SMEM(x.shape, x.dtype), self)

  def memory_spaces(self):
    """HBM for the per-step leaves, SMEM for `actual_steps`."""
    spaces = jax.tree.map(lambda _: pltpu.HBM, self)
    return dataclasses.replace(spaces, actual_steps=pltpu.SMEM)

  def in_specs(self):
    return jax.tree.map(
        lambda space: pl.BlockSpec(memory_space=space), self.memory_spaces()
    )

  def out_specs(self):
    return self.in_specs()

  def out_shape(self):
    return jax.tree.map(
        lambda x, space: space(x.shape, x.dtype), self, self.memory_spaces()
    )


# --- Scheduler Computation Core ---


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LoopCarry:
  hbm_offset: jax.Array
  count: jax.Array


def _mask_out_steps(
    step: jax.Array,
    schedule_smem: MlaSchedule,
    b_idx: jax.Array,
):
  """Masks out unused schedule entries at (step, b_idx)."""
  schedule_smem.s_idx[step, b_idx] = -1
  schedule_smem.q_idx[step, b_idx] = 0
  schedule_smem.k_idx[step, b_idx] = 0
  schedule_smem.is_last_k[step, b_idx] = 0
  schedule_smem.do_writeback[step, b_idx] = 0
  schedule_smem.dma_q[step, b_idx, 0] = 0
  schedule_smem.dma_q[step, b_idx, 1] = 0

  for i in range(schedule_smem.cfgs.bkv_p):
    schedule_smem.dma_kv_cache[step, b_idx, i, 0] = 0
    schedule_smem.dma_kv_cache[step, b_idx, i, 1] = 0

  for j in range(3):
    schedule_smem.dma_kv_new_fetch[step, b_idx, j] = 0

  for i in range(schedule_smem.cfgs.bkv_p_wb):
    for j in range(3):
      schedule_smem.dma_kv_new_wb[step, b_idx, i, j] = 0

  schedule_smem.total_wait_kv_in[step] = 0
  schedule_smem.total_wait_kv_out[step] = 0
  schedule_smem.total_wait_q_in[step] = 0
  schedule_smem.total_wait_o_out[step] = 0


def _write_schedule_to_hbm(
    schedule_smem: MlaSchedule,
    schedule_hbm: MlaSchedule,
    hbm_offset: jax.Array,
    num_steps: jax.Array,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.MlaConfigs,
):
  """Writes num_steps from schedule_smem to schedule_hbm."""
  num_lanes = pltpu.get_tpu_info().num_lanes
  hbm_offset_aligned = pl.multiple_of(hbm_offset, num_lanes)  # pytype: disable=bad-argument-type
  flat_hbm = schedule_hbm.step_leaves()
  flat_smem = schedule_smem.step_leaves()

  copies = []
  for h, s in zip(flat_hbm, flat_smem):
    element_size = s.shape[0] // cfgs.max_steps_ub
    copy_size = pl.multiple_of(
        utils.align_to(num_steps * element_size, num_lanes), num_lanes
    )
    dst_offset = pl.multiple_of(hbm_offset_aligned * element_size, num_lanes)
    copies.append(
        pltpu.make_async_copy(
            s.at[pl.ds(0, copy_size)],
            h.at[pl.ds(dst_offset, copy_size)],
            dma_sem.at[0],
        )
    )
  for copy in copies:
    copy.start()
  for copy in copies:
    copy.wait()


def _compute_waits(
    schedule: MlaSchedule,
    start_step: int | jax.Array,
    end_step: jax.Array,
    *,
    cfgs: configs.MlaConfigs,
):
  """Computes total wait lane counts for DMA synchronization."""

  @jax.named_scope("compute_waits")
  def body(step, _):
    # KV IN: every descriptor carries its token count (a lane-tile multiple).
    kv_in_tokens = 0
    for b in range(cfgs.batch_size):
      for i in range(cfgs.bkv_p):
        _, sz = schedule.get_dma_kv_cache(step, b, i)
        kv_in_tokens += sz
      _, _, fetch_sz = schedule.get_dma_kv_new_fetch(step, b)
      kv_in_tokens += fetch_sz

    schedule.total_wait_kv_in[step] = kv_in_tokens

    # KV OUT
    kv_out_tokens = 0
    for b in range(cfgs.batch_size):
      do_writeback = schedule.do_writeback[step, b] == 1
      for i in range(cfgs.bkv_p_wb):
        _, _, wb_sz = schedule.get_dma_kv_new_wb(step, b, i)
        kv_out_tokens += jnp.where(do_writeback, wb_sz, 0)

    schedule.total_wait_kv_out[step] = kv_out_tokens

    # Q IN
    q_in_tokens = 0
    for b in range(cfgs.batch_size):
      _, q_sz = schedule.get_dma_q(step, b)
      q_in_tokens += q_sz
    schedule.total_wait_q_in[step] = q_in_tokens

    # O OUT
    o_out_tokens = 0
    for b in range(cfgs.batch_size):
      is_last_k = schedule.is_last_k[step, b] == 1
      _, q_sz = schedule.get_dma_q(step, b)
      o_out_tokens += jnp.where(is_last_k, q_sz, 0)
    schedule.total_wait_o_out[step] = o_out_tokens

  jax.lax.fori_loop(start_step, end_step, body, None)


def flush_to_hbm(
    count: jax.Array,
    schedule: MlaSchedule,
    schedule_hbm_ref: MlaSchedule,
    hbm_offset: jax.Array,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.MlaConfigs,
):
  last_step = utils.align_to(count, cfgs.batch_size)
  num_steps = last_step // cfgs.batch_size

  @pl.loop(count, last_step)
  @jax.named_scope("mask_out_steps")
  def body(idx):
    step, b_idx = divmod(idx, cfgs.batch_size)
    _mask_out_steps(step, schedule, b_idx)

  _compute_waits(schedule, 0, num_steps, cfgs=cfgs)
  _write_schedule_to_hbm(
      schedule,
      schedule_hbm_ref,
      hbm_offset,
      num_steps,
      dma_sem,
      cfgs=cfgs,
  )
  return hbm_offset + cfgs.max_steps_ub, 0


def compute_metadata(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    page_indices_ref: jax.Ref,
    distribution_ref: jax.Ref,
    schedule: MlaSchedule,
    schedule_hbm_ref: MlaSchedule,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.MlaConfigs,
) -> LoopCarry:
  """Populates schedule metadata using native TPU jax.lax.fori_loop."""

  @jax.named_scope("k_loop")
  def k_loop(
      k_idx,
      carry: LoopCarry,
      *,
      s_idx,
      q_idx,
      q_src,
      q_sz_task,
      total_kv_len,
      historical_kv_len,
      causal_kv_len,
      end_k_idx,
  ):
    count = carry.count
    step, target_lane = divmod(count, cfgs.batch_size)

    schedule.s_idx[step, target_lane] = s_idx
    schedule.q_idx[step, target_lane] = q_idx
    schedule.k_idx[step, target_lane] = k_idx

    is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
    schedule.is_last_k[step, target_lane] = is_last_k

    schedule.dma_q[step, target_lane, 0] = q_src
    schedule.dma_q[step, target_lane, 1] = q_sz_task

    # Sequence KV timeline (`historical_kv_len` cached + `q_len` new tokens)
    # intersected with KV block `k_idx` (`[kv_token_start, +bkv_sz)`), bounded
    # above by `causal_kv_len` so `q_idx` never fetches new KV tokens past its
    # own causal horizon:
    #
    #   0                         historical_kv_len               causal_kv_len
    #   v                                 v                                    v
    #   +---------------------------------+------------------------------------+
    #   |            cached KV            |      causally visible new KV       |
    #   +---------------------------------+------------------------------------+
    #             :                       :                                    :
    #             |<----------------------+------------ kv_left -------------->|
    #             |<- kv_left_frm_cache ->|<------- kv_left_frm_new ---------->|
    #             :                       :               :
    #             +-----------------------+---------------+
    #             |     bkv_sz_cache      |    new_sz     |  KV block k_idx
    #             +-----------------------+---------------+
    #             |<-------------- bkv_sz --------------->|
    #             ^                                       ^
    #       kv_token_start                          kv_token_start
    #     (= k_idx * bkv_sz)                           + bkv_sz
    kv_token_start = k_idx * cfgs.bkv_sz
    kv_page_start = k_idx * cfgs.bkv_p
    kv_left = causal_kv_len - kv_token_start
    kv_left_frm_cache = jnp.maximum(historical_kv_len - kv_token_start, 0)
    kv_left_frm_new = kv_left - kv_left_frm_cache
    bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
    new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)
    p_offset = s_idx * cfgs.serve.pages_per_seq + kv_page_start

    # Writeback happens once per KV block, on the first `q_idx` whose causal
    # horizon covers the last new token in `k_idx` (`kv_block_end - 1`).
    kv_block_end = jnp.minimum(total_kv_len, kv_token_start + cfgs.bkv_sz)
    q_wb = jnp.maximum(0, kv_block_end - 1 - historical_kv_len) // cfgs.bq_sz
    do_writeback = jnp.where((new_sz > 0) & (q_idx == q_wb), 1, 0)
    schedule.do_writeback[step, target_lane] = do_writeback

    kv_to_read_from_cache = kv_left_frm_cache
    for i in range(cfgs.bkv_p):
      dma_sz = jnp.clip(kv_to_read_from_cache, 0, cfgs.serve.page_size)
      kv_to_read_from_cache -= dma_sz
      src_hbm_idx = jnp.minimum(p_offset + i, cfgs.serve.num_page_indices - 1)
      hbm_p_idx = page_indices_ref[src_hbm_idx]

      schedule.dma_kv_cache[step, target_lane, i, 0] = hbm_p_idx
      # Token count rounded up to the lane tile: only the tiles holding cached
      # tokens are fetched, not the whole page. Both ends are static: the
      # source starts at offset 0 of the page, the destination at
      # i * page_size.
      schedule.dma_kv_cache[step, target_lane, i, 1] = utils.align_to(
          dma_sz, cfgs.kv_token_align
      )

    align = cfgs.kv_token_align
    new_vmem_start = bkv_sz_cache
    new_vmem_end = bkv_sz_cache + new_sz

    # Fetch: the new tokens are contiguous in HBM and in VMEM, so one DMA
    # moves all of them, widened to the lane tile on both ends, and lands right
    # after the cached tiles; the stitch later shifts the tokens down to
    # `bkv_sz_cache`.
    new_hbm_start = q_src + q_sz_task - kv_left_frm_new
    fetch_hbm_off, _, fetch_aligned_sz = utils.align_span(
        new_hbm_start, new_sz, align
    )
    schedule.dma_kv_new_fetch[step, target_lane, 0] = fetch_hbm_off
    schedule.dma_kv_new_fetch[step, target_lane, 1] = utils.align_to(
        new_vmem_start, align
    )
    schedule.dma_kv_new_fetch[step, target_lane, 2] = jnp.where(
        new_sz > 0, fetch_aligned_sz, 0
    )

    # Writeback: one DMA per page, since page_indices scatters the pages,
    # widened to the lane tiles that hold the tokens. The tiles' leading
    # lanes are cached tokens that VMEM holds unchanged, and the trailing
    # ones lie past the sequence end, so rewriting them is harmless.
    def fill_dma_kv_new_wb(i, wb_vmem_start, wb_num_tokens):
      """Populates new-KV writeback `i` for `[wb_vmem_start, +wb_num_tokens)`."""
      seq_tok_idx = kv_token_start + wb_vmem_start
      logical_page = jnp.minimum(
          seq_tok_idx >> cfgs.serve.page_size_log2,
          cfgs.serve.pages_per_seq - 1,
      )
      page_table_idx = jnp.minimum(
          s_idx * cfgs.serve.pages_per_seq + logical_page,
          cfgs.serve.num_page_indices - 1,
      )
      phys_page = page_indices_ref[page_table_idx]
      offset_in_page = seq_tok_idx & cfgs.serve.page_size_mask
      wb_page_off, lead, wb_aligned_sz = utils.align_span(
          offset_in_page, wb_num_tokens, align
      )
      wb_aligned_sz = jnp.where(wb_num_tokens > 0, wb_aligned_sz, 0)

      schedule.dma_kv_new_wb[step, target_lane, i, 0] = (
          phys_page << cfgs.serve.page_size_log2
      ) | wb_page_off
      schedule.dma_kv_new_wb[step, target_lane, i, 1] = wb_vmem_start - lead
      schedule.dma_kv_new_wb[step, target_lane, i, 2] = wb_aligned_sz

    if cfgs.one_new_token:
      assert cfgs.bkv_p_wb == 1
      fill_dma_kv_new_wb(0, new_vmem_start, new_sz)
    else:
      for i in range(cfgs.bkv_p_wb):
        page_vmem_start = i * cfgs.serve.page_size
        page_vmem_end = page_vmem_start + cfgs.serve.page_size

        wb_vmem_start = jnp.maximum(page_vmem_start, new_vmem_start)
        wb_vmem_end = jnp.minimum(page_vmem_end, new_vmem_end)
        wb_num_tokens = jnp.maximum(0, wb_vmem_end - wb_vmem_start)

        fill_dma_kv_new_wb(i, wb_vmem_start, wb_num_tokens)

    def flush(carry: LoopCarry):
      hbm_offset = carry.hbm_offset
      new_hbm_offset, _ = flush_to_hbm(
          carry.count,
          schedule,
          schedule_hbm_ref,
          hbm_offset,
          dma_sem,
          cfgs=cfgs,
      )
      return LoopCarry(new_hbm_offset, jnp.int32(0))

    hbm_offset = carry.hbm_offset
    new_count = count + 1

    return jax.lax.cond(
        new_count < cfgs.max_steps_ub * cfgs.batch_size,
        lambda carry: carry,
        flush,
        LoopCarry(hbm_offset, new_count),
    )

  @jax.named_scope("q_loop")
  def q_loop(
      q_idx, carry, *, s_idx, q_start, q_end, total_kv_len, q_len, num_k
  ):
    q_src = q_start + q_idx * cfgs.bq_sz
    q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

    start_k_idx = 0
    historical_kv_len = total_kv_len - q_len
    if (sliding_window := cfgs.model.sliding_window) is not None:
      sw_start_idx = historical_kv_len + q_idx * cfgs.bq_sz - sliding_window + 1
      start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

    causal_kv_len = historical_kv_len + q_idx * cfgs.bq_sz + q_sz_task
    end_k_idx_causal = (causal_kv_len - 1) // cfgs.bkv_sz + 1
    end_k_idx = jnp.minimum(num_k, end_k_idx_causal)

    k_loop_fn = functools.partial(
        k_loop,
        s_idx=s_idx,
        q_idx=q_idx,
        q_src=q_src,
        q_sz_task=q_sz_task,
        total_kv_len=total_kv_len,
        historical_kv_len=historical_kv_len,
        causal_kv_len=causal_kv_len,
        end_k_idx=end_k_idx,
    )
    return jax.lax.fori_loop(start_k_idx, end_k_idx, k_loop_fn, carry)

  @jax.named_scope("seq_loop")
  def seq_loop(s_idx, carry):
    q_start = cu_q_lens_ref[s_idx]
    q_end = cu_q_lens_ref[s_idx + 1]
    total_kv_len = kv_lens_ref[s_idx]
    q_len = q_end - q_start

    num_q = pl.cdiv(q_len, cfgs.bq_sz)
    num_k = pl.cdiv(total_kv_len, cfgs.bkv_sz)

    q_loop_fn = functools.partial(
        q_loop,
        s_idx=s_idx,
        q_start=q_start,
        q_end=q_end,
        total_kv_len=total_kv_len,
        q_len=q_len,
        num_k=num_k,
    )
    return jax.lax.fori_loop(0, num_q, q_loop_fn, carry)

  start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)  # pytype: disable=bad-argument-type
  init_carry = LoopCarry(jnp.int32(0), jnp.int32(0))
  return jax.lax.fori_loop(start_seq_idx, end_seq_idx, seq_loop, init_carry)


def rpa_metadata_schedule_kernel(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    page_indices_ref: jax.Ref,
    distribution_ref: jax.Ref,
    schedule_hbm_ref: MlaSchedule,
    schedule_ref: MlaSchedule,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.MlaConfigs,
):
  """Generates HBM-to-VMEM DMA schedule for Batched MLA."""
  loop_carry = compute_metadata(
      cu_q_lens_ref,
      kv_lens_ref,
      page_indices_ref,
      distribution_ref,
      schedule_ref,
      schedule_hbm_ref,
      dma_sem,
      cfgs=cfgs,
  )
  count = loop_carry.count
  hbm_offset = loop_carry.hbm_offset
  steps = pl.cdiv(count, cfgs.batch_size) + hbm_offset
  # `actual_steps` is an SMEM output, so a scalar store is all it takes.
  schedule_hbm_ref.actual_steps[0] = steps  # pytype: disable=unsupported-operation

  # An empty buffer has nothing to flush, and flushing it anyway would advance
  # `hbm_offset` by another `max_steps_ub` -- past the end of the HBM schedule
  # whenever the step count landed exactly on a buffer boundary.
  @pl.when(count > 0)
  def _():
    flush_to_hbm(
        count,
        schedule_ref,
        schedule_hbm_ref,
        hbm_offset,
        dma_sem,
        cfgs=cfgs,
    )


def generate_mla_metadata(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    distribution: jax.Array,
    cfgs: configs.MlaConfigs,
) -> MlaSchedule:
  """Entry point for generating Batched MLA schedule metadata."""
  schedule_shaped_dtype = MlaSchedule.create_shape_dtype(cfgs)
  schedule_hbm = MlaSchedule.create_shape_dtype(
      cfgs, multiplier=cfgs.max_schedule_size_multiplier
  )
  # `out_specs` only types the refs inside the kernel; where XLA places the
  # output buffers comes from the `out_shape` avals. Plain ShapeDtypeStructs
  # carry no memory space, so XLA was free to put the schedule in VMEM and then
  # evict it to HBM with copy-start/done pairs before the attention kernel ran.
  # Tagging each leaf pins the per-step leaves in HBM and `actual_steps` in
  # SMEM.
  out_shape = schedule_hbm.out_shape()

  return pl.pallas_call(
      functools.partial(rpa_metadata_schedule_kernel, cfgs=cfgs),
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=4,
          in_specs=[],
          out_specs=schedule_hbm.out_specs(),
          scratch_shapes=[
              schedule_shaped_dtype.scratch_shapes(),
              pltpu.SemaphoreType.DMA((1,)),
          ],
      ),
      name="mla_metadata_schedule",
  )(cu_q_lens, kv_lens, page_indices, distribution)
