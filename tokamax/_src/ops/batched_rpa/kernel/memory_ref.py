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
"""Async HBM <-> VMEM multi-buffering references and per-slot DMA helpers."""

import dataclasses
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import configs


def _step_guard(
    step: int | jax.Array, total_steps: int | jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Returns `(is_no_op, block_idx)` for a possibly out-of-range pipeline step.

  Out-of-range steps occur in the pipeline prologue/epilogue. They are clamped
  to block 0 (so indexing stays in bounds) and flagged so that all DMA sizes
  can be forced to zero.

  Args:
    step: The pipeline step, which may fall outside `[0, total_steps)`.
    total_steps: The number of in-range steps.

  Returns:
    A tuple of `is_no_op`, true when `step` is out of range, and `block_idx`,
    the step clamped into `[0, total_steps)`.
  """
  is_no_op = jnp.logical_or(step < 0, step >= total_steps)
  block_idx = jnp.where(is_no_op, 0, jnp.maximum(step, 0))
  return is_no_op, block_idx


def _wait_lanes(
    ref: jax.Ref,
    wait_lanes: int | jax.Array,
    sem: jax.Ref,
    *,
    layout: configs.LayoutConfig,
) -> None:
  """Waits on `wait_lanes` `layout.dma_wait_word_dtype[num_lanes]` rows."""
  ref_words = ref.bitcast(layout.dma_wait_word_dtype)
  flat_ref = ref_words.reshape((-1, layout.num_lanes))
  pltpu.make_async_copy(
      flat_ref.at[pl.ds(0, wait_lanes), :],
      flat_ref.at[pl.ds(0, wait_lanes), :],
      sem,
  ).wait()


class _BaseBufferedRef(pltpu.BufferedRef):

  def __post_init__(self):
    # Pallas forbids buffer_count > 2 on output/inout refs; bypass check here.
    pass


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BypassRef(_BaseBufferedRef):
  """`pltpu.BufferedRef` bound to explicit `(step, slot, total_steps)` DMAs."""

  src_ref: Any = dataclasses.field(metadata=dict(static=True))
  dst_ref: Any = dataclasses.field(metadata=dict(static=True))

  def bind(self, src_ref=None, dst_ref=None):
    return dataclasses.replace(self, src_ref=src_ref, dst_ref=dst_ref)

  @classmethod
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool = False,
      **kwargs,
  ):
    standard_ref = _BaseBufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
    )
    return cls(
        src_ref=None,
        dst_ref=None,
        **kwargs,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )

  def open(self) -> None:
    """Optional prologue hook for collective/ring buffer subclasses."""

  def close(self) -> None:
    """Optional epilogue hook for collective/ring buffer subclasses."""

  def copy_in(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ) -> None:
    del step, slot, total_steps

  def wait_in(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ) -> None:
    del step, slot, total_steps

  def copy_out(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ) -> None:
    del step, slot, total_steps

  def wait_out(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ) -> None:
    del step, slot, total_steps


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class KVBufferedRefSeqAlongLane(BypassRef):
  """Fetches and writes back KV cache using `SEQ_ALONG_LANE` layout."""

  cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool = False,
      cfgs: configs.RpaConfigs | None = None,
      **kwargs,
  ):
    assert cfgs is not None
    assert buffer_type == pltpu.BufferType.INPUT_OUTPUT
    return super().create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        use_lookahead=use_lookahead,
        cfgs=cfgs,
        **kwargs,
    )

  def copy_in(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    kv_cache_hbm, new_kv_hbm, schedule_ref = self.src_ref
    sem = self.sem_recvs.at[slot]
    vmem_dst_lane = self.window_ref.at[slot]
    num_lanes = self.cfgs.layout.num_lanes
    for b in range(self.cfgs.batch_size):
      for i in range(self.cfgs.bkv_p_cache):
        cache_dma = schedule_ref.dma_kv_cache[block_idx, b, i]
        hbm_p_idx = cache_dma.src_hbm
        dst_off = pl.multiple_of(cache_dma.dst_vmem, num_lanes)
        sz = pl.multiple_of(jnp.where(is_no_op, 0, cache_dma.length), num_lanes)
        pltpu.make_async_copy(
            kv_cache_hbm.at[hbm_p_idx, :, :, :, pl.ds(0, sz)],
            vmem_dst_lane.at[b, :, :, :, pl.ds(dst_off, sz)],
            sem,
        ).start()

      # Entry 0 carries the single coalesced fetch for contiguous new tokens.
      new_dma = schedule_ref.dma_kv_new[block_idx, b, 0]
      src_new_off = pl.multiple_of(new_dma.fetch_hbm, num_lanes)
      dst_vmem_off = pl.multiple_of(new_dma.fetch_vmem, num_lanes)
      sz = pl.multiple_of(jnp.where(is_no_op, 0, new_dma.fetch_len), num_lanes)
      pltpu.make_async_copy(
          new_kv_hbm.at[:, :, :, pl.ds(src_new_off, sz)],
          vmem_dst_lane.at[b, :, :, :, pl.ds(dst_vmem_off, sz)],
          sem,
      ).start()

  def copy_out(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    kv_out_ref, _, schedule_ref = self.dst_ref
    sem = self.sem_sends.at[slot]
    vmem_src_lane = self.window_ref.at[slot]
    num_lanes = self.cfgs.layout.num_lanes
    for b in range(self.cfgs.batch_size):
      do_writeback = schedule_ref.tile_meta[block_idx, b].has_writeback
      for i in range(self.cfgs.bkv_p_new):
        new_dma = schedule_ref.dma_kv_new[block_idx, b, i]
        encoded_dst_hbm_off = new_dma.wb_hbm
        hbm_p_idx = encoded_dst_hbm_off >> self.cfgs.serve.page_size_log2
        dst_off = pl.multiple_of(
            encoded_dst_hbm_off & self.cfgs.serve.page_size_mask, num_lanes
        )
        src_vmem_off = pl.multiple_of(new_dma.wb_vmem, num_lanes)
        skip = jnp.logical_or(is_no_op, jnp.logical_not(do_writeback))
        sz = pl.multiple_of(jnp.where(skip, 0, new_dma.wb_len), num_lanes)
        pltpu.make_async_copy(
            vmem_src_lane.at[b, :, :, :, pl.ds(src_vmem_off, sz)],
            kv_out_ref.at[hbm_p_idx, :, :, :, pl.ds(dst_off, sz)],
            sem,
        ).start()

  def wait_in(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    _, _, schedule_ref = self.src_ref
    wait_lanes = jnp.where(is_no_op, 0, schedule_ref.waits[block_idx].kv_in)
    _wait_lanes(
        self.window_ref.at[slot],
        wait_lanes,
        self.sem_recvs.at[slot],
        layout=self.cfgs.layout,
    )

  def wait_out(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    _, _, schedule_ref = self.dst_ref
    wait_lanes = jnp.where(is_no_op, 0, schedule_ref.waits[block_idx].kv_out)
    _wait_lanes(
        self.window_ref.at[slot],
        wait_lanes,
        self.sem_sends.at[slot],
        layout=self.cfgs.layout,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class KVBufferedRefHeadAlongSublane(BypassRef):
  """Fetches and writes back KV cache using `HEAD_ALONG_SUBLANE` layout."""

  cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool = False,
      cfgs: configs.RpaConfigs | None = None,
      **kwargs,
  ):
    assert cfgs is not None
    assert buffer_type == pltpu.BufferType.INPUT_OUTPUT
    return super().create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        use_lookahead=use_lookahead,
        cfgs=cfgs,
        **kwargs,
    )

  def copy_in(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    kv_cache_hbm, new_kv_hbm, schedule_ref = self.src_ref
    sem = self.sem_recvs.at[slot]
    vmem_dst = self.window_ref.at[slot, :, :, : self.cfgs.kv_hbm_stride]
    kv_cache_hbm_flat = kv_cache_hbm.reshape(-1, *kv_cache_hbm.shape[2:])

    for b in range(self.cfgs.batch_size):
      for i in range(self.cfgs.bkv_p_cache):
        cache_dma = schedule_ref.dma_kv_cache[block_idx, b, i]
        src_off = cache_dma.src_hbm * self.cfgs.serve.page_size
        dst_off = cache_dma.dst_vmem
        sz = jnp.where(is_no_op, 0, cache_dma.length)
        pltpu.make_async_copy(
            kv_cache_hbm_flat.at[pl.ds(src_off, sz)],
            vmem_dst.at[b, pl.ds(dst_off, sz)],
            sem,
        ).start()

      new_dma_0 = schedule_ref.dma_kv_new[block_idx, b, 0]
      src_new_off = new_dma_0.fetch_hbm
      dst_vmem_off = new_dma_0.fetch_vmem
      total_new_sz = 0
      for i in range(self.cfgs.bkv_p_new):
        total_new_sz += schedule_ref.dma_kv_new[block_idx, b, i].fetch_len
      total_new_sz = jnp.where(is_no_op, 0, total_new_sz)
      pltpu.make_async_copy(
          new_kv_hbm.at[pl.ds(src_new_off, total_new_sz)],
          vmem_dst.at[b, pl.ds(dst_vmem_off, total_new_sz)],
          sem,
      ).start()

  def copy_out(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    kv_out_ref, _, schedule_ref = self.dst_ref
    sem = self.sem_sends.at[slot]
    kv_out_ref_flat = kv_out_ref.reshape(-1, *kv_out_ref.shape[2:])
    vmem_src = self.window_ref.at[slot, :, :, : self.cfgs.kv_hbm_stride]

    for b in range(self.cfgs.batch_size):
      do_writeback = schedule_ref.tile_meta[block_idx, b].has_writeback
      for i in range(self.cfgs.bkv_p_new):
        new_dma = schedule_ref.dma_kv_new[block_idx, b, i]
        dst_hbm_off = new_dma.wb_hbm
        src_vmem_off = new_dma.wb_vmem
        sz = jnp.where(
            jnp.logical_or(is_no_op, jnp.logical_not(do_writeback)),
            0,
            new_dma.wb_len,
        )
        pltpu.make_async_copy(
            vmem_src.at[b, pl.ds(src_vmem_off, sz)],
            kv_out_ref_flat.at[pl.ds(dst_hbm_off, sz)],
            sem,
        ).start()

  def wait_in(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    _, _, schedule_ref = self.src_ref
    wait_lanes = jnp.where(is_no_op, 0, schedule_ref.waits[block_idx].kv_in)
    _wait_lanes(
        self.window_ref.at[slot],
        wait_lanes,
        self.sem_recvs.at[slot],
        layout=self.cfgs.layout,
    )

  def wait_out(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    _, _, schedule_ref = self.dst_ref
    wait_lanes = jnp.where(is_no_op, 0, schedule_ref.waits[block_idx].kv_out)
    _wait_lanes(
        self.window_ref.at[slot],
        wait_lanes,
        self.sem_sends.at[slot],
        layout=self.cfgs.layout,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingQRef(BypassRef):
  """Fetches batched query tiles using precomputed schedule descriptors."""

  cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool = False,
      cfgs: configs.RpaConfigs | None = None,
      **kwargs,
  ):
    assert cfgs is not None
    assert buffer_type == pltpu.BufferType.INPUT
    return super().create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        use_lookahead=use_lookahead,
        cfgs=cfgs,
        **kwargs,
    )

  def copy_in(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    q_hbm, schedule_ref = self.src_ref
    sem = self.sem_recvs.at[slot]
    vmem_dst = self.window_ref.at[slot]

    for b in range(self.cfgs.batch_size):
      q_dma = schedule_ref.dma_q[block_idx, b]
      q_src = q_dma.src_hbm
      sz = jnp.where(is_no_op, 0, q_dma.length)
      pltpu.make_async_copy(
          q_hbm.at[:, pl.ds(q_src, sz)],
          vmem_dst.at[b, :, pl.ds(0, sz)],
          sem,
      ).start()

  def wait_in(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    _, schedule_ref = self.src_ref
    wait_lanes = jnp.where(is_no_op, 0, schedule_ref.waits[block_idx].q_in)
    _wait_lanes(
        self.window_ref.at[slot],
        wait_lanes,
        self.sem_recvs.at[slot],
        layout=self.cfgs.layout,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingORef(BypassRef):
  """Writes normalized attention output tiles back to HBM."""

  cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool = False,
      cfgs: configs.RpaConfigs | None = None,
      **kwargs,
  ):
    assert cfgs is not None
    assert buffer_type == pltpu.BufferType.OUTPUT
    return super().create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        use_lookahead=use_lookahead,
        cfgs=cfgs,
        **kwargs,
    )

  def copy_out(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    o_hbm, schedule_ref = self.dst_ref
    sem = self.sem_sends.at[slot]
    vmem_src = self.window_ref.at[slot]

    for b in range(self.cfgs.batch_size):
      is_last_k = schedule_ref.tile_meta[block_idx, b].is_final_k
      q_dma = schedule_ref.dma_q[block_idx, b]
      q_src = q_dma.src_hbm
      sz = jnp.where(
          jnp.logical_or(is_no_op, jnp.logical_not(is_last_k)), 0, q_dma.length
      )
      pltpu.make_async_copy(
          vmem_src.at[b, :, pl.ds(0, sz)],
          o_hbm.at[:, pl.ds(q_src, sz)],
          sem,
      ).start()

  def wait_out(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    o_hbm, schedule_ref = self.dst_ref
    wait_lanes = jnp.where(is_no_op, 0, schedule_ref.waits[block_idx].o_out)
    _wait_lanes(
        o_hbm,
        wait_lanes,
        self.sem_sends.at[slot],
        layout=self.cfgs.layout,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingLSERef(BypassRef):
  """Writes per-token log-sum-exp tiles back to HBM."""

  cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool = False,
      cfgs: configs.RpaConfigs | None = None,
      **kwargs,
  ):
    assert cfgs is not None
    assert buffer_type == pltpu.BufferType.OUTPUT
    return super().create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        use_lookahead=use_lookahead,
        cfgs=cfgs,
        **kwargs,
    )

  def copy_out(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    lse_hbm, schedule_ref = self.dst_ref
    sem = self.sem_sends.at[slot]
    vmem_src = self.window_ref.at[slot]

    for b in range(self.cfgs.batch_size):
      is_last_k = schedule_ref.tile_meta[block_idx, b].is_final_k
      q_dma = schedule_ref.dma_q[block_idx, b]
      q_src = q_dma.src_hbm
      q_sz = jnp.where(
          jnp.logical_or(is_no_op, jnp.logical_not(is_last_k)), 0, q_dma.length
      )
      pltpu.make_async_copy(
          vmem_src.at[b, :, pl.ds(0, q_sz)],
          lse_hbm.at[:, pl.ds(q_src, q_sz)],
          sem,
      ).start()

  def wait_out(
      self,
      step: int | jax.Array,
      slot: int | jax.Array,
      total_steps: int | jax.Array,
  ):
    is_no_op, block_idx = _step_guard(step, total_steps)
    lse_hbm, schedule_ref = self.dst_ref
    wait_lanes = jnp.where(is_no_op, 0, schedule_ref.waits[block_idx].lse_out)
    _wait_lanes(
        lse_hbm,
        wait_lanes,
        self.sem_sends.at[slot],
        layout=self.cfgs.layout,
    )


def create_allocs(
    kv_cache_hbm_ref: jax.Ref,
    q_hbm_ref: jax.Ref,
    lse_hbm_ref: jax.Ref | None,
    cfgs: configs.RpaConfigs,
) -> tuple[
    BatchingQRef,
    KVBufferedRefSeqAlongLane | KVBufferedRefHeadAlongSublane,
    BatchingORef,
    BatchingLSERef | None,
]:
  """Creates unbound multi-buffered VMEM allocations for Q, KV, O, and LSE."""
  in_slots = cfgs.num_input_slots
  out_slots = cfgs.num_output_slots
  kv_cache_spec = pl.BlockSpec(
      block_shape=cfgs.kv_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(buffer_count=in_slots, use_lookahead=False),
  )
  q_spec = pl.BlockSpec(
      block_shape=cfgs.q_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(buffer_count=in_slots, use_lookahead=False),
  )
  o_spec = pl.BlockSpec(
      block_shape=cfgs.q_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(buffer_count=out_slots, use_lookahead=False),
  )

  if cfgs.is_seq_along_lane:
    kv_cache_alloc_cls = KVBufferedRefSeqAlongLane
  else:
    kv_cache_alloc_cls = KVBufferedRefHeadAlongSublane

  kv_cache_alloc = kv_cache_alloc_cls.input_output(
      spec=kv_cache_spec,
      dtype_or_type=kv_cache_hbm_ref,
      buffer_count=in_slots,
      use_lookahead=False,
      cfgs=cfgs,
  )
  q_alloc = BatchingQRef.input(
      spec=q_spec,
      dtype_or_type=q_hbm_ref,
      buffer_count=in_slots,
      use_lookahead=False,
      cfgs=cfgs,
  )
  o_alloc = BatchingORef.output(
      spec=o_spec,
      dtype_or_type=q_hbm_ref,
      buffer_count=out_slots,
      use_lookahead=False,
      cfgs=cfgs,
  )

  lse_alloc = None
  if cfgs.serve.return_lse:
    lse_spec = pl.BlockSpec(
        block_shape=cfgs.lse_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i,),
        pipeline_mode=pl.Buffered(buffer_count=out_slots, use_lookahead=False),
    )
    lse_alloc = BatchingLSERef.output(
        spec=lse_spec,
        dtype_or_type=lse_hbm_ref,
        buffer_count=out_slots,
        use_lookahead=False,
        cfgs=cfgs,
    )

  return q_alloc, kv_cache_alloc, o_alloc, lse_alloc


def zero_initialize_buffers(
    q_alloc: BatchingQRef,
    kv_cache_alloc: KVBufferedRefSeqAlongLane | KVBufferedRefHeadAlongSublane,
    *,
    num_lanes: int,
) -> None:
  """Zero-initializes Q and KV VMEM buffers to prevent residual NaN pollution."""
  q_ref_flat = q_alloc.window_ref.bitcast(jnp.uint32).reshape(-1, num_lanes)
  kv_ref_flat = kv_cache_alloc.window_ref.bitcast(jnp.uint32).reshape(
      -1, num_lanes
  )
  q_ref_flat[...] = jnp.zeros_like(q_ref_flat)
  kv_ref_flat[...] = jnp.zeros_like(kv_ref_flat)
