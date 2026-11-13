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
"""DMA schedule data structures and HBM/SMEM flush helpers for RPA."""

from abc import ABC, abstractmethod
import dataclasses
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import utils


class FieldOffset:
  """A Python descriptor that generates the `.at[pos + offset]` lazy lookup.

  This is necessary because JAX does not support dynamically slicing a
  range (e.g. `data.at[pos:pos+4]`) using traced indices inside a loop,
  but it natively supports retrieving/updating single dynamically-indexed
  elements (e.g. `data.at[pos+1]`).
  """

  def __init__(self, offset: int | None = None, is_abstract: bool = False):
    if offset is None:
      assert is_abstract

    self.offset = offset
    self.__isabstractmethod__ = is_abstract

  def __get__(self, obj, objtype=None):
    return obj.data.at[obj.pos + self.offset]


@dataclasses.dataclass(frozen=True)
class DmaNew(ABC):
  data: Any
  pos: Any

  # New-KV fetch. The new tokens are contiguous in HBM and in VMEM, so a
  # single DMA covers them all: copy_in takes the addresses from entry 0 and
  # the length from the sum of fetch_val over all entries.
  fetch_hbm = FieldOffset(is_abstract=True)
  fetch_vmem = FieldOffset(is_abstract=True)
  # Writeback, which stays per-entry because page_indices scatters the
  # destinations. wb_hbm packs them as (page_idx << page_size_log2) | p_off.
  wb_hbm = FieldOffset(is_abstract=True)
  wb_vmem = FieldOffset(is_abstract=True)
  # Fetch/writeback token counts, packed per the subclass's field layout.
  _flags = FieldOffset(is_abstract=True)

  @staticmethod
  @abstractmethod
  def num_fields() -> int:
    ...

  @abstractmethod
  def set_flags(self, fetch_val, wb_val):
    ...

  @property
  @abstractmethod
  def fetch_val(self):
    ...

  @property
  @abstractmethod
  def wb_val(self):
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SeqAlongLaneDmaNew(DmaNew):
  fetch_hbm = FieldOffset(0)
  fetch_vmem = FieldOffset(1)
  wb_hbm = FieldOffset(2)
  wb_vmem = FieldOffset(3)
  _flags = FieldOffset(4)

  @staticmethod
  def num_fields() -> int:
    return 5

  @property
  def fetch_val(self):
    # First 16 bits are for fetch
    return self._flags[...] & 0xFFFF

  @property
  def wb_val(self):
    # Last 16 bits are for writeback
    return self._flags[...] >> 16

  def set_flags(self, fetch_val, wb_val):
    self._flags[...] = fetch_val | (wb_val << 16)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class HeadAlongSublaneDmaNew(DmaNew):
  # Each page writes back exactly the tokens it fetched, from the offset it
  # fetched them into, so vmem offsets and token counts alias down to 4 slots.
  wb_hbm = FieldOffset(0)
  fetch_hbm = FieldOffset(1)
  fetch_vmem = FieldOffset(2)
  wb_vmem = FieldOffset(2)
  _flags = FieldOffset(3)

  @staticmethod
  def num_fields() -> int:
    return 4

  @property
  def fetch_val(self):
    return self._flags[...]

  @property
  def wb_val(self):
    return self._flags[...]

  def set_flags(self, fetch_val, wb_val):
    self._flags[...] = fetch_val


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class HeadAlongSublaneDmaNewCP(DmaNew):
  """Like HeadAlongSublaneDmaNew but with separate fetch/wb flags for CP.

  wb_val encodes per-page ownership (dma_sz if this rank owns the page,
  else 0), so bref_override copy_out needs no CP-specific logic.
  """

  wb_hbm = FieldOffset(0)
  fetch_hbm = FieldOffset(1)
  fetch_vmem = FieldOffset(2)
  wb_vmem = FieldOffset(2)
  _fetch_flags = FieldOffset(3)
  _wb_flags = FieldOffset(4)

  @staticmethod
  def num_fields() -> int:
    return 5

  @property
  def fetch_val(self):
    return self._fetch_flags[...]

  @property
  def wb_val(self):
    return self._wb_flags[...]

  def set_flags(self, fetch_val, wb_val):
    self._fetch_flags[...] = fetch_val
    self._wb_flags[...] = wb_val


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
  """Maps physical 1-D data into logical N-D representation."""

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemArrayOfStructs(SmemWrapper):
  """Maps physical 1-D data into logical Array of Structs."""

  struct_cls: type[DmaNew] = dataclasses.field(metadata=dict(static=True))
  struct_size: int = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create_shape_dtype(cls, shape, struct_cls, struct_size):  # pyrefly: ignore[bad-override]
    assert struct_size == struct_cls.num_fields()
    return cls(
        data=jax.ShapeDtypeStruct((np.prod(shape) * struct_size,), jnp.int32),
        shape=shape,
        struct_cls=struct_cls,
        struct_size=struct_size,
    )

  def _get_pos(self, indices):
    strides = pl.strides_from_shape(self.shape)
    assert len(strides) == len(indices)

    pos = 0
    for stride, idx in zip(strides, indices):
      pos += stride * idx
    return pos * self.struct_size

  def __getitem__(self, indices):
    pos_start = self._get_pos(indices)
    return self.struct_cls(self.data, pos_start)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RpaSchedule:
  """Container for metadata arrays with integrated shape/spec logic."""

  s_idx: SmemWrapper  # [steps, batch]
  q_idx: SmemWrapper  # [steps, batch]
  k_idx: SmemWrapper  # [steps, batch]
  is_last_k: SmemWrapper  # [steps, batch]
  do_writeback: SmemWrapper  # [steps, batch]
  dma_q: SmemWrapper  # [steps, batch, 2]
  dma_kv_cache: SmemWrapper  # [steps, batch, bkv_p_cache, 3]
  dma_kv_new: SmemArrayOfStructs  # [steps, batch, bkv_p_new]
  total_wait_kv_in: SmemWrapper  # [steps]
  total_wait_kv_out: SmemWrapper  # [steps]
  total_wait_q_in: SmemWrapper  # [steps]
  total_wait_o_out: SmemWrapper  # [steps]
  total_wait_lse_out: SmemWrapper  # [steps]
  actual_steps: Any  # [1]

  cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))
  kv_interleaved_size: int = dataclasses.field(
      default=1, metadata=dict(static=True)
  )

  @classmethod
  def create_shape_dtype(
      cls,
      cfgs: configs.RpaConfigs,
      dma_kv_new_struct_cls: type[DmaNew] = SeqAlongLaneDmaNew,
      multiplier: int = 1,
      kv_interleaved_size: int = 1,
  ):
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
            (effective_max_steps, cfgs.batch_size, cfgs.bkv_p_cache, 3)
        ),
        dma_kv_new=SmemArrayOfStructs.create_shape_dtype(
            (
                effective_max_steps,
                cfgs.batch_size,
                cfgs.bkv_p_new,
            ),
            struct_cls=dma_kv_new_struct_cls,
            struct_size=dma_kv_new_struct_cls.num_fields(),
        ),
        total_wait_kv_in=steps_wrapper,
        total_wait_kv_out=steps_wrapper,
        total_wait_q_in=steps_wrapper,
        total_wait_o_out=steps_wrapper,
        total_wait_lse_out=steps_wrapper,
        actual_steps=jax.ShapeDtypeStruct((1,), jnp.int32),
        cfgs=cfgs,
        kv_interleaved_size=kv_interleaved_size,
    )

  def get_dma_kv_cache(
      self,
      step: jax.typing.ArrayLike,
      batch_idx: jax.typing.ArrayLike,
      page_idx: jax.typing.ArrayLike,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    # 0: src_hbm, 1: dst_vmem, 2: size
    src_off = self.dma_kv_cache[step, batch_idx, page_idx, 0]
    dst_off = self.dma_kv_cache[step, batch_idx, page_idx, 1]
    sz = self.dma_kv_cache[step, batch_idx, page_idx, 2]
    return src_off, dst_off, sz

  def get_dma_q(
      self, step: jax.typing.ArrayLike, batch_idx: jax.typing.ArrayLike
  ) -> tuple[jax.Array, jax.Array]:
    # 0: src_hbm, 1: size
    src_hbm = self.dma_q[step, batch_idx, 0]
    sz = self.dma_q[step, batch_idx, 1]
    return src_hbm, sz

  def scratch_shapes(self):
    """Returns a Pytree of SMEM scratch memory."""

    return jax.tree.map(
        lambda x: pltpu.SMEM(x.shape, x.dtype),
        self,
    )

  def in_specs(self):
    """Returns a Pytree of input BlockSpecs."""

    def wrapper(x):
      if x.size == 1:
        return pl.BlockSpec(memory_space=pltpu.SMEM)
      else:
        # Since we use maximum upper bound when allocating scheduler data,
        # it is not feasible to use scalar prefetch and fetch entire scheduler
        # data into the kernel. Instead, we stored it to HBM first and perform
        # dynamic sized DMA inside the kernel using actual number of steps.
        return pl.BlockSpec(memory_space=pltpu.HBM)

    return jax.tree.map(wrapper, self)

  def out_specs(self):
    """Returns a Pytree of output BlockSpecs."""

    return jax.tree.map(
        lambda x: pl.BlockSpec(memory_space=pltpu.HBM),
        self,
    )


def _mask_out_steps(
    step: jax.typing.ArrayLike,
    schedule_smem: RpaSchedule,
    b_idx: jax.typing.ArrayLike,
):
  """Mask out unvisited metadata schedule entries at (step, b_idx)."""
  schedule_smem.s_idx[step, b_idx] = -1
  schedule_smem.q_idx[step, b_idx] = 0
  schedule_smem.k_idx[step, b_idx] = 0
  schedule_smem.is_last_k[step, b_idx] = 0
  schedule_smem.do_writeback[step, b_idx] = 0

  schedule_smem.dma_q[step, b_idx, 0] = 0
  schedule_smem.dma_q[step, b_idx, 1] = 0

  for i in range(schedule_smem.cfgs.bkv_p_cache):
    schedule_smem.dma_kv_cache[step, b_idx, i, 0] = 0
    schedule_smem.dma_kv_cache[step, b_idx, i, 1] = 0
    schedule_smem.dma_kv_cache[step, b_idx, i, 2] = 0

  for i in range(schedule_smem.cfgs.bkv_p_new):
    dma_entry = schedule_smem.dma_kv_new[step, b_idx, i]
    dma_entry.fetch_hbm[...] = 0
    dma_entry.fetch_vmem[...] = 0
    dma_entry.wb_hbm[...] = 0
    dma_entry.wb_vmem[...] = 0
    dma_entry.set_flags(0, 0)

  schedule_smem.total_wait_kv_in[step] = 0
  schedule_smem.total_wait_kv_out[step] = 0
  schedule_smem.total_wait_q_in[step] = 0
  schedule_smem.total_wait_o_out[step] = 0
  schedule_smem.total_wait_lse_out[step] = 0


def _write_schedule_to_hbm(
    schedule_smem: RpaSchedule,
    schedule_hbm: RpaSchedule,
    hbm_offset: jax.typing.ArrayLike,
    num_steps: jax.typing.ArrayLike,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):
  """Writes `num_steps` of metadata from `schedule_smem` to `schedule_hbm` at `hbm_offset`."""
  hbm_offset_aligned = pl.multiple_of(hbm_offset, 128)  # pyrefly: ignore[bad-argument-type]
  flat_hbm = jax.tree_util.tree_leaves(schedule_hbm)
  flat_smem = jax.tree_util.tree_leaves(schedule_smem)
  dma_list = []
  for h, s in zip(flat_hbm, flat_smem):
    element_size = s.shape[0] // cfgs.max_steps_ub
    if h.shape[0] > 1:
      write_size = num_steps * element_size
      write_size = utils.align_to(write_size, 128)
      output_offset = hbm_offset_aligned * element_size
    else:
      write_size = h.shape[0]
      output_offset = 0

    copy = pltpu.make_async_copy(
        s.at[pl.ds(0, write_size)],
        h.at[pl.ds(output_offset, write_size)],
        dma_sem.at[0],
    )
    dma_list.append(copy)

  jax.tree.map(lambda x: x.start(), dma_list)
  jax.tree.map(lambda x: x.wait(), dma_list)


def _compute_waits(
    schedule: RpaSchedule,
    start_step: jax.typing.ArrayLike,
    end_step: jax.typing.ArrayLike,
    *,
    cfgs: configs.RpaConfigs,
):
  """Computes total wait row/lane counts for a range of steps."""

  kv_bytes_per_token = cfgs.kv_bytes_per_token
  q_bytes_per_token = cfgs.q_bytes_per_token
  o_bytes_per_token = cfgs.o_bytes_per_token

  # Reshape buffers to (-1, 128) in u32 to perform contiguous 512-byte lane
  # waits (1 row of 128 u32 words = 512 bytes).
  num_lanes = pltpu.get_tpu_info().num_lanes
  dma_chunk_size = num_lanes * 4

  @jax.named_scope("compute_waits")
  def body(step, _):
    # KV IN
    kv_in_tokens = 0
    for b in range(cfgs.batch_size):
      for i in range(cfgs.bkv_p_cache):
        _, _, sz = schedule.get_dma_kv_cache(step, b, i)
        kv_in_tokens += sz
      for i in range(cfgs.bkv_p_new):
        kv_in_tokens += schedule.dma_kv_new[step, b, i].fetch_val

    schedule.total_wait_kv_in[step] = (
        kv_in_tokens * kv_bytes_per_token
    ) // dma_chunk_size

    # KV OUT
    kv_out_tokens = 0
    for b in range(cfgs.batch_size):
      do_writeback = schedule.do_writeback[step, b] == 1
      for i in range(cfgs.bkv_p_new):
        dma_entry = schedule.dma_kv_new[step, b, i]
        kv_out_tokens += jnp.where(do_writeback, dma_entry.wb_val, 0)

    schedule.total_wait_kv_out[step] = (
        kv_out_tokens * kv_bytes_per_token
    ) // dma_chunk_size

    # Q IN
    q_in_tokens = 0
    for b in range(cfgs.batch_size):
      _, q_sz = schedule.get_dma_q(step, b)
      q_in_tokens += q_sz
    schedule.total_wait_q_in[step] = (
        q_in_tokens * q_bytes_per_token
    ) // dma_chunk_size

    # O OUT
    o_out_tokens = 0
    for b in range(cfgs.batch_size):
      is_last_k = schedule.is_last_k[step, b] == 1
      _, q_sz = schedule.get_dma_q(step, b)
      o_out_tokens += jnp.where(is_last_k, q_sz, 0)
    schedule.total_wait_o_out[step] = (
        o_out_tokens * o_bytes_per_token
    ) // dma_chunk_size

    # LSE OUT
    lse_bytes_per_token = (
        cfgs.model.num_kv_heads
        * cfgs.aligned_num_q_heads_per_kv_head
        * 128
        * jnp.dtype(cfgs.serve.dtype_out).itemsize
    )
    schedule.total_wait_lse_out[step] = (
        o_out_tokens * lse_bytes_per_token
    ) // dma_chunk_size

  jax.lax.fori_loop(start_step, end_step, body, None)


def flush_to_hbm(
    count: jax.typing.ArrayLike,
    schedule: RpaSchedule,
    schedule_hbm_ref: RpaSchedule,
    hbm_offset: jax.typing.ArrayLike,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):

  last_step = utils.align_to(count, cfgs.batch_size)
  @pl.loop(count, last_step)
  @jax.named_scope("mask_out_steps")
  def body(idx):
    step, b_idx = divmod(idx, cfgs.batch_size)
    _mask_out_steps(step, b_idx=b_idx, schedule_smem=schedule)

  _compute_waits(schedule, 0, last_step // cfgs.batch_size, cfgs=cfgs)
  _write_schedule_to_hbm(
      schedule,
      schedule_hbm_ref,
      hbm_offset,
      cfgs.max_steps_ub,
      dma_sem,
      cfgs=cfgs,
  )

  return hbm_offset + cfgs.max_steps_ub, 0


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LoopCarry:
  hbm_offset: jax.Array | int
  count: jax.Array | int


def advance_carry(
    carry: LoopCarry,
    schedule_ref: RpaSchedule,
    schedule_hbm_ref: RpaSchedule,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
) -> LoopCarry:
  """Increments the step counter and flushes SMEM schedule to HBM when full."""
  new_count = carry.count + 1

  def _flush(c: LoopCarry) -> LoopCarry:
    new_hbm_offset, _ = flush_to_hbm(
        c.count,
        schedule_ref,
        schedule_hbm_ref,
        c.hbm_offset,
        dma_sem,
        cfgs=cfgs,
    )
    return LoopCarry(new_hbm_offset, 0)

  return jax.lax.cond(
      new_count < cfgs.max_steps_ub * cfgs.batch_size,
      lambda c: c,
      _flush,
      LoopCarry(carry.hbm_offset, new_count),
  )
