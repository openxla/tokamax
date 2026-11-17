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
"""Self-contained RpaSchedule tape schema and SMEM/HBM chunk DMA helpers."""

import dataclasses
from typing import Any, ClassVar

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import utils


class FieldOffset:
  """Descriptor mapping a named struct field to `obj.data[obj.pos + offset]`."""

  def __init__(self, offset: int):
    self.offset = offset

  def __get__(self, obj: Any, objtype: Any = None) -> Any:
    if obj is None:
      return self
    return obj.data[obj.pos + self.offset]

  def __set__(self, obj: Any, value: Any) -> None:
    obj.data[obj.pos + self.offset] = value


class StructView:
  """View over `NUM_WORDS` contiguous `int32` words in `data` at `pos`."""

  NUM_WORDS: ClassVar[int]
  __slots__ = ("data", "pos")

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    fields: dict[str, int] = {}
    for base in reversed(cls.__mro__):
      for name, val in vars(base).items():
        if isinstance(val, FieldOffset):
          fields[name] = val.offset
    offsets = sorted(fields.values())
    if len(offsets) != len(set(offsets)):
      raise TypeError(
          f"{cls.__name__} has duplicate FieldOffset indices: {fields}"
      )
    if offsets != list(range(len(offsets))):
      raise TypeError(
          f"{cls.__name__} FieldOffsets must cover 0..{len(offsets) - 1}, got"
          f" {fields}"
      )
    if "NUM_WORDS" in vars(cls) and cls.NUM_WORDS != len(offsets):
      raise TypeError(
          f"{cls.__name__}.NUM_WORDS={cls.NUM_WORDS} != {len(offsets)}"
          " FieldOffsets"
      )
    cls.NUM_WORDS = len(offsets)

  def __init__(self, data: Any, pos: Any):
    self.data = data
    self.pos = pos

  def clear(self) -> None:
    """Zeroes all `NUM_WORDS` words of this struct entry."""
    for word in range(self.NUM_WORDS):
      self.data[self.pos + word] = 0


class TileMeta(StructView):
  """Per-`(step, lane)` compute & writeback metadata (6 `int32` words)."""

  NUM_WORDS = 6
  EMPTY_SEQ_IDX: ClassVar[int] = -1

  s_idx = FieldOffset(0)  # EMPTY_SEQ_IDX (-1) = empty lane; else seq index
  is_last_k = FieldOffset(1)  # 0 = partial K, 1 = final K (commit O / LSE)
  do_writeback = FieldOffset(2)  # 0 / 1: scatter new KV tokens into kv_cache
  causal_offset = FieldOffset(3)  # k_global - q_global (mask: q - k >= offset)
  k_min = FieldOffset(4)  # lower KV tile token bound (NEW_TOKENS_ONLY)
  k_max = FieldOffset(5)  # upper KV tile token bound (CACHE_ONLY)

  @property
  def has_writeback(self) -> Any:
    """Whether this tile scatters new KV tokens back into `kv_cache`."""
    return self.do_writeback == 1

  @property
  def is_final_k(self) -> Any:
    """Whether this is the final KV tile of the Q block (commits O / LSE)."""
    return self.is_last_k == 1


class SeqAlongLaneTileMeta(TileMeta):
  """`TileMeta` extended with `SEQ_ALONG_LANE` VMEM stitch offsets (9 words)."""

  NUM_WORDS = 9
  stitch_dst = FieldOffset(6)  # cache prefix length in VMEM [0, bkv_sz]
  stitch_src = FieldOffset(7)  # staged new-KV start offset in VMEM
  stitch_len = FieldOffset(8)  # valid new-KV token count to stitch


class QFetchDma(StructView):
  """Per-`(step, lane)` query fetch descriptor (2 `int32` words)."""

  NUM_WORDS = 2
  src_hbm = FieldOffset(0)  # token start index in packed q_hbm / o_hbm
  length = FieldOffset(1)  # valid query token count in [0, bq_sz]


class CacheFetchDma(StructView):
  """Per-`(step, lane, page_slot)` KV-cache fetch descriptor (3 words)."""

  NUM_WORDS = 3
  src_hbm = FieldOffset(0)  # physical page ID in HBM kv_cache
  dst_vmem = FieldOffset(1)  # token offset in VMEM KV slot
  length = FieldOffset(2)  # token count to fetch (lane-aligned for SEQ)


class NewKvDma(StructView):
  """Per-`(step, lane, page_slot)` new-KV fetch & writeback descriptor."""

  NUM_WORDS = 5
  fetch_hbm = FieldOffset(0)  # source token offset in new_kv_hbm
  fetch_vmem = FieldOffset(1)  # destination token offset in VMEM KV slot
  wb_hbm = FieldOffset(2)  # packed (phys_page << page_size_log2) | page_offset
  wb_vmem = FieldOffset(3)  # source token offset in VMEM KV slot for writeback
  packed_lens = FieldOffset(4)  # (fetch_len & 0xFFFF) | (wb_len << 16)

  @property
  def fetch_len(self) -> Any:
    """Unpacks new-KV fetch token count from the low 16 bits."""
    return self.packed_lens & 0xFFFF

  @property
  def wb_len(self) -> Any:
    """Unpacks KV-cache writeback token count from the high 16 bits."""
    return (self.packed_lens >> 16) & 0xFFFF

  def set_lens(self, fetch_len: Any, wb_len: Any) -> None:
    """Packs fetch (low 16 bits) and writeback (high 16 bits) token counts."""
    self.packed_lens = (fetch_len & 0xFFFF) | (wb_len << 16)


class StepWaits(StructView):
  """Per-`step` DMA wait row counts (in `layout.dma_wait_row_bytes` units)."""

  NUM_WORDS = 5
  kv_in = FieldOffset(0)
  kv_out = FieldOffset(1)
  q_in = FieldOffset(2)
  o_out = FieldOffset(3)
  lse_out = FieldOffset(4)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemStructArray:
  """Maps a flat 1-D `int32` buffer onto an N-D array of `StructView`s."""

  data: Any
  shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))
  struct_cls: type[StructView] = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create_shape_dtype(
      cls, shape: tuple[int, ...], struct_cls: type[StructView]
  ) -> "SmemStructArray":
    num_words = int(np.prod(shape)) * struct_cls.NUM_WORDS
    return cls(
        data=jax.ShapeDtypeStruct((num_words,), jnp.int32),
        shape=shape,
        struct_cls=struct_cls,
    )

  def _get_pos(self, indices: Any) -> Any:
    if not isinstance(indices, tuple):
      indices = (indices,)
    strides = pl.strides_from_shape(self.shape)
    assert len(strides) == len(indices)
    pos = 0
    for stride, idx in zip(strides, indices):
      pos += stride * idx
    return pos * self.struct_cls.NUM_WORDS

  def __getitem__(self, indices: Any) -> Any:
    return self.struct_cls(self.data, self._get_pos(indices))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RpaSchedule:
  """Self-contained schedule tape connecting planner and attention executor."""

  # HBM <-> SMEM schedule chunk DMAs align step offsets to 128 and pad SMEM
  # chunk reads to 1024 int32 words.
  DMA_STEP_ALIGN: ClassVar[int] = 128
  READ_WORD_ALIGN: ClassVar[int] = 1024

  tile_meta: (
      SmemStructArray  # [steps, batch] -> TileMeta | SeqAlongLaneTileMeta
  )
  dma_q: SmemStructArray  # [steps, batch] -> QFetchDma
  dma_kv_cache: SmemStructArray  # [steps, batch, bkv_p_cache] -> CacheFetchDma
  dma_kv_new: SmemStructArray  # [steps, batch, bkv_p_new] -> NewKvDma
  waits: SmemStructArray  # [steps] -> StepWaits
  actual_steps: Any  # int32[1]
  cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def bytes_per_step(cls, cfgs: configs.RpaConfigs) -> int:
    """Returns exact SMEM bytes per step summed from struct `NUM_WORDS`."""
    meta_cls = SeqAlongLaneTileMeta if cfgs.is_seq_along_lane else TileMeta
    lane_words = (
        meta_cls.NUM_WORDS
        + QFetchDma.NUM_WORDS
        + CacheFetchDma.NUM_WORDS * cfgs.bkv_p_cache
        + NewKvDma.NUM_WORDS * cfgs.bkv_p_new
    )
    return 4 * (lane_words * cfgs.batch_size + StepWaits.NUM_WORDS)

  @classmethod
  def max_steps_ub(cls, cfgs: configs.RpaConfigs) -> int:
    """Returns maximum SMEM schedule steps per chunk under the SMEM budget."""
    fixed_words = (
        cfgs.serve.num_seqs  # kv_lens
        + (cfgs.serve.num_seqs + 1)  # cu_q_lens
        + 2 * cfgs.page_window_slot_len  # page_indices windows
        + 3  # distribution
        + cfgs.batch_size  # lane_lengths
        + 1  # actual_steps
    )
    fixed_bytes = fixed_words * 4
    smem_limit_bytes = (
        pltpu.get_tpu_info().smem_capacity_bytes - 32 * 1024
    ) * cfgs.serve.smem_fraction_limit_for_schedule_generation
    available_bytes = smem_limit_bytes - fixed_bytes
    max_steps = available_bytes // cls.bytes_per_step(cfgs)
    num_lanes = cfgs.layout.num_lanes
    return int(max(1, max_steps // num_lanes) * num_lanes)

  @classmethod
  def create_shape_dtype(
      cls, cfgs: configs.RpaConfigs, multiplier: int = 1
  ) -> "RpaSchedule":
    """Creates the `ShapeDtypeStruct` PyTree for SMEM (`1`) or HBM (`>1`)."""
    steps = cls.max_steps_ub(cfgs) * multiplier
    batch = cfgs.batch_size
    meta_cls = SeqAlongLaneTileMeta if cfgs.is_seq_along_lane else TileMeta
    sched = cls(
        tile_meta=SmemStructArray.create_shape_dtype((steps, batch), meta_cls),
        dma_q=SmemStructArray.create_shape_dtype((steps, batch), QFetchDma),
        dma_kv_cache=SmemStructArray.create_shape_dtype(
            (steps, batch, cfgs.bkv_p_cache), CacheFetchDma
        ),
        dma_kv_new=SmemStructArray.create_shape_dtype(
            (steps, batch, cfgs.bkv_p_new), NewKvDma
        ),
        waits=SmemStructArray.create_shape_dtype((steps,), StepWaits),
        actual_steps=jax.ShapeDtypeStruct((1,), jnp.int32),
        cfgs=cfgs,
    )
    words_per_step = sum(a.data.shape[0] for a in sched.step_arrays()) // steps
    assert words_per_step * 4 == cls.bytes_per_step(cfgs)
    return sched

  def step_arrays(self) -> tuple[SmemStructArray, ...]:
    """Returns the step-indexed `SmemStructArray` leaves in deterministic order."""
    return (
        self.tile_meta,
        self.dma_q,
        self.dma_kv_cache,
        self.dma_kv_new,
        self.waits,
    )

  def scratch_shapes(self) -> Any:
    """Returns a matching `RpaSchedule` PyTree of SMEM scratch allocations."""
    return jax.tree.map(lambda x: pltpu.SMEM(x.shape, x.dtype), self)

  def in_specs(self) -> Any:
    """Returns input `BlockSpec`s (`HBM` for step tables, `SMEM` for count)."""
    hbm_spec = lambda arr: jax.tree.map(
        lambda _: pl.BlockSpec(memory_space=pltpu.HBM), arr
    )
    return dataclasses.replace(
        self,
        tile_meta=hbm_spec(self.tile_meta),
        dma_q=hbm_spec(self.dma_q),
        dma_kv_cache=hbm_spec(self.dma_kv_cache),
        dma_kv_new=hbm_spec(self.dma_kv_new),
        waits=hbm_spec(self.waits),
        actual_steps=pl.BlockSpec(memory_space=pltpu.SMEM),
    )

  def out_specs(self) -> Any:
    """Returns output `BlockSpec`s placing all schedule buffers in `HBM`."""
    return jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.HBM), self)

  def clear_slot(self, step: Any, b_idx: Any) -> None:
    """Masks out an unvisited `(step, b_idx)` schedule slot."""
    meta = self.tile_meta[step, b_idx]
    meta.clear()
    meta.s_idx = TileMeta.EMPTY_SEQ_IDX
    self.dma_q[step, b_idx].clear()
    for i in range(self.cfgs.bkv_p_cache):
      self.dma_kv_cache[step, b_idx, i].clear()
    for i in range(self.cfgs.bkv_p_new):
      self.dma_kv_new[step, b_idx, i].clear()
    self.waits[step].clear()

  def compute_waits(self, start_step: Any, end_step: Any) -> None:
    """Populates per-step DMA wait row counts over `[start_step, end_step)`."""
    cfgs = self.cfgs
    layout = cfgs.layout

    @jax.named_scope("compute_waits")
    def body(step, _):
      kv_in_tokens = 0
      kv_out_tokens = 0
      q_in_tokens = 0
      o_out_tokens = 0

      for b in range(cfgs.batch_size):
        meta = self.tile_meta[step, b]
        for i in range(cfgs.bkv_p_cache):
          kv_in_tokens += self.dma_kv_cache[step, b, i].length
        for i in range(cfgs.bkv_p_new):
          new_dma = self.dma_kv_new[step, b, i]
          kv_in_tokens += new_dma.fetch_len
          kv_out_tokens += jnp.where(meta.has_writeback, new_dma.wb_len, 0)

        q_sz = self.dma_q[step, b].length
        q_in_tokens += q_sz
        o_out_tokens += jnp.where(meta.is_final_k, q_sz, 0)

      waits = self.waits[step]
      waits.kv_in = layout.tokens_to_wait_rows(
          kv_in_tokens, layout.kv_bytes_per_token
      )
      waits.kv_out = layout.tokens_to_wait_rows(
          kv_out_tokens, layout.kv_bytes_per_token
      )
      waits.q_in = layout.tokens_to_wait_rows(
          q_in_tokens, layout.q_bytes_per_token
      )
      waits.o_out = layout.tokens_to_wait_rows(
          o_out_tokens, layout.o_bytes_per_token
      )
      waits.lse_out = layout.tokens_to_wait_rows(
          o_out_tokens, layout.lse_bytes_per_token
      )

    jax.lax.fori_loop(start_step, end_step, body, None)

  def write_chunk_to_hbm(
      self,
      schedule_hbm_ref: "RpaSchedule",
      hbm_offset: Any,
      num_steps: Any,
      dma_sem: jax.Ref,
  ) -> None:
    """Writes `num_steps` of SMEM schedule (`self`) to `schedule_hbm_ref`."""
    max_steps = self.max_steps_ub(self.cfgs)
    hbm_offset_aligned = pl.multiple_of(
        jnp.asarray(hbm_offset), self.DMA_STEP_ALIGN
    )
    dma_list = []
    for h_arr, s_arr in zip(schedule_hbm_ref.step_arrays(), self.step_arrays()):
      s = s_arr.data
      h = h_arr.data
      if s.shape[0] == 0:
        continue
      element_size = s.shape[0] // max_steps
      write_size = utils.align_to(num_steps * element_size, self.DMA_STEP_ALIGN)
      output_offset = hbm_offset_aligned * element_size
      dma_list.append(
          pltpu.make_async_copy(
              s.at[pl.ds(0, write_size)],
              h.at[pl.ds(output_offset, write_size)],
              dma_sem.at[0],
          )
      )
    dma_list.append(
        pltpu.make_async_copy(
            self.actual_steps.at[pl.ds(0, 1)],
            schedule_hbm_ref.actual_steps.at[pl.ds(0, 1)],
            dma_sem.at[0],
        )
    )
    for copy in dma_list:
      copy.start()
    for copy in dma_list:
      copy.wait()

  def read_chunk_from_hbm(
      self,
      schedule_hbm_ref: "RpaSchedule",
      start_step: Any,
      num_steps: Any,
      dma_sem: jax.Ref,
  ) -> jax.Array:
    """Reads `[start_step, start_step + num_steps)` from HBM into SMEM (`self`)."""
    max_steps = self.max_steps_ub(self.cfgs)
    aligned_start_step = (
        start_step // self.DMA_STEP_ALIGN
    ) * self.DMA_STEP_ALIGN
    prefix_steps = start_step - aligned_start_step
    total_steps = jnp.asarray(num_steps + prefix_steps)

    dma_list = []
    for h_arr, s_arr in zip(schedule_hbm_ref.step_arrays(), self.step_arrays()):
      s = s_arr.data
      h = h_arr.data
      if s.shape[0] == 0:
        continue
      element_size = s.shape[0] // max_steps
      read_size = utils.align_to(
          element_size * total_steps, self.READ_WORD_ALIGN
      )
      read_size = jnp.minimum(read_size, s.shape[0])
      src_off = pl.multiple_of(
          jnp.asarray(element_size * aligned_start_step), self.DMA_STEP_ALIGN
      )
      copy = pltpu.make_async_copy(
          h.at[pl.ds(src_off, read_size)],
          s.at[pl.ds(0, read_size)],
          dma_sem.at[0],
      )
      copy.start()
      dma_list.append(copy)
    for copy in dma_list:
      copy.wait()
    return total_steps

  def flush_to_hbm(
      self,
      count: Any,
      schedule_hbm_ref: "RpaSchedule",
      hbm_offset: Any,
      dma_sem: jax.Ref,
  ) -> tuple[Any, int]:
    """Masks trailing slots, computes wait rows, and flushes chunk to HBM."""
    cfgs = self.cfgs
    max_steps = self.max_steps_ub(cfgs)
    last_step = utils.align_to(count, cfgs.batch_size)

    @pl.loop(count, last_step)
    @jax.named_scope("mask_out_steps")
    def _mask_trailing(idx):
      step, b_idx = divmod(idx, cfgs.batch_size)
      self.clear_slot(step, b_idx)

    self.compute_waits(0, last_step // cfgs.batch_size)
    self.write_chunk_to_hbm(schedule_hbm_ref, hbm_offset, max_steps, dma_sem)
    return hbm_offset + max_steps, 0
