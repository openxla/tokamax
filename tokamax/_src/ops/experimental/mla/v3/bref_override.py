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
"""Pallas BufferedRef overrides for Batched MLA DMA pipelines.

- KVBufferedRefSeqAlongLane: Handles transposed [C_kv, K_pe] cache with dual new
KV inputs.
- BatchingQRef: Non-positional query (d_nope = 512) and positional query (d_pe =
128) inputs.
- BatchingORef: Output activations (d_nope = 512).
"""

import dataclasses
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import schedule

# To support older JAX versions.
if hasattr(pltpu, "BufferType"):
  BufferType = pltpu.BufferType
else:  # jax < 0.11
  from jax._src.pallas.mosaic.pipeline import BufferType  # pylint: disable=g-import-not-at-top

# The KV staging buffer needs separate (in, out) slot counts (see
# `MlaConfigs.kv_buffer_count`): its writeback is waited a step late, so with a
# single count a slot could be refilled before its writeback finished. Older
# JAX releases only take an int and would silently misbuild the buffer, so the
# kernel requires the feature rather than falling back.
SUPPORTS_IN_OUT_BUFFER_COUNT = hasattr(pltpu.BufferedRef, "in_buffer_count")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _BypassRef(pltpu.BufferedRef):
  """Helper class to safely bypass buffer_count checks during creation."""

  def __post_init__(self):
    # Pallas restricts n_buffer > 2 for output refs by default; override to
    # allow flexible pipelined buffer depths.
    pass


def _rebuild_with_cfgs(cls, standard_ref, cfgs):
  """Re-wraps a plain `pltpu.BufferedRef` as `cls`, attaching `cfgs`.

  Every subclass in this module adds exactly one field (`cfgs`) to
  `pltpu.BufferedRef`, so construction is a field-by-field copy of the standard
  ref plus that field. The copy is keyed by field *name*, so a Pallas release
  that renames or drops a `BufferedRef` field raises `TypeError` here rather
  than silently mis-wiring a buffer.

  This reaches into Pallas internals: `pltpu.BufferedRef` is not a stable public
  API. Verified against jax 0.11.x; revisit on JAX upgrades.
  """
  return cls(
      cfgs=cfgs,
      **{
          f.name: getattr(standard_ref, f.name)
          for f in dataclasses.fields(pltpu.BufferedRef)
      },
  )


# ==============================================================================
# Transposed KV Cache BufferedRef (SEQ_ALONG_LANE)
# ==============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class KVBufferedRefSeqAlongLane(_BypassRef):
  """Handles fetching/updating KV cache using SEQ_ALONG_LANE memory layout."""

  cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pytype: disable=signature-mismatch  # pyrefly: ignore[bad-override]
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: BufferType,
      buffer_count: int | tuple[int, int],
      use_lookahead: bool,
      cfgs: configs.MlaConfigs,
      **kwargs,
  ) -> "KVBufferedRefSeqAlongLane":
    assert buffer_type == BufferType.INPUT_OUTPUT
    if not SUPPORTS_IN_OUT_BUFFER_COUNT:
      raise NotImplementedError(
          "The MLA v3 kernel needs a JAX with separate (in, out) Pallas"
          f" buffer counts, which jax {jax.__version__} lacks."
      )

    standard_ref = _BypassRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
        **kwargs,
    )
    return _rebuild_with_cfgs(cls, standard_ref, cfgs)

  @jax.named_scope("kv_copy_in")
  def copy_in(
      self,
      src_ref: tuple[Any, Any, Any, schedule.MlaSchedule, Any],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # src_ref: (kv_cache_hbm, new_kv_c_hbm, new_k_pe_hbm, schedule_ref,
    #           kv_finite_ref)
    (
        kv_cache_hbm,
        new_kv_c_hbm,
        new_k_pe_hbm,
        schedule_ref,
        kv_finite_ref,
    ) = src_ref

    slot = self.current_copy_in_slot
    assert self.sem_recvs is not None
    sem: Any = self.sem_recvs.at[slot]
    block_idx = jnp.maximum(grid_indices[0], 0)

    assert self.window_ref is not None
    vmem_dst_lane: Any = self.window_ref.at[slot]
    num_lanes = self.cfgs.kv_token_align
    aligned_lkv_dim = self.cfgs.aligned_lkv_dim

    for b in range(self.cfgs.batch_size):
      # 1. Fetch cached paged tokens from 3D HBM cache
      with jax.named_scope("fetch_paged_kv_cache"):
        for i in range(self.cfgs.bkv_p):
          hbm_p_idx, sz = schedule_ref.get_dma_kv_cache(block_idx, b, i)
          dst_off = i * self.cfgs.serve.page_size

          @pl.when(sz > 0)
          def _start_paged_kv(hbm_p_idx=hbm_p_idx, dst_off=dst_off, sz=sz, b=b):
            sz = pl.multiple_of(sz, num_lanes)
            pltpu.make_async_copy(
                kv_cache_hbm.at[hbm_p_idx, :, pl.ds(0, sz)],
                vmem_dst_lane.at[b, :, pl.ds(dst_off, sz)],
                sem,
            ).start()

      # 2. Fetch unpaged new KV tokens from HBM.
      with jax.named_scope("fetch_new_kv"):
        fetch_hbm, fetch_vmem, fetch_sz = schedule_ref.get_dma_kv_new_fetch(
            block_idx, b
        )

        @pl.when(fetch_sz > 0)
        def _start_new_kv(
            fetch_hbm=fetch_hbm, fetch_vmem=fetch_vmem, sz=fetch_sz, b=b
        ):
          src_new_off = pl.multiple_of(fetch_hbm, num_lanes)
          dst_vmem_off = pl.multiple_of(fetch_vmem, num_lanes)
          sz = pl.multiple_of(sz, num_lanes)

          # new_kv_c_hbm -> vmem_dst_lane[:aligned_lkv_dim, :] (C_kv)
          with jax.named_scope("fetch_new_kv_c"):
            pltpu.make_async_copy(
                new_kv_c_hbm.at[:, pl.ds(src_new_off, sz)],
                vmem_dst_lane.at[b, :aligned_lkv_dim, pl.ds(dst_vmem_off, sz)],
                sem,
            ).start()

          # new_k_pe_hbm -> vmem_dst_lane[aligned_lkv_dim:, :] (K_pe).
          with jax.named_scope("fetch_new_k_pe"):
            pltpu.make_async_copy(
                new_k_pe_hbm.at[:, pl.ds(src_new_off, sz)],
                vmem_dst_lane.at[b, aligned_lkv_dim:, pl.ds(dst_vmem_off, sz)],
                sem,
            ).start()

    # 3. Zero what those DMAs leave uncovered, the first time each slot is
    # used. Every lane's DMAs are already in flight, so this overlaps them.
    self._zero_uncovered_on_first_use(
        schedule_ref, kv_finite_ref, slot, block_idx, vmem_dst_lane
    )

  def _zero_uncovered_on_first_use(
      self,
      schedule_ref: schedule.MlaSchedule,
      kv_finite_ref: Any,
      slot: jax.Array,
      block_idx: jax.Array,
      vmem_dst_lane: Any,
  ):
    """Zeroes the V rows a step's DMAs leave uncovered, once per slot.

    Attention computes over every column of `[0, hi)`. Masked columns get
    `p == 0` in P @ V, but `0 * NaN` is NaN and never-written VMEM may hold NaN
    fp8 bytes, so the V rows there must be finite. The K rows need nothing:
    masked scores are replaced with `mask_value` before the running max. The
    stitch only moves bits (u32 roll and select), so it cannot spread a NaN.

    A valid lane's DMAs cover a lane-tile-aligned prefix `[0, covered)`: cache
    page `i` lands at `i * page_size` with its token count rounded up to the
    tile, so the cached tiles end at `align_to(bkv_sz_cache, tile)`; the one
    new-KV fetch starts exactly there; and tile 0 is always one of them.
    Zeroing only `[covered, hi)` therefore never races an in-flight DMA.
    Afterwards every column of `[0, hi)` holds DMA'd, zeroed or stitched data,
    and later steps only overwrite a prefix with finite data, so each slot
    needs this once per kernel. `kv_finite_ref` records which ones are done.

    Padding lanes (`s_idx == -1`) are skipped. Nothing is DMA'd into or out of
    them, and they only occur after every real lane of the very last step of the
    schedule, so a slot whose first use contains a padding lane is never reused
    by a later step anyway.

    Args:
      schedule_ref: The SMEM schedule for the current chunk.
      kv_finite_ref: SMEM flags, one per slot (`(kv_num_slots,)`). Must be
        zeroed at kernel start.
      slot: The physical buffer slot this step's DMAs target.
      block_idx: The step being fetched.
      vmem_dst_lane: The window of `slot`, [batch_size, aligned_kv_dim, lanes].
    """
    cfgs = self.cfgs
    page_size = cfgs.serve.page_size
    tile = cfgs.kv_token_align
    # Decode only reads [0, bkv_sz); the prefill stitch rolls the whole lane,
    # so it needs the slack past bkv_sz as well.
    hi = cfgs.bkv_sz if cfgs.one_new_token else cfgs.kv_vmem_lanes
    if hi <= tile:
      return
    v_rows = cfgs.aligned_lkv_dim
    slot = jax.lax.convert_element_type(slot, jnp.int32)

    @pl.when(kv_finite_ref[slot] == 0)
    @jax.named_scope("zero_uncovered_kv")
    def _zero_slot():
      for b in range(cfgs.batch_size):

        @pl.when(schedule_ref.s_idx[block_idx, b] != -1)
        def _zero_lane(b=b):
          covered = jnp.int32(0)
          for i in range(cfgs.bkv_p):
            _, sz = schedule_ref.get_dma_kv_cache(block_idx, b, i)
            covered = jnp.maximum(
                covered, jnp.where(sz > 0, i * page_size + sz, 0)
            )
          _, fetch_vmem, fetch_sz = schedule_ref.get_dma_kv_new_fetch(
              block_idx, b
          )
          covered = jnp.maximum(
              covered,
              jnp.where(fetch_sz > 0, fetch_vmem + fetch_sz, 0),
          )

          # `covered` is a tile multiple, so the tiles at or past it are exactly
          # [covered, hi). A loop over just those tiles costs nothing when the
          # DMAs cover everything, where a static branch per tile would still
          # evaluate every branch. Tile 0 is always covered.
          @pl.loop(jnp.maximum(covered // tile, 1), hi // tile)
          def _zero_tile(t):
            start = pl.multiple_of(t * tile, tile)
            vmem_dst_lane[b, :v_rows, pl.ds(start, tile)] = jnp.zeros(
                (v_rows, tile), vmem_dst_lane.dtype
            )

      kv_finite_ref[slot] = 1

  def copy_out(
      self,
      dst_ref: tuple[jax.Ref, jax.Ref, jax.Ref, schedule.MlaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    with jax.named_scope("kv_copy_out"):
      # dst_ref: (kv_out_ref, _, _, schedule_ref, kv_finite_ref)
      kv_out_ref, _, _, schedule_ref, _ = dst_ref
      slot = self.current_copy_out_slot
      assert self.sem_sends is not None
      sem: Any = self.sem_sends.at[slot]
      block_idx = grid_indices[0]

      assert self.window_ref is not None
      vmem_src_lane: Any = self.window_ref.at[slot]
      num_lanes = self.cfgs.kv_token_align

      for b in range(self.cfgs.batch_size):

        @pl.when(schedule_ref.do_writeback[block_idx, b] == 1)
        def _wb_lane(b=b):
          for i in range(self.cfgs.bkv_p_wb):
            wb_hbm, wb_vmem, wb_sz = schedule_ref.get_dma_kv_new_wb(
                block_idx, b, i  # pyrefly: ignore[bad-argument-type]
            )

            # Write back concatenated into 3D kv_out_ref. Same adjacency
            # argument as `copy_in`: one copy covers both halves.
            @pl.when(wb_sz > 0)
            def _start_wb(wb_hbm=wb_hbm, wb_vmem=wb_vmem, sz=wb_sz, b=b):
              # wb_hbm packs (physical_page << page_size_log2) | offset_in_page.
              hbm_p_idx = wb_hbm >> self.cfgs.serve.page_size_log2
              dst_off = pl.multiple_of(
                  wb_hbm & self.cfgs.serve.page_size_mask, num_lanes
              )
              src_vmem_off = pl.multiple_of(wb_vmem, num_lanes)
              sz = pl.multiple_of(sz, num_lanes)
              pltpu.make_async_copy(
                  vmem_src_lane.at[b, :, pl.ds(src_vmem_off, sz)],
                  kv_out_ref.at[hbm_p_idx, :, pl.ds(dst_off, sz)],
                  sem,
              ).start()

  def wait_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, jax.Ref, schedule.MlaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    with jax.named_scope("wait_kv_copy_in"):
      _, _, _, schedule_ref, _ = src_ref
      slot = self.current_wait_in_slot
      assert self.sem_recvs is not None
      sem: Any = self.sem_recvs.at[slot]
      block_idx = grid_indices[0]

      kv_in_tokens = schedule_ref.total_wait_kv_in[block_idx]
      kv_in_tokens = pl.multiple_of(
          jnp.asarray(kv_in_tokens), self.cfgs.kv_token_align
      )

      assert self.window_ref is not None
      vmem_dst: Any = self.window_ref.at[slot]
      vmem_u32 = vmem_dst.bitcast(jnp.uint32)
      minor = self.cfgs.kv_vmem_lanes
      flat_dst = vmem_u32.reshape((-1, minor))
      sublanes = self.cfgs.aligned_kv_dim // self.cfgs.serve.packing_kv
      pltpu.make_async_copy(
          flat_dst.at[:sublanes, pl.ds(0, kv_in_tokens)],
          flat_dst.at[:sublanes, pl.ds(0, kv_in_tokens)],
          sem,
      ).wait()

  def wait_out(
      self,
      dst_ref: tuple[jax.Ref, jax.Ref, jax.Ref, schedule.MlaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    with jax.named_scope("wait_kv_copy_out"):
      _, _, _, schedule_ref, _ = dst_ref
      slot = self.current_wait_out_slot
      assert self.sem_sends is not None
      sem: Any = self.sem_sends.at[slot]
      block_idx = grid_indices[0]

      kv_out_tokens = schedule_ref.total_wait_kv_out[block_idx]
      kv_out_tokens = pl.multiple_of(
          jnp.asarray(kv_out_tokens), self.cfgs.kv_token_align
      )

      assert self.window_ref is not None
      vmem_src: Any = self.window_ref.at[slot]
      vmem_u32 = vmem_src.bitcast(jnp.uint32)
      minor = self.cfgs.kv_vmem_lanes
      flat_src = vmem_u32.reshape((-1, minor))
      sublanes = self.cfgs.aligned_kv_dim // self.cfgs.serve.packing_kv
      pltpu.make_async_copy(
          flat_src.at[:sublanes, pl.ds(0, kv_out_tokens)],
          flat_src.at[:sublanes, pl.ds(0, kv_out_tokens)],
          sem,
      ).wait()


# ==============================================================================
# Dedicated Query BufferedRef (BatchingQRef)
# ==============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingQRef(pltpu.BufferedRef):
  """Handles fetching Query block [Q_nope, Q_pe]."""

  cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pytype: disable=signature-mismatch  # pyrefly: ignore[bad-override]
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: configs.MlaConfigs,
      **kwargs,
  ) -> "BatchingQRef":
    assert buffer_type == BufferType.INPUT

    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
        **kwargs,
    )
    return _rebuild_with_cfgs(cls, standard_ref, cfgs)

  @jax.named_scope("q_copy_in")
  def copy_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # src_ref: (q_nope_hbm, q_pe_hbm, schedule_ref)
    q_nope_hbm, q_pe_hbm, schedule_ref = src_ref
    slot = self.current_copy_in_slot
    assert self.sem_recvs is not None
    sem: Any = self.sem_recvs.at[slot]
    assert self.window_ref is not None
    vmem_dst: Any = self.window_ref.at[slot]
    block_idx = grid_indices[0]

    aligned_lkv_dim = self.cfgs.aligned_lkv_dim

    for b in range(self.cfgs.batch_size):
      q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)  # pyrefly: ignore[bad-argument-type]

      # Copy Q_nope
      pltpu.make_async_copy(
          q_nope_hbm.at[pl.ds(q_src, q_sz), ...],
          vmem_dst.at[b, pl.ds(0, q_sz), :, :aligned_lkv_dim],
          sem,
      ).start()

      # Copy Q_pe
      pltpu.make_async_copy(
          q_pe_hbm.at[pl.ds(q_src, q_sz), ...],
          vmem_dst.at[b, pl.ds(0, q_sz), :, aligned_lkv_dim:],
          sem,
      ).start()

  def wait_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    _, _, schedule_ref = src_ref
    slot = self.current_wait_in_slot
    assert self.sem_recvs is not None
    sem: Any = self.sem_recvs.at[slot]
    block_idx = grid_indices[0]

    q_in_tokens = schedule_ref.total_wait_q_in[block_idx]

    itemsize = jnp.dtype(self.cfgs.serve.dtype_q).itemsize
    minor = self.cfgs.aligned_q_dim
    rows_per_token = (self.cfgs.aligned_num_q_heads * itemsize) // 4
    assert (
        rows_per_token % 8 == 0
    ), f"{rows_per_token=} must be sublane-aligned for the wait slice"
    wait_lanes = q_in_tokens * rows_per_token

    assert self.window_ref is not None
    vmem_dst: Any = self.window_ref.at[slot]
    vmem_u32 = vmem_dst.bitcast(jnp.uint32)
    flat_vmem = vmem_u32.reshape((-1, minor))
    pltpu.make_async_copy(
        flat_vmem.at[pl.ds(0, wait_lanes), :],
        flat_vmem.at[pl.ds(0, wait_lanes), :],
        sem,
    ).wait()


# ==============================================================================
# Output Activation BufferedRef (BatchingORef)
# ==============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingORef(pltpu.BufferedRef):
  """Handles storing final attention output activations (width d_nope = 512)."""

  cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pytype: disable=signature-mismatch  # pyrefly: ignore[bad-override]
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: configs.MlaConfigs,
      **kwargs,
  ) -> "BatchingORef":
    assert buffer_type == BufferType.OUTPUT

    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
        **kwargs,
    )
    return _rebuild_with_cfgs(cls, standard_ref, cfgs)

  def copy_out(
      self,
      dst_ref: tuple[jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # dst_ref: (o_hbm, schedule_ref)
    o_hbm, schedule_ref = dst_ref
    slot = self.current_copy_out_slot
    assert self.sem_sends is not None
    sem: Any = self.sem_sends.at[slot]
    assert self.window_ref is not None
    vmem_src: Any = self.window_ref.at[slot]
    block_idx = grid_indices[0]

    for b in range(self.cfgs.batch_size):
      is_last_k = schedule_ref.is_last_k[block_idx, b] == 1
      q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)  # pyrefly: ignore[bad-argument-type]
      q_sz = jnp.where(is_last_k, q_sz, 0)

      def _start_o(q_src=q_src, q_sz=q_sz, b=b):
        pltpu.make_async_copy(
            vmem_src.at[b, pl.ds(0, q_sz), ...],
            o_hbm.at[pl.ds(q_src, q_sz), ...],
            sem,
        ).start()

      # Output is emitted only at a sequence's last k-block; the rest issue a
      # zero-length copy. `wait_out` counts bytes, so this is invisible to it.
      _start_o()

  def wait_out(
      self,
      dst_ref: tuple[jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # dst_ref: (o_hbm, schedule_ref)
    _, schedule_ref = dst_ref
    slot = self.current_wait_out_slot
    assert self.sem_sends is not None
    sem: Any = self.sem_sends.at[slot]
    block_idx = grid_indices[0]
    minor = self.cfgs.aligned_lkv_dim
    itemsize = jnp.dtype(self.cfgs.serve.dtype_out).itemsize
    rows_per_token = (self.cfgs.aligned_num_q_heads * itemsize) // 4
    assert rows_per_token % 8 == 0
    o_tokens = schedule_ref.total_wait_o_out[block_idx]
    wait_lanes = o_tokens * rows_per_token
    assert self.window_ref is not None
    vmem_src: Any = self.window_ref.at[slot]
    vmem_u32 = vmem_src.bitcast(jnp.uint32)
    flat_src = vmem_u32.reshape((-1, minor))
    pltpu.make_async_copy(
        flat_src.at[pl.ds(0, wait_lanes), :],
        flat_src.at[pl.ds(0, wait_lanes), :],
        sem,
    ).wait()
