# Copyright 2026 Google LLC
# Copyright 2026 Rabdos AI
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
"""Prefill pipelines, padding and output transfers for fused GDN."""

from typing import Any, Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import memory_ref
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import compute as fused_compute
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import memory as fused_memory
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import metadata as fused_metadata


def run_prefill_pipeline(
    body: Any,
    qkv_ref: Any,
    b_ref: Any,
    a_ref: Any,
    metadata_ref: memory_ref.MetadataRef,
    *,
    chunk_size: int,
    before_first_tile: Callable[[], None] | None = None,
) -> None:
  """Stream a validated nonempty schedule through two activation slots.

  Each tile prefetches its successor. body receives the tile index and three
  eight-row-aligned VMEM windows; finish reads before returning and
  initialize consumed padding. Reload shared short-request windows into the
  next slot so padding writes remain private. before_first_tile runs once
  after starting the first input transfer, allowing independent work to
  overlap that load without exposing incomplete inputs.

  Args:
    body: Tile callback receiving its index and private QKV, beta, and decay
      windows.
    qkv_ref: HBM packed activations [tokens, 2 * n_kq * d_k + n_v * d_v].
    b_ref: HBM beta-gate inputs [tokens, n_v], with the activation dtype.
    a_ref: HBM decay-gate inputs [tokens, n_v], with the activation dtype.
    metadata_ref: Packed tile/request metadata refs holding ownership, flags,
      and cache slots.
    chunk_size: Number of token rows in one prefill tile.
    before_first_tile: Optional callback after the first input DMA starts,
      before its wait.
  """

  def run(qkv_tiles: Any, b_tiles: Any, a_tiles: Any, sem: Any) -> None:
    """Prime the first transfer, run the initialization hook, and iterate over
    tiles.

    Args:
      qkv_tiles: Two private QKV windows used to overlap prefill loads and
        compute.
      b_tiles: Two private beta-input windows used to overlap prefill loads and
        compute.
      a_tiles: Two private decay-input windows used to overlap prefill loads and
        compute.
      sem: DMA semaphore array for the private input-window transfers.
    """

    def transfer(p_id: jax.Array, *, wait: bool) -> None:
      """Start or wait for a tile's aligned QKV/gate window.

      Args:
        p_id: Zero-based packed prefill tile index.
        wait: True waits for the matching transfer; False starts it.
      """
      base = metadata_ref.get_record(p_id, 0).r_base
      size = metadata_ref.get_record(p_id, 0).r_size
      slot = p_id % 2
      start = pl.multiple_of((base // 8) * 8, 8)
      rows = pl.multiple_of(((base - start + size + 7) // 8) * 8, 8)
      fused_memory.dma(
          qkv_ref.at[pl.ds(start, rows), :],
          qkv_tiles.at[slot, pl.ds(0, rows), :],
          sem.at[slot, 0],
          wait=wait,
      )
      for operand, (source, destination) in enumerate(
          ((b_ref, b_tiles), (a_ref, a_tiles)), start=1
      ):
        fused_memory.dma(
            source.at[pl.ds(start, rows), :],
            destination.at[slot, pl.ds(0, rows), :],
            sem.at[slot, operand],
            wait=wait,
        )

    transfer(jnp.int32(0), wait=False)
    if before_first_tile is not None:
      before_first_tile()

    def step(p_id: jax.Array, unused: None) -> None:
      """Prefetch the next tile, wait for the current inputs, then call the tile
      body.

      Args:
        p_id: Zero-based packed prefill tile index.
        unused: Unused loop carry, always None.
      """

      @pl.when(p_id + 1 < metadata_ref.num_tiles[...])
      def prefetch() -> None:
        # The opposite slot is free; prefetch before waiting for current input.
        """Start the next input window when another tile exists."""
        transfer(p_id + 1, wait=False)

      transfer(p_id, wait=True)
      slot = p_id % 2
      body(p_id, qkv_tiles.at[slot], b_tiles.at[slot], a_tiles.at[slot])
      return unused

    jax.lax.fori_loop(0, metadata_ref.num_tiles[...], step, None)

  pl.run_scoped(
      run,
      pltpu.VMEM((2, chunk_size + 8, *qkv_ref.shape[1:]), qkv_ref.dtype),
      pltpu.VMEM((2, chunk_size + 8, *b_ref.shape[1:]), b_ref.dtype),
      pltpu.VMEM((2, chunk_size + 8, *a_ref.shape[1:]), a_ref.dtype),
      pltpu.SemaphoreType.DMA((2, 3)),
  )


def _clear_prefill_padding(
    qkv_slot_ref: Any,
    b_slot_ref: Any,
    a_slot_ref: Any,
    real_size: jax.Array,
    *,
    chunk_size: int,
    input_offset: jax.Array,
) -> None:
  """Zero inactive QKV/gate rows after input DMA completion.

  QKV windows are [chunk_size + 8, width]; gates are [chunk_size + 8, n_v].
  input_offset is in [0, 8), real_size in [1, chunk_size], and chunk_size
  must be a multiple of 64.

  Args:
    qkv_slot_ref: Private packed-QKV window for one prefill tile, including
      alignment rows.
    b_slot_ref: Private beta-input window for one prefill tile, including
      alignment rows.
    a_slot_ref: Private decay-input window for one prefill tile, including
      alignment rows.
    real_size: Valid token count in the current prefill tile.
    chunk_size: Number of token rows in one prefill tile.
    input_offset: Logical token offset within the private eight-row-aligned
      input window.
  """

  @pl.when(real_size < chunk_size)
  def clear_tail() -> None:
    # Mask the boundary and clear aligned tiles; avoid scalar VMEM row stores.
    """Clear padding from the valid tail onward, preserving earlier rows."""
    tail = real_size + input_offset
    row_base = pl.multiple_of((tail // 8) * 8, 8)
    for ref in (qkv_slot_ref, b_slot_ref, a_slot_ref):
      ref[pl.ds(row_base, 8), :] = jnp.where(
          jnp.arange(8)[:, None] < tail - row_base,
          ref[pl.ds(row_base, 8), :],
          jnp.zeros((8, ref.shape[1]), ref.dtype),
      )

    def clear_block(block: jax.Array, unused: None) -> None:
      """Clear a complete eight-row padding block.

      Args:
        block: Zero-based output or padding block index.
        unused: Unused loop carry, always None.
      """
      start = pl.multiple_of(block * 8, 8)
      for ref in (qkv_slot_ref, b_slot_ref, a_slot_ref):
        ref[pl.ds(start, 8), :] = jnp.zeros((8, ref.shape[1]), ref.dtype)
      return unused

    jax.lax.fori_loop(
        tail // 8 + 1, qkv_slot_ref.shape[0] // 8, clear_block, None
    )


def _prefill_inner(
    qkv_slot_ref: Any,
    b_slot_ref: Any,
    a_slot_ref: Any,
    conv_state_slot_ref: Any,
    recurrent_slot_ref: Any,
    out_slot_ref: Any,
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: Any,
    dense_conv_ref: tuple[Any, Any | None],
    carry_conv_scratch_ref: Any,
    carry_recurrent_scratch_ref: Any,
    qkv_head_major_scratch_ref: Any,
    grouped_state_scratch_ref: Any,
    output_carry_ref: Any,
    *,
    cfg: config.GDNConfig,
    p_id: jax.Array,
    output_offset: jax.Array,
    flush_rows: jax.Array,
) -> None:
  """Compute one prefill tile after input DMA and update its resident state.

  cfg uses one sequence and whole 64-row blocks; metadata/carries follow tile
  order. Output slot must be free. Conv carry stays FP32; BF16 recurrent
  cache needs separate FP32 carry, while an FP32 slot serves directly.
  Grouped state scratch is [n_kq, 128, v_per_kq * 128] FP32; output carry is
  [16, output_width] in the activation dtype.

  Args:
    qkv_slot_ref: Private packed-QKV window for one prefill tile, including
      alignment rows.
    b_slot_ref: Private beta-input window for one prefill tile, including
      alignment rows.
    a_slot_ref: Private decay-input window for one prefill tile, including
      alignment rows.
    conv_state_slot_ref: Mutable per-request convolution history prepared for
      cache writeback.
    recurrent_slot_ref: Mutable recurrent-state slice for the current
      request/member.
    out_slot_ref: Mutable VMEM output tile [chunk_size + 16, n_v * d_v].
    metadata_ref: Packed tile/request metadata refs holding ownership, flags,
      and cache slots.
    weights_ref: VMEM refs for log decay weights and timestep biases.
    dense_conv_ref: FP32 convolution tap and optional bias refs in head-major
      layout.
    carry_conv_scratch_ref: Mutable FP32 convolution history carried between
      prefill tiles.
    carry_recurrent_scratch_ref: Optional FP32 state carry used when the
      recurrent cache is BF16.
    qkv_head_major_scratch_ref: Shared FP32 head-major scratch; decode may
      borrow it before prefill.
    grouped_state_scratch_ref: Mutable FP32 scratch [n_kq, 128, (n_v // n_kq) *
      128].
    output_carry_ref: Mutable [16, n_v * d_v] partial output block in activation
      dtype.
    cfg: Static GDN configuration defining head layout, tile sizes, and dtypes.
    p_id: Zero-based packed prefill tile index.
    output_offset: Decode row position or prefill starting offset within the
      16-row carry.
    flush_rows: Number of output rows forming complete 16-row blocks.
  """
  # The caller initializes history once per request; later tiles use the carry.
  real_sizes = metadata_ref.get_record(p_id, 0).r_size.reshape((1,))

  _clear_prefill_padding(
      qkv_slot_ref,
      b_slot_ref,
      a_slot_ref,
      real_sizes[0],
      chunk_size=cfg.chunk_size,
      input_offset=output_offset % 8,
  )
  fused_compute.dense_conv_silu(
      qkv_slot_ref,
      real_sizes,
      dense_conv_ref,
      qkv_head_major_scratch_ref,
      conv_state_slot_ref,
      carry_conv_scratch_ref,
      cfg=cfg,
      input_offset=output_offset % 8,
  )

  q_end = cfg.num_kq_heads
  k_end = 2 * cfg.num_kq_heads
  q_large = qkv_head_major_scratch_ref[:q_end, : cfg.chunk_size, :][None, ...]
  k_large = qkv_head_major_scratch_ref[q_end:k_end, : cfg.chunk_size, :][
      None, ...
  ]
  v_large = qkv_head_major_scratch_ref[
      k_end : k_end + cfg.num_v_heads, : cfg.chunk_size, :
  ][None, ...]

  padding = cfg.aligned_num_v_heads - cfg.num_v_heads
  gate_pad = ((0, 0), (0, 0), (0, padding))
  b_values = fused_compute.native_gate_values(
      b_slot_ref, output_offset % 8, cfg.chunk_size
  )[None, ...]
  a_values = fused_compute.native_gate_values(
      a_slot_ref, output_offset % 8, cfg.chunk_size
  )[None, ...]
  b_large = jnp.pad(b_values, gate_pad)[:, None, :, :]
  a_large = jnp.pad(a_values, gate_pad)[:, None, :, :]
  a_log = jnp.pad(weights_ref.a_log[...], (0, padding))
  dt_bias = jnp.pad(weights_ref.dt_bias[...], (0, padding))

  # FP32 carry avoids rounding a BF16 cache at every tile boundary.
  state_prev = (
      recurrent_slot_ref[...]
      if carry_recurrent_scratch_ref is None
      else carry_recurrent_scratch_ref[...]
  )
  outputs, new_recurrent = fused_compute.chunked_gdn(
      real_sizes,
      q_large,
      k_large,
      v_large,
      b_large,
      a_large,
      state_prev,
      a_log,
      dt_bias,
      cfg,
      grouped_state_scratch_ref,
  )
  fused_compute.store_prefill_output(
      outputs[0], out_slot_ref, output_carry_ref, output_offset, flush_rows
  )
  if carry_recurrent_scratch_ref is not None:
    carry_recurrent_scratch_ref[...] = new_recurrent.astype(jnp.float32)
  recurrent_slot_ref[...] = new_recurrent.astype(recurrent_slot_ref.dtype)


def _output_transfer(
    metadata_ref: memory_ref.MetadataRef,
    out_ref: Any,
    output_scratch_ref: Any,
    output_sem_ref: Any,
    p_id: jax.Array,
    *,
    wait: bool,
) -> None:
  """Transfer complete aligned blocks from a valid, assembled output tile.

  Scratch is [2, chunk_size + 16, output_width] in the activation dtype; sem
  is [2]. Match each wait to the same tile before parity-slot reuse.

  Args:
    metadata_ref: Packed tile/request metadata refs holding ownership, flags,
      and cache slots.
    out_ref: HBM output array [tokens, n_v * d_v], written by the pipeline.
    output_scratch_ref: Two output-tile slots retained until their DMA stores
      complete.
    output_sem_ref: DMA semaphores for the output scratch slots.
    p_id: Zero-based packed prefill tile index.
    wait: True waits for the matching transfer; False starts it.
  """
  max_blocks = min(
      (output_scratch_ref.shape[1] - fused_compute.OUTPUT_ALIGNMENT)
      // fused_compute.OUTPUT_ALIGNMENT,
      out_ref.shape[0] // fused_compute.OUTPUT_ALIGNMENT,
  )
  if max_blocks == 0:
    return

  base = metadata_ref.get_record(p_id, 0).r_base
  size = metadata_ref.get_record(p_id, 0).r_size
  lo = pl.multiple_of(
      (base // fused_compute.OUTPUT_ALIGNMENT) * fused_compute.OUTPUT_ALIGNMENT,
      fused_compute.OUTPUT_ALIGNMENT,
  )
  blocks = ((base - lo) + size) // fused_compute.OUTPUT_ALIGNMENT
  slot = p_id % 2

  @pl.when(blocks > 0)
  def transfer() -> None:
    # Transfer complete 16-row blocks; dynamic extents avoid a branch per size.
    """Start or wait for the assembled output block transfer."""
    rows = pl.multiple_of(
        blocks * fused_compute.OUTPUT_ALIGNMENT,
        fused_compute.OUTPUT_ALIGNMENT,
    )
    fused_memory.dma(
        output_scratch_ref.at[slot, pl.ds(0, rows), :],
        out_ref.at[pl.ds(lo, rows), :],
        output_sem_ref.at[slot],
        wait=wait,
    )


def prefill_pipeline_inner(
    p_id: jax.Array,
    qkv_slot_ref: Any,
    b_slot_ref: Any,
    a_slot_ref: Any,
    recurrent_state_ref: Any,
    recurrent_state_out_ref: Any,
    recurrent_scratch_ref: Any,
    recurrent_load_sem_ref: Any,
    recurrent_store_sem_ref: Any,
    conv_state_ref: Any,
    conv_dma_scratch_ref: Any,
    conv_state_slot_ref: Any,
    conv_dma_sem_ref: Any,
    conv_store_scratch_ref: Any,
    conv_store_sem_ref: Any,
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: Any,
    dense_conv_ref: tuple[Any, Any | None],
    carry_conv_scratch_ref: Any,
    carry_recurrent_scratch_ref: Any,
    qkv_head_major_scratch_ref: Any,
    grouped_state_scratch_ref: Any,
    out_ref: Any,
    output_scratch_ref: Any,
    output_carry_ref: Any,
    output_sem_ref: Any,
    distribution_ref: Any,
    read_state_indices_ref: Any,
    read_offsets_ref: Any,
    *,
    cfg: config.GDNConfig,
) -> None:
  """Pipeline one scheduled tile's computation, histories and output transfers.

  Before entry, activation loads must be complete and any required
  first-request state loads started. Run tiles in request order. State
  scratch is [2, n_v, 128, 128] in the cache dtype, retained across each
  owner's tiles; conv load/store scratch is [2, max(1, kernel_size - 1),
  width]. Each load/store semaphore array is [2]. Drain the corresponding
  transfer before reusing an owner's parity slot. _prefill_inner defines
  carry/scratch use.

  Args:
    p_id: Zero-based packed prefill tile index.
    qkv_slot_ref: Private packed-QKV window for one prefill tile, including
      alignment rows.
    b_slot_ref: Private beta-input window for one prefill tile, including
      alignment rows.
    a_slot_ref: Private decay-input window for one prefill tile, including
      alignment rows.
    recurrent_state_ref: HBM recurrent cache [slots, n_v, d_k, d_v].
    recurrent_state_out_ref: HBM recurrent-cache destination alias, written only
      at owned slots.
    recurrent_scratch_ref: Two prefill recurrent-state parity slots in the cache
      dtype.
    recurrent_load_sem_ref: DMA semaphores for the two prefill recurrent-load
      slots.
    recurrent_store_sem_ref: DMA semaphores for the two prefill recurrent-store
      slots.
    conv_state_ref: HBM convolution cache; reads use prefix slots and writes use
      owned slots.
    conv_dma_scratch_ref: Two parity slots for loading convolution history in
      its cache dtype.
    conv_state_slot_ref: Mutable per-request convolution history prepared for
      cache writeback.
    conv_dma_sem_ref: DMA semaphores for the two convolution-load slots.
    conv_store_scratch_ref: Two parity slots retaining convolution history until
      stores complete.
    conv_store_sem_ref: DMA semaphores for the two convolution-store slots.
    metadata_ref: Packed tile/request metadata refs holding ownership, flags,
      and cache slots.
    weights_ref: VMEM refs for log decay weights and timestep biases.
    dense_conv_ref: FP32 convolution tap and optional bias refs in head-major
      layout.
    carry_conv_scratch_ref: Mutable FP32 convolution history carried between
      prefill tiles.
    carry_recurrent_scratch_ref: Optional FP32 state carry used when the
      recurrent cache is BF16.
    qkv_head_major_scratch_ref: Shared FP32 head-major scratch; decode may
      borrow it before prefill.
    grouped_state_scratch_ref: Mutable FP32 scratch [n_kq, 128, (n_v // n_kq) *
      128].
    out_ref: HBM output array [tokens, n_v * d_v], written by the pipeline.
    output_scratch_ref: Two output-tile slots retained until their DMA stores
      complete.
    output_carry_ref: Mutable [16, n_v * d_v] partial output block in activation
      dtype.
    output_sem_ref: DMA semaphores for the output scratch slots.
    distribution_ref: SMEM endpoints for decode, prefill, and active request
      prefixes.
    read_state_indices_ref: Per-request initial-state cache slots, separate from
      write ownership.
    read_offsets_ref: Per-request int32 read-slot offsets; public dispatch zeros
      prefill offsets.
    cfg: Static GDN configuration defining head layout, tile sizes, and dtypes.
  """
  owner = metadata_ref.get_record(p_id, 0).s_idx
  recurrent_slot_ref = recurrent_scratch_ref.at[pl.ds(owner % 2, 1)]

  def recurrent_transfer(
      request: jax.Array, *, to_hbm: bool, wait: bool
  ) -> None:
    """Transfer a request's recurrent state through its request-parity slot.

    Args:
      request: Request index whose initial or final state is transferred.
      to_hbm: True stores VMEM state to HBM; False loads HBM state into VMEM.
      wait: True waits for the matching transfer; False starts it.
    """
    fused_memory.recurrent_state_transfer(
        metadata_ref,
        read_state_indices_ref,
        read_offsets_ref,
        recurrent_state_out_ref if to_hbm else recurrent_state_ref,
        recurrent_scratch_ref,
        recurrent_load_sem_ref,
        recurrent_store_sem_ref,
        request,
        to_hbm=to_hbm,
        wait=wait,
    )

  def load_states(request: jax.Array, *, wait: bool) -> None:
    """Start or wait for history loads of a resumed request.

    Args:
      request: Request index whose initial or final state is transferred.
      wait: True waits for the matching transfer; False starts it.
    """

    @pl.when(
        fused_metadata.metadata_storage(metadata_ref.s_idx_has_initial_state)[
            request
        ]
    )
    def transfer() -> None:
      """Start or wait for both previous-state cache loads."""
      recurrent_transfer(request, to_hbm=False, wait=wait)
      fused_memory.conv_state_transfer(
          metadata_ref,
          read_state_indices_ref,
          read_offsets_ref,
          conv_state_ref,
          conv_dma_scratch_ref,
          conv_dma_sem_ref,
          request,
          to_hbm=False,
          wait=wait,
      )

  @pl.when(metadata_ref.get_record(p_id, 0).is_first_tile)
  def initialize_states() -> None:
    """Initialize the first tile of a request from loaded history or zero."""
    load_states(owner, wait=True)
    has_initial_state = fused_metadata.metadata_storage(
        metadata_ref.s_idx_has_initial_state
    )[owner]

    @pl.when(has_initial_state)
    def initialize_conv() -> None:
      """Copy convolution and any BF16 recurrent history into FP32 carries."""
      if cfg.kernel_size > 1:
        carry_conv_scratch_ref[...] = (
            conv_dma_scratch_ref[owner % 2, ...]
            .reshape(carry_conv_scratch_ref.shape)
            .astype(jnp.float32)
        )
      if carry_recurrent_scratch_ref is not None:
        carry_recurrent_scratch_ref[...] = recurrent_slot_ref[...].astype(
            jnp.float32
        )

    @pl.when(~has_initial_state)
    def zero_states() -> None:
      """Zero convolution and recurrent carries for a fresh request."""
      if cfg.kernel_size > 1:
        carry_conv_scratch_ref[...] = jnp.zeros(
            carry_conv_scratch_ref.shape, carry_conv_scratch_ref.dtype
        )
      recurrent_slot_ref[...] = jnp.zeros(
          recurrent_slot_ref.shape, recurrent_slot_ref.dtype
      )
      if carry_recurrent_scratch_ref is not None:
        carry_recurrent_scratch_ref[...] = jnp.zeros(
            carry_recurrent_scratch_ref.shape,
            carry_recurrent_scratch_ref.dtype,
        )

  @pl.when(p_id + 1 < metadata_ref.num_tiles[...])
  def prefetch_next_request() -> None:
    """Prefetch state when the next tile starts a new request."""
    next_p_id = p_id + 1

    @pl.when(metadata_ref.get_record(next_p_id, 0).is_first_tile)
    def start_next_load() -> None:
      """Release the next request's state slot, then start its history loads."""
      next_owner = metadata_ref.get_record(next_p_id, 0).s_idx

      # Drain the old owner before buffer reuse, even if the next owner is fresh.
      @pl.when(next_owner >= distribution_ref[0] + 2)
      def release_state_buffer() -> None:
        """Wait for an older recurrent store before that parity slot is reused."""
        recurrent_transfer(next_owner - 2, to_hbm=True, wait=True)

      load_states(next_owner, wait=False)

  base = metadata_ref.get_record(p_id, 0).r_base
  size = metadata_ref.get_record(p_id, 0).r_size
  output_offset = base % fused_compute.OUTPUT_ALIGNMENT
  flush_rows = (
      (output_offset + size) // fused_compute.OUTPUT_ALIGNMENT
  ) * fused_compute.OUTPUT_ALIGNMENT

  @pl.when(p_id >= 2)
  def wait_previous_output() -> None:
    """Drain the output store before reusing its scratch slot."""
    _output_transfer(
        metadata_ref,
        out_ref,
        output_scratch_ref,
        output_sem_ref,
        p_id - 2,
        wait=True,
    )

  out_slot_ref = output_scratch_ref.at[p_id % 2]
  _prefill_inner(
      qkv_slot_ref,
      b_slot_ref,
      a_slot_ref,
      conv_state_slot_ref,
      recurrent_slot_ref,
      out_slot_ref,
      metadata_ref,
      weights_ref,
      dense_conv_ref,
      carry_conv_scratch_ref,
      carry_recurrent_scratch_ref,
      qkv_head_major_scratch_ref,
      grouped_state_scratch_ref,
      output_carry_ref,
      cfg=cfg,
      p_id=p_id,
      output_offset=output_offset,
      flush_rows=flush_rows,
  )

  @pl.when(metadata_ref.get_record(p_id, 0).is_last_tile)
  def store_states() -> None:
    """Start final recurrent/convolution cache stores when a request finishes."""
    recurrent_transfer(owner, to_hbm=True, wait=False)

    @pl.when(owner >= distribution_ref[0] + 2)
    def wait_store_buffer_reuse() -> None:
      """Drain the older convolution store before replacing its history buffer."""
      fused_memory.conv_state_transfer(
          metadata_ref,
          read_state_indices_ref,
          read_offsets_ref,
          conv_state_ref,
          conv_store_scratch_ref,
          conv_store_sem_ref,
          owner - 2,
          to_hbm=True,
          wait=True,
      )

    if cfg.kernel_size > 1:
      conv_store_scratch_ref[owner % 2, ...] = conv_state_slot_ref[...].reshape(
          conv_store_scratch_ref.shape[1:]
      )
    fused_memory.conv_state_transfer(
        metadata_ref,
        read_state_indices_ref,
        read_offsets_ref,
        conv_state_ref,
        conv_store_scratch_ref,
        conv_store_sem_ref,
        owner,
        to_hbm=True,
        wait=False,
    )

  _output_transfer(
      metadata_ref,
      out_ref,
      output_scratch_ref,
      output_sem_ref,
      p_id,
      wait=False,
  )

  @pl.when(p_id == metadata_ref.num_tiles[...] - 1)
  def drain_output() -> None:
    """Drain output DMA when the final prefill tile has been processed."""
    _output_transfer(
        metadata_ref,
        out_ref,
        output_scratch_ref,
        output_sem_ref,
        p_id,
        wait=True,
    )

    @pl.when(p_id > 0)
    def drain_other_slot() -> None:
      """Wait for the other output slot when it still contains an outstanding
      store.
      """
      _output_transfer(
          metadata_ref,
          out_ref,
          output_scratch_ref,
          output_sem_ref,
          p_id - 1,
          wait=True,
      )


def finish_bucket_output(
    query_start_ref: Any,
    state_indices_ref: Any,
    distribution_ref: Any,
    out_ref: Any,
    output_refs: fused_memory.OutputRefs,
    *,
    schedule_valid: jax.Array,
) -> None:
  """Flush the partial carry and zero inactive rows after output stores drain.

  For a valid schedule the carry holds the final partial block; an invalid
  schedule has no active output. output_refs follows OutputRefs.

  Args:
    query_start_ref: SMEM int32 request token boundaries, with requests + 1
      entries.
    state_indices_ref: SMEM per-request write-cache slots; active owners must be
      distinct.
    distribution_ref: SMEM endpoints for decode, prefill, and active request
      prefixes.
    out_ref: HBM output array [tokens, n_v * d_v], written by the pipeline.
    output_refs: Output scratch, carry, semaphores, and optional final-tail ref.
    schedule_valid: Whether metadata validation accepted the active request
      prefix.
  """
  # Use the last active offset; padded offsets may truncate valid output.
  active_end = jnp.where(
      schedule_valid,
      query_start_ref[
          fused_metadata.active_request_count(
              distribution_ref, state_indices_ref.shape[0]
          )
      ],
      jnp.int32(0),
  )
  active_block = active_end // fused_compute.OUTPUT_ALIGNMENT
  active_rows = active_end % fused_compute.OUTPUT_ALIGNMENT
  row = jax.lax.broadcasted_iota(jnp.int32, output_refs.carry.shape, 0)
  output_refs.carry[...] = jnp.where(
      row < active_rows, output_refs.carry[...], 0
  )

  def finish_block(block: jax.Array, unused: None) -> None:
    """Build one final output block from carry plus zeros and store it
    synchronously.

    Args:
      block: Zero-based output or padding block index.
      unused: Unused loop carry, always None.
    """
    output_refs.scratch[0, : fused_compute.OUTPUT_ALIGNMENT, :] = jnp.where(
        block == active_block, output_refs.carry[...], 0
    )
    lo = pl.multiple_of(
        block * fused_compute.OUTPUT_ALIGNMENT, fused_compute.OUTPUT_ALIGNMENT
    )
    source = output_refs.scratch.at[
        0, pl.ds(0, fused_compute.OUTPUT_ALIGNMENT), :
    ]
    destination = out_ref.at[pl.ds(lo, fused_compute.OUTPUT_ALIGNMENT), :]
    fused_memory.dma_store_and_wait(source, destination, output_refs.sem.at[0])
    return unused

  if out_ref.shape[0] >= fused_compute.OUTPUT_ALIGNMENT:
    jax.lax.fori_loop(
        active_block,
        out_ref.shape[0] // fused_compute.OUTPUT_ALIGNMENT,
        finish_block,
        None,
    )

  if output_refs.tail is not None:
    output_refs.carry[...] = jnp.where(
        active_block == out_ref.shape[0] // fused_compute.OUTPUT_ALIGNMENT,
        output_refs.carry[...],
        0,
    )
    fused_memory.dma_store_and_wait(
        output_refs.carry, output_refs.tail, output_refs.sem.at[0]
    )
