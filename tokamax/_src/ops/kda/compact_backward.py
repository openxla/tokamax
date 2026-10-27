# Copyright 2026 Ant Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Ported from primatrix/pallas-kernel dca6a09feec2659f2777a6db02f2ee55560f991e.

import functools
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int

from tokamax._src.ops.kda.compact_hardware import get_tpu_config
from tokamax._src.ops.kda.compact_cost import (
    kda_bwd_mega_cost,
    kda_compact_output_cost,
    resolve_cost_estimate,
)
from tokamax._src.ops.kda.cp_utils import ContextParallelMetadata as CPContext
from tokamax._src.ops.kda.compact_backward_math import (
    _chunk_segment_metadata,
    _chunk_window_has_valid,
    _matmul_in_dtype,
    compute_intra_backward_sequential_to_chunk_refs,
    compute_m1_recompute,
    compute_reverse_cumsum_dg_scan,
    compute_wy_backward,
)
from tokamax._src.ops.kda.compact_segments import (
    same_segment_causal_mask,
    segmented_reverse_sum,
    segmented_suffix_last,
)
from tokamax._src.ops.kda.utils import get_interpret
from tokamax._src.ops.kda.compact_utils import align_up, compute_padded_cu_seqlens, prepare_chunk_indices

# Mega-kernel tuning knobs. Each benchmark process reads MB/CB before JAX
# tracing, so those variants can be swept without source edits. The backward
# pipeline uses independent two-way input and output ping-pong buffers; output
# storage never aliases an input tile.
KDA_DMA_CHUNK_BATCH = 6
KDA_DMA_INPUT_BUFFER_COUNT = 2
KDA_DMA_OUTPUT_BUFFER_COUNT = 2
KDA_DMA_MINI_BATCH_ENV = None
KDA_DMA_MINI_BATCH = (
    int(KDA_DMA_MINI_BATCH_ENV)
    if KDA_DMA_MINI_BATCH_ENV is not None
    else KDA_DMA_MINI_BATCH_ENV
)

# Keep the backward mega-kernel within a portable 64 MiB ceiling.  The
# hardware-specific compiler budget may be smaller (TPU v7 reserves 64 KiB),
# so every Pallas call below must also intersect this ceiling with the detected
# compiler limit.
KDA_BWD_SCOPED_VMEM_LIMIT_BYTES = 64 * 1024 * 1024
# Keep custom collective IDs stable and separate from JAX's automatic range.
KDA_BWD_CP_COLLECTIVE_ID = 101


def _cp_streaming_merge(
    dS_ref,
    dM_ref,
    merged_ref,
    buffers_ref,
    post_num,
    max_post_num,
    is_last,
    *,
    cp_size,
    axis_name,
):
  """Merge FP32 rank summaries with two VMEM packets and bounded semaphores.

  dS/merged are [MB,K,V], dM is [MB,K,K], and buffers is [2,MB,K,V+K].
  All ranks must pass the same max_post_num and execute this scope in the
  same batch/head-group order. Original packets are forwarded unchanged;
  only merged_ref consumes the associated downstream summaries.
  """
  V = dS_ref.shape[-1]
  merged_ref[:] = jnp.zeros(merged_ref.shape, dtype=jnp.float32)
  if cp_size == 1:
    return

  rank = jax.lax.axis_index(axis_name)
  left = (rank + cp_size - 1) % cp_size
  right = (rank + 1) % cp_size

  @pl.when(max_post_num > 0)
  def _communicate():
    buffers_ref[0, :, :, :V] = dS_ref[:]
    buffers_ref[0, :, :, V:] = dM_ref[:]

    @functools.partial(
        pl.run_scoped,
        send_sem=pltpu.SemaphoreType.DMA,
        recv_sem=pltpu.SemaphoreType.DMA,
        ready_sem=pltpu.SemaphoreType.REGULAR,
        startup_sem=pltpu.SemaphoreType.REGULAR,
    )
    def _ring(send_sem, recv_sem, ready_sem, startup_sem):
      # The second phase prevents an early next scope from satisfying a
      # neighbor's current startup barrier. Both peers must have entered the
      # scope before any remote access to its VMEM or local semaphores.
      barrier = pltpu.get_barrier_semaphore()
      for peer in (left, right):
        pl.semaphore_signal(
            barrier,
            device_id={axis_name: peer},
            device_id_type=pl.DeviceIdType.MESH,
        )
      pl.semaphore_wait(barrier, 2)
      for peer in (left, right):
        pl.semaphore_signal(
            startup_sem,
            device_id={axis_name: peer},
            device_id_type=pl.DeviceIdType.MESH,
        )
      pl.semaphore_wait(startup_sem, 2)

      @pl.when(max_post_num == 1)
      def _adjacent():
        op = pltpu.make_async_remote_copy(
            buffers_ref.at[0, :, :, :V],
            buffers_ref.at[1, :, :, :V],
            send_sem,
            recv_sem,
            device_id={axis_name: left},
            device_id_type=pl.DeviceIdType.MESH,
        )
        op.start()
        op.wait()
        merged_ref[:] = jnp.where(
            (post_num == 1) & (is_last == 0),
            buffers_ref[1, :, :, :V],
            jnp.zeros_like(merged_ref[:]),
        )

      @pl.when(max_post_num > 1)
      def _general():
        @pl.loop(0, cp_size - 1, unroll=False)
        def _step(i):
          work = i % 2
          recv = 1 - work
          # Entry to this iteration proves the previous receive semaphore
          # was consumed and the old work slot's send/merge has finished.
          # Credit, not alternating slots alone, bounds a fast sender.
          pl.semaphore_signal(
              ready_sem,
              device_id={axis_name: left},
              device_id_type=pl.DeviceIdType.MESH,
          )
          pl.semaphore_wait(ready_sem, 1)
          op = pltpu.make_async_remote_copy(
              buffers_ref.at[work],
              buffers_ref.at[recv],
              send_sem,
              recv_sem,
              device_id={axis_name: right},
              device_id_type=pl.DeviceIdType.MESH,
          )
          op.start()
          op.wait()
          origin = (rank + cp_size - i - 1) % cp_size

          @pl.when(
              (origin > rank) & (origin <= rank + post_num) & (is_last == 0)
          )
          def _merge_packet():
            merged_ref[:] = (
                jnp.matmul(
                    buffers_ref[recv, :, :, V:],
                    merged_ref[:],
                    preferred_element_type=jnp.float32,
                )
                + buffers_ref[recv, :, :, :V]
            )


@partial(jax.jit, static_argnames=["scale"])
def compute_dav(bdo, bvn, bAqk, scale, segment_ids=None):
  """Compute dAqk/dV, optionally masking a compact segmented chunk."""
  # Forward stores Aqk with its strict upper triangle zeroed, so masking it
  # again before the dV contraction only adds a full-tile VPU select.
  BT = bdo.shape[1]
  m_causal = (
      jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
      if segment_ids is None
      else same_segment_causal_mask(segment_ids)
  )
  bdAqk = jnp.matmul(
      bdo.astype(jnp.float32),
      bvn.transpose(0, 2, 1),
      preferred_element_type=jnp.float32,
  )
  bdAqk = jnp.where(m_causal[None, :, :], bdAqk * scale, 0.0)
  bAqk = jnp.where(m_causal[None], bAqk, 0)
  bdv0 = jnp.matmul(
      bAqk.transpose(0, 2, 1),
      bdo.astype(jnp.float32),
      preferred_element_type=jnp.float32,
  )
  return bdAqk, bdv0


@jax.jit
def compute_dv0(bdo, bAqk):
  """Compute the Aqk contribution to dV without materializing dAqk."""
  return jnp.matmul(
      bAqk.transpose(0, 2, 1),
      bdo.astype(jnp.float32),
      preferred_element_type=jnp.float32,
  )


def _packed_sequence_indices(boundaries, positions):
  """Map packed positions to sequences without a large token-by-capacity array."""
  if boundaries.shape[0] > 8193:
    # Right insertion skips empty sequences (including leading/interior ones),
    # exactly matching the legacy count of boundaries <= each token position.
    seq_idx = jnp.searchsorted(
        boundaries[1:], positions, side="right", method="scan"
    )
  else:
    seq_idx = jnp.sum(positions[:, None] >= boundaries[None, 1:], axis=1)
  return jnp.minimum(seq_idx, boundaries.shape[0] - 2)


def _compact_head_first_output_pallas(
    x,
    orig_cu_seqlens,
    aligned_cu_seqlens,
    t_out,
    chunk_size,
):
  """Compact one aligned head-first gradient into original packed order."""
  H, B, T_aligned, feature_size = x.shape
  compact_mb = min(16, H)
  while H % compact_mb != 0:
    compact_mb -= 1
  if orig_cu_seqlens.ndim == 1:
    orig_cu_seqlens = jnp.broadcast_to(
        orig_cu_seqlens[None, :], (B, orig_cu_seqlens.shape[0])
    )
  if aligned_cu_seqlens.ndim == 1:
    aligned_cu_seqlens = jnp.broadcast_to(
        aligned_cu_seqlens[None, :], (B, aligned_cu_seqlens.shape[0])
    )

  scratch_bytes = (
      compact_mb * T_aligned * feature_size
      + (t_out + chunk_size) * compact_mb * feature_size
  ) * x.dtype.itemsize
  while scratch_bytes > 48 * 1024 * 1024 and compact_mb > 1:
    compact_mb -= 1
    while H % compact_mb != 0:
      compact_mb -= 1
    scratch_bytes = (
        compact_mb * T_aligned * feature_size
        + (t_out + chunk_size) * compact_mb * feature_size
    ) * x.dtype.itemsize
  use_pallas = (
      not get_interpret()
      # This kernel prefetches both complete boundary arrays. Large capacities
      # use indexed HBM gathers instead, with no capacity-sized SMEM operand.
      and orig_cu_seqlens.size <= 8193
      and compact_mb % 8 == 0
      and t_out % 8 == 0
      and scratch_bytes <= 48 * 1024 * 1024
  )
  if not use_pallas:
    positions = jnp.arange(t_out, dtype=jnp.int32)
    batches = []
    for batch_idx in range(B):
      seq_idx = _packed_sequence_indices(orig_cu_seqlens[batch_idx], positions)
      padding_offsets = (
          aligned_cu_seqlens[batch_idx, :-1] - orig_cu_seqlens[batch_idx, :-1]
      )
      aligned_idx = jnp.clip(
          positions + padding_offsets[seq_idx], 0, T_aligned - 1
      )
      values = jnp.take(x[:, batch_idx], aligned_idx, axis=1)
      valid = positions < orig_cu_seqlens[batch_idx, -1]
      batches.append(
          jnp.where(valid[None, :, None], values, jnp.zeros_like(values))
      )
    return jnp.stack(batches, axis=1).transpose(1, 2, 0, 3)

  num_seqs = orig_cu_seqlens.shape[1] - 1

  def _compact_kernel(
      orig_ref,
      aligned_ref,
      src_hbm_ref,
      out_hbm_ref,
      src_ref,
      packed_ref,
  ):
    head_group = pl.program_id(0)
    batch = pl.program_id(1)
    head_start = head_group * compact_mb
    pltpu.sync_copy(
        src_hbm_ref.at[pl.ds(head_start, compact_mb), pl.ds(batch, 1), :, :],
        src_ref,
    )
    packed_ref[:] = jnp.zeros_like(packed_ref[:])

    @pl.loop(0, num_seqs, unroll=False)
    def _compact_seq(seq):
      orig_start = orig_ref[batch, seq]
      seq_len = orig_ref[batch, seq + 1] - orig_start
      aligned_start = aligned_ref[batch, seq]
      num_chunks = (seq_len + chunk_size - 1) // chunk_size

      @pl.loop(0, num_chunks, unroll=False)
      def _compact_chunk(chunk):
        aligned_token_start = pl.multiple_of(
            aligned_start + chunk * chunk_size, 16
        )
        values = src_ref[:, 0, pl.ds(aligned_token_start, chunk_size), :]
        packed_ref[
            0, pl.ds(orig_start + chunk * chunk_size, chunk_size), :, :
        ] = jnp.transpose(values, (1, 0, 2))

    pltpu.sync_copy(
        packed_ref.at[pl.ds(0, 1), pl.ds(0, t_out), :, :],
        out_hbm_ref.at[
            pl.ds(batch, 1),
            pl.ds(0, t_out),
            pl.ds(head_start, compact_mb),
            pl.ds(None),
        ],
    )

  return pl.pallas_call(
      _compact_kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          grid=(H // compact_mb, B),
          in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          scratch_shapes=[
              pltpu.VMEM((compact_mb, 1, T_aligned, feature_size), x.dtype),
              pltpu.VMEM(
                  (1, t_out + chunk_size, compact_mb, feature_size), x.dtype
              ),
          ],
      ),
      out_shape=jax.ShapeDtypeStruct((B, t_out, H, feature_size), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
          vmem_limit_bytes=min(
              get_tpu_config().vmem_limit_bytes,
              KDA_BWD_SCOPED_VMEM_LIMIT_BYTES,
          ),
      ),
      cost_estimate=kda_compact_output_cost(
          B,
          H,
          T_aligned,
          t_out,
          feature_size,
          dtype_bytes=x.dtype.itemsize,
      ),
  )(
      orig_cu_seqlens.astype(jnp.int32),
      aligned_cu_seqlens.astype(jnp.int32),
      x,
  )


def _bwd_mega_scratch_shapes(
    *,
    MB,
    K,
    V,
    BT,
    BT_PAD,
    CHUNK_BATCH,
    CP_ACTIVE,
    CP_SIZE,
    PACKED_INPUTS,
    BATCH_FIRST_INPUTS,
    FUSE_GATE_BWD=False,
    dtypes,
    B=1,
):
  """Shared head-group/backward allocations for tracing and tile selection."""
  MB_PAD = align_up(MB, 16)
  packed_buffer_rows = 2 * BT + 8
  DMA_INPUT_BUFFER_COUNT = KDA_DMA_INPUT_BUFFER_COUNT
  DMA_OUTPUT_BUFFER_COUNT = KDA_DMA_OUTPUT_BUFFER_COUNT
  state = {"dh_tmp_ref": pltpu.VMEM((MB, K, V), jnp.float32)}
  if CP_ACTIVE:
    state.update(
        {
            "dS_acc_ref": pltpu.VMEM((MB, K, V), jnp.float32),
            "M_acc_ref": pltpu.VMEM((MB, K, K), jnp.float32),
            "cp_buf_ref": pltpu.VMEM((2, MB, K, V + K), jnp.float32),
        }
    )
  backward = {
      "q_buf": pltpu.VMEM(
          (1, 1, 1, 1, 1, 1)
          if PACKED_INPUTS
          else (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K),
          dtypes["q"],
      ),
      "k_buf": pltpu.VMEM(
          (1, 1, 1, 1, 1, 1)
          if PACKED_INPUTS
          else (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K),
          dtypes["k"],
      ),
      "v_buf": pltpu.VMEM(
          (1, 1, 1, 1, 1, 1)
          if PACKED_INPUTS
          else (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, V),
          dtypes["v"],
      ),
      "do_buf": pltpu.VMEM(
          (1, 1, 1, 1, 1, 1)
          if PACKED_INPUTS
          else (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, V),
          dtypes["do"],
      ),
      "beta_buf": pltpu.VMEM(
          (1, 1)
          if PACKED_INPUTS
          else (DMA_INPUT_BUFFER_COUNT, CHUNK_BATCH * MB_PAD, 128),
          dtypes["beta"],
      ),
      "packed_q_buf": pltpu.VMEM(
          (
              (2, 1, packed_buffer_rows, MB, K)
              if BATCH_FIRST_INPUTS
              else (2, MB, 1, packed_buffer_rows, K)
          )
          if PACKED_INPUTS
          else (1, 1, 1, 1, 1),
          dtypes["q"],
      ),
      "packed_k_buf": pltpu.VMEM(
          (
              (2, 1, packed_buffer_rows, MB, K)
              if BATCH_FIRST_INPUTS
              else (2, MB, 1, packed_buffer_rows, K)
          )
          if PACKED_INPUTS
          else (1, 1, 1, 1, 1),
          dtypes["k"],
      ),
      "packed_v_buf": pltpu.VMEM(
          (
              (2, 1, packed_buffer_rows, MB, V)
              if BATCH_FIRST_INPUTS
              else (2, MB, 1, packed_buffer_rows, V)
          )
          if PACKED_INPUTS
          else (1, 1, 1, 1, 1),
          dtypes["v"],
      ),
      "packed_do_buf": pltpu.VMEM(
          (
              (2, 1, packed_buffer_rows, MB, V)
              if BATCH_FIRST_INPUTS
              else (2, MB, 1, packed_buffer_rows, V)
          )
          if PACKED_INPUTS
          else (1, 1, 1, 1, 1),
          dtypes["do"],
      ),
      "packed_beta_buf": pltpu.VMEM(
          (2, MB, 1, 16, 128) if PACKED_INPUTS else (1, 1, 1, 1, 1),
          dtypes["beta"],
      ),
      "Aqk_buf": pltpu.VMEM(
          (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, BT_PAD),
          dtypes["Aqk"],
      ),
      "Akk_buf": pltpu.VMEM(
          (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, BT_PAD),
          dtypes["Akk"],
      ),
      "g_buf": pltpu.VMEM(
          (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K), dtypes["g"]
      ),
      "raw_gate_buf": pltpu.VMEM(
          (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K)
          if FUSE_GATE_BWD
          else (1, 1, 1, 1, 1, 1),
          dtypes.get("raw_gate", dtypes["g"]),
      ),
      "h_buf": pltpu.VMEM(
          (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, K, V), dtypes["h"]
      ),
      # Independent output ping-pong must not alias an in-flight input DMA tile.
      "dq_buf": pltpu.VMEM(
          (DMA_OUTPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K), dtypes["dq"]
      ),
      "dk_buf": pltpu.VMEM(
          (DMA_OUTPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K), dtypes["dk"]
      ),
      "dv_buf": pltpu.VMEM(
          (DMA_OUTPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, V), dtypes["dv"]
      ),
      "db_buf": pltpu.VMEM(
          (DMA_OUTPUT_BUFFER_COUNT, CHUNK_BATCH * MB_PAD, 128), dtypes["db"]
      ),
      "dg_buf": pltpu.VMEM(
          (DMA_OUTPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K), dtypes["dg"]
      ),
      "gate_dA_acc": pltpu.VMEM(
          (MB, 128) if FUSE_GATE_BWD else (1, 1), jnp.float32
      ),
      "gate_dbias_acc": pltpu.VMEM(
          (MB, K) if FUSE_GATE_BWD else (1, 1), jnp.float32
      ),
      "gate_A_buf": pltpu.VMEM(
          (MB, K) if FUSE_GATE_BWD else (1, 1), jnp.float32
      ),
      "gate_bias_buf": pltpu.VMEM(
          (MB, K) if FUSE_GATE_BWD else (1, 1), jnp.float32
      ),
  }
  return state, backward


def _bwd_mega_vmem_bytes(state, backward) -> int:
  """Conservative tiled scratch bytes, excluding compiler temporaries.

  R/P/B scopes execute sequentially. With BT=64, K/V>=128 and BF16/FP32
  operands, B contains at least the storage required by either R or P.
  The enclosing state scope remains live throughout all three phases.
  """

  def size(spec):
    itemsize = jnp.dtype(spec.dtype).itemsize
    # Include physical padding even for dummy refs and small packed head
    # groups. Narrow elements may pack two rows into one 32-bit sublane.
    rows = align_up(spec.shape[-2], 8 * max(1, 4 // itemsize))
    cols = align_up(spec.shape[-1], 128)
    return math.prod(spec.shape[:-2]) * rows * cols * itemsize

  return sum(size(spec) for spec in (*state.values(), *backward.values()))


def _select_bwd_mega_tiling(
    *,
    H,
    K,
    V,
    NT,
    BT,
    dtypes,
    packed_inputs,
    cp_active,
    cp_size,
    chunk_batch,
    mini_batch,
    vmem_limit_bytes,
    fuse_gate_bwd=False,
    B=1,
) -> tuple[int, int]:
  """Choose legal DMA windows/head groups within the scoped VMEM budget.

  chunk_batch is an upper bound. An explicit mini_batch (including the
  environment override resolved by the caller) is exact and must fit.
  Automatic head groups shrink when local state, dimensions or dtypes demand it.
  CP communication storage is independent of the number of ranks.
  """
  assert NT > 0, f"NT must be positive, got {NT}"
  assert chunk_batch > 0, f"chunk_batch must be positive, got {chunk_batch}"
  assert H > 0, f"H must be positive, got {H}"
  assert not cp_active or (cp_size is not None and cp_size > 0)

  cp_mb16_supported = (
      cp_active
      and packed_inputs
      and dtypes["q"] == jnp.bfloat16
      and K == 128
      and V == 128
  )
  needs_fp32_budget = dtypes["q"] == jnp.float32 or cp_active
  max_mb = 16 if cp_mb16_supported else 8 if needs_fp32_budget else 16
  if mini_batch is not None:
    assert mini_batch > 0, f"mini_batch must be positive, got {mini_batch}"
    assert (
        H % mini_batch == 0
    ), f"H={H} must be divisible by mini_batch={mini_batch}"
    assert (
        not packed_inputs or mini_batch == H or mini_batch % 8 == 0
    ), "packed batch-first DMA requires mini_batch == H or mini_batch % 8 == 0"
    head_groups = (mini_batch,)
  else:
    head_groups = range(min(H, max_mb), 0, -1)

  budget = min(vmem_limit_bytes, KDA_BWD_SCOPED_VMEM_LIMIT_BYTES)
  # Scratch refs are only a lower bound on the compiled live set. Reserve
  # 12.5% for layout conversions, FP32 intermediates and compiler spill slots.
  scratch_budget = budget * 7 // 8
  for MB in head_groups:
    if H % MB or (packed_inputs and MB != H and MB % 8):
      continue
    cb_limit = min(chunk_batch, NT)
    if cp_active and packed_inputs and MB > 8:
      cb_limit = min(cb_limit, 2)
    for CB in range(cb_limit, 0, -1):
      # All caps, including the CP VMEM cap, precede the packed invariant.
      if packed_inputs and NT % CB:
        continue
      state, backward = _bwd_mega_scratch_shapes(
          B=B,
          MB=MB,
          K=K,
          V=V,
          BT=BT,
          BT_PAD=align_up(BT, 128),
          CHUNK_BATCH=CB,
          CP_ACTIVE=cp_active,
          CP_SIZE=cp_size,
          PACKED_INPUTS=packed_inputs,
          BATCH_FIRST_INPUTS=packed_inputs,
          FUSE_GATE_BWD=fuse_gate_bwd,
          dtypes=dtypes,
      )
      if _bwd_mega_vmem_bytes(state, backward) <= scratch_budget:
        return MB, CB
  raise AssertionError(
      "KDA backward cannot fit its VMEM scratch budget "
      f"({scratch_budget} bytes plus compiler headroom within {budget} bytes): "
      f"H={H}, K={K}, V={V}, cp_size={cp_size}, "
      f"mini_batch={mini_batch}, chunk_batch={chunk_batch}. "
      "Unset TOPS_KDA_MEGA_MINI_BATCH / mini_batch to allow automatic head "
      "group sizing, or reduce the local state dimensions."
  )


def _bwd_mega_kernel(
    chunk_seg_ids_ref,
    orig_cu_seqlens_ref,
    chunk_metadata_ref,
    cp_post_num_ref,
    cp_is_last_ref,
    original_segment_ids_ref,
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    Aqk_ref,
    Akk_ref,
    g_ref,
    h_ref,
    do_ref,
    dht_ref,
    initial_state_ref,
    raw_gate_ref,
    gate_A_log_ref,
    gate_dt_bias_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    db_ref,
    dg_ref,
    dh0_ref,
    h_all_ref,
    gate_dA_ref,
    gate_dbias_ref,
    *,
    H,
    B,
    BT,
    BT_PAD,
    K,
    V,
    NT,
    T_INPUT,
    SCALE,
    MB,
    MB_PAD,
    CHUNK_BATCH,
    DMA_INPUT_BUFFER_COUNT,
    DMA_OUTPUT_BUFFER_COUNT,
    DISABLE_RECOMPUTE,
    CP_ACTIVE,
    CP_SIZE,
    CP_AXIS_NAME,
    HAS_H0,
    HAS_DHT,
    RETURN_DH0,
    PACKED_INPUTS,
    MULTI_SEGMENT_PACKED,
    BATCH_FIRST_INPUTS,
    COMPACT_CHUNK_METADATA,
    PRECOMPUTED_CHUNK_WINDOWS,
    COMPACT_BOUNDARIES,
    ORIGINAL_TOKEN_SEGMENTS,
    FUSE_GATE_BWD,
):
  """Single-program M5 kernel with manual HBM<->VMEM DMA.

  Cross-phase state is allocated inside each batch/head-group iteration with
  pl.run_scoped. The enclosing pallas_call has no externally declared
  temporary VMEM.
  """
  compute_dtype = q_ref.dtype
  precision = (
      None if compute_dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
  )
  state_scratch, backward_scratch = _bwd_mega_scratch_shapes(
      B=B,
      MB=MB,
      K=K,
      V=V,
      BT=BT,
      BT_PAD=BT_PAD,
      CHUNK_BATCH=CHUNK_BATCH,
      CP_ACTIVE=CP_ACTIVE,
      CP_SIZE=CP_SIZE,
      PACKED_INPUTS=PACKED_INPUTS,
      BATCH_FIRST_INPUTS=BATCH_FIRST_INPUTS,
      FUSE_GATE_BWD=FUSE_GATE_BWD,
      dtypes={
          name: ref.dtype
          for name, ref in (
              ("q", q_ref),
              ("k", k_ref),
              ("v", v_ref),
              ("do", do_ref),
              ("beta", beta_ref),
              ("Aqk", Aqk_ref),
              ("Akk", Akk_ref),
              ("g", g_ref),
              ("raw_gate", raw_gate_ref),
              ("h", h_ref),
              ("dq", dq_ref),
              ("dk", dk_ref),
              ("dv", dv_ref),
              ("db", db_ref),
              ("dg", dg_ref),
          )
      },
  )

  def _matmul_low(lhs, rhs, precision=precision):
    return _matmul_in_dtype(lhs, rhs, compute_dtype, precision)

  @jax.jit
  def _dma_copy(src, dst, sem):
    cp = pltpu.make_async_copy(src, dst, sem)
    cp.start()
    cp.wait()

  @jax.jit
  def _dma_in(hbm_ref, vmem_ref, sem, mb_offset, b, c):
    src = hbm_ref.at[
        (
            pl.ds(mb_offset * MB, MB),
            pl.ds(b, 1),
            pl.ds(c, 1),
            pl.ds(None),
            pl.ds(None),
        )
    ]
    _dma_copy(src, vmem_ref, sem)

  @jax.jit
  def _dma_out(vmem_ref, hbm_ref, sem, mb_offset, b, c):
    dst = hbm_ref.at[
        (
            pl.ds(mb_offset * MB, MB),
            pl.ds(b, 1),
            pl.ds(c, 1),
            pl.ds(None),
            pl.ds(None),
        )
    ]
    _dma_copy(vmem_ref, dst, sem)

  def _async_copy(src, dst, sem, wait=False):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
      cp.wait()
    else:
      cp.start()

  def _dma_beta_in(beta_buf, sem, mb_offset, b, c, wait=False):
    """Copy one compact [head-group, chunk] beta tile into VMEM."""
    beta_offset = ((mb_offset * B + b) * NT + c) * MB_PAD
    src = beta_ref.at[(pl.ds(beta_offset, MB_PAD), pl.ds(None))]
    _async_copy(src, beta_buf, sem, wait=wait)

  @jax.jit
  def _dma_state_in(state_ref, vmem_ref, sem, mb_offset, b, seq):
    src = state_ref.at[
        (pl.ds(mb_offset * MB, MB), b, seq, pl.ds(None), pl.ds(None))
    ]
    _dma_copy(src, vmem_ref, sem)

  @jax.jit
  def _dma_state_out(vmem_ref, state_ref, sem, mb_offset, b, seq):
    dst = state_ref.at[
        (pl.ds(mb_offset * MB, MB), b, seq, pl.ds(None), pl.ds(None))
    ]
    _dma_copy(vmem_ref, dst, sem)

  packed_load_rows = BT + 8
  # A packed token DMA loads BT + 8 rows and may shift the window right by
  # at most BT rows at the end of the input. No branch reads beyond this.
  packed_buffer_rows = 2 * BT + 8
  packed_dma_rows = min(T_INPUT, packed_load_rows)

  def _packed_input_window(batch, chunk):
    if ORIGINAL_TOKEN_SEGMENTS:
      packed_start = chunk * BT
      remaining = T_INPUT - packed_start
    elif PRECOMPUTED_CHUNK_WINDOWS:
      packed_start = chunk_metadata_ref[batch, 0, chunk]
      remaining = chunk_metadata_ref[batch, 1, chunk]
    elif COMPACT_CHUNK_METADATA:
      # Canonical generated chunk metadata is fully determined by the
      # aligned sequence boundaries.  Reconstruct the location from the
      # O(N) table instead of scalar-prefetching O(T / BT) pairs into SMEM.
      num_seqs = chunk_metadata_ref.shape[-1] - 1
      chunk_start = chunk * BT
      seq_idx = jnp.int32(0)
      for seq_boundary in range(1, num_seqs + 1):
        seq_idx += (
            chunk_start >= chunk_metadata_ref[batch, seq_boundary]
        ).astype(jnp.int32)
      seq_idx = jnp.minimum(seq_idx, num_seqs - 1)
      block_idx = chunk - chunk_metadata_ref[batch, seq_idx] // BT
    else:
      # Preserve caller-provided mappings, which may intentionally use a
      # non-canonical chunk launch order.
      seq_idx = chunk_metadata_ref[batch, chunk, 0]
      block_idx = chunk_metadata_ref[batch, chunk, 1]
    if not ORIGINAL_TOKEN_SEGMENTS and not PRECOMPUTED_CHUNK_WINDOWS:
      bos = orig_cu_seqlens_ref[batch, seq_idx]
      eos = orig_cu_seqlens_ref[batch, seq_idx + 1]
      valid_chunk = block_idx * BT < eos - bos
      packed_start = jnp.where(valid_chunk, bos + block_idx * BT, jnp.int32(0))
      remaining = eos - packed_start
    window_start = packed_start - jnp.mod(packed_start, jnp.int32(8))
    source_start = jnp.minimum(
        window_start, jnp.int32(max(T_INPUT - packed_dma_rows, 0))
    )
    source_start = source_start - jnp.mod(source_start, jnp.int32(8))
    source_start = pl.multiple_of(source_start, 8)
    source_shift = (window_start - source_start) // jnp.int32(8)
    beta_group = window_start // jnp.int32(BT)
    beta_source_start = beta_group - jnp.mod(beta_group, jnp.int32(8))
    beta_source_start = pl.multiple_of(beta_source_start, 8)
    beta_source_shift = beta_group - beta_source_start
    packed_offset = jnp.mod(packed_start, jnp.int32(8))
    beta_group_offset = jnp.mod(packed_start // jnp.int32(8), jnp.int32(8))
    return (
        source_start,
        source_shift,
        beta_source_start,
        beta_source_shift,
        packed_offset,
        beta_group_offset,
        remaining,
    )

  def _packed_token_dma_descs(
      buf,
      mb_offset,
      batch,
      chunk,
      q_buf,
      k_buf,
      v_buf,
      do_buf,
      beta_buf,
      sems,
  ):
    source_start, _, beta_source_start, _, _, _, _ = _packed_input_window(
        batch, chunk
    )
    # Batch-first HBM tiles are 8-head tiled. MB is an 8-aligned packed-path
    # specialization, but the dynamic group index needs an explicit Mosaic
    # alignment annotation before it can be used as a DMA slice offset.
    head_start = pl.multiple_of(mb_offset * MB, 8)

    def _token_desc(src_ref, dst_ref, sem):
      if BATCH_FIRST_INPUTS:
        src = src_ref.at[
            pl.ds(batch, 1),
            pl.ds(source_start, packed_dma_rows),
            pl.ds(head_start, MB),
            pl.ds(None),
        ]
        dst = dst_ref.at[buf, :, pl.ds(0, packed_dma_rows), :, :]
      else:
        src = src_ref.at[
            pl.ds(head_start, MB),
            pl.ds(batch, 1),
            pl.ds(source_start, packed_dma_rows),
            pl.ds(None),
        ]
        dst = dst_ref.at[buf, :, :, pl.ds(0, packed_dma_rows), :]
      return pltpu.make_async_copy(
          src,
          dst,
          sem,
      )

    return (
        _token_desc(q_ref, q_buf, sems.at[0, buf]),
        _token_desc(k_ref, k_buf, sems.at[1, buf]),
        _token_desc(v_ref, v_buf, sems.at[2, buf]),
        _token_desc(do_ref, do_buf, sems.at[3, buf]),
        pltpu.make_async_copy(
            beta_ref.at[
                pl.ds(head_start, MB),
                pl.ds(batch, 1),
                pl.ds(beta_source_start, 16),
                pl.ds(None),
            ],
            beta_buf.at[buf],
            sems.at[4, buf],
        ),
    )

  def _align_packed_tail_window(buf, source_shift, *token_bufs):
    for shift in range(1, BT // 8 + 1):
      row_start = shift * 8

      @pl.when(source_shift == shift)
      def _shift(row_start=row_start):
        for token_buf in token_bufs:
          if BATCH_FIRST_INPUTS:
            token_buf[buf, :, :packed_load_rows, :, :] = token_buf[
                buf, :, row_start : row_start + packed_load_rows, :, :
            ]
          else:
            token_buf[buf, :, :, :packed_load_rows, :] = token_buf[
                buf, :, :, row_start : row_start + packed_load_rows, :
            ]

  def _packed_token_view(token_buf, buf, packed_offset):
    if BATCH_FIRST_INPUTS:
      branches = tuple(
          lambda operand, offset=offset: jnp.transpose(
              operand[buf, 0, offset : offset + BT, :, :], (1, 0, 2)
          )
          for offset in range(8)
      )
      return jax.lax.switch(packed_offset, branches, token_buf)
    branches = tuple(
        lambda operand, offset=offset: operand[
            buf, :, 0, offset : offset + BT, :
        ]
        for offset in range(8)
    )
    return jax.lax.switch(packed_offset, branches, token_buf)

  def _packed_beta_view(
      beta_buf, buf, beta_source_shift, beta_group_offset, packed_offset
  ):
    beta_group_branches = tuple(
        lambda operand, shift=shift: operand[buf, :, 0, shift : shift + 2, :]
        for shift in range(8)
    )
    beta_pair = jax.lax.switch(beta_source_shift, beta_group_branches, beta_buf)

    def _beta_aligned_window(x):
      branches = []
      for group_offset in range(8):
        start = group_offset * 8
        head = start + 8
        branches.append(
            lambda operand, start=start, head=head: jnp.concatenate(
                [operand[:, 0, start:BT], operand[:, 1, :head]], axis=1
            )
        )
      return jax.lax.switch(beta_group_offset, tuple(branches), x)

    beta_window = _beta_aligned_window(beta_pair)
    branches = tuple(
        lambda operand, offset=offset: operand[:, offset : offset + BT]
        for offset in range(8)
    )
    return jax.lax.switch(packed_offset, branches, beta_window)

  def main_loop_body(
      batch_idx,
      mb_offset,
      dh_tmp_ref,
      dS_acc_ref=None,
      M_acc_ref=None,
      cp_buf_ref=None,
  ):
    # Only CP consumes the first/last local segment IDs. Keeping this static
    # NT-wide scan out of the non-CP program materially shortens the Pallas
    # jaxpr embedded in large layer scans.
    if CP_ACTIVE:
      first_seg_id = chunk_seg_ids_ref[batch_idx, 0]
      first_valid_chunk = jnp.where(first_seg_id != 0, 0, NT)
      first_run_end = jnp.where(first_seg_id != 0, 1, 0)
      last_valid_chunk = jnp.where(first_seg_id != 0, 0, -1)
      for seg_i in range(1, NT):
        seg_i_id = chunk_seg_ids_ref[batch_idx, seg_i]
        first_valid_chunk = jnp.where(
            (first_seg_id == 0) & (seg_i_id != 0), seg_i, first_valid_chunk
        )
        first_run_end = jnp.where(
            (seg_i_id != 0)
            & (
                (first_seg_id == 0)
                | ((seg_i_id == first_seg_id) & (first_run_end == seg_i))
            ),
            seg_i + 1,
            first_run_end,
        )
        first_seg_id = jnp.where(
            (first_seg_id == 0) & (seg_i_id != 0), seg_i_id, first_seg_id
        )
        last_valid_chunk = jnp.where(seg_i_id != 0, seg_i, last_valid_chunk)

    # =================================================================
    # Phase 1: Forward recompute_h  (only when !disable_recompute)
    # =================================================================
    if not DISABLE_RECOMPUTE:

      @functools.partial(
          pl.run_scoped,
          k_fwd_buf=pltpu.VMEM(
              (
                  (1, 1, packed_buffer_rows, MB, K)
                  if BATCH_FIRST_INPUTS
                  else (1, MB, 1, packed_buffer_rows, K)
              )
              if PACKED_INPUTS
              else (MB, 1, 1, BT, K),
              k_ref.dtype,
          ),
          v_fwd_buf=pltpu.VMEM(
              (
                  (1, 1, packed_buffer_rows, MB, V)
                  if BATCH_FIRST_INPUTS
                  else (1, MB, 1, packed_buffer_rows, V)
              )
              if PACKED_INPUTS
              else (MB, 1, 1, BT, V),
              v_ref.dtype,
          ),
          beta_fwd_buf=pltpu.VMEM(
              (1, MB, 1, 16, 128) if PACKED_INPUTS else (MB_PAD, 128),
              beta_ref.dtype,
          ),
          Akk_fwd_buf=pltpu.VMEM((MB, 1, 1, BT, BT_PAD), Akk_ref.dtype),
          g_fwd_buf=pltpu.VMEM((MB, 1, 1, BT, K), g_ref.dtype),
          h_fwd_out_buf=pltpu.VMEM((MB, 1, 1, K, V), jnp.float32),
          g_fwd_scratch=pltpu.VMEM((MB, BT, K), jnp.float32),
          in_sems_fwd=pltpu.SemaphoreType.DMA((5,)),
          out_sem_fwd=pltpu.SemaphoreType.DMA,
          state_sem_fwd=pltpu.SemaphoreType.DMA,
      )
      def _forward_recompute_scoped(
          k_fwd_buf,
          v_fwd_buf,
          beta_fwd_buf,
          Akk_fwd_buf,
          g_fwd_buf,
          h_fwd_out_buf,
          g_fwd_scratch,
          in_sems_fwd,
          out_sem_fwd,
          state_sem_fwd,
      ):
        @pl.loop(0, NT, unroll=False)
        def forward_loop(chunk_id):
          # chunk_id: 0 -> NT-1  (forward scan)
          _seg_cur, seq_idx, is_valid, is_first_chunk, _is_last_chunk = (
              _chunk_segment_metadata(
                  chunk_seg_ids_ref, batch_idx, chunk_id, NT
              )
          )

          # Match PR #463's all-padding DMA skip. Invalid chunks are masked
          # below, so their VMEM buffers do not need to be refreshed.
          @pl.when(is_valid)
          def _load_forward_chunk():
            if PACKED_INPUTS:
              source_start, _, beta_source_start, _, _, _, _ = (
                  _packed_input_window(batch_idx, chunk_id)
              )
              head_start = pl.multiple_of(mb_offset * MB, 8)
              if BATCH_FIRST_INPUTS:
                k_src = k_ref.at[
                    pl.ds(batch_idx, 1),
                    pl.ds(source_start, packed_dma_rows),
                    pl.ds(head_start, MB),
                    pl.ds(None),
                ]
                v_src = v_ref.at[
                    pl.ds(batch_idx, 1),
                    pl.ds(source_start, packed_dma_rows),
                    pl.ds(head_start, MB),
                    pl.ds(None),
                ]
                k_dst = k_fwd_buf.at[0, :, pl.ds(0, packed_dma_rows), :, :]
                v_dst = v_fwd_buf.at[0, :, pl.ds(0, packed_dma_rows), :, :]
              else:
                k_src = k_ref.at[
                    pl.ds(head_start, MB),
                    pl.ds(batch_idx, 1),
                    pl.ds(source_start, packed_dma_rows),
                    pl.ds(None),
                ]
                v_src = v_ref.at[
                    pl.ds(head_start, MB),
                    pl.ds(batch_idx, 1),
                    pl.ds(source_start, packed_dma_rows),
                    pl.ds(None),
                ]
                k_dst = k_fwd_buf.at[0, :, :, pl.ds(0, packed_dma_rows), :]
                v_dst = v_fwd_buf.at[0, :, :, pl.ds(0, packed_dma_rows), :]
              _dma_copy(k_src, k_dst, in_sems_fwd.at[0])
              _dma_copy(v_src, v_dst, in_sems_fwd.at[1])
              _dma_copy(
                  beta_ref.at[
                      pl.ds(head_start, MB),
                      pl.ds(batch_idx, 1),
                      pl.ds(beta_source_start, 16),
                      pl.ds(None),
                  ],
                  beta_fwd_buf.at[0],
                  in_sems_fwd.at[2],
              )
            else:
              _dma_in(
                  k_ref,
                  k_fwd_buf,
                  in_sems_fwd.at[0],
                  mb_offset,
                  batch_idx,
                  chunk_id,
              )
              _dma_in(
                  v_ref,
                  v_fwd_buf,
                  in_sems_fwd.at[1],
                  mb_offset,
                  batch_idx,
                  chunk_id,
              )
              _dma_beta_in(
                  beta_fwd_buf,
                  in_sems_fwd.at[2],
                  mb_offset,
                  batch_idx,
                  chunk_id,
              )
              _dma_beta_in(
                  beta_fwd_buf,
                  in_sems_fwd.at[2],
                  mb_offset,
                  batch_idx,
                  chunk_id,
                  wait=True,
              )
            _dma_in(
                Akk_ref,
                Akk_fwd_buf,
                in_sems_fwd.at[3],
                mb_offset,
                batch_idx,
                chunk_id,
            )
            _dma_in(
                g_ref,
                g_fwd_buf,
                in_sems_fwd.at[4],
                mb_offset,
                batch_idx,
                chunk_id,
            )

          @pl.when(~is_valid)
          def _zero_forward_chunk():
            k_fwd_buf[:] = jnp.zeros_like(k_fwd_buf[:])
            v_fwd_buf[:] = jnp.zeros_like(v_fwd_buf[:])
            beta_fwd_buf[:] = jnp.zeros_like(beta_fwd_buf[:])
            Akk_fwd_buf[:] = jnp.zeros_like(Akk_fwd_buf[:])
            g_fwd_buf[:] = jnp.zeros_like(g_fwd_buf[:])

          g_fwd_scratch[:] = g_fwd_buf[:, 0, 0]
          g_fwd_scratch[:] = jnp.where(
              is_valid, g_fwd_scratch[:], jnp.zeros_like(g_fwd_scratch[:])
          )

          @pl.when(is_first_chunk)
          def _load_initial():
            if HAS_H0:
              if CP_ACTIVE:
                # Only the first valid run receives the incoming state.
                # Later runs can reuse its label but must start from zero.
                dh_tmp_ref[:] = jnp.zeros_like(dh_tmp_ref[:])

                @pl.when(chunk_id == first_valid_chunk)
                def _load_cp_first_state():
                  _dma_state_in(
                      initial_state_ref,
                      dh_tmp_ref,
                      state_sem_fwd,
                      mb_offset,
                      batch_idx,
                      0,
                  )

              else:
                _dma_state_in(
                    initial_state_ref,
                    dh_tmp_ref,
                    state_sem_fwd,
                    mb_offset,
                    batch_idx,
                    seq_idx,
                )
            else:
              dh_tmp_ref[:] = jnp.zeros_like(dh_tmp_ref[:])

          @pl.when(is_valid)
          def _recompute_h():
            h_cur = dh_tmp_ref[:].astype(jnp.float32)
            if PACKED_INPUTS:
              (
                  _,
                  source_shift,
                  _,
                  beta_source_shift,
                  packed_offset,
                  beta_group_offset,
                  remaining,
              ) = _packed_input_window(batch_idx, chunk_id)
              _align_packed_tail_window(0, source_shift, k_fwd_buf, v_fwd_buf)
              fk = _packed_token_view(k_fwd_buf, 0, packed_offset).astype(
                  compute_dtype
              )
              fv = _packed_token_view(v_fwd_buf, 0, packed_offset).astype(
                  compute_dtype
              )
              fb = _packed_beta_view(
                  beta_fwd_buf,
                  0,
                  beta_source_shift,
                  beta_group_offset,
                  packed_offset,
              ).astype(compute_dtype)
              fk_valid = (
                  jax.lax.broadcasted_iota(jnp.int32, fk.shape, 1) < remaining
              )
              fv_valid = (
                  jax.lax.broadcasted_iota(jnp.int32, fv.shape, 1) < remaining
              )
              fb_valid = (
                  jax.lax.broadcasted_iota(jnp.int32, fb.shape, 1) < remaining
              )
              fk = jnp.where(fk_valid, fk, jnp.zeros_like(fk))
              fv = jnp.where(fv_valid, fv, jnp.zeros_like(fv))
              fb = jnp.where(fb_valid, fb, jnp.zeros_like(fb))
            else:
              fk = k_fwd_buf[:, 0, 0].astype(compute_dtype)
              fv = v_fwd_buf[:, 0, 0].astype(compute_dtype)
              fb = beta_fwd_buf[:MB, :BT].astype(compute_dtype)
            fA = Akk_fwd_buf[:, 0, 0, :, :BT].astype(compute_dtype)
            fg = g_fwd_scratch[:].astype(jnp.float32)
            fg_exp = jnp.exp2(fg)
            fg_last = fg[:, BT - 1, :]
            fv_beta = fv * fb[:, :, None]
            fu = _matmul_low(fA, fv_beta)
            fw = jnp.matmul(
                fA.astype(jnp.float32),
                (fk * fb[:, :, None] * fg_exp).astype(jnp.float32),
                preferred_element_type=jnp.float32,
            )
            fvn = fu.astype(jnp.float32) - jnp.matmul(
                fw.astype(jnp.float32),
                h_cur,
                precision=precision,
                preferred_element_type=jnp.float32,
            )
            h_fwd_out_buf[:, 0, 0] = h_cur
            _dma_out(
                h_fwd_out_buf,
                h_all_ref,
                out_sem_fwd,
                mb_offset,
                batch_idx,
                chunk_id,
            )
            fkg = fk * jnp.exp2(fg[:, BT - 1 : BT, :] - fg)
            dh_tmp_ref[:] = dh_tmp_ref[:] * jnp.exp2(fg_last)[:, :, None]
            dh_tmp_ref[:] = dh_tmp_ref[:] + jnp.matmul(
                fkg.astype(jnp.float32).transpose(0, 2, 1),
                fvn,
                precision=precision,
                preferred_element_type=jnp.float32,
            )

    # =================================================================
    # Phase 2: Preprocess (dAv + CP accumulate)  (only when cp_active)
    # =================================================================
    if CP_ACTIVE:

      @functools.partial(
          pl.run_scoped,
          q_cp_buf=pltpu.VMEM(
              (2, 1, packed_buffer_rows, MB, K)
              if PACKED_INPUTS
              else (2, MB, 1, 1, BT, K),
              q_ref.dtype,
          ),
          k_cp_buf=pltpu.VMEM(
              (2, 1, packed_buffer_rows, MB, K)
              if PACKED_INPUTS
              else (2, MB, 1, 1, BT, K),
              k_ref.dtype,
          ),
          do_cp_buf=pltpu.VMEM(
              (2, 1, packed_buffer_rows, MB, V)
              if PACKED_INPUTS
              else (2, MB, 1, 1, BT, V),
              do_ref.dtype,
          ),
          beta_cp_buf=pltpu.VMEM(
              (2, MB, 1, 16, 128) if PACKED_INPUTS else (2, MB_PAD, 128),
              beta_ref.dtype,
          ),
          Aqk_cp_buf=pltpu.VMEM((2, MB, 1, 1, BT, BT_PAD), Aqk_ref.dtype),
          Akk_cp_buf=pltpu.VMEM((2, MB, 1, 1, BT, BT_PAD), Akk_ref.dtype),
          g_cp_buf=pltpu.VMEM((2, MB, 1, 1, BT, K), g_ref.dtype),
          g_cp_scratch=pltpu.VMEM((MB, BT, K), jnp.float32),
          in_sems_cp=pltpu.SemaphoreType.DMA((7, 2)),
      )
      def _cp_preprocess_scoped(
          q_cp_buf,
          k_cp_buf,
          do_cp_buf,
          beta_cp_buf,
          Aqk_cp_buf,
          Akk_cp_buf,
          g_cp_buf,
          g_cp_scratch,
          in_sems_cp,
      ):
        def _dma_in_cp(hbm_ref, vmem_ref, sem, buf, c, wait=False):
          src = hbm_ref.at[
              (
                  pl.ds(mb_offset * MB, MB),
                  pl.ds(batch_idx, 1),
                  pl.ds(c, 1),
                  pl.ds(None),
                  pl.ds(None),
              )
          ]
          _async_copy(src, vmem_ref.at[buf], sem, wait=wait)

        def _packed_cp_input_descs(buf, chunk_id):
          source_start, _, beta_source_start, _, _, _, _ = _packed_input_window(
              batch_idx, chunk_id
          )
          head_start = pl.multiple_of(mb_offset * MB, 8)

          def _token_desc(src_ref, dst_ref, sem):
            return pltpu.make_async_copy(
                src_ref.at[
                    pl.ds(batch_idx, 1),
                    pl.ds(source_start, packed_dma_rows),
                    pl.ds(head_start, MB),
                    pl.ds(None),
                ],
                dst_ref.at[buf, :, pl.ds(0, packed_dma_rows), :, :],
                sem,
            )

          return (
              _token_desc(q_ref, q_cp_buf, in_sems_cp.at[0, buf]),
              _token_desc(k_ref, k_cp_buf, in_sems_cp.at[1, buf]),
              _token_desc(do_ref, do_cp_buf, in_sems_cp.at[5, buf]),
              pltpu.make_async_copy(
                  beta_ref.at[
                      pl.ds(head_start, MB),
                      pl.ds(batch_idx, 1),
                      pl.ds(beta_source_start, 16),
                      pl.ds(None),
                  ],
                  beta_cp_buf.at[buf],
                  in_sems_cp.at[2, buf],
              ),
          )

        def _start_cp_input_dma(buf, chunk_id, is_valid):
          @pl.when(is_valid)
          def _start_valid_cp_input_dma():
            if PACKED_INPUTS:
              for desc in _packed_cp_input_descs(buf, chunk_id):
                desc.start()
            else:
              _dma_in_cp(q_ref, q_cp_buf, in_sems_cp.at[0, buf], buf, chunk_id)
              _dma_in_cp(k_ref, k_cp_buf, in_sems_cp.at[1, buf], buf, chunk_id)
              _dma_beta_in(
                  beta_cp_buf.at[buf],
                  in_sems_cp.at[2, buf],
                  mb_offset,
                  batch_idx,
                  chunk_id,
              )
            _dma_in_cp(
                Aqk_ref, Aqk_cp_buf, in_sems_cp.at[3, buf], buf, chunk_id
            )
            _dma_in_cp(
                Akk_ref, Akk_cp_buf, in_sems_cp.at[4, buf], buf, chunk_id
            )
            if not PACKED_INPUTS:
              _dma_in_cp(
                  do_ref, do_cp_buf, in_sems_cp.at[5, buf], buf, chunk_id
              )
            _dma_in_cp(g_ref, g_cp_buf, in_sems_cp.at[6, buf], buf, chunk_id)

          @pl.when(~is_valid)
          def _zero_invalid_cp_inputs():
            q_cp_buf[buf] = jnp.zeros_like(q_cp_buf[buf])
            k_cp_buf[buf] = jnp.zeros_like(k_cp_buf[buf])
            beta_cp_buf[buf] = jnp.zeros_like(beta_cp_buf[buf])
            Aqk_cp_buf[buf] = jnp.zeros_like(Aqk_cp_buf[buf])
            Akk_cp_buf[buf] = jnp.zeros_like(Akk_cp_buf[buf])
            do_cp_buf[buf] = jnp.zeros_like(do_cp_buf[buf])
            g_cp_buf[buf] = jnp.zeros_like(g_cp_buf[buf])

        def _wait_cp_input_dma(buf, chunk_id, is_valid):
          @pl.when(is_valid)
          def _wait_valid_cp_input_dma():
            if PACKED_INPUTS:
              for desc in _packed_cp_input_descs(buf, chunk_id):
                desc.wait()
              source_shift = _packed_input_window(batch_idx, chunk_id)[1]
              _align_packed_tail_window(
                  buf, source_shift, q_cp_buf, k_cp_buf, do_cp_buf
              )
            else:
              _dma_in_cp(
                  q_ref,
                  q_cp_buf,
                  in_sems_cp.at[0, buf],
                  buf,
                  chunk_id,
                  wait=True,
              )
              _dma_in_cp(
                  k_ref,
                  k_cp_buf,
                  in_sems_cp.at[1, buf],
                  buf,
                  chunk_id,
                  wait=True,
              )
              _dma_beta_in(
                  beta_cp_buf.at[buf],
                  in_sems_cp.at[2, buf],
                  mb_offset,
                  batch_idx,
                  chunk_id,
                  wait=True,
              )
            _dma_in_cp(
                Aqk_ref,
                Aqk_cp_buf,
                in_sems_cp.at[3, buf],
                buf,
                chunk_id,
                wait=True,
            )
            _dma_in_cp(
                Akk_ref,
                Akk_cp_buf,
                in_sems_cp.at[4, buf],
                buf,
                chunk_id,
                wait=True,
            )
            if not PACKED_INPUTS:
              _dma_in_cp(
                  do_ref,
                  do_cp_buf,
                  in_sems_cp.at[5, buf],
                  buf,
                  chunk_id,
                  wait=True,
              )
            _dma_in_cp(
                g_ref, g_cp_buf, in_sems_cp.at[6, buf], buf, chunk_id, wait=True
            )

        # Summarize only the first contiguous valid run. Labels may repeat
        # later, and explicit maps may contain leading invalid slots.
        first_segment_chunks = first_run_end
        cp_preprocess_nt = jnp.maximum(first_run_end - first_valid_chunk, 1)

        first_chunk_id = jnp.maximum(first_segment_chunks - 1, 0)
        _, _, first_is_valid, _, _ = _chunk_segment_metadata(
            chunk_seg_ids_ref, batch_idx, first_chunk_id, NT
        )
        _start_cp_input_dma(0, first_chunk_id, first_is_valid)

        @pl.loop(0, cp_preprocess_nt, unroll=False)
        def preprocess_loop(rev_idx):
          chunk_id = jnp.maximum(first_segment_chunks - 1 - rev_idx, 0)
          buf = rev_idx % 2
          next_buf = 1 - buf

          _seg_cur, _seq_idx, is_valid, _is_first_chunk, _is_last_chunk = (
              _chunk_segment_metadata(
                  chunk_seg_ids_ref, batch_idx, chunk_id, NT
              )
          )
          is_first_local_segment = (
              is_valid
              & (chunk_id >= first_valid_chunk)
              & (chunk_id < first_run_end)
          )

          _wait_cp_input_dma(buf, chunk_id, is_valid)

          @pl.when(rev_idx + 1 < cp_preprocess_nt)
          def _prefetch_next_cp_chunk():
            next_chunk_id = jnp.maximum(first_segment_chunks - 2 - rev_idx, 0)
            _, _, next_is_valid, _, _ = _chunk_segment_metadata(
                chunk_seg_ids_ref, batch_idx, next_chunk_id, NT
            )
            _start_cp_input_dma(next_buf, next_chunk_id, next_is_valid)

          g_cp_scratch[:] = g_cp_buf[buf, :, 0, 0]
          g_cp_scratch[:] = jnp.where(
              is_valid, g_cp_scratch[:], jnp.zeros_like(g_cp_scratch[:])
          )

          # Init CP accumulators at first chunk (rev_idx == 0 -> chunk_id == NT-1)
          @pl.when(rev_idx == 0)
          def _init_cp_acc():
            dS_acc_ref[:] = jnp.zeros((MB, K, V), dtype=jnp.float32)
            M_acc_ref[:] = jnp.broadcast_to(
                jnp.eye(K, dtype=jnp.float32), (MB, K, K)
            )

          if PACKED_INPUTS:
            (
                _,
                _,
                _,
                beta_source_shift,
                packed_offset,
                beta_group_offset,
                remaining,
            ) = _packed_input_window(batch_idx, chunk_id)
            bq = _packed_token_view(q_cp_buf, buf, packed_offset).astype(
                compute_dtype
            )
            bk = _packed_token_view(k_cp_buf, buf, packed_offset).astype(
                compute_dtype
            )
            bdo = _packed_token_view(do_cp_buf, buf, packed_offset).astype(
                compute_dtype
            )
            bb = _packed_beta_view(
                beta_cp_buf,
                buf,
                beta_source_shift,
                beta_group_offset,
                packed_offset,
            ).astype(compute_dtype)
            # A packed DMA window may include the next segment. Whole-chunk
            # is_valid and the Aqk/Akk masks do not protect Qg.T @ dO.
            # Match the per-token tail masking in packed_chunk_operands.
            bq_valid = (
                jax.lax.broadcasted_iota(jnp.int32, bq.shape, 1) < remaining
            )
            bk_valid = (
                jax.lax.broadcasted_iota(jnp.int32, bk.shape, 1) < remaining
            )
            bdo_valid = (
                jax.lax.broadcasted_iota(jnp.int32, bdo.shape, 1) < remaining
            )
            bb_valid = (
                jax.lax.broadcasted_iota(jnp.int32, bb.shape, 1) < remaining
            )
            bq = jnp.where(bq_valid, bq, jnp.zeros_like(bq))
            bk = jnp.where(bk_valid, bk, jnp.zeros_like(bk))
            bdo = jnp.where(bdo_valid, bdo, jnp.zeros_like(bdo))
            bb = jnp.where(bb_valid, bb, jnp.zeros_like(bb))
          else:
            bq = q_cp_buf[buf, :, 0, 0].astype(compute_dtype)
            bk = k_cp_buf[buf, :, 0, 0].astype(compute_dtype)
            bdo = do_cp_buf[buf, :, 0, 0].astype(compute_dtype)
            bb = beta_cp_buf[buf, :MB, :BT].astype(compute_dtype)
          bg = g_cp_scratch[:].astype(jnp.float32)
          bA = Akk_cp_buf[buf, :, 0, 0, :, :BT].astype(compute_dtype)
          bAqk = Aqk_cp_buf[buf, :, 0, 0, :, :BT].astype(compute_dtype)

          bq = jnp.where(is_valid, bq, jnp.zeros_like(bq))
          bk = jnp.where(is_valid, bk, jnp.zeros_like(bk))
          bb = jnp.where(is_valid, bb, jnp.zeros_like(bb))
          bA = jnp.where(is_valid, bA, jnp.zeros_like(bA))
          bAqk = jnp.where(is_valid, bAqk, jnp.zeros_like(bAqk))
          bdo = jnp.where(is_valid, bdo, jnp.zeros_like(bdo))

          # CP preprocessing only consumes dV. Avoid loading v/h and skip
          # their M1 contractions; the full backward stage recomputes them.
          bdv0 = compute_dv0(bdo, bAqk)
          bkg = bk * jnp.exp2(bg[:, BT - 1 : BT, :] - bg)
          bw = jnp.matmul(
              bA.astype(jnp.float32),
              (bk * bb[:, :, None] * jnp.exp2(bg)).astype(jnp.float32),
              preferred_element_type=jnp.float32,
          )
          bqg = bq * jnp.exp2(bg)

          @pl.when(is_first_local_segment)
          def _cp_preprocess():
            cp_decay = jnp.exp2(
                jnp.maximum(
                    bg[:, BT - 1, :], jnp.asarray(-126.0, dtype=jnp.float32)
                )
            )
            dv_cur = (
                jnp.matmul(
                    bkg, dS_acc_ref[:], preferred_element_type=jnp.float32
                )
                + bdv0
            )
            dS_acc_ref[:] = (
                dS_acc_ref[:] * cp_decay[:, :, None]
                + jnp.matmul(
                    bqg.transpose(0, 2, 1),
                    bdo,
                    preferred_element_type=jnp.float32,
                )
                * SCALE
                - jnp.matmul(
                    bw.transpose(0, 2, 1),
                    dv_cur,
                    preferred_element_type=jnp.float32,
                )
            )
            kM = jnp.matmul(
                bkg,
                M_acc_ref[:],
                precision=precision,
                preferred_element_type=jnp.float32,
            )
            WkM = jnp.matmul(
                bw.transpose(0, 2, 1),
                kM,
                precision=precision,
                preferred_element_type=jnp.float32,
            )
            M_acc_ref[:] = M_acc_ref[:] * cp_decay[:, :, None] - WkM

      _cp_streaming_merge(
          dS_acc_ref,
          M_acc_ref,
          dh_tmp_ref,
          cp_buf_ref,
          cp_post_num_ref[batch_idx, 0],
          cp_post_num_ref[batch_idx, 1],
          cp_is_last_ref[batch_idx],
          cp_size=CP_SIZE,
          axis_name=CP_AXIS_NAME,
      )

    # =================================================================
    # Phase 3: Backward (dhu + WY + intra + cumsum + write)  (always)
    # =================================================================
    @functools.partial(
        pl.run_scoped,
        **backward_scratch,
        in_sems_bwd=pltpu.SemaphoreType.DMA((10, DMA_INPUT_BUFFER_COUNT)),
        packed_in_sems_bwd=pltpu.SemaphoreType.DMA((5, 2)),
        out_sems_bwd=pltpu.SemaphoreType.DMA((5, DMA_OUTPUT_BUFFER_COUNT)),
        state_sems_bwd=pltpu.SemaphoreType.DMA((5,)),
    )
    def _backward_scoped(
        q_buf,
        k_buf,
        v_buf,
        do_buf,
        beta_buf,
        packed_q_buf,
        packed_k_buf,
        packed_v_buf,
        packed_do_buf,
        packed_beta_buf,
        Aqk_buf,
        Akk_buf,
        g_buf,
        raw_gate_buf,
        h_buf,
        dq_buf,
        dk_buf,
        dv_buf,
        db_buf,
        dg_buf,
        gate_dA_acc,
        gate_dbias_acc,
        gate_A_buf,
        gate_bias_buf,
        in_sems_bwd,
        packed_in_sems_bwd,
        out_sems_bwd,
        state_sems_bwd,
    ):
      gate_dA_acc[:] = jnp.zeros_like(gate_dA_acc[:])
      gate_dbias_acc[:] = jnp.zeros_like(gate_dbias_acc[:])
      if FUSE_GATE_BWD:
        head_start = pl.multiple_of(mb_offset * MB, 8)
        _dma_copy(
            gate_A_log_ref.at[pl.ds(head_start, MB), pl.ds(None)],
            gate_A_buf,
            state_sems_bwd.at[3],
        )
        _dma_copy(
            gate_dt_bias_ref.at[pl.ds(head_start, MB), pl.ds(None)],
            gate_bias_buf,
            state_sems_bwd.at[4],
        )
      # Every positive CHUNK_BATCH uses the contiguous-window path, including CB1.
      # Slots are consumed in reverse order so the cross-chunk dh recurrence
      # remains sequential.
      LOOP_NUM = pl.cdiv(NT, CHUNK_BATCH)
      REM = NT % CHUNK_BATCH
      DB_NT = LOOP_NUM * CHUNK_BATCH
      DB_TAIL_PAD = 0 if REM == 0 else CHUNK_BATCH - REM

      def start_dmas(descs):
        for desc in descs:
          desc.start()

      def wait_dmas(descs):
        for desc in descs:
          desc.wait()

      def write_gate_parameter_grads():
        if FUSE_GATE_BWD:
          head_start = mb_offset * MB
          _dma_copy(
              gate_dA_acc,
              gate_dA_ref.at[batch_idx, pl.ds(head_start, MB), pl.ds(None)],
              state_sems_bwd.at[3],
          )
          _dma_copy(
              gate_dbias_acc,
              gate_dbias_ref.at[batch_idx, pl.ds(head_start, MB), pl.ds(None)],
              state_sems_bwd.at[4],
          )

      def dma_in_batch_desc(hbm_ref, vmem_ref, sem, buf, start_c):
        src = hbm_ref.at[
            (
                pl.ds(mb_offset * MB, MB),
                pl.ds(batch_idx, 1),
                pl.ds(start_c, CHUNK_BATCH),
                pl.ds(None),
                pl.ds(None),
            )
        ]
        return pltpu.make_async_copy(src, vmem_ref.at[buf], sem)

      def dma_in_tail_desc(hbm_ref, vmem_ref, sem, buf):
        src = hbm_ref.at[
            (
                pl.ds(mb_offset * MB, MB),
                pl.ds(batch_idx, 1),
                pl.ds(0, REM),
                pl.ds(None),
                pl.ds(None),
            )
        ]
        return pltpu.make_async_copy(
            src, vmem_ref.at[buf, :, :, :REM, :, :], sem
        )

      def dma_in_batch_beta_desc(buf, start_c):
        beta_offset = ((mb_offset * B + batch_idx) * NT + start_c) * MB_PAD
        packed_heads = CHUNK_BATCH * MB_PAD
        src = beta_ref.at[(pl.ds(beta_offset, packed_heads), pl.ds(None))]
        return pltpu.make_async_copy(
            src, beta_buf.at[buf], in_sems_bwd.at[3, buf]
        )

      def dma_out_batch_desc(vmem_ref, hbm_ref, sem, buf, start_c):
        dst = hbm_ref.at[
            (
                pl.ds(mb_offset * MB, MB),
                pl.ds(batch_idx, 1),
                pl.ds(start_c, CHUNK_BATCH),
                pl.ds(None),
                pl.ds(None),
            )
        ]
        return pltpu.make_async_copy(vmem_ref.at[buf], dst, sem)

      def dma_out_tail_desc(vmem_ref, hbm_ref, sem, buf):
        dst = hbm_ref.at[
            (
                pl.ds(mb_offset * MB, MB),
                pl.ds(batch_idx, 1),
                pl.ds(0, REM),
                pl.ds(None),
                pl.ds(None),
            )
        ]
        return pltpu.make_async_copy(
            vmem_ref.at[buf, :, :, :REM, :, :], dst, sem
        )

      def dma_out_batch_db_desc(buf, start_c):
        db_offset = (
            (mb_offset * B + batch_idx) * DB_NT + start_c + DB_TAIL_PAD
        ) * MB_PAD
        packed_heads = CHUNK_BATCH * MB_PAD
        dst = db_ref.at[(pl.ds(db_offset, packed_heads), pl.ds(None))]
        return pltpu.make_async_copy(
            db_buf.at[buf], dst, out_sems_bwd.at[3, buf]
        )

      def dma_out_tail_db_desc(buf):
        # A complete beta/db tile avoids Jellyfish's failing masked BF16
        # StridedMemcopy path. Guard chunks are removed by the wrapper.
        db_offset = ((mb_offset * B + batch_idx) * DB_NT) * MB_PAD
        packed_heads = CHUNK_BATCH * MB_PAD
        dst = db_ref.at[(pl.ds(db_offset, packed_heads), pl.ds(None))]
        return pltpu.make_async_copy(
            db_buf.at[buf], dst, out_sems_bwd.at[3, buf]
        )

      def input_batch_descs(buf, start_c):
        token_descs = (
            []
            if PACKED_INPUTS
            else [
                dma_in_batch_desc(
                    q_ref, q_buf, in_sems_bwd.at[0, buf], buf, start_c
                ),
                dma_in_batch_desc(
                    k_ref, k_buf, in_sems_bwd.at[1, buf], buf, start_c
                ),
                dma_in_batch_desc(
                    v_ref, v_buf, in_sems_bwd.at[2, buf], buf, start_c
                ),
                dma_in_batch_beta_desc(buf, start_c),
                dma_in_batch_desc(
                    do_ref, do_buf, in_sems_bwd.at[6, buf], buf, start_c
                ),
            ]
        )
        descs = token_descs + [
            dma_in_batch_desc(
                Aqk_ref, Aqk_buf, in_sems_bwd.at[4, buf], buf, start_c
            ),
            dma_in_batch_desc(
                Akk_ref, Akk_buf, in_sems_bwd.at[5, buf], buf, start_c
            ),
            dma_in_batch_desc(
                g_ref, g_buf, in_sems_bwd.at[8, buf], buf, start_c
            ),
            dma_in_batch_desc(
                raw_gate_ref, raw_gate_buf, in_sems_bwd.at[9, buf], buf, start_c
            )
            if FUSE_GATE_BWD
            else None,
            dma_in_batch_desc(
                h_ref if DISABLE_RECOMPUTE else h_all_ref,
                h_buf,
                in_sems_bwd.at[7, buf],
                buf,
                start_c,
            ),
        ]
        return [desc for desc in descs if desc is not None]

      def input_tail_descs(buf):
        # The regular operands use one REM-wide DMA. beta deliberately
        # transfers a complete CHUNK_BATCH tile into the complete buffer;
        # this is still one tail DMA and avoids the BF16 prefix-DMA crash.
        token_descs = (
            []
            if PACKED_INPUTS
            else [
                dma_in_tail_desc(q_ref, q_buf, in_sems_bwd.at[0, buf], buf),
                dma_in_tail_desc(k_ref, k_buf, in_sems_bwd.at[1, buf], buf),
                dma_in_tail_desc(v_ref, v_buf, in_sems_bwd.at[2, buf], buf),
                dma_in_batch_beta_desc(buf, 0),
                dma_in_tail_desc(do_ref, do_buf, in_sems_bwd.at[6, buf], buf),
            ]
        )
        descs = token_descs + [
            dma_in_tail_desc(Aqk_ref, Aqk_buf, in_sems_bwd.at[4, buf], buf),
            dma_in_tail_desc(Akk_ref, Akk_buf, in_sems_bwd.at[5, buf], buf),
            dma_in_tail_desc(g_ref, g_buf, in_sems_bwd.at[8, buf], buf),
            dma_in_tail_desc(
                raw_gate_ref, raw_gate_buf, in_sems_bwd.at[9, buf], buf
            )
            if FUSE_GATE_BWD
            else None,
            dma_in_tail_desc(
                h_ref if DISABLE_RECOMPUTE else h_all_ref,
                h_buf,
                in_sems_bwd.at[7, buf],
                buf,
            ),
        ]
        return [desc for desc in descs if desc is not None]

      def output_batch_descs(buf, start_c):
        return (
            dma_out_batch_desc(
                dq_buf, dq_ref, out_sems_bwd.at[0, buf], buf, start_c
            ),
            dma_out_batch_desc(
                dk_buf, dk_ref, out_sems_bwd.at[1, buf], buf, start_c
            ),
            dma_out_batch_desc(
                dv_buf, dv_ref, out_sems_bwd.at[2, buf], buf, start_c
            ),
            dma_out_batch_db_desc(buf, start_c),
            dma_out_batch_desc(
                dg_buf, dg_ref, out_sems_bwd.at[4, buf], buf, start_c
            ),
        )

      def output_tail_descs(buf):
        return (
            dma_out_tail_desc(dq_buf, dq_ref, out_sems_bwd.at[0, buf], buf),
            dma_out_tail_desc(dk_buf, dk_ref, out_sems_bwd.at[1, buf], buf),
            dma_out_tail_desc(dv_buf, dv_ref, out_sems_bwd.at[2, buf], buf),
            dma_out_tail_db_desc(buf),
            dma_out_tail_desc(dg_buf, dg_ref, out_sems_bwd.at[4, buf], buf),
        )

      def input_window_has_valid(full_start_c, is_tail):
        full_has_valid = _chunk_window_has_valid(
            chunk_seg_ids_ref,
            batch_idx,
            full_start_c,
            CHUNK_BATCH,
        )
        if REM > 0:
          tail_has_valid = _chunk_window_has_valid(
              chunk_seg_ids_ref, batch_idx, 0, REM
          )
          return jnp.where(is_tail, tail_has_valid, full_has_valid)
        return full_has_valid

      def start_input_window(buf, full_start_c, is_tail, has_valid):
        if REM > 0:

          @pl.when(jnp.logical_and(is_tail, has_valid))
          def _():
            start_dmas(input_tail_descs(buf))

          @pl.when(jnp.logical_and(jnp.logical_not(is_tail), has_valid))
          def _():
            start_dmas(input_batch_descs(buf, full_start_c))

        else:

          @pl.when(has_valid)
          def _():
            start_dmas(input_batch_descs(buf, full_start_c))

      def wait_input_window(buf, full_start_c, is_tail, has_valid):
        if REM > 0:

          @pl.when(jnp.logical_and(is_tail, has_valid))
          def _():
            wait_dmas(input_tail_descs(buf))

          @pl.when(jnp.logical_and(jnp.logical_not(is_tail), has_valid))
          def _():
            wait_dmas(input_batch_descs(buf, full_start_c))

        else:

          @pl.when(has_valid)
          def _():
            wait_dmas(input_batch_descs(buf, full_start_c))

      def start_packed_chunk(buf, chunk_id):
        if PACKED_INPUTS:
          # Canonical aligned maps contain phantom slots up to N_max.
          # Do not fetch real tokens (or walk all sequence boundaries) for
          # those slots. Start/wait use the identical immutable predicate.
          valid = (
              jnp.bool_(True)
              if ORIGINAL_TOKEN_SEGMENTS
              else chunk_seg_ids_ref[batch_idx, chunk_id] != 0
          )

          @pl.when(valid)
          def _start_valid_packed_chunk():
            start_dmas(
                list(
                    _packed_token_dma_descs(
                        buf,
                        mb_offset,
                        batch_idx,
                        chunk_id,
                        packed_q_buf,
                        packed_k_buf,
                        packed_v_buf,
                        packed_do_buf,
                        packed_beta_buf,
                        packed_in_sems_bwd,
                    )
                )
            )

      def wait_packed_chunk(buf, chunk_id):
        if PACKED_INPUTS:
          valid = (
              jnp.bool_(True)
              if ORIGINAL_TOKEN_SEGMENTS
              else chunk_seg_ids_ref[batch_idx, chunk_id] != 0
          )

          @pl.when(valid)
          def _wait_valid_packed_chunk():
            wait_dmas(
                list(
                    _packed_token_dma_descs(
                        buf,
                        mb_offset,
                        batch_idx,
                        chunk_id,
                        packed_q_buf,
                        packed_k_buf,
                        packed_v_buf,
                        packed_do_buf,
                        packed_beta_buf,
                        packed_in_sems_bwd,
                    )
                )
            )

      def packed_chunk_operands(buf, chunk_id):
        (
            _,
            source_shift,
            _,
            beta_source_shift,
            packed_offset,
            beta_group_offset,
            remaining,
        ) = _packed_input_window(batch_idx, chunk_id)
        _align_packed_tail_window(
            buf,
            source_shift,
            packed_q_buf,
            packed_k_buf,
            packed_v_buf,
            packed_do_buf,
        )
        pq = _packed_token_view(packed_q_buf, buf, packed_offset)
        pk = _packed_token_view(packed_k_buf, buf, packed_offset)
        pv = _packed_token_view(packed_v_buf, buf, packed_offset)
        pdo = _packed_token_view(packed_do_buf, buf, packed_offset)
        pbeta = _packed_beta_view(
            packed_beta_buf,
            buf,
            beta_source_shift,
            beta_group_offset,
            packed_offset,
        )

        pq_valid = jax.lax.broadcasted_iota(jnp.int32, pq.shape, 1) < remaining
        pk_valid = jax.lax.broadcasted_iota(jnp.int32, pk.shape, 1) < remaining
        pv_valid = jax.lax.broadcasted_iota(jnp.int32, pv.shape, 1) < remaining
        pdo_valid = (
            jax.lax.broadcasted_iota(jnp.int32, pdo.shape, 1) < remaining
        )
        pbeta_valid = (
            jax.lax.broadcasted_iota(jnp.int32, pbeta.shape, 1) < remaining
        )
        pq = jnp.where(pq_valid, pq, jnp.zeros_like(pq))
        pk = jnp.where(pk_valid, pk, jnp.zeros_like(pk))
        pv = jnp.where(pv_valid, pv, jnp.zeros_like(pv))
        pdo = jnp.where(pdo_valid, pdo, jnp.zeros_like(pdo))
        pbeta = jnp.where(pbeta_valid, pbeta, jnp.zeros_like(pbeta))
        return pq, pk, pv, pbeta, pdo

      def compute_chunk(
          q_chunk_ref,
          k_chunk_ref,
          v_chunk_ref,
          beta_chunk_ref,
          Aqk_chunk_ref,
          Akk_chunk_ref,
          g_chunk_ref,
          raw_gate_chunk_ref,
          h_chunk_ref,
          do_chunk_ref,
          dq_chunk_ref,
          dk_chunk_ref,
          dv_chunk_ref,
          db_chunk_ref,
          dg_chunk_ref,
          chunk_id,
      ):
        with jax.named_scope("_chunk_segment_metadata"):
          _seg_cur, seq_idx, is_valid, is_first_chunk, is_last_chunk = (
              _chunk_segment_metadata(
                  chunk_seg_ids_ref, batch_idx, chunk_id, NT
              )
          )

        if CP_ACTIVE:
          # Only the final run carries the downstream state gradient. Earlier
          # runs must reset even when their label matches that final run.
          @pl.when(is_last_chunk & (chunk_id != last_valid_chunk))
          def _load_dht_cp():
            if HAS_DHT:
              _dma_state_in(
                  dht_ref,
                  dh_tmp_ref,
                  state_sems_bwd.at[1],
                  mb_offset,
                  batch_idx,
                  seq_idx,
              )
            else:
              dh_tmp_ref[:] = jnp.zeros_like(dh_tmp_ref[:])

        else:

          @pl.when(is_last_chunk)
          def load_dht():
            if HAS_DHT:
              with jax.named_scope("_dma_state_in"):
                _dma_state_in(
                    dht_ref,
                    dh_tmp_ref,
                    state_sems_bwd.at[1],
                    mb_offset,
                    batch_idx,
                    seq_idx,
                )
            elif not ORIGINAL_TOKEN_SEGMENTS:
              # Aligned chunks use the first-token segment ID for their edge
              # metadata, so a new reverse segment must reset the carried
              # state here. Compact chunks may contain a second fragment that
              # continues into the next chunk; their exact last-token edge is
              # applied below by ``continues_to_next`` instead.
              dh_tmp_ref[:] = jnp.zeros_like(dh_tmp_ref[:])

        @pl.when(jnp.logical_not(is_valid))
        def _zero_invalid_slot():
          dq_chunk_ref[:] = jnp.zeros_like(dq_chunk_ref[:])
          dk_chunk_ref[:] = jnp.zeros_like(dk_chunk_ref[:])
          dv_chunk_ref[:] = jnp.zeros_like(dv_chunk_ref[:])
          db_chunk_ref[:] = jnp.zeros_like(db_chunk_ref[:])
          dg_chunk_ref[:] = jnp.zeros_like(dg_chunk_ref[:])

        if ORIGINAL_TOKEN_SEGMENTS:
          compact_split = chunk_metadata_ref[batch_idx, 0, chunk_id]
          compact_valid_end = chunk_metadata_ref[batch_idx, 1, chunk_id]
          needs_segment_math = (compact_split < compact_valid_end) | (
              compact_valid_end < BT
          )

        def _compute_valid_chunk(segment_math, full_segment_metadata=False):
          with jax.named_scope("mega_compute_prepare"):
            bq = q_chunk_ref[:].astype(compute_dtype)
            bk = k_chunk_ref[:].astype(compute_dtype)
            bv = v_chunk_ref[:].astype(compute_dtype)
            bg = g_chunk_ref[:].astype(jnp.float32)
            bb = (
                beta_chunk_ref[:].astype(compute_dtype)
                if PACKED_INPUTS
                else beta_chunk_ref[pl.ds(0, MB), :BT].astype(compute_dtype)
            )
            bA = Akk_chunk_ref[:, :, :BT].astype(compute_dtype)
            bdo = do_chunk_ref[:].astype(compute_dtype)
            bh = h_chunk_ref[:].astype(jnp.float32)
            if ORIGINAL_TOKEN_SEGMENTS:
              # The experimental compact contract allows at most two
              # fragments per chunk. Reconstruct token IDs from the
              # scalar-prefetched boundary metadata instead of DMAing a
              # Bx256 int32 window for every chunk/head group.
              first_segment_id = chunk_seg_ids_ref[batch_idx, chunk_id]
              split = compact_split
              valid_end = compact_valid_end
              last_segment_id = chunk_metadata_ref[batch_idx, 2, chunk_id]
              next_chunk = jnp.minimum(chunk_id + 1, NT - 1)
              next_segment_id = jnp.where(
                  chunk_id + 1 < NT,
                  chunk_seg_ids_ref[batch_idx, next_chunk],
                  jnp.int32(0),
              )
              if segment_math:
                token_idx = jnp.arange(BT, dtype=jnp.int32)
                if full_segment_metadata:
                  segment_tile = original_segment_ids_ref[
                      batch_idx, chunk_id, :
                  ].astype(jnp.int32)
                else:
                  segment_tile = jnp.where(
                      token_idx < split,
                      first_segment_id,
                      jnp.where(
                          token_idx < valid_end, last_segment_id, jnp.int32(0)
                      ),
                  )
                segment_rows = jnp.broadcast_to(
                    segment_tile[None, :], bq.shape[:2]
                )
              else:
                segment_tile = None
                segment_rows = None
            else:
              segment_tile = None
              segment_rows = None
            if segment_math:
              terminal_g = segmented_suffix_last(bg, segment_rows)
              g_exp_last = jnp.exp2(terminal_g[:, 0, :])
              valid_rows = segment_rows != 0
              first_fragment = valid_rows & (
                  segment_rows == segment_rows[:, :1]
              )
              last_fragment = valid_rows & (
                  segment_rows == segment_rows[:, -1:]
              )
            else:
              g_exp_last = jnp.exp2(bg[:, BT - 1, :])

          with jax.named_scope("mega_compute_m1"):
            bAqk = Aqk_chunk_ref[:, :, :BT].astype(compute_dtype)
            _, w, bvn, bqg, bkg = compute_m1_recompute(
                bq,
                bk,
                bv,
                bb,
                bA,
                bg,
                bh,
                dtype=compute_dtype,
                compute_dtype=compute_dtype,
                segment_ids=segment_rows if segment_math else None,
            )
            bw = w.astype(jnp.float32)
          with jax.named_scope("mega_compute_dav"):
            bdAqk, bdv0 = compute_dav(
                bdo, bvn, bAqk, SCALE, segment_tile if segment_math else None
            )
          del bAqk, w

          # Finish the recurrence before WY/intra so bw/bqg do not
          # remain live across the largest contractions.
          with jax.named_scope("mega_compute_state_dv"):
            dh = dh_tmp_ref[:].astype(jnp.float32)
            if ORIGINAL_TOKEN_SEGMENTS:
              continues_to_next = (last_segment_id != 0) & (
                  last_segment_id == next_segment_id
              )
              dh = jnp.where(continues_to_next, dh, jnp.zeros_like(dh))
              if segment_math:
                bkg = bkg * last_fragment.astype(bkg.dtype)[..., None]
            bdv = (
                jnp.matmul(
                    bkg.astype(jnp.float32),
                    dh,
                    preferred_element_type=jnp.float32,
                )
                + bdv0
            )
          with jax.named_scope("mega_compute_dh"):
            dh_next = dh * g_exp_last[:, :, None]
            if ORIGINAL_TOKEN_SEGMENTS and segment_math:
              same_edge_segment = (segment_tile[0] != 0) & (
                  segment_tile[0] == segment_tile[BT - 1]
              )
              dh_next = jnp.where(
                  same_edge_segment, dh_next, jnp.zeros_like(dh_next)
              )
              bqg = bqg * first_fragment.astype(bqg.dtype)[..., None]
            dh_next = dh_next + jnp.matmul(
                jnp.concatenate([bqg * SCALE, -bw], axis=1)
                .astype(jnp.float32)
                .transpose(0, 2, 1),
                jnp.concatenate([bdo.astype(jnp.float32), bdv], axis=1),
                preferred_element_type=jnp.float32,
            )
            dh_tmp_ref[:] = dh_next
          del bw, bqg, bkg, bdv0, g_exp_last, dh_next

          with jax.named_scope("mega_compute_wy"):
            (
                dq_acc,
                dk_acc,
                b_dvb,
                db_acc,
                dg_acc,
                dAkk_local,
                b_dgk,
            ) = compute_wy_backward(
                bdo,
                bdv,
                bvn,
                bv,
                bh,
                dh,
                bq,
                bk,
                bg,
                bb,
                bA,
                SCALE,
                precision,
                return_b_dgk=True,
                segment_ids=segment_rows if segment_math else None,
            )
          del bdo, bdv, bvn, bv, bh, dh, bA

          # Consume dV immediately instead of carrying b_dvb through
          # the intra contraction and reverse cumsum.
          with jax.named_scope("mega_compute_output"):
            dv_chunk_ref[:] = (b_dvb * bb[:, :, None]).astype(
                dv_chunk_ref.dtype
            )
          del b_dvb

          with jax.named_scope("mega_compute_intra"):
            db_chunk_ref[:] = jnp.zeros_like(db_chunk_ref[:])
            # Only native all-gather CP preserves the FP32 intra cotangent.
            # External all-to-all callers have CP_ACTIVE=False and retain
            # the original ref-store/reload rounding.
            direct_fp32_dg = CP_ACTIVE and dg_chunk_ref.dtype != jnp.float32
            dg_total = compute_intra_backward_sequential_to_chunk_refs(
                bq,
                bk,
                bg,
                bb,
                bdAqk,
                dAkk_local,
                dq_acc,
                dk_acc,
                db_acc,
                dg_acc,
                dq_chunk_ref,
                dk_chunk_ref,
                db_chunk_ref,
                dg_chunk_ref,
                operand_dtype=q_ref.dtype,
                b_dgk=b_dgk,
                return_dg=direct_fp32_dg,
            )
          del dq_acc, dk_acc, db_acc, dg_acc, dAkk_local
          del bq, bk, bg, bb, bdAqk, b_dgk

          with jax.named_scope("mega_compute_cumsum"):
            if not direct_fp32_dg:
              dg_total = dg_chunk_ref[:].astype(jnp.float32)
            dg_reverse_cumsum = (
                segmented_reverse_sum(
                    dg_total,
                    jnp.broadcast_to(segment_tile[None, :], dg_total.shape[:2]),
                )
                if segment_math
                else compute_reverse_cumsum_dg_scan(dg_total)
            )
          with jax.named_scope("mega_compute_output"):
            if FUSE_GATE_BWD:
              head_start = mb_offset * MB
              raw_gate = raw_gate_chunk_ref[:].astype(jnp.float32)
              gate_bias = gate_bias_buf[:].reshape(MB, 1, K)
              biased_gate = raw_gate + gate_bias
              gate_A = jnp.exp(gate_A_buf[:, 0])[:, None, None]
              gate_sigmoid = jax.nn.sigmoid(gate_A * biased_gate)
              dg_final = (
                  dg_reverse_cumsum
                  * (-5.0 * gate_sigmoid * (1.0 - gate_sigmoid))
                  * gate_A
              )
              gate_dA_acc[:] += jnp.sum(dg_final * biased_gate, axis=(1, 2))[
                  :, None
              ]
              gate_dbias_acc[:] += jnp.sum(dg_final, axis=1)
              dg_chunk_ref[:] = dg_final.astype(dg_chunk_ref.dtype)
            else:
              dg_chunk_ref[:] = dg_reverse_cumsum.astype(dg_chunk_ref.dtype)

        if ORIGINAL_TOKEN_SEGMENTS:

          general_segment_math = chunk_metadata_ref[batch_idx, 3, chunk_id] > 2

          @pl.when(is_valid & needs_segment_math & general_segment_math)
          def _compute_general_boundary_chunk():
            _compute_valid_chunk(True, True)

          @pl.when(is_valid & needs_segment_math & ~general_segment_math)
          def _compute_boundary_chunk():
            _compute_valid_chunk(True)

          @pl.when(is_valid & ~needs_segment_math)
          def _compute_full_chunk():
            _compute_valid_chunk(False)

        else:

          @pl.when(is_valid)
          def _compute_aligned_chunk():
            _compute_valid_chunk(False)

        if RETURN_DH0:

          @pl.when(is_first_chunk)
          def write_dh0():
            with jax.named_scope("_dma_state_out"):
              _dma_state_out(
                  dh_tmp_ref,
                  dh0_ref,
                  state_sems_bwd.at[2],
                  mb_offset,
                  batch_idx,
                  seq_idx,
              )

      assert DMA_INPUT_BUFFER_COUNT == 2 and DMA_OUTPUT_BUFFER_COUNT == 2

      if PACKED_INPUTS:
        # The packed wrapper selects a divisor of NT, so every DMA window is
        # complete. Keep one loop over chunks and express CB window ownership
        # with boundary predicates; this preserves batched residual/output
        # DMA without nesting a per-window chunk loop around the packed
        # selector tree.
        assert REM == 0
        WINDOW_NUM = NT // CHUNK_BATCH
        first_window_start = NT - CHUNK_BATCH
        first_window_has_valid = _chunk_window_has_valid(
            chunk_seg_ids_ref,
            batch_idx,
            first_window_start,
            CHUNK_BATCH,
        )
        with jax.named_scope("mega_dma_input_start"):
          start_input_window(
              0,
              first_window_start,
              False,
              first_window_has_valid,
          )
        start_packed_chunk(0, NT - 1)

        @pl.loop(0, NT, unroll=False)
        def packed_flat_loop(rev_idx):
          chunk_id = NT - 1 - rev_idx
          window_idx = rev_idx // CHUNK_BATCH
          rev_slot = rev_idx % CHUNK_BATCH
          slot = CHUNK_BATCH - 1 - rev_slot
          input_buf = window_idx % DMA_INPUT_BUFFER_COUNT
          next_input_buf = 1 - input_buf
          output_buf = window_idx % DMA_OUTPUT_BUFFER_COUNT
          packed_buf = rev_idx % 2
          window_start = NT - (window_idx + 1) * CHUNK_BATCH
          is_window_start = rev_slot == 0
          is_window_end = slot == 0
          window_has_valid = _chunk_window_has_valid(
              chunk_seg_ids_ref,
              batch_idx,
              window_start,
              CHUNK_BATCH,
          )

          @pl.when(is_window_start & window_has_valid)
          def _wait_current_window():
            with jax.named_scope("mega_dma_input_wait"):
              wait_dmas(input_batch_descs(input_buf, window_start))

          has_next_window = window_idx + 1 < WINDOW_NUM
          next_window_start = jnp.maximum(window_start - CHUNK_BATCH, 0)
          next_window_has_valid = _chunk_window_has_valid(
              chunk_seg_ids_ref,
              batch_idx,
              next_window_start,
              CHUNK_BATCH,
          )

          @pl.when(is_window_start & has_next_window & next_window_has_valid)
          def _prefetch_next_window():
            with jax.named_scope("mega_dma_input_prefetch"):
              start_dmas(input_batch_descs(next_input_buf, next_window_start))

          @pl.when(is_window_start & (window_idx >= DMA_OUTPUT_BUFFER_COUNT))
          def _wait_reused_output():
            previous_window = window_idx - DMA_OUTPUT_BUFFER_COUNT
            previous_start_c = NT - (previous_window + 1) * CHUNK_BATCH
            with jax.named_scope("mega_dma_output_wait"):
              wait_dmas(output_batch_descs(output_buf, previous_start_c))

          wait_packed_chunk(packed_buf, chunk_id)

          @pl.when(rev_idx + 1 < NT)
          def _prefetch_next_packed_chunk():
            start_packed_chunk(1 - packed_buf, chunk_id - 1)

          valid = (
              jnp.bool_(True)
              if ORIGINAL_TOKEN_SEGMENTS
              else chunk_seg_ids_ref[batch_idx, chunk_id] != 0
          )

          @pl.when(valid)
          def _compute_valid_packed_slot():
            q_chunk, k_chunk, v_chunk, beta_chunk, do_chunk = (
                packed_chunk_operands(packed_buf, chunk_id)
            )
            compute_chunk(
                q_chunk,
                k_chunk,
                v_chunk,
                beta_chunk,
                Aqk_buf.at[input_buf, :, 0, slot, :, :],
                Akk_buf.at[input_buf, :, 0, slot, :, :],
                g_buf.at[input_buf, :, 0, slot, :, :],
                raw_gate_buf.at[input_buf, :, 0, slot, :, :]
                if FUSE_GATE_BWD
                else g_buf.at[input_buf, :, 0, slot, :, :],
                h_buf.at[input_buf, :, 0, slot, :, :],
                do_chunk,
                dq_buf.at[output_buf, :, 0, slot, :, :],
                dk_buf.at[output_buf, :, 0, slot, :, :],
                dv_buf.at[output_buf, :, 0, slot, :, :],
                db_buf.at[output_buf, pl.ds(slot * MB_PAD, MB_PAD), :],
                dg_buf.at[output_buf, :, 0, slot, :, :],
                chunk_id,
            )

          if not ORIGINAL_TOKEN_SEGMENTS:

            @pl.when(~valid)
            def _zero_phantom_outputs():
              # No input read, state update or DMA wait for a phantom slot.
              # Its output window is still written exactly once as before.
              for operand in (dq_buf, dk_buf, dv_buf, dg_buf):
                operand[output_buf, :, 0, slot, :, :] = jnp.zeros_like(
                    operand[output_buf, :, 0, slot, :, :]
                )
              db_buf[output_buf, pl.ds(slot * MB_PAD, MB_PAD), :] = (
                  jnp.zeros_like(
                      db_buf[output_buf, pl.ds(slot * MB_PAD, MB_PAD), :]
                  )
              )

          @pl.when(is_window_end)
          def _launch_window_output():
            with jax.named_scope("mega_dma_output_start"):
              start_dmas(output_batch_descs(output_buf, window_start))

        # Both output slots may still own an in-flight DMA after the loop.
        last_window = WINDOW_NUM - 1
        with jax.named_scope("mega_dma_output_drain"):
          if WINDOW_NUM > 1:
            previous_window = WINDOW_NUM - 2
            previous_buf = previous_window % DMA_OUTPUT_BUFFER_COUNT
            previous_start_c = NT - (previous_window + 1) * CHUNK_BATCH
            wait_dmas(output_batch_descs(previous_buf, previous_start_c))
          last_buf = last_window % DMA_OUTPUT_BUFFER_COUNT
          last_start_c = NT - (last_window + 1) * CHUNK_BATCH
          wait_dmas(output_batch_descs(last_buf, last_start_c))
        write_gate_parameter_grads()
        return

      initial_is_tail = REM > 0 and LOOP_NUM == 1
      initial_start_c = 0 if initial_is_tail else NT - CHUNK_BATCH
      initial_has_valid = input_window_has_valid(
          initial_start_c, initial_is_tail
      )
      with jax.named_scope("mega_dma_input_start"):
        start_input_window(
            0, initial_start_c, initial_is_tail, initial_has_valid
        )

      # input ping-pong and output ping
      @pl.loop(0, LOOP_NUM, unroll=False)
      def async_loop(loop_idx):
        input_buf = loop_idx % 2
        next_input_buf = 1 - input_buf
        output_buf = loop_idx % 2
        full_start_c = NT - (loop_idx + 1) * CHUNK_BATCH
        is_tail = loop_idx == LOOP_NUM - 1
        if REM > 0:
          safe_full_start_c = jnp.where(is_tail, 0, full_start_c)
        else:
          is_tail = False
          safe_full_start_c = full_start_c

        has_valid = input_window_has_valid(safe_full_start_c, is_tail)
        with jax.named_scope("mega_dma_input_wait"):
          wait_input_window(input_buf, safe_full_start_c, is_tail, has_valid)

        @pl.when(loop_idx + 1 < LOOP_NUM)
        def prefetch_next_window():
          next_full_start_c = NT - (loop_idx + 2) * CHUNK_BATCH
          if REM > 0:
            next_is_tail = loop_idx + 1 == LOOP_NUM - 1
            next_safe_start_c = jnp.where(next_is_tail, 0, next_full_start_c)
          else:
            next_is_tail = False
            next_safe_start_c = next_full_start_c
          next_has_valid = input_window_has_valid(
              next_safe_start_c, next_is_tail
          )
          with jax.named_scope("mega_dma_input_prefetch"):
            start_input_window(
                next_input_buf,
                next_safe_start_c,
                next_is_tail,
                next_has_valid,
            )

        # Output(i - 2) owns the same ping-pong slot as output(i). Wait
        # only when that slot is about to be reused; output(i - 1) remains
        # in flight in the other slot while this window computes.
        @pl.when(loop_idx >= DMA_OUTPUT_BUFFER_COUNT)
        def _wait_reused_output():
          previous_window = loop_idx - DMA_OUTPUT_BUFFER_COUNT
          previous_start_c = NT - (previous_window + 1) * CHUNK_BATCH
          with jax.named_scope("mega_dma_output_wait"):
            wait_dmas(output_batch_descs(output_buf, previous_start_c))

        if PACKED_INPUTS:
          first_slot = (
              jnp.where(is_tail, jnp.int32(REM - 1), jnp.int32(CHUNK_BATCH - 1))
              if REM > 0
              else jnp.int32(CHUNK_BATCH - 1)
          )
          first_chunk = jnp.where(
              is_tail, first_slot, safe_full_start_c + first_slot
          )
          start_packed_chunk(0, first_chunk)

        # =================== compute start ====================================
        @pl.when(jnp.logical_not(is_tail))
        def _process_full_window():
          @pl.loop(0, CHUNK_BATCH, unroll=False)
          def _process_full_slot(rev_slot):
            slot = CHUNK_BATCH - 1 - rev_slot
            chunk_id = safe_full_start_c + slot
            if PACKED_INPUTS:
              packed_buf = rev_slot % 2
              wait_packed_chunk(packed_buf, chunk_id)

              @pl.when(rev_slot + 1 < CHUNK_BATCH)
              def _prefetch_next_packed_chunk():
                start_packed_chunk(1 - packed_buf, chunk_id - 1)

              q_chunk, k_chunk, v_chunk, beta_chunk, do_chunk = (
                  packed_chunk_operands(packed_buf, chunk_id)
              )
            else:
              q_chunk = q_buf.at[input_buf, :, 0, slot, :, :]
              k_chunk = k_buf.at[input_buf, :, 0, slot, :, :]
              v_chunk = v_buf.at[input_buf, :, 0, slot, :, :]
              beta_chunk = beta_buf.at[
                  input_buf, pl.ds(slot * MB_PAD, MB_PAD), :
              ]
              do_chunk = do_buf.at[input_buf, :, 0, slot, :, :]
            compute_chunk(
                q_chunk,
                k_chunk,
                v_chunk,
                beta_chunk,
                Aqk_buf.at[input_buf, :, 0, slot, :, :],
                Akk_buf.at[input_buf, :, 0, slot, :, :],
                g_buf.at[input_buf, :, 0, slot, :, :],
                raw_gate_buf.at[input_buf, :, 0, slot, :, :]
                if FUSE_GATE_BWD
                else g_buf.at[input_buf, :, 0, slot, :, :],
                h_buf.at[input_buf, :, 0, slot, :, :],
                do_chunk,
                dq_buf.at[output_buf, :, 0, slot, :, :],
                dk_buf.at[output_buf, :, 0, slot, :, :],
                dv_buf.at[output_buf, :, 0, slot, :, :],
                db_buf.at[output_buf, pl.ds(slot * MB_PAD, MB_PAD), :],
                dg_buf.at[output_buf, :, 0, slot, :, :],
                chunk_id,
            )

        if REM > 0:

          @pl.when(is_tail)
          def _process_tail_window():
            @pl.loop(0, REM, unroll=False)
            def _process_tail_slot(rev_slot):
              slot = REM - 1 - rev_slot
              if PACKED_INPUTS:
                packed_buf = rev_slot % 2
                wait_packed_chunk(packed_buf, slot)

                @pl.when(rev_slot + 1 < REM)
                def _prefetch_next_packed_chunk():
                  start_packed_chunk(1 - packed_buf, slot - 1)

                q_chunk, k_chunk, v_chunk, beta_chunk, do_chunk = (
                    packed_chunk_operands(packed_buf, slot)
                )
              else:
                q_chunk = q_buf.at[input_buf, :, 0, slot, :, :]
                k_chunk = k_buf.at[input_buf, :, 0, slot, :, :]
                v_chunk = v_buf.at[input_buf, :, 0, slot, :, :]
                beta_chunk = beta_buf.at[
                    input_buf, pl.ds(slot * MB_PAD, MB_PAD), :
                ]
                do_chunk = do_buf.at[input_buf, :, 0, slot, :, :]
              compute_chunk(
                  q_chunk,
                  k_chunk,
                  v_chunk,
                  beta_chunk,
                  Aqk_buf.at[input_buf, :, 0, slot, :, :],
                  Akk_buf.at[input_buf, :, 0, slot, :, :],
                  g_buf.at[input_buf, :, 0, slot, :, :],
                  raw_gate_buf.at[input_buf, :, 0, slot, :, :]
                  if FUSE_GATE_BWD
                  else g_buf.at[input_buf, :, 0, slot, :, :],
                  h_buf.at[input_buf, :, 0, slot, :, :],
                  do_chunk,
                  dq_buf.at[output_buf, :, 0, slot, :, :],
                  dk_buf.at[output_buf, :, 0, slot, :, :],
                  dv_buf.at[output_buf, :, 0, slot, :, :],
                  db_buf.at[output_buf, pl.ds(slot * MB_PAD, MB_PAD), :],
                  dg_buf.at[output_buf, :, 0, slot, :, :],
                  slot,
              )

        # =================== compute end ====================================

        @pl.when(is_tail)
        def _launch_tail_output():
          with jax.named_scope("mega_dma_output_start"):
            start_dmas(output_tail_descs(output_buf))

        @pl.when(jnp.logical_not(is_tail))
        def _launch_full_output():
          with jax.named_scope("mega_dma_output_start"):
            start_dmas(output_batch_descs(output_buf, safe_full_start_c))

      # Both output slots may still own an in-flight DMA after the loop.
      # Drain the older (penultimate) window first, then the final window.
      last_window = LOOP_NUM - 1
      with jax.named_scope("mega_dma_output_drain"):
        if LOOP_NUM > 1:
          previous_window = LOOP_NUM - 2
          previous_buf = previous_window % DMA_OUTPUT_BUFFER_COUNT
          previous_start_c = NT - (previous_window + 1) * CHUNK_BATCH
          wait_dmas(output_batch_descs(previous_buf, previous_start_c))
        last_buf = last_window % DMA_OUTPUT_BUFFER_COUNT
        if REM > 0:
          wait_dmas(output_tail_descs(last_buf))
        else:
          last_start_c = NT - (last_window + 1) * CHUNK_BATCH
          wait_dmas(output_batch_descs(last_buf, last_start_c))
      write_gate_parameter_grads()
      return

  # Execute the body in batch-major, head-group-minor order. Scratch refs
  # live for exactly one (batch, head-group) iteration.
  @pl.loop(0, B, unroll=False)
  def batch_loop(batch_idx):
    @pl.loop(0, H // MB, unroll=False)
    def head_group_loop(mb_offset):
      if CP_ACTIVE:

        @functools.partial(pl.run_scoped, **state_scratch)
        def _cp_head_group_scoped(
            dh_tmp_ref,
            dS_acc_ref,
            M_acc_ref,
            cp_buf_ref,
        ):
          main_loop_body(
              batch_idx,
              mb_offset,
              dh_tmp_ref,
              dS_acc_ref,
              M_acc_ref,
              cp_buf_ref,
          )

      else:

        @functools.partial(pl.run_scoped, **state_scratch)
        def _non_cp_head_group_scoped(dh_tmp_ref):
          main_loop_body(batch_idx, mb_offset, dh_tmp_ref)


@partial(
    jax.jit,
    static_argnames=[
        "chunk_size",
        "use_exp2",
        "lower_bound",
        "scale",
        "mini_batch",
        "chunk_batch",
        "return_dh0",
        "N_MAX",
        "disable_recompute",
        "cp_active",
        "has_initial_state",
        "cp_size",
        "cp_axis_name",
        "cp_context",
        "batch_first_inputs",
        "cast_dg_to_q_dtype",
        "cost_estimate_flops",
        "original_token_segments",
        "fuse_gate_bwd",
    ],
)
def chunk_kda_bwd_fusion(
    q: Float[Array, "H B T K"],
    k: Float[Array, "H B T K"],
    v: Float[Array, "H B T V"],
    beta: Float[Array, "H B T"],
    Aqk: Float[Array, "H B T BT_ALIGN"],
    Akk: Float[Array, "H B T BT_ALIGN"],
    g: Float[Array, "H B T K"],
    h: Float[Array, "H B NT K V"] | None,
    do: Float[Array, "H B T V"],
    dht: Float[Array, "B N_MAX H K V"] | None,
    initial_state: Float[Array, "B N_MAX H K V"] | None,
    scale: float,
    *,
    segment_ids: Int[Array, "B T"] | None = None,
    cu_seqlens: Int[Array, "B N_PLUS_1"] | None = None,
    chunk_indices: Int[Array, "B NT 2"] | None = None,
    chunk_size: int = 64,
    use_exp2: bool = True,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    mini_batch: int | None = None,
    chunk_batch: int = KDA_DMA_CHUNK_BATCH,
    return_dh0: bool = True,
    disable_recompute: bool = True,
    cp_active: bool = False,
    has_initial_state: bool = False,
    cp_context: CPContext | None = None,
    cp_size: int | None = None,
    cp_axis_name: str | None = None,
    N_MAX: int | None = None,
    batch_first_inputs: bool = False,
    cast_dg_to_q_dtype: bool = False,
    cost_estimate_flops: int = -1,
    original_token_segment_ids: Int[Array, "B T_INPUT"] | None = None,
    original_token_segments: bool = False,
    raw_gate: Float[Array, "B T H K"] | None = None,
    gate_A_log: Float[Array, "H"] | None = None,
    gate_dt_bias: Float[Array, "HK"] | None = None,
    fuse_gate_bwd: bool = False,
) -> tuple[
    Float[Array, "H B T K"],  # q
    Float[Array, "H B T K"],  # k
    Float[Array, "H B T V"],  # v
    Float[Array, "H B T"],  # beta
    Float[Array, "H B T K"],  # g
    Float[Array, "B N_MAX H K V"] | None,  # dh0
]:
  """M5 mega kernel: fuses M1+dAv+M4 (all phases) into a single Pallas call.

  All branch decisions (``disable_recompute`` and ``cp_active``) are resolved
  at trace time via ``static_argnames`` — the kernel body is a single source
  of truth for every supported path. Intra contractions always preserve the
  BF16 RHS low bits.

  Same return signature as ``_fused_dhu_wy_intra_cumsum_pallas_jit``.
  Eliminates HBM round-trips for w, qg, kg, v_new, dAqk, dv0.
  ``K`` and ``V`` must be multiples of 128. ``Aqk``/``Akk`` may expose
  either the logical ``BT`` width or its 128-aligned physical width; only
  legacy logical-width inputs are padded before the Pallas call.
  """
  if batch_first_inputs:
    B, T_INPUT, H, K = q.shape
  else:
    H, B, T_INPUT, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  T = Aqk.shape[2]
  NT = T // BT
  BT_pad = pl.cdiv(BT, 128) * 128
  BETA_pad = 128
  # Packed custom-VJP callers can request the final public gate dtype here.
  # Casting in the kernel halves dg write/compaction traffic for BF16 inputs.
  dg_out_dtype = q.dtype if cast_dg_to_q_dtype else g.dtype

  assert chunk_size == 64, f"only chunk_size=64 is supported, got {chunk_size}"
  packed_inputs = cu_seqlens is not None and batch_first_inputs
  assert (
      not batch_first_inputs or packed_inputs
  ), "batch-first token inputs are supported only by native packed backward"
  assert T % BT == 0, f"residual T={T} must be divisible by chunk_size={BT}"
  assert (
      not packed_inputs
  ) or q.dtype == jnp.bfloat16, (
      "native packed backward currently supports BF16 token inputs"
  )
  assert (
      not packed_inputs
  ) or T_INPUT % 8 == 0, (
      f"native packed backward output T={T_INPUT} must be divisible by 8"
  )
  assert use_exp2 is True, "only use_exp2=True is supported"
  assert K % 128 == 0, f"K={K} must be a multiple of 128"
  assert V % 128 == 0, f"V={V} must be a multiple of 128"
  assert Aqk.shape[-1] in (
      BT,
      BT_pad,
  ), f"Aqk last dimension must be {BT} or {BT_pad}, got {Aqk.shape[-1]}"
  assert Akk.shape[-1] in (
      BT,
      BT_pad,
  ), f"Akk last dimension must be {BT} or {BT_pad}, got {Akk.shape[-1]}"
  assert (disable_recompute is False) or (
      h is not None
  ), "h is required when disable_recompute=True"
  assert (cp_active is False) or (
      cp_context is not None
      and cp_size is not None
      and cp_size > 0
      and cp_axis_name is not None
  )
  assert (segment_ids is None) or (
      N_MAX is not None
  ), "N_MAX must be provided for varlen input"
  assert (
      dht is None or dht.ndim == 5
  ), f"dht must be 5D [B, N_MAX, H, K, V], got shape {dht.shape}"
  assert (
      initial_state is None or initial_state.ndim == 5
  ), f"initial_state must be 5D [B, N_MAX, H, K, V], got shape {initial_state.shape}"
  assert original_token_segments == (
      original_token_segment_ids is not None
  ), "original_token_segments and original_token_segment_ids must be enabled together"
  assert (
      not original_token_segments
  ) or packed_inputs, (
      "original-token segment metadata requires native packed inputs"
  )
  assert not fuse_gate_bwd or (
      original_token_segments
      and raw_gate is not None
      and gate_A_log is not None
      and gate_dt_bias is not None
      and lower_bound == -5.0
  ), "fused gate backward requires compact production lower-bound inputs"
  if original_token_segments:
    assert original_token_segment_ids.shape == (B, T_INPUT), (
        "original_token_segment_ids must match packed inputs: "
        f"expected {(B, T_INPUT)}, got {original_token_segment_ids.shape}"
    )
    assert T == T_INPUT, (
        "compact residual and input token domains must match: "
        f"T={T}, T_INPUT={T_INPUT}"
    )

  vmem_limit_bytes = min(
      get_tpu_config().vmem_limit_bytes, KDA_BWD_SCOPED_VMEM_LIMIT_BYTES
  )
  MB, chunk_batch = _select_bwd_mega_tiling(
      B=B,
      H=H,
      K=K,
      V=V,
      NT=NT,
      BT=BT,
      dtypes={
          "q": q.dtype,
          "k": k.dtype,
          "v": v.dtype,
          "do": do.dtype,
          "beta": beta.dtype,
          "Aqk": Aqk.dtype,
          "Akk": Akk.dtype,
          "g": g.dtype,
          "h": h.dtype if h is not None else jnp.dtype(jnp.float32),
          "dq": q.dtype,
          "dk": k.dtype,
          "dv": v.dtype,
          "db": beta.dtype,
          "dg": dg_out_dtype,
          "raw_gate": raw_gate.dtype if raw_gate is not None else q.dtype,
      },
      packed_inputs=packed_inputs,
      cp_active=cp_active,
      cp_size=cp_size,
      chunk_batch=chunk_batch,
      mini_batch=KDA_DMA_MINI_BATCH if mini_batch is None else mini_batch,
      vmem_limit_bytes=vmem_limit_bytes,
      fuse_gate_bwd=fuse_gate_bwd,
  )
  assert not packed_inputs or NT % chunk_batch == 0

  # Dynamic chunk-slot indexing into the packed BF16 beta/dbeta tile must
  # advance by a 16-row-aligned stride on TPU. Eight-row padding works only
  # when Mosaic can constant-fold the slot index.
  MB_PAD = align_up(MB, 16)
  N_HG = H // MB
  DB_NT = ((NT + chunk_batch - 1) // chunk_batch) * chunk_batch
  DB_TAIL_PAD = DB_NT - NT
  N_MAX = (
      cu_seqlens.shape[-1] - 1
      if packed_inputs
      else 1
      if (segment_ids is None)
      else N_MAX
  )

  if packed_inputs:
    if cu_seqlens.ndim == 1:
      cu_seqlens = jnp.broadcast_to(
          cu_seqlens[None, :], (B, cu_seqlens.shape[0])
      )
    if original_token_segments:
      aligned_cu_seqlens = cu_seqlens
      compact_chunk_metadata = False
      segment_chunks = original_token_segment_ids.reshape(B, NT, BT)
      chunk_seg_ids = segment_chunks[..., 0]
      valid = segment_chunks != 0
      valid_end = jnp.sum(valid, axis=-1, dtype=jnp.int32)
      token_idx = jnp.arange(BT, dtype=jnp.int32)
      starts_later_fragment = valid & (
          segment_chunks != chunk_seg_ids[..., None]
      )
      split = jnp.min(
          jnp.where(starts_later_fragment, token_idx, jnp.int32(BT)), axis=-1
      )
      split = jnp.minimum(split, valid_end)
      last_idx = jnp.maximum(valid_end - 1, 0)[..., None]
      last_segment_id = jnp.take_along_axis(segment_chunks, last_idx, axis=-1)[
          ..., 0
      ]
      fragment_counts = jnp.sum(
          valid
          & jnp.concatenate(
              (
                  jnp.ones((B, NT, 1), dtype=jnp.bool_),
                  segment_chunks[..., 1:] != segment_chunks[..., :-1],
              ),
              axis=-1,
          ),
          axis=-1,
          dtype=jnp.int32,
      )
      # Field-major (B, 4, NT): the scalar-prefetch SMEM tile then pads the
      # 4-wide field axis to 8 instead of padding the chunk axis 4 -> 128.
      # A (B, NT, 4) layout costs NT * 128 * 4 B = 2 MiB of SMEM at 256K and
      # overflows the 1 MiB budget ('prefetched SMEM operand 2').
      chunk_metadata = jnp.stack(
          (split, valid_end, last_segment_id, fragment_counts), axis=1
      )
      chunk_indices = jnp.zeros((B, NT, 2), dtype=jnp.int32)
    else:
      aligned_cu_seqlens = compute_padded_cu_seqlens(cu_seqlens, BT)
      compact_chunk_metadata = chunk_indices is None
      if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(aligned_cu_seqlens, BT, max_T=T)
      if chunk_indices.ndim == 2:
        chunk_indices = jnp.broadcast_to(chunk_indices[None, :, :], (B, NT, 2))
      chunk_indices = chunk_indices.astype(jnp.int32)
      seq_idx = chunk_indices[..., 0]
      block_idx = chunk_indices[..., 1]
      seq_lens = jnp.diff(cu_seqlens, axis=-1)
      chunk_lens = jnp.take_along_axis(seq_lens, seq_idx, axis=1)
      chunk_seg_ids = jnp.where(
          block_idx * BT < chunk_lens, seq_idx + 1, 0
      ).astype(jnp.int32)
      # Generated chunk indices are canonical and can be recovered from the
      # aligned boundaries inside the kernel. Explicit mappings retain the
      # legacy full table so their ordering semantics remain unchanged.
      chunk_metadata = (
          aligned_cu_seqlens if compact_chunk_metadata else chunk_indices
      )
  else:
    is_varlen = segment_ids is not None
    segment_ids = (
        jnp.ones((B, T), dtype=jnp.int32) if not is_varlen else segment_ids
    )
    chunk_seg_ids = segment_ids.reshape(B, NT, BT)[:, :, 0]
    cu_seqlens = jnp.zeros((B, 1), dtype=jnp.int32)
    aligned_cu_seqlens = cu_seqlens
    chunk_indices = jnp.zeros((B, NT, 2), dtype=jnp.int32)
    compact_chunk_metadata = False
    # Non-packed kernels never read chunk_metadata_ref. Scalar prefetch
    # allocates its operand even when unused: a dummy [B,NT,2] table pads
    # the last axis to 128 and exceeds 1 MiB at NT=8193. Use a scalar dummy,
    # not a capacity-sized allocation; real packed maps are unchanged.
    chunk_metadata = jnp.zeros((1,), dtype=jnp.int32)

  # Scalar prefetch copies the entire operand into SMEM, including phantom
  # segments. N_MAX=262144 needs (N_MAX+1)*4 bytes before alignment, exceeding
  # the 1 MiB SMEM budget. Keep public capacity/state indexing unchanged;
  # derive only the chunk windows and CP summary length needed by this kernel.
  precomputed_chunk_windows = (
      packed_inputs
      and not original_token_segments
      and cu_seqlens.shape[-1] > 8193
  )
  compact_boundaries = packed_inputs and (
      original_token_segments or precomputed_chunk_windows
  )
  kernel_cu_seqlens = cu_seqlens
  if compact_boundaries:
    first_seq = jnp.maximum(chunk_seg_ids[:, :1] - 1, 0)
    kernel_cu_seqlens = (
        jnp.take_along_axis(jnp.diff(cu_seqlens, axis=-1), first_seq, axis=1)
        if cp_active
        else jnp.zeros((B, 1), dtype=jnp.int32)
    )
  if precomputed_chunk_windows:
    bos = jnp.take_along_axis(cu_seqlens, seq_idx, axis=1)
    eos = jnp.take_along_axis(cu_seqlens, seq_idx + 1, axis=1)
    packed_start = jnp.where(
        block_idx * BT < eos - bos, bos + block_idx * BT, 0
    )
    # Field-major avoids the TPU's padding of a trailing two-element axis.
    chunk_metadata = jnp.stack((packed_start, eos - packed_start), axis=1)
    compact_chunk_metadata = False

  has_dht = dht is not None
  dht_arr = dht if has_dht else jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32)
  initial_state_arr = (
      initial_state
      if initial_state is not None
      # HAS_H0 is static, so the kernel never indexes this operand when the
      # caller has no initial state. Keep the ABI placeholder scalar-sized;
      # making it [B, N_MAX, H, K, V] creates an otherwise-unused FP32 buffer
      # that can dominate executable memory for native segment IDs.
      else jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32)
  )

  # Match the compact PR #463 M4 layout: one 128-wide row per head/chunk,
  # rather than expanding every beta scalar to a 128-wide row.
  if packed_inputs:
    beta_T = int(align_up(T_INPUT, 8 * BT)) + 8 * BT
    beta_r = jnp.pad(beta, ((0, 0), (0, 0), (0, beta_T - T_INPUT))).reshape(
        H, B, beta_T // BT, BT
    )
    beta_r = jnp.pad(beta_r, ((0, 0), (0, 0), (0, 0), (0, BETA_pad - BT)))
  else:
    beta_r = beta.reshape(N_HG, MB, B, NT, BT).transpose(0, 2, 3, 1, 4)
    beta_r = jnp.pad(
        beta_r,
        ((0, 0), (0, 0), (0, 0), (0, MB_PAD - MB), (0, BETA_pad - BT)),
    ).reshape(N_HG * B * NT * MB_PAD, BETA_pad)

  h_r = h if h is not None else jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32)
  initial_state_arr = initial_state_arr.astype(jnp.float32).transpose(
      2, 0, 1, 3, 4
  )

  # ── CP metadata arrays (per-batch) ──
  if cp_active and cp_context is not None:
    cp_post_num = jnp.broadcast_to(
        jnp.asarray(cp_context.post_num_ranks, dtype=jnp.int32).reshape(B), (B,)
    )
    cp_max_post = jax.lax.pmax(cp_post_num, cp_axis_name)
    cp_post_num = jnp.stack((cp_post_num, cp_max_post), axis=-1)
    cp_is_last = jnp.broadcast_to(
        jnp.asarray(cp_context.is_last_rank, dtype=jnp.int32).reshape(B), (B,)
    )
  else:
    cp_post_num = jnp.zeros((B, 2), dtype=jnp.int32)
    cp_is_last = jnp.zeros((B,), dtype=jnp.int32)

  kernel = partial(
      _bwd_mega_kernel,
      H=H,
      B=B,
      SCALE=scale,
      BT=BT,
      BT_PAD=BT_pad,
      K=K,
      V=V,
      NT=NT,
      T_INPUT=T_INPUT,
      MB=MB,
      MB_PAD=MB_PAD,
      CHUNK_BATCH=chunk_batch,
      DMA_INPUT_BUFFER_COUNT=KDA_DMA_INPUT_BUFFER_COUNT,
      DMA_OUTPUT_BUFFER_COUNT=KDA_DMA_OUTPUT_BUFFER_COUNT,
      DISABLE_RECOMPUTE=disable_recompute,
      CP_ACTIVE=cp_active,
      CP_SIZE=cp_size,
      CP_AXIS_NAME=cp_axis_name,
      HAS_H0=has_initial_state,
      HAS_DHT=has_dht,
      RETURN_DH0=return_dh0,
      PACKED_INPUTS=packed_inputs,
      MULTI_SEGMENT_PACKED=packed_inputs and N_MAX > 1,
      BATCH_FIRST_INPUTS=batch_first_inputs,
      COMPACT_CHUNK_METADATA=compact_chunk_metadata,
      PRECOMPUTED_CHUNK_WINDOWS=precomputed_chunk_windows,
      COMPACT_BOUNDARIES=compact_boundaries,
      ORIGINAL_TOKEN_SEGMENTS=original_token_segments,
      FUSE_GATE_BWD=fuse_gate_bwd,
  )
  out_shape = [
      jax.ShapeDtypeStruct((H, B, NT, BT, K), q.dtype),  # dq
      jax.ShapeDtypeStruct((H, B, NT, BT, K), k.dtype),  # dk
      jax.ShapeDtypeStruct((H, B, NT, BT, V), v.dtype),  # dv
      jax.ShapeDtypeStruct(
          (N_HG * B * DB_NT * MB_PAD, BETA_pad), beta.dtype
      ),  # db
      jax.ShapeDtypeStruct((H, B, NT, BT, K), dg_out_dtype),  # dg
      jax.ShapeDtypeStruct(
          (H, B, N_MAX, K, V) if return_dh0 else (1, 1, 1, 1, 1), jnp.float32
      ),  # dh0
      jax.ShapeDtypeStruct(
          (H, B, NT, K, V) if not disable_recompute else (1, 1, 1, 1, 1),
          jnp.float32,
      ),
      jax.ShapeDtypeStruct(
          (B, H, 128) if fuse_gate_bwd else (1, 1, 1), jnp.float32
      ),
      jax.ShapeDtypeStruct(
          (B, H, K) if fuse_gate_bwd else (1, 1, 1), jnp.float32
      ),
  ]

  any_spec = pl.BlockSpec(memory_space=pl.ANY)

  q_r = q if packed_inputs else q.reshape(H, B, NT, BT, K)
  k_r = k if packed_inputs else k.reshape(H, B, NT, BT, K)
  v_r = v if packed_inputs else v.reshape(H, B, NT, BT, V)
  g_r = g.reshape(H, B, NT, BT, K)
  raw_gate_r = (
      raw_gate.transpose(2, 0, 1, 3).reshape(H, B, NT, BT, K)
      if fuse_gate_bwd
      else jnp.zeros((1, 1, 1, 1, 1), dtype=q.dtype)
  )
  gate_A_log_r = (
      jnp.broadcast_to(gate_A_log[:, None], (H, K))
      if fuse_gate_bwd
      else jnp.zeros((1, 1), jnp.float32)
  )
  gate_dt_bias_r = (
      gate_dt_bias.reshape(H, K)
      if fuse_gate_bwd
      else jnp.zeros((1, 1), jnp.float32)
  )
  original_segment_ids_r = (
      original_token_segment_ids.reshape(B, NT, BT)
      if original_token_segments
      else jnp.zeros((1, 1, 1), dtype=jnp.int32)
  )

  def _pad_bt_4d(x):
    if x.shape[-1] < BT_pad:
      return jnp.pad(
          x,
          ((0, 0), (0, 0), (0, 0), (0, BT_pad - x.shape[-1])),
      )
    return x

  Aqk_r = _pad_bt_4d(Aqk).reshape(H, B, NT, BT, BT_pad)
  Akk_r = _pad_bt_4d(Akk).reshape(H, B, NT, BT, BT_pad)
  do_r = do if packed_inputs else do.reshape(H, B, NT, BT, V)
  dht_arr = dht_arr.astype(jnp.float32).transpose(2, 0, 1, 3, 4)
  (
      dq_r,
      dk_r,
      dv_r,
      db_r,
      dg_r,
      dh0_r,
      _,
      gate_dA_r,
      gate_dbias_r,
  ) = pl.pallas_call(
      kernel,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=5,
          grid=(),
          # Segment IDs are indexed directly by the general compact body, so stage
          # this small (B * T * int32) metadata tensor in VMEM. Other large inputs
          # remain in ANY/HBM and use the kernel's explicit DMA pipeline.
          in_specs=(
              [
                  pl.BlockSpec(
                      block_shape=original_segment_ids_r.shape,
                      memory_space=pltpu.VMEM,
                  )
              ]
              + [any_spec] * 14
          ),
          out_specs=[any_spec] * 9,
      ),
      compiler_params=pltpu.CompilerParams(
          disable_bounds_checks=True,
          collective_id=KDA_BWD_CP_COLLECTIVE_ID if cp_active else None,
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      interpret=get_interpret(),
      cost_estimate=resolve_cost_estimate(
          cost_estimate_flops, kda_bwd_mega_cost(H, B, NT, BT, K, V)
      ),
  )(
      chunk_seg_ids,
      kernel_cu_seqlens.astype(jnp.int32),
      chunk_metadata.astype(jnp.int32),
      cp_post_num,
      cp_is_last,
      original_segment_ids_r,
      q_r,
      k_r,
      v_r,
      beta_r,
      Aqk_r,
      Akk_r,
      g_r,
      h_r,
      do_r,
      dht_arr,
      initial_state_arr,
      raw_gate_r,
      gate_A_log_r,
      gate_dt_bias_r,
  )

  db_r = db_r.reshape(N_HG, B, DB_NT, MB_PAD, BETA_pad)[..., :MB, :BT]
  if DB_TAIL_PAD:
    REM = NT % chunk_batch
    db_r = jnp.concatenate(
        [db_r[:, :, :REM], db_r[:, :, chunk_batch:]],
        axis=2,
    )
  db_r = db_r.transpose(0, 3, 1, 2, 4).reshape(H, B, T)

  if packed_inputs:
    dq_aligned = dq_r.reshape(H, B, T, K)
    dk_aligned = dk_r.reshape(H, B, T, K)
    dv_aligned = dv_r.reshape(H, B, T, V)
    dg_aligned = dg_r.reshape(H, B, T, K)
    # A single packed segment whose static token extent is BT-aligned has no
    # alignment gaps: aligned and public token positions are identical.  Skip
    # all four gradient compactors and expose the existing buffers directly in
    # batch-first order.  Besides removing post-processing, this avoids the
    # compactor shrinking to a one-head DMA for long contexts when constrained
    # by VMEM; one-head slices are illegal for batch-first 8-head HBM tiles.
    single_aligned_segment = original_token_segments or (
        cu_seqlens.shape[-1] == 2 and T_INPUT % BT == 0 and T == T_INPUT
    )
    if single_aligned_segment:
      dq_packed = dq_aligned.transpose(1, 2, 0, 3)
      dk_packed = dk_aligned.transpose(1, 2, 0, 3)
      dv_packed = dv_aligned.transpose(1, 2, 0, 3)
      dg_packed = dg_aligned.transpose(1, 2, 0, 3)
    else:
      dq_packed = _compact_head_first_output_pallas(
          dq_aligned, cu_seqlens, aligned_cu_seqlens, T_INPUT, BT
      )
      dk_packed = _compact_head_first_output_pallas(
          dk_aligned, cu_seqlens, aligned_cu_seqlens, T_INPUT, BT
      )
      dv_packed = _compact_head_first_output_pallas(
          dv_aligned, cu_seqlens, aligned_cu_seqlens, T_INPUT, BT
      )
      dg_packed = _compact_head_first_output_pallas(
          dg_aligned, cu_seqlens, aligned_cu_seqlens, T_INPUT, BT
      )
    if original_token_segments:
      # The compact backward already writes db in the original token grid;
      # aligned_cu_seqlens equals cu_seqlens, so the generic gather below
      # would only materialize an identity permutation.
      db_packed_out = db_r
    else:
      packed_positions = jnp.arange(T_INPUT, dtype=jnp.int32)
      db_packed_batches = []
      for batch_idx in range(B):
        seq_idx = _packed_sequence_indices(
            cu_seqlens[batch_idx], packed_positions
        )
        padding_offsets = (
            aligned_cu_seqlens[batch_idx, :-1] - cu_seqlens[batch_idx, :-1]
        )
        aligned_idx = packed_positions + padding_offsets[seq_idx]
        aligned_idx = jnp.clip(aligned_idx, 0, T - 1)
        db_batch = jnp.take(db_r[:, batch_idx], aligned_idx, axis=1)
        valid = packed_positions < cu_seqlens[batch_idx, -1]
        db_packed_batches.append(
            jnp.where(valid[None, :], db_batch, jnp.zeros_like(db_batch))
        )
      db_packed_out = jnp.stack(db_packed_batches, axis=1)
    dh0_out = dh0_r.transpose(1, 2, 0, 3, 4)
    if initial_state is not None:
      dh0_out = dh0_out.astype(initial_state.dtype)
    outputs = (
        dq_packed,
        dk_packed,
        dv_packed,
        db_packed_out,
        dg_packed,
        dh0_out if return_dh0 else None,
    )
    if fuse_gate_bwd:
      return outputs + (
          jnp.sum(gate_dA_r[:, :, 0], axis=0).astype(gate_A_log.dtype),
          jnp.sum(gate_dbias_r, axis=0).reshape(-1).astype(gate_dt_bias.dtype),
      )
    return outputs

  dh0_out = dh0_r.transpose(1, 2, 0, 3, 4)
  if initial_state is not None:
    dh0_out = dh0_out.astype(initial_state.dtype)
  return (
      dq_r.reshape(H, B, T, K),
      dk_r.reshape(H, B, T, K),
      dv_r.reshape(H, B, T, V),
      db_r,
      dg_r.reshape(H, B, T, K),
      dh0_out if return_dh0 else None,
  )
