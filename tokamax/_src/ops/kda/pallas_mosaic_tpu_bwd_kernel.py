# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Pallas/Mosaic TPU backward kernels for experimental KDA."""

# pylint: disable=invalid-name

from __future__ import annotations

import contextlib
import dataclasses
import functools
from functools import partial
import math
import os

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.kda.common import (
    chunk_gated_delta_rule_fwd_h,
    estimate_mini_batch,
    get_max_mini_batch,
    get_tpu_limits,
    get_vmem_limit_bytes,
    kda_gate_chunk_cumsum,
    tri_inv_xor_dnc,
)
from tokamax._src.ops.kda.cp_utils import (
    ContextParallelMetadata,
    _merge_dht,
    all_gather_into_tensor,
    cp_associative_scan_bwd,
    cp_canonicalize_packed_segment_ids,
    cp_shift_left,
)
from tokamax._src.ops.kda.pallas_mosaic_tpu_fwd_kernel import (
    chunk_kda_fwd_h_o_varlen,
)
from tokamax._src.ops.kda.pallas_mosaic_tpu_types import (
    GdnResiduals,
    KdaResiduals,
)
from tokamax._src.ops.kda.utils import (
    _align_seqs,
    _unalign_output,
    align_up,
    canonicalize_packed_segment_ids,
    causal_conv1d_silu_bwd_bt,
    cdiv,
    compute_packed_chunk_metadata,
    exp,
    exp2,
    get_chain_precision,
    get_interpret,
    l2norm_bwd,
    segment_ids_to_seqlens,
)


# =============================================================================
# Context-parallel backward pre-process
# =============================================================================

# Dimensions.
#   B: batch size
#   H: number of value heads (H_QK for query/key heads when GQA is active)
#   MB: mini-batch of heads/chunks processed per grid point
#   T: sequence length
#   NT: number of chunks (T // BT)
#   BT: chunk size (64 or 128)
#   BC: sub-chunk block size (16)
#   N: max number of segments per batch element
#   K: key/query head dimension
#   V: value head dimension
#   GW: gate channel width (1 for scalar gate, K for per-channel gate)


def _chunk_gated_delta_rule_bwd_dhu_pre_process_kernel(
    # Inputs.
    chunk_active_ref,  # SMEM [NT] int32 -- per-chunk activeness mask
    q_ref,  # HBM [H*NT*BT, K]
    k_ref,  # HBM [H*NT*BT, K]
    w_ref,  # HBM [H*NT*BT, K]
    do_ref,  # HBM [H*NT*BT, V]
    dv_ref,  # HBM [H*NT*BT, V]
    gk_ref,  # HBM [H*NT*BT, K]
    # Outputs.
    dS_ext_ref,  # HBM [H, K, V]
    dM_ref,  # HBM [H, K, K]
    # Scratch.
    q_scratch_ref,  # VMEM [2, MB, BT, K]
    k_scratch_ref,  # VMEM [2, MB, BT, K]
    w_scratch_ref,  # VMEM [2, MB, BT, K]
    do_scratch_ref,  # VMEM [2, MB, BT, V]
    dv_scratch_ref,  # VMEM [2, MB, BT, V]
    gk_scratch_ref,  # VMEM [2, MB, BT, K]
    dh_acc_ref,  # VMEM [MB, K, V] fp32
    M_acc_ref,  # VMEM [MB, K, K] fp32
    sems,  # DMA [8, 2]
    *,
    H,
    NT,
    BT,
    K,
    MB,
    SCALE,
    USE_EXP2,
):
  """Reverse-time chunk-recurrence kernel with hand-pipelined DMA.

  Single-program kernel (grid=()) that loops over N_HG*NT iterations,
  processing head-groups of MB heads sequentially and chunks in reverse
  time order.  Double-buffered input DMA overlaps with MXU compute.
  MB heads are batched via ``dot_general`` batch dim (no Python unroll).
  """
  N_HG = H // MB
  TOTAL = N_HG * NT
  eye_k = jnp.eye(K, dtype=jnp.float32)
  SEM_OUT = 6 * MB
  precision = (
      None if q_ref.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
  )

  def _async_copy(src, dst, sem, wait=False):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
      cp.wait()
    else:
      cp.start()

  def _iter_to_hg_i(it):
    hg = it // NT
    i_rev = it % NT
    return hg, NT - 1 - i_rev

  def start_input_dma(buf, hg, i):
    for h_local in range(MB):
      off = (hg * MB + h_local) * NT * BT + i * BT
      _async_copy(q_ref.at[pl.ds(off, BT), pl.ds(None)], q_scratch_ref.at[buf, h_local], sems.at[0 * MB + h_local, buf])
      _async_copy(k_ref.at[pl.ds(off, BT), pl.ds(None)], k_scratch_ref.at[buf, h_local], sems.at[1 * MB + h_local, buf])
      _async_copy(w_ref.at[pl.ds(off, BT), pl.ds(None)], w_scratch_ref.at[buf, h_local], sems.at[2 * MB + h_local, buf])
      _async_copy(do_ref.at[pl.ds(off, BT), pl.ds(None)], do_scratch_ref.at[buf, h_local], sems.at[3 * MB + h_local, buf])
      _async_copy(dv_ref.at[pl.ds(off, BT), pl.ds(None)], dv_scratch_ref.at[buf, h_local], sems.at[4 * MB + h_local, buf])
      _async_copy(gk_ref.at[pl.ds(off, BT), pl.ds(None)], gk_scratch_ref.at[buf, h_local], sems.at[5 * MB + h_local, buf])

  def wait_input_dma(buf, hg, i):
    for h_local in range(MB):
      off = (hg * MB + h_local) * NT * BT + i * BT
      _async_copy(q_ref.at[pl.ds(off, BT), pl.ds(None)], q_scratch_ref.at[buf, h_local], sems.at[0 * MB + h_local, buf], True)
      _async_copy(k_ref.at[pl.ds(off, BT), pl.ds(None)], k_scratch_ref.at[buf, h_local], sems.at[1 * MB + h_local, buf], True)
      _async_copy(w_ref.at[pl.ds(off, BT), pl.ds(None)], w_scratch_ref.at[buf, h_local], sems.at[2 * MB + h_local, buf], True)
      _async_copy(do_ref.at[pl.ds(off, BT), pl.ds(None)], do_scratch_ref.at[buf, h_local], sems.at[3 * MB + h_local, buf], True)
      _async_copy(dv_ref.at[pl.ds(off, BT), pl.ds(None)], dv_scratch_ref.at[buf, h_local], sems.at[4 * MB + h_local, buf], True)
      _async_copy(gk_ref.at[pl.ds(off, BT), pl.ds(None)], gk_scratch_ref.at[buf, h_local], sems.at[5 * MB + h_local, buf], True)

  # ── Pre-loop prologue: start DMA for first iteration ────────────────
  hg0, i0 = _iter_to_hg_i(0)
  start_input_dma(0, hg0, i0)

  # ── Main loop ────────────────────────────────────────────────────────
  @pl.loop(0, TOTAL, unroll=False)
  def body(it):
    buf = it % 2
    hg = it // NT
    i_rev = it % NT
    i = NT - 1 - i_rev

    # 1. Wait ping-in (current buffer DMA complete)
    wait_input_dma(buf, hg, i)

    # 2. Trigger pong-in (start DMA for next iteration into other buffer)
    @pl.when(it + 1 < TOTAL)
    def _():
      next_hg, next_i = _iter_to_hg_i(it + 1)
      start_input_dma(1 - buf, next_hg, next_i)

    # 3. Wait prev output DMA (ping out) — at head-group boundary
    @pl.when((i_rev == 0) & (hg > 0))
    def _():
      prev_off = (hg - 1) * MB
      _async_copy(dh_acc_ref, dS_ext_ref.at[pl.ds(prev_off, MB), pl.ds(None), pl.ds(None)], sems.at[SEM_OUT, 0], wait=True)
      _async_copy(M_acc_ref, dM_ref.at[pl.ds(prev_off, MB), pl.ds(None), pl.ds(None)], sems.at[SEM_OUT + 1, 0], wait=True)

    # 4. Reset accumulators at start of each head group
    @pl.when(i_rev == 0)
    def _():
      dh_acc_ref[...] = jnp.zeros_like(dh_acc_ref[...])
      M_acc_ref[...] = jnp.broadcast_to(eye_k, (MB, K, K))

    # 5. Compute on ping (only for active chunks)
    is_active = chunk_active_ref[i] != 0
    @pl.when(is_active)
    def _():
      bq = q_scratch_ref[buf].astype(jnp.float32)    # [MB, BT, K]
      bk = k_scratch_ref[buf].astype(jnp.float32)    # [MB, BT, K]
      bw = w_scratch_ref[buf].astype(jnp.float32)    # [MB, BT, K]
      bdo = do_scratch_ref[buf].astype(jnp.float32)   # [MB, BT, V]
      bdv = dv_scratch_ref[buf]                        # [MB, BT, V]
      bgk = gk_scratch_ref[buf].astype(jnp.float32)   # [MB, BT, K]

      dh = dh_acc_ref[...]   # [MB, K, V]
      M = M_acc_ref[...]     # [MB, K, K]

      # (1) dv_cur = k @ dh + dv → [MB, BT, V]
      dv_cur = jax.lax.dot_general(
        bk, dh,
        (((2,), (1,)), ((0,), (0,))),
        precision=precision,
        preferred_element_type=jnp.float32,
      ) + bdv

      # decay
      gk_last = jnp.maximum(bgk[:, BT - 1, :], jnp.asarray(-126.0, dtype=jnp.float32))
      decay = exp2(gk_last) if USE_EXP2 else exp(gk_last)   # [MB, K]

      # (2) dh *= decay → [MB, K, V]
      dh_new = dh * decay[:, :, None]

      # (3) dh += q.T @ do * scale - w.T @ dv_cur → [MB, K, V]
      dh_new = dh_new + (
        jax.lax.dot_general(
          bq, bdo,
          (((1,), (1,)), ((0,), (0,))),
          precision=precision,
          preferred_element_type=jnp.float32,
        ) * SCALE
        - jax.lax.dot_general(
          bw, dv_cur,
          (((1,), (1,)), ((0,), (0,))),
          precision=precision,
          preferred_element_type=jnp.float32,
        )
      )

      # (4) kM = k @ M → [MB, BT, K]
      kM = jax.lax.dot_general(
        bk, M,
        (((2,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
      )

      # (5) WkM = w.T @ kM → [MB, K, K]
      WkM = jax.lax.dot_general(
        bw, kM,
        (((1,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
      )
      M_new = M * decay[:, :, None] - WkM

      dh_acc_ref[...] = dh_new
      M_acc_ref[...] = M_new

    # 6. Trigger ping-out at end of each head group
    @pl.when(i_rev == NT - 1)
    def _():
      _async_copy(dh_acc_ref, dS_ext_ref.at[pl.ds(hg * MB, MB), pl.ds(None), pl.ds(None)], sems.at[SEM_OUT, 0])
      _async_copy(M_acc_ref, dM_ref.at[pl.ds(hg * MB, MB), pl.ds(None), pl.ds(None)], sems.at[SEM_OUT + 1, 0])

  # ── Post-loop epilogue: wait for final output DMA ────────────────────
  last_off = (N_HG - 1) * MB
  _async_copy(dh_acc_ref, dS_ext_ref.at[pl.ds(last_off, MB), pl.ds(None), pl.ds(None)], sems.at[SEM_OUT, 0], wait=True)
  _async_copy(M_acc_ref, dM_ref.at[pl.ds(last_off, MB), pl.ds(None), pl.ds(None)], sems.at[SEM_OUT + 1, 0], wait=True)


@functools.partial(
  jax.jit,
  static_argnames=[
    "scale",
    "chunk_size",
    "use_exp2",
  ],
)
@jaxtyping.jaxtyped
def chunk_gated_delta_rule_bwd_dhu_pre_process(
  q: Float[Array, "H B T K"],
  k: Float[Array, "H B T K"],
  w: Float[Array, "H B T K"],
  do: Float[Array, "H B T V"],
  dv: Float[Array, "H B T V"],
  gk: Float[Array, "H B T K"],
  scale: float,
  segment_ids: Int[Array, "T"] | Int[Array, "B T"],
  chunk_size: int = 64,
  use_exp2: bool = True,
) -> tuple[Float[Array, "B H K V"], Float[Array, "B H K K"]]:
  """Builds the CP backward summary for the first real local segment.

  Only that segment receives cross-rank state in forward, so only its affine
  backward summary is communicated upstream. Batches are handled independently.
  """
  H, B, T, K = q.shape
  V = do.shape[-1]
  BT = chunk_size
  NT = T // BT

  # Handle B > 1 by looping over batch elements with the B=1 kernel.
  if B > 1:
    ds_list, dm_list = [], []
    for b in range(B):
      seg_b = segment_ids[b] if segment_ids.ndim == 2 else segment_ids
      ds_b, dm_b = chunk_gated_delta_rule_bwd_dhu_pre_process(
        q=q[:, b:b+1], k=k[:, b:b+1], w=w[:, b:b+1],
        do=do[:, b:b+1], dv=dv[:, b:b+1], gk=gk[:, b:b+1],
        scale=scale, segment_ids=seg_b,
        chunk_size=chunk_size, use_exp2=use_exp2,
      )
      ds_list.append(ds_b)
      dm_list.append(dm_b)
    return jnp.concatenate(ds_list, axis=0), jnp.concatenate(dm_list, axis=0)

  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  assert K % 128 == 0, f"K={K} must be a multiple of 128 (TPU lane alignment)"
  assert V % 128 == 0, f"V={V} must be a multiple of 128 (TPU lane alignment)"
  seg = segment_ids
  if seg.ndim == 2:
    seg = seg[0]

  if NT == 0:
    return (
      jnp.zeros([B, H, K, V], dtype=jnp.float32),
      jnp.zeros([B, H, K, K], dtype=jnp.float32),
    )

  # ── Precompute per-chunk activeness mask (SMEM scalar prefetch) ──────
  first_real_idx = jnp.argmax((seg != 0).astype(jnp.int32))
  first_seg_id = seg[first_real_idx]
  has_real = first_seg_id != 0
  chunk_seg_ids = seg.reshape(NT, BT)[:, 0]
  chunk_active = ((chunk_seg_ids == first_seg_id) & has_real).astype(jnp.int32)

  # ── Auto MB: maximise VMEM utilisation ──
  q_elem = 2 if q.dtype == jnp.bfloat16 else 4
  gk_elem = 2 if gk.dtype == jnp.bfloat16 else 4
  do_elem = 2 if do.dtype == jnp.bfloat16 else 4
  per_head = (
      2 * BT * (3 * K * q_elem + K * gk_elem + 2 * V * do_elem)
      + (K * V + K * K) * 4
  )
  MB = estimate_mini_batch(per_head, H)

  # ── Reshape inputs: [H, 1, T, dim] → [H*NT*BT, dim] (no transpose) ──
  q_r = q.reshape(H * NT * BT, K)
  k_r = k.reshape(H * NT * BT, K)
  w_r = w.reshape(H * NT * BT, K)
  do_r = do.reshape(H * NT * BT, V)
  dv_r = dv.reshape(H * NT * BT, V)
  gk_r = gk.reshape(H * NT * BT, K)

  # ── Output shapes: [H, K, V] and [H, K, K] ──────────────────────────
  dS_ext_spec = jax.ShapeDtypeStruct([H, K, V], jnp.float32)
  dM_spec = jax.ShapeDtypeStruct([H, K, K], jnp.float32)

  # ── Scratch: double-buffered inputs + accumulators + semaphores ──────
  scratch_shapes = [
    pltpu.VMEM((2, MB, BT, K), q.dtype),   # q
    pltpu.VMEM((2, MB, BT, K), k.dtype),   # k
    pltpu.VMEM((2, MB, BT, K), w.dtype),   # w
    pltpu.VMEM((2, MB, BT, V), do.dtype),  # do
    pltpu.VMEM((2, MB, BT, V), dv.dtype),  # dv
    pltpu.VMEM((2, MB, BT, K), gk.dtype),  # gk
    pltpu.VMEM((MB, K, V), jnp.float32),   # dh accumulator
    pltpu.VMEM((MB, K, K), jnp.float32),   # M accumulator
    pltpu.SemaphoreType.DMA((6 * MB + 2, 2)),  # 6*MB input + 2 output channels
  ]

  in_specs = [pl.BlockSpec(memory_space=pl.ANY)] * 6
  out_specs = [pl.BlockSpec(memory_space=pl.ANY)] * 2

  interpret = get_interpret()

  dS_ext_raw, dM_raw = pl.pallas_call(
    functools.partial(
      _chunk_gated_delta_rule_bwd_dhu_pre_process_kernel,
      H=H, NT=NT, BT=BT, K=K, MB=MB,
      SCALE=float(scale),
      USE_EXP2=use_exp2,
    ),
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=1,
      grid=(),
      in_specs=in_specs,
      out_specs=out_specs,
      scratch_shapes=scratch_shapes,
    ),
    out_shape=[dS_ext_spec, dM_spec],
    interpret=interpret,
  )(chunk_active, q_r, k_r, w_r, do_r, dv_r, gk_r)

  # ── Reshape outputs: [H, K, V] → [1, H, K, V] ──────────────────────
  dS_ext = dS_ext_raw.reshape(B, H, K, V)
  dM = dM_raw.reshape(B, H, K, K)

  return dS_ext, dM


@jaxtyping.jaxtyped
def kda_gate_bwd(
  g: Float[Array, "H B T GW"],
  a_log: Float[Array, "H"],
  *,
  dyg: Float[Array, "H B T GW"],
  delta_time_bias: Float[Array, "H*GW"] | None = None,
  lower_bound: float | None = None,
  valid_mask: Array | None = None,
) -> tuple[
    Float[Array, "H B T GW"],
    Float[Array, "H"],
    Float[Array, "H*GW"] | None,
]:
  """Differentiates the KDA gate with respect to g, a_log, and delta_time_bias.

  The standard path differentiates `-exp(a_log) * softplus(g + bias)`; the
  lower-bound path differentiates the sigmoid form.
  """
  H, GW = g.shape[0], g.shape[-1]

  g_f = g.astype(jnp.float32)
  dyg_f = dyg.astype(jnp.float32)
  if valid_mask is not None:
    mask_4d = valid_mask[None, :, :, None]
    g_f = jnp.where(mask_4d, g_f, 0.0)
    dyg_f = jnp.where(mask_4d, dyg_f, 0.0)

  # Apply bias if present
  if delta_time_bias is not None:
    g_f = g_f + delta_time_bias.reshape(H, 1, 1, GW).astype(jnp.float32)

  if lower_bound is None:
    # Forward: yg = -exp(a_log) * softplus(g + bias)
    # softplus(x) = log(1 + exp(x)), d/dx softplus(x) = sigmoid(x)
    b_A = -jnp.exp(a_log.astype(jnp.float32))  # [H]
    b_yg = b_A.reshape(H, 1, 1, 1) * jax.nn.softplus(g_f)  # [H, B, T, GW]
    dg_f = b_A.reshape(H, 1, 1, 1) * (dyg_f * jax.nn.sigmoid(g_f))  # [H, B, T, GW]
    dA_per_elem = dyg_f * b_yg  # [H, B, T, GW]
  else:
    # Forward: yg = lower_bound * sigmoid(exp(a_log) * g)
    b_A = jnp.exp(a_log.astype(jnp.float32))  # [H]
    b_inner = b_A.reshape(H, 1, 1, 1) * g_f  # [H, B, T, GW]
    b_sig = jax.nn.sigmoid(b_inner)
    b_dsig = b_sig * (1.0 - b_sig)
    dg_f = dyg_f * (lower_bound * b_dsig) * b_A.reshape(H, 1, 1, 1)  # [H, B, T, GW]
    dA_per_elem = dg_f * g_f  # [H, B, T, GW]

  if valid_mask is not None:
    mask_4d = valid_mask[None, :, :, None]
    dg_f = jnp.where(mask_4d, dg_f, 0.0)
    dA_per_elem = jnp.where(mask_4d, dA_per_elem, 0.0)

  # dA: reduce over all dims except H (axis 0) → [H]
  reduce_axes = (1, 2, 3)
  dA = jnp.sum(dA_per_elem, axis=reduce_axes)
  dA = dA.astype(a_log.dtype)

  # Cast dg back to input dtype
  dg = dg_f.astype(g.dtype)

  # dbias: sum dg over B, T (axes 1, 2) → [H, GW] → [H*GW]
  if delta_time_bias is not None:
    dbias = jnp.sum(dg_f, axis=(1, 2)).reshape(-1)  # [H*GW]
    dbias = dbias.astype(delta_time_bias.dtype)
  else:
    dbias = None

  return dg, dA, dbias


# =============================================================================
# KDA backward fusion kernels
# =============================================================================

def _chunk_segment_metadata(chunk_seg_ids, batch_idx, chunk_id, NT):
  """Return segment metadata for one chunk from compressed per-chunk IDs.

  chunk_seg_ids: [B, NT] int32 — ``segment_ids.reshape(B, NT, BT)[:, :, 0]``.
  batch_idx: which batch element (0..B-1).
  chunk_id: chunk index within that batch (0..NT-1).
  NT: per-batch chunk count (T // BT).
  Returns: (seg_cur, seq_idx, is_valid, is_first_chunk, is_last_chunk).
  """
  chunk_id = jnp.asarray(chunk_id, dtype=jnp.int32)
  batch_idx = jnp.asarray(batch_idx, dtype=jnp.int32)

  seg_cur = chunk_seg_ids[batch_idx, chunk_id]
  is_valid = seg_cur != 0

  prev_seg = jnp.where(chunk_id == 0, jnp.int32(0),
                        chunk_seg_ids[batch_idx, jnp.maximum(chunk_id - 1, 0)])
  next_seg = jnp.where(chunk_id + 1 >= NT, jnp.int32(0),
                        chunk_seg_ids[batch_idx, jnp.minimum(chunk_id + 1, NT - 1)])

  is_first_chunk = is_valid & (prev_seg != seg_cur)
  is_last_chunk = is_valid & (next_seg != seg_cur)
  seq_idx = jnp.maximum(seg_cur - 1, 0).astype(jnp.int32)
  return seg_cur, seq_idx, is_valid, is_first_chunk, is_last_chunk

# =====================================================================
# M1: recompute_w_u_fwd + compute_v_new_from_h
# =====================================================================

@partial(jax.jit, static_argnames=["dtype", "two_pass_bf16"])
def compute_m1_recompute(
    bq, bk, bv, bb, bA, bg, bh, dtype, two_pass_bf16: bool = False
):
  """Recomputes (u, w, v_new, qg, kg) from a saved state."""
  if dtype != jnp.bfloat16:
    precision = jax.lax.Precision.HIGHEST
  elif two_pass_bf16:
    precision = jax.lax.Precision.HIGH
  else:
    precision = None
  g_exp = jnp.exp2(bg)
  v_beta = bv * bb[:, :, None]
  u = jnp.matmul(
      bA.astype(jnp.float32),
      v_beta.astype(jnp.float32),
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  w = jnp.matmul(bA.astype(jnp.float32), (bk * bb[:, :, None] * g_exp).astype(jnp.float32),
                 precision=precision,
                 preferred_element_type=jnp.float32)
  u_mat = u if two_pass_bf16 else u.astype(dtype)
  w_mat = w if two_pass_bf16 else w.astype(dtype)
  v_new = u_mat.astype(jnp.float32) - jnp.matmul(
      w_mat.astype(jnp.float32),
      bh.astype(jnp.float32),
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  qg = bq * g_exp
  kg = bk * jnp.exp2(bg[:, -1:, :] - bg)
  return u, w, v_new, qg, kg


def _recompute_w_u_fwd(
    q,
    k,
    v,
    beta,
    A,
    gk,
    chunk_size,
    *,
    two_pass_bf16: bool = False,
    packed_tok_meta: jax.Array | None = None,
):
  """Recompute WY intermediates for the low-memory backward path."""
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT

  A_chunks = A.reshape(H * B * NT, BT, BT).astype(jnp.float32)
  q_chunks = q.reshape(H * B * NT, BT, K)
  k_chunks = k.reshape(H * B * NT, BT, K)
  v_chunks = v.reshape(H * B * NT, BT, V)
  beta_chunks = beta.reshape(H * B * NT, BT, 1)
  # Preserve gate width (K for per-channel, 1 for scalar) and broadcast in products below.
  g_chunks = gk.reshape(H * B * NT, BT, gk.shape[-1])
  valid_f = None
  if packed_tok_meta is not None:
    def _expand_meta(sublane: int) -> jax.Array:
      m = packed_tok_meta[:, :, sublane, :].astype(jnp.float32)
      return jnp.broadcast_to(m[None, :, :, :, None], (H, B, NT, BT, 1)).reshape(
          H * B * NT, BT, 1
      )

    m_in_f = _expand_meta(2)
    m_out_f = _expand_meta(3)
    valid_f = _expand_meta(4)
    q_chunks = (q_chunks.astype(jnp.float32) * valid_f).astype(q.dtype)
    k_chunks = (k_chunks.astype(jnp.float32) * valid_f).astype(k.dtype)
    v_chunks = (v_chunks.astype(jnp.float32) * valid_f).astype(v.dtype)
    beta_chunks = beta_chunks.astype(jnp.float32) * valid_f
    g_exp = m_in_f * jnp.exp2(m_in_f * g_chunks)
    g_exp_out = m_out_f * jnp.exp2(
        m_out_f * (g_chunks[:, -1:, :] - g_chunks)
    )
  else:
    g_exp = jnp.exp2(g_chunks)
    g_exp_out = jnp.exp2(g_chunks[:, -1:, :] - g_chunks)
  if q.dtype != jnp.bfloat16:
    precision = jax.lax.Precision.HIGHEST
  elif two_pass_bf16:
    precision = jax.lax.Precision.HIGH
  else:
    precision = None

  wy_dtype = (
      jnp.float32
      if (two_pass_bf16 and q.dtype == jnp.bfloat16)
      else q.dtype
  )
  u_f32 = jnp.matmul(
      A_chunks,
      (v_chunks * beta_chunks).astype(jnp.float32),
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  w_f32 = jnp.matmul(
      A_chunks,
      (k_chunks * beta_chunks * g_exp).astype(jnp.float32),
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  if valid_f is not None:
    u_f32 = u_f32 * valid_f
    w_f32 = w_f32 * valid_f
  u = u_f32.astype(wy_dtype)
  w = w_f32.astype(wy_dtype)
  qg = (q_chunks * g_exp).astype(wy_dtype)
  kg = (k_chunks * g_exp_out).astype(wy_dtype)

  return (
      w.reshape(H, B, T, K),
      u.reshape(H, B, T, V),
      qg.reshape(H, B, T, K),
      kg.reshape(H, B, T, K),
  )


def compute_dhu_recurrence(
    bkg, dh, bdv0, dh_tmp, g_exp_last, bqg, bw, bdo, scale, precision
):
  bdv = jnp.matmul(
      bkg,
      dh,
      precision=precision,
      preferred_element_type=jnp.float32,
  ) + bdv0
  dh_new = dh_tmp * g_exp_last[:, :, None] + jnp.matmul(
    jnp.concatenate([bqg * scale, -bw], axis=1).transpose(0, 2, 1),
    jnp.concatenate([bdo.astype(jnp.float32), bdv], axis=1),
    precision=precision,
    preferred_element_type=jnp.float32,
  )
  return bdv, dh_new


def compute_dhu_recurrence_wfree(
    bkg, dh, bdv0, dh_tmp, g_exp_last, bqg, bkbg, bA, bdo, scale, precision
):
  """w-free chain: dh_new uses k_beta_g^T (A^T dv) instead of w^T dv."""
  bdv = (
      jnp.matmul(
          bkg,
          dh,
          precision=precision,
          preferred_element_type=jnp.float32,
      )
      + bdv0
  )
  b_dvb = jnp.matmul(
      bA.transpose(0, 2, 1),
      bdv,
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  dh_new = dh_tmp * g_exp_last[:, :, None] + jnp.matmul(
      jnp.concatenate([bqg * scale, -bkbg], axis=1).transpose(0, 2, 1),
      jnp.concatenate([bdo.astype(jnp.float32), b_dvb], axis=1),
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  return bdv, b_dvb, dh_new


@partial(jax.jit, static_argnames=["scale", "precision"])
def compute_wy_backward(
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
    scale,
    precision,
    b_dvb=None,
    g_exp=None,
    g_exp_last=None,
    g_exp_decay=None,
    same_doc=None,
):
  BT = bdo.shape[1]
  bh_t = bh.transpose(0, 2, 1)
  dk_acc = jnp.matmul(bvn, dh.transpose(0, 2, 1), precision=precision, preferred_element_type=jnp.float32)
  if b_dvb is None:
    bA_t = bA.transpose(0, 2, 1)
    b_dvb = jnp.matmul(
        bA_t, bdv, precision=precision, preferred_element_type=jnp.float32
    )
  db_acc = (b_dvb * bv).sum(axis=2)
  # One packed multiply against h^T gives both state-side gradients.
  # dkgb = A^T dw = -A^T dv h^T = -b_dvb h^T, so neither b_dw = -dv h^T nor the
  # A^T @ b_dw multiply is needed.
  qk_h = jnp.matmul(
      jnp.concatenate([bdo, b_dvb], axis=1),
      bh_t,
      precision=precision,
      preferred_element_type=jnp.float32,
  )  # [MB, 2BT, K]
  dq_acc = qk_h[:, :BT] * scale
  dkgb = -qk_h[:, BT:]

  if g_exp is None:
    g_exp = jnp.exp2(bg)
  if g_exp_last is None:
    g_exp_last = g_exp[:, BT - 1, :]
  b_dgk = (bh * dh).sum(axis=2) * g_exp_last
  dq_acc = dq_acc * g_exp
  if g_exp_decay is None:
    g_exp_decay = jnp.exp2(bg[:, BT - 1 : BT, :] - bg)
  dk_acc = dk_acc * g_exp_decay

  gb = g_exp * bb[:, :, None]
  kg_local = bk * g_exp
  db_acc = db_acc + (dkgb * kg_local).sum(axis=2)

  kdk = bk * dk_acc
  b_dgk = b_dgk + kdk.sum(axis=1)
  idx = jnp.arange(BT, dtype=jnp.int32)
  m_last = (idx == BT - 1).astype(jnp.float32)
  dg_acc = (bq * dq_acc - kdk + m_last[None, :, None] * b_dgk[:, None, :] + kg_local * dkgb * bb[:, :, None])
  dk_acc = dk_acc + dkgb * gb

  # dAkk in one multiply. With dA = du (v*beta)^T
  # + dw (k*beta*g)^T, u = A v_beta, w = A k_beta_g, dw = -dv h^T and
  # v_new = u - w h:  A^T dA A^T = (A^T dv) (u - w h)^T = b_dvb v_new^T.
  # The strict-lower mask on dA can be dropped because A^T is unit upper
  # triangular, so A^T (upper part) A^T has no strict-lower entries.
  row_l = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
  col_l = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
  m_lower = row_l > col_l
  if same_doc is not None:
    m_lower = m_lower & same_doc
  dAkk_local = jnp.matmul(
      b_dvb,
      bvn.transpose(0, 2, 1),
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  dAkk_local = jnp.where(m_lower[None, :, :], -dAkk_local, 0.0)
  return dq_acc, dk_acc, b_dvb, db_acc, dg_acc, dAkk_local


def _fused_recompute_w_u_vnew_from_h_kernel(
  k_ref,
  v_ref,
  beta_ref,
  A_ref,
  q_ref,
  g_ref,
  h_ref,
  w_ref,
  qg_ref,
  kg_ref,
  v_new_ref,
  *,
  BT,
  K,
  V,
  MB,
  two_pass_bf16: bool = False,
):
  """Compute w, qg, kg, and v_new for MB chunk tiles.

  Per chunk:
      u     = A @ (v * beta)
      w     = A @ (k * beta * exp2(g))
      qg    = q * exp2(g)
      kg    = k * exp2(g_last - g)
      v_new = u - w @ h
  """
  q = q_ref[:]  # [MB, BT, K]
  k = k_ref[:]  # [MB, BT, K]
  v = v_ref[:]  # [MB, BT, V]
  beta = beta_ref[:]  # [MB, BT]
  A = A_ref[:]  # [MB, BT, BT]
  g = g_ref[:]  # [MB, BT, K]
  h = h_ref[:]  # [MB, K, V]

  _, w, v_new, qg, kg = compute_m1_recompute(
      q, k, v, beta, A, g, h, k_ref.dtype, two_pass_bf16=two_pass_bf16
  )

  w_ref[:] = w.astype(w_ref.dtype)
  qg_ref[:] = qg.astype(qg_ref.dtype)
  kg_ref[:] = kg.astype(kg_ref.dtype)
  v_new_ref[:] = v_new.astype(v_new_ref.dtype)


@partial(jax.jit, static_argnames=["chunk_size", "mini_batch", "two_pass_bf16"])
@jaxtyping.jaxtyped
def fused_recompute_w_u_vnew_from_h_pallas(
  q: Float[Array, "H B T K"],
  k: Float[Array, "H B T K"],
  v: Float[Array, "H B T V"],
  beta: Float[Array, "H B T"],
  A: Float[Array, "H B T BT"],
  g: Float[Array, "H B T K"],
  h: Float[Array, "H B NT K V"],
  chunk_size: int = 64,
  mini_batch: int | None = None,
  two_pass_bf16: bool = False,
) -> tuple[
    Float[Array, "H B T K"],
    Float[Array, "H B T K"],
    Float[Array, "H B T K"],
    Float[Array, "H B T V"],
]:
  """Fuses recomputation of w/qg/kg with v_new from a saved state.

  The kernel computes `u` internally and avoids materializing it in HBM.
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT
  total = H * B * NT
  total_orig = total  # saved for unpadding before _ir reshape

  _, _, h_num_chunks, _, _ = h.shape
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  assert A.shape[-1] == BT, (
      f"A.shape[-1]={A.shape[-1]} must equal chunk_size={BT}"
  )
  assert h_num_chunks == NT, f"h has NT={h_num_chunks}, expected {NT}"

  def _r(x, d):
    return x.reshape(total, BT, d)

  q_r = _r(q, K)
  k_r = _r(k, K)
  v_r = _r(v, V)
  g_r = _r(g, K)
  beta_r = beta.reshape(total, BT)
  A_r = A.reshape(-1, BT, BT)
  h_r = h.reshape(total, K, V)

  wy_dtype = (
      jnp.float32
      if (two_pass_bf16 and q.dtype == jnp.bfloat16)
      else q.dtype
  )
  align_minor = get_tpu_limits().block_align_minor
  if mini_batch is None:
    q_elem = jnp.dtype(q.dtype).itemsize
    k_elem = jnp.dtype(k.dtype).itemsize
    v_elem = jnp.dtype(v.dtype).itemsize
    g_elem = jnp.dtype(g.dtype).itemsize
    beta_elem = jnp.dtype(beta.dtype).itemsize
    a_elem = jnp.dtype(A.dtype).itemsize
    h_elem = jnp.dtype(h.dtype).itemsize
    wy_elem = jnp.dtype(wy_dtype).itemsize
    in_bytes = (
        BT * K * (q_elem + k_elem)
        + BT * V * v_elem
        + BT * K * g_elem
        + BT * beta_elem
        + BT * BT * a_elem
        + K * V * h_elem
    )
    out_bytes = (3 * BT * K + BT * V) * wy_elem
    per_chunk = in_bytes + out_bytes
    vmem_budget = get_vmem_limit_bytes()
    hw_mb = max(1, vmem_budget // per_chunk)
    MB = min(hw_mb, total, 32)
    # align upward to block_align_minor (TPU dim=-2 constraint)
    MB = min(align_up(MB, align_minor), total)
    # try to make total divisible by MB; reduce MB first, pad as fallback
    need_pad = False
    if total % MB != 0:
      mb_try = MB
      while mb_try > 1:
        mb_try -= 1
        if total % mb_try == 0 and (mb_try == total or mb_try % align_minor == 0):
          MB = mb_try
          break
      else:
        need_pad = True
  else:
    MB = mini_batch
    assert total % MB == 0, f"total={total} must be divisible by mini_batch={MB}"
    assert MB % align_minor == 0 or MB == total, (
      f"mini_batch={MB} must be divisible by {align_minor} or equal to total={total}"
    )
    need_pad = False

  # ---- pad if needed: round total up to a multiple of MB ----
  if need_pad:
    total_padded = ((total + MB - 1) // MB) * MB
    pad_width = total_padded - total
    q_r = jnp.pad(q_r, [(0, pad_width), (0, 0), (0, 0)])
    k_r = jnp.pad(k_r, [(0, pad_width), (0, 0), (0, 0)])
    v_r = jnp.pad(v_r, [(0, pad_width), (0, 0), (0, 0)])
    g_r = jnp.pad(g_r, [(0, pad_width), (0, 0), (0, 0)])
    beta_r = jnp.pad(beta_r, [(0, pad_width), (0, 0)])
    A_r = jnp.pad(A_r, [(0, pad_width), (0, 0), (0, 0)])
    h_r = jnp.pad(h_r, [(0, pad_width), (0, 0), (0, 0)])
    total = total_padded

  def _spec2(d1, d2=None):
    if d2 is None:
      return pl.BlockSpec(block_shape=(MB, d1), index_map=lambda idx: (idx, 0))
    return pl.BlockSpec(block_shape=(MB, d1, d2), index_map=lambda idx: (idx, 0, 0))

  in_specs = [
    _spec2(BT, K),  # k
    _spec2(BT, V),  # v
    _spec2(BT),     # beta  ← 2D [MB, BT]
    _spec2(BT, BT),  # A
    _spec2(BT, K),  # q
    _spec2(BT, K),  # g
    _spec2(K, V),  # h
  ]
  out_specs = [
    _spec2(BT, K),  # w
    _spec2(BT, K),  # qg
    _spec2(BT, K),  # kg
    _spec2(BT, V),  # v_new
  ]
  out_shape = [
    jax.ShapeDtypeStruct((total, BT, K), wy_dtype),
    jax.ShapeDtypeStruct((total, BT, K), wy_dtype),
    jax.ShapeDtypeStruct((total, BT, K), wy_dtype),
    jax.ShapeDtypeStruct((total, BT, V), wy_dtype),
  ]

  kernel = partial(
    _fused_recompute_w_u_vnew_from_h_kernel,
    BT=BT,
    K=K,
    V=V,
    MB=MB,
    two_pass_bf16=two_pass_bf16,
  )

  w_r, qg_r, kg_r, v_new_r = pl.pallas_call(
    kernel,
    out_shape=out_shape,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=(total // MB,),
      in_specs=in_specs,
      out_specs=out_specs,
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel",),
      disable_bounds_checks=True,
      vmem_limit_bytes=get_tpu_limits().vmem_limit_bytes,
    ),
    interpret=get_interpret(),
  )(k_r, v_r, beta_r, A_r, q_r, g_r, h_r)

  def _ir(x, d):
    return x[:total_orig].reshape(H, B, T, d)

  return (
    _ir(w_r, K),
    _ir(qg_r, K),
    _ir(kg_r, K),
    _ir(v_new_r, V),
  )


# ══════════════════════════════════════════════════════════════════════════
# Shared L1 helpers for the fused backward kernel
# ══════════════════════════════════════════════════════════════════════════


@partial(jax.jit, static_argnames=["precision", "per_channel_gate"])
def compute_intra_backward(
    bq,
    bk,
    bg,
    bb,
    dAqk,
    dAkk,
    dq_acc,
    dk_acc,
    db_acc,
    dg_acc,
    precision,
    per_channel_gate=True,
    decay=None,
    same_doc=None,
    seg_tok=None,
    act_tok=None,
    valid_int=None,
):
  BT = bq.shape[1]
  row_bt = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
  col_bt = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
  causal_mask = row_bt >= col_bt
  strict_mask = row_bt > col_bt
  if same_doc is not None:
    causal_mask = causal_mask & same_doc
    strict_mask = strict_mask & same_doc
  dAqk_full = jnp.where(causal_mask[None], dAqk, 0.0).astype(jnp.float32)
  dAkk_full = jnp.where(strict_mask[None], dAkk, 0.0).astype(jnp.float32)

  if not per_channel_gate:
    # Scalar gate: skip the BC-block decomposition. Blocking is only needed to
    # keep `exp2` bounded for per-channel gates. With a scalar gate,
    # `gs[r] - gs[c] <= 0` under the causal mask, so the full [BT, BT] decay is
    # formed once and folded into the cotangents. This uses 2 packed matmuls
    # (non-contracting / contracting 2*BT) instead of 8 with contraction BC.
    if decay is None:
      gs = bg[:, :, 0]  # [MB, BT]
      _causal = causal_mask[None]
      _diff = gs[:, :, None] - gs[:, None, :]  # [MB, BT, BT]
      decay = jnp.where(_causal, jnp.exp2(jnp.where(_causal, _diff, 0.0)), 0.0)

    dAqk_d = dAqk_full * decay
    dAkk_d = dAkk_full * decay
    _b1 = (((2,), (1,)), ((0,), (0,)))
    _b1t = (((1,), (1,)), ((0,), (0,)))
    _dot = lambda a, b, d: jax.lax.dot_general(
        a, b, d, preferred_element_type=jnp.float32, precision=precision
    )

    dq_dk_row = _dot(jnp.concatenate([dAqk_d, dAkk_d], axis=1), bk, _b1)
    dq_intra = dq_dk_row[:, :BT]
    dk_row_pre = dq_dk_row[:, BT:]
    # beta indexes the row, matching the blocked path's `dAkk_diag * beta_b`.
    dk_col = _dot(
        jnp.concatenate([dAqk_d, dAkk_d * bb[:, :, None]], axis=1),
        jnp.concatenate([bq, bk], axis=1),
        _b1t,
    )

    db_intra = jnp.sum(bk * dk_row_pre, axis=-1)
    dk_row = bb[:, :, None] * dk_row_pre
    dk_intra = dk_row + dk_col
    dg_intra = bq * dq_intra + bk * (dk_row - dk_col)
    return (
        dq_acc + dq_intra,
        dk_acc + dk_intra,
        db_acc + db_intra,
        dg_acc + dg_intra,
    )

  BC = min(16, BT)
  NC = BT // BC
  K = bq.shape[2]
  MB = bq.shape[0]

  def _to_blocks(x):
    return jnp.stack([x[:, i * BC:(i + 1) * BC] for i in range(NC)])
  def _from_blocks(x):
    if NC == 1:
      return x[0]
    return jnp.concatenate([x[i] for i in range(NC)], axis=1)

  q_b = _to_blocks(bq)
  k_b = _to_blocks(bk)
  g_b = _to_blocks(bg)
  beta_b = _to_blocks(bb[:, :, None])

  dAqk_diag = jnp.stack(
    [dAqk_full[:, i * BC:(i + 1) * BC, i * BC:(i + 1) * BC] for i in range(NC)])
  dAkk_diag = jnp.stack(
    [dAkk_full[:, i * BC:(i + 1) * BC, i * BC:(i + 1) * BC] for i in range(NC)])

  # Only per-channel gates reach this point; scalar gates returned above.
  act_b_k = None
  seg_b_k = None
  if seg_tok is not None:
    assert (
        same_doc is not None
        and act_tok is not None
        and valid_int is not None
    )
    bt_1d = jnp.arange(BT, dtype=jnp.int32)
    row_blk = row_bt // BC
    same_blk_doc = (row_blk == (col_bt // BC)) & same_doc
    first_t = jnp.min(jnp.where(same_blk_doc, col_bt, BT), axis=1)
    last_t = jnp.max(jnp.where(same_blk_doc, col_bt, 0), axis=1)
    ref_t = (first_t + last_t) // 2
    ref_t = valid_int * ref_t + (1 - valid_int) * bt_1d
    ref_r = jnp.broadcast_to(ref_t[:, None], (BT, BT))
    pick_ref_bt = (col_bt == ref_r).astype(jnp.float32)
    gn_ref_all = jax.lax.dot_general(
        pick_ref_bt,
        bg,
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).transpose(1, 0, 2)
    g_max = _to_blocks(gn_ref_all)
    act_bt_k = jnp.broadcast_to(act_tok[:, None], (BT, K))
    act_b_k = jnp.stack([act_bt_k[i * BC : (i + 1) * BC] for i in range(NC)])
    seg_bt_k = jnp.broadcast_to(seg_tok[:, None], (BT, K))
    seg_b_k = jnp.stack([seg_bt_k[i * BC : (i + 1) * BC] for i in range(NC)])
    valid_bt_k = jnp.broadcast_to(valid_int[:, None], (BT, K)).astype(
        jnp.float32
    )
    valid_b_f32 = jnp.stack(
        [valid_bt_k[i * BC : (i + 1) * BC] for i in range(NC)]
    )[:, None, :, :]
    row_d = valid_b_f32 * jnp.exp2(valid_b_f32 * (g_b - g_max))
    col_d = valid_b_f32 * jnp.exp2(valid_b_f32 * (g_max - g_b))
  else:
    # Re-center around block max to prevent exp2 overflow across rows.
    g_max = jnp.max(g_b, axis=2, keepdims=True)
    row_d = jnp.exp2(g_b - g_max)
    col_d = jnp.exp2(g_max - g_b)
  k_til = k_b * col_d
  q_hat = q_b * row_d
  k_hat = k_b * beta_b * row_d

  NM = NC * MB
  _b1 = (((2,), (1,)), ((0,), (0,)))
  _b1t = (((1,), (1,)), ((0,), (0,)))
  def _f(x):
    _, _, d2, d3 = x.shape
    return x.reshape(NM, d2, d3)

  _sc_row = lambda x: _f(row_d) * x
  _sc_col = lambda x: _f(col_d) * x
  _dAkk_col = dAkk_diag

  dq_all = _sc_row(
      jax.lax.dot_general(
          _f(dAqk_diag),
          _f(k_til),
          _b1,
          preferred_element_type=jnp.float32,
          precision=precision,
      )
  ).reshape(NC, MB, BC, K)
  dk_row_pre_all = _sc_row(
      jax.lax.dot_general(
          _f(dAkk_diag),
          _f(k_til),
          _b1,
          preferred_element_type=jnp.float32,
          precision=precision,
      )
  ).reshape(NC, MB, BC, K)
  dk_col_all = _sc_col(
      jax.lax.dot_general(
          _f(dAqk_diag),
          _f(q_hat),
          _b1t,
          preferred_element_type=jnp.float32,
          precision=precision,
      )
      + jax.lax.dot_general(
          _f(_dAkk_col),
          _f(k_hat),
          _b1t,
          preferred_element_type=jnp.float32,
          precision=precision,
      )
  ).reshape(NC, MB, BC, K)

  row_pairs = [(ii, ij) for ii in range(NC) for ij in range(ii)]
  if row_pairs:
    Aqk_rp = jnp.stack(
      [dAqk_full[:, ii * BC:(ii + 1) * BC, ij * BC:(ij + 1) * BC]
       for (ii, ij) in row_pairs])
    Akk_rp = jnp.stack(
      [dAkk_full[:, ii * BC:(ii + 1) * BC, ij * BC:(ij + 1) * BC]
       for (ii, ij) in row_pairs])
    k_rp = jnp.stack([k_b[ij] for (_, ij) in row_pairs])
    g_rp = jnp.stack([g_b[ij] for (_, ij) in row_pairs])
    if seg_tok is not None:
      assert act_b_k is not None and seg_b_k is not None
      gn_prev_b = jnp.concatenate(
          [g_b[0:1, :, 0:1, :], g_b[:-1, :, BC - 1 : BC, :]], axis=0
      )
      d_prev_b = jnp.concatenate(
          [jnp.zeros((1, 1, K), dtype=jnp.int32), act_b_k[:-1, BC - 1 : BC, :]],
          axis=0,
      )
      g_ref_rp = jnp.stack(
          [g_b[ii - 1, :, BC - 1 : BC, :] for (ii, _) in row_pairs]
      )
      d_prev_rp = jnp.stack([d_prev_b[ii] for (ii, _) in row_pairs])
      seg_ij_rp = jnp.stack([seg_b_k[ij] for (_, ij) in row_pairs])
      m_ij_rp = ((seg_ij_rp == d_prev_rp) & (d_prev_rp > 0)).astype(
          jnp.float32
      )[:, None, :, :]
      k_dec_rp = m_ij_rp * k_rp * jnp.exp2(m_ij_rp * (g_ref_rp - g_rp))
      m_ii_r = ((seg_b_k == d_prev_b) & (d_prev_b > 0)).astype(jnp.float32)[
          :, None, :, :
      ]
      row_decay = m_ii_r * jnp.exp2(m_ii_r * (g_b - gn_prev_b))
    else:
      g_ref_rp = jnp.stack([g_b[ii, :, 0:1, :] for (ii, _) in row_pairs])
      k_dec_rp = k_rp * jnp.exp2(g_ref_rp - g_rp)
      row_decay = jnp.exp2(g_b - g_b[:, :, 0:1, :])

    NR = len(row_pairs) * MB
    dq_rp = jax.lax.dot_general(
      Aqk_rp.reshape(NR, BC, BC), k_dec_rp.reshape(NR, BC, K),
      _b1, preferred_element_type=jnp.float32, precision=precision,
    ).reshape(len(row_pairs), MB, BC, K)
    dkpre_rp = jax.lax.dot_general(
      Akk_rp.reshape(NR, BC, BC), k_dec_rp.reshape(NR, BC, K),
      _b1, preferred_element_type=jnp.float32, precision=precision,
    ).reshape(len(row_pairs), MB, BC, K)

    dq_row_acc = [q_b[ii] * 0.0 for ii in range(NC)]
    dkpre_row_acc = [k_b[ii] * 0.0 for ii in range(NC)]
    for p_idx, (ii, _) in enumerate(row_pairs):
      dq_row_acc[ii] = dq_row_acc[ii] + dq_rp[p_idx]
      dkpre_row_acc[ii] = dkpre_row_acc[ii] + dkpre_rp[p_idx]
    dq_all = jnp.stack(
      [dq_all[ii] + row_decay[ii] * dq_row_acc[ii] for ii in range(NC)])
    dk_row_pre_all = jnp.stack(
      [dk_row_pre_all[ii] + row_decay[ii] * dkpre_row_acc[ii] for ii in range(NC)])

  col_pairs = [(ii, ij) for ii in range(NC) for ij in range(ii + 1, NC)]
  if col_pairs:
    Aqk_cp = jnp.stack(
      [dAqk_full[:, ij * BC:(ij + 1) * BC, ii * BC:(ii + 1) * BC]
       for (ii, ij) in col_pairs])
    Akk_cp = jnp.stack(
      [dAkk_full[:, ij * BC:(ij + 1) * BC, ii * BC:(ii + 1) * BC]
       for (ii, ij) in col_pairs])
    q_cp = jnp.stack([q_b[ij] for (_, ij) in col_pairs])
    k_cp = jnp.stack([k_b[ij] for (_, ij) in col_pairs])
    g_cp = jnp.stack([g_b[ij] for (_, ij) in col_pairs])
    beta_cp = jnp.stack([beta_b[ij] for (_, ij) in col_pairs])
    if seg_tok is not None:
      assert act_b_k is not None and seg_b_k is not None
      gn_end_b = g_b[:, :, BC - 1 : BC, :]
      d_end_b = jnp.concatenate(
          [act_b_k[:-1, BC - 1 : BC, :], jnp.zeros((1, 1, K), dtype=jnp.int32)],
          axis=0,
      )
      g_ref_cp = jnp.stack(
          [g_b[ii, :, BC - 1 : BC, :] for (ii, _) in col_pairs]
      )
      d_end_cp = jnp.stack([d_end_b[ii] for (ii, _) in col_pairs])
      seg_ij_cp = jnp.stack([seg_b_k[ij] for (_, ij) in col_pairs])
      m_ij_cp = ((seg_ij_cp == d_end_cp) & (d_end_cp > 0)).astype(
          jnp.float32
      )[:, None, :, :]
      row_d_cp = m_ij_cp * jnp.exp2(m_ij_cp * (g_cp - g_ref_cp))
      m_ii_c = ((seg_b_k == d_end_b) & (d_end_b > 0)).astype(jnp.float32)[
          :, None, :, :
      ]
      col_decay = m_ii_c * jnp.exp2(m_ii_c * (gn_end_b - g_b))
    else:
      g_ref_cp = jnp.stack([g_b[ii, :, BC - 1:BC, :] for (ii, _) in col_pairs])
      row_d_cp = jnp.exp2(g_cp - g_ref_cp)
      col_decay = jnp.exp2(g_b[:, :, BC - 1:BC, :] - g_b)
    q_dec_cp = q_cp * row_d_cp
    kb_dec_cp = k_cp * beta_cp * row_d_cp

    NCP = len(col_pairs) * MB
    dk_col_cp = (
      jax.lax.dot_general(Aqk_cp.reshape(NCP, BC, BC),
                          q_dec_cp.reshape(NCP, BC, K), _b1t,
                          preferred_element_type=jnp.float32, precision=precision)
      + jax.lax.dot_general(Akk_cp.reshape(NCP, BC, BC),
                            kb_dec_cp.reshape(NCP, BC, K), _b1t,
                            preferred_element_type=jnp.float32, precision=precision)
    ).reshape(len(col_pairs), MB, BC, K)

    dk_col_acc = [k_b[ii] * 0.0 for ii in range(NC)]
    for p_idx, (ii, _) in enumerate(col_pairs):
      dk_col_acc[ii] = dk_col_acc[ii] + dk_col_cp[p_idx]
    dk_col_all = jnp.stack(
      [dk_col_all[ii] + col_decay[ii] * dk_col_acc[ii] for ii in range(NC)])

  dq_intra = _from_blocks(dq_all)
  dk_row_pre = _from_blocks(dk_row_pre_all)
  dk_col = _from_blocks(dk_col_all)
  db_intra = jnp.sum(bk * dk_row_pre, axis=-1)
  dk_row = bb[:, :, None] * dk_row_pre
  dk_intra = dk_row + dk_col
  dg_intra = bq * dq_intra + bk * (dk_row - dk_col)

  dq_total = dq_acc + dq_intra
  dk_total = dk_acc + dk_intra
  db_total = db_acc + db_intra
  dg_total = dg_acc + dg_intra
  return dq_total, dk_total, db_total, dg_total


@jax.jit
def compute_reverse_cumsum_dg(dg_total, act_tok=None):
  BT = dg_total.shape[1]
  row = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
  col = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
  mask_b = row <= col
  if act_tok is not None:
    act_r = jnp.broadcast_to(act_tok[:, None], (BT, BT))
    act_c = jnp.broadcast_to(act_tok[None, :], (BT, BT))
    mask_b = mask_b & (act_r == act_c)
  cumsum_mask = mask_b.astype(jnp.float32)
  return jax.lax.dot_general(
    cumsum_mask, dg_total, (((1,), (1,)), ((), ())),
    precision=jax.lax.Precision.HIGHEST,
  ).transpose(1, 0, 2)


@jax.jit
def compute_forward_cumsum_g(g_raw, act_tok=None):
  BT = g_raw.shape[1]
  row = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
  col = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
  mask_b = row >= col
  if act_tok is not None:
    act_r = jnp.broadcast_to(act_tok[:, None], (BT, BT))
    act_c = jnp.broadcast_to(act_tok[None, :], (BT, BT))
    mask_b = mask_b & (act_r == act_c)
  cumsum_mask = mask_b.astype(jnp.float32)
  return jax.lax.dot_general(
      cumsum_mask,
      g_raw,
      (((1,), (1,)), ((), ())),
      precision=jax.lax.Precision.HIGHEST,
  ).transpose(1, 0, 2)


# =====================================================================
# M4: dhu + WY + intra backward + chunk-local reverse cumsum
# =====================================================================


def _fused_dhu_wy_intra_cumsum_kernel(
    step,
    # Inputs. All refs except the first two are single VMEM blocks from
    # emit_pipeline; optional inputs absent from the pipeline arrive as None.
    chunk_seg_ids_smem_ref,  # [B, NT] i32 when is_varlen, else None (SMEM)
    keep_smem_ref,  # [B, NT] i32 when packed, else None (SMEM)
    q_ref,  # [MB_QK, 1, 1, BT, K], or [1, BT, MB_QK * K] with
    # native_qkv_do_layout
    k_ref,  # [MB_QK, 1, 1, BT, K], or [1, BT, MB_QK * K] with
    # native_qkv_do_layout
    v_ref,  # [MB, 1, 1, BT, V], or [1, BT, MB * V] with native_qkv_do_layout
    v_new_ref,  # [MB, 1, 1, BT, V] or None
    qg_ref,  # [MB, 1, 1, BT, K] or None
    kg_ref,  # [MB, 1, 1, BT, K] or None
    w_ref,  # [MB, 1, 1, BT, K] or None
    g_ref,  # [MB, 1, 1, 1, BT]  scalar gate, width-1 axis kept off
    # lanes; else [MB, 1, 1, BT, K]
    beta_ref,  # [MB, 1, 1, 1, BT]  width-1 axis kept off lanes
    A_ref,  # [MB, 1, 1, BT, BT], or [1, 1, MB, BT, BT] with
    # native_state_layout
    h_ref,  # [MB, 1, 1, K, V], or [1, 1, MB, K, V] with
    # native_state_layout
    do_ref,  # [MB, 1, 1, BT, V], or [1, BT, MB * V] with native_qkv_do_layout
    dv0_ref,  # [MB, 1, 1, BT, V] or None
    dAqk_ref,  # [MB, 1, 1, BT, BT] or None
    dht_ref,  # [MB, 1, 1, K, V] or None  block of this chunk's segment
    rstd_q_ref,  # [MB_QK, 1, 1, 1, BT] or None
    rstd_k_ref,  # [MB_QK, 1, 1, 1, BT] or None
    meta_ref,  # [1, 1, 8, BT] i32 or None
    # Outputs.
    dq_ref,  # [MB_QK, 1, 1, BT, K], or [1, BT, MB_QK * K] with
    # native_qkv_do_layout; dqkv_dtype (f32 by default)
    dk_ref,  # [MB_QK, 1, 1, BT, K], or [1, BT, MB_QK * K] with
    # native_qkv_do_layout; dqkv_dtype (f32 by default)
    dv_ref,  # [MB, 1, 1, BT, V], or [1, BT, MB * V] with
    # native_qkv_do_layout; dqkv_dtype (f32 by default)
    db_ref,  # [MB, 1, 1, 1, BT] f32
    dg_ref,  # [MB, 1, 1, 1, BT] f32 scalar gate; else [MB, 1, 1, BT, K]
    dh0_ref,  # [MB, 1, 1, K, V] f32 or None  block of this chunk's segment
    # Scratch.
    dh_scratch,  # [MB, K, V] f32  dh carried across the reversed chunk axis
    *,
    BT,
    K,
    V,
    NT,
    scale,
    MB,
    per_channel_gate=True,
    gqa_repeats=1,
    native_state_layout=False,
    native_qkv_do_layout=False,
    use_qk_l2norm_in_kernel=False,
    g_is_cumsummed=False,
    is_varlen=True,
    two_pass_bf16=False,
):
  """Fuse Dhu, WY, intra backward, and reverse cumsum for one chunk tile."""
  _, batch_idx, rev_c = step.index
  chunk_id = NT - 1 - rev_c
  if is_varlen:
    _, _, is_valid, is_first_chunk, is_last_chunk = _chunk_segment_metadata(
        chunk_seg_ids_smem_ref,
        batch_idx,
        chunk_id,
        NT,
    )
  else:
    is_valid = jnp.bool_(True)
    is_last_chunk = rev_c == 0
    is_first_chunk = rev_c == NT - 1
  if q_ref.dtype != jnp.bfloat16:
    precision = jax.lax.Precision.HIGHEST
  elif two_pass_bf16:
    precision = jax.lax.Precision.HIGH
  else:
    precision = jax.lax.Precision.DEFAULT

  @pl.when(~is_valid)
  def _():
    dh_scratch[:] = jnp.zeros_like(dh_scratch)

  seg_tok = act_tok = valid_int = None
  valid_f32 = m_in_f32 = m_out_f32 = None
  same_doc = None
  if meta_ref is not None:
    meta_tile = meta_ref[0, 0]  # [8, BT] int32
    seg_tok = meta_tile[0]
    act_tok = meta_tile[1]
    m_in_f32 = meta_tile[2].astype(jnp.float32)
    m_out_f32 = meta_tile[3].astype(jnp.float32)
    valid_int = meta_tile[4]
    valid_f32 = valid_int.astype(jnp.float32)
    seg_r = jnp.broadcast_to(seg_tok[:, None], (BT, BT))
    seg_c = jnp.broadcast_to(seg_tok[None, :], (BT, BT))
    same_doc = (seg_r == seg_c) & (seg_r > 0)
  @pl.when(is_last_chunk)
  def _():
    if dht_ref is None:
      dh_scratch[:] = jnp.zeros_like(dh_scratch)
    else:
      dh_scratch[:] = dht_ref[:, 0, 0, :].astype(dh_scratch.dtype)

  dh = dh_scratch[:].astype(jnp.float32)
  # Native GQA: q and k arrive at the key head count (MB // gqa_repeats heads
  # per tile) and are broadcast to the MB value heads in VMEM, so no repeated
  # copy of q and k is needed in HBM.
  MB_QK = MB // gqa_repeats
  if native_qkv_do_layout:
    # Unpack lane-aligned [BT, D] head slices from [1, BT, M * D] windows
    # directly into [M, BT, D] without any VMEM transpose.
    bq_raw = jnp.stack(
        [q_ref[0, :, h * K : (h + 1) * K] for h in range(MB_QK)], axis=0
    ).astype(jnp.float32)
    bk_raw = jnp.stack(
        [k_ref[0, :, h * K : (h + 1) * K] for h in range(MB_QK)], axis=0
    ).astype(jnp.float32)
    bv = jnp.stack(
        [v_ref[0, :, h * V : (h + 1) * V] for h in range(MB)], axis=0
    ).astype(jnp.float32)
    bdo = jnp.stack(
        [do_ref[0, :, h * V : (h + 1) * V] for h in range(MB)], axis=0
    ).astype(jnp.float32)
  else:
    bq_raw = q_ref[:, 0, 0].astype(jnp.float32)
    bk_raw = k_ref[:, 0, 0].astype(jnp.float32)
    bv = v_ref[:, 0, 0].astype(jnp.float32)
    bdo = do_ref[:, 0, 0].astype(jnp.float32)

  if valid_f32 is not None:
    valid_3d = valid_f32[None, :, None]
    bq_raw = bq_raw * valid_3d
    bk_raw = bk_raw * valid_3d
    bv = bv * valid_3d
    bdo = bdo * valid_3d

  brstd_q = brstd_k = None
  bq_unit = bk_unit = None
  if use_qk_l2norm_in_kernel:
    q_sq = jnp.sum(bq_raw * bq_raw, axis=-1, keepdims=True)
    k_sq = jnp.sum(bk_raw * bk_raw, axis=-1, keepdims=True)
    brstd_q = jnp.where(
        q_sq == 0.0,
        0.0,
        jax.lax.rsqrt(jnp.where(q_sq == 0.0, 1.0, q_sq) + 1e-6),
    )
    brstd_k = jnp.where(
        k_sq == 0.0,
        0.0,
        jax.lax.rsqrt(jnp.where(k_sq == 0.0, 1.0, k_sq) + 1e-6),
    )
    bq_unit = bq_raw * brstd_q
    bk_unit = bk_raw * brstd_k
    bq_k = bq_unit.astype(q_ref.dtype).astype(jnp.float32)
    bk_k = bk_unit.astype(k_ref.dtype).astype(jnp.float32)
  else:
    bq_k = bq_raw
    bk_k = bk_raw

  if gqa_repeats > 1:
    bq = jnp.repeat(bq_k, gqa_repeats, axis=0)
    bk = jnp.repeat(bk_k, gqa_repeats, axis=0)
  else:
    bq = bq_k
    bk = bk_k

  # Transpose scalar gate from [.., 1, BT] back to [MB, BT, 1].
  if g_ref.shape[-2] == 1:
    bg = g_ref[:, 0, 0, 0].astype(jnp.float32)[..., None]
  else:
    bg = g_ref[:, 0, 0].astype(jnp.float32)
  if v_new_ref is None and not g_is_cumsummed:
    if valid_f32 is not None:
      bg = bg * valid_f32[None, :, None]
    if bg.shape[-1] == 1:
      row_g = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
      col_g = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
      _fwd_b = row_g <= col_g
      if act_tok is not None:
        act_r = jnp.broadcast_to(act_tok[:, None], (BT, BT))
        act_c = jnp.broadcast_to(act_tok[None, :], (BT, BT))
        _fwd_b = _fwd_b & (act_r == act_c)
      _fwd_mask = _fwd_b.astype(jnp.float32)
      bg = jnp.matmul(
          bg[..., 0],
          _fwd_mask,
          precision=jax.lax.Precision.HIGHEST,
          preferred_element_type=jnp.float32,
      )[..., None]
    else:
      bg = compute_forward_cumsum_g(bg, act_tok=act_tok)
  bg_scalar = bg
  _gate_narrow = bg.shape[-1] != K
  if m_in_f32 is not None:
    m_in_3d = m_in_f32[None, :, None]
    g_exp = m_in_3d * jnp.exp2(m_in_3d * bg)
  else:
    g_exp = jnp.exp2(bg)
  if keep_smem_ref is not None:
    keep_f32 = keep_smem_ref[batch_idx, chunk_id].astype(jnp.float32)
    g_exp_last = keep_f32 * jnp.exp2(keep_f32 * bg[:, BT - 1, :])
  else:
    g_exp_last = g_exp[:, BT - 1, :]
  if m_out_f32 is not None:
    m_out_3d = m_out_f32[None, :, None]
    g_exp_decay = m_out_3d * jnp.exp2(
        m_out_3d * (bg[:, BT - 1 : BT, :] - bg)
    )
  else:
    g_exp_decay = None
  bb = beta_ref[:, 0, 0, 0].astype(jnp.float32)
  if valid_f32 is not None:
    bb = bb * valid_f32[None, :]
  if native_state_layout:
    bh = h_ref[0, 0].astype(jnp.float32)
  else:
    bh = h_ref[:, 0, 0].astype(jnp.float32)

  decay_mat = None
  m_causal = None
  m_strict = None
  qk_dot_cached = None
  if v_new_ref is None or A_ref is None:
    row_bt = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
    col_bt = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
    m_causal = row_bt >= col_bt
    m_strict = row_bt > col_bt
    if same_doc is not None:
      m_causal = m_causal & same_doc
      m_strict = m_strict & same_doc
    diff_g = bg_scalar - jnp.swapaxes(bg_scalar, 1, 2)
    safe_diff = jnp.where(m_causal[None, :, :], diff_g, 0.0)
    decay_mat = jnp.where(m_causal[None, :, :], jnp.exp2(safe_diff), 0.0)

  if A_ref is None:
    if v_new_ref is None:
      qk_kk_dot = jax.lax.dot_general(
          jnp.concatenate([bq_k, bk_k], axis=1),
          bk_k,
          (((2,), (2,)), ((0,), (0,))),
          precision=precision,
          preferred_element_type=jnp.float32,
      )
      qk_dot_cached = qk_kk_dot[:, :BT]
      kk_dot = qk_kk_dot[:, BT:]
    else:
      kk_dot = jax.lax.dot_general(
          bk_k,
          bk_k,
          (((2,), (2,)), ((0,), (0,))),
          precision=precision,
          preferred_element_type=jnp.float32,
      )
    if gqa_repeats > 1:
      kk_dot = jnp.repeat(kk_dot, gqa_repeats, axis=0)
    assert decay_mat is not None and m_strict is not None
    L_bwd = jnp.where(
        m_strict[None, :, :],
        kk_dot * decay_mat * bb[:, :, None],
        0.0,
    )
    bA = tri_inv_xor_dnc(
        L_bwd,
        use_fp32=(q_ref.dtype == jnp.float32),
        two_pass_bf16=two_pass_bf16,
    )
  elif native_state_layout:
    # A and h blocks are [1, 1, MB, ...] in the forward's [B, NC, H, ...] layout.
    bA = A_ref[0, 0].astype(jnp.float32)
  else:
    bA = A_ref[:, 0, 0].astype(jnp.float32)

  bkbg = None
  if qg_ref is not None and kg_ref is not None and w_ref is not None:
    bqg = qg_ref[:, 0, 0].astype(jnp.float32)
    bkg = kg_ref[:, 0, 0].astype(jnp.float32)
    bw = w_ref[:, 0, 0].astype(jnp.float32)
  else:
    if g_exp_decay is None:
      g_exp_decay = jnp.exp2(bg[:, BT - 1 : BT, :] - bg)
    if two_pass_bf16:
      bqg = bq * g_exp
      bkg = bk * g_exp_decay
    else:
      bqg = (bq * g_exp).astype(q_ref.dtype).astype(jnp.float32)
      bkg = (bk * g_exp_decay).astype(k_ref.dtype).astype(jnp.float32)
    bkbg = bk * (bb[:, :, None] * g_exp)  # k * beta * g, f32
    if v_new_ref is None:
      # w-free path: w = A @ k_beta_g is never formed.
      bw = None
    else:
      bw = jnp.matmul(
          bA,
          bkbg,
          precision=precision,
          preferred_element_type=jnp.float32,
      )
      if valid_f32 is not None:
        bw = bw * valid_f32[None, :, None]
      if not two_pass_bf16:
        bw = bw.astype(k_ref.dtype).astype(jnp.float32)

  bdv0 = None
  bdAqk = None
  if v_new_ref is not None:
    bvn = v_new_ref[:, 0, 0].astype(jnp.float32)
    bdv0 = dv0_ref[:, 0, 0].astype(jnp.float32)
    bdAqk = dAqk_ref[:, 0, 0].astype(jnp.float32)
    if valid_f32 is not None:
      valid_3d = valid_f32[None, :, None]
      bvn = bvn * valid_3d
      bdv0 = bdv0 * valid_3d
  elif bw is None:
    # Two-matmul v_new = A @ (v * beta - k_beta_g @ h); no u, no w.
    assert bkbg is not None
    bvn = jnp.matmul(
        bA,
        bv * bb[:, :, None]
        - jnp.matmul(
            bkbg,
            bh,
            precision=precision,
            preferred_element_type=jnp.float32,
        ),
        precision=precision,
        preferred_element_type=jnp.float32,
    )
    if valid_f32 is not None:
      bvn = bvn * valid_f32[None, :, None]
  else:
    bu = jnp.matmul(
        bA,
        (bv * bb[:, :, None]).astype(jnp.float32),
        precision=precision,
        preferred_element_type=jnp.float32,
    )
    if valid_f32 is not None:
      bu = bu * valid_f32[None, :, None]
    if not two_pass_bf16:
      bu = bu.astype(v_ref.dtype).astype(jnp.float32)
    bvn = bu - jnp.matmul(
        bw,
        bh,
        precision=precision,
        preferred_element_type=jnp.float32,
    )
    if valid_f32 is not None:
      bvn = bvn * valid_f32[None, :, None]
  if v_new_ref is None:
    assert m_causal is not None and decay_mat is not None
    # Compute q @ k^T on the MB_QK key heads before repeating across GQA groups.
    if qk_dot_cached is not None:
      qk_dot = qk_dot_cached
    else:
      qk_dot = jax.lax.dot_general(
          bq_k,
          bk_k,
          (((2,), (2,)), ((0,), (0,))),
          precision=precision,
          preferred_element_type=jnp.float32,
      )
    if gqa_repeats > 1:
      qk_dot = jnp.repeat(qk_dot, gqa_repeats, axis=0)
    bAqk = qk_dot * decay_mat * scale
    do_vn_dot = jax.lax.dot_general(
        bdo,
        bvn,
        (((2,), (2,)), ((0,), (0,))),
        precision=precision,
        preferred_element_type=jnp.float32,
    )
    bdAqk = jnp.where(m_causal[None, :, :], do_vn_dot * scale, 0.0)
    bdv0 = jax.lax.dot_general(
        bAqk,
        bdo,
        (((1,), (1,)), ((0,), (0,))),
        precision=precision,
        preferred_element_type=jnp.float32,
    )

  # --- dhu reverse recurrence ---
  assert bdv0 is not None and bdAqk is not None
  if bw is None:
    # w-free chain: w^T dv = k_beta_g^T (A^T dv) = k_beta_g^T b_dvb.
    assert bkbg is not None
    bdv, b_dvb, dh_new = compute_dhu_recurrence_wfree(
        bkg,
        dh,
        bdv0,
        dh_scratch[:],
        g_exp_last,
        bqg,
        bkbg,
        bA,
        bdo,
        scale,
        precision,
    )
  else:
    bdv, dh_new = compute_dhu_recurrence(
        bkg,
        dh,
        bdv0,
        dh_scratch[:],
        g_exp_last,
        bqg,
        bw,
        bdo,
        scale,
        precision,
    )
    b_dvb = None
  dh_scratch[:] = dh_new

  # --- WY backward ---
  dq_acc, dk_acc, b_dvb, db_acc, dg_acc, dAkk_local = compute_wy_backward(
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
      scale,
      precision,
      b_dvb=b_dvb,
      g_exp=g_exp,
      g_exp_last=g_exp_last,
      g_exp_decay=g_exp_decay,
      same_doc=same_doc,
  )

  # --- Intra backward + reverse cumsum ---
  dq_total, dk_total, db_total, dg_total = compute_intra_backward(
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
      precision=precision,
      per_channel_gate=per_channel_gate,
      decay=decay_mat,
      same_doc=same_doc,
      seg_tok=seg_tok,
      act_tok=act_tok,
      valid_int=valid_int,
  )
  if _gate_narrow:
    # Scalar gate: all K channels share one gate, so sum the gate gradient over
    # K first (exact by linearity) and run the reverse cumsum on [MB, BT]
    # instead of [MB, BT, K].
    dg_scalar = jnp.sum(dg_total, axis=-1)  # [MB, BT]
    row_rev = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
    col_rev = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
    # _rev_mask[s, t] = 1 where s >= t, so dg_scalar @ _rev_mask is the
    # reverse cumsum. The trailing axis restores the [MB, BT, 1] gate shape.
    rev_b = row_rev >= col_rev
    if act_tok is not None:
      act_r = jnp.broadcast_to(act_tok[:, None], (BT, BT))
      act_c = jnp.broadcast_to(act_tok[None, :], (BT, BT))
      rev_b = rev_b & (act_r == act_c)
    _rev_mask = rev_b.astype(jnp.float32)
    dg_reverse_cumsum = jnp.matmul(
        dg_scalar,
        _rev_mask,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )[..., None]
  else:
    dg_reverse_cumsum = compute_reverse_cumsum_dg(dg_total, act_tok=act_tok)

  # Sum value heads back to key heads in VMEM before writing dq and dk.
  if gqa_repeats > 1:
    _red = lambda x: x.reshape(MB_QK, gqa_repeats, *x.shape[1:]).sum(axis=1)
    dq_total = _red(dq_total)
    dk_total = _red(dk_total)
  if use_qk_l2norm_in_kernel:
    assert brstd_q is not None and brstd_k is not None
    assert bq_unit is not None and bk_unit is not None
    dq_total = brstd_q * (
        dq_total - bq_unit * jnp.sum(bq_unit * dq_total, axis=-1, keepdims=True)
    )
    dk_total = brstd_k * (
        dk_total - bk_unit * jnp.sum(bk_unit * dk_total, axis=-1, keepdims=True)
    )
  elif rstd_q_ref is not None and rstd_k_ref is not None:
    bq_norm = bq_k
    bk_norm = bk_k
    brstd_q = rstd_q_ref[:, 0, 0, 0].astype(jnp.float32)[..., None]
    brstd_k = rstd_k_ref[:, 0, 0, 0].astype(jnp.float32)[..., None]
    dq_total = brstd_q * (
        dq_total - bq_norm * jnp.sum(bq_norm * dq_total, axis=-1, keepdims=True)
    )
    dk_total = brstd_k * (
        dk_total - bk_norm * jnp.sum(bk_norm * dk_total, axis=-1, keepdims=True)
    )
  dv_total = b_dvb * bb[:, :, None]
  if valid_f32 is not None:
    valid_3d = valid_f32[None, :, None]
    dq_total = dq_total * valid_3d
    dk_total = dk_total * valid_3d
    dv_total = dv_total * valid_3d
    db_total = db_total * valid_f32[None, :]
    dg_reverse_cumsum = dg_reverse_cumsum * valid_3d
  if native_qkv_do_layout:
    dq_cast = dq_total.astype(dq_ref.dtype)
    dk_cast = dk_total.astype(dk_ref.dtype)
    dv_cast = dv_total.astype(dv_ref.dtype)
    dq_ref[...] = jnp.concatenate([dq_cast[h] for h in range(MB_QK)], axis=-1)[
        None, ...
    ]
    dk_ref[...] = jnp.concatenate([dk_cast[h] for h in range(MB_QK)], axis=-1)[
        None, ...
    ]
    dv_ref[...] = jnp.concatenate([dv_cast[h] for h in range(MB)], axis=-1)[
        None, ...
    ]
  else:
    dq_ref[:, 0, 0] = dq_total.astype(dq_ref.dtype)
    dk_ref[:, 0, 0] = dk_total.astype(dk_ref.dtype)
    dv_ref[:, 0, 0] = dv_total.astype(dv_ref.dtype)
  db_ref[:, 0, 0, 0] = db_total.astype(db_ref.dtype)
  if dg_ref.shape[-2] == 1:
    dg_ref[:, 0, 0, 0] = dg_reverse_cumsum[..., 0].astype(dg_ref.dtype)
  else:
    dg_ref[:, 0, 0] = dg_reverse_cumsum.astype(dg_ref.dtype)

  if dh0_ref is not None:

    @pl.when(is_first_chunk)
    def _():
      dh0_ref[:, 0, 0, :] = dh_scratch[:].astype(dh0_ref.dtype)


@partial(
    jax.jit,
    static_argnames=[
        "chunk_size",
        "use_exp2",
        "scale",
        "mini_batch",
        "return_dh0",
        "max_num_segments",
        "per_channel_gate",
        "gqa_repeats",
        "dqkv_dtype",
        "native_state_layout",
        "native_bt_qkv",
        "use_qk_l2norm_in_kernel",
        "g_is_cumsummed",
        "two_pass_bf16",
    ],
)
@jaxtyping.jaxtyped
def _fused_dhu_wy_intra_cumsum_pallas_jit(
    # `H_QK` is `H // gqa_repeats`, and equals H when gqa_repeats is 1.
    # When `qkv_conv` is passed, `q`, `k`, and `v` may be None and the kernel
    # reads `qkv_conv [B, T, 2*H_QK*K + H*V]` and `do [B, T, H, V]` in place
    # and writes `dy_conv [B, T, 2*H_QK*K + H*V]` directly.
    q: Float[Array, "..."] | None,
    k: Float[Array, "..."] | None,
    v: Float[Array, "..."] | None,
    v_new: Float[Array, "H B T V"] | None,
    qg: Float[Array, "H B T K"] | None,
    kg: Float[Array, "H B T K"] | None,
    w: Float[Array, "H B T K"] | None,
    g: Float[Array, "H B T GW"],
    beta: Float[Array, "H B T"],
    # [H, B, T, BT] and [H, B, NT, K, V], or with native_state_layout=True the
    # forward's own [B, NT, H, BT, BT] and [B, NT, H, K, V] (no transpose).
    # `A` may be None on scalar-gate paths to rebuild `(I + L)^{-1}` in VMEM.
    A: Float[Array, "..."] | None,
    h: Float[Array, "..."],
    # [H, B, T, V], or [B, T, H, V] / [B, T, H * V] when qkv_conv is passed.
    do: Float[Array, "..."],
    dv0: Float[Array, "H B T V"] | None,
    dAqk: Float[Array, "H B T BT"] | None,
    dht: Float[Array, "B N H K V"] | None,
    scale: float,
    *,
    segment_ids: Int[Array, "B T"] | None = None,
    chunk_size: int = 64,
    use_exp2: bool = True,
    per_channel_gate: bool = True,
    mini_batch: int | None = None,
    return_dh0: bool = True,
    max_num_segments: int | None = None,
    gqa_repeats: int = 1,
    rstd_q: Float[Array, "H_QK B T ..."] | None = None,
    rstd_k: Float[Array, "H_QK B T ..."] | None = None,
    dqkv_dtype=None,
    native_state_layout: bool = False,
    native_bt_qkv: bool = False,
    qkv_conv: Float[Array, "B T C"] | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    g_is_cumsummed: bool = False,
    two_pass_bf16: bool = False,
    packed_tok_meta: Int[Array, "B NT 8 BT"] | None = None,
    packed_chunk_keep: Int[Array, "B NT"] | None = None,
) -> tuple[
    Float[Array, "..."],
    Float[Array, "..."] | None,
    Float[Array, "..."] | None,
    Float[Array, "H B T"],
    Float[Array, "H B T GW"],
    Float[Array, "B N_OUT H K V"] | None,
]:
  """Fuses Dhu recurrence, WY backward, intra backward, and gate cumsum.

  `g` is expected in base-2 (`log2`) units, pre-cumsum when `v_new is None and
  not g_is_cumsummed` (the kernel computes the chunk-local forward cumsum in
  VMEM) and post-cumsum when `v_new is not None or g_is_cumsummed`. The
  returned `dg` is the gradient with respect to the natural-log gate
  `g_ln = g * ln(2)`.

  When `use_qk_l2norm_in_kernel=True`, `q` and `k` (or the `q`/`k` slices of
  `qkv_conv`) must be unnormalized; the kernel applies both L2-normalization
  and its analytical backward VJP in VMEM.

  `segment_ids` applies per-batch varlen boundaries, and `return_dh0`
  controls whether the initial-state gradient is materialized.
  """
  # `q` and `k` carry the key head count; everything else carries the value
  # head count. H below is always the value head count.
  *_, K, V = h.shape
  H, _, _, GW = g.shape
  native_aliased_qkv = qkv_conv is not None
  native_qkv_do_layout = native_aliased_qkv or native_bt_qkv
  if native_aliased_qkv:
    if native_bt_qkv:
      raise ValueError("native_bt_qkv must be False when qkv_conv is provided.")
    if q is not None or k is not None or v is not None:
      raise ValueError("q, k, and v must be None when qkv_conv is provided.")
    B, T, C = qkv_conv.shape
    if H % gqa_repeats != 0:
      raise ValueError(f"H={H} must be divisible by gqa_repeats={gqa_repeats}.")
    H_QK = H // gqa_repeats
    act_dtype = qkv_conv.dtype
  else:
    if q is None or k is None or v is None:
      raise ValueError(
          "q, k, and v must all be provided when qkv_conv is None."
      )
    if native_bt_qkv:
      B, T, H_QK, K = q.shape
    else:
      H_QK, B, T, K = q.shape
    H = H_QK * gqa_repeats
    V = v.shape[-1]
    C = 2 * H_QK * K + H * V
    act_dtype = q.dtype
  BT = chunk_size
  NT = T // BT
  q_size = H_QK * K
  k_size = H_QK * K
  v_size = H * V
  if native_qkv_do_layout:
    if native_aliased_qkv and C != q_size + k_size + v_size:
      raise ValueError(f"qkv_conv.shape[-1]={C} != {q_size + k_size + v_size}.")
    if K % 128 != 0 or V % 128 != 0:
      raise NotImplementedError(
          f"native qkv layout requires K={K}, V={V} multiples of 128."
      )
    if BT % 8 != 0:
      raise NotImplementedError(
          f"native qkv layout requires chunk_size={BT} divisible by 8."
      )
    if do.shape not in ((B, T, H, V), (B, T, v_size)):
      raise ValueError(
          f"do.shape={do.shape}, expected {(B, T, H, V)} or {(B, T, v_size)}."
      )
  else:
    if do.shape != (H, B, T, V):
      raise ValueError(f"do.shape={do.shape}, expected {(H, B, T, V)}.")

  is_varlen = segment_ids is not None
  if (packed_tok_meta is None) != (packed_chunk_keep is None):
    raise ValueError(
        "packed_tok_meta and packed_chunk_keep must be both provided or both"
        " None."
    )
  has_packed_meta = packed_tok_meta is not None
  if is_varlen and has_packed_meta:
    raise ValueError(
        "segment_ids (aligned varlen) and packed_tok_meta are mutually"
        " exclusive."
    )
  if T % BT != 0:
    raise ValueError(f"T={T} must be divisible by chunk_size={BT}.")
  if A is None and per_channel_gate:
    raise NotImplementedError(
        "In-VMEM Akk rebuild (A=None) only supports per_channel_gate=False."
    )
  if A is not None and A.shape[-1] != BT:
    raise ValueError(f"A.shape[-1]={A.shape[-1]} must equal chunk_size={BT}.")
  if native_state_layout:
    if A is not None and A.shape != (B, NT, H, BT, BT):
      raise ValueError(f"A.shape={A.shape}, expected {(B, NT, H, BT, BT)}.")
    if h.shape != (B, NT, H, K, V):
      raise ValueError(f"h.shape={h.shape}, expected {(B, NT, H, K, V)}.")
  else:
    if A is not None and A.shape not in ((H, B, T, BT), (H, B, NT, BT, BT)):
      raise ValueError(
          f"A.shape={A.shape}, expected {(H, B, T, BT)} or"
          f" {(H, B, NT, BT, BT)}."
      )
    if h.shape != (H, B, NT, K, V):
      raise ValueError(f"h.shape={h.shape}, expected {(H, B, NT, K, V)}.")
  if not use_exp2:
    raise NotImplementedError(
        "fused_dhu_wy_intra_cumsum_pallas currently expects log2 gates and"
        " use_exp2=True."
    )

  chunk_seg_ids = segment_ids.reshape(B, NT, BT)[:, :, 0] if is_varlen else None
  if dht is not None:
    N = dht.shape[1]  # dht is [B, N, H, K, V], N is dim 1
  elif is_varlen:
    N = max_num_segments
  else:
    N = 1  # uniform: one sequence per batch element
  if dht is not None and dht.shape[1] != N:
    raise ValueError(f"dht has N={dht.shape[1]}, expected {N}.")

  if not ((qg is None) == (kg is None) == (w is None)):
    raise ValueError(
        "qg, kg, and w must either all be provided or all be None."
    )
  if not ((v_new is None) == (dv0 is None) == (dAqk is None)):
    raise ValueError(
        "v_new, dv0, and dAqk must either all be provided or all be None."
    )
  if not ((rstd_q is None) == (rstd_k is None)):
    raise ValueError(
        "rstd_q and rstd_k must either both be provided or both be None."
    )
  has_qg_kg_w = qg is not None
  has_vn_dv0_dAqk = v_new is not None
  has_rstd_qk = rstd_q is not None
  if use_qk_l2norm_in_kernel and has_rstd_qk:
    raise ValueError(
        "rstd_q and rstd_k must be None when use_qk_l2norm_in_kernel=True."
    )
  if not has_vn_dv0_dAqk and per_channel_gate:
    raise NotImplementedError(
        "Inline v_new/dv0/dAqk recomputation (v_new=None) only supports"
        " per_channel_gate=False."
    )
  expected_gw = K if per_channel_gate else 1
  if GW != expected_gw:
    raise ValueError(
        f"g.shape[-1]={GW} does not match expected {expected_gw} for"
        f" per_channel_gate={per_channel_gate}."
    )
  scalar_gate = not per_channel_gate
  dqkv_out_dtype = jnp.float32 if dqkv_dtype is None else dqkv_dtype
  has_A = A is not None
  if mini_batch is None:
    elem_size = 2 if act_dtype == jnp.bfloat16 else 4
    dqkv_elem_size = 2 if dqkv_out_dtype == jnp.bfloat16 else 4
    h_elem_size = jnp.dtype(h.dtype).itemsize
    vn_elem_size = jnp.dtype(v_new.dtype).itemsize if v_new is not None else 0
    dv0_elem_size = jnp.dtype(dv0.dtype).itemsize if dv0 is not None else 0
    qgw_elem_size = jnp.dtype(w.dtype).itemsize if w is not None else 0
    a_elem_size = jnp.dtype(A.dtype).itemsize if A is not None else 0
    da_elem_size = jnp.dtype(dAqk.dtype).itemsize if dAqk is not None else 0
    gw_elems = 1 if scalar_gate else K
    offdiag_pairs = (
        (BT // 16) * (BT // 16 - 1) // 2 if (per_channel_gate and BT > 64) else 0
    )
    io_per_head = (
        (2 * BT * K + 2 * BT * V + BT * gw_elems + BT) * elem_size
        + (3 * BT * K * qgw_elem_size if has_qg_kg_w else 0)
        + BT * V * (vn_elem_size + dv0_elem_size)
        + K * V * (h_elem_size + (4 if dht is not None else 0))
        + BT * BT * (a_elem_size + da_elem_size)
        + (2 * BT * K + BT * V) * dqkv_elem_size
        + (
            (1 + (1 if return_dh0 else 0)) * K * V
            + 5 * BT * K
            + BT * V
            + BT * gw_elems
            + BT
            + (5 if not has_A else 2) * BT * BT
            + 3 * offdiag_pairs * 16 * K
        )
        * 4
    )
    per_head = io_per_head + io_per_head * 3 // 2
    vmem_budget = get_vmem_limit_bytes()
    # Scale down VMEM budget for chunk sizes > 64 (override via KDA_BWD_VMEM_FRACTION).
    _frac = float(os.environ.get("KDA_BWD_VMEM_FRACTION", "0") or 0)
    if _frac > 0:
      vmem_budget = int(vmem_budget * _frac)
    elif BT > 64:
      vmem_budget = (vmem_budget * 64) // BT
    is_stateless_bf16_scalar = (
        act_dtype == jnp.bfloat16
        and not per_channel_gate
        and not has_qg_kg_w
        and not has_vn_dv0_dAqk
        and dht is None
        and not return_dh0
        and dqkv_out_dtype == jnp.bfloat16
        and BT <= 64
    )
    MB = estimate_mini_batch(
        per_head,
        H,
        max_mb=32 if is_stateless_bf16_scalar else None,
        vmem_budget=vmem_budget,
        multiple_of=gqa_repeats,
    )  # pylint: disable=invalid-name
  else:
    MB = mini_batch
  if H % MB != 0:
    raise NotImplementedError(f"H={H} must be divisible by MB={MB}.")
  if MB % gqa_repeats != 0:
    raise NotImplementedError(
        f"MB={MB} must be divisible by gqa_repeats={gqa_repeats}."
    )

  # Keep [H, B, ...] layout; reshape T → (NT, BT) only. No transpose.
  # B is an independent dimension handled by a separate grid axis.
  if native_aliased_qkv:
    q_r = qkv_conv
    k_r = qkv_conv
    v_r = qkv_conv
    do_r = do.reshape(B, T, H * V)
  elif native_bt_qkv:
    assert q is not None and k is not None and v is not None
    q_r = q.reshape(B, T, q_size)
    k_r = k.reshape(B, T, k_size)
    v_r = v.reshape(B, T, v_size)
    do_r = do.reshape(B, T, v_size)
  else:
    assert q is not None and k is not None and v is not None
    q_r = q.reshape(H_QK, B, NT, BT, K)
    k_r = k.reshape(H_QK, B, NT, BT, K)
    v_r = v.reshape(H, B, NT, BT, V)
    do_r = do.reshape(H, B, NT, BT, V)
  rstd_q_r = rstd_q.reshape(H_QK, B, NT, 1, BT) if rstd_q is not None else None
  rstd_k_r = rstd_k.reshape(H_QK, B, NT, 1, BT) if rstd_k is not None else None
  vn_r = v_new.reshape(H, B, NT, BT, V) if v_new is not None else None
  qg_r = qg.reshape(H, B, NT, BT, K) if qg is not None else None
  kg_r = kg.reshape(H, B, NT, BT, K) if kg is not None else None
  w_r = w.reshape(H, B, NT, BT, K) if w is not None else None
  # Store width-1 axes as [.., 1, BT] to avoid 128-lane minor-axis padding in HBM.
  g_r = (
      g.reshape(H, B, NT, 1, BT) if scalar_gate else g.reshape(H, B, NT, BT, GW)
  )
  beta_r = beta.reshape(H, B, NT, 1, BT)
  if A is None:
    A_r = None
  elif native_state_layout:
    A_r = A  # [B, NT, H, BT, BT], read in place
  else:
    A_r = A.reshape(H, B, NT, BT, BT)
  h_r = h  # [B, NT, H, K, V] when native_state_layout else [H, B, NT, K, V]
  dv0_r = dv0.reshape(H, B, NT, BT, V) if dv0 is not None else None
  dAqk_r = dAqk.reshape(H, B, NT, BT, BT) if dAqk is not None else None

  def idx_chunk(head_group, batch, chunk, chunk_seg_ids_smem_ref):
    del chunk_seg_ids_smem_ref
    return (head_group, batch, NT - 1 - chunk, 0, 0)

  def idx_state(head_group, batch, chunk, chunk_seg_ids_smem_ref):
    if not is_varlen:
      return (head_group, batch, 0, 0, 0)
    chunk_id = NT - 1 - chunk
    _, seq_idx, _, _, _ = _chunk_segment_metadata(
        chunk_seg_ids_smem_ref, batch, chunk_id, NT
    )
    return (head_group, batch, seq_idx, 0, 0)

  # dht_arr [B, N, H, K, V] → [H, B, N, K, V] only when dht is provided.
  dht_arr = (
      dht.astype(jnp.float32).transpose(2, 0, 1, 3, 4)
      if dht is not None
      else None
  )

  # q/k blocks carry MB // gqa_repeats key heads for the MB value heads of the
  # tile. MB divides H and gqa_repeats divides MB, so this is exact.
  MB_QK = MB // gqa_repeats
  interpret = get_interpret()
  if native_qkv_do_layout and interpret:
    raise NotImplementedError(
        "native qkv layout does not support Pallas interpret mode"
    )
  if native_aliased_qkv:
    tile_q = MB_QK * K
    tile_k = MB_QK * K
    tile_v = MB * V
    q_spec = pl.BlockSpec(
        (1, BT, pl.Element(tile_q)),
        lambda hg, b, c, _: (b, NT - 1 - c, pl.multiple_of(hg * tile_q, 128)),
    )
    k_spec = pl.BlockSpec(
        (1, BT, pl.Element(tile_k)),
        lambda hg, b, c, _: (
            b,
            NT - 1 - c,
            pl.multiple_of(q_size + hg * tile_k, 128),
        ),
    )
    v_spec = pl.BlockSpec(
        (1, BT, pl.Element(tile_v)),
        lambda hg, b, c, _: (
            b,
            NT - 1 - c,
            pl.multiple_of(q_size + k_size + hg * tile_v, 128),
        ),
    )
    do_spec = pl.BlockSpec(
        (1, BT, tile_v), lambda hg, b, c, _: (b, NT - 1 - c, hg)
    )
  elif native_bt_qkv:
    tile_q = MB_QK * K
    tile_v = MB * V
    q_spec = pl.BlockSpec(
        (1, BT, tile_q), lambda hg, b, c, _: (b, NT - 1 - c, hg)
    )
    k_spec = q_spec
    v_spec = pl.BlockSpec(
        (1, BT, tile_v), lambda hg, b, c, _: (b, NT - 1 - c, hg)
    )
    do_spec = v_spec
  else:
    q_spec = pl.BlockSpec((MB_QK, 1, 1, BT, K), index_map=idx_chunk)
    k_spec = q_spec
    v_spec = pl.BlockSpec((MB, 1, 1, BT, V), index_map=idx_chunk)
    do_spec = v_spec
  rstd_spec = (
      pl.BlockSpec((MB_QK, 1, 1, 1, BT), index_map=idx_chunk)
      if has_rstd_qk
      else None
  )
  # qg/kg/w, when provided by the caller, are per value head.
  qgw_spec = (
      pl.BlockSpec((MB, 1, 1, BT, K), index_map=idx_chunk)
      if has_qg_kg_w
      else None
  )
  vn_dv0_spec = (
      pl.BlockSpec((MB, 1, 1, BT, V), index_map=idx_chunk)
      if has_vn_dv0_dAqk
      else None
  )
  g_spec = (
      pl.BlockSpec((MB, 1, 1, 1, BT), index_map=idx_chunk)
      if scalar_gate
      else pl.BlockSpec((MB, 1, 1, BT, GW), index_map=idx_chunk)
  )
  b_spec = pl.BlockSpec((MB, 1, 1, 1, BT), index_map=idx_chunk)
  dAqk_spec = (
      pl.BlockSpec((MB, 1, 1, BT, BT), index_map=idx_chunk)
      if has_vn_dv0_dAqk
      else None
  )
  A_spec = (
      pl.BlockSpec((MB, 1, 1, BT, BT), index_map=idx_chunk)
      if A_r is not None
      else None
  )
  h_spec = pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_chunk)
  if native_state_layout:

    def idx_chunk_bnh(head_group, batch, chunk, chunk_seg_ids_smem_ref):
      del chunk_seg_ids_smem_ref
      return (batch, NT - 1 - chunk, head_group, 0, 0)

    if A_r is not None:
      A_spec = pl.BlockSpec((1, 1, MB, BT, BT), index_map=idx_chunk_bnh)
    h_spec = pl.BlockSpec((1, 1, MB, K, V), index_map=idx_chunk_bnh)
  state_spec = pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_state)
  dht_spec = state_spec if dht_arr is not None else None
  meta_spec = (
      pl.BlockSpec(
          (1, 1, 8, BT),
          lambda hg, b, c, _: (b, NT - 1 - c, 0, 0),
      )
      if has_packed_meta
      else None
  )

  kernel = partial(
      _fused_dhu_wy_intra_cumsum_kernel,
      scale=scale,
      per_channel_gate=per_channel_gate,
      gqa_repeats=gqa_repeats,
      native_state_layout=native_state_layout,
      native_qkv_do_layout=native_qkv_do_layout,
      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
      g_is_cumsummed=g_is_cumsummed,
      is_varlen=is_varlen,
      two_pass_bf16=two_pass_bf16,
      BT=BT,
      K=K,
      V=V,
      NT=NT,
      MB=MB,
  )
  dh_tmp = pltpu.VMEM((MB, K, V), jnp.float32)
  # dq/dk/dv may be written in the activation dtype (e.g. bf16), which halves
  # their HBM writes. db/dg stay f32 for the gate sums.
  b_out_shape = jax.ShapeDtypeStruct((H, B, NT, 1, BT), jnp.float32)
  g_out_shape = (
      jax.ShapeDtypeStruct((H, B, NT, 1, BT), jnp.float32)
      if scalar_gate
      else jax.ShapeDtypeStruct((H, B, NT, BT, GW), jnp.float32)
  )
  if native_aliased_qkv:
    out_shape = [
        jax.ShapeDtypeStruct((B, T, C), dqkv_out_dtype),
        b_out_shape,
        g_out_shape,
    ]
  elif native_bt_qkv:
    out_shape = [
        jax.ShapeDtypeStruct((B, T, q_size), dqkv_out_dtype),
        jax.ShapeDtypeStruct((B, T, k_size), dqkv_out_dtype),
        jax.ShapeDtypeStruct((B, T, v_size), dqkv_out_dtype),
        b_out_shape,
        g_out_shape,
    ]
  else:
    out_shape = [
        jax.ShapeDtypeStruct((H_QK, B, NT, BT, K), dqkv_out_dtype),
        jax.ShapeDtypeStruct((H_QK, B, NT, BT, K), dqkv_out_dtype),
        jax.ShapeDtypeStruct((H, B, NT, BT, V), dqkv_out_dtype),
        b_out_shape,
        g_out_shape,
    ]
  if return_dh0:
    out_shape.append(jax.ShapeDtypeStruct((H, B, N, K, V), jnp.float32))

  in_specs = [
      q_spec,
      k_spec,
      v_spec,
      vn_dv0_spec,
      qgw_spec,
      qgw_spec,
      qgw_spec,
      g_spec,
      b_spec,
      A_spec,
      h_spec,
      do_spec,
      vn_dv0_spec,
      dAqk_spec,
      dht_spec,
      rstd_spec,
      rstd_spec,
      meta_spec,
  ]
  # dq and dk are written at the key head count H_QK.
  out_specs = [q_spec, k_spec, v_spec, b_spec, g_spec]
  if return_dh0:
    out_specs.append(state_spec)
  inputs = [
      q_r,
      k_r,
      v_r,
      vn_r,
      qg_r,
      kg_r,
      w_r,
      g_r,
      beta_r,
      A_r,
      h_r,
      do_r,
      dv0_r,
      dAqk_r,
      dht_arr,
      rstd_q_r,
      rstd_k_r,
      packed_tok_meta,
  ]
  # Optional inputs are None. They are left out of the pipeline and passed to
  # the kernel body as None.
  present = [x is not None for x in inputs]
  in_specs = [s for s, p in zip(in_specs, present) if p]
  if native_aliased_qkv:
    # Pass qkv_conv once as an HBM operand; pipeline_kernel expands it to 3
    # refs.
    assert qkv_conv is not None
    hbm_inputs = [qkv_conv] + [x for x in inputs[3:] if x is not None]
  else:
    hbm_inputs = [x for x in inputs if x is not None]
  num_pipe_in = len(in_specs)
  num_pipe_out = len(out_specs)
  num_hbm_in = len(hbm_inputs)
  num_hbm_out = len(out_shape)

  def bind_seg_ids(spec, seg_ids_smem_ref):
    # emit_pipeline index maps only receive the grid indices.
    index_map = spec.index_map
    return pl.BlockSpec(
        spec.block_shape, lambda *idx: index_map(*idx, seg_ids_smem_ref)
    )

  # Ref order: optional SMEM refs, HBM inputs, HBM outputs, VMEM scratch.
  def pipeline_kernel(*all_refs):
    seg_ids_smem_ref = None
    keep_smem_ref = None
    offset = 0
    if is_varlen:
      seg_ids_smem_ref = all_refs[offset]
      offset += 1
    if has_packed_meta:
      keep_smem_ref = all_refs[offset]
      offset += 1
    refs = all_refs[offset:]
    in_hbm_refs = refs[:num_hbm_in]
    out_hbm_refs = refs[num_hbm_in : num_hbm_in + num_hbm_out]
    scratch_vmem_refs = refs[num_hbm_in + num_hbm_out :]
    if native_aliased_qkv:
      pipe_in_refs = (
          in_hbm_refs[0],
          in_hbm_refs[0],
          in_hbm_refs[0],
      ) + tuple(in_hbm_refs[1:])
      pipe_out_refs = (
          out_hbm_refs[0],
          out_hbm_refs[0],
          out_hbm_refs[0],
      ) + tuple(out_hbm_refs[1:])
    else:
      pipe_in_refs = in_hbm_refs
      pipe_out_refs = out_hbm_refs

    def body(step, *block_vmem_refs):
      blocks = iter(block_vmem_refs[:num_pipe_in])
      kernel_inputs = [next(blocks) if p else None for p in present]
      out_blocks = list(
          block_vmem_refs[num_pipe_in : num_pipe_in + num_pipe_out]
      )
      if not return_dh0:
        out_blocks.append(None)
      kernel(
          step,
          seg_ids_smem_ref,
          keep_smem_ref,
          *kernel_inputs,
          *out_blocks,
          *block_vmem_refs[num_pipe_in + num_pipe_out :],
      )

    tpu_ctx = (
        jax.sharding.use_abstract_mesh(
            jax.sharding.AbstractMesh(
                (),
                (),
                (),
                abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
            )
        )
        if (
            interpret
            and not pltpu.is_tpu_device()
            and jax.sharding.get_abstract_mesh().empty
        )
        else contextlib.nullcontext()
    )
    with tpu_ctx:
      pltpu.emit_pipeline(
          body,
          grid=(H // MB, B, NT),
          in_specs=[bind_seg_ids(s, seg_ids_smem_ref) for s in in_specs],
          out_specs=[bind_seg_ids(s, seg_ids_smem_ref) for s in out_specs],
          _explicit_indices=True,
      )(*pipe_in_refs, *pipe_out_refs, scratches=tuple(scratch_vmem_refs))

  # Operands stay in HBM. Inside a single kernel invocation, emit_pipeline
  # walks the (head group, batch, reversed chunk) grid and pipelines the block
  # copies. Chunks run in order: `dh_tmp` carries dh from one chunk to the
  # previous one.
  hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
  smem_args = []
  if chunk_seg_ids is not None:
    smem_args.append(chunk_seg_ids)
  if packed_chunk_keep is not None:
    smem_args.append(packed_chunk_keep.astype(jnp.int32))
  pallas_args = [*smem_args, *hbm_inputs]
  total_bwd_chunks = H * B * NT
  tri_inv_matmuls = max(0, 2 * int(math.log2(BT)) - 3) + (
      2 if two_pass_bf16 and act_dtype == jnp.bfloat16 else 0
  )
  qk_kk_flops = (
      ((4 if not has_vn_dv0_dAqk else 2) * BT * BT * K) // gqa_repeats
      if not has_A
      else ((2 * BT * BT * K) // gqa_repeats if not has_vn_dv0_dAqk else 0)
  )
  bwd_flops_per_chunk = (
      12 * BT * K * V
      + 8 * BT * BT * K
      + qk_kk_flops
      + (10 if not has_vn_dv0_dAqk else 6) * BT * BT * V
      + (2 * tri_inv_matmuls * BT * BT * BT if not has_A else 0)
  )
  bwd_transcendentals_per_chunk = (
      4 * BT * K if per_channel_gate else (BT * BT + 3 * BT)
  )
  bwd_bytes = sum(
      int(x.size) * jnp.dtype(x.dtype).itemsize for x in pallas_args
  ) + sum(int(s.size) * jnp.dtype(s.dtype).itemsize for s in out_shape)
  bwd_cost_estimate = pl.CostEstimate(
      flops=total_bwd_chunks * bwd_flops_per_chunk,
      transcendentals=total_bwd_chunks * bwd_transcendentals_per_chunk,
      bytes_accessed=bwd_bytes,
  )
  pallas_out = pl.pallas_call(
      pipeline_kernel,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(smem_args),
          grid=(),
          in_specs=[hbm_spec] * num_hbm_in,
          out_specs=[hbm_spec] * num_hbm_out,
          scratch_shapes=[dh_tmp],
      ),
      compiler_params=pltpu.CompilerParams(
          disable_bounds_checks=True,
          vmem_limit_bytes=(
              None if interpret else get_tpu_limits().vmem_limit_bytes
          ),
      ),
      cost_estimate=bwd_cost_estimate,
      interpret=interpret,
      name="_fused_dhu_wy_intra_cumsum_pallas_jit",
  )(*pallas_args)

  if native_aliased_qkv:
    if return_dh0:
      dy_conv_r, db_r, dg_r, dh0_r = pallas_out
      dh0_out = dh0_r.transpose(1, 2, 0, 3, 4)
    else:
      dy_conv_r, db_r, dg_r = pallas_out
      dh0_out = None
    return (
        dy_conv_r,
        None,
        None,
        db_r.reshape(H, B, T),
        dg_r.reshape(H, B, T, GW),
        dh0_out,
    )

  if return_dh0:
    dq_r, dk_r, dv_r, db_r, dg_r, dh0_r = pallas_out
    dh0_out = dh0_r.transpose(1, 2, 0, 3, 4)
  else:
    dq_r, dk_r, dv_r, db_r, dg_r = pallas_out
    dh0_out = None
  if native_bt_qkv:
    return (
        dq_r.reshape(B, T, H_QK, K),
        dk_r.reshape(B, T, H_QK, K),
        dv_r.reshape(B, T, H, V),
        db_r.reshape(H, B, T),
        dg_r.reshape(H, B, T, GW),
        dh0_out,
    )
  return (
      dq_r.reshape(H_QK, B, T, K),
      dk_r.reshape(H_QK, B, T, K),
      dv_r.reshape(H, B, T, V),
      db_r.reshape(H, B, T),
      dg_r.reshape(H, B, T, GW),
      dh0_out,
  )


RCP_LN2 = 1.0 / math.log(2)


# =====================================================================
# chunk_kda_bwd_dAv  —  Pallas kernel
# =====================================================================


def _chunk_kda_bwd_dAv_kernel(
  v_ref,
  A_ref,
  do_ref,
  dA_ref,
  dv_ref,
  *,
  scale,
  BT,
  BV,
  NV,
  V,
  MB,
  two_pass_bf16=False,
):
  bv = v_ref[:]  # [MB, BT, V]
  bA = A_ref[:]  # [MB, BT, BT]
  bdo = do_ref[:].astype(jnp.float32)  # [MB, BT, V], cast in VMEM
  if do_ref.dtype != jnp.bfloat16:
    precision = jax.lax.Precision.HIGHEST
  elif two_pass_bf16:
    precision = jax.lax.Precision.HIGH
  else:
    precision = None

  m_causal = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
  bA_masked = jnp.where(m_causal[None, :, :], bA, 0.0)  # [MB, BT, BT]

  b_dA = jnp.zeros((MB, BT, BT), jnp.float32)
  dv_blocks = []

  for i_v in range(NV):
    vs = i_v * BV
    ve = vs + BV

    b_v_blk = bv[:, :, vs:ve]    # [MB, BT, BV]
    b_do_blk = bdo[:, :, vs:ve]  # [MB, BT, BV]

    # dA += do @ v^T — contract BV (dim 2), batch MB (dim 0)
    b_dA += jax.lax.dot_general(
      b_do_blk,
      b_v_blk,
      (((2,), (2,)), ((0,), (0,))),
      precision=precision,
      preferred_element_type=jnp.float32,
    )

    # dv = A^T @ do — contract BT_row (dim 1), batch MB (dim 0)
    b_dv_blk = jax.lax.dot_general(
      bA_masked,
      b_do_blk.astype(bA_masked.dtype),
      (((1,), (1,)), ((0,), (0,))),
      precision=precision,
      preferred_element_type=jnp.float32,
    )
    dv_blocks.append(b_dv_blk)

  b_dv = jnp.concatenate(dv_blocks, axis=2) if NV > 1 else dv_blocks[0]

  # Apply causal mask and scale
  b_dA = jnp.where(m_causal[None, :, :], b_dA * scale, 0.0)

  dA_ref[:] = b_dA
  # Store in the output dtype: dv is an f32 partial gradient even when the
  # cotangent `do` is bf16.
  dv_ref[:] = b_dv.astype(dv_ref.dtype)


@functools.partial(
  jax.jit,
  static_argnames=[
    "chunk_size",
    "scale",
    "block_V",
    "mini_batch",
    "two_pass_bf16",
  ],
)
@jaxtyping.jaxtyped
def chunk_kda_bwd_dAv_kernel(
  q: Float[Array, "H B T K"],
  k: Float[Array, "H B T K"],
  v: Float[Array, "H B T V"],
  do: Float[Array, "H B T V"],
  A: Float[Array, "H B T BT"],
  scale: float,
  chunk_size: int = 64,
  block_V: int | None = None,
  mini_batch: int | None = None,
  *,
  two_pass_bf16: bool = False,
) -> tuple[
    Float[Array, "H B T BT"],
    Float[Array, "H B T V"],
]:
  """Computes attention and value gradients with a tiled Pallas kernel.

  `block_V` controls value-dimension tiling; `mini_batch` controls the
  flattened chunks processed by each program.
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT

  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  assert A.shape[-1] == BT, (
      f"A.shape[-1]={A.shape[-1]} must equal chunk_size={BT}"
  )

  BV = block_V if block_V is not None else V
  assert V % BV == 0, f"V={V} must be divisible by block_V={BV}"
  NV = V // BV
  BH = B * H

  v_r = v.reshape(BH * NT, BT, V)
  A_r = A.reshape(BH * NT, BT, BT)
  do_r = do.reshape(BH * NT, BT, V)

  total = BH * NT

  # ---- auto-compute mini-batch (MB) to maximise VMEM utilisation ----
  if mini_batch is None:
    v_elem = jnp.dtype(v.dtype).itemsize
    a_elem = jnp.dtype(A.dtype).itemsize
    do_elem = jnp.dtype(do.dtype).itemsize
    in_bytes = BT * V * (v_elem + do_elem) + BT * BT * a_elem
    out_bytes = (BT * BT + BT * V) * 4
    scratch_bytes = (3 * BT * BT + 2 * BT * V) * 4
    per_chunk = in_bytes + out_bytes + scratch_bytes
    MB = estimate_mini_batch(per_chunk, total)
  else:
    MB = mini_batch
    assert total % MB == 0, f"total={total} must be divisible by mini_batch={MB}"
  def _spec3(d1, d2):
    return pl.BlockSpec(block_shape=(MB, d1, d2), index_map=lambda idx: (idx, 0, 0))

  in_specs = [
    _spec3(BT, V),   # v
    _spec3(BT, BT),  # A
    _spec3(BT, V),   # do
  ]

  out_specs = [
    _spec3(BT, BT),  # dA
    _spec3(BT, V),   # dv
  ]

  out_shape = [
      jax.ShapeDtypeStruct((total, BT, BT), jnp.float32),
      # f32 (not `do.dtype`): this partial gradient is accumulated by the fused
      # backward kernel.
      jax.ShapeDtypeStruct((total, BT, V), jnp.float32),
  ]

  kernel = partial(
    _chunk_kda_bwd_dAv_kernel,
    scale=scale,
    BT=BT,
    BV=BV,
    NV=NV,
    V=V,
    MB=MB,
    two_pass_bf16=two_pass_bf16,
  )

  interpret = get_interpret()

  dA_r, dv_r = pl.pallas_call(
    kernel,
    out_shape=out_shape,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=(total // MB,),
      in_specs=in_specs,
      out_specs=out_specs,
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel",),
      disable_bounds_checks=True,
      vmem_limit_bytes=None if interpret else get_tpu_limits().vmem_limit_bytes,
    ),
    cost_estimate=pl.CostEstimate(
      flops=4 * total * BT * BT * V,
      transcendentals=0,
      bytes_accessed=(
          int(v_r.size) * jnp.dtype(v_r.dtype).itemsize
          + int(A_r.size) * jnp.dtype(A_r.dtype).itemsize
          + int(do_r.size) * jnp.dtype(do_r.dtype).itemsize
          + total * (BT * BT + BT * V) * 4
      ),
    ),
    interpret=interpret,
  )(v_r, A_r, do_r)

  # [HB*NT, BT, X] -> [H, B, T, X]
  def _ir(x, d):
    return x.reshape(H, B, T, d)

  return _ir(dA_r, BT), _ir(dv_r, V)


# =====================================================================
# chunk_kda_bwd_custom  —  backward adapter and 6-stage orchestrator
# =====================================================================


def _conv1d_silu_bwd(
    x_raw: jax.Array,
    w: jax.Array,
    bias: jax.Array | None,
    dout: jax.Array,
    aligned_segment_ids: jax.Array | None = None,
    *,
    packed_canonical_seg: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  """Computes VJP of depthwise causal conv1d + SiLU, supporting GQA head repeats."""
  H, B, T, D = x_raw.shape
  H_w, W, _ = w.shape
  repeats = H // H_w

  def _fwd_fn(x_in, w_in, b_in):
    x_f32 = x_in.astype(jnp.float32)
    w_f32 = w_in.astype(jnp.float32)
    acc = x_f32 * w_f32[:, None, W - 1 : W, :]
    for lag in range(1, min(W, x_in.shape[2])):
      x_lag = jnp.pad(x_f32[:, :, :-lag, :], ((0, 0), (0, 0), (lag, 0), (0, 0)))
      if packed_canonical_seg is not None:
        same_seg = (
            packed_canonical_seg[:, lag:] == packed_canonical_seg[:, :-lag]
        ) & (packed_canonical_seg[:, lag:] > 0)
        same_seg = jnp.pad(same_seg, ((0, 0), (lag, 0)))
        x_lag = jnp.where(same_seg[None, :, :, None], x_lag, 0.0)
      elif aligned_segment_ids is not None:
        same_seg = (
            aligned_segment_ids[:, lag:] == aligned_segment_ids[:, :-lag]
        ) & (aligned_segment_ids[:, lag:] >= 0)
        same_seg = jnp.pad(same_seg, ((0, 0), (lag, 0)))
        x_lag = jnp.where(same_seg[None, :, :, None], x_lag, 0.0)
      acc = acc + x_lag * w_f32[:, None, W - 1 - lag : W - lag, :]
    if b_in is not None:
      acc = acc + b_in.astype(jnp.float32)[:, None, None, :]
    out = jax.nn.silu(acc)
    if packed_canonical_seg is not None:
      out = jnp.where((packed_canonical_seg > 0)[None, :, :, None], out, 0.0)
    return out.astype(x_in.dtype)

  if repeats > 1:
    x_raw_k = x_raw[::repeats]
    dout_k = jnp.sum(
        dout.astype(x_raw.dtype).reshape(H_w, repeats, B, T, D), axis=1
    )
    _, vjp_fn = jax.vjp(_fwd_fn, x_raw_k, w, bias)
    dx_raw_k, dw, db = vjp_fn(dout_k)
    dx_raw = (
        jnp.zeros_like(x_raw).at[::repeats].set(dx_raw_k.astype(x_raw.dtype))
    )
  else:
    _, vjp_fn = jax.vjp(_fwd_fn, x_raw, w, bias)
    dx_raw, dw, db = vjp_fn(dout.astype(x_raw.dtype))
    dx_raw = dx_raw.astype(x_raw.dtype)

  return (
      dx_raw,
      dw.astype(w.dtype),
      (db.astype(bias.dtype) if bias is not None and db is not None else None),
  )


def _local_cp_bwd_summary_kernel(
    *refs,
    BT: int,
    MB: int,
    NT: int,
    K_PADSIZE: int,
    V_ALIGNED: int,
    SCALE: float,
    HAS_PACKED_META: bool,
):
  """Pallas reverse chunk-state summary scan keeping `(dS, dM)` in VMEM."""
  if HAS_PACKED_META:
    (
        keep_ref,
        qg_ref,
        kg_ref,
        w_ref,
        do_ref,
        dv0_ref,
        gk_ref,
        meta_ref,
        ds_out_ref,
        dm_out_ref,
        ds_scratch,
        dm_scratch,
    ) = refs
  else:
    keep_ref = meta_ref = None
    (
        qg_ref,
        kg_ref,
        w_ref,
        do_ref,
        dv0_ref,
        gk_ref,
        ds_out_ref,
        dm_out_ref,
        ds_scratch,
        dm_scratch,
    ) = refs

  i_b = pl.program_id(1)
  c_rev = pl.program_id(2)
  chunk_id = NT - 1 - c_rev

  @pl.when(c_rev == 0)
  def _init():
    ds_scratch[...] = jnp.zeros((MB, K_PADSIZE, V_ALIGNED), dtype=jnp.float32)
    dm_scratch[...] = jnp.broadcast_to(
        jnp.eye(K_PADSIZE, dtype=jnp.float32), (MB, K_PADSIZE, K_PADSIZE)
    )

  ds_acc = ds_scratch[...]
  dm_acc = dm_scratch[...]
  qg_f = qg_ref[:, 0, :].astype(jnp.float32)
  kg_f = kg_ref[:, 0, :].astype(jnp.float32)
  w_f = w_ref[:, 0, :].astype(jnp.float32)
  do_f = do_ref[:, 0, :].astype(jnp.float32)
  dv0_f = dv0_ref[:, 0, :].astype(jnp.float32)
  gk_last = gk_ref[:, 0, :][:, BT - 1, :].astype(jnp.float32)

  if HAS_PACKED_META:
    assert keep_ref is not None and meta_ref is not None
    valid_f = meta_ref[0, 0, 0][4].astype(jnp.float32)[None, :, None]
    do_f = do_f * valid_f
    dv0_f = dv0_f * valid_f
    keep_f = keep_ref[i_b, chunk_id].astype(jnp.float32)
    decay = keep_f * jnp.exp2(keep_f * gk_last)
  else:
    decay = jnp.exp2(gk_last)

  prec = get_chain_precision()
  dv_s = dv0_f + jnp.matmul(
      kg_f, ds_acc, precision=prec, preferred_element_type=jnp.float32
  )
  qg_t = qg_f.transpose(0, 2, 1)
  w_t = w_f.transpose(0, 2, 1)
  ds_next = (
      decay[:, :, None] * ds_acc
      + SCALE
      * jnp.matmul(
          qg_t, do_f, precision=prec, preferred_element_type=jnp.float32
      )
      - jnp.matmul(
          w_t, dv_s, precision=prec, preferred_element_type=jnp.float32
      )
  )
  kg_dm = jnp.matmul(
      kg_f, dm_acc, precision=prec, preferred_element_type=jnp.float32
  )
  dm_next = decay[:, :, None] * dm_acc - jnp.matmul(
      w_t, kg_dm, precision=prec, preferred_element_type=jnp.float32
  )
  ds_scratch[...] = ds_next
  dm_scratch[...] = dm_next

  @pl.when(c_rev == NT - 1)
  def _store():
    ds_out_ref[0, :, :] = ds_scratch[...].astype(ds_out_ref.dtype)
    dm_out_ref[0, :, :] = dm_scratch[...].astype(dm_out_ref.dtype)


def compute_local_cp_bwd_summary(
    *,
    qg: jax.Array,
    kg: jax.Array,
    w: jax.Array,
    do: jax.Array,
    dv0: jax.Array,
    gk: jax.Array,
    scale: float,
    chunk_size: int,
    packed_chunk_keep: jax.Array | None = None,
    packed_tok_meta: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Computes local reverse affine summary `(dS_local, dM_local)` in fp32."""
  H, B, T, K = qg.shape
  V = do.shape[-1]
  GW = gk.shape[-1]
  BT = chunk_size
  NT = T // BT
  if NT == 0:
    return (
        jnp.zeros((B, H, K, V), dtype=jnp.float32),
        jnp.broadcast_to(jnp.eye(K, dtype=jnp.float32), (B, H, K, K)),
    )

  K_PADSIZE = int(align_up(K, 128))
  V_ALIGNED = int(align_up(V, 128))
  q_elem = jnp.dtype(qg.dtype).itemsize
  do_elem = jnp.dtype(do.dtype).itemsize
  gk_elem = jnp.dtype(gk.dtype).itemsize
  per_head = (
      2
      * BT
      * (
          3 * K_PADSIZE * q_elem
          + 2 * V_ALIGNED * do_elem
          + GW * gk_elem
      )
      + 3 * (K_PADSIZE * V_ALIGNED + K_PADSIZE * K_PADSIZE) * 4
  )
  MB = estimate_mini_batch(per_head, H)

  def _pad_last(x: jax.Array, pad: int) -> jax.Array:
    return jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, pad))) if pad > 0 else x

  qg_t = _pad_last(qg, K_PADSIZE - K)
  kg_t = _pad_last(kg, K_PADSIZE - K)
  w_t = _pad_last(w, K_PADSIZE - K)
  do_t = _pad_last(do, V_ALIGNED - V)
  dv0_t = _pad_last(dv0, V_ALIGNED - V)
  gk_t = _pad_last(gk, (K_PADSIZE - K) if GW == K else 0)
  gw_block = K_PADSIZE if GW == K else GW

  def _in_rev_map(h, b, c_rev, *_):
    return (h, b, NT - 1 - c_rev, 0)

  def _out_map(h, b, c_rev, *_):
    del c_rev
    return (b, h, 0, 0)

  has_packed_meta = (
      packed_tok_meta is not None and packed_chunk_keep is not None
  )
  in_specs = [
      pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_in_rev_map),
      pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_in_rev_map),
      pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_in_rev_map),
      pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_in_rev_map),
      pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_in_rev_map),
      pl.BlockSpec([MB, 1, BT, gw_block], index_map=_in_rev_map),
  ]
  call_args: list[jax.Array] = []
  if has_packed_meta:
    assert packed_chunk_keep is not None and packed_tok_meta is not None
    call_args.append(packed_chunk_keep.astype(jnp.int32))
    in_specs.append(
        pl.BlockSpec(
            [1, 1, 1, 8, BT],
            index_map=lambda h, b, c_rev, *_: (0, b, NT - 1 - c_rev, 0, 0),
        )
    )
  call_args.extend([qg_t, kg_t, w_t, do_t, dv0_t, gk_t])
  if has_packed_meta:
    assert packed_tok_meta is not None
    call_args.append(packed_tok_meta[None, ...])

  out_specs = [
      pl.BlockSpec([1, MB, K_PADSIZE, V_ALIGNED], index_map=_out_map),
      pl.BlockSpec([1, MB, K_PADSIZE, K_PADSIZE], index_map=_out_map),
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((B, H, K_PADSIZE, V_ALIGNED), jnp.float32),
      jax.ShapeDtypeStruct((B, H, K_PADSIZE, K_PADSIZE), jnp.float32),
  ]
  scratches = [
      pltpu.VMEM((MB, K_PADSIZE, V_ALIGNED), jnp.float32),
      pltpu.VMEM((MB, K_PADSIZE, K_PADSIZE), jnp.float32),
  ]
  total_chunks = H * B * NT
  cost = pl.CostEstimate(
      flops=total_chunks * BT * K_PADSIZE * (6 * V_ALIGNED + 4 * K_PADSIZE),
      transcendentals=total_chunks * gw_block,
      bytes_accessed=sum(
          int(x.size) * jnp.dtype(x.dtype).itemsize for x in call_args
      )
      + sum(int(s.size) * 4 for s in out_shapes),
  )
  ds_pad, dm_pad = pl.pallas_call(
      functools.partial(
          _local_cp_bwd_summary_kernel,
          BT=BT,
          MB=MB,
          NT=NT,
          K_PADSIZE=K_PADSIZE,
          V_ALIGNED=V_ALIGNED,
          SCALE=scale,
          HAS_PACKED_META=has_packed_meta,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1 if has_packed_meta else 0,
          grid=(H // MB, B, NT),
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=scratches,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=(
              None if get_interpret() else get_tpu_limits().vmem_limit_bytes
          ),
      ),
      out_shape=out_shapes,
      cost_estimate=cost,
      interpret=get_interpret(),
  )(*call_args)
  return ds_pad[..., :K, :V], dm_pad[..., :K, :K]


def _gdn_local_cp_bwd_summary_kernel(
    *refs,
    BT: int,
    MB: int,
    MB_QK: int,
    NT: int,
    K: int,
    V: int,
    K_PADSIZE: int,
    V_ALIGNED: int,
    GQA_REPEATS: int,
    SCALE: float,
    USE_QK_L2NORM: bool,
    TWO_PASS_BF16: bool,
    NATIVE_QKV_DO: bool,
    ALIASED_QKV: bool,
    HAS_AKK_IN: bool,
    HAS_PACKED_META: bool,
):
  """Pallas reverse scan for GDN `(dS_local, dM_local)` + transient `Akk`."""
  idx = 0
  keep_ref = None
  if HAS_PACKED_META:
    keep_ref = refs[idx]
    idx += 1

  q_ref = refs[idx]
  k_ref = refs[idx + 1]
  do_ref = refs[idx + 2]
  beta_ref = refs[idx + 3]
  g_ref = refs[idx + 4]
  idx += 5

  akk_in_ref = None
  if HAS_AKK_IN:
    akk_in_ref = refs[idx]
    idx += 1

  meta_ref = None
  if HAS_PACKED_META:
    meta_ref = refs[idx]
    idx += 1

  ds_out_ref = refs[idx]
  dm_out_ref = refs[idx + 1]
  idx += 2

  akk_out_ref = None
  if not HAS_AKK_IN:
    akk_out_ref = refs[idx]
    idx += 1

  ds_scratch = refs[idx]
  dm_scratch = refs[idx + 1]

  act_dtype = q_ref.dtype
  i_b = pl.program_id(1)
  c_rev = pl.program_id(2)
  chunk_id = NT - 1 - c_rev

  @pl.when(c_rev == 0)
  def _init():
    ds_scratch[...] = jnp.zeros((MB, K_PADSIZE, V_ALIGNED), dtype=jnp.float32)
    dm_scratch[...] = jnp.broadcast_to(
        jnp.eye(K_PADSIZE, dtype=jnp.float32), (MB, K_PADSIZE, K_PADSIZE)
    )

  ds_acc = ds_scratch[...]
  dm_acc = dm_scratch[...]

  if NATIVE_QKV_DO:
    q_tile = q_ref[0] if ALIASED_QKV else q_ref[0, 0]
    k_tile = k_ref[0] if ALIASED_QKV else k_ref[0, 0]
    do_tile = do_ref[0, 0]
    q_f = jnp.stack(
        [q_tile[:, h * K : (h + 1) * K] for h in range(MB_QK)], axis=0
    ).astype(jnp.float32)
    k_f = jnp.stack(
        [k_tile[:, h * K : (h + 1) * K] for h in range(MB_QK)], axis=0
    ).astype(jnp.float32)
    do_f = jnp.stack(
        [do_tile[:, h * V : (h + 1) * V] for h in range(MB)], axis=0
    ).astype(jnp.float32)
  else:
    q_f = q_ref[:, 0, :].astype(jnp.float32)
    k_f = k_ref[:, 0, :].astype(jnp.float32)
    do_f = do_ref[:, 0, :].astype(jnp.float32)

  bb_f = beta_ref[:, 0, 0, 0].astype(jnp.float32)
  bg_f = g_ref[:, 0, 0, 0].astype(jnp.float32)[..., None]

  if HAS_PACKED_META:
    assert meta_ref is not None and keep_ref is not None
    meta_tile = meta_ref[0, 0, 0]
    seg_tok = meta_tile[0]
    m_in_f = meta_tile[2].astype(jnp.float32)[None, :, None]
    m_out_f = meta_tile[3].astype(jnp.float32)[None, :, None]
    valid_f = meta_tile[4].astype(jnp.float32)[None, :, None]
    q_f = q_f * valid_f
    k_f = k_f * valid_f
    do_f = do_f * valid_f
    bb_f = bb_f * valid_f[..., 0]
    seg_r = jnp.broadcast_to(seg_tok[:, None], (BT, BT))
    seg_c = jnp.broadcast_to(seg_tok[None, :], (BT, BT))
    same_doc = (seg_r == seg_c) & (seg_r > 0)
  else:
    m_in_f = m_out_f = valid_f = same_doc = None

  if USE_QK_L2NORM:
    q_sq = jnp.sum(q_f * q_f, axis=-1, keepdims=True)
    k_sq = jnp.sum(k_f * k_f, axis=-1, keepdims=True)
    rstd_q = jnp.where(
        q_sq == 0.0,
        0.0,
        jax.lax.rsqrt(jnp.where(q_sq == 0.0, 1.0, q_sq) + 1e-6),
    )
    rstd_k = jnp.where(
        k_sq == 0.0,
        0.0,
        jax.lax.rsqrt(jnp.where(k_sq == 0.0, 1.0, k_sq) + 1e-6),
    )
    q_f = (q_f * rstd_q).astype(act_dtype).astype(jnp.float32)
    k_f = (k_f * rstd_k).astype(act_dtype).astype(jnp.float32)

  # Intra-chunk matmuls mirror `_fused_dhu_wy_intra_cumsum_kernel`; the
  # `(dS, dM)` chain follows `get_chain_precision` like the forward summary.
  if act_dtype != jnp.bfloat16:
    prec = jax.lax.Precision.HIGHEST
  elif TWO_PASS_BF16:
    prec = jax.lax.Precision.HIGH
  else:
    prec = jax.lax.Precision.DEFAULT
  chain_prec = get_chain_precision()
  if HAS_AKK_IN:
    qk_dot = jnp.matmul(
        q_f,
        k_f.transpose(0, 2, 1),
        precision=prec,
        preferred_element_type=jnp.float32,
    )
    kk_dot = None
  else:
    qk_kk = jnp.matmul(
        jnp.concatenate([q_f, k_f], axis=1),
        k_f.transpose(0, 2, 1),
        precision=prec,
        preferred_element_type=jnp.float32,
    )
    qk_dot = qk_kk[:, :BT]
    kk_dot = qk_kk[:, BT:]

  if GQA_REPEATS > 1:
    q_v = jnp.repeat(q_f, GQA_REPEATS, axis=0)
    k_v = jnp.repeat(k_f, GQA_REPEATS, axis=0)
    qk_dot = jnp.repeat(qk_dot, GQA_REPEATS, axis=0)
    if kk_dot is not None:
      kk_dot = jnp.repeat(kk_dot, GQA_REPEATS, axis=0)
  else:
    q_v, k_v = q_f, k_f

  row_bt = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
  col_bt = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
  m_causal = row_bt >= col_bt
  m_strict = row_bt > col_bt
  if same_doc is not None:
    m_causal = m_causal & same_doc
    m_strict = m_strict & same_doc
  diff_g = bg_f - bg_f.transpose(0, 2, 1)
  decay_mat = jnp.where(
      m_causal[None], jnp.exp2(jnp.where(m_causal[None], diff_g, 0.0)), 0.0
  )
  aqk_c = qk_dot * decay_mat * SCALE
  dv0_c = jnp.matmul(
      aqk_c.transpose(0, 2, 1),
      do_f,
      precision=prec,
      preferred_element_type=jnp.float32,
  )

  if HAS_AKK_IN:
    assert akk_in_ref is not None
    a_inv = akk_in_ref[:, 0, :].astype(jnp.float32)
  else:
    assert kk_dot is not None and akk_out_ref is not None
    l_bwd = jnp.where(
        m_strict[None], kk_dot * decay_mat * bb_f[:, :, None], 0.0
    )
    a_inv = tri_inv_xor_dnc(
        l_bwd,
        use_fp32=(act_dtype == jnp.float32),
        two_pass_bf16=TWO_PASS_BF16,
    )
    akk_out_ref[:, 0, :] = a_inv.astype(akk_out_ref.dtype)

  if m_in_f is not None and m_out_f is not None:
    g_exp_in = m_in_f * jnp.exp2(m_in_f * bg_f)
    g_exp_out = m_out_f * jnp.exp2(
        m_out_f * (bg_f[:, BT - 1 : BT, :] - bg_f)
    )
  else:
    g_exp_in = jnp.exp2(bg_f)
    g_exp_out = jnp.exp2(bg_f[:, BT - 1 : BT, :] - bg_f)

  if HAS_PACKED_META:
    assert keep_ref is not None
    keep_f = keep_ref[i_b, chunk_id].astype(jnp.float32)
    decay_last = keep_f * jnp.exp2(keep_f * bg_f[:, BT - 1, :])
  else:
    decay_last = jnp.exp2(bg_f[:, BT - 1, :])

  qg_c = q_v * g_exp_in
  kg_c = k_v * g_exp_out
  w_c = jnp.matmul(
      a_inv,
      k_v * bb_f[:, :, None] * g_exp_in,
      precision=prec,
      preferred_element_type=jnp.float32,
  )
  if valid_f is not None:
    w_c = w_c * valid_f

  dv_s = dv0_c + jnp.matmul(
      kg_c, ds_acc, precision=chain_prec, preferred_element_type=jnp.float32
  )
  qg_t = qg_c.transpose(0, 2, 1)
  w_t = w_c.transpose(0, 2, 1)
  ds_next = (
      decay_last[:, :, None] * ds_acc
      + SCALE
      * jnp.matmul(
          qg_t, do_f, precision=chain_prec, preferred_element_type=jnp.float32
      )
      - jnp.matmul(
          w_t, dv_s, precision=chain_prec, preferred_element_type=jnp.float32
      )
  )
  kg_dm = jnp.matmul(
      kg_c, dm_acc, precision=chain_prec, preferred_element_type=jnp.float32
  )
  dm_next = decay_last[:, :, None] * dm_acc - jnp.matmul(
      w_t, kg_dm, precision=chain_prec, preferred_element_type=jnp.float32
  )
  ds_scratch[...] = ds_next
  dm_scratch[...] = dm_next

  @pl.when(c_rev == NT - 1)
  def _store():
    ds_out_ref[0, :, :] = ds_scratch[...].astype(ds_out_ref.dtype)
    dm_out_ref[0, :, :] = dm_scratch[...].astype(dm_out_ref.dtype)


def compute_gdn_local_cp_bwd_summary(
    *,
    qkv_conv: jax.Array | None,
    q_bt: jax.Array | None,
    k_bt: jax.Array | None,
    beta_hbt: jax.Array,
    g_hbt: jax.Array,
    do_bt: jax.Array,
    akk: jax.Array | None,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    scale: float,
    chunk_size: int,
    use_qk_l2norm: bool,
    two_pass_bf16: bool,
    mini_batch: int | None = None,
    packed_tok_meta: jax.Array | None = None,
    packed_chunk_keep: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes GDN `(dS_local, dM_local)` in Pallas and reuses rebuilt `Akk`."""
  B, T, H_V, V = do_bt.shape
  H_QK = num_k_heads
  K = head_k_dim
  assert H_V == num_v_heads and V == head_v_dim
  BT = chunk_size
  NT = T // BT
  gqa_repeats = H_V // H_QK
  act_source = qkv_conv if qkv_conv is not None else q_bt
  assert act_source is not None
  act_dtype = act_source.dtype

  K_PADSIZE = int(align_up(K, 128))
  V_ALIGNED = int(align_up(V, 128))
  interpret = get_interpret()
  can_native_qkv_do = (
      K == K_PADSIZE and V == V_ALIGNED and BT % 8 == 0 and not interpret
  )

  act_elem = jnp.dtype(act_dtype).itemsize
  akk_bytes = 4 if (two_pass_bf16 or act_dtype == jnp.float32) else act_elem
  io_bytes = 2 * (
      (2 * BT * K_PADSIZE + BT * V_ALIGNED) * act_elem
      + 2 * BT * 4
      + BT * BT * akk_bytes
      + (K_PADSIZE * V_ALIGNED + K_PADSIZE * K_PADSIZE) * 4
  )
  scratch_bytes = (
      2 * (K_PADSIZE * V_ALIGNED + K_PADSIZE * K_PADSIZE)
      + 4 * BT * K_PADSIZE
      + 2 * BT * V_ALIGNED
      + (6 if akk is None else 2) * BT * BT
  ) * 4
  per_head = io_bytes + scratch_bytes
  vmem_limit = get_vmem_limit_bytes()
  vmem_budget = (vmem_limit * 64) // BT if BT > 64 else vmem_limit
  max_mb = get_max_mini_batch(chunk_size=BT if BT > 128 else None)
  auto_mb = estimate_mini_batch(
      per_head,
      H_V,
      max_mb=max_mb,
      vmem_budget=vmem_budget,
      multiple_of=gqa_repeats,
  )
  if (
      mini_batch is not None
      and mini_batch <= H_V
      and H_V % mini_batch == 0
      and mini_batch % gqa_repeats == 0
      and mini_batch <= max(max_mb, gqa_repeats)
      and mini_batch * per_head <= vmem_budget
  ):
    MB = mini_batch
  else:
    MB = auto_mb
  MB_QK = MB // gqa_repeats

  q_size = H_QK * K
  k_size = H_QK * K
  aliased_qkv = False
  if can_native_qkv_do:
    tile_q = MB_QK * K
    tile_v = MB * V
    if qkv_conv is not None:
      aliased_qkv = True
      q_arg = qkv_conv
      k_arg = qkv_conv
      q_spec = pl.BlockSpec(
          (pl.Element(1), pl.Element(BT), pl.Element(tile_q)),
          lambda hg, b, c_rev, *_: (
              b,
              pl.multiple_of((NT - 1 - c_rev) * BT, 8),
              pl.multiple_of(hg * tile_q, 128),
          ),
      )
      k_spec = pl.BlockSpec(
          (pl.Element(1), pl.Element(BT), pl.Element(tile_q)),
          lambda hg, b, c_rev, *_: (
              b,
              pl.multiple_of((NT - 1 - c_rev) * BT, 8),
              pl.multiple_of(q_size + hg * tile_q, 128),
          ),
      )
    else:
      assert q_bt is not None and k_bt is not None
      q_arg = q_bt.reshape(B, NT, BT, q_size)
      k_arg = k_bt.reshape(B, NT, BT, k_size)
      q_spec = pl.BlockSpec(
          (1, 1, BT, tile_q),
          lambda hg, b, c_rev, *_: (b, NT - 1 - c_rev, 0, hg),
      )
      k_spec = q_spec
    do_arg = do_bt.reshape(B, NT, BT, H_V * V)
    do_spec = pl.BlockSpec(
        (1, 1, BT, tile_v),
        lambda hg, b, c_rev, *_: (b, NT - 1 - c_rev, 0, hg),
    )
  else:
    if qkv_conv is not None:
      q_hbt = (
          qkv_conv[:, :, :q_size].reshape(B, T, H_QK, K).transpose(2, 0, 1, 3)
      )
      k_hbt = (
          qkv_conv[:, :, q_size : q_size + k_size]
          .reshape(B, T, H_QK, K)
          .transpose(2, 0, 1, 3)
      )
    else:
      assert q_bt is not None and k_bt is not None
      q_hbt = q_bt.transpose(2, 0, 1, 3)
      k_hbt = k_bt.transpose(2, 0, 1, 3)
    do_hbt = do_bt.transpose(2, 0, 1, 3)
    if K_PADSIZE > K:
      q_hbt = jnp.pad(q_hbt, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
      k_hbt = jnp.pad(k_hbt, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
    if V_ALIGNED > V:
      do_hbt = jnp.pad(do_hbt, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
    q_arg = q_hbt
    k_arg = k_hbt
    do_arg = do_hbt
    q_spec = pl.BlockSpec(
        (MB_QK, 1, BT, K_PADSIZE),
        lambda hg, b, c_rev, *_: (hg, b, NT - 1 - c_rev, 0),
    )
    k_spec = q_spec
    do_spec = pl.BlockSpec(
        (MB, 1, BT, V_ALIGNED),
        lambda hg, b, c_rev, *_: (hg, b, NT - 1 - c_rev, 0),
    )

  beta_r = beta_hbt.reshape(H_V, B, NT, 1, BT)
  g_r = g_hbt.reshape(H_V, B, NT, 1, BT)
  narrow_spec = pl.BlockSpec(
      (MB, 1, 1, 1, BT),
      lambda hg, b, c_rev, *_: (hg, b, NT - 1 - c_rev, 0, 0),
  )

  has_akk_in = akk is not None
  has_packed_meta = (
      packed_tok_meta is not None and packed_chunk_keep is not None
  )
  in_specs = [q_spec, k_spec, do_spec, narrow_spec, narrow_spec]
  call_args: list[jax.Array] = []
  if has_packed_meta:
    assert packed_chunk_keep is not None
    call_args.append(packed_chunk_keep.astype(jnp.int32))
  call_args.extend([q_arg, k_arg, do_arg, beta_r, g_r])

  if has_akk_in:
    assert akk is not None
    in_specs.append(
        pl.BlockSpec(
            (MB, 1, BT, BT),
            lambda hg, b, c_rev, *_: (hg, b, NT - 1 - c_rev, 0),
        )
    )
    call_args.append(akk)

  if has_packed_meta:
    assert packed_tok_meta is not None
    in_specs.append(
        pl.BlockSpec(
            (1, 1, 1, 8, BT),
            lambda hg, b, c_rev, *_: (0, b, NT - 1 - c_rev, 0, 0),
        )
    )
    call_args.append(packed_tok_meta[None, ...])

  def _out_map(hg, b, c_rev, *_):
    del c_rev
    return (b, hg, 0, 0)

  out_specs = [
      pl.BlockSpec((1, MB, K_PADSIZE, V_ALIGNED), index_map=_out_map),
      pl.BlockSpec((1, MB, K_PADSIZE, K_PADSIZE), index_map=_out_map),
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((B, H_V, K_PADSIZE, V_ALIGNED), jnp.float32),
      jax.ShapeDtypeStruct((B, H_V, K_PADSIZE, K_PADSIZE), jnp.float32),
  ]
  akk_out_dtype = jnp.float32 if two_pass_bf16 else act_dtype
  if not has_akk_in:
    out_specs.append(
        pl.BlockSpec(
            (MB, 1, BT, BT),
            lambda hg, b, c_rev, *_: (hg, b, NT - 1 - c_rev, 0),
        )
    )
    out_shapes.append(jax.ShapeDtypeStruct((H_V, B, T, BT), akk_out_dtype))

  scratches = [
      pltpu.VMEM((MB, K_PADSIZE, V_ALIGNED), jnp.float32),
      pltpu.VMEM((MB, K_PADSIZE, K_PADSIZE), jnp.float32),
  ]
  total_chunks = H_V * B * NT
  tri_inv_matmuls = max(0, 2 * int(math.log2(BT)) - 3) + (
      2 if two_pass_bf16 and act_dtype == jnp.bfloat16 else 0
  )
  tri_flops = 0 if has_akk_in else 2 * tri_inv_matmuls * BT * BT * BT
  qk_kk_flops = ((2 if has_akk_in else 4) * BT * BT * K_PADSIZE) // gqa_repeats
  if aliased_qkv:
    assert qkv_conv is not None
    in_bytes = int(B * T * 2 * q_size) * jnp.dtype(qkv_conv.dtype).itemsize
    extra_in = [do_arg, beta_r, g_r]
    if has_akk_in and akk is not None:
      extra_in.append(akk)
    if has_packed_meta:
      assert packed_chunk_keep is not None and packed_tok_meta is not None
      extra_in.extend([packed_chunk_keep.astype(jnp.int32), packed_tok_meta])
    in_bytes += sum(
        int(x.size) * jnp.dtype(x.dtype).itemsize for x in extra_in
    )
  else:
    in_bytes = sum(
        int(x.size) * jnp.dtype(x.dtype).itemsize for x in call_args
    )
  out_bytes = sum(
      int(s.size) * jnp.dtype(s.dtype).itemsize for s in out_shapes
  )
  cost = pl.CostEstimate(
      flops=total_chunks
      * (
          BT * K_PADSIZE * (6 * V_ALIGNED + 4 * K_PADSIZE + 2 * BT)
          + qk_kk_flops
          + 2 * BT * BT * V_ALIGNED
          + tri_flops
      ),
      transcendentals=total_chunks * (BT * BT + 3 * BT),
      bytes_accessed=in_bytes + out_bytes,
  )
  pallas_outs = pl.pallas_call(
      functools.partial(
          _gdn_local_cp_bwd_summary_kernel,
          BT=BT,
          MB=MB,
          MB_QK=MB_QK,
          NT=NT,
          K=K,
          V=V,
          K_PADSIZE=K_PADSIZE,
          V_ALIGNED=V_ALIGNED,
          GQA_REPEATS=gqa_repeats,
          SCALE=scale,
          USE_QK_L2NORM=use_qk_l2norm,
          TWO_PASS_BF16=two_pass_bf16,
          NATIVE_QKV_DO=can_native_qkv_do,
          ALIASED_QKV=aliased_qkv,
          HAS_AKK_IN=has_akk_in,
          HAS_PACKED_META=has_packed_meta,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1 if has_packed_meta else 0,
          grid=(H_V // MB, B, NT),
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=scratches,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=(
              None if get_interpret() else get_tpu_limits().vmem_limit_bytes
          ),
      ),
      out_shape=out_shapes,
      cost_estimate=cost,
      interpret=interpret,
  )(*call_args)

  if has_akk_in:
    ds_pad, dm_pad = pallas_outs
    assert akk is not None
    akk_eff = akk
  else:
    ds_pad, dm_pad, akk_eff = pallas_outs
  return ds_pad[..., :K, :V], dm_pad[..., :K, :K], akk_eff


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "use_qk_l2norm",
        "use_gate_in_kernel",
        "lower_bound",
        "disable_recompute",
        "context_parallel_metadata",
        "chunk_size",
        "max_num_segments",
        "has_initial_state",
        "per_channel_gate",
        "mini_batch",
        "two_pass_bf16",
    ],
)
@jaxtyping.jaxtyped
def chunk_kda_bwd_custom(
    scale: float | None,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    disable_recompute: bool,
    context_parallel_metadata: ContextParallelMetadata | None,
    chunk_size: int,
    max_num_segments: int | None,
    has_initial_state: bool,
    residuals: KdaResiduals,
    grad_outputs: tuple[
        Float[Array, "H B T_ORIG V"],
        Float[Array, "B H K V"] | Float[Array, "B N H K V"] | None,
    ],
    *,
    per_channel_gate: bool = True,
    mini_batch: int | None = None,
    two_pass_bf16: bool = False,
) -> tuple[
    Float[Array, "H B T_ORIG K"],
    Float[Array, "H B T_ORIG K"],
    Float[Array, "H B T_ORIG V"],
    Float[Array, "H B T_ORIG GW"],
    Float[Array, "H B T_ORIG"],
    Float[Array, "H"] | None,
    Float[Array, "H*GW"] | None,
    Float[Array, "B N H K V"] | None,
    None,
    Float[Array, "H_Q W K"] | None,
    Float[Array, "H_K W K"] | None,
    Float[Array, "H W V"] | None,
    Float[Array, "H_Q K"] | None,
    Float[Array, "H_K K"] | None,
    Float[Array, "H V"] | None,
]:
  """Runs the full KDA backward pipeline from forward residuals."""
  do, dht = grad_outputs
  # `do` stays in its input dtype (normally bf16) and is cast to f32 inside each
  # kernel after it is loaded into VMEM. Accumulation is f32 through
  # `preferred_element_type`.
  q = residuals.q
  k = residuals.k
  v = residuals.v
  beta = residuals.beta
  g_cumsum = residuals.g_cumsum
  Aqk = residuals.aqk
  Akk = residuals.akk
  initial_state = residuals.initial_state
  g_org = residuals.g_org
  a_log = residuals.a_log
  delta_time_bias = residuals.delta_time_bias
  h = residuals.h
  v_new = residuals.v_new
  g_dtype_marker = residuals.g_dtype_marker
  rstd_q = residuals.q_rstd
  rstd_k = residuals.k_rstd
  cu_seqlens = residuals.cu_seqlens
  aligned_cu = residuals.aligned_cu_seqlens
  chunk_indices_bwd = residuals.chunk_indices
  segment_ids_aligned = residuals.aligned_segment_ids
  segment_ids = residuals.segment_ids
  cp_metadata = residuals.cp_metadata

  cu_seqlens_bwd = None
  T_orig = None
  if cu_seqlens is not None:
    _, _, T_orig, _ = do.shape
    [do], [], _, _ = _align_seqs(
        [do],
        [],
        cu_seqlens,
        align=chunk_size,
        aligned_cu_seqlens=aligned_cu,
    )
    cu_seqlens_bwd = aligned_cu
  elif do.shape[2] < q.shape[2]:
    T_orig = do.shape[2]
    do = jnp.pad(do, ((0, 0), (0, 0), (0, q.shape[2] - T_orig), (0, 0)))

  _cp_active = (
      context_parallel_metadata is not None
      and context_parallel_metadata.is_cp_enabled
  )
  _legacy_varlen_cp = _cp_active and (
      cu_seqlens is not None or aligned_cu is not None
  )
  _inplace_cp = _cp_active and not _legacy_varlen_cp

  if _legacy_varlen_cp:
    assert context_parallel_metadata is not None
    if segment_ids is None:
      raise ValueError("backward CP requires rank-local `segment_ids`.")
    if cp_metadata is None:
      raise ValueError("backward CP requires metadata retained by forward.")
    is_first_rank, is_last_rank, pre_num_ranks, post_num_ranks = cp_metadata
    context_parallel_metadata = dataclasses.replace(
        context_parallel_metadata,
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
        pre_num_ranks=pre_num_ranks,
        post_num_ranks=post_num_ranks,
    )

  effective_max_num_segments = max_num_segments
  if aligned_cu is not None:
    effective_max_num_segments = aligned_cu.shape[-1] - 1

  original_cu_seqlens = cu_seqlens
  g = g_cumsum
  cu_seqlens = cu_seqlens_bwd
  chunk_indices = chunk_indices_bwd
  segment_ids = (
      segment_ids_aligned
      if segment_ids_aligned is not None
      else segment_ids
  )
  max_num_segments = effective_max_num_segments

  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT
  scale = K ** -0.5 if scale is None else scale
  halo_len_raw = (
      residuals.q_raw.shape[2] - T if residuals.q_raw is not None else 0
  )
  use_inplace_packed = (
      cu_seqlens is None
      and segment_ids_aligned is None
      and segment_ids is not None
      and not _legacy_varlen_cp
  )
  seg_canon = None
  seg_canon_conv = None
  packed_tok_meta = None
  packed_chunk_keep = None
  packed_valid_mask = None
  if use_inplace_packed:
    assert segment_ids is not None
    seg_2d = segment_ids[None, :] if segment_ids.ndim == 1 else segment_ids
    if _inplace_cp:
      assert context_parallel_metadata is not None
      seg_canon, act_doc, is_new_doc, halo_seg_canon, _ = (
          cp_canonicalize_packed_segment_ids(
              seg_2d,
              axis_name=context_parallel_metadata.axis_name,
              cp_size=context_parallel_metadata.cp_size,
              halo_len=halo_len_raw,
              precomputed_cp_halo=residuals.cp_seg_halo,
          )
      )
      seg_canon_conv = (
          jnp.concatenate(
              [halo_seg_canon[:, -halo_len_raw:], seg_canon], axis=1
          )
          if halo_len_raw > 0 and halo_seg_canon is not None
          else seg_canon
      )
      packed_tok_meta, packed_chunk_keep = compute_packed_chunk_metadata(
          seg_canon,
          act_doc,
          is_new_doc,
          BT,
          has_initial_state=False,
          conv_halo_seg=halo_seg_canon,
      )
    else:
      seg_canon, act_doc, is_new_doc = canonicalize_packed_segment_ids(seg_2d)
      seg_canon_conv = seg_canon
      packed_tok_meta, packed_chunk_keep = compute_packed_chunk_metadata(
          seg_canon,
          act_doc,
          is_new_doc,
          BT,
          has_initial_state=(initial_state is not None),
      )
    packed_valid_mask = seg_canon > 0
  elif (cu_seqlens is None) and (segment_ids is not None):
    # per-batch cu_seqlens [B, N+1]
    caller_max_num_segments = max_num_segments
    if caller_max_num_segments is not None:
      max_num_segments = caller_max_num_segments
    else:
      if initial_state is None:
        raise ValueError(
            "`max_num_segments` is required when `segment_ids` is provided "
            "without `initial_state`."
        )
      # Varlen: (B, N, H, K, V)
      max_num_segments = initial_state.shape[1]
    cu_seqlens = segment_ids_to_seqlens(segment_ids, max_segs=max_num_segments)

  # max_num_segments must be provided when segment_ids are used, unless
  # cu_seqlens was already derived upstream or in-place packing is active.
  if (
      segment_ids is not None
      and not use_inplace_packed
      and cu_seqlens is None
      and max_num_segments is None
  ):
    raise ValueError(
        "`max_num_segments` is required when segment metadata is unavailable."
    )
  _, _, do_seq_len, _ = do.shape
  if do_seq_len != T:
    raise ValueError(f"aligned do has T={do_seq_len}, expected {T}")
  if Akk is None:
    raise ValueError("KDA backward requires Akk in residuals.")
  if Aqk.shape[-1] != BT or Akk.shape[-1] != BT:
    raise ValueError(f"Aqk/Akk block size must equal chunk_size={BT}")

  # When use_gate_in_kernel=True and disable_recompute=False, g_cumsum is
  # None from forward (freed to save memory) and will be recomputed in
  # Stage 0 below from g_org.  Only assert non-None when we won't recompute.
  if not (use_gate_in_kernel and not disable_recompute):
    if g is None:
      raise ValueError("g (post-cumsum, log2 space) must be provided")
  if T % BT != 0:
    raise ValueError(f"T={T} must be divisible by chunk_size={BT}")

  # ============= assert input shapes and static properties =============
  # initial_state/dht: [B, H, K, V] (non-varlen) or [N, H, K, V] (varlen)

  if disable_recompute:
    # Path A: save-h fast path.
    if use_gate_in_kernel:
      if a_log is None:
        raise ValueError("a_log is required when use_gate_in_kernel=True")
      assert g_org is not None
      g_cumsum = kda_gate_chunk_cumsum(
        g=g_org,
        a_log=a_log,
        chunk_size=BT,
        scale=RCP_LN2,
        delta_time_bias=delta_time_bias,
        lower_bound=lower_bound,
        packed_tok_meta=packed_tok_meta,
      )
      g = g_cumsum  # already [H,B,T,K]

    if h is None:
      raise ValueError("saved h is required when recompute is disabled")
    _, _, h_num_chunks, _, _ = h.shape
    if h_num_chunks != NT:
      raise ValueError(f"saved h has NT={h_num_chunks}, expected {NT}")

    assert g is not None
    if v_new is not None:
      # v_new was saved in forward residuals; w, qg, kg are computed on-the-fly
      # in VMEM inside _fused_dhu_wy_intra_cumsum_kernel.
      w, qg, kg = None, None, None
    elif packed_tok_meta is not None:
      w, u_rec, qg, kg = _recompute_w_u_fwd(
          q=q,
          k=k,
          v=v,
          beta=beta,
          A=Akk,
          gk=g,
          chunk_size=BT,
          two_pass_bf16=two_pass_bf16,
          packed_tok_meta=packed_tok_meta,
      )
      prec_vn = (
          jax.lax.Precision.HIGHEST
          if q.dtype != jnp.bfloat16
          else (jax.lax.Precision.HIGH if two_pass_bf16 else None)
      )
      v_new_c = u_rec.reshape(H, B, NT, BT, V).astype(
          jnp.float32
      ) - jnp.matmul(
          w.reshape(H, B, NT, BT, K).astype(jnp.float32),
          h.astype(jnp.float32),
          precision=prec_vn,
          preferred_element_type=jnp.float32,
      )
      valid_5d = packed_tok_meta[:, :, 4, :].astype(jnp.float32)[
          None, :, :, :, None
      ]
      wy_dtype = (
          jnp.float32
          if (two_pass_bf16 and v.dtype == jnp.bfloat16)
          else v.dtype
      )
      v_new = (v_new_c * valid_5d).reshape(H, B, T, V).astype(wy_dtype)
    else:
      # M1 fusion: recompute w/qg/kg + v_new in one kernel (no u HBM round-trip).
      # Broadcast scalar gate to key width K for the recompute helper.
      g_wide = (
          g
          if g.shape[-1] == q.shape[-1]
          else jnp.broadcast_to(g, g.shape[:-1] + (q.shape[-1],))
      )
      w, qg, kg, v_new = fused_recompute_w_u_vnew_from_h_pallas(
          q=q,
          k=k,
          v=v,
          beta=beta,
          A=Akk,
          g=g_wide,
          h=h,
          chunk_size=BT,
          two_pass_bf16=two_pass_bf16,
      )
  else:
    # Path B: full recompute fallback.
    if use_gate_in_kernel:
      if a_log is None:
        raise ValueError("a_log is required when use_gate_in_kernel=True")
      assert g_org is not None
      g_cumsum = kda_gate_chunk_cumsum(
        g=g_org,
        a_log=a_log,
        chunk_size=BT,
        scale=RCP_LN2,
        delta_time_bias=delta_time_bias,
        lower_bound=lower_bound,
        packed_tok_meta=packed_tok_meta,
      )
      g = g_cumsum  # already [H,B,T,K]

    # recompute_w_u_fwd is natively [H,B,T,X]
    w, u, qg, kg = _recompute_w_u_fwd(
      k=k,
      v=v,
      beta=beta,
      A=Akk,
      q=q,
      gk=g,
      chunk_size=BT,
      two_pass_bf16=two_pass_bf16,
      packed_tok_meta=packed_tok_meta,
    )
    if kg is None:
      raise RuntimeError("KDA recompute did not produce gated keys.")

    assert g is not None
    if packed_tok_meta is not None:
      stage34_cu = jnp.broadcast_to(
          jnp.asarray([0, T], dtype=jnp.int32), (B, 2)
      )
      block_ids = jnp.arange(NT, dtype=jnp.int32)
      fixed_ci = jnp.stack([jnp.zeros_like(block_ids), block_ids], axis=-1)
      stage34_ci = jnp.broadcast_to(fixed_ci[None, :, :], (B, NT, 2))
      stage34_h0 = (
          initial_state[:, None, ...]
          if (initial_state is not None and initial_state.ndim == 4)
          else initial_state
      )
      _, _, h, v_new = chunk_kda_fwd_h_o_varlen(
          w=w,
          u=u,
          kg=kg,
          gk=g,
          q=q,
          A=Aqk,
          cu_seqlens=stage34_cu,
          chunk_indices=stage34_ci,
          initial_state=stage34_h0,
          output_final_state=False,
          scale=scale,
          chunk_size=BT,
          store_h=True,
          store_v_new=True,
          mini_batch=mini_batch,
          two_pass_bf16=two_pass_bf16,
          packed_tok_meta=packed_tok_meta,
          packed_chunk_keep=packed_chunk_keep,
      )
    else:
      # Pass scalar gates via `g` with plain keys `k`; pass per-channel gates
      # via `gk` with gated keys `kg`.
      scalar_gate = g.shape[-1] == 1
      h, v_new, _ = chunk_gated_delta_rule_fwd_h(
          k=k if scalar_gate else kg,
          w=w,
          u=u,
          g=g[..., 0] if scalar_gate else None,
          gk=None if scalar_gate else g,
          initial_state=initial_state,
          output_final_state=False,
          chunk_size=chunk_size,
          cu_seqlens=cu_seqlens,
          chunk_indices=chunk_indices,
          use_exp2=True,
      )
      # Varlen: pad h from NT_total to NT chunks (padding chunks get zero state)
      _, _, h_num_chunks, _, _ = h.shape
      if cu_seqlens is not None and h_num_chunks < NT:
        h = jnp.pad(
            h, ((0, 0), (0, 0), (0, NT - h_num_chunks), (0, 0), (0, 0))
        )

  assert v_new is not None
  if packed_valid_mask is not None:
    do = jnp.where(packed_valid_mask[None, :, :, None], do, 0.0)
  # ---- Stage 1: dAqk and initial dv ----
  dAqk, dv = chunk_kda_bwd_dAv_kernel(
    q=q,
    k=k,
    v=v_new,
    do=do,
    A=Aqk,
    scale=scale,
    chunk_size=chunk_size,
    two_pass_bf16=two_pass_bf16,
  )

  if _inplace_cp:
    assert context_parallel_metadata is not None
    if qg is None or kg is None or w is None:
      assert g is not None
      g_wide = (
          g
          if g.shape[-1] == q.shape[-1]
          else jnp.broadcast_to(g, g.shape[:-1] + (q.shape[-1],))
      )
      w, _, qg, kg = _recompute_w_u_fwd(
          k=k,
          v=v,
          beta=beta,
          A=Akk,
          q=q,
          gk=g_wide,
          chunk_size=BT,
          two_pass_bf16=two_pass_bf16,
          packed_tok_meta=packed_tok_meta,
      )
    assert g is not None
    ds_local, dm_local = compute_local_cp_bwd_summary(
        qg=qg,
        kg=kg,
        w=w,
        do=do,
        dv0=dv,
        gk=g,
        scale=scale,
        chunk_size=BT,
        packed_chunk_keep=packed_chunk_keep,
        packed_tok_meta=packed_tok_meta,
    )
    dht = cp_associative_scan_bwd(
        ds_local,
        dm_local,
        axis_name=context_parallel_metadata.axis_name,
        cp_size=context_parallel_metadata.cp_size,
    )
    initial_state = None
  elif _legacy_varlen_cp:
    assert context_parallel_metadata is not None
    if segment_ids is None:
      raise ValueError("backward CP requires rank-local segment_ids")
    if context_parallel_metadata.post_num_ranks is None:
      raise ValueError("backward CP requires post_num_ranks")
    if context_parallel_metadata.is_last_rank is None:
      raise ValueError("backward CP requires is_last_rank")
    # pre_process requires segment_ids length == aligned T.
    # Caller's segment_ids is un-aligned (length T_orig); pad with 0
    # (= padding seg id, OOB chunks naturally inactive).
    if segment_ids.ndim == 1:
      segment_ids = segment_ids[None,]
    T_seg = segment_ids.shape[-1]
    if T_seg < T:
      pad_width = ((0, 0), (0, T - T_seg))
      segment_ids = jnp.pad(segment_ids, pad_width)
    elif T_seg > T:
      segment_ids = segment_ids[..., :T]
    if qg is None or kg is None or w is None:
      assert g is not None
      g_wide = (
          g
          if g.shape[-1] == q.shape[-1]
          else jnp.broadcast_to(g, g.shape[:-1] + (q.shape[-1],))
      )
      w, _, qg, kg = _recompute_w_u_fwd(
          k=k,
          v=v,
          beta=beta,
          A=Akk,
          q=q,
          gk=g_wide,
          chunk_size=BT,
          two_pass_bf16=two_pass_bf16,
      )
    # Inputs already in [H, B, T, X]; pre_process consumes this layout
    # directly (no transpose round-trip).
    dS_ext, dM = chunk_gated_delta_rule_bwd_dhu_pre_process(
      q=qg,
      k=kg,
      w=w,
      do=do,
      dv=dv,
      gk=g,
      scale=scale,
      segment_ids=segment_ids,
      chunk_size=BT,
      use_exp2=True,
    )
    # Pack dS_ext [B,H,K,V] and dM [B,H,K,K] into one tensor along the last
    # axis so a single all_gather covers both, halving the CP collective cost.
    packed = jnp.concatenate([dS_ext, dM], axis=-1)  # [B, H, K, V+K]
    packed_all, _ = all_gather_into_tensor(packed, context_parallel_metadata.axis_name)
    dS_ext_all = packed_all[..., :V]                  # [cp, B, H, K, V]
    dM_all = packed_all[..., V:V + K]                 # [cp, B, H, K, K]
    rank = jax.lax.axis_index(context_parallel_metadata.axis_name)
    post_num = jnp.asarray(context_parallel_metadata.post_num_ranks)
    is_last = jnp.asarray(context_parallel_metadata.is_last_rank)
    ds_list = []
    for b in range(B):
      ds_b = _merge_dht(
        dS_ext_all[:, b:b+1],  # [cp, 1, H, K, V]
        dM_all[:, b:b+1],      # [cp, 1, H, K, K]
        rank=rank,
        post_num_ranks=post_num[b],
        is_last_rank=is_last[b],
      )
      ds_list.append(ds_b)  # [1, H, K, V]
    dS_in = jnp.concatenate(ds_list, axis=0)  # [B, H, K, V]

    # dS_in: [B, H, K, V] — merged downstream gradient for each batch element.
    # M4 expects dht as [B, N, H, K, V] with per-batch segment indexing.
    # dS_in[b] goes to slot (last_seg_id_b - 1) in its own N dimension.
    assert max_num_segments is not None
    N = max_num_segments
    dht = jnp.zeros((B, N, H, K, V), dtype=jnp.float32)
    max_per_batch = jnp.max(segment_ids, axis=1)  # [B]
    for b in range(B):
      last_seg_id_b = max_per_batch[b]
      has_real_b = last_seg_id_b > 0
      dht_slot = jnp.maximum(last_seg_id_b - 1, 0)
      dht_slot = jnp.minimum(dht_slot, N - 1)
      dht_value_b = jnp.where(has_real_b, dS_in[b], jnp.zeros_like(dS_in[b]))
      dht = dht.at[b, dht_slot, :, :, :].set(dht_value_b)
    initial_state = None

  # M4 requires 2D segment_ids [B, T]
  if segment_ids is not None and segment_ids.ndim == 1:
    segment_ids = segment_ids[None,]

  # ---- Stage 2+3+4+5: M4 mega fusion (dhu + WY + intra + cumsum) ----
  # M4 expects dht as [B, N, H, K, V].
  # Normalize 4D dht to 5D (uniform: insert N=1 at dim1).
  if dht is not None and dht.ndim == 4 and (
      segment_ids is None or use_inplace_packed
  ):
    dht = dht[:, None, :, :, :]
  dht_m4 = dht
  _l2norm_in_kernel = os.environ.get("KDA_L2NORM_BWD_IN_KERNEL", "1") == "1"
  dq, dk, dv, db, dg, dh0 = _fused_dhu_wy_intra_cumsum_pallas_jit(
      per_channel_gate=per_channel_gate,
      q=q,
      k=k,
      v=v,
      v_new=v_new,
      qg=qg,
      kg=kg,
      w=w,
      g=g,
      beta=beta,
      A=Akk,
      h=h,
      do=do,
      dv0=dv,
      dAqk=dAqk,
      dht=dht_m4,
      scale=scale,
      segment_ids=None if use_inplace_packed else segment_ids,
      chunk_size=chunk_size,
      use_exp2=True,
      return_dh0=initial_state is not None,
      max_num_segments=max_num_segments,
      mini_batch=mini_batch,
      rstd_q=rstd_q if (use_qk_l2norm and _l2norm_in_kernel) else None,
      rstd_k=rstd_k if (use_qk_l2norm and _l2norm_in_kernel) else None,
      two_pass_bf16=two_pass_bf16,
      packed_tok_meta=packed_tok_meta,
      packed_chunk_keep=packed_chunk_keep,
  )

  dA, dbias = None, None
  if use_gate_in_kernel:
    assert g_org is not None and a_log is not None
    dg, dA, dbias = kda_gate_bwd(
      g=g_org,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      dyg=dg,
      lower_bound=lower_bound,
      valid_mask=packed_valid_mask,
    )

  # Non-varlen: squeeze dh0 5D [B,1,H,K,V] → 4D [B,H,K,V]
  # (varlen callers pass 5D, so dh0 stays 5D)
  _is_varlen = cu_seqlens is not None or (
      segment_ids is not None and not use_inplace_packed
  )
  if dh0 is not None and dh0.ndim == 5 and not _is_varlen:
    dh0 = dh0[:, 0]

  cu_seqlens = original_cu_seqlens
  if use_qk_l2norm and not _l2norm_in_kernel:
    assert rstd_q is not None and rstd_k is not None
    dq = l2norm_bwd(q, rstd_q, dq)
    dk = l2norm_bwd(k, rstd_k, dk)
    if packed_valid_mask is not None:
      mask_4d = packed_valid_mask[None, :, :, None]
      dq = jnp.where(mask_4d, dq, 0.0)
      dk = jnp.where(mask_4d, dk, 0.0)

  dw_q = dw_k = dw_v = None
  db_q = db_k = db_v = None
  if residuals.q_raw is not None:
    assert residuals.k_raw is not None and residuals.v_raw is not None
    assert (
        residuals.conv_weight_q is not None
        and residuals.conv_weight_k is not None
        and residuals.conv_weight_v is not None
    )
    if halo_len_raw > 0:
      assert context_parallel_metadata is not None
      axis_name = context_parallel_metadata.axis_name
      cp_size = context_parallel_metadata.cp_size
      pad_spec = ((0, 0), (0, 0), (halo_len_raw, 0), (0, 0))
      dq_ext, dw_q, db_q = _conv1d_silu_bwd(
          residuals.q_raw,
          residuals.conv_weight_q,
          residuals.conv_bias_q,
          jnp.pad(dq, pad_spec),
          segment_ids_aligned,
          packed_canonical_seg=seg_canon_conv,
      )
      dk_ext, dw_k, db_k = _conv1d_silu_bwd(
          residuals.k_raw,
          residuals.conv_weight_k,
          residuals.conv_bias_k,
          jnp.pad(dk, pad_spec),
          segment_ids_aligned,
          packed_canonical_seg=seg_canon_conv,
      )
      dv_ext, dw_v, db_v = _conv1d_silu_bwd(
          residuals.v_raw,
          residuals.conv_weight_v,
          residuals.conv_bias_v,
          jnp.pad(dv, pad_spec),
          segment_ids_aligned,
          packed_canonical_seg=seg_canon_conv,
      )
      dq = dq_ext[:, :, halo_len_raw:, :].at[:, :, -halo_len_raw:, :].add(
          cp_shift_left(dq_ext[:, :, :halo_len_raw, :], axis_name, cp_size)
      )
      dk = dk_ext[:, :, halo_len_raw:, :].at[:, :, -halo_len_raw:, :].add(
          cp_shift_left(dk_ext[:, :, :halo_len_raw, :], axis_name, cp_size)
      )
      dv = dv_ext[:, :, halo_len_raw:, :].at[:, :, -halo_len_raw:, :].add(
          cp_shift_left(dv_ext[:, :, :halo_len_raw, :], axis_name, cp_size)
      )
    else:
      dq, dw_q, db_q = _conv1d_silu_bwd(
          residuals.q_raw,
          residuals.conv_weight_q,
          residuals.conv_bias_q,
          dq,
          segment_ids_aligned,
          packed_canonical_seg=seg_canon,
      )
      dk, dw_k, db_k = _conv1d_silu_bwd(
          residuals.k_raw,
          residuals.conv_weight_k,
          residuals.conv_bias_k,
          dk,
          segment_ids_aligned,
          packed_canonical_seg=seg_canon,
      )
      dv, dw_v, db_v = _conv1d_silu_bwd(
          residuals.v_raw,
          residuals.conv_weight_v,
          residuals.conv_bias_v,
          dv,
          segment_ids_aligned,
          packed_canonical_seg=seg_canon,
      )

  if cu_seqlens is not None:
    dq = _unalign_output(dq, cu_seqlens, aligned_cu, T_orig)
    dk = _unalign_output(dk, cu_seqlens, aligned_cu, T_orig)
    dv = _unalign_output(dv, cu_seqlens, aligned_cu, T_orig)
    dg = _unalign_output(dg, cu_seqlens, aligned_cu, T_orig)
    db = _unalign_output(db, cu_seqlens, aligned_cu, T_orig)
  elif T_orig is not None:
    dq = dq[:, :, :T_orig, :]
    dk = dk[:, :, :T_orig, :]
    dv = dv[:, :, :T_orig, :]
    dg = dg[:, :, :T_orig, :]
    db = db[:, :, :T_orig]

  if dh0 is not None and dh0.ndim == 4 and has_initial_state:
    dh0 = dh0[:, None]

  return (
      dq.astype(q.dtype),
      dk.astype(k.dtype),
      dv.astype(v.dtype),
      dg.astype(g_dtype_marker.dtype),
      db.astype(beta.dtype),
      dA,
      dbias,
      dh0,
      None,
      dw_q,
      dw_k,
      dw_v,
      db_q,
      db_k,
      db_v,
  )


def chunk_gdn_bwd_custom(
    scale: float,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    chunk_size: int,
    has_initial_state: bool,
    has_fused_qkv: bool,
    residuals: GdnResiduals,
    grad_outputs: tuple[
        Float[Array, "B T H_V V"],
        Float[Array, "B H_V K V"] | None,
    ],
    *,
    context_parallel_metadata: ContextParallelMetadata | None = None,
    mini_batch: int | None = None,
    two_pass_bf16: bool = False,
) -> tuple[
    Float[Array, "B T C_QKV"] | None,
    Float[Array, "B T H_QK K"] | None,
    Float[Array, "B T H_QK K"] | None,
    Float[Array, "B T H_V V"] | None,
    Float[Array, "B T H_V"],
    Float[Array, "B T H_V"],
    Float[Array, "H_V"] | None,
    Float[Array, "H_V"] | None,
    Float[Array, "B H_V K V"] | None,
    Float[Array, "W C_QKV"] | None,
    Float[Array, "C_QKV"] | None,
]:
  """Runs the natural-layout `[B, T, ...]` GDN backward pipeline."""
  do_bt, dht = grad_outputs
  H_QK = num_k_heads
  H_V = num_v_heads
  K = head_k_dim
  V = head_v_dim
  BT = chunk_size
  gqa_repeats = H_V // H_QK
  q_size = H_QK * K
  k_size = H_QK * K
  B, T, _, _ = do_bt.shape
  act_dtype = do_bt.dtype
  _cp_active = (
      context_parallel_metadata is not None
      and context_parallel_metadata.is_cp_enabled
  )
  halo_len = (
      residuals.qkv_raw.shape[1] - T
      if residuals.conv_weight is not None and residuals.qkv_raw is not None
      else 0
  )

  seg_canon = None
  seg_canon_ext = None
  packed_tok_meta = None
  packed_chunk_keep = None
  packed_valid_mask = None
  if residuals.segment_ids is not None:
    if _cp_active:
      assert context_parallel_metadata is not None
      seg_canon, act_doc, is_new_doc, halo_seg_canon, _ = (
          cp_canonicalize_packed_segment_ids(
              residuals.segment_ids,
              axis_name=context_parallel_metadata.axis_name,
              cp_size=context_parallel_metadata.cp_size,
              halo_len=halo_len,
              precomputed_cp_halo=residuals.cp_seg_halo,
          )
      )
      seg_canon_ext = (
          jnp.concatenate([halo_seg_canon, seg_canon], axis=1)
          if halo_len > 0 and halo_seg_canon is not None
          else seg_canon
      )
      packed_tok_meta, packed_chunk_keep = compute_packed_chunk_metadata(
          seg_canon,
          act_doc,
          is_new_doc,
          BT,
          has_initial_state=False,
          conv_halo_seg=halo_seg_canon,
      )
    else:
      seg_canon, act_doc, is_new_doc = canonicalize_packed_segment_ids(
          residuals.segment_ids
      )
      seg_canon_ext = seg_canon
      packed_tok_meta, packed_chunk_keep = compute_packed_chunk_metadata(
          seg_canon,
          act_doc,
          is_new_doc,
          BT,
          has_initial_state=has_initial_state,
      )
    packed_valid_mask = seg_canon > 0

  g = residuals.g_cumsum
  if g is None:
    if not use_gate_in_kernel or residuals.g_org is None or residuals.a_log is None:
      raise ValueError("GDN backward requires g_cumsum or (g_org, a_log).")
    g = kda_gate_chunk_cumsum(
        g=residuals.g_org,
        a_log=residuals.a_log,
        chunk_size=BT,
        scale=RCP_LN2,
        delta_time_bias=residuals.delta_time_bias,
        lower_bound=None,
        packed_tok_meta=packed_tok_meta,
    )
  if residuals.h is None:
    raise ValueError("GDN backward requires h in residuals.")

  akk_eff = residuals.akk
  if _cp_active:
    assert context_parallel_metadata is not None
    ds_local, dm_local, akk_eff = compute_gdn_local_cp_bwd_summary(
        qkv_conv=residuals.qkv_conv,
        q_bt=residuals.q,
        k_bt=residuals.k,
        beta_hbt=residuals.beta,
        g_hbt=g,
        do_bt=do_bt,
        akk=residuals.akk,
        num_k_heads=H_QK,
        num_v_heads=H_V,
        head_k_dim=K,
        head_v_dim=V,
        scale=scale,
        chunk_size=BT,
        use_qk_l2norm=use_qk_l2norm,
        two_pass_bf16=two_pass_bf16,
        mini_batch=mini_batch,
        packed_tok_meta=packed_tok_meta,
        packed_chunk_keep=packed_chunk_keep,
    )
    dht = cp_associative_scan_bwd(
        ds_local,
        dm_local,
        axis_name=context_parallel_metadata.axis_name,
        cp_size=context_parallel_metadata.cp_size,
    )

  dht_5d = dht[:, None] if dht is not None else None
  interpret = get_interpret()
  can_native_qkv_do = (
      K % 128 == 0 and V % 128 == 0 and BT % 8 == 0 and not interpret
  )

  dy_conv = None
  dq_bt = dk_bt = dv_bt = None
  if residuals.qkv_conv is not None and can_native_qkv_do:
    dy_conv, _, _, db_hbt, dg_hbt, dh0 = _fused_dhu_wy_intra_cumsum_pallas_jit(
        q=None,
        k=None,
        v=None,
        v_new=None,
        qg=None,
        kg=None,
        w=None,
        g=g,
        beta=residuals.beta,
        A=akk_eff,
        h=residuals.h,
        do=do_bt,
        dv0=None,
        dAqk=None,
        dht=dht_5d,
        scale=scale,
        chunk_size=BT,
        use_exp2=True,
        per_channel_gate=False,
        mini_batch=mini_batch,
        return_dh0=has_initial_state,
        gqa_repeats=gqa_repeats,
        dqkv_dtype=act_dtype,
        qkv_conv=residuals.qkv_conv,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        g_is_cumsummed=True,
        two_pass_bf16=two_pass_bf16,
        packed_tok_meta=packed_tok_meta,
        packed_chunk_keep=packed_chunk_keep,
    )
  elif residuals.qkv_conv is None and can_native_qkv_do:
    dq_bt, dk_bt, dv_bt, db_hbt, dg_hbt, dh0 = (
        _fused_dhu_wy_intra_cumsum_pallas_jit(
            q=residuals.q,
            k=residuals.k,
            v=residuals.v,
            v_new=None,
            qg=None,
            kg=None,
            w=None,
            g=g,
            beta=residuals.beta,
            A=akk_eff,
            h=residuals.h,
            do=do_bt,
            dv0=None,
            dAqk=None,
            dht=dht_5d,
            scale=scale,
            chunk_size=BT,
            use_exp2=True,
            per_channel_gate=False,
            mini_batch=mini_batch,
            return_dh0=has_initial_state,
            gqa_repeats=gqa_repeats,
            dqkv_dtype=act_dtype,
            native_bt_qkv=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            g_is_cumsummed=True,
            two_pass_bf16=two_pass_bf16,
            packed_tok_meta=packed_tok_meta,
            packed_chunk_keep=packed_chunk_keep,
        )
    )
  else:
    if residuals.qkv_conv is not None:
      qkv_c = residuals.qkv_conv
      q_hbt = qkv_c[:, :, :q_size].reshape(B, T, H_QK, K).transpose(2, 0, 1, 3)
      k_hbt = (
          qkv_c[:, :, q_size : q_size + k_size]
          .reshape(B, T, H_QK, K)
          .transpose(2, 0, 1, 3)
      )
      v_hbt = (
          qkv_c[:, :, q_size + k_size :]
          .reshape(B, T, H_V, V)
          .transpose(2, 0, 1, 3)
      )
    else:
      assert (
          residuals.q is not None
          and residuals.k is not None
          and residuals.v is not None
      )
      q_hbt = residuals.q.transpose(2, 0, 1, 3)
      k_hbt = residuals.k.transpose(2, 0, 1, 3)
      v_hbt = residuals.v.transpose(2, 0, 1, 3)
    do_hbt = do_bt.transpose(2, 0, 1, 3)
    dq_hbt, dk_hbt, dv_hbt, db_hbt, dg_hbt, dh0 = (
        _fused_dhu_wy_intra_cumsum_pallas_jit(
            q=q_hbt,
            k=k_hbt,
            v=v_hbt,
            v_new=None,
            qg=None,
            kg=None,
            w=None,
            g=g,
            beta=residuals.beta,
            A=akk_eff,
            h=residuals.h,
            do=do_hbt,
            dv0=None,
            dAqk=None,
            dht=dht_5d,
            scale=scale,
            chunk_size=BT,
            use_exp2=True,
            per_channel_gate=False,
            mini_batch=mini_batch,
            return_dh0=has_initial_state,
            gqa_repeats=gqa_repeats,
            dqkv_dtype=act_dtype,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            g_is_cumsummed=True,
            two_pass_bf16=two_pass_bf16,
            packed_tok_meta=packed_tok_meta,
            packed_chunk_keep=packed_chunk_keep,
        )
    )
    assert dq_hbt is not None and dk_hbt is not None and dv_hbt is not None
    dq_bt = dq_hbt.transpose(1, 2, 0, 3)
    dk_bt = dk_hbt.transpose(1, 2, 0, 3)
    dv_bt = dv_hbt.transpose(1, 2, 0, 3)
    if residuals.qkv_conv is not None:
      dy_conv = jnp.concatenate(
          [
              dq_bt.reshape(B, T, q_size),
              dk_bt.reshape(B, T, k_size),
              dv_bt.reshape(B, T, H_V * V),
          ],
          axis=-1,
      )

  da_log = ddt_bias = None
  if use_gate_in_kernel:
    assert residuals.g_org is not None and residuals.a_log is not None
    dg_hbt, da_log, ddt_bias = kda_gate_bwd(
        g=residuals.g_org,
        a_log=residuals.a_log,
        delta_time_bias=residuals.delta_time_bias,
        dyg=dg_hbt,
        lower_bound=None,
        valid_mask=packed_valid_mask,
    )
  dg_bt = dg_hbt[..., 0].transpose(1, 2, 0).astype(residuals.g_dtype_marker.dtype)

  if use_beta_sigmoid_in_kernel:
    b_f32 = residuals.beta.astype(jnp.float32)
    db_hbt = db_hbt.astype(jnp.float32) * b_f32 * (1.0 - b_f32)
  db_bt = db_hbt.transpose(1, 2, 0).astype(jnp.dtype(residuals.beta_dtype))

  dconv_w = dconv_b = None
  if residuals.conv_weight is not None:
    assert residuals.qkv_raw is not None and dy_conv is not None
    if halo_len > 0:
      assert context_parallel_metadata is not None
      dy_conv_ext = jnp.pad(dy_conv, ((0, 0), (halo_len, 0), (0, 0)))
      dqkv_ext, dconv_w, dconv_b = causal_conv1d_silu_bwd_bt(
          residuals.qkv_raw,
          residuals.conv_weight,
          residuals.conv_bias,
          dy_conv_ext,
          canonical_segment_ids=seg_canon_ext,
      )
      dqkv_in = dqkv_ext[:, halo_len:, :].at[:, -halo_len:, :].add(
          cp_shift_left(
              dqkv_ext[:, :halo_len, :],
              context_parallel_metadata.axis_name,
              context_parallel_metadata.cp_size,
          )
      )
    else:
      dqkv_in, dconv_w, dconv_b = causal_conv1d_silu_bwd_bt(
          residuals.qkv_raw,
          residuals.conv_weight,
          residuals.conv_bias,
          dy_conv,
          canonical_segment_ids=seg_canon,
      )
  else:
    dqkv_in = dy_conv

  if has_fused_qkv:
    assert dqkv_in is not None
    dqkv_out = dqkv_in.astype(act_dtype)
    dq_out = dk_out = dv_out = None
  elif dqkv_in is not None:
    dq_out = dqkv_in[:, :, :q_size].reshape(B, T, H_QK, K).astype(act_dtype)
    dk_out = (
        dqkv_in[:, :, q_size : q_size + k_size]
        .reshape(B, T, H_QK, K)
        .astype(act_dtype)
    )
    dv_out = dqkv_in[:, :, q_size + k_size :].reshape(B, T, H_V, V).astype(act_dtype)
    dqkv_out = None
  else:
    assert dq_bt is not None and dk_bt is not None and dv_bt is not None
    dq_out = dq_bt.astype(act_dtype)
    dk_out = dk_bt.astype(act_dtype)
    dv_out = dv_bt.astype(act_dtype)
    dqkv_out = None

  if dh0 is not None and dh0.ndim == 5:
    dh0 = dh0[:, 0]
  if dh0 is not None and residuals.initial_state is not None:
    dh0 = dh0.astype(residuals.initial_state.dtype)

  return (
      dqkv_out,
      dq_out,
      dk_out,
      dv_out,
      dg_bt,
      db_bt,
      da_log,
      ddt_bias,
      dh0,
      dconv_w,
      dconv_b,
  )
