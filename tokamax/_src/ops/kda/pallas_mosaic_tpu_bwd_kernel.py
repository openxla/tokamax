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

from __future__ import annotations

import dataclasses
import functools
from functools import partial
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member

from tokamax._src import jaxtyping
from tokamax._src.ops.kda.common import (
    chunk_gated_delta_rule_fwd_h,
    estimate_mini_batch,
    get_tpu_limits,
    kda_gate_chunk_cumsum,
)
from tokamax._src.ops.kda.cp_utils import (
    ContextParallelMetadata,
    _merge_dht,
    all_gather_into_tensor,
)
from tokamax._src.ops.kda.pallas_mosaic_tpu_types import KdaResiduals
from tokamax._src.ops.kda.utils import (
    _align_seqs,
    _unalign_output,
    align_up,
    cdiv,
    exp,
    exp2,
    get_interpret,
    l2norm_bwd,
    segment_ids_to_seqlens,
)


# =============================================================================
# Context-parallel backward pre-process
# =============================================================================


def _chunk_gated_delta_rule_bwd_dhu_pre_process_kernel(
  chunk_active_ref,   # SMEM [NT] int32 -- per-chunk activeness mask
  q_ref,              # HBM [H*NT*BT, K]
  k_ref,              # HBM [H*NT*BT, K]
  w_ref,              # HBM [H*NT*BT, K]
  do_ref,             # HBM [H*NT*BT, V]
  dv_ref,             # HBM [H*NT*BT, V]
  gk_ref,             # HBM [H*NT*BT, K]
  # outputs
  dS_ext_ref,         # HBM [H, K, V]
  dM_ref,             # HBM [H, K, K]
  # scratch: 6 double-buffered inputs (with MB batch dim)
  q_scratch_ref,      # VMEM [2, MB, BT, K]
  k_scratch_ref,      # VMEM [2, MB, BT, K]
  w_scratch_ref,      # VMEM [2, MB, BT, K]
  do_scratch_ref,     # VMEM [2, MB, BT, V]
  dv_scratch_ref,     # VMEM [2, MB, BT, V]
  gk_scratch_ref,     # VMEM [2, MB, BT, K]
  # scratch: accumulators (with MB batch dim)
  dh_acc_ref,         # VMEM [MB, K, V] fp32
  M_acc_ref,          # VMEM [MB, K, K] fp32
  # semaphores
  sems,               # DMA [8, 2]
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

  # ── Auto MB: maximise VMEM utilisation, capped at min(H, 16) ──
  q_elem = 2 if q.dtype == jnp.bfloat16 else 4
  gk_elem = 2 if gk.dtype == jnp.bfloat16 else 4
  do_elem = 2 if do.dtype == jnp.bfloat16 else 4
  per_head = (
    2 * BT * (3 * K * q_elem + K * gk_elem + 2 * V * do_elem)
    + (K * V + K * K) * 4
  )
  vmem_budget = 8 * 1024 * 1024
  MB = max(1, vmem_budget // per_head)
  MB = min(MB, H, 16)
  while H % MB != 0 and MB > 1:
    MB -= 1

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
  g: Float[Array, "H B T K"],
  a_log: Float[Array, "H"],
  *,
  dyg: Float[Array, "H B T K"],
  delta_time_bias: Float[Array, "H*K"] | None = None,
  lower_bound: float | None = None,
) -> tuple[
    Float[Array, "H B T K"],
    Float[Array, "H"],
    Float[Array, "H*K"] | None,
]:
  """Differentiates the KDA gate with respect to g, a_log, and delta_time_bias.

  The standard path differentiates `-exp(a_log) * softplus(g + bias)`; the
  lower-bound path differentiates the sigmoid form.
  """
  H, K = g.shape[0], g.shape[-1]

  g_f = g.astype(jnp.float32)
  dyg_f = dyg.astype(jnp.float32)

  # Apply bias if present
  if delta_time_bias is not None:
    g_f = g_f + delta_time_bias.reshape(H, 1, 1, K).astype(jnp.float32)

  if lower_bound is None:
    # Forward: yg = -exp(a_log) * softplus(g + bias)
    # softplus(x) = log(1 + exp(x)), d/dx softplus(x) = sigmoid(x)
    b_A = -jnp.exp(a_log.astype(jnp.float32))  # [H]
    b_yg = b_A.reshape(H, 1, 1, 1) * jax.nn.softplus(g_f)  # [H, B, T, K]
    dg_f = b_A.reshape(H, 1, 1, 1) * (dyg_f * jax.nn.sigmoid(g_f))  # [H, B, T, K]
    dA_per_elem = dyg_f * b_yg  # [H, B, T, K]
  else:
    # Forward: yg = lower_bound * sigmoid(exp(a_log) * g)
    b_A = jnp.exp(a_log.astype(jnp.float32))  # [H]
    b_inner = b_A.reshape(H, 1, 1, 1) * g_f  # [H, B, T, K]
    b_sig = jax.nn.sigmoid(b_inner)
    b_dsig = b_sig * (1.0 - b_sig)
    dg_f = dyg_f * (lower_bound * b_dsig) * b_A.reshape(H, 1, 1, 1)  # [H, B, T, K]
    dA_per_elem = dg_f * g_f  # [H, B, T, K]

  # dA: reduce over all dims except H (axis 0) → [H]
  reduce_axes = (1, 2, 3)
  dA = jnp.sum(dA_per_elem, axis=reduce_axes)
  dA = dA.astype(a_log.dtype)

  # Cast dg back to input dtype
  dg = dg_f.astype(g.dtype)

  # dbias: sum dg over B, T (axes 1, 2) → [H, K] → [H*K]
  if delta_time_bias is not None:
    dbias = jnp.sum(dg_f, axis=(1, 2)).reshape(-1)  # [H*K]
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

@partial(jax.jit, static_argnames=["dtype"])
def compute_m1_recompute(bq, bk, bv, bb, bA, bg, bh, dtype):
  """Recomputes (u, w, v_new, qg, kg) from a saved state."""
  precision = (
      None if dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
  )
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
  u_mat = u.astype(dtype)
  w_mat = w.astype(dtype)
  v_new = u_mat.astype(jnp.float32) - jnp.matmul(
      w_mat.astype(jnp.float32),
      bh.astype(jnp.float32),
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  qg = bq * g_exp
  kg = bk * jnp.exp2(bg[:, -1:, :] - bg)
  return u, w, v_new, qg, kg


def _recompute_w_u_fwd(q, k, v, beta, A, gk, chunk_size):
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
  g_exp = jnp.exp2(g_chunks)
  precision = (
      None if q.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
  )

  u = jnp.matmul(
      A_chunks,
      (v_chunks * beta_chunks).astype(jnp.float32),
      precision=precision,
      preferred_element_type=jnp.float32,
  ).astype(v.dtype)
  w = jnp.matmul(
      A_chunks,
      (k_chunks * beta_chunks * g_exp).astype(jnp.float32),
      precision=precision,
      preferred_element_type=jnp.float32,
  ).astype(k.dtype)
  qg = (q_chunks * g_exp).astype(q.dtype)
  kg = (k_chunks * jnp.exp2(g_chunks[:, -1:, :] - g_chunks)).astype(k.dtype)

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
    bdo, bdv, bvn, bv, bh, dh, bq, bk, bg, bb, bA, scale, precision, b_dvb=None
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

  g_exp = jnp.exp2(bg)
  g_exp_last = g_exp[:, BT - 1, :]
  b_dgk = (bh * dh).sum(axis=2) * g_exp_last
  dq_acc = dq_acc * g_exp
  dk_acc = dk_acc * jnp.exp2(bg[:, BT - 1:BT, :] - bg)

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
  m_lower = idx[:, None] > idx[None, :]
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

  u, w, v_new, qg, kg = compute_m1_recompute(q, k, v, beta, A, g, h, k_ref.dtype)

  w_ref[:] = w.astype(w_ref.dtype)
  qg_ref[:] = qg.astype(qg_ref.dtype)
  kg_ref[:] = kg.astype(kg_ref.dtype)
  v_new_ref[:] = v_new.astype(v_new_ref.dtype)


@partial(jax.jit, static_argnames=["chunk_size", "mini_batch"])
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

  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  assert A.shape[-1] == BT, (
      f"A.shape[-1]={A.shape[-1]} must equal chunk_size={BT}"
  )
  assert h.shape[2] == NT, f"h has NT={h.shape[2]}, expected {NT}"

  def _r(x, d):
    return x.reshape(total, BT, d)

  q_r = _r(q, K)
  k_r = _r(k, K)
  v_r = _r(v, V)
  g_r = _r(g, K)
  beta_r = beta.reshape(total, BT)
  A_r = A.reshape(-1, BT, BT)
  h_r = h.reshape(total, K, V)

  align_minor = get_tpu_limits().block_align_minor
  if mini_batch is None:
    elem_size = 2 if v.dtype == jnp.bfloat16 else 4
    in_bytes = (3 * BT * K + BT * V + BT + BT * BT + K * V) * elem_size  # beta: BT bytes (2D)
    out_bytes = (3 * BT * K + BT * V) * elem_size
    per_chunk = in_bytes + out_bytes
    vmem_budget = 8 * 1024 * 1024
    hw_mb = max(1, vmem_budget // per_chunk)
    MB = min(hw_mb, total)
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
    jax.ShapeDtypeStruct((total, BT, K), q.dtype),
    jax.ShapeDtypeStruct((total, BT, K), q.dtype),
    jax.ShapeDtypeStruct((total, BT, K), q.dtype),
    jax.ShapeDtypeStruct((total, BT, V), v.dtype),
  ]

  kernel = partial(
    _fused_recompute_w_u_vnew_from_h_kernel,
    BT=BT,
    K=K,
    V=V,
    MB=MB,
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
):
  BT = bq.shape[1]
  idx = jnp.arange(BT, dtype=jnp.int32)
  causal_mask = idx[:, None] >= idx[None, :]
  dAqk_full = jnp.where(causal_mask[None], dAqk, 0.0).astype(jnp.float32)
  dAkk_full = dAkk.astype(jnp.float32)

  if not per_channel_gate:
    # Scalar gate: skip the BC-block decomposition. Blocking is only needed to
    # keep `exp2` bounded for per-channel gates. With a scalar gate,
    # `gs[r] - gs[c] <= 0` under the causal mask, so the full [BT, BT] decay is
    # formed once and folded into the cotangents. This uses 4 matmuls with
    # contraction BT instead of 8 with contraction BC.
    gs = bg[:, :, 0]  # [MB, BT]
    row = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
    col = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
    _causal = (row >= col)[None]
    _diff = gs[:, :, None] - gs[:, None, :]  # [MB, BT, BT]
    decay = jnp.where(_causal, jnp.exp2(jnp.where(_causal, _diff, 0.0)), 0.0)

    dAqk_d = dAqk_full * decay
    dAkk_d = dAkk_full * decay
    _b1 = (((2,), (1,)), ((0,), (0,)))
    _b1t = (((1,), (1,)), ((0,), (0,)))
    _dot = lambda a, b, d: jax.lax.dot_general(
        a, b, d, preferred_element_type=jnp.float32, precision=precision
    )

    dq_intra = _dot(dAqk_d, bk, _b1)
    dk_row_pre = _dot(dAkk_d, bk, _b1)
    # beta indexes the row, matching the blocked path's `dAkk_diag * beta_b`.
    dk_col = _dot(dAqk_d, bq, _b1t) + _dot(dAkk_d * bb[:, :, None], bk, _b1t)

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
  # Re-center around block max to prevent exp2 overflow across rows.
  g_max = jnp.max(g_b, axis=2, keepdims=True)
  row_d = jnp.exp2(g_b - g_max)
  col_d = jnp.exp2(g_max - g_b)
  k_til = k_b * col_d
  q_hat = q_b * row_d
  k_hat = k_b * beta_b * row_d

  K = bq.shape[2]
  MB = bq.shape[0]
  NM = NC * MB
  _b1 = (((2,), (1,)), ((0,), (0,)))
  _b1t = (((1,), (1,)), ((0,), (0,)))
  def _f(x):
    return x.reshape(NM, x.shape[2], x.shape[3])

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
    g_ref_rp = jnp.stack([g_b[ii, :, 0:1, :] for (ii, _) in row_pairs])
    k_dec_rp = k_rp * jnp.exp2(g_ref_rp - g_rp)

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
    row_decay = jnp.exp2(g_b - g_b[:, :, 0:1, :])
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
    g_ref_cp = jnp.stack([g_b[ii, :, BC - 1:BC, :] for (ii, _) in col_pairs])

    row_d_cp = jnp.exp2(g_cp - g_ref_cp)
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
    col_decay = jnp.exp2(g_b[:, :, BC - 1:BC, :] - g_b)
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
def compute_reverse_cumsum_dg(dg_total):
  BT = dg_total.shape[1]
  idx = jnp.arange(BT, dtype=jnp.int32)
  cumsum_mask = (idx[:, None] <= idx[None, :]).astype(jnp.float32)
  return jax.lax.dot_general(
    cumsum_mask, dg_total, (((1,), (1,)), ((), ())),
    precision=jax.lax.Precision.HIGHEST,
  ).transpose(1, 0, 2)


@jax.jit
def compute_forward_cumsum_g(g_raw):
  BT = g_raw.shape[1]
  idx = jnp.arange(BT, dtype=jnp.int32)
  cumsum_mask = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
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
    # Inputs. All refs except the first are single VMEM blocks from
    # emit_pipeline; optional inputs absent from the pipeline arrive as None.
    chunk_seg_ids_smem_ref,  # [B, NT] i32  full array, SMEM scalar prefetch
    q_ref,  # [MB_QK, 1, 1, BT, K]  MB_QK = MB // gqa_repeats
    k_ref,  # [MB_QK, 1, 1, BT, K]
    v_ref,  # [MB, 1, 1, BT, V]
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
    do_ref,  # [MB, 1, 1, BT, V]
    dv0_ref,  # [MB, 1, 1, BT, V] or None
    dAqk_ref,  # [MB, 1, 1, BT, BT] or None
    dht_ref,  # [MB, 1, 1, K, V]  block of this chunk's segment
    rstd_q_ref,  # [MB_QK, 1, 1, 1, BT] or None
    rstd_k_ref,  # [MB_QK, 1, 1, 1, BT] or None
    # Outputs.
    dq_ref,  # [MB_QK, 1, 1, BT, K] dqkv_dtype (f32 by default)
    dk_ref,  # [MB_QK, 1, 1, BT, K] dqkv_dtype (f32 by default)
    dv_ref,  # [MB, 1, 1, BT, V] dqkv_dtype (f32 by default)
    db_ref,  # [MB, 1, 1, 1, BT] f32
    dg_ref,  # [MB, 1, 1, 1, BT] f32 scalar gate; else [MB, 1, 1, BT, K]
    dh0_ref,  # [MB, 1, 1, K, V] f32  block of this chunk's segment
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
):
  """Fuse Dhu, WY, intra backward, and reverse cumsum for one chunk tile."""
  head_group = pl.program_id(0)
  batch_idx = pl.program_id(1)
  rev_c = pl.program_id(2)
  chunk_id = NT - 1 - rev_c
  _, seq_idx, _, is_first_chunk, is_last_chunk = _chunk_segment_metadata(
      chunk_seg_ids_smem_ref,
      batch_idx,
      chunk_id,
      NT,
  )
  precision = (
      jax.lax.Precision.DEFAULT
      if q_ref.dtype == jnp.bfloat16
      else jax.lax.Precision.HIGHEST
  )

  @pl.when(is_last_chunk)
  def _():
    dh_scratch[:] = dht_ref[:, 0, 0, :].astype(dh_scratch.dtype)

  dh = dh_scratch[:].astype(jnp.float32)
  # Native GQA: q and k arrive at the key head count (MB // gqa_repeats heads
  # per tile) and are broadcast to the MB value heads in VMEM, so no repeated
  # copy of q and k is needed in HBM.
  bq = q_ref[:, 0, 0].astype(jnp.float32)
  bk = k_ref[:, 0, 0].astype(jnp.float32)
  if gqa_repeats > 1:
    bq = jnp.repeat(bq, gqa_repeats, axis=0)
    bk = jnp.repeat(bk, gqa_repeats, axis=0)
  bv = v_ref[:, 0, 0].astype(jnp.float32)
  # Transpose scalar gate from [.., 1, BT] back to [MB, BT, 1] and broadcast to K in VMEM.
  if g_ref.shape[-2] == 1:
    bg = g_ref[:, 0, 0, 0].astype(jnp.float32)[..., None]
  else:
    bg = g_ref[:, 0, 0].astype(jnp.float32)
  if v_new_ref is None:
    bg = compute_forward_cumsum_g(bg)
  bg_scalar = bg
  _gate_narrow = bg.shape[-1] != K
  if _gate_narrow:
    bg = jnp.broadcast_to(bg, bg.shape[:-1] + (K,))
  g_exp_last = jnp.exp2(bg[:, BT - 1, :])
  bb = beta_ref[:, 0, 0, 0].astype(jnp.float32)
  if native_state_layout:
    # A and h blocks are [1, 1, MB, ...] in the forward's [B, NC, H, ...] layout.
    bA = A_ref[0, 0].astype(jnp.float32)
    bh = h_ref[0, 0].astype(jnp.float32)
  else:
    bA = A_ref[:, 0, 0].astype(jnp.float32)
    bh = h_ref[:, 0, 0].astype(jnp.float32)
  bdo = do_ref[:, 0, 0].astype(jnp.float32)

  bkbg = None
  if qg_ref is not None and kg_ref is not None and w_ref is not None:
    bqg = qg_ref[:, 0, 0].astype(jnp.float32)
    bkg = kg_ref[:, 0, 0].astype(jnp.float32)
    bw = w_ref[:, 0, 0].astype(jnp.float32)
  else:
    g_exp = jnp.exp2(bg)
    bqg = (bq * g_exp).astype(q_ref.dtype).astype(jnp.float32)
    bkg = (
        (bk * jnp.exp2(bg[:, BT - 1 : BT, :] - bg))
        .astype(k_ref.dtype)
        .astype(jnp.float32)
    )
    bkbg = bk * bb[:, :, None] * g_exp  # k * beta * g, f32
    if v_new_ref is None:
      # w-free path: w = A @ k_beta_g is never formed.
      bw = None
    else:
      bw = (
          jnp.matmul(
              bA,
              bkbg,
              precision=precision,
              preferred_element_type=jnp.float32,
          )
          .astype(k_ref.dtype)
          .astype(jnp.float32)
      )

  bdv0 = None
  bdAqk = None
  if v_new_ref is not None:
    bvn = v_new_ref[:, 0, 0].astype(jnp.float32)
    bdv0 = dv0_ref[:, 0, 0].astype(jnp.float32)
    bdAqk = dAqk_ref[:, 0, 0].astype(jnp.float32)
  elif bw is None:
    assert bkbg is not None
    # Two-matmul v_new = A @ (v * beta - k_beta_g @ h); no u, no w.
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
  else:
    bu = (
        jnp.matmul(
            bA,
            (bv * bb[:, :, None]).astype(jnp.float32),
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        .astype(v_ref.dtype)
        .astype(jnp.float32)
    )
    bvn = bu - jnp.matmul(
        bw,
        bh,
        precision=precision,
        preferred_element_type=jnp.float32,
    )
  if v_new_ref is None:
    idx_bt = jnp.arange(BT, dtype=jnp.int32)
    m_causal = idx_bt[:, None] >= idx_bt[None, :]
    diff_g = bg_scalar - jnp.swapaxes(bg_scalar, 1, 2)
    safe_diff = jnp.where(m_causal[None, :, :], diff_g, 0.0)
    decay_mat = jnp.where(m_causal[None, :, :], jnp.exp2(safe_diff), 0.0)
    qk_dot = jax.lax.dot_general(
        bq,
        bk,
        (((2,), (2,)), ((0,), (0,))),
        precision=precision,
        preferred_element_type=jnp.float32,
    )
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
    assert bkbg is not None
    # w-free chain: w^T dv = k_beta_g^T (A^T dv) = k_beta_g^T b_dvb.
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
  )
  if _gate_narrow:
    # Scalar gate: all K channels share one gate, so sum the gate gradient over
    # K first (exact by linearity) and run the reverse cumsum on [MB, BT]
    # instead of [MB, BT, K].
    dg_scalar = jnp.sum(dg_total, axis=-1)  # [MB, BT]
    idx = jnp.arange(BT, dtype=jnp.int32)
    # _rev_mask[s, t] = 1 where s >= t, so dg_scalar @ _rev_mask is the
    # reverse cumsum. The trailing axis restores the [MB, BT, 1] gate shape.
    _rev_mask = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    dg_reverse_cumsum = jnp.matmul(
        dg_scalar,
        _rev_mask,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )[..., None]
  else:
    dg_reverse_cumsum = compute_reverse_cumsum_dg(dg_total)

  # Sum value heads back to key heads in VMEM before writing dq and dk.
  if gqa_repeats > 1:
    MB_QK = dq_ref.shape[0]
    _red = lambda x: x.reshape(MB_QK, gqa_repeats, *x.shape[1:]).sum(axis=1)
    dq_total = _red(dq_total)
    dk_total = _red(dk_total)
  if rstd_q_ref is not None and rstd_k_ref is not None:
    bq_norm = q_ref[:, 0, 0].astype(jnp.float32)
    bk_norm = k_ref[:, 0, 0].astype(jnp.float32)
    brstd_q = rstd_q_ref[:, 0, 0, 0].astype(jnp.float32)[..., None]
    brstd_k = rstd_k_ref[:, 0, 0, 0].astype(jnp.float32)[..., None]
    dq_total = brstd_q * (
        dq_total - bq_norm * jnp.sum(bq_norm * dq_total, axis=-1, keepdims=True)
    )
    dk_total = brstd_k * (
        dk_total - bk_norm * jnp.sum(bk_norm * dk_total, axis=-1, keepdims=True)
    )
  dq_ref[:, 0, 0] = dq_total.astype(dq_ref.dtype)
  dk_ref[:, 0, 0] = dk_total.astype(dk_ref.dtype)
  dv_ref[:, 0, 0] = (b_dvb * bb[:, :, None]).astype(dv_ref.dtype)
  db_ref[:, 0, 0, 0] = db_total.astype(db_ref.dtype)
  if dg_ref.shape[-2] == 1:
    dg_ref[:, 0, 0, 0] = dg_reverse_cumsum[..., 0].astype(dg_ref.dtype)
  else:
    dg_ref[:, 0, 0] = dg_reverse_cumsum.astype(dg_ref.dtype)

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
    ],
)
@jaxtyping.jaxtyped
def _fused_dhu_wy_intra_cumsum_pallas_jit(
    # `H_QK` is `H // gqa_repeats`, and equals H when gqa_repeats is 1.
    q: Float[Array, "H_QK B T K"],
    k: Float[Array, "H_QK B T K"],
    v: Float[Array, "H B T V"],
    v_new: Float[Array, "H B T V"] | None,
    qg: Float[Array, "H B T K"] | None,
    kg: Float[Array, "H B T K"] | None,
    w: Float[Array, "H B T K"] | None,
    g: Float[Array, "H B T GW"],
    beta: Float[Array, "H B T"],
    # [H, B, T, BT] and [H, B, NT, K, V], or with native_state_layout=True the
    # forward's own [B, NT, H, BT, BT] and [B, NT, H, K, V] (no transpose).
    A: Float[Array, "..."],
    h: Float[Array, "..."],
    do: Float[Array, "H B T V"],
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
) -> tuple[
    Float[Array, "H_QK B T K"],
    Float[Array, "H_QK B T K"],
    Float[Array, "H B T V"],
    Float[Array, "H B T"],
    Float[Array, "H B T GW"],
    Float[Array, "B N_OUT H K V"] | None,
]:
  """Fuses Dhu recurrence, WY backward, intra backward, and gate cumsum.

  `segment_ids` applies per-batch varlen boundaries, and `return_dh0`
  controls whether the initial-state gradient is materialized.
  """
  # `q` and `k` carry the key head count; everything else carries the value
  # head count. H below is always the value head count.
  H_QK, B, T, K = q.shape
  H = H_QK * gqa_repeats
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT
  HB = H * B

  is_varlen = segment_ids is not None
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  assert A.shape[-1] == BT, (
      f"A.shape[-1]={A.shape[-1]} must equal chunk_size={BT}"
  )
  if native_state_layout:
    assert A.shape == (B, NT, H, BT, BT), f"A.shape={A.shape}"
    assert h.shape == (B, NT, H, K, V), f"h.shape={h.shape}"
  else:
    assert h.shape[2] == NT, f"h has NT={h.shape[2]}, expected {NT}"
  assert use_exp2 is True, (
    "fused_dhu_wy_intra_cumsum_pallas currently expects log2 gates and use_exp2=True"
  )

  segment_ids = jnp.ones((B, T), dtype=jnp.int32) if not is_varlen else segment_ids
  chunk_seg_ids = segment_ids.reshape(B, NT, BT)[:, :, 0]  # [B, NT]
  if dht is not None:
    N = dht.shape[1]  # dht is [B, N, H, K, V], N is dim 1
  elif is_varlen:
    N = max_num_segments
  else:
    N = 1  # uniform: one sequence per batch element
  if dht is not None:
    assert dht.shape[1] == N, f"dht has N={dht.shape[1]}, expected {N}"
  dht_arr = dht if dht is not None else jnp.zeros((B, N, H, K, V), dtype=jnp.float32)

  has_qg_kg_w = qg is not None and kg is not None and w is not None
  has_vn_dv0_dAqk = v_new is not None and dv0 is not None and dAqk is not None
  has_rstd_qk = rstd_q is not None and rstd_k is not None
  if mini_batch is None:
    elem_size = 2 if q.dtype == jnp.bfloat16 else 4
    num_k_io = 8 if has_qg_kg_w else 5
    io_per_head = (
        num_k_io * BT * K + 4 * BT * V + BT + 3 * BT * BT + 2 * K * V
    ) * elem_size + (K * V + 5 * BT * K + BT * V + BT + 2 * BT * BT + K * V) * 4
    per_head = io_per_head + io_per_head * 3 // 2
    hw = get_tpu_limits()
    vmem_budget = hw.vmem_limit_bytes
    # Scale down VMEM budget for chunk sizes > 64 (override via KDA_BWD_VMEM_FRACTION).
    import os as _os  # pylint: disable=g-import-not-at-top

    _frac = float(_os.environ.get("KDA_BWD_VMEM_FRACTION", "0") or 0)
    if _frac > 0:
      vmem_budget = int(vmem_budget * _frac)
    elif BT > 64:
      vmem_budget = (vmem_budget * 64) // BT
    MB = max(1, vmem_budget // per_head)
    MB = min(MB, H, 16)
    while H % MB != 0 and MB > 1:
      MB -= 1
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  # Keep [H, B, ...] layout; reshape T → (NT, BT) only. No transpose.
  # B is an independent dimension handled by a separate grid axis.
  q_r = q.reshape(H_QK, B, NT, BT, K)
  k_r = k.reshape(H_QK, B, NT, BT, K)
  rstd_q_r = rstd_q.reshape(H_QK, B, NT, 1, BT) if has_rstd_qk else None
  rstd_k_r = rstd_k.reshape(H_QK, B, NT, 1, BT) if has_rstd_qk else None
  v_r = v.reshape(H, B, NT, BT, V)
  vn_r = v_new.reshape(H, B, NT, BT, V) if has_vn_dv0_dAqk else None
  qg_r = qg.reshape(H, B, NT, BT, K) if has_qg_kg_w else None
  kg_r = kg.reshape(H, B, NT, BT, K) if has_qg_kg_w else None
  w_r = w.reshape(H, B, NT, BT, K) if has_qg_kg_w else None
  GW = g.shape[-1]
  scalar_gate = GW == 1
  # Store width-1 axes as [.., 1, BT] to avoid 128-lane minor-axis padding in HBM.
  g_r = (
      g.reshape(H, B, NT, 1, BT) if scalar_gate else g.reshape(H, B, NT, BT, GW)
  )
  beta_r = beta.reshape(H, B, NT, 1, BT)
  if native_state_layout:
    A_r = A  # [B, NT, H, BT, BT], read in place
    h_r = h  # [B, NT, H, K, V], read in place
  else:
    A_r = A.reshape(H, B, NT, BT, BT)
    h_r = h  # already [H, B, NT, K, V]
  do_r = do.reshape(H, B, NT, BT, V)
  dv0_r = dv0.reshape(H, B, NT, BT, V) if has_vn_dv0_dAqk else None
  dAqk_r = dAqk.reshape(H, B, NT, BT, BT) if has_vn_dv0_dAqk else None

  def idx_chunk(head_group, batch, chunk, chunk_seg_ids_smem_ref):
    return (head_group, batch, NT - 1 - chunk, 0, 0)

  def idx_state(head_group, batch, chunk, chunk_seg_ids_smem_ref):
    chunk_id = NT - 1 - chunk
    _, seq_idx, _, _, _ = _chunk_segment_metadata(
        chunk_seg_ids_smem_ref, batch, chunk_id, NT
    )
    return (head_group, batch, seq_idx, 0, 0)

  # dht_arr [B, N, H, K, V] → [H, B, N, K, V]
  dht_arr = dht_arr.transpose(2, 0, 1, 3, 4)

  # q/k blocks carry MB // gqa_repeats key heads for the MB value heads of the
  # tile. MB divides H and gqa_repeats divides MB, so this is exact.
  MB_QK = MB // gqa_repeats
  assert (
      MB % gqa_repeats == 0
  ), f"MB={MB} must be divisible by gqa_repeats={gqa_repeats}"
  qk_spec = pl.BlockSpec((MB_QK, 1, 1, BT, K), index_map=idx_chunk)
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
  v_spec = pl.BlockSpec((MB, 1, 1, BT, V), index_map=idx_chunk)
  vn_dv0_spec = v_spec if has_vn_dv0_dAqk else None
  g_spec = (
      pl.BlockSpec((MB, 1, 1, 1, BT), index_map=idx_chunk)
      if scalar_gate
      else pl.BlockSpec((MB, 1, 1, BT, GW), index_map=idx_chunk)
  )
  b_spec = pl.BlockSpec((MB, 1, 1, 1, BT), index_map=idx_chunk)
  A_spec = pl.BlockSpec((MB, 1, 1, BT, BT), index_map=idx_chunk)
  dAqk_spec = A_spec if has_vn_dv0_dAqk else None
  h_spec = pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_chunk)
  if native_state_layout:

    def idx_chunk_bnh(head_group, batch, chunk, chunk_seg_ids_smem_ref):
      return (batch, NT - 1 - chunk, head_group, 0, 0)

    A_spec = pl.BlockSpec((1, 1, MB, BT, BT), index_map=idx_chunk_bnh)
    h_spec = pl.BlockSpec((1, 1, MB, K, V), index_map=idx_chunk_bnh)
  state_spec = pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_state)

  kernel = partial(
      _fused_dhu_wy_intra_cumsum_kernel,
      scale=scale,
      per_channel_gate=per_channel_gate,
      gqa_repeats=gqa_repeats,
      native_state_layout=native_state_layout,
      BT=BT,
      K=K,
      V=V,
      NT=NT,
      MB=MB,
  )
  dh_tmp = pltpu.VMEM((MB, K, V), jnp.float32)
  # dq/dk/dv may be written in the activation dtype (e.g. bf16), which halves
  # their HBM writes. db/dg stay f32 for the gate sums.
  dqkv_out_dtype = jnp.float32 if dqkv_dtype is None else dqkv_dtype
  out_shape = [
      jax.ShapeDtypeStruct((H_QK, B, NT, BT, K), dqkv_out_dtype),
      jax.ShapeDtypeStruct((H_QK, B, NT, BT, K), dqkv_out_dtype),
      jax.ShapeDtypeStruct((H, B, NT, BT, V), dqkv_out_dtype),
      jax.ShapeDtypeStruct((H, B, NT, 1, BT), jnp.float32),
      jax.ShapeDtypeStruct((H, B, NT, 1, BT), jnp.float32)
      if scalar_gate
      else jax.ShapeDtypeStruct((H, B, NT, BT, GW), jnp.float32),
      jax.ShapeDtypeStruct((H, B, N, K, V), jnp.float32),
  ]

  in_specs = [
      qk_spec,
      qk_spec,
      v_spec,
      vn_dv0_spec,
      qgw_spec,
      qgw_spec,
      qgw_spec,
      g_spec,
      b_spec,
      A_spec,
      h_spec,
      v_spec,
      vn_dv0_spec,
      dAqk_spec,
      state_spec,
      rstd_spec,
      rstd_spec,
  ]
  # dq and dk are written at the key head count H_QK.
  out_specs = [qk_spec, qk_spec, v_spec, b_spec, g_spec, state_spec]
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
  ]
  # Optional inputs are None. They are left out of the pipeline and passed to
  # the kernel body as None.
  present = [x is not None for x in inputs]
  in_specs = [s for s, p in zip(in_specs, present) if p]
  inputs = [x for x in inputs if x is not None]
  num_in = len(inputs)
  num_out = len(out_specs)

  def bind_seg_ids(spec, seg_ids_smem_ref):
    # emit_pipeline index maps only receive the grid indices.
    index_map = spec.index_map
    return pl.BlockSpec(
        spec.block_shape, lambda *idx: index_map(*idx, seg_ids_smem_ref)
    )

  # Ref order: SMEM seg ids, HBM inputs, HBM outputs, VMEM scratch.
  def pipeline_kernel(seg_ids_smem_ref, *refs):
    in_hbm_refs = refs[:num_in]
    out_hbm_refs = refs[num_in : num_in + num_out]
    scratch_vmem_refs = refs[num_in + num_out :]

    def body(*block_vmem_refs):
      blocks = iter(block_vmem_refs[:num_in])
      kernel_inputs = [next(blocks) if p else None for p in present]
      kernel(seg_ids_smem_ref, *kernel_inputs, *block_vmem_refs[num_in:])

    pltpu.emit_pipeline(
        body,
        grid=(H // MB, B, NT),
        in_specs=[bind_seg_ids(s, seg_ids_smem_ref) for s in in_specs],
        out_specs=[bind_seg_ids(s, seg_ids_smem_ref) for s in out_specs],
    )(*in_hbm_refs, *out_hbm_refs, scratches=tuple(scratch_vmem_refs))

  # Operands stay in HBM. Inside a single kernel invocation, emit_pipeline
  # walks the (head group, batch, reversed chunk) grid and pipelines the block
  # copies. Chunks run in order: `dh_tmp` carries dh from one chunk to the
  # previous one.
  hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
  dq_r, dk_r, dv_r, db_r, dg_r, dh0_r = pl.pallas_call(
      pipeline_kernel,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(),
          in_specs=[hbm_spec] * num_in,
          out_specs=[hbm_spec] * num_out,
          scratch_shapes=[dh_tmp],
      ),
      compiler_params=pltpu.CompilerParams(
          disable_bounds_checks=True,
          vmem_limit_bytes=get_tpu_limits().vmem_limit_bytes,
      ),
      interpret=get_interpret(),
      name="_fused_dhu_wy_intra_cumsum_pallas_jit",
  )(chunk_seg_ids, *inputs)

  dh0_out = dh0_r.transpose(1, 2, 0, 3, 4) if return_dh0 else None
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
):
  bv = v_ref[:]  # [MB, BT, V]
  bA = A_ref[:]  # [MB, BT, BT]
  bdo = do_ref[:].astype(jnp.float32)  # [MB, BT, V], cast in VMEM
  precision = (
      None if A_ref.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
  )

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
    elem_size = 2 if v.dtype == jnp.bfloat16 else 4
    in_bytes = (2 * BT * V + BT * BT) * elem_size
    out_bytes = (BT * BT * 4 + BT * V * 2)
    per_chunk = in_bytes + out_bytes
    # Pin MB=1 for BT > 64. `per_chunk` above counts only the pipelined I/O
    # blocks, not the in-kernel f32 [MB, BT, BT] intermediates (masked A and
    # the dA accumulator), which grow quadratically with BT.
    max_mb = 1 if BT > 64 else 32
    MB = estimate_mini_batch(per_chunk, total, max_mb=max_mb)
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
      vmem_limit_bytes=get_tpu_limits().vmem_limit_bytes,
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
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  """Computes VJP of depthwise causal conv1d + SiLU, supporting GQA head repeats."""
  H, B, T, D = x_raw.shape
  H_w, W, _ = w.shape
  repeats = H // H_w

  def _fwd_fn(x_in, w_in, b_in):
    x_f32 = x_in.astype(jnp.float32)
    w_f32 = w_in.astype(jnp.float32)
    acc = x_f32 * w_f32[:, None, W - 1 : W, :]
    for lag in range(1, W):
      x_lag = jnp.pad(x_f32[:, :, :-lag, :], ((0, 0), (0, 0), (lag, 0), (0, 0)))
      if aligned_segment_ids is not None:
        same_seg = (
            aligned_segment_ids[:, lag:] == aligned_segment_ids[:, :-lag]
        ) & (aligned_segment_ids[:, lag:] >= 0)
        same_seg = jnp.pad(same_seg, ((0, 0), (lag, 0)))
        x_lag = jnp.where(same_seg[None, :, :, None], x_lag, 0.0)
      acc = acc + x_lag * w_f32[:, None, W - 1 - lag : W - lag, :]
    if b_in is not None:
      acc = acc + b_in.astype(jnp.float32)[:, None, None, :]
    return jax.nn.silu(acc).astype(x_in.dtype)

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
) -> tuple[
    Float[Array, "H B T_ORIG K"],
    Float[Array, "H B T_ORIG K"],
    Float[Array, "H B T_ORIG V"],
    Float[Array, "H B T_ORIG GW"],
    Float[Array, "H B T_ORIG"],
    Float[Array, "H"] | None,
    Float[Array, "H*K"] | None,
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
    T_orig = do.shape[2]
    [do], [], _, _ = _align_seqs(
        [do],
        [],
        cu_seqlens,
        align=chunk_size,
        aligned_cu_seqlens=aligned_cu,
    )
    cu_seqlens_bwd = aligned_cu

  if (
      context_parallel_metadata is not None
      and context_parallel_metadata.is_cp_enabled
  ):
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
  if (cu_seqlens is None) and (segment_ids is not None):
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
  # cu_seqlens was already derived upstream (e.g. from forward residuals).
  if (
      segment_ids is not None
      and cu_seqlens is None
      and max_num_segments is None
  ):
    raise ValueError(
        "`max_num_segments` is required when segment metadata is unavailable."
    )
  if do.shape[2] != T:
    raise ValueError(f"aligned do has T={do.shape[2]}, expected {T}")
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
  _cp_active = (
      context_parallel_metadata is not None
      and context_parallel_metadata.is_cp_enabled
  )

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
      )
      g = g_cumsum  # already [H,B,T,K]

    if h is None:
      raise ValueError("saved h is required when recompute is disabled")
    if h.shape[2] != NT:
      raise ValueError(f"saved h has NT={h.shape[2]}, expected {NT}")

    assert g is not None
    if v_new is not None:
      # v_new was saved in forward residuals; w, qg, kg are computed on-the-fly
      # in VMEM inside _fused_dhu_wy_intra_cumsum_kernel.
      w, qg, kg = None, None, None
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
    )
    if kg is None:
      raise RuntimeError("KDA recompute did not produce gated keys.")

    # Pass scalar gates via `g` with plain keys `k`; pass per-channel gates via `gk` with gated keys `kg`.
    assert g is not None
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
    if cu_seqlens is not None and h.shape[2] < NT:
      h = jnp.pad(h, ((0, 0), (0, 0), (0, NT - h.shape[2]), (0, 0), (0, 0)))

  assert v_new is not None
  # ---- Stage 1: dAqk and initial dv ----
  dAqk, dv = chunk_kda_bwd_dAv_kernel(
    q=q,
    k=k,
    v=v_new,
    do=do,
    A=Aqk,
    scale=scale,
    chunk_size=chunk_size,
  )

  if _cp_active:
    assert context_parallel_metadata is not None
    if segment_ids is None:
      raise ValueError("backward CP requires rank-local segment_ids")
    if context_parallel_metadata.post_num_ranks is None:
      raise ValueError("backward CP requires post_num_ranks")
    if context_parallel_metadata.is_last_rank is None:
      raise ValueError("backward CP requires is_last_rank")
    # pre_process requires segment_ids length == aligned T (q.shape[2]).
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
  if dht is not None and dht.ndim == 4 and segment_ids is None:
    dht = dht[:, None, :, :, :]
  dht_m4 = dht
  import os as _os  # pylint: disable=g-import-not-at-top

  _l2norm_in_kernel = _os.environ.get("KDA_L2NORM_BWD_IN_KERNEL", "1") == "1"
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
      segment_ids=segment_ids,
      chunk_size=chunk_size,
      use_exp2=True,
      return_dh0=initial_state is not None,
      max_num_segments=max_num_segments,
      rstd_q=rstd_q if (use_qk_l2norm and _l2norm_in_kernel) else None,
      rstd_k=rstd_k if (use_qk_l2norm and _l2norm_in_kernel) else None,
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
    )

  # Non-varlen: squeeze dh0 5D [B,1,H,K,V] → 4D [B,H,K,V]
  # (varlen callers pass 5D, so dh0 stays 5D)
  _is_varlen = cu_seqlens is not None or (segment_ids is not None)
  if dh0 is not None and dh0.ndim == 5 and not _is_varlen:
    dh0 = dh0[:, 0]

  cu_seqlens = original_cu_seqlens
  if use_qk_l2norm and not _l2norm_in_kernel:
    assert rstd_q is not None and rstd_k is not None
    dq = l2norm_bwd(q, rstd_q, dq)
    dk = l2norm_bwd(k, rstd_k, dk)

  dw_q = dw_k = dw_v = None
  db_q = db_k = db_v = None
  if residuals.q_raw is not None:
    assert residuals.k_raw is not None and residuals.v_raw is not None
    assert (
        residuals.conv_weight_q is not None
        and residuals.conv_weight_k is not None
        and residuals.conv_weight_v is not None
    )
    dq, dw_q, db_q = _conv1d_silu_bwd(
        residuals.q_raw,
        residuals.conv_weight_q,
        residuals.conv_bias_q,
        dq,
        segment_ids_aligned,
    )
    dk, dw_k, db_k = _conv1d_silu_bwd(
        residuals.k_raw,
        residuals.conv_weight_k,
        residuals.conv_bias_k,
        dk,
        segment_ids_aligned,
    )
    dv, dw_v, db_v = _conv1d_silu_bwd(
        residuals.v_raw,
        residuals.conv_weight_v,
        residuals.conv_bias_v,
        dv,
        segment_ids_aligned,
    )

  if cu_seqlens is not None:
    dq = _unalign_output(dq, cu_seqlens, aligned_cu, T_orig)
    dk = _unalign_output(dk, cu_seqlens, aligned_cu, T_orig)
    dv = _unalign_output(dv, cu_seqlens, aligned_cu, T_orig)
    dg = _unalign_output(dg, cu_seqlens, aligned_cu, T_orig)
    db = _unalign_output(db, cu_seqlens, aligned_cu, T_orig)

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
