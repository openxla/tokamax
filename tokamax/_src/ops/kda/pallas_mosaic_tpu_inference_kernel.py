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
"""Inference-only native segment-ID KDA forward Pallas kernel.

This module is intentionally separate from chunk_fwd_mega.py so inference
optimizations cannot change the training, recompute, or context-parallel paths.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tokamax._src.ops.kda.utils import get_interpret
from tokamax._src.ops.kda.utils import align_up
from tokamax._src.ops.kda.common import estimate_mini_batch, get_tpu_limits

_RCP_LN2 = 1.0 / math.log(2)

CHUNK_KIND_ALL_PAD = 0
CHUNK_KIND_FULL_IN_SEGMENT = 1
CHUNK_KIND_BOUNDARY = 2
CHUNK_KIND_PARTIAL_PAD = 3
CHUNK_FLAG_LAST_REAL = 4
_MAX_SEGS_PER_CHUNK = 32


def _build_chunk_metadata(segment_ids, chunk_size):
  """Build per-chunk metadata from segment IDs with zero denoting padding."""
  orig_ndim = segment_ids.ndim
  if orig_ndim == 1:
    segment_ids = segment_ids[None, :]
  batch, tokens = segment_ids.shape
  block_tokens = int(chunk_size)
  num_blocks = tokens // block_tokens
  seg = segment_ids.astype(jnp.int32).reshape(batch, num_blocks, block_tokens)
  seg_first = seg[:, :, 0]
  valid_mask = seg > 0
  pos = jnp.arange(block_tokens, dtype=jnp.int32)
  last_valid_idx = jnp.where(valid_mask, pos[None, None, :], -1).max(axis=2)
  has_any = last_valid_idx >= 0
  safe_idx = jnp.where(has_any, last_valid_idx, 0)
  seg_last = jnp.where(
      has_any,
      jnp.take_along_axis(seg, safe_idx[:, :, None], axis=2)[:, :, 0],
      0,
  )
  seg_prev = jnp.concatenate(
      [
          jnp.zeros((batch, num_blocks, 1), dtype=jnp.int32),
          seg[:, :, :-1],
      ],
      axis=2,
  )
  is_new = (seg != seg_prev) & (seg > 0)
  distinct = is_new.sum(axis=2).astype(jnp.int32)
  has_pad = (seg == 0).any(axis=2)
  chunk_kind = jnp.where(
      distinct == 0,
      jnp.int32(CHUNK_KIND_ALL_PAD),
      jnp.where(
          distinct >= 2,
          jnp.int32(CHUNK_KIND_BOUNDARY),
          jnp.where(
              has_pad,
              jnp.int32(CHUNK_KIND_PARTIAL_PAD),
              jnp.int32(CHUNK_KIND_FULL_IN_SEGMENT),
          ),
      ),
  )
  seg_id = jnp.where(
      chunk_kind == CHUNK_KIND_FULL_IN_SEGMENT, seg_first, 0
  ).astype(jnp.int32)
  if orig_ndim == 1:
    return chunk_kind[0], seg_first[0], seg_last[0], seg_id[0]
  return chunk_kind, seg_first, seg_last, seg_id


# =====================================================================
# Native segment_ids mega kernel -- avoids _align_seqs gather/scatter
# =====================================================================


def _fwd_mega_kernel_native_segids(
    # Scalar-prefetch metadata
    chunk_kind_meta_ref,  # [B, NT_META_PAD]
    seg_first_meta_ref,  # [B, NT_META_PAD]
    seg_last_meta_ref,  # [B, NT_META_PAD]
    # Prefetch: segment_ids
    seg_ids_ref,  # [B, 128] (128-aligned block)
    # Inputs
    q_ref,  # [MB, 1, BT, K_PAD]
    k_ref,  # [MB, 1, BT, K_PAD]
    v_ref,  # [MB, 1, BT, V_ALIGNED]
    g_ref,  # [MB, 1, BT, K_PAD]
    beta_ref,  # [MB, 1, 1, 1, BT]
    h0_ref,  # auto window or full HBM ref for manual DMA
    A_scale_ref,  # [MB, 1, 1, 1] or None
    dt_bias_ref,  # [MB, 1, 1, K_PAD] or None
    # Outputs
    o_ref,  # [MB, 1, BT, V_ALIGNED]
    ht_ref,  # all segments or streaming [2, MB, 1, K_PAD, V]
    Aqk_ref,  # [MB, 1, BT, BT]
    Akk_ref,  # [MB, 1, BT, BT]
    g_cumsum_out_ref,  # [MB, 1, BT, K_PAD]
    chunk_h_out_ref,  # [MB, 1, 1, K_PAD, V_ALIGNED] per-chunk h
    # Scratch (VMEM persists across the NT loop for the same head and batch)
    scratch_ref,  # [MB, K_PAD, V_ALIGNED] KV state
    prev_seg_ref,  # [1] previous segment ID
    final_state_dma_ref,  # [MB, K_PAD, V_ALIGNED] state write staging
    q_dma_ref,  # [1, BT, MB, K_PAD] direct-offset staging
    k_dma_ref,  # [1, BT, MB, K_PAD]
    v_dma_ref,  # [1, BT, MB, V_PAD]
    g_dma_ref,  # [1, BT, MB, K_PAD]
    beta_dma_ref,  # [1, BT, 1, 128]
    input_dma_sems,  # [5] DMA semaphores
    state_dma_sem,  # DMA semaphore for conditional state transfers
    *,
    NT,
    BT,
    N_max,
    scale,
    cumsum_scale,
    MB,
    K_PAD,
    V_PAD,
    OUTPUT_PRECISION,
    safe_gate,
    NORMALIZE_QK,
    use_gate_in_kernel,
    lower_bound,
    USE_NEUMANN,
    QK_BC,
    INV_BC,
    SKIP_STAGE4_MASK,
    PACK_HEAD_INV,
    CLIP_BETA_IN_KERNEL,
    PACKED_METADATA,
    HAS_H0,
    STORE_RESIDUALS,
    STORE_H,
    STORE_FINAL_STATE,
    BATCH_FIRST,
    MANUAL_H0_DMA,
    MANUAL_HT_DMA,
    OVERLAP_H0_DMA,
    OVERLAP_HT_DMA,
    RESIDUAL_CHUNK_LAYOUT,
    SEGMENT_LOCAL,
    DIRECT_INPUT_DMA,
):
  """Native segment_ids mega kernel body."""
  i_b = pl.program_id(1)
  i_c = pl.program_id(2)
  chain_start = i_c == 0
  t0 = i_c * BT

  def _load_token_segment_ids():
    pass
    seg_full = seg_ids_ref[i_b, :].astype(jnp.int32)  # [128]
    if BT == 128:
      return seg_full
    seg_first_half = seg_full[:BT]
    seg_second_half = seg_full[BT : 2 * BT]
    use_second = ((i_c * BT) % 128) >= BT
    return jnp.where(use_second, seg_second_half, seg_first_half)

  if PACKED_METADATA:
    packed_metadata = chunk_kind_meta_ref[i_b, i_c]
    encoded_kind = packed_metadata & jnp.int32(0x7)
    first_seg = (packed_metadata >> jnp.int32(3)) & jnp.int32(0x3FFF)
    last_seg = packed_metadata >> jnp.int32(17)
  else:
    encoded_kind = chunk_kind_meta_ref[i_b, i_c]
    first_seg = seg_first_meta_ref[i_b, i_c]
    last_seg = seg_last_meta_ref[i_b, i_c]
  kind = encoded_kind & jnp.int32(0x3)
  is_last_real = (encoded_kind & jnp.int32(CHUNK_FLAG_LAST_REAL)) != 0

  pass

  def _dma_state_to_ht(src_ref, segment_id):
    h_start = pl.program_id(0) * MB
    cp = pltpu.make_async_copy(
        src_ref,
        ht_ref.at[
            segment_id - 1,
            pl.ds(h_start, MB),
            i_b,
            pl.ds(None),
            pl.ds(None),
        ],
        state_dma_sem,
    )
    cp.start()
    cp.wait()

  def _read_h0(segment_id):
    return h0_ref[segment_id - 1, :, 0, :, :].astype(jnp.float32)

  # ALL_PAD ----------------------------------------------------------
  @pl.when(kind == CHUNK_KIND_ALL_PAD)
  def _all_pad():
    # Segment-local uses a rectangular grid. Idle programs may map to a
    # chunk owned by another segment, so they must not write that block.
    o_ref[0] = jnp.zeros([BT, MB, V_PAD], dtype=o_ref.dtype)
    pass

  # FULL_IN_SEGMENT fast path ---------------------------------------
  @pl.when(kind == CHUNK_KIND_FULL_IN_SEGMENT)
  def _full():
    prev_seg = prev_seg_ref[0]
    seg_changed = (first_seg != prev_seg) | chain_start

    if STORE_FINAL_STATE:

      @pl.when(seg_changed & (~chain_start))
      def _save_prev():
        if MANUAL_HT_DMA:
          _dma_state_to_ht(scratch_ref, prev_seg)
        else:
          ht_ref[prev_seg - 1, ...] = scratch_ref[...].astype(ht_ref.dtype)[
              :, None, :, :
          ]

    @pl.when(seg_changed)
    def _init():
      scratch_ref[...] = jnp.zeros([MB, K_PAD, V_PAD], dtype=jnp.float32)

    if HAS_H0:

      @pl.when(seg_changed)
      def _load_h0():
        if MANUAL_H0_DMA:
          h_start = pl.program_id(0) * MB
          cp = pltpu.make_async_copy(
              h0_ref.at[
                  first_seg - 1,
                  pl.ds(h_start, MB),
                  i_b,
                  pl.ds(None),
                  pl.ds(None),
              ],
              scratch_ref,
              state_dma_sem,
          )
          cp.start()
          cp.wait()
        else:
          scratch_ref[...] = _read_h0(first_seg)

    # --- Stage 1: Gate cumsum ---
    q = q_ref[0].transpose(1, 0, 2).astype(jnp.float32)
    k = k_ref[0].transpose(1, 0, 2).astype(jnp.float32)
    v = v_ref[0].transpose(1, 0, 2).astype(jnp.float32)
    g_raw = g_ref[0].transpose(1, 0, 2).astype(jnp.float32)
    beta = (
        beta_dma_ref[0, :, 0, :MB].transpose(1, 0).astype(jnp.float32)
        if DIRECT_INPUT_DMA
        else (
            beta_ref[0, 0].transpose(1, 0).astype(jnp.float32)
            if BATCH_FIRST and beta_ref.ndim == 4
            else beta_ref[:, 0, 0, 0, :].astype(jnp.float32)
        )
    )
    beta = jnp.clip(beta, 0, 1)
    q *= jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k *= jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    q = q.astype(jnp.bfloat16).astype(jnp.float32)
    k = k.astype(jnp.bfloat16).astype(jnp.float32)

    # Gate activation (matches cu_seqlens kernel)
    g_f32 = g_raw
    if use_gate_in_kernel:
      dt_b = dt_bias_ref[:, 0, 0]
      g_f32 = g_f32 + dt_b[:, None, :]
      A_scale = A_scale_ref[:, 0, 0, 0]
      if lower_bound is None:
        g_f32 = -A_scale[:, None, None] * jax.nn.softplus(g_f32)
      else:
        g_f32 = lower_bound * jax.nn.sigmoid(A_scale[:, None, None] * g_f32)

    g_cumsum = g_f32 * cumsum_scale
    shift = 1
    while shift < BT:
      shifted = jnp.concatenate(
          [
              jnp.zeros_like(g_cumsum[:, :shift]),
              g_cumsum[:, :-shift],
          ],
          axis=1,
      )
      g_cumsum = g_cumsum + shifted
      shift *= 2

    # --- Stage 2: Intra-chunk solve ---
    BC = QK_BC
    NC = BT // BC
    beta_f32 = beta[:, :, None]
    ref_idx = BC // 2 if safe_gate else 0
    row_iota_bc_bt = jax.lax.broadcasted_iota(jnp.int32, (BC, BT), 0)
    col_iota_bc_bt = jax.lax.broadcasted_iota(jnp.int32, (BC, BT), 1)
    Aqk_rows, L_rows = [], []
    k_eng_prefix = None
    prev_gn = None
    for i_sc in range(NC):
      i_s = i_sc * BC
      q_i, k_i = q[:, i_s : i_s + BC], k[:, i_s : i_s + BC]
      g_i = g_cumsum[:, i_s : i_s + BC]
      beta_i = beta_f32[:, i_s : i_s + BC]
      gn = g_i[:, ref_idx : ref_idx + 1, :]
      diff_i = g_i - gn
      exp_i = jnp.exp2(diff_i)
      q_eg, k_eg = q_i * exp_i, k_i * exp_i
      j_end = i_s + BC
      k_eng_current = k_i * jnp.exp2(-diff_i)
      if i_sc == 0:
        k_eng_prefix = k_eng_current
      else:
        ref_decay = jnp.exp2(gn - prev_gn)
        k_eng_prefix = jnp.concatenate(
            [k_eng_prefix * ref_decay, k_eng_current],
            axis=1,
        )
      prev_gn = gn
      qk_eg = jnp.concatenate([q_eg, k_eg], axis=1)
      qk_dot_valid = jax.lax.dot_general(
          qk_eg,
          k_eng_prefix,
          (((2,), (2,)), ((0,), (0,))),
          precision=OUTPUT_PRECISION,
          preferred_element_type=jnp.float32,
      )
      if j_end < BT:
        qk_dot = jnp.concatenate(
            [
                qk_dot_valid,
                jnp.zeros((MB, 2 * BC, BT - j_end), dtype=jnp.float32),
            ],
            axis=2,
        )
      else:
        qk_dot = qk_dot_valid
      Aqk_r = qk_dot[:, :BC] * scale
      Akk_r = qk_dot[:, BC:] * beta_i
      ind = (col_iota_bc_bt >= i_s) & (col_iota_bc_bt < i_s + BC)
      cl = col_iota_bc_bt - i_s
      Aqk_r = jnp.where((~ind) | (row_iota_bc_bt >= cl), Aqk_r, 0.0)
      Akk_r = jnp.where((~ind) | (row_iota_bc_bt > cl), Akk_r, 0.0)
      Aqk_rows.append(Aqk_r)
      L_rows.append(Akk_r)

    g_last = g_cumsum[:, BT - 1 : BT, :]
    kg = k_eng_prefix * jnp.exp2(g_last - prev_gn)
    Aqk = jnp.concatenate(Aqk_rows, axis=1)
    L = jnp.concatenate(L_rows, axis=1)

    v_beta = v * beta_f32
    k_eg_beta = k * jnp.exp2(g_cumsum) * beta_f32
    I_bt = jnp.eye(BT, dtype=jnp.float32)
    _dot = lambda a, b: jax.lax.dot_general(
        a,
        b,
        (((2,), (1,)), ((0,), (0,))),
        precision=OUTPUT_PRECISION,
        preferred_element_type=jnp.float32,
    )

    BC_inv = INV_BC
    NC_inv = BT // BC_inv
    inv_dt = jnp.float32
    L_inv = L.astype(inv_dt)
    solve_bt = BT
    solve_I = I_bt
    if PACK_HEAD_INV:
      # Pair independent heads in the MXU matrix dimensions. A
      # 64x64 batched dot occupies a physical 128x128 tile, so two
      # heads can share that tile without changing either BC8
      # triangular system.
      pair_mb = MB // 2
      L_pair = L_inv.reshape(pair_mb, 2, BT, BT)
      z = jnp.zeros_like(L_pair[:, 0])
      L_inv = jnp.concatenate(
          [
              jnp.concatenate([L_pair[:, 0], z], axis=2),
              jnp.concatenate([z, L_pair[:, 1]], axis=2),
          ],
          axis=1,
      )
      solve_bt = 2 * BT
      solve_I = jnp.eye(solve_bt, dtype=inv_dt)
    _row = jax.lax.broadcasted_iota(
        jnp.int32, (solve_bt, solve_bt), dimension=0
    )
    _col = jax.lax.broadcasted_iota(
        jnp.int32, (solve_bt, solve_bt), dimension=1
    )
    _same = (_row // BC_inv == _col // BC_inv).astype(inv_dt)
    L_diag = L_inv * _same[None]
    F = L_inv - L_diag
    neg_Ld = -L_diag
    S = solve_I[None] + neg_Ld
    Mk = neg_Ld
    num_diag_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[BC_inv]
    for _ in range(num_diag_steps):
      Mk = _dot(Mk, Mk)
      S = _dot(S, solve_I[None] + Mk)
    P = S
    rhs = jnp.concatenate(
        [v_beta.astype(inv_dt), k_eg_beta.astype(inv_dt)], axis=-1
    )
    if PACK_HEAD_INV:
      rhs = rhs.reshape(MB // 2, 2 * BT, V_PAD + K_PAD)
    if NC_inv == 1:
      result = _dot(P, rhs)
    else:
      F_and_rhs = jnp.concatenate([F, rhs], axis=-1)
      P_merged = _dot(P, F_and_rhs)
      G = P_merged[:, :, :solve_bt]
      P_rhs = P_merged[:, :, solve_bt:]
      H_mat = -G
      inv_I_G = solve_I[None] + H_mat
      Hk = H_mat
      log2_NC_inv = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[NC_inv]
      num_horner_steps = log2_NC_inv - 1
      if num_horner_steps > 0:
        Hk = _dot(Hk, Hk)
        for _ in range(num_horner_steps - 1):
          merged_lhs = jnp.concatenate([inv_I_G, Hk], axis=1)
          merged_products = _dot(merged_lhs, Hk)
          inv_I_G = inv_I_G + merged_products[:, :solve_bt]
          Hk = merged_products[:, solve_bt:]
        inv_I_G = inv_I_G + _dot(inv_I_G, Hk)
      result = _dot(inv_I_G, P_rhs)
    if PACK_HEAD_INV:
      result = result.reshape(MB, BT, V_PAD + K_PAD)
    u = result[:, :, :V_PAD]
    w = result[:, :, V_PAD : V_PAD + K_PAD]
    if NC_inv == 1:
      A_inv = P

    # --- Stage 3+4: State + Output ---
    b_h = scratch_ref[...]
    pass
    b_qg = q * jnp.exp2(jnp.maximum(g_cumsum, -126.0))
    b_v_o = jnp.matmul(
        jnp.concatenate([w, b_qg], axis=1),
        b_h,
        precision=OUTPUT_PRECISION,
        preferred_element_type=jnp.float32,
    )
    b_v_new = u - b_v_o[:, :BT]
    b_o = b_v_o[:, BT:] * scale
    b_A = Aqk.astype(jnp.float32)
    b_o_h = jnp.matmul(
        jnp.concatenate([b_A, kg.transpose(0, 2, 1)], axis=1),
        b_v_new,
        precision=OUTPUT_PRECISION,
        preferred_element_type=jnp.float32,
    )
    b_o += b_o_h[:, :BT]
    o_ref[0] = b_o.transpose(1, 0, 2).astype(o_ref.dtype)

    b_gk_last = g_cumsum[:, BT - 1, :]
    b_h_new = b_h * jnp.exp2(b_gk_last)[:, :, None] + b_o_h[:, BT:]
    scratch_ref[...] = b_h_new
    pass
    prev_seg_ref[...] = jnp.broadcast_to(first_seg, (128,))

    if STORE_FINAL_STATE:

      @pl.when(is_last_real)
      def _final():
        if MANUAL_HT_DMA:
          _dma_state_to_ht(scratch_ref, first_seg)
        else:
          ht_ref[first_seg - 1, ...] = b_h_new.astype(ht_ref.dtype)[
              :, None, :, :
          ]

  # PARTIAL_PAD ------------------------------------------------------
  @pl.when(kind == CHUNK_KIND_PARTIAL_PAD)
  def _partial():
    seg = _load_token_segment_ids()
    prev_seg = prev_seg_ref[0]
    seg_changed = (first_seg != prev_seg) | chain_start

    if STORE_FINAL_STATE:

      @pl.when(seg_changed & (~chain_start))
      def _save_prev():
        if MANUAL_HT_DMA:
          _dma_state_to_ht(scratch_ref, prev_seg)
        else:
          ht_ref[prev_seg - 1, ...] = scratch_ref[...].astype(ht_ref.dtype)[
              :, None, :, :
          ]

    @pl.when(seg_changed)
    def _init():
      scratch_ref[...] = jnp.zeros([MB, K_PAD, V_PAD], dtype=jnp.float32)

    if HAS_H0:

      @pl.when(seg_changed)
      def _load_h0():
        if MANUAL_H0_DMA:
          h_start = pl.program_id(0) * MB
          cp = pltpu.make_async_copy(
              h0_ref.at[
                  first_seg - 1,
                  pl.ds(h_start, MB),
                  i_b,
                  pl.ds(None),
                  pl.ds(None),
              ],
              scratch_ref,
              state_dma_sem,
          )
          cp.start()
          cp.wait()
        else:
          scratch_ref[...] = _read_h0(first_seg)

    vm = (seg > 0).astype(jnp.float32)  # [BT] valid mask
    q = q_ref[0].transpose(1, 0, 2).astype(jnp.float32) * vm[None, :, None]
    k = k_ref[0].transpose(1, 0, 2).astype(jnp.float32) * vm[None, :, None]
    v = v_ref[0].transpose(1, 0, 2).astype(jnp.float32) * vm[None, :, None]
    g_raw = g_ref[0].transpose(1, 0, 2).astype(jnp.float32) * vm[None, :, None]
    beta = (
        beta_dma_ref[0, :, 0, :MB].transpose(1, 0).astype(jnp.float32)
        if DIRECT_INPUT_DMA
        else (
            beta_ref[0, 0].transpose(1, 0).astype(jnp.float32)
            if BATCH_FIRST and beta_ref.ndim == 4
            else beta_ref[:, 0, 0, 0, :].astype(jnp.float32)
        )
    ) * vm[None, :]
    beta = jnp.clip(beta, 0, 1)
    q *= jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k *= jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    q = q.astype(jnp.bfloat16).astype(jnp.float32)
    k = k.astype(jnp.bfloat16).astype(jnp.float32)

    # Gate activation (must match FULL chunk)
    g_f32 = g_raw
    if use_gate_in_kernel:
      dt_b = dt_bias_ref[:, 0, 0]
      g_f32 = g_f32 + dt_b[:, None, :] * vm[None, :, None]
      A_scale = A_scale_ref[:, 0, 0, 0]
      if lower_bound is None:
        g_f32 = -A_scale[:, None, None] * jax.nn.softplus(g_f32)
      else:
        g_f32 = lower_bound * jax.nn.sigmoid(A_scale[:, None, None] * g_f32)
      g_f32 = g_f32 * vm[None, :, None]

    g_cumsum = g_f32 * cumsum_scale
    shift = 1
    while shift < BT:
      shifted = jnp.concatenate(
          [
              jnp.zeros_like(g_cumsum[:, :shift]),
              g_cumsum[:, :-shift],
          ],
          axis=1,
      )
      g_cumsum = g_cumsum + shifted
      shift *= 2

    BC = QK_BC
    NC = BT // BC
    beta_f32 = beta[:, :, None]
    ref_idx = BC // 2 if safe_gate else 0
    row_iota_bc_bt = jax.lax.broadcasted_iota(jnp.int32, (BC, BT), 0)
    col_iota_bc_bt = jax.lax.broadcasted_iota(jnp.int32, (BC, BT), 1)
    Aqk_rows, L_rows = [], []
    k_eng_prefix = None
    prev_gn = None
    for i_sc in range(NC):
      i_s = i_sc * BC
      q_i, k_i = q[:, i_s : i_s + BC], k[:, i_s : i_s + BC]
      g_i = g_cumsum[:, i_s : i_s + BC]
      beta_i = beta_f32[:, i_s : i_s + BC]
      gn = g_i[:, ref_idx : ref_idx + 1, :]
      diff_i = g_i - gn
      exp_i = jnp.exp2(diff_i)
      q_eg, k_eg = q_i * exp_i, k_i * exp_i
      j_end = i_s + BC
      k_eng_current = k_i * jnp.exp2(-diff_i)
      if i_sc == 0:
        k_eng_prefix = k_eng_current
      else:
        ref_decay = jnp.exp2(gn - prev_gn)
        k_eng_prefix = jnp.concatenate(
            [k_eng_prefix * ref_decay, k_eng_current],
            axis=1,
        )
      prev_gn = gn
      qk_eg = jnp.concatenate([q_eg, k_eg], axis=1)
      qk_dot_valid = jax.lax.dot_general(
          qk_eg,
          k_eng_prefix,
          (((2,), (2,)), ((0,), (0,))),
          precision=OUTPUT_PRECISION,
          preferred_element_type=jnp.float32,
      )
      if j_end < BT:

        qk_dot = jnp.concatenate(
            [
                qk_dot_valid,
                jnp.zeros((MB, 2 * BC, BT - j_end), dtype=jnp.float32),
            ],
            axis=2,
        )
      else:
        qk_dot = qk_dot_valid
      Aqk_r = qk_dot[:, :BC] * scale
      Akk_r = qk_dot[:, BC:] * beta_i
      ind = (col_iota_bc_bt >= i_s) & (col_iota_bc_bt < i_s + BC)
      cl = col_iota_bc_bt - i_s
      Aqk_r = jnp.where((~ind) | (row_iota_bc_bt >= cl), Aqk_r, 0.0)
      Akk_r = jnp.where((~ind) | (row_iota_bc_bt > cl), Akk_r, 0.0)
      Aqk_rows.append(Aqk_r)
      L_rows.append(Akk_r)
    Aqk = jnp.concatenate(Aqk_rows, axis=1)
    L = jnp.concatenate(L_rows, axis=1)
    v_beta = v * beta_f32
    k_eg_beta = k * jnp.exp2(g_cumsum) * beta_f32
    I_bt = jnp.eye(BT, dtype=jnp.float32)
    _dot = lambda a, b: jax.lax.dot_general(
        a,
        b,
        (((2,), (1,)), ((0,), (0,))),
        precision=OUTPUT_PRECISION,
        preferred_element_type=jnp.float32,
    )
    BC_inv = INV_BC
    NC_inv = BT // BC_inv
    inv_dt = jnp.float32
    L_inv = L.astype(inv_dt)
    _row = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
    _col = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
    _same = (_row // BC_inv == _col // BC_inv).astype(inv_dt)
    L_diag = L_inv * _same[None]
    F = L_inv - L_diag
    neg_Ld = -L_diag
    S = I_bt[None] + neg_Ld
    Mk = neg_Ld
    num_diag_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[BC_inv]
    for _ in range(num_diag_steps):
      Mk = _dot(Mk, Mk)
      S = _dot(S, I_bt[None] + Mk)
    P = S
    rhs = jnp.concatenate(
        [v_beta.astype(inv_dt), k_eg_beta.astype(inv_dt)], axis=-1
    )
    if NC_inv == 1:
      result = _dot(P, rhs)
    else:
      F_and_rhs = jnp.concatenate([F, rhs], axis=-1)
      P_merged = _dot(P, F_and_rhs)
      G = P_merged[:, :, :BT]
      P_rhs = P_merged[:, :, BT:]
      H_mat = -G
      inv_I_G = I_bt[None] + H_mat
      Hk = H_mat
      log2_NC_inv = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[NC_inv]
      num_horner_steps = log2_NC_inv - 1
      if num_horner_steps > 0:
        Hk = _dot(Hk, Hk)
        for _ in range(num_horner_steps - 1):
          merged_lhs = jnp.concatenate([inv_I_G, Hk], axis=1)
          merged_products = _dot(merged_lhs, Hk)
          inv_I_G = inv_I_G + merged_products[:, :BT]
          Hk = merged_products[:, BT:]
        inv_I_G = inv_I_G + _dot(inv_I_G, Hk)
      result = _dot(inv_I_G, P_rhs)
    u = result[:, :, :V_PAD]
    w = result[:, :, V_PAD : V_PAD + K_PAD]
    if NC_inv == 1:
      A_inv = P
    g_last = (g_cumsum * vm[None, :, None]).min(axis=1, keepdims=True)
    kg = k * jnp.exp2(g_last - g_cumsum) * vm[None, :, None]

    b_h = scratch_ref[...]
    pass
    b_qg = q * jnp.exp2(jnp.maximum(g_cumsum, -126.0))
    b_v_o = jnp.matmul(
        jnp.concatenate([w, b_qg], axis=1),
        b_h,
        precision=OUTPUT_PRECISION,
        preferred_element_type=jnp.float32,
    )
    b_v_new = u - b_v_o[:, :BT]
    b_o = b_v_o[:, BT:] * scale
    b_A = Aqk.astype(jnp.float32)
    b_o_h = jnp.matmul(
        jnp.concatenate([b_A, kg.transpose(0, 2, 1)], axis=1),
        b_v_new,
        precision=OUTPUT_PRECISION,
        preferred_element_type=jnp.float32,
    )
    b_o += b_o_h[:, :BT]
    b_o = b_o * vm[None, :, None]
    o_ref[0] = b_o.transpose(1, 0, 2).astype(o_ref.dtype)

    b_gk_last = g_last[:, 0, :]
    b_h_new = b_h * jnp.exp2(b_gk_last)[:, :, None] + b_o_h[:, BT:]
    scratch_ref[...] = b_h_new
    if STORE_FINAL_STATE:
      if MANUAL_HT_DMA:
        _dma_state_to_ht(scratch_ref, first_seg)
      else:
        ht_ref[first_seg - 1, ...] = b_h_new.astype(ht_ref.dtype)[:, None, :, :]
    pass
    prev_seg_ref[...] = jnp.broadcast_to(first_seg, (128,))

    if STORE_FINAL_STATE:

      @pl.when(is_last_real)
      def _final():
        if MANUAL_HT_DMA:
          _dma_state_to_ht(scratch_ref, first_seg)
        else:
          ht_ref[first_seg - 1, ...] = b_h_new.astype(ht_ref.dtype)[
              :, None, :, :
          ]

  # BOUNDARY: two segments in one chunk -----------------------------
  @pl.when(kind == CHUNK_KIND_BOUNDARY)
  def _boundary():
    # Native boundary blocks may contain any number of segments. The
    # source's two-segment solve silently omitted middle segments. Use a
    # recurrence for this uncommon block kind; full blocks retain the MXU
    # fused solve. No aligned inputs or backward residuals are produced.
    seg = _load_token_segment_ids()
    q = q_ref[0].transpose(1, 0, 2).astype(jnp.float32)
    k = k_ref[0].transpose(1, 0, 2).astype(jnp.float32)
    v = v_ref[0].transpose(1, 0, 2).astype(jnp.float32)
    raw_g = g_ref[0].transpose(1, 0, 2).astype(jnp.float32)
    beta = (
        beta_ref[0, 0].transpose(1, 0)
        if beta_ref.ndim == 4
        else beta_ref[:, 0, 0, 0, :]
    ).astype(jnp.float32)
    beta = jnp.clip(beta, 0, 1)
    q = (
        (q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6))
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    k = (
        (k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6))
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    gate = lower_bound * jax.nn.sigmoid(
        A_scale_ref[:, 0, 0, 0][:, None, None]
        * (raw_g + dt_bias_ref[:, 0, 0][:, None, :])
    )
    previous = jnp.where(chain_start, 0, prev_seg_ref[0])
    state = jnp.where(
        chain_start, jnp.zeros_like(scratch_ref[...]), scratch_ref[...]
    )

    def token_step(t, carry):
      state, previous, output = carry
      token_mask = (
          jax.lax.broadcasted_iota(jnp.int32, (BT,), dimension=0) == t
      )
      current = jnp.max(jnp.where(token_mask, seg, 0))
      valid = current > 0
      changed = valid & (current != previous)
      if STORE_FINAL_STATE:

        @pl.when(changed & (previous > 0))
        def save_previous():
          if MANUAL_HT_DMA:
            final_state_dma_ref[...] = state
            _dma_state_to_ht(final_state_dma_ref, previous)
          else:
            ht_ref[previous - 1, ...] = state[:, None, :, :]

      if HAS_H0:
        if MANUAL_H0_DMA:

          @pl.when(changed)
          def load_initial():
            copy = pltpu.make_async_copy(
                h0_ref.at[
                    jnp.maximum(current - 1, 0),
                    pl.ds(pl.program_id(0) * MB, MB),
                    i_b,
                    pl.ds(None),
                    pl.ds(None),
                ],
                final_state_dma_ref,
                state_dma_sem,
            )
            copy.start()
            copy.wait()

          initial = final_state_dma_ref[...]
        else:
          initial = _read_h0(jnp.maximum(current, 1))
      else:
        initial = jnp.zeros_like(state)
      state = jnp.where(changed, initial, state)
      # Dynamic array slices have no Pallas TPU lowering.
      def token(x):
        token_position = jax.lax.broadcasted_iota(
            jnp.int32, x.shape, dimension=1
        )
        return jnp.sum(jnp.where(token_position == t, x, 0), axis=1)

      kt = token(k)
      decayed = state * jnp.exp(token(gate))[:, :, None]
      correction = token(beta[:, :, None]) * (
          token(v) - jnp.sum(kt[:, :, None] * decayed, axis=1)
      )
      updated = decayed + kt[:, :, None] * correction[:, None, :]
      state = jnp.where(valid, updated, state)
      value = jnp.where(
          valid, jnp.sum(token(q)[:, :, None] * state, axis=1) * scale, 0
      )
      output_position = jax.lax.broadcasted_iota(
          jnp.int32, output.shape, dimension=1
      )
      output = jnp.where(output_position == t, value[:, None, :], output)
      return state, jnp.where(valid, current, previous), output

    state, previous, output = jax.lax.fori_loop(
        0,
        BT,
        token_step,
        (state, previous, jnp.zeros((MB, BT, V_PAD), jnp.float32)),
    )
    scratch_ref[...] = state
    prev_seg_ref[...] = jnp.broadcast_to(previous, (128,))
    o_ref[0] = output.transpose(1, 0, 2).astype(o_ref.dtype)
    if STORE_FINAL_STATE:
      if MANUAL_HT_DMA:
        _dma_state_to_ht(scratch_ref, previous)
      else:
        ht_ref[previous - 1, ...] = state[:, None, :, :]


def _fwd_mega_kernel_native_segids_packed(packed_metadata_ref, *args, **kwargs):
  return _fwd_mega_kernel_native_segids(
      packed_metadata_ref,
      packed_metadata_ref,
      packed_metadata_ref,
      *args,
      **kwargs,
  )


_NATIVE_SEGIDS_STATIC_ARGNAMES = [
    "output_final_state",
    "scale",
    "chunk_size",
    "store_h",
    "store_v_new",
    "disable_recompute",
    "only_fwd",
    "safe_gate",
    "use_qk_l2norm_in_kernel",
    "use_gate_in_kernel",
    "lower_bound",
    "mini_batch",
    "N_max",
    "residual_chunk_layout",
    "batch_first",
]


def _chunk_kda_fwd_native_segids_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    segment_ids: jax.Array,
    initial_state=None,
    output_final_state=False,
    scale=1.0,
    chunk_size=64,
    store_h=False,
    store_v_new=False,
    disable_recompute=False,
    only_fwd=False,
    safe_gate=True,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    mini_batch=None,
    N_max=None,
    residual_chunk_layout=False,
    batch_first=False,
):
  """Native segment_ids mega kernel wrapper. No _align_seqs."""
  if (
      not only_fwd
      or not batch_first
      or store_h
      or store_v_new
      or disable_recompute
  ):
    raise ValueError("This kernel supports batch-first inference only")
  B, T, H, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  pass
  pass

  # Normalize segment_ids to [B, T]
  if segment_ids.ndim == 1:
    segment_ids = segment_ids[None, :]
    if B > 1:
      segment_ids = jnp.broadcast_to(segment_ids, (B, T))

  segment_local = False
  direct_input_dma = False
  original_T = T
  original_segment_ids = segment_ids
  aligned_cu = None
  pass

  NT = T // BT
  packed_metadata = True
  chunk_kind, seg_first, seg_last, seg_id = _build_chunk_metadata(
      segment_ids, BT
  )
  next_kind = jnp.concatenate(
      [
          chunk_kind[:, 1:],
          jnp.zeros((B, 1), dtype=jnp.int32),
      ],
      axis=1,
  )
  is_last_real_chunk = (chunk_kind != CHUNK_KIND_ALL_PAD) & (
      next_kind == CHUNK_KIND_ALL_PAD
  )
  chunk_kind = chunk_kind | (
      is_last_real_chunk.astype(jnp.int32) * CHUNK_FLAG_LAST_REAL
  )
  if N_max is None:
    N_max = int(
        segment_ids.max()
    )  # Fallback: should not happen when called via dispatch
  # The packed representation uses 3 kind/flag bits and two 14-bit segment IDs.
  # Preserve the legacy three-prefetch ABI for unusually large N_max.
  if packed_metadata and N_max >= 16384:
    packed_metadata = False

  K_PAD = int(align_up(K, 128))
  V_ALIGNED = int(align_up(V, 128))
  # Pad T to 128-aligned for segment_ids block spec (TPU requires last dim % 128 == 0)
  T_PAD_S = int(align_up(T, 128))
  NT_S = T_PAD_S // 128  # number of 128-blocks in T

  def _pad(x, t):
    p = t - x.shape[-1]
    return jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, p))) if p > 0 else x

  q_t, k_t, v_t, g_t = (
      _pad(q, K_PAD),
      _pad(k, K_PAD),
      _pad(v, V_ALIGNED),
      _pad(g, K_PAD),
  )
  pass
  # Pad segment_ids to [B, T_PAD_S] (128-aligned in T dim)
  if segment_ids.shape[-1] < T_PAD_S:
    segment_ids = jnp.pad(
        segment_ids, ((0, 0), (0, T_PAD_S - segment_ids.shape[-1]))
    )

  # Rebuild metadata with padded T (chunk_size still BT, but more chunks now due to padding)
  # Metadata arrays need last dim >= 128 or == array dim. Pad NT to 128 if needed.
  NT_meta = T // BT  # original number of chunks
  NT_meta_PAD = max(int(align_up(NT_meta, 128)), 128)  # at least 128
  chunk_kind_padded = jnp.pad(
      chunk_kind,
      ((0, 0), (0, NT_meta_PAD - NT_meta)),
      constant_values=CHUNK_KIND_ALL_PAD,
  )
  seg_first_padded = jnp.pad(seg_first, ((0, 0), (0, NT_meta_PAD - NT_meta)))
  seg_last_padded = jnp.pad(seg_last, ((0, 0), (0, NT_meta_PAD - NT_meta)))
  max_segment_chunks = None
  pass

  if mini_batch is not None:
    MB = mini_batch
  per_head_bytes = (
      8 * K_PAD * V_ALIGNED * 4
      + 24 * BT * max(K_PAD, V_ALIGNED) * 4
      + 32 * BT * BT * 4
  )
  MB = estimate_mini_batch(per_head_bytes, H, max_mb=32)
  while H % MB != 0:
    MB //= 2
  pass

  # The batch-first beta window is copy-free when one program owns the full
  # head dimension. Smaller head groups use the head-first layout whose last
  # dimension is the complete 64-token block and therefore TPU legal.
  beta_batch_first = batch_first and only_fwd and MB == H
  beta_t = (
      beta.reshape(B, NT, BT, H)
      if beta_batch_first
      else beta.transpose(2, 0, 1).reshape(H, B, NT, 1, BT)
  )

  use_neumann = q.dtype == jnp.bfloat16
  # With gate values as low as -5, a 32-token block can form masked
  # pre-causal products near 2**155 before the mask is applied. Keeping the
  # rescaling window at 16 bounds those products below the FP32/BF16 range.
  qk_bc = 16
  # A monolithic 64x64 Neumann polynomial can reach O(1e17)
  # intermediates for correlated normalized keys with near-zero decay. The
  # mathematically cancelling result then loses precision in FP32. Eight
  # 8-token diagonal blocks bound the cancellation before the block-level
  # composition on TPU MXU.
  inv_bc = 8
  # Stage 2 has already made Aqk causal in all three native segment-ID
  # chunk variants.  In inference, avoid rebuilding and applying the same
  # 64x64 mask on the Stage 4 dependency chain.  Keep the training lowering
  # unchanged because it shares this kernel body.
  skip_stage4_mask = only_fwd
  pack_head_inv = MB % 2 == 0
  clip_beta_in_kernel = only_fwd
  overlap_h0_dma = True
  overlap_ht_dma = True
  _prec = (
      jax.lax.Precision.DEFAULT if use_neumann else jax.lax.Precision.HIGHEST
  )

  # Prepare h0: broadcast to [1, H, 1, K_PAD, V_ALIGNED] so each grid point
  # can read its own MB-sized block via _h0_map (avoids OOB on dim1).
  has_h0 = initial_state is not None
  manual_state_dma = N_max >= 6
  manual_h0_dma = has_h0 and N_max > 6
  if has_h0:
    h0 = initial_state
    if h0.ndim == 4:
      h0 = h0[None, ...]  # [1, N, H, K, V]
    if h0.shape[0] < B:
      h0 = jnp.broadcast_to(h0, (B,) + h0.shape[1:])

    # Pass ALL N_max initial states for per-segment loading
    # h0: [B, N_max, H, K, V] -> [N_max, H, B, K, V] -> pad -> [N_max, H, B, K_PAD, V_ALIGNED]
    h0 = h0[:, :, :, :, :].transpose(1, 2, 0, 3, 4)  # [N_max, H, B, K, V]
    if K_PAD > K:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, K_PAD - K), (0, 0)))
    if V_ALIGNED > V:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
    h0_in = h0.astype(jnp.float32)  # [N_max, H, B, K_PAD, V_ALIGNED]
  else:
    h0_in = None

  # Prepare A_log and dt_bias for gate activation inside kernel
  if use_gate_in_kernel and A_log is not None:
    H_A = A_log.shape[0]
    n_rep = H // H_A
    A_expanded = (
        jnp.repeat(A_log.astype(jnp.float32), n_rep)
        if n_rep > 1
        else A_log.astype(jnp.float32)
    )
    A_scale_in = jnp.exp(A_expanded).reshape(H, 1, 1, 1)
  else:
    A_scale_in = jnp.zeros((H, 1, 1, 1), dtype=jnp.float32)

  if use_gate_in_kernel and dt_bias is not None:
    H_A = A_log.shape[0]
    n_rep = H // H_A
    db_2d = dt_bias.reshape(-1)[: H_A * K].reshape(H_A, K).astype(jnp.float32)
    if n_rep > 1:
      db_2d = jnp.repeat(db_2d, n_rep, axis=0)
    db_in = db_2d[:, None, None, :].astype(jnp.float32)  # [H, 1, 1, K]
    if K_PAD > K:
      db_in = jnp.pad(db_in, ((0, 0), (0, 0), (0, 0), (0, K_PAD - K)))
    db_in = jnp.broadcast_to(db_in, (H, 1, 1, K_PAD))
    # db_in stays [H, 1, 1, K_PAD]
  else:
    db_in = jnp.zeros((H, 1, 1, K_PAD), dtype=jnp.float32)

  # Block specs -- index maps must accept all prefetch refs as args
  # Note: index map returns block offset (in units of block size), not array element offset
  def _seg_map(h, b, c, *refs):
    # segment_ids has shape [B, T_PAD_S], block size is [B, 128]
    # So we need to return (b_offset, seg_offset) where seg_offset is in units of 128
    block_128 = (c * BT) // 128
    return (0, block_128)

  def _in_map(h, b, c, *refs):
    return (b, c, h, 0) if batch_first else (h, b, c, 0)

  def _h0_map(h, b, c, *refs):
    return (0, h, b, 0, 0)

  def _out_map(h, b, c, *refs):
    return (h, b, c, 0)

  def _out_chunk_map(h, b, c, *refs):
    return (h, b, c, 0, 0)

  def _beta_map(h, b, c, *refs):
    return (b, c, 0, h) if beta_batch_first else (h, b, c, 0, 0)

  def _ht_map(h, b, c, *refs):
    return (0, h, b, 0, 0)

  pass

  seg_spec = pl.BlockSpec([B, 128], index_map=_seg_map)
  q_spec = pl.BlockSpec(
      [1, BT, MB, K_PAD] if batch_first else [MB, 1, BT, K_PAD],
      index_map=_in_map,
  )
  k_spec = pl.BlockSpec(
      [1, BT, MB, K_PAD] if batch_first else [MB, 1, BT, K_PAD],
      index_map=_in_map,
  )
  v_spec = pl.BlockSpec(
      [1, BT, MB, V_ALIGNED] if batch_first else [MB, 1, BT, V_ALIGNED],
      index_map=_in_map,
  )
  g_spec = pl.BlockSpec(
      [1, BT, MB, K_PAD] if batch_first else [MB, 1, BT, K_PAD],
      index_map=_in_map,
  )
  beta_spec = (
      pl.BlockSpec([1, 1, BT, MB], index_map=_beta_map)
      if beta_batch_first
      else pl.BlockSpec([MB, 1, 1, 1, BT], index_map=_beta_map)
  )
  h0_spec = (
      (
          pl.BlockSpec(memory_space=pl.ANY)
          if manual_h0_dma
          else pl.BlockSpec(
              [N_max, MB, 1, K_PAD, V_ALIGNED],
              index_map=_h0_map,
          )
      )
      if has_h0
      else None
  )

  def _alog_map(h, b, c, *refs):
    return (h, 0, 0, 0)

  def _dtbias_map(h, b, c, *refs):
    return (h, 0, 0, 0)

  alog_spec = pl.BlockSpec([MB, 1, 1, 1], index_map=_alog_map)
  dtbias_spec = pl.BlockSpec([MB, 1, 1, K_PAD], index_map=_dtbias_map)

  def _o_map(h, b, c, *refs):
    return (b, c, h, 0)

  pass

  o_spec = pl.BlockSpec([1, BT, MB, V_ALIGNED], index_map=_o_map)
  store_final_state = output_final_state or store_h
  manual_ht_dma = store_final_state and manual_state_dma
  ht_spec = (
      (
          pl.BlockSpec(memory_space=pl.ANY)
          if manual_ht_dma
          else pl.BlockSpec(
              [N_max, MB, 1, K_PAD, V_ALIGNED],
              index_map=_ht_map,
          )
      )
      if store_final_state
      else None
  )

  store_residuals = not only_fwd
  store_chunk_h = store_residuals and store_h
  aqk_spec = (
      pl.BlockSpec(
          [MB, 1, 1, BT, BT] if residual_chunk_layout else [MB, 1, BT, BT],
          index_map=_out_chunk_map if residual_chunk_layout else _out_map,
      )
      if store_residuals
      else None
  )
  akk_spec = (
      pl.BlockSpec(
          [MB, 1, 1, BT, BT] if residual_chunk_layout else [MB, 1, BT, BT],
          index_map=_out_chunk_map if residual_chunk_layout else _out_map,
      )
      if store_residuals
      else None
  )
  g_cumsum_spec = (
      pl.BlockSpec([MB, 1, BT, K_PAD], index_map=_out_map)
      if store_residuals
      else None
  )
  chunk_h_spec = (
      pl.BlockSpec(
          [MB, 1, 1, K_PAD, V_ALIGNED],
          index_map=lambda h, b, c, *r: (h, b, c, 0, 0),
      )
      if store_chunk_h
      else None
  )

  aqk_shape = (
      jax.ShapeDtypeStruct(
          (H, B, NT, BT, BT) if residual_chunk_layout else (H, B, T, BT),
          q.dtype,
      )
      if store_residuals
      else None
  )
  akk_shape = (
      jax.ShapeDtypeStruct(
          (H, B, NT, BT, BT) if residual_chunk_layout else (H, B, T, BT),
          q.dtype,
      )
      if store_residuals
      else None
  )
  g_cumsum_shape = (
      jax.ShapeDtypeStruct((H, B, T, K_PAD), jnp.float32)
      if store_residuals
      else None
  )

  chunk_h_shape = (
      jax.ShapeDtypeStruct((H, B, NT, K_PAD, V_ALIGNED), jnp.float32)
      if store_chunk_h
      else None
  )

  grid = (
      (H // MB, B, N_max, max_segment_chunks)
      if segment_local
      else (H // MB, B, NT)
  )
  if packed_metadata:
    native_kernel = _fwd_mega_kernel_native_segids_packed
    scalar_prefetch_count = 1
    packed_meta = (
        chunk_kind_padded.astype(jnp.int32)
        | (seg_first_padded.astype(jnp.int32) << jnp.int32(3))
        | (seg_last_padded.astype(jnp.int32) << jnp.int32(17))
    )
    scalar_inputs = (packed_meta,)
  else:
    native_kernel = _fwd_mega_kernel_native_segids
    scalar_prefetch_count = 3
    scalar_inputs = (
        chunk_kind_padded,
        seg_first_padded,
        seg_last_padded,
    )

  o_out, ht_out, Aqk_out, Akk_out, g_cumsum_out, chunk_h_out = pl.pallas_call(
      functools.partial(
          native_kernel,
          NT=NT,
          BT=BT,
          N_max=N_max,
          scale=scale,
          cumsum_scale=_RCP_LN2,
          MB=MB,
          K_PAD=K_PAD,
          V_PAD=V_ALIGNED,
          OUTPUT_PRECISION=_prec,
          safe_gate=safe_gate,
          NORMALIZE_QK=use_qk_l2norm_in_kernel,
          use_gate_in_kernel=use_gate_in_kernel,
          lower_bound=lower_bound,
          USE_NEUMANN=use_neumann,
          QK_BC=qk_bc,
          INV_BC=inv_bc,
          SKIP_STAGE4_MASK=skip_stage4_mask,
          PACK_HEAD_INV=pack_head_inv,
          CLIP_BETA_IN_KERNEL=clip_beta_in_kernel,
          PACKED_METADATA=packed_metadata,
          HAS_H0=has_h0,
          STORE_RESIDUALS=store_residuals,
          STORE_H=store_chunk_h,
          STORE_FINAL_STATE=store_final_state,
          BATCH_FIRST=batch_first,
          MANUAL_H0_DMA=manual_h0_dma,
          MANUAL_HT_DMA=manual_ht_dma,
          OVERLAP_H0_DMA=overlap_h0_dma,
          OVERLAP_HT_DMA=overlap_ht_dma,
          RESIDUAL_CHUNK_LAYOUT=residual_chunk_layout,
          SEGMENT_LOCAL=segment_local,
          DIRECT_INPUT_DMA=direct_input_dma,
      ),
      out_shape=[
          jax.ShapeDtypeStruct(
              (B, T, H, V_ALIGNED) if batch_first else (H, B, T, V_ALIGNED),
              v.dtype,
          ),
          (
              jax.ShapeDtypeStruct((N_max, H, B, K_PAD, V_ALIGNED), jnp.float32)
              if store_final_state
              else None
          ),
          aqk_shape,
          akk_shape,
          g_cumsum_shape,
          chunk_h_shape,
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=scalar_prefetch_count,
          grid=grid,
          in_specs=[
              seg_spec,
              q_spec,
              k_spec,
              v_spec,
              g_spec,
              beta_spec,
              h0_spec,
              alog_spec,
              dtbias_spec,
          ],
          out_specs=[
              o_spec,
              ht_spec,
              aqk_spec,
              akk_spec,
              g_cumsum_spec,
              chunk_h_spec,
          ],
          scratch_shapes=[
              pltpu.VMEM((MB, K_PAD, V_ALIGNED), jnp.float32),
              pltpu.VMEM((128,), jnp.int32),
              pltpu.VMEM((MB, K_PAD, V_ALIGNED), jnp.float32),
              (
                  pltpu.VMEM((1, BT, MB, K_PAD), q.dtype)
                  if direct_input_dma
                  else None
              ),
              (
                  pltpu.VMEM((1, BT, MB, K_PAD), k.dtype)
                  if direct_input_dma
                  else None
              ),
              (
                  pltpu.VMEM((1, BT, MB, V_ALIGNED), v.dtype)
                  if direct_input_dma
                  else None
              ),
              (
                  pltpu.VMEM((1, BT, MB, K_PAD), g.dtype)
                  if direct_input_dma
                  else None
              ),
              (
                  pltpu.VMEM((1, BT, 1, 128), beta.dtype)
                  if direct_input_dma
                  else None
              ),
              pltpu.SemaphoreType.DMA((5,)) if direct_input_dma else None,
              pltpu.SemaphoreType.DMA,
          ],
      ),
      interpret=pltpu.InterpretParams(detect_races=True)
      if get_interpret()
      else False,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              ("parallel", "parallel", "parallel", "arbitrary")
              if segment_local
              else ("parallel", "parallel", "arbitrary")
          ),
          disable_bounds_checks=True,
          vmem_limit_bytes=get_tpu_limits().vmem_limit_bytes,
      ),
  )(
      *scalar_inputs,
      segment_ids,
      q_t,
      k_t,
      v_t,
      g_t,
      beta_t,
      h0_in,
      A_scale_in,
      db_in,
  )

  # Match the selected contract: [B, T, H, V] or [H, B, T, V].
  o_out = o_out[..., :V]
  pass
  # ht_out: [1, H, B, K_PAD, V_ALIGNED] -> [B, 1, H, K, V]
  if ht_out is not None:
    ht_out = ht_out[:, :, :, :K, :V].transpose(
        2, 0, 1, 3, 4
    )  # [B, N_max, H, K, V]

  # Empty segment slots have no writer in the native kernel. Preserve h0
  # for them, or zero when no initial state was supplied.
  if ht_out is not None:
    present = (
        original_segment_ids[:, :, None]
        == jnp.arange(1, N_max + 1)[None, None, :]
    ).any(axis=1)
    empty_state = (
        jnp.zeros_like(ht_out)
        if initial_state is None
        else jnp.broadcast_to(initial_state, ht_out.shape)
    )
    ht_out = jnp.where(present[:, :, None, None, None], ht_out, empty_state)
  final_state = ht_out if store_final_state else None

  # Trim g_cumsum: [H, B, T, K_PAD] -> [H, B, T, K]
  if g_cumsum_out is not None and K_PAD > K:
    g_cumsum_out = g_cumsum_out[:, :, :, :K]

  if chunk_h_out is not None:
    chunk_h_out = chunk_h_out[:, :, :, :K, :V]
  return (
      o_out,
      final_state,
      g_cumsum_out,
      Aqk_out,
      Akk_out,
      None,
      None,
      None,
      None,
      None,
      chunk_h_out,
      None,
  )


_chunk_kda_fwd_native_segids = jax.jit(
    _chunk_kda_fwd_native_segids_impl,
    static_argnames=_NATIVE_SEGIDS_STATIC_ARGNAMES,
)
