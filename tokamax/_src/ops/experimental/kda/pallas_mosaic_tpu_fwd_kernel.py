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
"""Pallas/Mosaic TPU forward kernels for experimental KDA."""

from __future__ import annotations

import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member

from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda.common import (
  estimate_mini_batch,
  get_tpu_limits,
  RCP_LN2,
)
from tokamax._src.ops.experimental.kda.cp_utils import (
  ContextParallelMetadata,
  _merge_initial_state,
  all_gather_into_tensor,
)
from tokamax._src.ops.experimental.kda.pallas_mosaic_tpu_types import KdaResiduals
from tokamax._src.ops.experimental.kda.utils import (
  _unalign_output,
  align_up,
  exp2,
  get_interpret,
  prepare_chunk_indices,
)


# ─── Context-Parallel Pre-Process ──────────────────────────────────────────


def _pre_process_kernel(
  seqlens_ref,        # scalar prefetch: cu_seqlens [N+1]
  chunk_to_seq_ref,   # scalar prefetch: chunk -> seq mapping [NT]
  last_seg_idx_ref,   # scalar prefetch: real last seg idx [1], int32
  k_ref,    # [MB, 1, BT, K_PADSIZE]
  w_ref,    # [MB, 1, BT, K_PADSIZE]
  u_ref,    # [MB, 1, BT, V_ALIGNED]
  gk_ref,   # [MB, 1, BT, K_PADSIZE]
  S_ext_ref,  # [MB, 1, K_PADSIZE, V_ALIGNED]
  M_ref,      # [MB, 1, K_PADSIZE, K_PADSIZE]
  h_scratch,  # [MB, K_PADSIZE, V_ALIGNED]
  m_scratch,  # [MB, K_PADSIZE, K_PADSIZE]
  *,
  BT,
  MB,
):
  """Fused (S_ext, M) pre-process Pallas kernel for KDA CP forward.

  MB heads form the batch dimension of ``dot_general``
  (no Python unroll). For each program point (h_group, i_c):
    - seq_idx = chunk_to_seq[i_c] — which segment this chunk belongs to.
    - last_seg_idx = scalar prefetch from launcher — the **REAL** last
      segment idx (largest i where ``cu_seqlens[i+1] > cu_seqlens[i]``).
      Computed in the launcher rather than reading ``chunk_to_seq[NT-1]``
      because ``prepare_chunk_indices``'s ``jnp.repeat(total_repeat_length=...)``
      pads trailing slots with the **last input value** (= the last
      trailing-padding seg id when ``cu_seqlens`` has max_num_segments padding),
      which would mis-identify the real last seg.
    - Only chunks of the LAST segment contribute to (S_ext, M); others
      no-op (BlockSpec DMA still happens but compute is skipped).

  Updates per chunk (last segment only):
    M_c    = Diag(exp2(gk_last_c)) - K_c^T @ W_c        # [MB, K, K]
    dS_c   = K_c^T @ U_c                                # [MB, K, V]
    h_acc  = M_c @ h_acc + dS_c                         # [MB, K, V]
    m_acc  = M_c @ m_acc                                # [MB, K, K]
  """
  i_c = pl.program_id(1)
  seq_idx = chunk_to_seq_ref[i_c]
  last_seg_idx = last_seg_idx_ref[0]
  bos = seqlens_ref[seq_idx]
  eos = seqlens_ref[seq_idx + 1]
  t0 = i_c * BT

  K = k_ref.shape[-1]
  V = u_ref.shape[-1]
  eye_k = jnp.eye(K, dtype=jnp.float32)

  # ── Init scratch at LAST segment's first chunk ──
  @pl.when((t0 == bos) & (seq_idx == last_seg_idx))
  def _init():
    h_scratch[...] = jnp.zeros((MB, K, V), dtype=jnp.float32)
    m_scratch[...] = jnp.broadcast_to(eye_k, (MB, K, K))

  # ── Update only on LAST segment chunks ──
  @pl.when(seq_idx == last_seg_idx)
  def _update():
    K_all = k_ref[:,0,:].astype(jnp.float32)                       # [MB, BT, K]
    W_all = w_ref[:,0,:].astype(jnp.float32)                       # [MB, BT, K]
    U_all = u_ref[:,0,:].astype(jnp.float32)                       # [MB, BT, V]
    gk_last_all = gk_ref[:, 0, BT - 1].astype(jnp.float32)     # [MB, K]

    # M_c per head: Diag(exp2(gk_last)) - K^T @ W
    decay_all = jnp.exp2(jnp.maximum(gk_last_all, -126.0))     # [MB, K]
    diag_all = decay_all[..., None] * eye_k                    # [MB, K, K]

    # K^T @ W: contract BT (axis 1), batch MB (axis 0) -> [MB, K, K]
    KW_all = jax.lax.dot_general(
      K_all, W_all,
      (((1,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
    )
    M_all = diag_all - KW_all                                  # [MB, K, K]

    # dS_c = K^T @ U: contract BT, batch MB -> [MB, K, V]
    dS_all = jax.lax.dot_general(
      K_all, U_all,
      (((1,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
    )

    # h_acc = M @ h_acc + dS: M [MB, K_out, K_in] @ h [MB, K_in, V] -> [MB, K_out, V]
    # contract K_in (M.axis 2 / h.axis 1), batch MB
    h_new = jax.lax.dot_general(
      M_all, h_scratch[...],
      (((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
    ) + dS_all                                                 # [MB, K, V]
    m_new = jax.lax.dot_general(
      M_all, m_scratch[...],
      (((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
    )                                                          # [MB, K, K]
    h_scratch[...] = h_new
    m_scratch[...] = m_new

  # ── Write outputs at LAST segment's last chunk ──
  @pl.when((t0 + BT >= eos) & (seq_idx == last_seg_idx))
  def _store():
    S_ext_ref[:,0,:] = h_scratch[...].astype(S_ext_ref.dtype)
    M_ref[:,0,:] = m_scratch[...].astype(M_ref.dtype)


def _pre_process_pallas(
  k: jax.Array,    # [H, B=1, T_local, K]
  w: jax.Array,    # [H, B=1, T_local, K]
  u: jax.Array,    # [H, B=1, T_local, V]
  gk: jax.Array,   # [H, B=1, T_local, K]
  cu_seqlens: jax.Array,
  chunk_indices: jax.Array,
  chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
  """Pallas launcher for (S_ext, M) pre-process. Returns shape
  ``([1, H, K, V], [1, H, K, K])`` both fp32.

  Assumes ``T_local`` is already a multiple of ``chunk_size`` (caller's
  ``_align_seqs`` has run) and ``chunk_indices = prepare_chunk_indices(...)``
  has been pre-computed by the caller.
  """
  H, B, T_local, K = k.shape
  V = u.shape[-1]
  BT = chunk_size

  # Handle B > 1 by looping over batch elements with the B=1 kernel.
  if B > 1:
    s_list, m_list = [], []
    for b in range(B):
      # chunk_indices: [B, NT, 2] → [NT, 2]
      ci_b = chunk_indices[b] if chunk_indices.ndim == 3 else chunk_indices
      cu_b = cu_seqlens[b] if cu_seqlens.ndim == 2 else cu_seqlens
      s_b, m_b = _pre_process_pallas(
        k=k[:, b:b+1], w=w[:, b:b+1], u=u[:, b:b+1], gk=gk[:, b:b+1],
        cu_seqlens=cu_b, chunk_indices=ci_b, chunk_size=BT,
      )
      s_list.append(s_b)
      m_list.append(m_b)
    return jnp.concatenate(s_list, axis=1), jnp.concatenate(m_list, axis=1)

  # B=1 path: squeeze cu_seqlens to 1D and chunk_indices to 2D if needed
  if cu_seqlens.ndim == 2:
    assert cu_seqlens.shape[0] == 1, f"B=1 but cu_seqlens has shape {cu_seqlens.shape}"
    cu_seqlens = cu_seqlens[0]
  if chunk_indices.ndim == 3:
    assert chunk_indices.shape[0] == 1, f"B=1 but chunk_indices has shape {chunk_indices.shape}"
    chunk_indices = chunk_indices[0]
  NT = len(chunk_indices)
  assert T_local % BT == 0
  assert K <= 256, "pre_process does not support K > 256"

  K_PADSIZE = int(align_up(K, 128))
  V_ALIGNED = int(align_up(V, 128))

  # ── Auto MB: maximise VMEM utilisation, capped at min(H, 16) ──
  # scratch per head: h_acc [K, V] + m_acc [K, K], all fp32.
  per_head_scratch = (K_PADSIZE * V_ALIGNED + K_PADSIZE * K_PADSIZE) * 4
  vmem_budget = 8 * 1024 * 1024
  MB = max(1, vmem_budget // per_head_scratch)
  MB = min(MB, H, 16)
  while H % MB != 0 and MB > 1:
    MB -= 1

  # ── Pad K/V then T (reserve trailing BT for safe gather) ──

  def _pad_kdim_then_t(x, dim_pad):
    if dim_pad > 0:
      x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, dim_pad)))
    return x  # [H, B, T_alloc, D]

  k_t = _pad_kdim_then_t(k.astype(jnp.float32), K_PADSIZE - K)
  w_t = _pad_kdim_then_t(w.astype(jnp.float32), K_PADSIZE - K)
  u_t = _pad_kdim_then_t(u.astype(jnp.float32), V_ALIGNED - V)
  gk_t = _pad_kdim_then_t(gk.astype(jnp.float32), K_PADSIZE - K)

  chunk_to_seq = chunk_indices[:, 0].astype(jnp.int32)

  # ── Real last seg idx (largest i where cu_seqlens[i+1] > cu_seqlens[i]) ──
  # MUST NOT derive from chunk_to_seq[NT-1] inside the kernel: when
  # cu_seqlens is max_num_segments-padded with trailing zero-length segments,
  # prepare_chunk_indices's `jnp.repeat(total_repeat_length=...)` pads
  # the trailing chunk_to_seq slots with the LAST input seg id (= the
  # trailing-padding seg id), which would point at a phantom segment.
  # Computing real_last_seg_idx here from cu_seqlens directly is robust
  # to any max_num_segments padding scheme.
  seg_lens = jnp.diff(cu_seqlens)  # [N]
  seg_indices = jnp.arange(seg_lens.shape[0], dtype=jnp.int32)
  real_last_seg_idx = jnp.maximum(
    jnp.max(jnp.where(seg_lens > 0, seg_indices, jnp.int32(-1))),
    jnp.int32(0),
  ).astype(jnp.int32)
  real_last_seg_idx_arr = real_last_seg_idx[None]  # [1] for scalar prefetch

  def _in_index_map(h, c, seqlens_ref, chunk_to_seq_ref, last_seg_idx_ref):
    return (h, 0, c, 0)

  def _out_index_map(h, c, seqlens_ref, chunk_to_seq_ref, last_seg_idx_ref):
    return (h, 0, 0, 0)

  bspec_k = pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_in_index_map)
  bspec_u = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_in_index_map)
  S_ext_spec = pl.BlockSpec(
    [MB, 1, K_PADSIZE, V_ALIGNED], index_map=_out_index_map
  )
  M_spec = pl.BlockSpec(
    [MB, 1, K_PADSIZE, K_PADSIZE], index_map=_out_index_map
  )

  S_ext_shape = jax.ShapeDtypeStruct(
    [H, 1, K_PADSIZE, V_ALIGNED], jnp.float32
  )
  M_shape = jax.ShapeDtypeStruct(
    [H, 1, K_PADSIZE, K_PADSIZE], jnp.float32
  )

  scratch_shapes = [
    pltpu.VMEM((MB, K_PADSIZE, V_ALIGNED), jnp.float32),  # h_acc
    pltpu.VMEM((MB, K_PADSIZE, K_PADSIZE), jnp.float32),  # m_acc
  ]
  grid = (H // MB, NT)
  interpret = get_interpret()

  S_ext_pad, M_pad = pl.pallas_call(
    functools.partial(_pre_process_kernel, BT=BT, MB=MB),
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=3,
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_u, bspec_k],
      out_specs=[S_ext_spec, M_spec],
      scratch_shapes=scratch_shapes,
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "arbitrary"),
    ),
    out_shape=[S_ext_shape, M_shape],
    interpret=interpret,
  )(cu_seqlens.astype(jnp.int32), chunk_to_seq, real_last_seg_idx_arr,
    k_t, w_t, u_t, gk_t)

  # Trim K/V padding back to original shape.
  S_ext = S_ext_pad[..., :K, :V]
  M = M_pad[..., :K, :K]
  return S_ext, M


@jaxtyping.jaxtyped
def chunk_gated_delta_rule_fwd_h_pre_process(
  k: Float[Array, "H B T_LOCAL K"],
  w: Float[Array, "H B T_LOCAL K"],
  u: Float[Array, "H B T_LOCAL V"],
  gk: Float[Array, "H B T_LOCAL K"],
  cu_seqlens: Int[Array, "B N_CU"] | Int[Array, "N_CU"],
  chunk_indices: (
      Int[Array, "B NT 2"] | Int[Array, "NT 2"] | None
  ) = None,
  chunk_size: int = 64,
  use_exp2: bool = True,
) -> tuple[Float[Array, "H B K V"], Float[Array, "H B K K"]]:
  """Builds the CP affine summary for the last rank-local segment.

  Only that segment can continue on the next rank. The fused kernel assumes a
  zero incoming state and returns the contribution and transition matrix used
  by `_merge_initial_state`. Inputs must be chunk-aligned in log2 gate space.
  """
  H, B, T_local, K = k.shape
  V = u.shape[-1]
  BT = chunk_size
  N_local = cu_seqlens.shape[-1] - 1

  assert use_exp2, "KDA pre-process requires use_exp2=True (gates are log2 space)"
  assert K <= 256, (
    "current pre-process does not support head dimension larger than 256."
  )
  assert T_local % BT == 0, (
    f"T_local={T_local} must be divisible by chunk_size={BT}; "
    f"caller must pre-align via _align_seqs."
  )
  assert N_local >= 1, (
    f"cu_seqlens must have at least 2 entries (one segment); got "
    f"shape {cu_seqlens.shape}"
  )

  if chunk_indices is None:
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT, max_T=T_local)

  S_ext, M = _pre_process_pallas(
    k=k, w=w, u=u, gk=gk,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    chunk_size=BT,
  )

  assert S_ext.dtype == jnp.float32, f"S_ext must be fp32, got {S_ext.dtype}"
  assert M.dtype == jnp.float32, f"M must remain fp32, got {M.dtype}"
  return S_ext, M


@jaxtyping.jaxtyped
def _prepare_cp_initial_state(
    *,
    kg: Float[Array, "H B T_LOCAL K"],
    w: Float[Array, "H B T_LOCAL K"],
    u: Float[Array, "H B T_LOCAL V"],
    gk: Float[Array, "H B T_LOCAL K"],
    cu_seqlens: Int[Array, "B N_CU"],
    chunk_indices: Int[Array, "B NT 2"],
    context_parallel_metadata: ContextParallelMetadata,
    chunk_size: int,
) -> jax.Array:
  """Builds the rank-local initial state for context parallel forward.

  Each rank summarizes its last local segment as an affine state transform,
  gathers those summaries across the CP axis, and merges the upstream ranks
  that continue into this rank's first segment. Only that first segment gets
  the merged state; all other rank-local segments start from zero.

  Args:
    kg: Chunk-aligned gated keys in log2 gate space.
    w: Chunk-aligned delta-rule weights.
    u: Chunk-aligned delta-rule values.
    gk: Chunk-aligned cumulative gates in log2 space.
    cu_seqlens: Batched chunk-aligned rank-local sequence boundaries.
    chunk_indices: Batched chunk mapping derived from ``cu_seqlens``.
    context_parallel_metadata: CP axis and derived rank-chain metadata.
    chunk_size: Kernel chunk size.

  Returns:
    Initial states with shape ``[B, N, H, K, V]`` in fp32.
  """
  H, B, _, K = kg.shape
  V = u.shape[-1]
  if cu_seqlens.ndim != 2 or cu_seqlens.shape[0] != B:
    raise ValueError(
        "cu_seqlens must have shape [B, N_CU] matching the input batch; "
        f"got {cu_seqlens.shape} for B={B}."
    )
  if (
      chunk_indices.ndim != 3
      or chunk_indices.shape[0] != B
      or chunk_indices.shape[-1] != 2
  ):
    raise ValueError(
        "chunk_indices must have shape [B, NT, 2] matching the input batch; "
        f"got {chunk_indices.shape} for B={B}."
    )
  N = cu_seqlens.shape[-1] - 1

  S_ext_local, M_local = chunk_gated_delta_rule_fwd_h_pre_process(
      k=kg,
      w=w,
      u=u,
      gk=gk,
      cu_seqlens=cu_seqlens,
      chunk_indices=chunk_indices,
      chunk_size=chunk_size,
      use_exp2=True,
  )
  S_ext_all, _ = all_gather_into_tensor(
      S_ext_local, context_parallel_metadata.axis_name
  )
  M_all, _ = all_gather_into_tensor(M_local, context_parallel_metadata.axis_name)
  rank = jax.lax.axis_index(context_parallel_metadata.axis_name)

  pre_num = context_parallel_metadata.pre_num_ranks
  is_first = context_parallel_metadata.is_first_rank
  assert pre_num is not None and is_first is not None
  if B > 1 and hasattr(pre_num, "ndim") and pre_num.ndim > 0:
    pre_num_arr = jnp.asarray(pre_num)
    is_first_arr = jnp.asarray(is_first)
    s_in_list = []
    for batch_index in range(B):
      s_in_list.append(
          _merge_initial_state(
              S_ext_all[:, :, batch_index : batch_index + 1],
              M_all[:, :, batch_index : batch_index + 1],
              rank,
              pre_num_arr[batch_index],
              is_first_arr[batch_index],
          )
      )
    s_in_first = jnp.concatenate(s_in_list, axis=1)
  else:
    s_in_first = _merge_initial_state(
        S_ext_all,
        M_all,
        rank,
        pre_num,
        is_first,
    )

  s_in_first = jnp.transpose(s_in_first, (1, 0, 2, 3))
  return (
      jnp.zeros((B, N, H, K, V), dtype=jnp.float32)
      .at[:, 0]
      .set(s_in_first)
  )


# =============================================================================
# Fused intra-chunk utilities
# =============================================================================

def _solve_unit_lower_triangular_batched(A, b):
  """Solve (I + A) x = b exactly for batched inputs.

  Uses block forward substitution with block size 16 for TPU MXU
  utilization.

  Args:
      A: (MB, N, N) strictly lower triangular matrix (float32).
      b: (MB, N, D) right-hand side matrix (float32).

  Returns:
      x: (MB, N, D) exact solution matrix.
  """
  MB, N, D = b.shape
  BS = 16
  num_blocks = N // BS
  A = A.astype(jnp.float32)
  b = b.astype(jnp.float32)

  blocks = [b[:, i * BS : (i + 1) * BS, :] for i in range(num_blocks)]

  for i in range(num_blocks):
    start = i * BS
    end = (i + 1) * BS

    A_ii = A[:, start:end, start:end]  # [MB, BS, BS]
    x_block = blocks[i]  # [MB, BS, D]

    rows = [x_block[:, r, :] for r in range(BS)]  # list of [MB, D]
    for j in range(BS):
      if j > 0:
        vec = A_ii[:, j, :j]  # [MB, j]
        mat = jnp.stack(rows[:j], axis=1)  # [MB, j, D]
        # [MB, 1, j] @ [MB, j, D] → [MB, 1, D] → squeeze
        correction = jnp.matmul(vec[:, None, :], mat).squeeze(1)
        rows[j] = rows[j] - correction

    x_block = jnp.stack(rows, axis=1)  # [MB, BS, D]
    blocks[i] = x_block

    if i < num_blocks - 1:
      x_rest = jnp.concatenate(blocks[i + 1 :], axis=1)  # [MB, rest, D]
      A_rest = A[:, (i + 1) * BS :, start:end]  # [MB, rest, BS]

      update = jnp.matmul(
        A_rest,
        x_block,
        preferred_element_type=jnp.float32,
      )
      x_rest = x_rest - update

      remaining = num_blocks - 1 - i
      for idx in range(remaining):
        blocks[i + 1 + idx] = x_rest[:, idx * BS : (idx + 1) * BS, :]

  return jnp.concatenate(blocks, axis=1)


def _solve_unit_lower_triangular_neumann_batched(A, b):
  """Solve (I + A) x = b with a blocked finite Neumann series.

  The strictly lower triangular system is split into 8-token diagonal blocks.
  Each diagonal block is inverted with a finite Neumann series, then the
  block-off-diagonal system is inverted with Horner doubling.

  Args:
      A: (MB, N, N) strictly lower triangular matrix.
      b: (MB, N, D) right-hand side matrix.

  Returns:
      A pair containing the solution with shape (MB, N, D) and the inverse of
      (I + A) with shape (MB, N, N), both in float32.
  """
  N = A.shape[-1]
  block_size = 8
  num_blocks = N // block_size
  inv_dtype = jnp.float32
  identity = jnp.eye(N, dtype=inv_dtype)

  # Batched dot: [MB, M, K] @ [MB, K, N] -> [MB, M, N].
  def dot_batch(lhs, rhs):
    return jax.lax.dot_general(
        lhs,
        rhs,
        (((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

  A = A.astype(inv_dtype)
  block_ids = jnp.arange(N, dtype=jnp.int32) // block_size
  same_block = (block_ids[:, None] == block_ids[None, :]).astype(inv_dtype)
  A_diag = A * same_block[None]
  A_off_diag = A - A_diag

  neg_A_diag = -A_diag
  diag_inverse = identity[None] + neg_A_diag
  power = neg_A_diag
  num_diag_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[block_size]
  for _ in range(num_diag_steps):
    power = dot_batch(power, power)
    diag_inverse = dot_batch(diag_inverse, identity[None] + power)

  if num_blocks == 1:
    return dot_batch(diag_inverse, b.astype(inv_dtype)), diag_inverse

  # Fuse P @ [F | b] to share a matmul, where P is the block-diagonal
  # inverse and F is the block-off-diagonal part of A.
  off_diag_and_rhs = jnp.concatenate(
      [A_off_diag, b.astype(inv_dtype)], axis=-1
  )
  merged = dot_batch(diag_inverse, off_diag_and_rhs)
  transformed_off_diag = merged[:, :, :N]
  transformed_rhs = merged[:, :, N:]

  # (I + G)^-1 = sum(k=0..num_blocks-1) (-G)^k. The block-strictly-lower
  # matrix G is nilpotent, so Horner doubling evaluates this finite series.
  neg_transformed_off_diag = -transformed_off_diag
  off_diag_inverse = identity[None] + neg_transformed_off_diag
  power = neg_transformed_off_diag
  log2_num_blocks = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[num_blocks]
  for _ in range(log2_num_blocks - 1):
    power = dot_batch(power, power)
    off_diag_inverse = off_diag_inverse + dot_batch(off_diag_inverse, power)

  result = dot_batch(off_diag_inverse, transformed_rhs)
  inverse = dot_batch(off_diag_inverse, diag_inverse)
  return result, inverse


# =============================================================================
# Fused gate cumsum and intra-chunk solve
# =============================================================================

def _fused_gate_intra_kernel(
  q_ref,
  k_ref,
  g_ref,
  beta_ref,
  v_ref,
  a_log_ref,
  delta_time_bias_ref,
  u_out_ref,
  w_out_ref,
  qg_out_ref,
  kg_out_ref,
  Aqk_out_ref,
  Akk_inv_out_ref,
  g_cumsum_out_ref,
  *,
  chunk_size: int,
  head_dim: int,
  value_dim: int,
  scale: float,
  cumsum_scale: float,
  fuse_cumsum: bool,
  disable_recompute: bool,
  safe_gate: bool,
  use_gate_in_kernel: bool,
  lower_bound: float | None,
  mini_batch: int = 1,
):
  """Fused Pallas kernel: gate activation + cumsum + BC=16 Aqk/L + Neumann inversion.

  When fuse_cumsum=True, g_ref contains raw gate values (not cumsum'd).
  The kernel applies gate activation (if use_gate_in_kernel) and chunk-local
  prefix sum via tril matmul before proceeding to the Aqk/L construction.

  For bfloat16 inputs, uses Neumann series inversion.
  For float32 inputs, falls back to exact forward substitution.

  All refs have leading dims from BlockSpec: [1, MB, 1, BT, D].
  a_log_ref: [1, MB, 1, 1, 1] — per-head scalar.
  delta_time_bias_ref: [1, MB, 1, 1, K] — per-head bias vector.

  MB heads are processed simultaneously via batch-vectorized ops.

  Args:
      mini_batch: int — number of heads processed per grid point (MB).
  """
  dtype = q_ref.dtype
  BT = chunk_size
  BC = 16
  NC = BT // BC
  K = head_dim
  V = value_dim
  MB = mini_batch

  # Load all MB heads at once
  q = q_ref[:, 0, 0]        # [MB, BT, K]
  k = k_ref[:, 0, 0]        # [MB, BT, K]
  g = g_ref[:, 0, 0]        # [MB, BT, K]
  beta = beta_ref[:, 0, 0]  # [MB, BT, 1]
  v = v_ref[:, 0, 0]        # [MB, BT, V]

  # --- Gate activation + cumsum ---
  g_f32 = g.astype(jnp.float32)

  if use_gate_in_kernel:
    dt_b = delta_time_bias_ref[:, 0, 0, 0]        # [MB, K]
    g_f32 = g_f32 + dt_b[:, None, :]       # [MB, BT, K]
    A_val = a_log_ref[:, 0, 0, 0, 0]       # [MB]
    if lower_bound is None:
      g_f32 = -jnp.exp(A_val)[:, None, None] * jax.nn.softplus(g_f32)
    else:
      g_f32 = lower_bound * jax.nn.sigmoid(
        jnp.exp(A_val)[:, None, None] * g_f32
      )

  if fuse_cumsum:
    tril = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))
    # tril[BT,BT] @ g_f32[MB,BT,K] → [MB,BT,K]: contract on BT, batch on MB
    g_cumsum = jax.lax.dot_general(
      tril, g_f32, (((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
    ).transpose(1, 0, 2) * cumsum_scale  # [BT,MB,K] → [MB,BT,K]
  else:
    g_cumsum = g_f32

  q_f32 = q.astype(jnp.float32)
  k_f32 = k.astype(jnp.float32)
  beta_f32 = beta.astype(jnp.float32)

  # --- BC=16 sub-block factored Aqk/L (fla safe-gate style, j-batched) ---
  # Per i_sc, batch the j_sc loop into a single (MB, 2*BC, K) x (MB, BT, K) matmul:
  #   q_eg[m,r,k]   = q_i[m,r,k] * exp2(g_i[m,r,k] - gn_ref[m,k])
  #   k_eg[m,r,k]   = k_i[m,r,k] * exp2(g_i[m,r,k] - gn_ref[m,k])    (Akk left)
  #   k_eng[m,t,k]  = k[m,t,k]   * exp2(gn_ref[m,k] - g_cumsum[m,t,k]) (full BT)
  # so Aqk[m,r,t]  = sum_k q_i[m,r,k] * k[m,t,k] * exp2(g_i[m,r,k] - g_cumsum[m,t,k]).
  # Stack q_eg and k_eg along row dim and do one matmul -> split Aqk/Akk.
  #
  # Anti-causal rows (t >= (i_sc+1)*BC) have positive `gn_ref - g_cumsum[t]`
  # for negative-gate cumsum, which can overflow exp2. Mask those diffs to 0
  # BEFORE exp2, then zero the resulting rows so they contribute 0 to matmul.
  #
  # Use broadcasted_iota for 2D indices to avoid Mosaic-unsupported i1
  # shape casts (e.g. (BT,) -> (BT, 1) on bool tensors).
  ref_idx = BC // 2 if safe_gate else 0
  row_iota_bt_k = jax.lax.broadcasted_iota(jnp.int32, (BT, K), dimension=0)
  row_iota_bc_bt = jax.lax.broadcasted_iota(jnp.int32, (BC, BT), dimension=0)
  col_iota_bc_bt = jax.lax.broadcasted_iota(jnp.int32, (BC, BT), dimension=1)

  Aqk_rows = []
  L_rows = []
  for i_sc in range(NC):
    i_s = i_sc * BC
    q_i = q_f32[:, i_s : i_s + BC]       # [MB, BC, K]
    k_i = k_f32[:, i_s : i_s + BC]       # [MB, BC, K]
    g_i = g_cumsum[:, i_s : i_s + BC]    # [MB, BC, K]
    beta_i = beta_f32[:, i_s : i_s + BC]  # [MB, BC, 1]

    gn_ref = g_i[:, ref_idx : ref_idx + 1, :]   # [MB, 1, K] — query sub-block ref
    diff_i = g_i - gn_ref                        # [MB, BC, K], <= 0
    exp_diff_i = jnp.exp2(diff_i)
    q_eg = q_i * exp_diff_i     # [MB, BC, K]
    k_eg = k_i * exp_diff_i     # [MB, BC, K]

    # j-side: full BT rows. Mask anti-causal rows BEFORE exp2 (avoids
    # overflow when gn_ref - g_cumsum[t] is large positive). The 2D mask
    # is built from broadcasted_iota to bypass the i1 reshape limitation.
    valid_j = (row_iota_bt_k < (i_s + BC)).astype(jnp.float32)   # [BT, K]
    diff_j_safe = (gn_ref - g_cumsum) * valid_j[None]              # [MB, BT, K]
    k_eng_full = k_f32 * jnp.exp2(diff_j_safe) * valid_j[None]    # [MB, BT, K]

    # Stack q_eg / k_eg and do one matmul: [MB, 2*BC, K] x [MB, K, BT] -> [MB, 2*BC, BT]
    qk_eg = jnp.concatenate([q_eg, k_eg], axis=1)  # [MB, 2*BC, K]
    qk_dot = jax.lax.dot_general(
      qk_eg, k_eng_full, (((2,), (2,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
    )  # [MB, 2*BC, BT]
    Aqk_row = qk_dot[:, :BC] * scale      # [MB, BC, BT]
    Akk_row = qk_dot[:, BC:] * beta_i     # [MB, BC, BT]

    # Apply diagonal-block masks (causal for Aqk, strict-lower for Akk).
    # All masks built from 2D iota; no i1 shape casts needed.
    in_diag = (col_iota_bc_bt >= i_s) & (col_iota_bc_bt < i_s + BC)  # [BC, BT]
    col_local = col_iota_bc_bt - i_s                                   # [BC, BT]
    causal_diag = row_iota_bc_bt >= col_local                          # [BC, BT]
    strict_diag = row_iota_bc_bt > col_local                           # [BC, BT]
    aqk_keep = (~in_diag) | causal_diag
    akk_keep = (~in_diag) | strict_diag
    Aqk_row = jnp.where(aqk_keep[None], Aqk_row, jnp.float32(0.0))
    Akk_row = jnp.where(akk_keep[None], Akk_row, jnp.float32(0.0))

    Aqk_rows.append(Aqk_row)
    L_rows.append(Akk_row)

  Aqk = jnp.concatenate(Aqk_rows, axis=1).astype(dtype)  # [MB, BT, BT]
  L = jnp.concatenate(L_rows, axis=1)                     # [MB, BT, BT]

  # --- Solve (I + L) x = rhs ---
  v_beta = v.astype(jnp.float32) * beta_f32          # [MB, BT, V]
  k_eg_beta = k_f32 * jnp.exp2(g_cumsum) * beta_f32  # [MB, BT, K]
  rhs = jnp.concatenate([v_beta, k_eg_beta], axis=-1)  # [MB, BT, V+K]

  use_neumann = dtype != jnp.float32

  if use_neumann:
    result, A_inv = _solve_unit_lower_triangular_neumann_batched(L, rhs)
  else:
    # --- Exact forward substitution (fp32) ---
    I_bt = jnp.eye(BT, dtype=jnp.float32)
    I_bt_batch = jnp.broadcast_to(I_bt, (MB, BT, BT))
    combined_b = jnp.concatenate(
      [rhs, I_bt_batch], axis=-1
    )  # [MB, BT, V+K+BT]
    full_result = _solve_unit_lower_triangular_batched(L, combined_b)
    result = full_result[:, :, : V + K]
    A_inv = full_result[:, :, V + K :]

  u = result[:, :, :V]         # [MB, BT, V]
  w = result[:, :, V : V + K]  # [MB, BT, K]

  # --- kg, qg ---
  g_last = g_cumsum[:, BT - 1 : BT, :]  # [MB, 1, K]
  kg = k_f32 * exp2(g_last - g_cumsum)
  qg = q_f32 * exp2(g_cumsum) if disable_recompute else jnp.zeros_like(q_f32)

  # --- Store all MB heads ---
  u_out_ref[:, 0, 0] = u.astype(u_out_ref.dtype)
  w_out_ref[:, 0, 0] = w.astype(w_out_ref.dtype)
  qg_out_ref[:, 0, 0] = qg.astype(qg_out_ref.dtype)
  kg_out_ref[:, 0, 0] = kg.astype(kg_out_ref.dtype)
  Aqk_out_ref[:, 0, 0] = Aqk.astype(Aqk_out_ref.dtype)
  Akk_inv_out_ref[:, 0, 0] = A_inv.astype(Akk_inv_out_ref.dtype)
  g_cumsum_out_ref[:, 0, 0] = g_cumsum


def _compute_intra_fused_mini_batch(H, BT, K, V, dtype=None):
  """Auto-compute mini-batch size for intra fused kernel.

  Each head needs VMEM for intermediates: BT*K + BT*V + 2*BT*BT (Aqk, L).
  Uses hardware VMEM budget via ``estimate_mini_batch``.
  """
  elem_size = dtype.itemsize if dtype is not None else 4
  per_head = (BT * K + BT * V + 2 * BT * BT) * elem_size
  return estimate_mini_batch(per_head, H, max_mb=16)


@functools.partial(
  jax.jit,
  static_argnames=[
    "chunk_size",
    "scale",
    "safe_gate",
    "disable_recompute",
    "cumsum_scale",
    "use_gate_in_kernel",
    "lower_bound",
    "mini_batch",
  ],
)
@jaxtyping.jaxtyped
def pallas_kda_fwd_intra_fused(
  q: Float[Array, "H B T K"],
  k: Float[Array, "H B T K"],
  v: Float[Array, "H B T V"],
  g: Float[Array, "H B T K"],
  beta: Float[Array, "H B T"],
  scale: float,
  chunk_size: int = 64,
  safe_gate: bool = True,
  disable_recompute: bool = False,
  cumsum_scale: float = RCP_LN2,
  a_log: Float[Array, "H"] | None = None,
  delta_time_bias: Float[Array, "H*K"] | None = None,
  use_gate_in_kernel: bool = False,
  lower_bound: float | None = None,
  mini_batch: int | None = None,
) -> tuple[
    Float[Array, "H B T K"],
    Float[Array, "H B T V"],
    Float[Array, "H B T K"] | None,
    Float[Array, "H B T K"],
    Float[Array, "H B T BT"],
    Float[Array, "H B T BT"],
    Float[Array, "H B T K"],
]:
  """Fuses gate cumsum with the fixed-length intra-chunk solve.

  Heads are mini-batched to amortize DMA. `qg` is retained only when
  `disable_recompute` is true.
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  NC = T // BT

  if use_gate_in_kernel:
    assert a_log is not None, "a_log required when use_gate_in_kernel=True"

  if mini_batch is None:
    MB = _compute_intra_fused_mini_batch(H, BT, K, V, dtype=q.dtype)
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  # [H, B, T, K] -> [H, B, NC, BT, K]
  q_r = q.reshape(H, B, NC, BT, K)
  k_r = k.reshape(H, B, NC, BT, K)
  g_r = g.reshape(H, B, NC, BT, K)
  beta_r = beta.reshape(H, B, NC, BT, 1)
  v_r = v.reshape(H, B, NC, BT, V)

  if use_gate_in_kernel:
    assert a_log is not None
    a_log_r = a_log.astype(jnp.float32).reshape(H, 1, 1, 1, 1)
    if delta_time_bias is not None:
      delta_time_bias_r = delta_time_bias.astype(jnp.float32).reshape(H, 1, 1, 1, K)
    else:
      delta_time_bias_r = jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)
  else:
    a_log_r = jnp.zeros((H, 1, 1, 1, 1), dtype=jnp.float32)
    delta_time_bias_r = jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)

  grid = (H // MB, B, NC)

  def _make_spec(last_dim):
    return pl.BlockSpec(
      index_map=lambda i, j, l: (i, j, l, 0, 0),
      block_shape=(MB, 1, 1, BT, last_dim),
    )

  def _make_per_head_spec(last_dim):
    return pl.BlockSpec(
      index_map=lambda i, j, l: (i, 0, 0, 0, 0),
      block_shape=(MB, 1, 1, 1, last_dim),
    )

  (u_r, w_r, qg_r, kg_r, Aqk_r, Akk_inv_r, g_cumsum_r) = pl.pallas_call(
    functools.partial(
      _fused_gate_intra_kernel,
      chunk_size=BT,
      head_dim=K,
      value_dim=V,
      scale=scale,
      cumsum_scale=cumsum_scale,
      fuse_cumsum=True,
      disable_recompute=disable_recompute,
      safe_gate=safe_gate,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
      mini_batch=MB,
    ),
    interpret=get_interpret(),
    out_shape=[
      jax.ShapeDtypeStruct((H, B, NC, BT, V), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), jnp.float32),
    ],
    in_specs=[
      _make_spec(K),
      _make_spec(K),
      _make_spec(K),
      _make_spec(1),
      _make_spec(V),
      _make_per_head_spec(1),
      _make_per_head_spec(K),
    ],
    out_specs=[
      _make_spec(V),
      _make_spec(K),
      _make_spec(K),
      _make_spec(K),
      _make_spec(BT),
      _make_spec(BT),
      _make_spec(K),
    ],
    grid=grid,
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "parallel"),
    ),
  )(q_r, k_r, g_r, beta_r, v_r, a_log_r, delta_time_bias_r)

  # --- Reshape back to [H, B, T, D] (head-first) ---
  w_out = w_r.reshape(H, B, T, K)
  u_out = u_r.reshape(H, B, T, V)
  kg_out = kg_r.reshape(H, B, T, K)
  qg_out = (
    qg_r.reshape(H, B, T, K)
    if disable_recompute else None
  )
  Aqk_flat = Aqk_r.reshape(H, B, NC * BT, BT)
  Akk_flat = Akk_inv_r.reshape(H, B, NC * BT, BT)
  g_cumsum_out = g_cumsum_r.reshape(H, B, T, K)

  return w_out, u_out, qg_out, kg_out, Aqk_flat, Akk_flat, g_cumsum_out


def kda_fwd_intra_fused(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array,
  beta: jax.Array,
  scale: float,
  cu_seqlens: jax.Array | None = None,
  chunk_size: int = 64,
  chunk_indices: jax.Array | None = None,
  safe_gate: bool = True,
  disable_recompute: bool = False,
  cumsum_scale: float = RCP_LN2,
  a_log: jax.Array | None = None,
  delta_time_bias: jax.Array | None = None,
  use_gate_in_kernel: bool = False,
  lower_bound: float | None = None,
):
  """Fused gate cumsum + intra-chunk solve entry.

  Args:
      g: [H, B, T, K] -- raw gate input (NOT cumsum'd).
      Other args configure the intra-chunk solve and gate activation.

  Returns:
      7-tuple: (w, u, qg, kg, Aqk, Akk, g_cumsum).
  """
  assert chunk_size == 64, f"Expected chunk_size=64, got {chunk_size}"

  # The caller has already BT-aligned varlen inputs, so the same contiguous
  # fused kernel handles both fixed-length and variable-length batches.
  return pallas_kda_fwd_intra_fused(
    q=q, k=k, v=v, g=g, beta=beta,
    scale=scale, chunk_size=chunk_size,
    safe_gate=safe_gate, disable_recompute=disable_recompute,
    cumsum_scale=cumsum_scale,
    a_log=a_log, delta_time_bias=delta_time_bias,
    use_gate_in_kernel=use_gate_in_kernel,
    lower_bound=lower_bound,
  )


# =============================================================================
# Fused state propagation and output
# =============================================================================

def _chunk_kda_fwd_h_o_varlen_kernel(
  seqlens_ref,       # scalar prefetch: cu_seqlens [N+1]
  chunk_to_seq_ref,  # scalar prefetch: chunk -> seq mapping [NT]
  # Stage 3 inputs
  w_ref,      # [MB, 1, BT, K_PADSIZE]
  u_ref,      # [MB, 1, BT, V_ALIGNED]
  kg_ref,     # [MB, 1, BT, K_PADSIZE]
  gk_ref,     # [MB, 1, BT, K_PADSIZE]  -- g_cumsum
  # Stage 4 inputs
  q_ref,      # [MB, 1, BT, K_PADSIZE]
  A_ref,      # [MB, 1, BT, BT]
  # Optional initial state
  h0_ref,     # [1, MB, K_PADSIZE, V_ALIGNED] or None
  # Outputs
  o_ref,      # [MB, 1, BT, V_ALIGNED]
  ht_ref,     # [1, MB, K_PADSIZE, V_ALIGNED] or None
  h_out_ref,    # [MB, 1, 1, K_PADSIZE, V_ALIGNED] or None  -- per-chunk pre-update h
  v_new_out_ref,  # [MB, 1, BT, V_ALIGNED] or None         -- per-token v_new
  # Scratch
  scratch_ref,  # [MB, K_PADSIZE, V_ALIGNED]
  *,
  BT,
  scale,
  USE_INITIAL_STATE,
  STORE_FINAL_STATE,
  STORE_H,
  STORE_V_NEW,
  MB,
  OUTPUT_PRECISION,
):
  """Fused Stage 3+4 Pallas kernel body for varlen with H-dim mini-batch.

  Grid is (H // MB, B, NT). For each program point (h_group, i_b, i_c):
    - MB heads [h_group*MB .. h_group*MB+MB) are processed per grid point
      via batched matmuls over the MB dimension.
    - i_b is the batch index, i_c is the chunk index within that batch.
    - seq_idx = chunk_to_seq[i_b, i_c] identifies which sequence this chunk
      belongs to within batch i_b.
    - At t0 == bos: init scratch (h0 or zeros) for this sequence
    - At t0 + BT >= eos: store final state for this sequence
  """
  i_b = pl.program_id(1)
  i_c = pl.program_id(2)
  seq_idx = chunk_to_seq_ref[i_b, i_c]

  bos = seqlens_ref[i_b, seq_idx]
  eos = seqlens_ref[i_b, seq_idx + 1]
  t0 = i_c * BT

  K = w_ref.shape[3]
  V = u_ref.shape[3]

  # === Init state (first chunk of THIS sequence) — uniform across all MB heads ===
  @pl.when(t0 == bos)
  def _():
    scratch_ref[:] = jnp.zeros([MB, K, V], dtype=jnp.float32)
    if USE_INITIAL_STATE:
      scratch_ref[:] = h0_ref[0, 0].astype(jnp.float32)  # [MB, K, V]

  # === Stage 3+4 work — batched over MB heads ===
  # h: pre-update state for all MB heads in this tile.
  b_h = scratch_ref[:]   # [MB, K, V]

  # Spill pre-update h (saved residual for bwd; tagged with
  # checkpoint_name("kda_residuals") at the custom_vjp boundary).
  if STORE_H:
    h_out_ref[:, 0, 0] = b_h.astype(h_out_ref.dtype)  # [MB, K, V]

  b_w = w_ref[:,0,:]   # [MB, BT, K]
  b_u = u_ref[:,0,:]   # [MB, BT, V]

  # Stage 3 delta correction: v_new = u - w @ h
  # [MB, BT, K] @ [MB, K, V] -> [MB, BT, V]
  # HIGHEST precision: v_new feeds directly into the recursive state update.
  b_v_new = b_u.astype(jnp.float32) - jnp.matmul(
    b_w.astype(jnp.float32), b_h,
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )  # [MB, BT, V]

  # Spill v_new (used by bwd Stage 0 when disable_recompute=True).
  if STORE_V_NEW:
    v_new_out_ref[:,0,:] = b_v_new.astype(v_new_out_ref.dtype)

  # Stage 4 inter-chunk output: scale * (q * exp2(g - g_ref)) @ (h * exp2(g_ref))
  # Reference-point stabilization mirrors the production varlen kernel's
  # g_ref = g[0] choice for bit-identical numerics.
  b_q = q_ref[:,0,:]                              # [MB, BT, K]
  b_g = gk_ref[:,0,:].astype(jnp.float32)         # [MB, BT, K]
  b_A = A_ref[:,0,:]                               # [MB, BT, BT]

  b_g_ref_row = b_g[:, 0:1, :]                # [MB, 1, K]
  b_qg = b_q.astype(jnp.float32) * jnp.exp2(
    jnp.maximum(b_g - b_g_ref_row, -126.0)
  )                                            # [MB, BT, K]
  b_h_scaled = b_h * jnp.exp2(
    jnp.maximum(b_g_ref_row[:, 0, :], -126.0)
  )[:, :, None]                                # [MB, K, V]

  # [MB, BT, K] @ [MB, K, V] -> [MB, BT, V]
  # Output-only GEMM: bf16 inputs use DEFAULT (no extra precision to preserve),
  # fp32 inputs use HIGHEST to maintain full mantissa fidelity.
  b_o = jnp.matmul(
    b_qg, b_h_scaled,
    precision=OUTPUT_PRECISION,
    preferred_element_type=jnp.float32,
  ) * scale                                    # [MB, BT, V]

  # Stage 4 intra-chunk: A @ v_new
  # Apply lower-triangular mask: Aqk is causal by construction, but masking
  # here matches the original per-head loop behaviour and guards against
  # any tiny upper-triangle fp noise from the Neumann intra-chunk solve.
  m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]  # [BT, BT]
  b_A_f32 = jnp.where(m_s[None, :, :], b_A.astype(jnp.float32), 0.0)
  # [MB, BT, BT] @ [MB, BT, V] -> [MB, BT, V]
  b_o = b_o + jnp.matmul(
    b_A_f32, b_v_new,
    precision=OUTPUT_PRECISION,
    preferred_element_type=jnp.float32,
  )                                            # [MB, BT, V]

  o_ref[:,0,:] = b_o.astype(o_ref.dtype)

  # Stage 3 state update: h = decay(h) + kg^T @ v_new
  # HIGHEST precision: accumulates directly into the recursive hidden state.
  b_gk_last = gk_ref[:,0,:][:, BT - 1, :].astype(jnp.float32)  # [MB, K]
  b_h_new = b_h * jnp.exp2(b_gk_last)[:, :, None]          # [MB, K, V] decay

  b_kg = kg_ref[:,0,:]   # [MB, BT, K]
  # [MB, K, BT] @ [MB, BT, V] -> [MB, K, V]
  b_h_new = b_h_new + jnp.matmul(
    b_kg.astype(jnp.float32).transpose(0, 2, 1), b_v_new,
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )
  scratch_ref[:] = b_h_new

  # === Final state (last chunk of THIS sequence) ===
  @pl.when(t0 + BT >= eos)
  def _():
    if STORE_FINAL_STATE:
      ht_ref[0, 0] = scratch_ref[:].astype(ht_ref.dtype)  # [MB, K, V]


@functools.partial(
  jax.jit,
  static_argnames=[
    "output_final_state",
    "scale",
    "chunk_size",
    "store_h",
    "store_v_new",
    "store_intermediates",
    "mini_batch",
  ],
)
@jaxtyping.jaxtyped
def chunk_kda_fwd_h_o_varlen(
  w: Float[Array, "H B T K"],
  u: Float[Array, "H B T V"],
  kg: Float[Array, "H B T K"],
  gk: Float[Array, "H B T K"],
  q: Float[Array, "H B T K"],
  A: Float[Array, "H B T BT"],
  cu_seqlens: Int[Array, "N_CU"] | Int[Array, "B N_CU"],
  chunk_indices: Int[Array, "NT 2"] | Int[Array, "B NT 2"] | None = None,
  initial_state: Float[Array, "B N H K V"] | None = None,
  output_final_state: bool = False,
  scale: float = 1.0,
  chunk_size: int = 64,
  store_h: bool = False,
  store_v_new: bool = False,
  store_intermediates: bool | None = None,
  mini_batch: int | None = None,
) -> tuple[
    Float[Array, "H B T V"],
    Float[Array, "B N H K V"] | None,
    Float[Array, "H B NT K V"] | None,
    Float[Array, "H B T V"] | None,
]:
  """Fuses varlen state propagation with output projection.

  The recurrent state remains in VMEM. `store_h` and `store_v_new` spill
  only the intermediates needed by backward; `store_intermediates` is the
  legacy alias that enables both.
  """
  # Back-compat shim: old single-flag callers map to both stores.
  if store_intermediates is not None:
    store_h = store_h or store_intermediates
    store_v_new = store_v_new or store_intermediates
  H, B, T, K = q.shape
  V = u.shape[-1]
  BT = chunk_size

  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  # Ensure cu_seqlens is 2D [B, N+1] for kernel block specs
  if cu_seqlens.ndim == 1:
    cu_seqlens = jnp.broadcast_to(cu_seqlens[None, :], (B, cu_seqlens.shape[0]))
  assert A.shape[-1] == BT, (
      f"A.shape[-1]={A.shape[-1]} must equal chunk_size={BT}"
  )

  N = cu_seqlens.shape[-1] - 1
  if initial_state is not None:
    assert initial_state.shape[1] == N, (
        f"initial_state has N={initial_state.shape[1]}, expected {N}"
    )
  assert K <= 256, "current kernel does not support K > 256."

  hw = get_tpu_limits()
  K_PADSIZE = int(align_up(K, hw.block_align_major))
  V_ALIGNED = int(align_up(V, hw.block_align_major))

  # ---- auto-compute mini-batch (MB) to maximise VMEM utilisation ----
  if mini_batch is None:
    elem_size = q.dtype.itemsize
    # scratch per head: h_state[K_PADSIZE*V_ALIGNED] in f32 + i/o buffers
    in_bytes = (BT * K_PADSIZE + BT * V_ALIGNED + BT) * elem_size
    out_bytes = BT * V_ALIGNED * elem_size
    scratch_bytes = K_PADSIZE * V_ALIGNED * 4  # float32 accumulator
    per_head = in_bytes + out_bytes + scratch_bytes
    MB = estimate_mini_batch(per_head, H, max_mb=16)
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  # Generate chunk_to_seq mapping from chunk_indices
  if chunk_indices is None:
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
  NT = chunk_indices.shape[-2]
  chunk_to_seq = chunk_indices[..., 0].astype(jnp.int32)  # [NT] or [B, NT]
  # Kernel block specs index with [b, c], so ensure 2D [B, NT].
  if chunk_to_seq.ndim == 1:
    chunk_to_seq = jnp.broadcast_to(chunk_to_seq[None, :], (B, NT))

  # grid = (H // MB, NT) with NT = T // BT, so chunk index c ranges
  # 0..NT-1 and the maximum T offset accessed is (NT-1)*BT+BT = T.
  # No extra trailing chunk is ever touched, so T_alloc == T suffices.
  T_alloc = T

  # Pad K (last dim) to K_PADSIZE if needed, then transpose to
  # [B=1, H, T, K_PADSIZE]. No T-dim padding required (see T_alloc above).
  # Inputs stay in their original dtype (typically bf16); the kernel casts
  # tile-level slices to f32 on the fly in VMEM, avoiding a full-tensor
  # bf16->f32 cast in HBM that doubles memory and DMA bandwidth.
  def _pad_kdim_then_t(x, dim_pad):
    if dim_pad > 0:
      x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, dim_pad)))
    return x

  w_t  = _pad_kdim_then_t(w,  K_PADSIZE - K)
  kg_t = _pad_kdim_then_t(kg, K_PADSIZE - K)
  gk_t = _pad_kdim_then_t(gk, K_PADSIZE - K)
  q_t  = _pad_kdim_then_t(q,  K_PADSIZE - K)
  u_t  = _pad_kdim_then_t(u,  V_ALIGNED - V)

  # A is [B, T, H, BT]; transpose to [B=1, H, T, BT]. No T padding needed.
  A_t = A  # [H, B, T, BT]

  # h0: [B, N, H, K, V] -> pad K and V -> [B, N, H, K_PADSIZE, V_ALIGNED]
  if initial_state is not None:
    h0 = initial_state
    if V_ALIGNED > V:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
    if K_PADSIZE > K:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
  else:
    h0 = None

  # Index maps with MB heads per grid point. Scalar prefetch order:
  # (seqlens_ref, chunk_to_seq_ref).
  # Inputs are [H, B, T_alloc, X]; BlockSpec slices MB heads at h*MB.
  def _t_index_map(h, b, c, seqlens_ref, chunk_to_seq_ref):
    return (h, b, c, 0)

  def _A_index_map(h, b, c, seqlens_ref, chunk_to_seq_ref):
    return (h, b, c, 0)

  bspec_k = pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_t_index_map)
  bspec_v = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
  bspec_a = pl.BlockSpec([MB, 1, BT, BT],        index_map=_A_index_map)
  bspec_h0 = (
    pl.BlockSpec(
      [1, 1, MB, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (
        b, chunk_to_seq_ref[b, c], h, 0, 0
      ),
    )
    if h0 is not None else None
  )

  # Output specs.
  o_spec = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
  ht_spec = (
    pl.BlockSpec(
      [1, 1, MB, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (
        b, chunk_to_seq_ref[b, c], h, 0, 0
      ),
    )
    if output_final_state else None
  )
  # Per-chunk h spill (pre-update, used by bwd save-h fast path). Layout
  # [H, B, NT, K_PADSIZE, V_ALIGNED]
  h_out_spec = (
    pl.BlockSpec(
      [MB, 1, 1, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (
        h, b, c, 0, 0
      ),
    )
    if store_h else None
  )
  # Per-token v_new spill, [H, B, T_alloc, V_ALIGNED].
  v_new_out_spec = (
    pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
    if store_v_new else None
  )

  o_shape = jax.ShapeDtypeStruct([H, B, T_alloc, V_ALIGNED], jnp.float32)
  ht_shape = (
    jax.ShapeDtypeStruct([B, N, H, K_PADSIZE, V_ALIGNED], jnp.float32)
    if output_final_state else None
  )
  # h_per_chunk dtype mirrors u (matches unfused chunk_gated_delta_rule_fwd_h
  # which produces h in u.dtype, typically bf16/fp32).
  h_out_shape = (
    jax.ShapeDtypeStruct([H, B, NT, K_PADSIZE, V_ALIGNED], u.dtype)
    if store_h else None
  )
  v_new_out_shape = (
    jax.ShapeDtypeStruct([H, B, T_alloc, V_ALIGNED], u.dtype)
    if store_v_new else None
  )

  scratch = pltpu.VMEM((MB, K_PADSIZE, V_ALIGNED), jnp.float32)
  grid = (H // MB, B, NT)
  interpret = get_interpret()

  # bf16 inputs: DEFAULT precision is lossless (operands already have ~7-bit
  # mantissa); fp32 inputs: HIGHEST preserves full 23-bit mantissa fidelity.
  # State-update matmuls always use HIGHEST regardless (recursive accumulation).
  _output_prec = (
    jax.lax.Precision.DEFAULT
    if q.dtype == jnp.bfloat16
    else jax.lax.Precision.HIGHEST
  )

  o_out, ht_out, h_out, v_new_out = pl.pallas_call(
    functools.partial(
      _chunk_kda_fwd_h_o_varlen_kernel,
      BT=BT,
      scale=scale,
      USE_INITIAL_STATE=(h0 is not None),
      STORE_FINAL_STATE=output_final_state,
      STORE_H=store_h,
      STORE_V_NEW=store_v_new,
      MB=MB,
      OUTPUT_PRECISION=_output_prec,
    ),
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=2,
      grid=grid,
      in_specs=[
        bspec_k,   # w
        bspec_v,   # u
        bspec_k,   # kg
        bspec_k,   # gk
        bspec_k,   # q
        bspec_a,   # A
        bspec_h0,  # h0
      ],
      out_specs=[o_spec, ht_spec, h_out_spec, v_new_out_spec],
      scratch_shapes=[scratch],
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "arbitrary"),
      disable_bounds_checks=True,
    ),
    out_shape=[o_shape, ht_shape, h_out_shape, v_new_out_shape],
    interpret=interpret,
  )(cu_seqlens.astype(jnp.int32), chunk_to_seq, w_t, u_t, kg_t, gk_t, q_t, A_t, h0)

  # Post-process: o is [H, 1, T, V_ALIGNED]
  if V_ALIGNED > V:
    o_out = o_out[..., :V]
  o_out = o_out.astype(u.dtype)

  if output_final_state and ht_out is not None:
    if V_ALIGNED > V:
      ht_out = ht_out[..., :V]
    if K_PADSIZE > K:
      ht_out = ht_out[..., :K, :]

    # Handle empty sequences: sequences with no chunks never execute kernel code,
    # so their final_state is uninitialized. Fill them with initial_state or zeros.
    seq_lens = jnp.diff(cu_seqlens, axis=-1)
    empty_mask = (seq_lens == 0)  # [B, N]
    if initial_state is not None:
      # For empty sequences, final_state should equal initial_state
      fill_value = initial_state[:, :, :, :K, :V]
    else:
      # For empty sequences without initial_state, final_state should be zeros
      fill_value = jnp.zeros((B, N, H, K, V), dtype=ht_out.dtype)
    # Use where to selectively replace empty sequence states
    ht_out = jnp.where(empty_mask[:, :, None, None, None], fill_value, ht_out)
  else:
    ht_out = None

  # Post-process spilled intermediates (independently controlled).
  if store_h:
    # h_out: [1, NT, H, K_PADSIZE, V_ALIGNED] -> trim K/V padding back.
    if V_ALIGNED > V:
      h_out = h_out[..., :V]
    if K_PADSIZE > K:
      h_out = h_out[..., :K, :]

  else:
    h_out = None
  if store_v_new:
    if V_ALIGNED > V:
      v_new_out = v_new_out[..., :V]
  else:
    v_new_out = None


  return o_out, ht_out, h_out, v_new_out


@jaxtyping.jaxtyped
def chunk_kda_fwd_custom(
    q: Float[Array, "H B T_ALIGNED K"],
    k: Float[Array, "H B T_ALIGNED K"],
    v: Float[Array, "H B T_ALIGNED V"],
    g: Float[Array, "H B T_ALIGNED K"],
    beta: Float[Array, "H B T_ALIGNED"],
    a_log: Float[Array, "H"] | None = None,
    delta_time_bias: Float[Array, "H*K"] | None = None,
    scale: float | None = None,
    initial_state: (
        Float[Array, "B H K V"] | Float[Array, "B N H K V"] | None
    ) = None,
    output_final_state: bool = False,
    use_gate_in_kernel: bool = False,
    segment_ids: Int[Array, "B T"] | None = None,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    disable_recompute: bool = True,
    context_parallel_metadata: ContextParallelMetadata | None = None,
    chunk_size: int = 64,
    return_residuals: bool = False,
    cu_seqlens: Int[Array, "B N_CU"] | None = None,
    aligned_cu_seqlens: Int[Array, "B N_CU"] | None = None,
    chunk_indices: Int[Array, "B NT 2"] | None = None,
    aligned_segment_ids: Int[Array, "B T_ALIGNED"] | None = None,
    q_rstd: Float[Array, "H B T_ALIGNED"] | None = None,
    k_rstd: Float[Array, "H B T_ALIGNED"] | None = None,
) -> tuple[
    tuple[
        Float[Array, "H B T V"],
        Float[Array, "B H K V"] | Float[Array, "B N H K V"] | None,
    ],
    KdaResiduals | None,
]:
  """Runs forward on inputs canonicalized by the Tokamax TPU adapter."""
  original_cu_seqlens = cu_seqlens
  cu_seqlens = aligned_cu_seqlens
  save_for_backward = return_residuals and disable_recompute

  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size

  _cp_active = (
      context_parallel_metadata is not None
      and context_parallel_metadata.is_cp_enabled
  )
  cp_metadata = None
  if _cp_active:
    assert context_parallel_metadata is not None
    cp_metadata = (
        context_parallel_metadata.is_first_rank,
        context_parallel_metadata.is_last_rank,
        context_parallel_metadata.pre_num_ranks,
        context_parallel_metadata.post_num_ranks,
    )
    if any(value is None for value in cp_metadata):
      raise ValueError("Enabled CP context is missing derived rank metadata.")

  _is_varlen = cu_seqlens is not None
  # ------------------------------------------------------------------
  # Step 1 + 2 (Fused): Gate cumsum + Intra-chunk solve
  # ------------------------------------------------------------------
  scale_val = 1.0 / math.sqrt(K) if scale is None else scale
  w, u, qg, kg, Aqk, Akk, g_cumsum = kda_fwd_intra_fused(
    q=q,
    k=k,
    v=v,
    g=g,
    beta=beta,
    scale=scale_val,
    cu_seqlens=cu_seqlens,
    chunk_size=BT,
    chunk_indices=chunk_indices,
    safe_gate=safe_gate,
    disable_recompute=save_for_backward,
    cumsum_scale=RCP_LN2,
    a_log=a_log,
    delta_time_bias=delta_time_bias,
    use_gate_in_kernel=use_gate_in_kernel,
    lower_bound=lower_bound,
  )

  if _cp_active:
    assert (
        cu_seqlens is not None
        and chunk_indices is not None
        and context_parallel_metadata is not None
    )
    initial_state = _prepare_cp_initial_state(
        kg=kg,
        w=w,
        u=u,
        gk=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        context_parallel_metadata=context_parallel_metadata,
        chunk_size=BT,
    )

  # ------------------------------------------------------------------
  # Step 3 + Step 4: Inter-chunk state + Output (gather/scatter for varlen)
  #
  # The inter-chunk kernel and output kernel require BT-aligned
  # cu_seqlens (they index blocks via bos // BT).  For non-aligned
  # varlen sequences we reuse the existing _align_seqs / _unalign_output
  # utilities to gather inputs into a chunk-aligned layout, run Stages
  # 3 & 4, then scatter the output back.
  #
  # Padding semantics (relied on for correctness):
  #   - k/w/u/q/Aqk at padded tail positions = 0 (from _align_seqs's
  #     jnp.pad with default fill value 0). With zero k/w/u/q the state
  #     update reduces to h_pad = h_{t-1} * exp(g_pad) + 0, and the
  #     output reduces to q_pad * exp(...) @ h = 0 * ... = 0.
  #   - g_cumsum at padded tail positions = 0 ⇒ exp(g_pad) = 1, so the
  #     hidden state is carried through padded positions unchanged.
  # ------------------------------------------------------------------
  # Stage 1/2 already operate in BT-aligned layout; derived tensors inherit
  # that layout. Use the fused Stage 3+4 kernel for both varlen and fixed
  # inputs. A fixed batch is represented as one sequence [0, T] per batch.
  if _is_varlen:
    stage34_cu_seqlens = cu_seqlens
    stage34_chunk_indices = chunk_indices
    stage34_initial_state = initial_state
  else:
    stage34_cu_seqlens = jnp.broadcast_to(
      jnp.asarray([0, T], dtype=jnp.int32), (B, 2)
    )
    block_ids = jnp.arange(T // BT, dtype=jnp.int32)
    fixed_chunk_indices = jnp.stack(
      [jnp.zeros_like(block_ids), block_ids], axis=-1
    )
    stage34_chunk_indices = jnp.broadcast_to(
      fixed_chunk_indices[None, :, :], (B, T // BT, 2)
    )
    stage34_initial_state = (
      initial_state[:, None, ...] if initial_state is not None else None
    )

  o, final_state, h, v_new = chunk_kda_fwd_h_o_varlen(
    w=w,
    u=u,
    kg=kg,
    gk=g_cumsum,
    q=q,
    A=Aqk,
    cu_seqlens=stage34_cu_seqlens,
    chunk_indices=stage34_chunk_indices,
    initial_state=stage34_initial_state,
    output_final_state=output_final_state,
    scale=scale,
    chunk_size=BT,
    store_h=disable_recompute,
    store_v_new=False,
  )
  if not _is_varlen and final_state is not None:
    final_state = final_state[:, 0]

  # ------------------------------------------------------------------
  # Drop intermediates that backward will recompute or never consume.
  # ------------------------------------------------------------------
  if not save_for_backward:
    w, u, qg, kg, v_new = None, None, None, None, None
    h = None
    if use_gate_in_kernel:
      g_cumsum = None

  output = o
  cu_seqlens = original_cu_seqlens

  if aligned_cu_seqlens is not None:
    if segment_ids is None:
      raise ValueError("Aligned varlen metadata requires `segment_ids`.")
    output = _unalign_output(
        output,
        cu_seqlens,
        aligned_cu_seqlens,
        segment_ids.shape[1],
    )

  output = (output.astype(q.dtype), final_state)
  if not return_residuals:
    return output, None

  g_org = g if use_gate_in_kernel else None

  # Keep the prepared inputs: the Op-level VJP also retains the original
  # arguments, but these copies may be varlen-aligned and L2-normalized.
  residuals = KdaResiduals(
      q=q,
      k=k,
      v=v,
      beta=beta,
      g_cumsum=g_cumsum,
      aqk=Aqk,
      akk=Akk,
      initial_state=initial_state,
      g_org=g_org,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      h=h,
      g_dtype_marker=jnp.zeros((), dtype=g.dtype),
      q_rstd=q_rstd,
      k_rstd=k_rstd,
      cu_seqlens=cu_seqlens,
      aligned_cu_seqlens=aligned_cu_seqlens,
      chunk_indices=chunk_indices,
      aligned_segment_ids=aligned_segment_ids,
      segment_ids=segment_ids,
      cp_metadata=cp_metadata,
  )
  return output, residuals
