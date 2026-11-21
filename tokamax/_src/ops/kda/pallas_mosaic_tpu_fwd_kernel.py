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
from jax import ad_checkpoint
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.kda.common import (
    RCP_LN2,
    estimate_mini_batch,
    get_max_mini_batch,
    get_tpu_limits,
    tri_inv_xor_dnc,
)
from tokamax._src.ops.kda.cp_utils import (
    ContextParallelMetadata,
    _merge_initial_state,
    all_gather_into_tensor,
    cp_associative_scan_fwd,
    cp_canonicalize_packed_segment_ids,
    cp_shift_right,
)
from tokamax._src.ops.kda.pallas_mosaic_tpu_types import GdnResiduals
from tokamax._src.ops.kda.pallas_mosaic_tpu_types import KdaResiduals
from tokamax._src.ops.kda.utils import (
    _unalign_output,
    align_up,
    canonicalize_packed_segment_ids,
    causal_conv1d_silu_fwd_bt,
    compute_packed_chunk_metadata,
    exp2,
    get_chain_precision,
    get_interpret,
    prepare_chunk_indices,
)

# Rows of the previous chunk kept for the depthwise conv; one f32 sublane tile.
# The conv weights are left-padded to this many taps, so W <= _CONV_HALO.
_CONV_HALO = 8

# Dimensions.
#   B: batch size
#   H: number of value heads (H_QK for query/key heads when GQA is active)
#   MB: mini-batch of heads processed per grid point
#   T: sequence length
#   NT: number of chunks (T // BT)
#   BT: chunk size (64 or 128)
#   BC: sub-chunk block size (16)
#   K: key/query head dimension
#   V: value head dimension
#   GW: gate channel width (1 for scalar gate, K for per-channel gate)

# ─── Context-Parallel Pre-Process ──────────────────────────────────────────


def _pre_process_kernel(
    # Inputs.
    seqlens_ref,  # scalar prefetch: cu_seqlens [N+1]
    chunk_to_seq_ref,  # scalar prefetch: chunk -> seq mapping [NT]
    last_seg_idx_ref,  # scalar prefetch: real last seg idx [1], int32
    k_ref,  # [MB, 1, BT, K_PADSIZE]
    w_ref,  # [MB, 1, BT, K_PADSIZE]
    u_ref,  # [MB, 1, BT, V_ALIGNED]
    gk_ref,  # [MB, 1, BT, K_PADSIZE]
    # Outputs.
    S_ext_ref,  # [MB, 1, K_PADSIZE, V_ALIGNED]
    M_ref,  # [MB, 1, K_PADSIZE, K_PADSIZE]
    # Scratch.
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

  # ── Auto MB: maximise VMEM utilisation ──
  # scratch per head: h_acc [K, V] + m_acc [K, K], all fp32.
  per_head_scratch = (K_PADSIZE * V_ALIGNED + K_PADSIZE * K_PADSIZE) * 4
  MB = estimate_mini_batch(per_head_scratch, H)

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
# Fused gate cumsum and intra-chunk solve
# =============================================================================


def _scalar_gate_intra(
    q_f32,
    k_f32,
    g_cumsum,
    beta_f32,
    *,
    BT: int,
    scale: float,
    gqa_repeats: int = 1,
    precision: jax.lax.Precision | None = None,
    seg_tok: jax.Array | None = None,
):
  """Computes intra-chunk Aqk and L for a scalar gate (Gated Delta Net)."""
  gs = g_cumsum[:, :, 0]  # [MB, BT]
  row = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
  col = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
  causal = row >= col
  strict = row > col
  if seg_tok is not None:
    seg_r = jnp.broadcast_to(seg_tok[:, None], (BT, BT))
    seg_c = jnp.broadcast_to(seg_tok[None, :], (BT, BT))
    same_doc = (seg_r == seg_c) & (seg_r > 0)
    causal = causal & same_doc
    strict = strict & same_doc

  # Mask pairwise differences before exp2 to prevent overflow on anti-causal entries.
  diff = gs[:, :, None] - gs[:, None, :]  # [MB, BT, BT]
  decay = jnp.exp2(jnp.where(causal[None], diff, jnp.float32(0.0)))

  qk = jax.lax.dot_general(
      q_f32,
      k_f32,
      (((2,), (2,)), ((0,), (0,))),
      precision=precision,
      preferred_element_type=jnp.float32,
  )  # [MB_QK, BT, BT]
  kk = jax.lax.dot_general(
      k_f32,
      k_f32,
      (((2,), (2,)), ((0,), (0,))),
      precision=precision,
      preferred_element_type=jnp.float32,
  )  # [MB_QK, BT, BT]
  if gqa_repeats > 1:
    qk = jnp.repeat(qk, gqa_repeats, axis=0)
    kk = jnp.repeat(kk, gqa_repeats, axis=0)

  zero = jnp.float32(0.0)
  Aqk = jnp.where(causal[None], qk * decay * scale, zero)
  L = jnp.where(strict[None], kk * decay * beta_f32, zero)
  return Aqk, L


def _make_chunk_halo(
    x_r: jax.Array,
    aligned_segment_ids: jax.Array | None = None,
    halo: int = _CONV_HALO,
    rank_halo: jax.Array | None = None,
) -> jax.Array:
  """Extracts the last `halo` tokens of chunk l-1 for each chunk l."""
  prev = x_r[:, :, :-1, -halo:, :]
  if aligned_segment_ids is not None:
    _, B, NC, BT, _ = x_r.shape
    seg_r = aligned_segment_ids.reshape(B, NC, BT)
    same_seg = (seg_r[:, 1:, 0] == seg_r[:, :-1, -1]) & (seg_r[:, 1:, 0] >= 0)
    prev = jnp.where(same_seg[None, :, :, None, None], prev, 0)
  if rank_halo is None:
    return jnp.pad(prev, ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
  return jnp.concatenate(
      [rank_halo.astype(prev.dtype)[:, :, None, :, :], prev], axis=2
  )


def _apply_causal_conv1d_silu(
    x_curr: jax.Array,
    x_prev: jax.Array,
    w_pad: jax.Array,
    bias: jax.Array,
    W: int,
    *,
    conv_lags_tok: jax.Array | None = None,
    valid_f32: jax.Array | None = None,
) -> jax.Array:
  """Computes depthwise causal conv1d (W <= _CONV_HALO) + SiLU in fp32."""
  x_curr_f32 = x_curr.astype(jnp.float32)
  x_prev_f32 = x_prev.astype(jnp.float32)
  w_f32 = w_pad.astype(jnp.float32)
  acc = x_curr_f32 * w_f32[:, _CONV_HALO - 1 : _CONV_HALO, :]
  for lag in range(1, W):
    x_lag = jnp.concatenate(
        [x_prev_f32[:, _CONV_HALO - lag :, :], x_curr_f32[:, :-lag, :]],
        axis=1,
    )
    if conv_lags_tok is not None:
      lag_f32 = ((conv_lags_tok >> lag) & 1).astype(jnp.float32)
      x_lag = x_lag * lag_f32[None, :, None]
    acc = acc + x_lag * w_f32[:, _CONV_HALO - 1 - lag : _CONV_HALO - lag, :]
  acc = acc + bias.astype(jnp.float32)[:, None, :]
  out = jax.nn.silu(acc)
  if valid_f32 is not None:
    out = out * valid_f32[None, :, None]
  return out


def _fused_gate_intra_kernel(
    # Inputs. All refs are single VMEM blocks.
    q_ref,  # [MB, 1, 1, BT, K]
    k_ref,  # [MB, 1, 1, BT, K]
    g_ref,  # [MB, 1, 1, 1, BT]  scalar gate, width-1 axis kept
    # off lanes; else [MB, 1, 1, BT, K]
    beta_ref,  # [MB, 1, 1, 1, BT]  width-1 axis kept off lanes
    v_ref,  # [MB, 1, 1, BT, V]
    a_log_ref,  # [MB, 1, 1, 1, 1] f32  per head
    delta_time_bias_ref,  # [MB, 1, 1, 1, K] f32  per head
    # With use_conv1d_in_kernel, nine conv inputs come first, all f32 except
    # the halos (input dtype); `HALO` is `_CONV_HALO`:
    #   q_prev_ref, k_prev_ref  [MB, 1, 1, HALO, K]  last HALO rows of the
    #   v_prev_ref              [MB, 1, 1, HALO, V]  previous chunk, zeroed
    #                                                across segment starts
    #   w_q_ref, w_k_ref        [MB, 1, 1, HALO, K]  W taps left-padded to
    #   w_v_ref                 [MB, 1, 1, HALO, V]  HALO; same block for
    #                                                every chunk
    #   b_q_ref, b_k_ref        [MB, 1, 1, 1, K]
    #   b_v_ref                 [MB, 1, 1, 1, V]
    # Outputs (k dtype unless noted):
    #   u_out_ref               [MB, 1, 1, BT, V]
    #   w_out_ref, kg_out_ref   [MB, 1, 1, BT, K]
    #   Aqk_out_ref             [MB, 1, 1, BT, BT]
    #   Akk_inv_out_ref         [MB, 1, 1, BT, BT]
    #   g_cumsum_out_ref        [MB, 1, 1, BT, GW] f32
    # and, with use_conv1d_in_kernel, q_norm_ref [MB, 1, 1, BT, K]; with
    # return_conv_intermediates also k_norm_ref [MB, 1, 1, BT, K] and
    # v_silu_ref [MB, 1, 1, BT, V] (v dtype); with use_qk_l2norm also
    # q_rstd_ref, k_rstd_ref [MB, 1, 1, 1, BT] f32.
    *conv_in_and_out_refs,
    chunk_size: int,
    head_dim: int,
    value_dim: int,
    scale: float,
    cumsum_scale: float,
    fuse_cumsum: bool,
    disable_recompute: bool,
    safe_gate: bool,
    per_channel_gate: bool,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    mini_batch: int = 1,
    gqa_repeats: int = 1,
    native_qkv_layout: bool = False,
    write_q_norm: bool = False,
    use_conv1d_in_kernel: bool = False,
    use_qk_l2norm: bool = False,
    return_conv_intermediates: bool = False,
    conv_kernel_size: int = 4,
    store_akk: bool = True,
    two_pass_bf16: bool = False,
    has_packed_meta: bool = False,
):
  """Fused Pallas kernel: gate activation + cumsum + Aqk/L + xor_dnc inversion.

  When fuse_cumsum=True, g_ref contains raw gate values (not cumsum'd).
  The kernel applies gate activation (if use_gate_in_kernel) and chunk-local
  prefix sum via tril matmul before proceeding to the Aqk/L construction.

  Inverts `(I + L)` via vectorized block-doubling (`tri_inv_xor_dnc`), using
  optional 2-pass BF16 residual refinement when `two_pass_bf16=True`.

  Ref shapes are listed next to the parameters.

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
  MB_QK = MB // gqa_repeats

  q_prev_ref = k_prev_ref = v_prev_ref = None
  w_q_ref = w_k_ref = w_v_ref = None
  b_q_ref = b_k_ref = b_v_ref = None
  if use_conv1d_in_kernel:
    (
        q_prev_ref,
        k_prev_ref,
        v_prev_ref,
        w_q_ref,
        w_k_ref,
        w_v_ref,
        b_q_ref,
        b_k_ref,
        b_v_ref,
        *rem_refs,
    ) = conv_in_and_out_refs
  else:
    rem_refs = conv_in_and_out_refs

  meta_ref = None
  if has_packed_meta:
    meta_ref = rem_refs[0]
    out_refs = rem_refs[1:]
  else:
    out_refs = rem_refs

  seg_tok = act_tok = valid_int = conv_lags_tok = None
  valid_f32 = m_in_f32 = m_out_f32 = None
  if has_packed_meta:
    assert meta_ref is not None
    meta_tile = meta_ref[0, 0, 0]  # [8, BT] int32
    seg_tok = meta_tile[0]
    act_tok = meta_tile[1]
    m_in_f32 = meta_tile[2].astype(jnp.float32)
    m_out_f32 = meta_tile[3].astype(jnp.float32)
    valid_int = meta_tile[4]
    valid_f32 = valid_int.astype(jnp.float32)
    conv_lags_tok = meta_tile[5]

  # `qg` is not an output: it is not part of `KdaResiduals` and the backward
  # recomputes it. `Akk_inv_out_ref` is omitted when `store_akk=False`.
  if store_akk:
    (
        u_out_ref,
        w_out_ref,
        kg_out_ref,
        Aqk_out_ref,
        Akk_inv_out_ref,
        g_cumsum_out_ref,
    ) = out_refs[:6]
    extra_out_refs = out_refs[6:]
  else:
    Akk_inv_out_ref = None
    (
        u_out_ref,
        w_out_ref,
        kg_out_ref,
        Aqk_out_ref,
        g_cumsum_out_ref,
    ) = out_refs[:5]
    extra_out_refs = out_refs[5:]

  # Load all MB heads at once
  if native_qkv_layout:
    q_in = jnp.stack(
        [q_ref[0, :, h * K : (h + 1) * K] for h in range(MB_QK)], axis=0
    )
    k_in = jnp.stack(
        [k_ref[0, :, h * K : (h + 1) * K] for h in range(MB_QK)], axis=0
    )
    v_in = jnp.stack(
        [v_ref[0, :, h * V : (h + 1) * V] for h in range(MB)], axis=0
    )
  else:
    q_in = q_ref[:, 0, 0]  # [MB_QK, BT, K]
    k_in = k_ref[:, 0, 0]  # [MB_QK, BT, K]
    v_in = v_ref[:, 0, 0]  # [MB, BT, V]
  # Transpose width-1 inputs stored as [.., 1, BT] back to [MB, BT, 1].
  if g_ref.shape[-2] == 1:
    g = g_ref[:, 0, 0, 0][..., None]  # [MB, BT] -> [MB, BT, 1]
  else:
    g = g_ref[:, 0, 0]  # [MB, BT, K]
  beta = beta_ref[:, 0, 0, 0][..., None]  # [MB, BT] -> [MB, BT, 1]

  if use_conv1d_in_kernel:
    assert (
        q_prev_ref is not None
        and k_prev_ref is not None
        and v_prev_ref is not None
        and w_q_ref is not None
        and w_k_ref is not None
        and w_v_ref is not None
        and b_q_ref is not None
        and b_k_ref is not None
        and b_v_ref is not None
    )
    q_prev = q_prev_ref[:, 0, 0]  # [MB, _CONV_HALO, K]
    k_prev = k_prev_ref[:, 0, 0]  # [MB, _CONV_HALO, K]
    v_prev = v_prev_ref[:, 0, 0]  # [MB, _CONV_HALO, V]
    w_q = w_q_ref[:, 0, 0]  # [MB, _CONV_HALO, K]
    w_k = w_k_ref[:, 0, 0]  # [MB, _CONV_HALO, K]
    w_v = w_v_ref[:, 0, 0]  # [MB, _CONV_HALO, V]
    b_q = b_q_ref[:, 0, 0, 0]  # [MB, K]
    b_k = b_k_ref[:, 0, 0, 0]  # [MB, K]
    b_v = b_v_ref[:, 0, 0, 0]  # [MB, V]

    q_silu = _apply_causal_conv1d_silu(
        q_in,
        q_prev,
        w_q,
        b_q,
        conv_kernel_size,
        conv_lags_tok=conv_lags_tok,
        valid_f32=valid_f32,
    ).astype(dtype)
    k_silu = _apply_causal_conv1d_silu(
        k_in,
        k_prev,
        w_k,
        b_k,
        conv_kernel_size,
        conv_lags_tok=conv_lags_tok,
        valid_f32=valid_f32,
    ).astype(dtype)
    v_silu = _apply_causal_conv1d_silu(
        v_in,
        v_prev,
        w_v,
        b_v,
        conv_kernel_size,
        conv_lags_tok=conv_lags_tok,
        valid_f32=valid_f32,
    ).astype(dtype)

    q_rstd_val = k_rstd_val = None
    if use_qk_l2norm:
      q_f32_s = q_silu.astype(jnp.float32)
      k_f32_s = k_silu.astype(jnp.float32)
      q_sq = jnp.sum(q_f32_s * q_f32_s, axis=-1, keepdims=True)
      k_sq = jnp.sum(k_f32_s * k_f32_s, axis=-1, keepdims=True)
      q_rstd_val = jnp.where(
          q_sq == 0.0,
          0.0,
          jax.lax.rsqrt(jnp.where(q_sq == 0.0, 1.0, q_sq) + 1e-6),
      )
      k_rstd_val = jnp.where(
          k_sq == 0.0,
          0.0,
          jax.lax.rsqrt(jnp.where(k_sq == 0.0, 1.0, k_sq) + 1e-6),
      )
      q = (q_f32_s * q_rstd_val).astype(dtype)
      k = (k_f32_s * k_rstd_val).astype(dtype)
    else:
      q = q_silu
      k = k_silu
    v = v_silu

    q_norm_ref = extra_out_refs[0]
    q_norm_ref[:, 0, 0] = q
    if return_conv_intermediates:
      k_norm_ref = extra_out_refs[1]
      v_silu_ref = extra_out_refs[2]
      k_norm_ref[:, 0, 0] = k
      v_silu_ref[:, 0, 0] = v
      if use_qk_l2norm:
        assert q_rstd_val is not None and k_rstd_val is not None
        q_rstd_ref = extra_out_refs[3]
        k_rstd_ref = extra_out_refs[4]
        q_rstd_ref[:, 0, 0, 0] = q_rstd_val[..., 0]
        k_rstd_ref[:, 0, 0, 0] = k_rstd_val[..., 0]
  else:
    if has_packed_meta:
      assert valid_f32 is not None
      valid_3d = valid_f32[None, :, None]
      q_in = (q_in.astype(jnp.float32) * valid_3d).astype(dtype)
      k_in = (k_in.astype(jnp.float32) * valid_3d).astype(dtype)
      v_in = (v_in.astype(jnp.float32) * valid_3d).astype(dtype)
    if use_qk_l2norm:
      q_f32_s = q_in.astype(jnp.float32)
      k_f32_s = k_in.astype(jnp.float32)
      q_sq = jnp.sum(q_f32_s * q_f32_s, axis=-1, keepdims=True)
      k_sq = jnp.sum(k_f32_s * k_f32_s, axis=-1, keepdims=True)
      q_rstd_val = jnp.where(
          q_sq == 0.0,
          0.0,
          jax.lax.rsqrt(jnp.where(q_sq == 0.0, 1.0, q_sq) + 1e-6),
      )
      k_rstd_val = jnp.where(
          k_sq == 0.0,
          0.0,
          jax.lax.rsqrt(jnp.where(k_sq == 0.0, 1.0, k_sq) + 1e-6),
      )
      q = (q_f32_s * q_rstd_val).astype(dtype)
      k = (k_f32_s * k_rstd_val).astype(dtype)
    else:
      q = q_in
      k = k_in
    v = v_in
    if write_q_norm:
      q_norm_ref = extra_out_refs[0]
      q_norm_ref[:, 0, 0] = q

  # --- Gate activation + cumsum ---
  g_f32 = g.astype(jnp.float32)
  if has_packed_meta:
    assert valid_f32 is not None
    g_f32 = g_f32 * valid_f32[None, :, None]

  if use_gate_in_kernel:
    dt_b = delta_time_bias_ref[:, 0, 0, 0]        # [MB, GW]
    g_f32 = g_f32 + dt_b[:, None, :]              # [MB, BT, GW]
    A_val = a_log_ref[:, 0, 0, 0, 0]              # [MB]
    if lower_bound is None:
      g_f32 = -jnp.exp(A_val)[:, None, None] * jax.nn.softplus(g_f32)
    else:
      g_f32 = lower_bound * jax.nn.sigmoid(
        jnp.exp(A_val)[:, None, None] * g_f32
      )
    if has_packed_meta:
      assert valid_f32 is not None
      g_f32 = g_f32 * valid_f32[None, :, None]

  if fuse_cumsum:
    same_span_bt = None
    if has_packed_meta:
      assert act_tok is not None
      act_r = jnp.broadcast_to(act_tok[:, None], (BT, BT))
      act_c = jnp.broadcast_to(act_tok[None, :], (BT, BT))
      same_span_bt = act_r == act_c
    if g_f32.shape[-1] == 1:
      row_g = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
      col_g = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
      _fwd_mask_b = row_g <= col_g
      if same_span_bt is not None:
        _fwd_mask_b = _fwd_mask_b & same_span_bt
      _fwd_mask = _fwd_mask_b.astype(jnp.float32)
      g_cumsum = (
          jnp.matmul(
              g_f32[..., 0],
              _fwd_mask,
              precision=jax.lax.Precision.HIGHEST,
              preferred_element_type=jnp.float32,
          )[..., None]
          * cumsum_scale
      )
    else:
      row_g = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
      col_g = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
      tril_b = row_g >= col_g
      if same_span_bt is not None:
        tril_b = tril_b & same_span_bt
      tril = tril_b.astype(jnp.float32)
      # tril[BT,BT] @ g_f32[MB,BT,K] → [MB,BT,K]: contract on BT, batch on MB
      g_cumsum = jax.lax.dot_general(
        tril, g_f32, (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
      ).transpose(1, 0, 2) * cumsum_scale  # [BT,MB,K] → [MB,BT,K]
  else:
    g_cumsum = g_f32

  q_f32_qk = q.astype(jnp.float32)
  k_f32_qk = k.astype(jnp.float32)
  if per_channel_gate and gqa_repeats > 1:
    q_f32 = jnp.repeat(q_f32_qk, gqa_repeats, axis=0)
    k_f32 = jnp.repeat(k_f32_qk, gqa_repeats, axis=0)
  else:
    q_f32 = q_f32_qk
    k_f32 = k_f32_qk
  beta_f32 = beta.astype(jnp.float32)
  if has_packed_meta:
    assert valid_f32 is not None
    beta_f32 = beta_f32 * valid_f32[None, :, None]

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

  if dtype == jnp.float32:
    solve_prec = jax.lax.Precision.HIGHEST
  elif two_pass_bf16:
    solve_prec = jax.lax.Precision.HIGH
  else:
    solve_prec = jax.lax.Precision.DEFAULT

  gn_ref_all = None
  seg_bt_k = None
  same_doc_bt = None
  bt_1d = None
  if per_channel_gate and has_packed_meta:
    assert (
        seg_tok is not None
        and act_tok is not None
        and valid_int is not None
    )
    bt_1d = jnp.arange(BT, dtype=jnp.int32)
    bt_row = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
    bt_col = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
    seg_r = jnp.broadcast_to(seg_tok[:, None], (BT, BT))
    seg_c = jnp.broadcast_to(seg_tok[None, :], (BT, BT))
    same_doc_bt = (seg_r == seg_c) & (seg_r > 0)
    same_blk_doc = ((bt_row // BC) == (bt_col // BC)) & same_doc_bt
    first_t = jnp.min(jnp.where(same_blk_doc, bt_col, BT), axis=1)
    last_t = jnp.max(jnp.where(same_blk_doc, bt_col, 0), axis=1)
    ref_t = (first_t + last_t) // 2 if safe_gate else first_t
    ref_t = valid_int * ref_t + (1 - valid_int) * bt_1d
    ref_r = jnp.broadcast_to(ref_t[:, None], (BT, BT))
    pick_ref_bt = (bt_col == ref_r).astype(jnp.float32)
    gn_ref_all = jax.lax.dot_general(
        pick_ref_bt,
        g_cumsum,
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).transpose(1, 0, 2)  # [MB, BT, K]
    seg_bt_k = jnp.broadcast_to(seg_tok[:, None], (BT, K))
  Aqk_rows = []
  L_rows = []
  for i_sc in range(NC) if per_channel_gate else ():
    i_s = i_sc * BC
    q_i = q_f32[:, i_s : i_s + BC]       # [MB, BC, K]
    k_i = k_f32[:, i_s : i_s + BC]       # [MB, BC, K]
    g_i = g_cumsum[:, i_s : i_s + BC]    # [MB, BC, K]
    beta_i = beta_f32[:, i_s : i_s + BC]  # [MB, BC, 1]

    if has_packed_meta:
      assert (
          gn_ref_all is not None
          and seg_bt_k is not None
          and same_doc_bt is not None
          and bt_1d is not None
          and seg_tok is not None
          and act_tok is not None
      )
      gn_ref_i = gn_ref_all[:, i_s : i_s + BC]  # [MB, BC, K]
      valid_i = seg_bt_k[i_s : i_s + BC] > 0    # [BC, K]
      diff_i = jnp.where(valid_i[None], g_i - gn_ref_i, 0.0)
      exp_diff_i = jnp.where(valid_i[None], jnp.exp2(diff_i), 0.0)
      q_eg = q_i * exp_diff_i
      k_eg = k_i * exp_diff_i

      in_diag_k = (row_iota_bt_k >= i_s) & (row_iota_bt_k < i_s + BC)
      valid_diag_j = in_diag_k & (seg_bt_k > 0)
      if i_sc > 0:
        d_prev = jnp.sum(act_tok * (bt_1d == (i_s - 1)).astype(jnp.int32))
        is_dprev_1d = (
            (bt_1d >= i_s).astype(jnp.int32)
            * (bt_1d < i_s + BC).astype(jnp.int32)
            * (seg_tok == d_prev).astype(jnp.int32)
            * (d_prev > 0).astype(jnp.int32)
        )
        any_dprev_i = jnp.max(is_dprev_1d)
        t0 = jnp.minimum(
            jnp.min(is_dprev_1d * bt_1d + (1 - is_dprev_1d) * BT), BT - 1
        )
        pick_t0 = (row_iota_bt_k == t0).astype(jnp.float32)
        gn_dprev = jnp.sum(
            gn_ref_all * pick_t0[None], axis=1, keepdims=True
        )  # [MB, 1, K]
        d_prev_2d = jnp.broadcast_to(d_prev, (BT, K))
        any_dprev_2d = jnp.broadcast_to(any_dprev_i, (BT, K))
        valid_pre_j = (
            (row_iota_bt_k < i_s)
            & (seg_bt_k == d_prev_2d)
            & (d_prev_2d > 0)
            & (any_dprev_2d > 0)
        )
        valid_j_b = valid_pre_j | valid_diag_j
        gn_ref_j = jnp.where(
            (row_iota_bt_k < i_s)[None], gn_dprev, gn_ref_all
        )
      else:
        valid_j_b = valid_diag_j
        gn_ref_j = gn_ref_all
      diff_j_safe = jnp.where(valid_j_b[None], gn_ref_j - g_cumsum, 0.0)
      k_eng_full = jnp.where(
          valid_j_b[None], k_f32 * jnp.exp2(diff_j_safe), 0.0
      )
    else:
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
      precision=solve_prec,
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
    if has_packed_meta:
      assert same_doc_bt is not None
      same_row = same_doc_bt[i_s : i_s + BC] & (col_iota_bc_bt < i_s + BC)
      aqk_keep = aqk_keep & same_row
      akk_keep = akk_keep & same_row
    Aqk_row = jnp.where(aqk_keep[None], Aqk_row, jnp.float32(0.0))
    Akk_row = jnp.where(akk_keep[None], Akk_row, jnp.float32(0.0))

    Aqk_rows.append(Aqk_row)
    L_rows.append(Akk_row)

  if per_channel_gate:
    Aqk = jnp.concatenate(Aqk_rows, axis=1)  # [MB, BT, BT] f32
    L = jnp.concatenate(L_rows, axis=1)  # [MB, BT, BT] f32
  else:
    Aqk, L = _scalar_gate_intra(
        q_f32_qk,
        k_f32_qk,
        g_cumsum,
        beta_f32,
        BT=BT,
        scale=scale,
        gqa_repeats=gqa_repeats,
        precision=solve_prec if dtype == jnp.float32 else None,
        seg_tok=seg_tok,
    )
    if gqa_repeats > 1:
      k_f32 = jnp.repeat(k_f32_qk, gqa_repeats, axis=0)

  # --- Solve (I + L) x = rhs via vectorized block-doubling (`xor_dnc`) ---
  v_beta = v.astype(jnp.float32) * beta_f32          # [MB, BT, V]
  if has_packed_meta:
    assert m_in_f32 is not None
    m_in_3d = m_in_f32[None, :, None]
    k_eg_beta = m_in_3d * k_f32 * jnp.exp2(m_in_3d * g_cumsum) * beta_f32
  else:
    k_eg_beta = k_f32 * jnp.exp2(g_cumsum) * beta_f32  # [MB, BT, K]
  rhs = jnp.concatenate([v_beta, k_eg_beta], axis=-1)  # [MB, BT, V+K]

  A_inv = tri_inv_xor_dnc(
      L,
      use_fp32=(dtype == jnp.float32),
      two_pass_bf16=two_pass_bf16,
  )
  result = jax.lax.dot_general(
      A_inv,
      rhs,
      (((2,), (1,)), ((0,), (0,))),
      precision=solve_prec,
      preferred_element_type=jnp.float32,
  )

  u = result[:, :, :V]         # [MB, BT, V]
  w = result[:, :, V : V + K]  # [MB, BT, K]

  # --- kg ---
  g_last = g_cumsum[:, BT - 1 : BT, :]  # [MB, 1, K]
  if has_packed_meta:
    assert m_out_f32 is not None
    m_out_3d = m_out_f32[None, :, None]
    kg = m_out_3d * k_f32 * exp2(m_out_3d * (g_last - g_cumsum))
  else:
    kg = k_f32 * exp2(g_last - g_cumsum)

  # --- Store all MB heads ---
  u_out_ref[:, 0, 0] = u.astype(u_out_ref.dtype)
  w_out_ref[:, 0, 0] = w.astype(w_out_ref.dtype)
  kg_out_ref[:, 0, 0] = kg.astype(kg_out_ref.dtype)
  Aqk_out_ref[:, 0, 0] = Aqk.astype(Aqk_out_ref.dtype)
  if Akk_inv_out_ref is not None:
    Akk_inv_out_ref[:, 0, 0] = A_inv.astype(Akk_inv_out_ref.dtype)
  g_cumsum_out_ref[:, 0, 0] = g_cumsum


def _compute_intra_fused_mini_batch(
    H,
    BT,
    K,
    V,
    dtype=None,
    *,
    per_channel_gate: bool = True,
    store_akk: bool = True,
    two_pass_bf16: bool = False,
    multiple_of: int = 1,
):
  """Auto-compute mini-batch size for intra fused kernel.

  Models double-buffered I/O plus f32 intra-chunk scratch (`Aqk`, `L`,
  `tri_inv_xor_dnc` stages, and `rhs`) at `BT > 64` so `BT=128` fits within
  64 MiB TPU7x VMEM even at `H_V=128`.
  """
  elem_size = jnp.dtype(dtype).itemsize if dtype is not None else 4
  wy_elem = 4 if (two_pass_bf16 and elem_size == 2) else elem_size
  if BT <= 64:
    per_head = (BT * K + BT * V + 2 * BT * BT) * wy_elem
    return estimate_mini_batch(per_head, H, multiple_of=multiple_of)

  gw = K if per_channel_gate else 1
  akk_elems = BT * BT if store_akk else 0
  io_bytes = 2 * (
      (2 * BT * K + BT * V) * elem_size
      + (2 * BT * K + BT * V + BT * BT + akk_elems) * wy_elem
      + BT * gw * 4
  )
  scratch_bt2 = 7 if per_channel_gate else 5
  scratch_bytes = (
      3 * BT * K + 2 * BT * (V + K) + BT * gw + scratch_bt2 * BT * BT
  ) * 4
  per_head = io_bytes + scratch_bytes
  max_mb = get_max_mini_batch() // (2 if per_channel_gate else 1)
  return estimate_mini_batch(
      per_head, H, max_mb=max_mb, multiple_of=multiple_of
  )


@functools.partial(
    jax.jit,
    static_argnames=[
        "chunk_size",
        "scale",
        "safe_gate",
        "per_channel_gate",
        "disable_recompute",
        "cumsum_scale",
        "use_gate_in_kernel",
        "lower_bound",
        "mini_batch",
        "use_conv1d_in_kernel",
        "use_qk_l2norm",
        "return_conv_intermediates",
        "num_k_heads",
        "head_k_dim",
        "head_v_dim",
        "native_bt_qkv",
        "write_q_norm",
        "store_akk",
        "two_pass_bf16",
    ],
)
@jaxtyping.jaxtyped
def pallas_kda_fwd_intra_fused(
    q: Float[Array, "..."] | None,
    k: Float[Array, "..."] | None,
    v: Float[Array, "..."] | None,
    g: Float[Array, "H B T GW"],
    beta: Float[Array, "H B T"],
    scale: float,
    chunk_size: int = 64,
    safe_gate: bool = True,
    per_channel_gate: bool = True,
    disable_recompute: bool = False,
    cumsum_scale: float = RCP_LN2,
    a_log: Float[Array, "H"] | None = None,
    delta_time_bias: Float[Array, "H*GW"] | None = None,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    mini_batch: int | None = None,
    conv_weight_q: Float[Array, "H_Q W K"] | None = None,
    conv_weight_k: Float[Array, "H_K W K"] | None = None,
    conv_weight_v: Float[Array, "H W V"] | None = None,
    conv_bias_q: Float[Array, "H_Q K"] | None = None,
    conv_bias_k: Float[Array, "H_K K"] | None = None,
    conv_bias_v: Float[Array, "H V"] | None = None,
    use_conv1d_in_kernel: bool = False,
    use_qk_l2norm: bool = False,
    return_conv_intermediates: bool = False,
    aligned_segment_ids: Int[Array, "B T"] | None = None,
    qkv_conv: Float[Array, "B T C_QKV"] | None = None,
    num_k_heads: int | None = None,
    head_k_dim: int | None = None,
    head_v_dim: int | None = None,
    native_bt_qkv: bool = False,
    write_q_norm: bool = False,
    store_akk: bool = True,
    two_pass_bf16: bool = False,
    packed_tok_meta: Int[Array, "B NC 8 BT"] | None = None,
    rank_halo_q: Float[Array, "..."] | None = None,
    rank_halo_k: Float[Array, "..."] | None = None,
    rank_halo_v: Float[Array, "..."] | None = None,
) -> tuple[
    Float[Array, "H B T K"],
    Float[Array, "H B T V"],
    Float[Array, "H B T K"],
    Float[Array, "H B T BT"],
    Float[Array, "H B T BT"] | None,
    Float[Array, "H B T GW"],
    Float[Array, "H_QK B T K"] | None,
    Float[Array, "H B T K"] | None,
    Float[Array, "H B T V"] | None,
    Float[Array, "H B T"] | None,
    Float[Array, "H B T"] | None,
]:
  """Fuses gate cumsum with the fixed-length intra-chunk solve.

  Heads are mini-batched to amortize DMA. When `use_conv1d_in_kernel` is
  true, causal conv1d + SiLU (+ optional L2 norm) are fused into the kernel.
  """
  H, B, T, GW = g.shape
  BT = chunk_size
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  NC = T // BT

  if qkv_conv is not None:
    assert (
        num_k_heads is not None
        and head_k_dim is not None
        and head_v_dim is not None
    )
    H_QK = num_k_heads
    K = head_k_dim
    V = head_v_dim
    act_dtype = qkv_conv.dtype
  else:
    assert q is not None and k is not None and v is not None
    if native_bt_qkv:
      _, _, H_QK, K = q.shape
      V = v.shape[-1]
    else:
      H_QK, _, _, K = q.shape
      V = v.shape[-1]
    act_dtype = k.dtype
  assert H % H_QK == 0, f"H={H} must be divisible by H_QK={H_QK}"
  gqa_repeats = H // H_QK
  native_qkv_layout = qkv_conv is not None or native_bt_qkv

  if use_gate_in_kernel:
    assert a_log is not None, "a_log required when use_gate_in_kernel=True"

  if mini_batch is None:
    MB = _compute_intra_fused_mini_batch(
        H,
        BT,
        K,
        V,
        dtype=act_dtype,
        per_channel_gate=per_channel_gate,
        store_akk=store_akk,
        two_pass_bf16=two_pass_bf16,
        multiple_of=gqa_repeats,
    )
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"
    assert (
        MB % gqa_repeats == 0
    ), f"mini_batch={MB} must be divisible by gqa_repeats={gqa_repeats}"
  MB_QK = MB // gqa_repeats

  if qkv_conv is not None:
    q_r = qkv_conv
    k_r = qkv_conv
    v_r = qkv_conv
  elif native_bt_qkv:
    assert q is not None and k is not None and v is not None
    q_r = q.reshape(B, T, H_QK * K)
    k_r = k.reshape(B, T, H_QK * K)
    v_r = v.reshape(B, T, H * V)
  else:
    assert q is not None and k is not None and v is not None
    # [H_QK, B, T, K] -> [H_QK, B, NC, BT, K]
    q_r = q.reshape(H_QK, B, NC, BT, K)
    k_r = k.reshape(H_QK, B, NC, BT, K)
    v_r = v.reshape(H, B, NC, BT, V)
  scalar_gate = GW == 1
  # Store width-1 axes as [.., 1, BT] to avoid 128-lane minor-axis padding in HBM.
  g_r = (
      g.reshape(H, B, NC, 1, BT) if scalar_gate else g.reshape(H, B, NC, BT, GW)
  )
  beta_r = beta.reshape(H, B, NC, 1, BT)

  if use_gate_in_kernel:
    assert a_log is not None
    a_log_r = a_log.astype(jnp.float32).reshape(H, 1, 1, 1, 1)
    if delta_time_bias is not None:
      delta_time_bias_r = delta_time_bias.astype(jnp.float32).reshape(
          H, 1, 1, 1, GW
      )
    else:
      delta_time_bias_r = jnp.zeros((H, 1, 1, 1, GW), dtype=jnp.float32)
  else:
    a_log_r = jnp.zeros((H, 1, 1, 1, 1), dtype=jnp.float32)
    delta_time_bias_r = jnp.zeros((H, 1, 1, 1, GW), dtype=jnp.float32)

  grid = (H // MB, B, NC)

  def _make_spec(last_dim):
    return pl.BlockSpec(
      index_map=lambda i, j, l: (i, j, l, 0, 0),
      block_shape=(MB, 1, 1, BT, last_dim),
    )

  def _make_halo_spec(last_dim):
    return pl.BlockSpec(
        index_map=lambda i, j, l: (i, j, l, 0, 0),
        block_shape=(MB, 1, 1, _CONV_HALO, last_dim),
    )

  def _make_weight_spec(last_dim):
    return pl.BlockSpec(
        index_map=lambda i, j, l: (i, 0, 0, 0, 0),
        block_shape=(MB, 1, 1, _CONV_HALO, last_dim),
    )

  def _make_per_head_spec(last_dim):
    return pl.BlockSpec(
      index_map=lambda i, j, l: (i, 0, 0, 0, 0),
      block_shape=(MB, 1, 1, 1, last_dim),
    )

  def _make_narrow_spec():
    """For an operand stored `[.., 1, BT]` to keep its size-1 axis off lanes."""
    return pl.BlockSpec(
        index_map=lambda i, j, l: (i, j, l, 0, 0),
        block_shape=(MB, 1, 1, 1, BT),
    )

  def _make_qk_spec(last_dim):
    return pl.BlockSpec(
        index_map=lambda i, j, l: (i, j, l, 0, 0),
        block_shape=(MB_QK, 1, 1, BT, last_dim),
    )

  if qkv_conv is not None:
    q_size = H_QK * K
    k_size = H_QK * K
    tile_q = MB_QK * K
    tile_k = MB_QK * K
    tile_v = MB * V
    q_spec = pl.BlockSpec(
        (pl.Element(1), pl.Element(BT), pl.Element(tile_q)),
        lambda hg, b, c: (
            b,
            pl.multiple_of(c * BT, 8),
            pl.multiple_of(hg * tile_q, 128),
        ),
    )
    k_spec = pl.BlockSpec(
        (pl.Element(1), pl.Element(BT), pl.Element(tile_k)),
        lambda hg, b, c: (
            b,
            pl.multiple_of(c * BT, 8),
            pl.multiple_of(q_size + hg * tile_k, 128),
        ),
    )
    v_spec = pl.BlockSpec(
        (pl.Element(1), pl.Element(BT), pl.Element(tile_v)),
        lambda hg, b, c: (
            b,
            pl.multiple_of(c * BT, 8),
            pl.multiple_of(q_size + k_size + hg * tile_v, 128),
        ),
    )
  elif native_bt_qkv:
    tile_q = MB_QK * K
    tile_v = MB * V
    q_spec = pl.BlockSpec((1, BT, tile_q), lambda hg, b, c: (b, c, hg))
    k_spec = q_spec
    v_spec = pl.BlockSpec((1, BT, tile_v), lambda hg, b, c: (b, c, hg))
  else:
    q_spec = _make_qk_spec(K)
    k_spec = _make_qk_spec(K)
    v_spec = _make_spec(V)

  in_args = [q_r, k_r, g_r, beta_r, v_r, a_log_r, delta_time_bias_r]
  in_specs = [
      q_spec,
      k_spec,
      _make_narrow_spec() if scalar_gate else _make_spec(GW),
      _make_narrow_spec(),
      v_spec,
      _make_per_head_spec(1),
      _make_per_head_spec(GW),
  ]
  wy_dtype = (
      jnp.float32
      if (two_pass_bf16 and act_dtype == jnp.bfloat16)
      else act_dtype
  )
  out_shape = [
      jax.ShapeDtypeStruct((H, B, NC, BT, V), wy_dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), wy_dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), wy_dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), wy_dtype),
  ]
  out_specs = [
      _make_spec(V),
      _make_spec(K),
      _make_spec(K),
      _make_spec(BT),
  ]
  if store_akk:
    out_shape.append(jax.ShapeDtypeStruct((H, B, NC, BT, BT), wy_dtype))
    out_specs.append(_make_spec(BT))
  out_shape.append(jax.ShapeDtypeStruct((H, B, NC, BT, GW), jnp.float32))
  out_specs.append(_make_spec(GW))

  conv_kernel_size = 4
  if use_conv1d_in_kernel:
    assert (
        k is not None
        and v is not None
        and conv_weight_q is not None
        and conv_weight_k is not None
        and conv_weight_v is not None
    )
    H_Q, W, _ = conv_weight_q.shape
    H_K, _, _ = conv_weight_k.shape
    # Internal backstop; the Mosaic backend rejects other widths up front.
    assert (
        1 <= W <= _CONV_HALO
    ), f"conv_weight kernel size W={W} must be in [1, {_CONV_HALO}]"
    conv_kernel_size = W
    q_prev_r = _make_chunk_halo(
        q_r, aligned_segment_ids, halo=_CONV_HALO, rank_halo=rank_halo_q
    )
    k_prev_r = _make_chunk_halo(
        k_r, aligned_segment_ids, halo=_CONV_HALO, rank_halo=rank_halo_k
    )
    v_prev_r = _make_chunk_halo(
        v_r, aligned_segment_ids, halo=_CONV_HALO, rank_halo=rank_halo_v
    )
    w_q_exp = (
        jnp.repeat(conv_weight_q, H // H_Q, axis=0)
        if H_Q < H
        else conv_weight_q
    )
    w_k_exp = (
        jnp.repeat(conv_weight_k, H // H_K, axis=0)
        if H_K < H
        else conv_weight_k
    )
    b_q_exp = (
        (
            jnp.repeat(conv_bias_q, H // conv_bias_q.shape[0], axis=0)
            if conv_bias_q.shape[0] < H
            else conv_bias_q
        )
        if conv_bias_q is not None
        else None
    )
    b_k_exp = (
        (
            jnp.repeat(conv_bias_k, H // conv_bias_k.shape[0], axis=0)
            if conv_bias_k.shape[0] < H
            else conv_bias_k
        )
        if conv_bias_k is not None
        else None
    )
    w_q_pad = jnp.pad(
        w_q_exp.astype(jnp.float32), ((0, 0), (_CONV_HALO - W, 0), (0, 0))
    ).reshape(H, 1, 1, _CONV_HALO, K)
    w_k_pad = jnp.pad(
        w_k_exp.astype(jnp.float32), ((0, 0), (_CONV_HALO - W, 0), (0, 0))
    ).reshape(H, 1, 1, _CONV_HALO, K)
    w_v_pad = jnp.pad(
        conv_weight_v.astype(jnp.float32),
        ((0, 0), (_CONV_HALO - W, 0), (0, 0)),
    ).reshape(H, 1, 1, _CONV_HALO, V)
    b_q_r = (
        b_q_exp.astype(jnp.float32).reshape(H, 1, 1, 1, K)
        if b_q_exp is not None
        else jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)
    )
    b_k_r = (
        b_k_exp.astype(jnp.float32).reshape(H, 1, 1, 1, K)
        if b_k_exp is not None
        else jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)
    )
    b_v_r = (
        conv_bias_v.astype(jnp.float32).reshape(H, 1, 1, 1, V)
        if conv_bias_v is not None
        else jnp.zeros((H, 1, 1, 1, V), dtype=jnp.float32)
    )

    in_args.extend([
        q_prev_r,
        k_prev_r,
        v_prev_r,
        w_q_pad,
        w_k_pad,
        w_v_pad,
        b_q_r,
        b_k_r,
        b_v_r,
    ])
    in_specs.extend([
        _make_halo_spec(K),
        _make_halo_spec(K),
        _make_halo_spec(V),
        _make_weight_spec(K),
        _make_weight_spec(K),
        _make_weight_spec(V),
        _make_per_head_spec(K),
        _make_per_head_spec(K),
        _make_per_head_spec(V),
    ])

    out_shape.append(jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype))
    out_specs.append(_make_spec(K))
    if return_conv_intermediates:
      out_shape.extend([
          jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),
          jax.ShapeDtypeStruct((H, B, NC, BT, V), v.dtype),
      ])
      out_specs.extend([
          _make_spec(K),
          _make_spec(V),
      ])
      if use_qk_l2norm:
        out_shape.extend([
            jax.ShapeDtypeStruct((H, B, NC, 1, BT), jnp.float32),
            jax.ShapeDtypeStruct((H, B, NC, 1, BT), jnp.float32),
        ])
        out_specs.extend([
            _make_narrow_spec(),
            _make_narrow_spec(),
        ])
  elif write_q_norm:
    out_shape.append(jax.ShapeDtypeStruct((H_QK, B, NC, BT, K), act_dtype))
    out_specs.append(_make_qk_spec(K))

  has_packed_meta = packed_tok_meta is not None
  if has_packed_meta:
    assert packed_tok_meta is not None
    meta_r = packed_tok_meta[None, ...]  # [1, B, NC, 8, BT]
    in_args.append(meta_r)
    in_specs.append(
        pl.BlockSpec(
            index_map=lambda i, j, l: (0, j, l, 0, 0),
            block_shape=(1, 1, 1, 8, BT),
        )
    )

  total_chunks = H * B * NC
  tri_inv_matmuls = max(0, 2 * int(math.log2(BT)) - 3) + (
      2 if two_pass_bf16 and act_dtype == jnp.bfloat16 else 0
  )
  intra_flops_per_chunk = (
      4 * BT * BT * K // gqa_repeats
      + 2 * BT * BT * (V + K + GW)
      + 2 * tri_inv_matmuls * BT * BT * BT
  )
  intra_transcendentals_per_chunk = (
      2 * BT * K if per_channel_gate else (BT * BT + 2 * BT)
  )
  # When `qkv_conv` is aliased across q/k/v specs, count its HBM bytes once.
  unique_in_args = (
      [qkv_conv, g_r, beta_r, a_log_r, delta_time_bias_r]
      if qkv_conv is not None
      else in_args
  )
  intra_bytes = sum(
      int(x.size) * jnp.dtype(x.dtype).itemsize for x in unique_in_args
  ) + sum(int(s.size) * jnp.dtype(s.dtype).itemsize for s in out_shape)
  intra_cost_estimate = pl.CostEstimate(
      flops=total_chunks * intra_flops_per_chunk,
      transcendentals=total_chunks * intra_transcendentals_per_chunk,
      bytes_accessed=intra_bytes,
  )

  pallas_outs = pl.pallas_call(
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
          per_channel_gate=per_channel_gate,
          use_gate_in_kernel=use_gate_in_kernel,
          lower_bound=lower_bound,
          mini_batch=MB,
          gqa_repeats=gqa_repeats,
          native_qkv_layout=native_qkv_layout,
          write_q_norm=write_q_norm,
          use_conv1d_in_kernel=use_conv1d_in_kernel,
          use_qk_l2norm=use_qk_l2norm,
          return_conv_intermediates=return_conv_intermediates,
          conv_kernel_size=conv_kernel_size,
          store_akk=store_akk,
          two_pass_bf16=two_pass_bf16,
          has_packed_meta=has_packed_meta,
      ),
      interpret=get_interpret(),
      out_shape=out_shape,
      in_specs=in_specs,
      out_specs=out_specs,
      grid=grid,
      cost_estimate=intra_cost_estimate,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
      ),
  )(*in_args)

  if store_akk:
    u_r, w_r, kg_r, Aqk_r, Akk_inv_r, g_cumsum_r = pallas_outs[:6]
    extra_outs = pallas_outs[6:]
  else:
    Akk_inv_r = None
    u_r, w_r, kg_r, Aqk_r, g_cumsum_r = pallas_outs[:5]
    extra_outs = pallas_outs[5:]

  # --- Reshape back to [H, B, T, D] (head-first) ---
  w_out = w_r.reshape(H, B, T, K)
  u_out = u_r.reshape(H, B, T, V)
  kg_out = kg_r.reshape(H, B, T, K)
  Aqk_flat = Aqk_r.reshape(H, B, NC * BT, BT)
  Akk_flat = (
      Akk_inv_r.reshape(H, B, NC * BT, BT) if Akk_inv_r is not None else None
  )
  g_cumsum_out = g_cumsum_r.reshape(H, B, T, GW)

  q_norm_out = k_norm_out = v_silu_out = q_rstd_out = k_rstd_out = None
  if use_conv1d_in_kernel:
    q_norm_out = extra_outs[0].reshape(H, B, T, K)
    if return_conv_intermediates:
      k_norm_out = extra_outs[1].reshape(H, B, T, K)
      v_silu_out = extra_outs[2].reshape(H, B, T, V)
      if use_qk_l2norm:
        q_rstd_out = extra_outs[3].reshape(H, B, T)
        k_rstd_out = extra_outs[4].reshape(H, B, T)
  elif write_q_norm:
    q_norm_out = extra_outs[0].reshape(H_QK, B, T, K)

  return (
      w_out,
      u_out,
      kg_out,
      Aqk_flat,
      Akk_flat,
      g_cumsum_out,
      q_norm_out,
      k_norm_out,
      v_silu_out,
      q_rstd_out,
      k_rstd_out,
  )


def kda_fwd_intra_fused(
    q: jax.Array | None,
    k: jax.Array | None,
    v: jax.Array | None,
    g: jax.Array,
    beta: jax.Array,
    scale: float,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
    chunk_indices: jax.Array | None = None,
    safe_gate: bool = True,
    per_channel_gate: bool = True,
    disable_recompute: bool = False,
    cumsum_scale: float = RCP_LN2,
    a_log: jax.Array | None = None,
    delta_time_bias: jax.Array | None = None,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    conv_weight_q: jax.Array | None = None,
    conv_weight_k: jax.Array | None = None,
    conv_weight_v: jax.Array | None = None,
    conv_bias_q: jax.Array | None = None,
    conv_bias_k: jax.Array | None = None,
    conv_bias_v: jax.Array | None = None,
    use_conv1d_in_kernel: bool = False,
    use_qk_l2norm: bool = False,
    return_conv_intermediates: bool = False,
    aligned_segment_ids: jax.Array | None = None,
    mini_batch: int | None = None,
    qkv_conv: jax.Array | None = None,
    num_k_heads: int | None = None,
    head_k_dim: int | None = None,
    head_v_dim: int | None = None,
    native_bt_qkv: bool = False,
    write_q_norm: bool = False,
    store_akk: bool = True,
    two_pass_bf16: bool = False,
    packed_tok_meta: jax.Array | None = None,
    rank_halo_q: jax.Array | None = None,
    rank_halo_k: jax.Array | None = None,
    rank_halo_v: jax.Array | None = None,
):
  """Fused gate cumsum + intra-chunk solve entry.

  Args:
      g: [H, B, T, K] for a per-channel gate or [H, B, T, 1] for a scalar gate
        -- raw gate input (NOT cumsum'd). Other args configure the intra-chunk
        solve and gate activation.

  Returns:
      11-tuple: (w, u, kg, Aqk, Akk, g_cumsum, q_norm, k_norm, v_silu, q_rstd,
      k_rstd).
  """
  assert (
      per_channel_gate is False or chunk_size in (64, 128)
  ), (
      "per-channel gate is validated at chunk_size in (64, 128); got"
      f" {chunk_size}"
  )
  assert chunk_size in (
      16,
      32,
      64,
      128,
      256,
      512,
  ), f"chunk_size must be a power of two in [16, 512]; got {chunk_size}"

  return pallas_kda_fwd_intra_fused(
      q=q,
      k=k,
      v=v,
      g=g,
      beta=beta,
      scale=scale,
      chunk_size=chunk_size,
      safe_gate=safe_gate,
      per_channel_gate=per_channel_gate,
      disable_recompute=disable_recompute,
      cumsum_scale=cumsum_scale,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
      mini_batch=mini_batch,
      conv_weight_q=conv_weight_q,
      conv_weight_k=conv_weight_k,
      conv_weight_v=conv_weight_v,
      conv_bias_q=conv_bias_q,
      conv_bias_k=conv_bias_k,
      conv_bias_v=conv_bias_v,
      use_conv1d_in_kernel=use_conv1d_in_kernel,
      use_qk_l2norm=use_qk_l2norm,
      return_conv_intermediates=return_conv_intermediates,
      aligned_segment_ids=aligned_segment_ids,
      qkv_conv=qkv_conv,
      num_k_heads=num_k_heads,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      native_bt_qkv=native_bt_qkv,
      write_q_norm=write_q_norm,
      store_akk=store_akk,
      two_pass_bf16=two_pass_bf16,
      packed_tok_meta=packed_tok_meta,
      rank_halo_q=rank_halo_q,
      rank_halo_k=rank_halo_k,
      rank_halo_v=rank_halo_v,
  )


# =============================================================================
# Fused state propagation and output
# =============================================================================


def _chunk_kda_fwd_h_o_varlen_kernel(
    seqlens_ref,  # scalar prefetch: cu_seqlens [N+1]
    chunk_to_seq_ref,  # scalar prefetch: chunk -> seq mapping [NT]
    *in_out_scratch_refs,
    BT,
    scale,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
    STORE_H,
    STORE_V_NEW,
    MB,
    GQA_REPEATS,
    NATIVE_OUT_LAYOUT,
    OUTPUT_PRECISION,
    HAS_PACKED_META=False,
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
  if HAS_PACKED_META:
    (
        keep_ref,
        w_ref,
        u_ref,
        kg_ref,
        gk_ref,
        q_ref,
        A_ref,
        h0_ref,
        meta_ref,
        o_ref,
        ht_ref,
        h_out_ref,
        v_new_out_ref,
        scratch_ref,
    ) = in_out_scratch_refs
  else:
    keep_ref = None
    meta_ref = None
    (
        w_ref,
        u_ref,
        kg_ref,
        gk_ref,
        q_ref,
        A_ref,
        h0_ref,
        o_ref,
        ht_ref,
        h_out_ref,
        v_new_out_ref,
        scratch_ref,
    ) = in_out_scratch_refs

  i_b = pl.program_id(1)
  i_c = pl.program_id(2)
  seq_idx = chunk_to_seq_ref[i_b, i_c]

  bos = seqlens_ref[i_b, seq_idx]
  eos = seqlens_ref[i_b, seq_idx + 1]
  t0 = i_c * BT

  K = w_ref.shape[-1]
  V = u_ref.shape[-1]

  seg_tok = m_in_f32 = valid_f32 = None
  if HAS_PACKED_META:
    assert meta_ref is not None
    meta_tile = meta_ref[0, 0, 0]  # [8, BT] int32
    seg_tok = meta_tile[0]
    m_in_f32 = meta_tile[2].astype(jnp.float32)
    valid_f32 = meta_tile[4].astype(jnp.float32)

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
  # v_new feeds directly into the recursive state update, so this matmul and
  # the state update below are the two on the serial chain. Their precision
  # is switchable; see `get_chain_precision`.
  b_v_new = b_u.astype(jnp.float32) - jnp.matmul(
      b_w.astype(jnp.float32),
      b_h,
      precision=get_chain_precision(),
      preferred_element_type=jnp.float32,
  )  # [MB, BT, V]
  if HAS_PACKED_META:
    assert valid_f32 is not None
    b_v_new = b_v_new * valid_f32[None, :, None]

  # Spill v_new (used by bwd Stage 0 when disable_recompute=True).
  if STORE_V_NEW:
    v_new_out_ref[:,0,:] = b_v_new.astype(v_new_out_ref.dtype)

  # Stage 4 inter-chunk output: scale * (q * exp2(g - g_ref)) @ (h * exp2(g_ref))
  # Reference-point stabilization mirrors the production varlen kernel's
  # g_ref = g[0] choice for bit-identical numerics.
  b_q = q_ref[:,0,:]                              # [MB_QK, BT, K]
  if GQA_REPEATS > 1:
    b_q = jnp.repeat(b_q, GQA_REPEATS, axis=0)    # [MB, BT, K]
  b_g = gk_ref[:,0,:].astype(jnp.float32)         # [MB, BT, K]
  b_A = A_ref[:,0,:]                               # [MB, BT, BT]

  b_g_ref_row = b_g[:, 0:1, :]                # [MB, 1, K]
  if HAS_PACKED_META:
    assert m_in_f32 is not None
    m_in_3d = m_in_f32[None, :, None]
    diff_qg = m_in_3d * jnp.maximum(b_g - b_g_ref_row, -126.0)
    b_qg = m_in_3d * b_q.astype(jnp.float32) * jnp.exp2(diff_qg)
  else:
    b_qg = b_q.astype(jnp.float32) * jnp.exp2(
      jnp.maximum(b_g - b_g_ref_row, -126.0)
    )                                            # [MB, BT, K]
  b_h_scaled = b_h * jnp.exp2(
    jnp.maximum(b_g_ref_row[:, 0, :], -126.0)
  )[:, :, None]                                # [MB, K, V]

  # [MB, BT, K] @ [MB, K, V] -> [MB, BT, V]
  # Output-only GEMM: bf16 inputs use DEFAULT (no extra precision to preserve),
  # fp32 inputs use DEFAULT/HIGHEST to maintain full mantissa fidelity.
  b_o = jnp.matmul(
    b_qg, b_h_scaled,
    precision=OUTPUT_PRECISION,
    preferred_element_type=jnp.float32,
  ) * scale                                    # [MB, BT, V]

  # Stage 4 intra-chunk: A @ v_new
  # Apply lower-triangular mask: Aqk is causal by construction, but masking
  # here matches the original per-head loop behaviour and guards against
  # any tiny upper-triangle fp noise from the intra-chunk solve.
  row_s = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=0)
  col_s = jax.lax.broadcasted_iota(jnp.int32, (BT, BT), dimension=1)
  m_s = row_s >= col_s
  if HAS_PACKED_META:
    assert seg_tok is not None
    seg_r = jnp.broadcast_to(seg_tok[:, None], (BT, BT))
    seg_c = jnp.broadcast_to(seg_tok[None, :], (BT, BT))
    same_doc = (seg_r == seg_c) & (seg_r > 0)
    m_s = m_s & same_doc
  b_A_f32 = jnp.where(m_s[None, :, :], b_A.astype(jnp.float32), 0.0)
  # [MB, BT, BT] @ [MB, BT, V] -> [MB, BT, V]
  b_o = b_o + jnp.matmul(
    b_A_f32, b_v_new,
    precision=OUTPUT_PRECISION,
    preferred_element_type=jnp.float32,
  )                                            # [MB, BT, V]
  if HAS_PACKED_META:
    assert valid_f32 is not None
    b_o = b_o * valid_f32[None, :, None]

  if NATIVE_OUT_LAYOUT:
    b_o_cast = b_o.astype(o_ref.dtype)
    o_ref[...] = jnp.concatenate([b_o_cast[h] for h in range(MB)], axis=-1)[
        None, ...
    ]
  else:
    o_ref[:,0,:] = b_o.astype(o_ref.dtype)

  # Stage 3 state update: h = decay(h) + kg^T @ v_new
  # On the serial chain: accumulates directly into the recursive hidden state,
  # so its precision comes from `get_chain_precision` (HIGHEST by default).
  b_gk_last = gk_ref[:,0,:][:, BT - 1, :].astype(jnp.float32)  # [MB, K]
  if HAS_PACKED_META:
    assert keep_ref is not None
    keep_f32 = keep_ref[i_b, i_c].astype(jnp.float32)
    b_h_new = keep_f32 * (
        b_h * jnp.exp2(keep_f32 * b_gk_last)[:, :, None]
    )
  else:
    b_h_new = b_h * jnp.exp2(b_gk_last)[:, :, None]          # [MB, K, V] decay

  b_kg = kg_ref[:,0,:]   # [MB, BT, K]
  # [MB, K, BT] @ [MB, BT, V] -> [MB, K, V]
  b_h_new = b_h_new + jnp.matmul(
      b_kg.astype(jnp.float32).transpose(0, 2, 1),
      b_v_new,
      precision=get_chain_precision(),
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
    "native_out_layout",
    "two_pass_bf16",
  ],
)
@jaxtyping.jaxtyped
def chunk_kda_fwd_h_o_varlen(
    w: Float[Array, "H B T K"],
    u: Float[Array, "H B T V"],
    kg: Float[Array, "H B T K"],
    gk: Float[Array, "H B T GW"],
    q: Float[Array, "H_QK B T K"],
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
    native_out_layout: bool = False,
    two_pass_bf16: bool = False,
    packed_tok_meta: Int[Array, "B NT 8 BT"] | None = None,
    packed_chunk_keep: Int[Array, "B NT"] | None = None,
) -> tuple[
    Float[Array, "..."],
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
  H, B, T, K = w.shape
  H_QK = q.shape[0]
  assert H % H_QK == 0, f"H={H} must be divisible by H_QK={H_QK}"
  gqa_repeats = H // H_QK
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
    q_elem = jnp.dtype(q.dtype).itemsize
    wy_elem = jnp.dtype(w.dtype).itemsize
    o_elem = q_elem if native_out_layout else 4
    # scratch per head: h_state[K_PADSIZE*V_ALIGNED] in f32 + i/o buffers
    in_bytes = (
        BT * K_PADSIZE * q_elem // gqa_repeats
        + (2 * BT * K_PADSIZE + BT * V_ALIGNED + BT * BT) * wy_elem
        + BT
        * (K_PADSIZE if gk.shape[-1] == K else 1)
        * jnp.dtype(gk.dtype).itemsize
    )
    out_bytes = (
        BT * V_ALIGNED * o_elem
        + (K_PADSIZE * V_ALIGNED * wy_elem if store_h else 0)
        + (BT * V_ALIGNED * wy_elem if store_v_new else 0)
    )
    scratch_bytes = K_PADSIZE * V_ALIGNED * 4  # float32 accumulator
    per_head = in_bytes + out_bytes + scratch_bytes
    MB = estimate_mini_batch(per_head, H, multiple_of=gqa_repeats)  # pylint: disable=invalid-name
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"
    assert (
        MB % gqa_repeats == 0
    ), f"mini_batch={MB} must be divisible by gqa_repeats={gqa_repeats}"
  MB_QK = MB // gqa_repeats

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

  GW = gk.shape[-1]  # K for a per-channel gate, 1 for a scalar one
  w_t  = _pad_kdim_then_t(w,  K_PADSIZE - K)
  kg_t = _pad_kdim_then_t(kg, K_PADSIZE - K)
  gk_t = _pad_kdim_then_t(gk, (K_PADSIZE - K) if GW == K else 0)
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
  # (seqlens_ref, chunk_to_seq_ref[, keep_ref]).
  # Inputs are [H, B, T_alloc, X]; BlockSpec slices MB heads at h*MB.
  def _t_index_map(h, b, c, *_):
    return (h, b, c, 0)

  def _A_index_map(h, b, c, *_):
    return (h, b, c, 0)

  bspec_k = pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_t_index_map)
  bspec_q = pl.BlockSpec([MB_QK, 1, BT, K_PADSIZE], index_map=_t_index_map)
  # Narrow when the gate is scalar; identical to bspec_k otherwise.
  bspec_gk = pl.BlockSpec(
      [MB, 1, BT, K_PADSIZE if GW == K else GW], index_map=_t_index_map
  )
  bspec_v = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
  bspec_a = pl.BlockSpec([MB, 1, BT, BT],        index_map=_A_index_map)
  bspec_h0 = (
    pl.BlockSpec(
      [1, 1, MB, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref, *_: (
        b, chunk_to_seq_ref[b, c], h, 0, 0
      ),
    )
    if h0 is not None else None
  )

  # Output specs.
  if native_out_layout:
    assert V == V_ALIGNED, f"native_out_layout requires aligned V={V}"
    o_spec = pl.BlockSpec(
        [1, BT, MB * V],
        index_map=lambda h, b, c, *_: (b, c, h),
    )
    o_shape = jax.ShapeDtypeStruct([B, T_alloc, H * V], q.dtype)
  else:
    o_spec = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
    o_shape = jax.ShapeDtypeStruct([H, B, T_alloc, V_ALIGNED], jnp.float32)
  ht_spec = (
    pl.BlockSpec(
      [1, 1, MB, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref, *_: (
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
      index_map=lambda h, b, c, *_: (
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
  # This applies to the output matmuls only. The two serial-chain matmuls
  # (`w @ h` and the state update) use `get_chain_precision` instead.
  if q.dtype != jnp.bfloat16:
    _output_prec = jax.lax.Precision.HIGHEST
  elif two_pass_bf16:
    _output_prec = jax.lax.Precision.HIGH
  else:
    _output_prec = jax.lax.Precision.DEFAULT

  has_packed_meta = packed_tok_meta is not None
  if (packed_tok_meta is None) != (packed_chunk_keep is None):
    raise ValueError(
        "packed_tok_meta and packed_chunk_keep must be both provided or both"
        " None."
    )
  in_specs_h_o = [
      bspec_k,  # w
      bspec_v,  # u
      bspec_k,  # kg
      bspec_gk,  # gk
      bspec_q,  # q
      bspec_a,  # A
      bspec_h0,  # h0
  ]
  call_args_h_o: list[jax.Array | None] = [
      cu_seqlens.astype(jnp.int32),
      chunk_to_seq,
  ]
  if has_packed_meta:
    assert packed_tok_meta is not None and packed_chunk_keep is not None
    call_args_h_o.append(packed_chunk_keep.astype(jnp.int32))
    in_specs_h_o.append(
        pl.BlockSpec(
            [1, 1, 1, 8, BT],
            index_map=lambda h, b, c, *_: (0, b, c, 0, 0),
        )
    )
  call_args_h_o.extend([w_t, u_t, kg_t, gk_t, q_t, A_t, h0])
  if has_packed_meta:
    assert packed_tok_meta is not None
    call_args_h_o.append(packed_tok_meta[None, ...])

  total_h_o_chunks = H * B * NT
  h_o_flops = total_h_o_chunks * (6 * BT * K * V + 2 * BT * BT * V)
  h_o_transcendentals = total_h_o_chunks * (BT + 1) * GW
  h_o_in_tensors = (w_t, u_t, kg_t, gk_t, q_t, A_t, h0)
  h_o_out_shapes = (o_shape, ht_shape, h_out_shape, v_new_out_shape)
  h_o_bytes = sum(
      int(x.size) * jnp.dtype(x.dtype).itemsize
      for x in h_o_in_tensors
      if x is not None
  ) + sum(
      int(s.size) * jnp.dtype(s.dtype).itemsize
      for s in h_o_out_shapes
      if s is not None
  )
  h_o_cost_estimate = pl.CostEstimate(
      flops=h_o_flops,
      transcendentals=h_o_transcendentals,
      bytes_accessed=h_o_bytes,
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
          GQA_REPEATS=gqa_repeats,
          NATIVE_OUT_LAYOUT=native_out_layout,
          OUTPUT_PRECISION=_output_prec,
          HAS_PACKED_META=has_packed_meta,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3 if has_packed_meta else 2,
          grid=grid,
          in_specs=in_specs_h_o,
          out_specs=[o_spec, ht_spec, h_out_spec, v_new_out_spec],
          scratch_shapes=[scratch],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[o_shape, ht_shape, h_out_shape, v_new_out_shape],
      cost_estimate=h_o_cost_estimate,
      interpret=interpret,
  )(*call_args_h_o)

  if native_out_layout:
    o_out = o_out.reshape(B, T_alloc, H, V)
  else:
    # Post-process: o is [H, B, T, V_ALIGNED]
    if V_ALIGNED > V:
      o_out = o_out[..., :V]
    o_out = o_out.astype(q.dtype)

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


def _local_cp_fwd_summary_kernel(
    *refs,
    BT: int,
    MB: int,
    NT: int,
    K_PADSIZE: int,
    V_ALIGNED: int,
    HAS_KEEP: bool,
):
  """Pallas forward chunk-state summary scan keeping `(S, M)` in VMEM."""
  if HAS_KEEP:
    (
        keep_ref,
        kg_ref,
        w_ref,
        u_ref,
        gk_ref,
        s_out_ref,
        m_out_ref,
        s_scratch,
        m_scratch,
    ) = refs
  else:
    keep_ref = None
    (
        kg_ref,
        w_ref,
        u_ref,
        gk_ref,
        s_out_ref,
        m_out_ref,
        s_scratch,
        m_scratch,
    ) = refs

  i_b = pl.program_id(1)
  i_c = pl.program_id(2)

  @pl.when(i_c == 0)
  def _init():
    s_scratch[...] = jnp.zeros((MB, K_PADSIZE, V_ALIGNED), dtype=jnp.float32)
    m_scratch[...] = jnp.broadcast_to(
        jnp.eye(K_PADSIZE, dtype=jnp.float32), (MB, K_PADSIZE, K_PADSIZE)
    )

  s_acc = s_scratch[...]
  m_acc = m_scratch[...]
  b_kg = kg_ref[:, 0, :].astype(jnp.float32)
  b_w = w_ref[:, 0, :].astype(jnp.float32)
  b_u = u_ref[:, 0, :].astype(jnp.float32)
  b_gk_last = gk_ref[:, 0, :][:, BT - 1, :].astype(jnp.float32)
  if HAS_KEEP:
    assert keep_ref is not None
    keep_f = keep_ref[i_b, i_c].astype(jnp.float32)
    decay = keep_f * jnp.exp2(keep_f * b_gk_last)
  else:
    decay = jnp.exp2(b_gk_last)

  prec = get_chain_precision()
  v_new_s = b_u - jnp.matmul(
      b_w, s_acc, precision=prec, preferred_element_type=jnp.float32
  )
  kg_t = b_kg.transpose(0, 2, 1)
  s_next = decay[:, :, None] * s_acc + jnp.matmul(
      kg_t, v_new_s, precision=prec, preferred_element_type=jnp.float32
  )
  w_m = jnp.matmul(
      b_w, m_acc, precision=prec, preferred_element_type=jnp.float32
  )
  m_next = decay[:, :, None] * m_acc - jnp.matmul(
      kg_t, w_m, precision=prec, preferred_element_type=jnp.float32
  )
  s_scratch[...] = s_next
  m_scratch[...] = m_next

  @pl.when(i_c == NT - 1)
  def _store():
    s_out_ref[0, :, :] = s_scratch[...].astype(s_out_ref.dtype)
    m_out_ref[0, :, :] = m_scratch[...].astype(m_out_ref.dtype)


def compute_local_cp_fwd_summary(
    *,
    kg: jax.Array,
    w: jax.Array,
    u: jax.Array,
    gk: jax.Array,
    chunk_size: int,
    packed_chunk_keep: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Computes local rank affine state summary `(S_local, M_local)` in fp32."""
  H, B, T, K = kg.shape
  V = u.shape[-1]
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
  act_elem = jnp.dtype(kg.dtype).itemsize
  gk_elem = jnp.dtype(gk.dtype).itemsize
  per_head = (
      2 * BT * (2 * K_PADSIZE * act_elem + V_ALIGNED * act_elem + GW * gk_elem)
      + 3 * (K_PADSIZE * V_ALIGNED + K_PADSIZE * K_PADSIZE) * 4
  )
  MB = estimate_mini_batch(per_head, H)

  def _pad_last(x: jax.Array, pad: int) -> jax.Array:
    return jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, pad))) if pad > 0 else x

  kg_t = _pad_last(kg, K_PADSIZE - K)
  w_t = _pad_last(w, K_PADSIZE - K)
  u_t = _pad_last(u, V_ALIGNED - V)
  gk_t = _pad_last(gk, (K_PADSIZE - K) if GW == K else 0)
  gw_block = K_PADSIZE if GW == K else GW

  def _in_map(h, b, c, *_):
    return (h, b, c, 0)

  def _out_map(h, b, c, *_):
    del c
    return (b, h, 0, 0)

  has_keep = packed_chunk_keep is not None
  in_specs = [
      pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_in_map),
      pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_in_map),
      pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_in_map),
      pl.BlockSpec([MB, 1, BT, gw_block], index_map=_in_map),
  ]
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
  call_args = (
      [packed_chunk_keep.astype(jnp.int32)] if has_keep else []
  ) + [kg_t, w_t, u_t, gk_t]
  total_chunks = H * B * NT
  cost = pl.CostEstimate(
      flops=total_chunks * 4 * BT * K_PADSIZE * (V_ALIGNED + K_PADSIZE),
      transcendentals=total_chunks * gw_block,
      bytes_accessed=sum(
          int(x.size) * jnp.dtype(x.dtype).itemsize for x in call_args
      )
      + sum(int(s.size) * 4 for s in out_shapes),
  )
  s_pad, m_pad = pl.pallas_call(
      functools.partial(
          _local_cp_fwd_summary_kernel,
          BT=BT,
          MB=MB,
          NT=NT,
          K_PADSIZE=K_PADSIZE,
          V_ALIGNED=V_ALIGNED,
          HAS_KEEP=has_keep,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1 if has_keep else 0,
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
  return s_pad[..., :K, :V], m_pad[..., :K, :K]


@jaxtyping.jaxtyped
def chunk_kda_fwd_custom(
    q: Float[Array, "H B T_ALIGNED K"],
    k: Float[Array, "H B T_ALIGNED K"],
    v: Float[Array, "H B T_ALIGNED V"],
    g: Float[Array, "H B T_ALIGNED GW"],
    beta: Float[Array, "H B T_ALIGNED"],
    a_log: Float[Array, "H"] | None = None,
    delta_time_bias: Float[Array, "H*GW"] | None = None,
    scale: float | None = None,
    initial_state: (
        Float[Array, "B H K V"] | Float[Array, "B N H K V"] | None
    ) = None,
    output_final_state: bool = False,
    use_gate_in_kernel: bool = False,
    segment_ids: Int[Array, "B T"] | None = None,
    safe_gate: bool = True,
    per_channel_gate: bool = True,
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
    conv_weight_q: Float[Array, "H_Q W K"] | None = None,
    conv_weight_k: Float[Array, "H_K W K"] | None = None,
    conv_weight_v: Float[Array, "H W V"] | None = None,
    conv_bias_q: Float[Array, "H_Q K"] | None = None,
    conv_bias_k: Float[Array, "H_K K"] | None = None,
    conv_bias_v: Float[Array, "H V"] | None = None,
    use_conv1d_in_kernel: bool = False,
    use_qk_l2norm: bool = False,
    mini_batch: int | None = None,
    two_pass_bf16: bool = False,
    cp_checkpoint_name: str | None = None,
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
  _is_varlen = cu_seqlens is not None
  _inplace_cp = _cp_active and not _is_varlen
  cp_metadata = None
  if _cp_active and _is_varlen:
    assert context_parallel_metadata is not None
    cp_metadata = (
        context_parallel_metadata.is_first_rank,
        context_parallel_metadata.is_last_rank,
        context_parallel_metadata.pre_num_ranks,
        context_parallel_metadata.post_num_ranks,
    )
    if any(value is None for value in cp_metadata):
      raise ValueError("Enabled CP context is missing derived rank metadata.")

  cp_ckpt = (
      (
          cp_checkpoint_name
          or context_parallel_metadata.cp_residual_checkpoint_name
      )
      if _inplace_cp and context_parallel_metadata is not None
      else None
  )
  halo_len_raw = (
      max(
          conv_weight_q.shape[1],
          conv_weight_k.shape[1],
          conv_weight_v.shape[1],
      )
      - 1
      if (
          _inplace_cp
          and use_conv1d_in_kernel
          and conv_weight_q is not None
          and conv_weight_k is not None
          and conv_weight_v is not None
      )
      else 0
  )
  packed_tok_meta = None
  packed_chunk_keep = None
  cp_seg_halo = None
  if segment_ids is not None and not _is_varlen and aligned_segment_ids is None:
    if _inplace_cp:
      assert context_parallel_metadata is not None
      seg_canon, act_doc, is_new_doc, halo_seg_canon, cp_seg_halo = (
          cp_canonicalize_packed_segment_ids(
              segment_ids,
              axis_name=context_parallel_metadata.axis_name,
              cp_size=context_parallel_metadata.cp_size,
              halo_len=halo_len_raw,
              cp_checkpoint_name=cp_ckpt,
          )
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
          segment_ids
      )
      packed_tok_meta, packed_chunk_keep = compute_packed_chunk_metadata(
          seg_canon,
          act_doc,
          is_new_doc,
          BT,
          has_initial_state=(initial_state is not None),
      )

  # ------------------------------------------------------------------
  # Step 1 + 2 (Fused): Gate cumsum + Intra-chunk solve
  # ------------------------------------------------------------------
  q_raw_in, k_raw_in, v_raw_in = q, k, v
  rank_halo_q = rank_halo_k = rank_halo_v = None
  if _inplace_cp and use_conv1d_in_kernel and halo_len_raw > 0:
    assert context_parallel_metadata is not None
    axis_name = context_parallel_metadata.axis_name
    cp_size = context_parallel_metadata.cp_size
    rank_halo_q = cp_shift_right(q[:, :, -_CONV_HALO:, :], axis_name, cp_size)
    rank_halo_k = cp_shift_right(k[:, :, -_CONV_HALO:, :], axis_name, cp_size)
    rank_halo_v = cp_shift_right(v[:, :, -_CONV_HALO:, :], axis_name, cp_size)
    if cp_ckpt is not None:
      rank_halo_q = ad_checkpoint.checkpoint_name(rank_halo_q, name=cp_ckpt)
      rank_halo_k = ad_checkpoint.checkpoint_name(rank_halo_k, name=cp_ckpt)
      rank_halo_v = ad_checkpoint.checkpoint_name(rank_halo_v, name=cp_ckpt)
    q_raw_in = jnp.concatenate(
        [rank_halo_q[:, :, -halo_len_raw:, :], q], axis=2
    )
    k_raw_in = jnp.concatenate(
        [rank_halo_k[:, :, -halo_len_raw:, :], k], axis=2
    )
    v_raw_in = jnp.concatenate(
        [rank_halo_v[:, :, -halo_len_raw:, :], v], axis=2
    )

  scale_val = 1.0 / math.sqrt(K) if scale is None else scale
  (
      w,
      u,
      kg,
      Aqk,
      Akk,
      g_cumsum,
      q_norm,
      k_norm,
      v_silu,
      q_rstd_conv,
      k_rstd_conv,
  ) = kda_fwd_intra_fused(
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
      per_channel_gate=per_channel_gate,
      disable_recompute=save_for_backward,
      cumsum_scale=RCP_LN2,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
      conv_weight_q=conv_weight_q,
      conv_weight_k=conv_weight_k,
      conv_weight_v=conv_weight_v,
      conv_bias_q=conv_bias_q,
      conv_bias_k=conv_bias_k,
      conv_bias_v=conv_bias_v,
      use_conv1d_in_kernel=use_conv1d_in_kernel,
      use_qk_l2norm=(use_qk_l2norm and use_conv1d_in_kernel),
      return_conv_intermediates=(use_conv1d_in_kernel and return_residuals),
      aligned_segment_ids=aligned_segment_ids,
      mini_batch=mini_batch,
      store_akk=return_residuals,
      two_pass_bf16=two_pass_bf16,
      packed_tok_meta=packed_tok_meta,
      rank_halo_q=rank_halo_q,
      rank_halo_k=rank_halo_k,
      rank_halo_v=rank_halo_v,
  )
  if use_conv1d_in_kernel:
    assert q_norm is not None
    q_stage34 = q_norm
    if return_residuals:
      assert k_norm is not None and v_silu is not None
      q_res, k_res, v_res = q_norm, k_norm, v_silu
      q_rstd_res, k_rstd_res = q_rstd_conv, k_rstd_conv
      q_raw_res, k_raw_res, v_raw_res = q_raw_in, k_raw_in, v_raw_in
    else:
      q_res = k_res = v_res = q_rstd_res = k_rstd_res = None
      q_raw_res = k_raw_res = v_raw_res = None
  else:
    q_stage34 = q
    q_res, k_res, v_res = q, k, v
    q_rstd_res, k_rstd_res = q_rstd, k_rstd
    q_raw_res = k_raw_res = v_raw_res = None

  if _inplace_cp:
    assert context_parallel_metadata is not None
    s_local, m_local = compute_local_cp_fwd_summary(
        kg=kg,
        w=w,
        u=u,
        gk=g_cumsum,
        chunk_size=BT,
        packed_chunk_keep=packed_chunk_keep,
    )
    initial_state = cp_associative_scan_fwd(
        s_local,
        m_local,
        axis_name=context_parallel_metadata.axis_name,
        cp_size=context_parallel_metadata.cp_size,
    )
    if cp_ckpt is not None:
      initial_state = ad_checkpoint.checkpoint_name(
          initial_state, name=cp_ckpt
      )
  elif _cp_active:
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
      q=q_stage34,
      A=Aqk,
      cu_seqlens=stage34_cu_seqlens,
      chunk_indices=stage34_chunk_indices,
      initial_state=stage34_initial_state,
      output_final_state=output_final_state,
      scale=scale_val,
      chunk_size=BT,
      store_h=disable_recompute,
      store_v_new=disable_recompute,
      mini_batch=mini_batch,
      two_pass_bf16=two_pass_bf16,
      packed_tok_meta=packed_tok_meta,
      packed_chunk_keep=packed_chunk_keep,
  )
  if not _is_varlen and final_state is not None:
    final_state = final_state[:, 0]

  # ------------------------------------------------------------------
  # Drop intermediates that backward will recompute or never consume.
  # ------------------------------------------------------------------
  if not save_for_backward:
    w, u, kg, v_new = None, None, None, None
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
  assert q_res is not None and k_res is not None and v_res is not None
  residuals = KdaResiduals(
      q=q_res,
      k=k_res,
      v=v_res,
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
      q_rstd=q_rstd_res,
      k_rstd=k_rstd_res,
      cu_seqlens=cu_seqlens,
      aligned_cu_seqlens=aligned_cu_seqlens,
      chunk_indices=chunk_indices,
      aligned_segment_ids=aligned_segment_ids,
      segment_ids=segment_ids,
      cp_metadata=cp_metadata,
      v_new=v_new,
      q_raw=q_raw_res,
      k_raw=k_raw_res,
      v_raw=v_raw_res,
      conv_weight_q=conv_weight_q if use_conv1d_in_kernel else None,
      conv_weight_k=conv_weight_k if use_conv1d_in_kernel else None,
      conv_weight_v=conv_weight_v if use_conv1d_in_kernel else None,
      conv_bias_q=conv_bias_q if use_conv1d_in_kernel else None,
      conv_bias_k=conv_bias_k if use_conv1d_in_kernel else None,
      conv_bias_v=conv_bias_v if use_conv1d_in_kernel else None,
      chunk_size=BT,
      two_pass_bf16=two_pass_bf16,
      cp_seg_halo=cp_seg_halo,
  )
  return output, residuals


@jaxtyping.jaxtyped
def chunk_gdn_fwd_custom(
    query: Float[Array, "B T H_QK K"] | None,
    key: Float[Array, "B T H_QK K"] | None,
    value: Float[Array, "B T H_V V"] | None,
    gate: Float[Array, "B T H_V"],
    beta: Float[Array, "B T H_V"],
    *,
    qkv: Float[Array, "B T C_QKV"] | None = None,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_weight: Float[Array, "W C_QKV"] | None = None,
    conv_bias: Float[Array, "C_QKV"] | None = None,
    a_log: Float[Array, "H_V"] | None = None,
    delta_time_bias: Float[Array, "H_V"] | None = None,
    scale: float,
    initial_state: Float[Array, "B H_V K V"] | None = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = True,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = True,
    disable_recompute: bool = True,
    chunk_size: int = 64,
    segment_ids: Int[Array, "B T"] | None = None,
    context_parallel_metadata: ContextParallelMetadata | None = None,
    return_residuals: bool = False,
    mini_batch: int | None = None,
    two_pass_bf16: bool = False,
    cp_checkpoint_name: str | None = None,
    precomputed_initial_state: Float[Array, "B H_V K V"] | None = None,
    precomputed_qkv_conv: Float[Array, "B T C_QKV"] | None = None,
    precomputed_qkv_raw: Float[Array, "..."] | None = None,
    precomputed_cp_seg_halo: (
        tuple[Int[Array, "B"], Int[Array, "B T_HALO"] | None] | None
    ) = None,
) -> tuple[
    tuple[Float[Array, "B T H_V V"], Float[Array, "B H_V K V"] | None],
    GdnResiduals | None,
]:
  """Natural-layout `[B, T, ...]` GDN forward sharing KDA Pallas TPU kernels."""
  save_for_backward = return_residuals and disable_recompute
  H_QK = num_k_heads
  H_V = num_v_heads
  K = head_k_dim
  V = head_v_dim
  q_size = H_QK * K
  k_size = H_QK * K
  v_size = H_V * V
  BT = chunk_size
  _cp_active = (
      context_parallel_metadata is not None
      and context_parallel_metadata.is_cp_enabled
  )
  cp_ckpt = (
      (
          cp_checkpoint_name
          or context_parallel_metadata.cp_residual_checkpoint_name
      )
      if _cp_active and context_parallel_metadata is not None
      else None
  )
  halo_len = (
      conv_weight.shape[0] - 1
      if (
          _cp_active
          and precomputed_qkv_conv is None
          and conv_weight is not None
          and conv_weight.shape[0] > 1
      )
      else 0
  )

  packed_tok_meta = None
  packed_chunk_keep = None
  seg_canon = None
  seg_canon_ext = None
  cp_seg_halo = precomputed_cp_seg_halo
  if segment_ids is not None:
    if _cp_active:
      assert context_parallel_metadata is not None
      seg_canon, act_doc, is_new_doc, halo_seg_canon, cp_seg_halo = (
          cp_canonicalize_packed_segment_ids(
              segment_ids,
              axis_name=context_parallel_metadata.axis_name,
              cp_size=context_parallel_metadata.cp_size,
              halo_len=halo_len,
              cp_checkpoint_name=cp_ckpt,
              precomputed_cp_halo=precomputed_cp_seg_halo,
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
          segment_ids
      )
      seg_canon_ext = seg_canon
      packed_tok_meta, packed_chunk_keep = compute_packed_chunk_metadata(
          seg_canon,
          act_doc,
          is_new_doc,
          BT,
          has_initial_state=(initial_state is not None),
      )

  qkv_raw = None
  qkv_conv = None
  if precomputed_qkv_conv is not None:
    B, T, _ = precomputed_qkv_conv.shape
    act_dtype = precomputed_qkv_conv.dtype
    qkv_conv = precomputed_qkv_conv
    qkv_raw = precomputed_qkv_raw
  elif qkv is not None or conv_weight is not None:
    if qkv is not None:
      B, T, _ = qkv.shape
      act_dtype = qkv.dtype
      qkv_in = qkv
    else:
      assert query is not None and key is not None and value is not None
      B, T, _, _ = query.shape
      act_dtype = query.dtype
      qkv_in = jnp.concatenate(
          [
              query.reshape(B, T, q_size),
              key.reshape(B, T, k_size),
              value.reshape(B, T, v_size),
          ],
          axis=-1,
      )
    if conv_weight is not None:
      if halo_len > 0:
        assert context_parallel_metadata is not None
        qkv_halo = cp_shift_right(
            qkv_in[:, -halo_len:, :],
            context_parallel_metadata.axis_name,
            context_parallel_metadata.cp_size,
        )
        if cp_ckpt is not None:
          qkv_halo = ad_checkpoint.checkpoint_name(qkv_halo, name=cp_ckpt)
        qkv_ext = jnp.concatenate([qkv_halo, qkv_in], axis=1)
        qkv_raw = qkv_ext if return_residuals else None
        qkv_conv = causal_conv1d_silu_fwd_bt(
            qkv_ext,
            conv_weight,
            conv_bias,
            canonical_segment_ids=seg_canon_ext,
        )[:, halo_len:, :]
      else:
        qkv_raw = qkv_in if return_residuals else None
        qkv_conv = causal_conv1d_silu_fwd_bt(
            qkv_in, conv_weight, conv_bias, canonical_segment_ids=seg_canon
        )
    else:
      qkv_conv = qkv_in
  else:
    assert query is not None and key is not None and value is not None
    B, T, _, _ = query.shape
    act_dtype = query.dtype

  interpret = get_interpret()
  can_native_qkv = (
      K % 128 == 0 and V % 128 == 0 and BT % 8 == 0 and not interpret
  )
  can_native_out = V % 128 == 0 and BT % 8 == 0 and not interpret

  if qkv_conv is not None and can_native_qkv:
    q_intra = k_intra = v_intra = None
    qkv_conv_intra = qkv_conv
    native_bt_qkv = False
  elif qkv_conv is None and can_native_qkv:
    q_intra, k_intra, v_intra = query, key, value
    qkv_conv_intra = None
    native_bt_qkv = True
  elif qkv_conv is not None:
    q_intra = (
        qkv_conv[:, :, :q_size].reshape(B, T, H_QK, K).transpose(2, 0, 1, 3)
    )
    k_intra = (
        qkv_conv[:, :, q_size : q_size + k_size]
        .reshape(B, T, H_QK, K)
        .transpose(2, 0, 1, 3)
    )
    v_intra = (
        qkv_conv[:, :, q_size + k_size :]
        .reshape(B, T, H_V, V)
        .transpose(2, 0, 1, 3)
    )
    qkv_conv_intra = None
    native_bt_qkv = False
  else:
    assert query is not None and key is not None and value is not None
    q_intra = query.transpose(2, 0, 1, 3)
    k_intra = key.transpose(2, 0, 1, 3)
    v_intra = value.transpose(2, 0, 1, 3)
    qkv_conv_intra = None
    native_bt_qkv = False

  g_hbt = gate.transpose(2, 0, 1)[..., None]
  beta_hbt = beta.transpose(2, 0, 1)
  if use_beta_sigmoid_in_kernel:
    beta_act = jax.nn.sigmoid(beta_hbt.astype(jnp.float32))
  else:
    beta_act = beta_hbt

  need_q_from_intra = (
      use_qk_l2norm
      or qkv_conv_intra is not None
      or native_bt_qkv
      or segment_ids is not None
  )
  store_akk = save_for_backward and BT < 128
  (
      w,
      u,
      kg,
      Aqk,
      Akk,
      g_cumsum,
      q_stage34,
      _,
      _,
      _,
      _,
  ) = kda_fwd_intra_fused(
      q=q_intra,
      k=k_intra,
      v=v_intra,
      g=g_hbt,
      beta=beta_act,
      scale=scale,
      chunk_size=BT,
      safe_gate=safe_gate,
      per_channel_gate=False,
      disable_recompute=save_for_backward,
      cumsum_scale=RCP_LN2,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=None,
      use_conv1d_in_kernel=False,
      use_qk_l2norm=use_qk_l2norm,
      return_conv_intermediates=False,
      mini_batch=mini_batch,
      qkv_conv=qkv_conv_intra,
      num_k_heads=H_QK,
      head_k_dim=K,
      head_v_dim=V,
      native_bt_qkv=native_bt_qkv,
      write_q_norm=need_q_from_intra,
      store_akk=store_akk,
      two_pass_bf16=two_pass_bf16,
      packed_tok_meta=packed_tok_meta,
  )
  if not need_q_from_intra:
    q_stage34 = q_intra
  assert q_stage34 is not None

  if _cp_active:
    if precomputed_initial_state is not None:
      initial_state = precomputed_initial_state
    else:
      assert context_parallel_metadata is not None
      s_local, m_local = compute_local_cp_fwd_summary(
          kg=kg,
          w=w,
          u=u,
          gk=g_cumsum,
          chunk_size=BT,
          packed_chunk_keep=packed_chunk_keep,
      )
      initial_state = cp_associative_scan_fwd(
          s_local,
          m_local,
          axis_name=context_parallel_metadata.axis_name,
          cp_size=context_parallel_metadata.cp_size,
      )
      if cp_ckpt is not None:
        initial_state = ad_checkpoint.checkpoint_name(
            initial_state, name=cp_ckpt
        )

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

  o_raw, final_state, h, _ = chunk_kda_fwd_h_o_varlen(
      w=w,
      u=u,
      kg=kg,
      gk=g_cumsum,
      q=q_stage34,
      A=Aqk,
      cu_seqlens=stage34_cu_seqlens,
      chunk_indices=stage34_chunk_indices,
      initial_state=stage34_initial_state,
      output_final_state=output_final_state,
      scale=scale,
      chunk_size=BT,
      store_h=save_for_backward,
      store_v_new=False,
      mini_batch=mini_batch,
      native_out_layout=can_native_out,
      two_pass_bf16=two_pass_bf16,
      packed_tok_meta=packed_tok_meta,
      packed_chunk_keep=packed_chunk_keep,
  )
  if final_state is not None:
    final_state = final_state[:, 0]

  out_bt = (
      o_raw.astype(act_dtype)
      if can_native_out
      else o_raw.transpose(1, 2, 0, 3).astype(act_dtype)
  )
  output = (out_bt, final_state)
  if not return_residuals:
    return output, None

  save_split_qkv = save_for_backward and qkv_conv is None
  if not save_for_backward:
    if not _cp_active:
      qkv_conv = None
      qkv_raw = None
    g_cumsum = None
    Akk = None
    h = None
  residuals = GdnResiduals(
      qkv_conv=qkv_conv,
      q=query if save_split_qkv else None,
      k=key if save_split_qkv else None,
      v=value if save_split_qkv else None,
      beta=beta_act,
      g_cumsum=g_cumsum,
      akk=Akk,
      h=h,
      initial_state=initial_state,
      g_org=g_hbt if use_gate_in_kernel else None,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      qkv_raw=qkv_raw,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      segment_ids=segment_ids,
      g_dtype_marker=jnp.zeros((), dtype=gate.dtype),
      chunk_size=BT,
      beta_dtype=jnp.dtype(beta.dtype).name,
      two_pass_bf16=two_pass_bf16,
      cp_seg_halo=cp_seg_halo,
  )
  return output, residuals
