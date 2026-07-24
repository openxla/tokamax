# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""KDA helpers shared by the Pallas TPU forward and backward paths."""

from __future__ import annotations

import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member

from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda.utils import (
  align_up,
  cdiv,
  exp,
  exp2,
  get_interpret,
  get_tpu_config,
  pad_to_multiple,
)
RCP_LN2 = 1.0 / math.log(2)

# =============================================================================
# Mini-batch sizing and shared forward recurrence
# =============================================================================


def estimate_mini_batch(
    per_tile_bytes: int,
    total: int,
    *,
    max_mb: int = 16,
    vmem_budget: int | None = None,
    align_minor: int | None = None,
) -> int:
  """Estimate the optimal mini-batch size to maximise VMEM utilisation.

  This mirrors the backward auto-tune pattern used throughout the KDA kernels:
  compute the largest MB that fits within the hardware VMEM budget, cap it,
  and then adjust downward so that ``total`` is evenly divisible by ``MB``.

  Args:
      per_tile_bytes: Estimated VMEM footprint (bytes) for **one** tile/head.
      total: Number of tiles (or heads) to partition.
      max_mb: Upper bound on MB (default 16).
      vmem_budget: VMEM budget in bytes. ``None`` (default) queries
          ``get_tpu_config().vmem_limit_bytes`` at call time.
      align_minor: TPU block_align_minor constraint. ``None`` (default)
          queries ``get_tpu_config().block_align_minor``.  When provided,
          the function prefers MB values that satisfy this alignment.

  Returns:
      Mini-batch size ``MB`` such that ``total % MB == 0`` (best-effort).
  """
  if vmem_budget is None or align_minor is None:
    hw = get_tpu_config()
    if vmem_budget is None:
      vmem_budget = hw.vmem_limit_bytes
    if align_minor is None:
      align_minor = hw.block_align_minor

  per_tile_bytes = max(1, per_tile_bytes)
  MB = max(1, vmem_budget // per_tile_bytes)
  MB = max(1, min(MB, total, max_mb))

  # Try to find an MB that divides total evenly.
  while total % MB != 0 and MB > 1:
    MB -= 1

  return MB


# =============================================================================
# Chunk-local cumsum helpers
# =============================================================================

# =============================================================================
# Fixed-length: Hillis-Steele scan (optimal for head_first=False)
# =============================================================================


def _hillis_steele_scan(
  x: jax.Array,
  axis: int,
) -> jax.Array:
  """Inclusive prefix scan with O(log n) parallel depth along ``axis``.

  Each iteration adds values from ``2 ** d`` positions to the left, so the
  dependency horizon doubles every round instead of advancing one element at a
  time as in a sequential cumsum.

  Args:
      x:    input tensor.
      axis: axis along which to compute the inclusive prefix scan.

  Returns:
      Tensor with the same shape as ``x``.
  """
  axis = axis % x.ndim
  length = x.shape[axis]
  assert length >= 1, f"scan axis must be non-empty, got length={length}"

  if length == 1:
    return x

  num_steps = int(math.log2(length))
  assert length == 1 << num_steps, (
    f"Hillis-Steele scan requires power-of-2 length, got {length}"
  )

  acc = x
  for i in range(num_steps):
    stride = 1 << i
    prefix = jax.lax.slice_in_dim(acc, 0, stride, axis=axis)
    suffix = jax.lax.slice_in_dim(acc, stride, length, axis=axis)
    shifted = jax.lax.slice_in_dim(acc, 0, length - stride, axis=axis)
    acc = jax.lax.concatenate([prefix, suffix + shifted], dimension=axis)

  return acc

# =============================================================================
# Public API
# =============================================================================

@functools.partial(
  jax.jit,
  static_argnames=["chunk_size", "reverse", "scale", "output_dtype"],
)
@jaxtyping.jaxtyped
def chunk_local_cumsum_vector(
  g: Float[Array, "H B T K"],
  chunk_size: int,
  reverse: bool = False,
  scale: float | None = None,
  output_dtype: jax.typing.DTypeLike = jnp.float32,
) -> Float[Array, "H B T K"]:
  """Computes prefix or suffix sums independently within each chunk."""
  assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
    "chunk_size must be power of 2"
  )

  BT = chunk_size
  out_dtype = output_dtype or g.dtype

  H, B, T, S = g.shape

  NT = (T + BT - 1) // BT
  T_padded = NT * BT
  pad_t = T_padded - T

  g_work = jnp.pad(g, ((0, 0), (0, 0), (0, pad_t), (0, 0))) if pad_t > 0 else g
  g_chunked = g_work.reshape(H, B, NT, BT, S).astype(jnp.float32)

  if reverse:
    cum_mask = jnp.triu(jnp.ones((BT, BT), dtype=jnp.float32))
  else:
    cum_mask = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))

  o_chunked = jnp.einsum(
    "ij,hbnjs->hbnis",
    cum_mask,
    g_chunked,
    precision=jax.lax.Precision.HIGHEST,
  )
  o = o_chunked.reshape(H, B, T_padded, S)[:, :, :T, :]

  if scale is not None:
    o = o * scale

  return o.astype(out_dtype)

# =============================================================================
# Delta-rule hidden-state kernels
# =============================================================================

# ---------------------------------------------------------------------------
# Pallas TPU kernels for delta-rule inter-chunk state propagation
# ---------------------------------------------------------------------------


# ── Varlen Pallas kernel ────────────────────────────────────────────────────


def _chunk_gated_delta_rule_fwd_varlen_kernel(
  seqlens_ref,
  chunk_to_seq_ref,
  k_ref,
  v_ref,
  w_ref,
  g_ref,
  gk_ref,
  h0_ref,
  h_ref,
  v_new_ref,
  ht_ref,
  scratch_ref,
  *,
  NT,
  USE_G,
  USE_GK,
  USE_INITIAL_STATE,
  STORE_FINAL_STATE,
  SAVE_NEW_VALUE,
  USE_EXP2,
  MINI_BATCH: int = 1,
):
  """Delta-rule inter-chunk state forward pass kernel for varlen.

  Grid is (H // MB, B, NT). For each program point (h_group, i_b, i_c):
    - Processes MB heads per grid point via batch-vectorized ops.
    - i_b is the batch index, i_c is the chunk index within that batch.
    - seq_idx = chunk_to_seq[i_b, i_c] identifies which sequence this chunk
      belongs to within batch i_b.
    - At t0 == bos: init scratch (h0 or zeros) for this sequence.
    - At t0 + BT >= eos: store final state for this sequence.

  Args:
      MINI_BATCH: int — number of heads processed per grid point (MB).
  """
  i_b = pl.program_id(1)
  i_c = pl.program_id(2)
  seq_idx = chunk_to_seq_ref[i_b, i_c]

  bos = seqlens_ref[i_b, seq_idx]
  eos = seqlens_ref[i_b, seq_idx + 1]

  BT = k_ref.shape[2]
  t0 = i_c * BT
  K, V = k_ref.shape[-1], v_ref.shape[-1]

  @pl.when(t0 == bos)
  def _():
    if USE_INITIAL_STATE:
      scratch_ref[:] = h0_ref[0, 0].astype(jnp.float32)  # [MB, K, V]
    else:
      scratch_ref[:] = jnp.zeros_like(scratch_ref[:])

  # h output shape: [MB, 1, 1, K, V] — MB is the H dimension
  h_ref[:, 0, 0] = scratch_ref[:].astype(h_ref.dtype)  # [MB, K, V]

  b_k = k_ref[:,0,:].astype(jnp.float32)    # [MB, BT, K]
  b_w = w_ref[:,0,:].astype(jnp.float32)    # [MB, BT, K]
  b_u = v_ref[:,0,:].astype(jnp.float32)    # [MB, BT, V]

  # v_new = v - w @ state: [MB, BT, K] @ [MB, K, V] → [MB, BT, V]
  b_v = b_u - jnp.matmul(
    b_w,
    scratch_ref[:],
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )
  if SAVE_NEW_VALUE:
    v_new_ref[:,0,:] = b_v.astype(v_new_ref.dtype)

  if USE_G:
    b_g = g_ref[:, 0, :, 0].astype(jnp.float32)  # [MB, BT]
    b_g_last = b_g[:, BT - 1]                      # [MB]
    if USE_EXP2:
      b_v = b_v * exp2(b_g_last[:, None] - b_g)[:, :, None]
      scratch_ref[:] = scratch_ref[:] * exp2(b_g_last)[:, None, None]
    else:
      b_v = b_v * exp(b_g_last[:, None] - b_g)[:, :, None]
      scratch_ref[:] = scratch_ref[:] * exp(b_g_last)[:, None, None]

  if USE_GK:
    b_gk_last = gk_ref[:, 0, BT - 1].astype(jnp.float32)  # [MB, K]
    if USE_EXP2:
      scratch_ref[:] = scratch_ref[:] * exp2(b_gk_last)[:, :, None]
    else:
      scratch_ref[:] = scratch_ref[:] * exp(b_gk_last)[:, :, None]

  # state += k^T @ v: [MB, K, BT] @ [MB, BT, V] → [MB, K, V]
  scratch_ref[:] = scratch_ref[:] + jnp.matmul(
    b_k.transpose(0, 2, 1),
    b_v,
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )

  @pl.when(t0 + BT >= eos)
  def _():
    if STORE_FINAL_STATE:
      ht_ref[0, 0] = scratch_ref[:].astype(ht_ref.dtype)


@functools.partial(
  jax.jit,
  static_argnames=[
    "output_final_state",
    "chunk_size",
    "BV",
    "save_new_value",
    "use_exp2",
    "mini_batch",
  ],
)
@jaxtyping.jaxtyped
def _chunk_gated_delta_rule_fwd_varlen(
  k: Float[Array, "H B T K"],
  w: Float[Array, "H B T K"],
  v: Float[Array, "H B T V"],
  seqlens: Int[Array, "B N_CU"],
  chunk_indices: Int[Array, "B NT 2"],
  g: Float[Array, "H B T"] | None = None,
  gk: Float[Array, "H B T K"] | None = None,
  initial_state: Float[Array, "B N H K V"] | None = None,
  output_final_state: bool = False,
  chunk_size: int = 64,
  BV: int = 64,
  save_new_value: bool = True,
  use_exp2: bool = False,
  mini_batch: int | None = None,
) -> tuple[
    Float[Array, "H B NT K V"],
    Float[Array, "H B T V"] | None,
    Float[Array, "B N H K V"] | None,
]:
  """Launches inter-chunk state propagation for aligned varlen inputs."""
  H, B, T, K = k.shape
  V = v.shape[-1]
  BT = chunk_size

  N = seqlens.shape[-1] - 1
  assert K <= 256, "current kernel does not support head dimension larger than 256."

  # --- Varlen launcher ---
  k = k.astype(jnp.float32)
  w = w.astype(jnp.float32)
  u_f32 = v.astype(jnp.float32)

  K_PADSIZE = int(align_up(K, 128))
  V_ALIGNED = int(align_up(V, 128))

  NT = chunk_indices.shape[-2]
  chunk_to_seq = chunk_indices[:, :, 0].astype(jnp.int32)  # [B, NT]

  T_alloc = T

  k_pad = (
    jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
    if K_PADSIZE > K
    else k
  )
  w_pad = (
    jnp.pad(w, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
    if K_PADSIZE > K
    else w
  )
  k_t = k_pad
  w_t = w_pad

  v_pad = (
    jnp.pad(u_f32, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
    if V_ALIGNED > V
    else u_f32
  )
  v_t = v_pad

  if g is not None:
    g_fp32 = g.astype(jnp.float32).reshape(H, B, T, 1)
    g_fp32 = pad_to_multiple(g_fp32, 128, -1, 0)
    g_t = g_fp32
  else:
    g_t = None

  if gk is not None:
    gk_fp32 = gk.astype(jnp.float32)
    if K_PADSIZE > K:
      gk_fp32 = jnp.pad(gk_fp32, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
    gk_t = gk_fp32
  else:
    gk_t = None

  if initial_state is not None:
    h0 = initial_state
    if V_ALIGNED > V:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
    if K_PADSIZE > K:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
  else:
    h0 = None

  # --- Mini-batch auto-compute ---
  if mini_batch is None:
    per_head = K_PADSIZE * V_ALIGNED * 4  # scratch state bytes per head
    vmem_budget = 8 * 1024 * 1024  # 8 MB
    MB = max(1, vmem_budget // per_head)
    MB = min(MB, H, 16)
    while H % MB != 0 and MB > 1:
      MB -= 1
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  g_pad_size = g_t.shape[-1] if g_t is not None else 128
  h_spec = jax.ShapeDtypeStruct([H, B, NT, K_PADSIZE, V_ALIGNED], k.dtype)
  v_new_spec = (
    jax.ShapeDtypeStruct([H, B, T_alloc, V_ALIGNED], jnp.float32)
    if save_new_value
    else None
  )
  ht_spec = (
    jax.ShapeDtypeStruct([B, N, H, K_PADSIZE, V_ALIGNED], jnp.float32)
    if output_final_state
    else None
  )

  def _t_index_map(h, b, c, seqlens_ref, chunk_to_seq_ref):
    return (h, b, c, 0)

  def _h_index_map(h, b, c, seqlens_ref, chunk_to_seq_ref):
    return (h, b, c, 0, 0)

  k_blockspec = pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_t_index_map)
  v_blockspec = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
  w_blockspec = pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_t_index_map)
  g_blockspec = (
    pl.BlockSpec([MB, 1, BT, g_pad_size], index_map=_t_index_map)
    if g is not None
    else None
  )
  gk_blockspec = (
    pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_t_index_map)
    if gk is not None
    else None
  )
  h0_blockspec = (
    pl.BlockSpec(
      [1, 1, MB, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (b, chunk_to_seq_ref[b, c], h, 0, 0)
    )
    if initial_state is not None
    else None
  )

  h_blockspec_out = pl.BlockSpec(
    [MB, 1, 1, K_PADSIZE, V_ALIGNED], index_map=_h_index_map
  )
  v_new_blockspec_out = (
    pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
    if save_new_value
    else None
  )
  ht_blockspec_out = (
    pl.BlockSpec(
      [1, 1, MB, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (b, chunk_to_seq_ref[b, c], h, 0, 0)
    )
    if output_final_state
    else None
  )

  scratch = pltpu.VMEM((MB, K_PADSIZE, V_ALIGNED), jnp.float32)
  grid = (H // MB, B, NT)
  interpret = get_interpret()

  h_out, v_new_out, ht_out = pl.pallas_call(
    functools.partial(
      _chunk_gated_delta_rule_fwd_varlen_kernel,
      NT=NT,
      USE_G=(g is not None),
      USE_GK=(gk is not None),
      USE_INITIAL_STATE=(initial_state is not None),
      STORE_FINAL_STATE=output_final_state,
      SAVE_NEW_VALUE=save_new_value,
      USE_EXP2=use_exp2,
      MINI_BATCH=MB,
    ),
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=2,
      grid=grid,
      in_specs=[
        k_blockspec,
        v_blockspec,
        w_blockspec,
        g_blockspec,
        gk_blockspec,
        h0_blockspec,
      ],
      out_specs=[h_blockspec_out, v_new_blockspec_out, ht_blockspec_out],
      scratch_shapes=[scratch],
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "arbitrary")
    ),
    out_shape=[h_spec, v_new_spec, ht_spec],
    interpret=interpret,
  )(seqlens, chunk_to_seq, k_t, v_t, w_t, g_t, gk_t, h0)

  h_out = h_out[:, :, :, :K, :V]
  v_new_out = (
    v_new_out[:, :, :T, :V] if save_new_value else None
  )

  if output_final_state and ht_out is not None:
    ht_out = ht_out[:, :, :, :K, :V]

    # Handle empty sequences: sequences with no chunks never execute kernel code,
    # so their final_state is uninitialized. Fill them with initial_state or zeros.
    seq_lens = jnp.diff(seqlens, axis=-1)  # [B, N]
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

  return h_out, v_new_out, ht_out


# ── Non-varlen Pallas kernel ────────────────────────────────────────────────


def _chunk_gated_delta_rule_fwd_kernel(
  k_ref,  # [1, 1, BT, K_PADSIZE]
  v_ref,  # [1, 1, BT, V_ALIGNED]
  w_ref,  # [1, 1, BT, K_PADSIZE]
  g_ref,  # [1, 1, BT, G_PAD]
  gk_ref,  # [1, 1, BT, K_PADSIZE]
  h0_ref,  # [1, 1, K_PADSIZE, V_ALIGNED]
  # outputs
  h_ref,  # [1, NT, 1, K_PADSIZE, V_ALIGNED]
  v_new_ref,  # [1, 1, BT, V_ALIGNED]
  ht_ref,  # [1, 1, K_PADSIZE, V_ALIGNED]
  scratch_ref,  # [K_PADSIZE, V_ALIGNED]
  *,
  NT,
  USE_EXP2,
):
  """Pallas kernel for one chunk of the gated delta-rule inter-chunk forward pass.

  This kernel is invoked on a grid of (B, H, NT). For each chunk ``idx_nt``
  it performs the following steps:

  1. **Initialise state** (chunk 0 only): set scratch (the running hidden
     state ``h``) to zeros or to the provided initial state ``h0``.
  2. **Snapshot**: write the *pre-update* state into ``h_ref`` so that the
     intra-chunk backward pass can use it.
  3. **Delta correction**: ``v_new = u - w @ h`` — correct the raw value
     by subtracting the projection of the current state through ``w``.
  4. **Gated state update**: apply scalar gate ``g`` and/or per-dim gate
     ``gk`` to decay the state, then accumulate the outer product
     ``k^T @ v_new`` into the running hidden state.
  5. **Final state** (last chunk only): write the state out to ``ht_ref``
     so it can be returned as the final hidden state.
  """

  idx_nt = pl.program_id(2)

  BT = k_ref.shape[2]
  K, V = k_ref.shape[-1], v_ref.shape[-1]
  b_k = k_ref[0, 0]

  @pl.when(idx_nt == 0)
  def _():
    scratch_ref[...] = jnp.zeros([K, V], dtype=jnp.float32)
    if h0_ref is not None:
      scratch_ref[...] = h0_ref[0, 0].astype(jnp.float32)

  h_ref[0, 0, idx_nt] = scratch_ref[...].astype(h_ref.dtype)

  b_w = w_ref[0, 0]
  b_v = jnp.dot(
    b_w.astype(jnp.float32),
    scratch_ref[...],
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )
  b_u = v_ref[0, 0]
  b_v = b_u.astype(b_v.dtype) - b_v
  if v_new_ref is not None:
    v_new_ref[0, 0] = b_v.astype(v_new_ref.dtype)

  if g_ref is not None:
    b_g = g_ref[0, 0, :, 0]
    b_g_last = g_ref[0, 0, BT - 1, 0].astype(jnp.float32)
    if USE_EXP2:
      b_v = b_v * exp2(b_g_last - b_g)[:, None]
      b_g_last = exp2(b_g_last)
    else:
      b_v = b_v * exp(b_g_last - b_g)[:, None]
      b_g_last = exp(b_g_last)
    scratch_ref[...] *= b_g_last
  if gk_ref is not None:
    b_gk_last = gk_ref[0, 0, BT - 1].astype(jnp.float32)
    if USE_EXP2:
      scratch_ref[...] *= exp2(b_gk_last)[:, None]
    else:
      scratch_ref[...] *= exp(b_gk_last)[:, None]

  scratch_ref[...] += jnp.dot(
    b_k.astype(jnp.float32).T,
    b_v.astype(jnp.float32),
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )

  @pl.when(idx_nt == NT - 1)
  def _():
    if ht_ref is not None:
      ht_ref[0, 0] = scratch_ref[...].astype(ht_ref.dtype)


@functools.partial(
  jax.jit,
  static_argnames=[
    "output_final_state",
    "chunk_size",
    "save_new_value",
    "use_exp2",
  ],
)
def _chunk_gated_delta_rule_fwd(
  k: jax.Array,
  w: jax.Array,
  u: jax.Array,
  g: jax.Array | None = None,
  gk: jax.Array | None = None,
  initial_state: jax.Array | None = None,
  output_final_state: bool = False,
  chunk_size: int = 64,
  save_new_value: bool = True,
  use_exp2: bool = False,
):
  """Non-varlen launcher for the chunked gated delta rule forward pass.

  Requires T % chunk_size == 0 (enforced by ``chunk_gated_delta_rule_fwd_h``).
  Uses the dedicated ``_chunk_gated_delta_rule_fwd_kernel`` -- NOT the varlen
  path.  Follows _chunk_gla_fwd_o_gk's block-spec style: inputs are transposed
  to (H, B, T, dim) so each grid point loads one head's full T-length sequence.

  Grid:  (B, H, NT)

  Args:
      k: [H, B, T, K] -- Keys.
      w: [H, B, T, K] -- Correction weights.
      u: [H, B, T, V] -- Delta-corrected values from intra-chunk.
      g: [H, B, T] -- Scalar per-head gate (optional).
      gk: [H, B, T, K] -- Per-element gate (optional).
      initial_state: [B, H, K, V] -- Initial hidden state (optional).
      output_final_state: Whether to return final hidden state.
      chunk_size: Chunk size.
      save_new_value: Whether to compute and return v_new.
      use_exp2: Use exp2 for gate computation.

  Returns:
      h: [B, NT, H, K, V] -- Hidden states before each chunk.
      v_new: [B, T, H, V] or None -- Delta-corrected values.
      final_state: [B, H, K, V] or None.
  """
  H, B, T, K = k.shape
  V = u.shape[-1]
  BT = chunk_size
  NT = T // BT  # exact -- T % BT == 0 enforced by caller

  BV = 128  # must be >=128 so pl.ds(idx_v*BV, BV) on last dim is provably 128-element-aligned on TPU
  K_PADSIZE = int(align_up(K, 128))  # pad K to 128-element blocks
  V_ALIGNED = int(align_up(V, BV))  # pad V to BV-element blocks (multiple of 128)
  NV = V_ALIGNED // BV

  # -- Pad and transpose inputs to (H, B, T, dim) layout ---
  # k, w: [B, T, H, K] -> pad K -> transpose -> (B, H, T, K_PADSIZE)
  k_pad = (
    jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K))) if K_PADSIZE > K else k
  )
  w_pad = (
    jnp.pad(w, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K))) if K_PADSIZE > K else w
  )
  k_t = k_pad  # (H, B, T, K_PADSIZE)
  w_t = w_pad  # (H, B, T, K_PADSIZE)

  # u (values): [B, T, H, V] -> pad V -> transpose -> (H, B, T, V_ALIGNED)
  u_pad = (
    jnp.pad(u, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V))) if V_ALIGNED > V else u
  )
  v_t = u_pad  # (H, B, T, V_ALIGNED)

  # g (scalar gate): [H, B, T] -> float32 -> [H, B, T, 1] -> pad last -> [H, B, T, G_PAD]
  if g is not None:
    g_fp32 = g.astype(jnp.float32).reshape(H, B, T, 1)
    g_fp32 = pad_to_multiple(g_fp32, 128, -1, 0)  # (H, B, T, 128)
    g_t = g_fp32  # (H, B, T, 128)
  else:
    g_t = None

  # gk (per-dim gate): [H, B, T, K] -> float32 -> pad K -> (H, B, T, K_PADSIZE)
  if gk is not None:
    gk_fp32 = gk.astype(jnp.float32)
    if K_PADSIZE > K:
      gk_fp32 = jnp.pad(gk_fp32, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
    gk_t = gk_fp32  # (H, B, T, K_PADSIZE)
  else:
    gk_t = None

  # h0 (initial state): [N=B, H, K, V] -> pad V, K -> transpose -> (B, H, K_PADSIZE, V_ALIGNED)
  if initial_state is not None:
    h0 = initial_state
    if V_ALIGNED > V:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
    if K_PADSIZE > K:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
  else:
    h0 = None

  # -- Output shapes ---
  # h stored as [B, NT, H, K_PADSIZE, V_ALIGNED] with V before K (varlen convention).
  h_spec = jax.ShapeDtypeStruct([H, B, NT, K_PADSIZE, V_ALIGNED], k.dtype)
  v_new_spec = (
    jax.ShapeDtypeStruct([H, B, T, V_ALIGNED], jnp.float32) if save_new_value else None
  )
  ht_spec = (
    jax.ShapeDtypeStruct([B, H, K_PADSIZE, V_ALIGNED], jnp.float32)
    if output_final_state
    else None
  )

  # -- Block specs ---
  g_pad_size = (
    g_t.shape[-1] if g_t is not None else 128
  )  # always 128 after pad_to_multiple

  k_blockspec = pl.BlockSpec(
    [1, 1, BT, K_PADSIZE], index_map=lambda b, h, nt: (h, b, nt, 0)
  )
  v_blockspec = pl.BlockSpec(
    [1, 1, BT, V_ALIGNED], index_map=lambda b, h, nt: (h, b, nt, 0)
  )
  w_blockspec = pl.BlockSpec(
    [1, 1, BT, K_PADSIZE], index_map=lambda b, h, nt: (h, b, nt, 0)
  )
  g_blockspec = (
    pl.BlockSpec([1, 1, BT, g_pad_size], index_map=lambda b, h, nt: (h, b, nt, 0))
    if g is not None
    else None
  )
  gk_blockspec = (
    pl.BlockSpec([1, 1, BT, K_PADSIZE], index_map=lambda b, h, nt: (h, b, nt, 0))
    if gk is not None
    else None
  )
  h0_blockspec = (
    pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=lambda b, h, nt: (b, h, 0, 0))
    if initial_state is not None
    else None
  )

  h_blockspec_out = pl.BlockSpec(
    [1, 1, NT, K_PADSIZE, V_ALIGNED], lambda b, h, nt: (h, b, 0, 0, 0)
  )
  v_new_blockspec_out = (
    pl.BlockSpec([1, 1, BT, V_ALIGNED], lambda b, h, nt: (h, b, nt, 0))
    if save_new_value
    else None
  )
  ht_blockspec_out = (
    pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], lambda b, h, nt: (b, h, 0, 0))
    if output_final_state
    else None
  )

  scratch = pltpu.VMEM((K_PADSIZE, V_ALIGNED), jnp.float32)
  scratch_shapes = [scratch]

  grid = (B, H, NT)
  interpret = get_interpret()
  h_out, v_new_out, ht_out = pl.pallas_call(
    functools.partial(
      _chunk_gated_delta_rule_fwd_kernel,
      NT=NT,
      USE_EXP2=use_exp2,
    ),
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=grid,
      in_specs=[
        k_blockspec,
        v_blockspec,
        w_blockspec,
        g_blockspec,
        gk_blockspec,
        h0_blockspec,
      ],
      out_specs=[h_blockspec_out, v_new_blockspec_out, ht_blockspec_out],
      scratch_shapes=scratch_shapes,
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=(
        "parallel",
        "parallel",
        "arbitrary",
      ),
      # vmem_limit_bytes=32 * 1024 * 1024,
      disable_bounds_checks=True,
    ),
    out_shape=[h_spec, v_new_spec, ht_spec],
    interpret=interpret,
  )(k_t, v_t, w_t, g_t, gk_t, h0)

  # -- Post-process outputs ---
  # h: [H, B, NT, K_PADSIZE, V_ALIGNED] -> trim K and V padding -> [H, B, NT, K, V]
  h_out = h_out[:, :, :, :K, :V]

  # v_new: [B, H, T, V_ALIGNED] -> [B, T, H,  V]
  if save_new_value:
    if V_ALIGNED > V:
      v_new_out = v_new_out[:, :, :, :V]
  else:
    v_new_out = None

  # ht: [B, H, K_PADSIZE, V_ALIGNED] -> trim K and V padding -> [B, H, K, V]
  if output_final_state:
    ht_out = ht_out[:, :, :K, :V]
  else:
    ht_out = None

  return h_out, v_new_out, ht_out


# ── Public dispatch function ────────────────────────────────────────────────


@jaxtyping.jaxtyped
def chunk_gated_delta_rule_fwd_h(
  k: Float[Array, "H B T K"],
  w: Float[Array, "H B T K"],
  u: Float[Array, "H B T V"],
  g: Float[Array, "H B T"] | None = None,
  gk: Float[Array, "H B T K"] | None = None,
  initial_state: (
      Float[Array, "N_STATE H K V"]
      | Float[Array, "B N_STATE H K V"]
      | None
  ) = None,
  output_final_state: bool = False,
  chunk_size: int = 64,
  save_new_value: bool = True,
  use_exp2: bool = True,
  cu_seqlens: Int[Array, "N_CU"] | Int[Array, "B N_CU"] | None = None,
  chunk_indices: Int[Array, "NT 2"] | Int[Array, "B NT 2"] | None = None,
) -> tuple[
    Float[Array, "H B NT K V"],
    Float[Array, "H B T V"] | None,
    Float[Array, "B H K V"] | Float[Array, "B N H K V"] | None,
]:
  """Dispatches inter-chunk state propagation to fixed or varlen launchers."""
  H, B, T, K = k.shape
  V = u.shape[-1]
  BT = chunk_size
  NT = cdiv(T, BT)
  N = B if cu_seqlens is None else cu_seqlens.shape[-1] - 1

  if cu_seqlens is not None:
    if initial_state is not None and initial_state.ndim == 4:
      initial_state = initial_state[None, :, :, :, :]
  else:
    if initial_state is not None and initial_state.ndim == 5:
      initial_state = initial_state[:, 0]
  assert K <= 256, "current kernel does not support head dimension larger than 256."

  if cu_seqlens is None:
    assert T % chunk_size == 0, (
      "For non-varlen input, T must be divisible by chunk_size"
    )
    return _chunk_gated_delta_rule_fwd(
      k,
      w,
      u,
      g=g,
      gk=gk,
      initial_state=initial_state,
      output_final_state=output_final_state,
      chunk_size=chunk_size,
      save_new_value=save_new_value,
      use_exp2=use_exp2,
    )
  else:
    # _chunk_gated_delta_rule_fwd_varlen expects B-batched inputs:
    #   seqlens: [B, N+1], chunk_indices: [B, NT, 2],
    #   initial_state: [B, N, H, K, V].
    # When B=1 the caller may pass 1D seqlens and 2D chunk_indices;
    # unsqueeze them here. initial_state must always be 5D.
    _varlen_cu = cu_seqlens
    _varlen_ci = chunk_indices
    _varlen_h0 = initial_state
    if _varlen_cu is not None and _varlen_cu.ndim == 1:
      _varlen_cu = _varlen_cu[None, :]
    if _varlen_ci is not None and _varlen_ci.ndim == 2:
      _varlen_ci = _varlen_ci[None, :, :]
    h, v_new, final_state = _chunk_gated_delta_rule_fwd_varlen(
      k,
      w,
      u,
      seqlens=_varlen_cu,
      chunk_indices=_varlen_ci,
      g=g,
      gk=gk,
      initial_state=_varlen_h0,
      output_final_state=output_final_state,
      chunk_size=chunk_size,
      save_new_value=save_new_value,
      use_exp2=use_exp2,
    )
    return h, v_new, final_state


@jaxtyping.jaxtyped
def kda_gate_chunk_cumsum(
  g: Float[Array, "H B T K"],
  A_log: Float[Array, "H"],
  chunk_size: int,
  scale: float | None = None,
  dt_bias: Float[Array, "H*K"] | None = None,
  output_dtype: jax.typing.DTypeLike = jnp.float32,
  lower_bound: float | None = None,
) -> Float[Array, "H B T K"]:
  """Applies the KDA gate activation and its chunk-local cumulative sum."""
  H, B, T, K = g.shape
  g_f32 = g.astype(jnp.float32)

  if dt_bias is not None:
    g_f32 = g_f32 + dt_bias.astype(jnp.float32).reshape(H, 1, 1, K)

  A = A_log.astype(jnp.float32)

  if lower_bound is None:
    g_act = -jnp.exp(A).reshape(H, 1, 1, 1) * jax.nn.softplus(g_f32)
  else:
    g_act = lower_bound * jax.nn.sigmoid(jnp.exp(A).reshape(H, 1, 1, 1) * g_f32)

  return chunk_local_cumsum_vector(
    g_act,
    chunk_size=chunk_size,
    scale=scale,
    output_dtype=output_dtype or jnp.float32,
  )
