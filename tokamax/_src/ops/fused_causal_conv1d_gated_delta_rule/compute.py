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
"""Convolution, recurrent updates and output packing for fused GDN."""

from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import compute_gdn
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config

OUTPUT_ALIGNMENT = 16

_L2_NORM_EPSILON = 1e-6
# Batched preparation needs at least two complete 128-row recurrent chunks.
_MIN_BATCHED_PREPARE_ROWS = 256


def native_gate_values(
    gate_ref: Any, input_offset: jax.Array, rows: int
) -> jax.Array:
  """Select gate rows from a private eight-row-aligned DMA window.

  Convert BF16/FP32 [window_rows, value_heads] before rotating to avoid
  unaligned VMEM loads; discard the prefix belonging to another request.

  Args:
    gate_ref: Private aligned gate-input window, shaped [window_rows,
      value_heads].
    input_offset: Logical token offset within the private eight-row-aligned
      input window.
    rows: Number of token rows to return or allocate.

  Returns:
    FP32 gate values [rows, value_heads] at the logical token start.
  """

  def shifted() -> jax.Array:
    """Return FP32 gate rows with the alignment prefix rotated away."""
    values = gate_ref[...].astype(jnp.float32)
    return pltpu.roll(
        values, shift=(-input_offset) % gate_ref.shape[0], axis=0
    )[:rows, :]

  if rows == 1:
    return shifted()

  return jax.lax.cond(
      input_offset == 0,
      lambda: gate_ref[:rows, :].astype(jnp.float32),
      shifted,
  )


def _dot_bf16_batched(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
  """Multiply BF16 operands with FP32 accumulation.

  State projections use _state_dot_bf16x3; Gram and solve rounding can still
  propagate through recurrent updates.

  Args:
    lhs: Left product operand, shaped [batch, rows, inner].
    rhs: Right product operand, shaped [batch, inner, columns].

  Returns:
    FP32 products [batch, rows, columns] computed with BF16 operands.
  """
  return jax.lax.dot(
      lhs.astype(jnp.bfloat16),
      rhs.astype(jnp.bfloat16),
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      precision=jax.lax.Precision.DEFAULT,
      preferred_element_type=jnp.float32,
  )


def _dot_bf16_head_pairs(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
  """Pack short, independent head products into diagonal blocks.

  Larger products use a batched dot to avoid discarded cross-head work.

  Args:
    lhs: Left product operand, shaped [batch, rows, inner].
    rhs: Right product operand, shaped [batch, inner, columns].

  Returns:
    FP32 products [heads, rows, columns] in the original head order.
  """
  heads, rows, inner = lhs.shape
  columns = rhs.shape[-1]
  pairs = heads // 2
  # Two heads must fit the 128-row/column tile.
  if not pairs or rows > 64 or inner > 64 or columns != 128:
    return _dot_bf16_batched(lhs, rhs)
  paired_lhs = (
      lhs[: 2 * pairs].astype(jnp.bfloat16).reshape(pairs, 2, rows, inner)
  )
  zeros = jnp.zeros((pairs, rows, inner), dtype=jnp.bfloat16)
  paired_lhs = jnp.concatenate(
      [
          jnp.concatenate([paired_lhs[:, 0], zeros], axis=2),
          jnp.concatenate([zeros, paired_lhs[:, 1]], axis=2),
      ],
      axis=1,
  )
  paired_rhs = (
      rhs[: 2 * pairs].astype(jnp.bfloat16).reshape(pairs, 2 * inner, columns)
  )
  result = _dot_bf16_batched(paired_lhs, paired_rhs).reshape(
      2 * pairs, rows, columns
  )
  if heads % 2:
    result = jnp.concatenate(
        [result, _dot_bf16_batched(lhs[-1:], rhs[-1:])], axis=0
    )
  return result


def _dot_bf16_head_pairs_wide(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
  """Pack two head products into a 128x256 MXU tile; keep its diagonal blocks.

  lhs is [heads, rows, inner], rhs [heads, inner, columns]. Packing requires
  rows and inner <= 64 and columns == 128; other sizes use a batched dot.
  Return [heads, rows, columns] FP32; an unpaired head uses a single dot.

  Args:
    lhs: Left product operand, shaped [batch, rows, inner].
    rhs: Right product operand, shaped [batch, inner, columns].

  Returns:
    FP32 products [heads, rows, columns] with cross-head blocks discarded.
  """
  heads, rows, inner = lhs.shape
  columns = rhs.shape[-1]
  pairs = heads // 2
  if not pairs or rows > 64 or inner > 64 or columns != 128:
    return _dot_bf16_batched(lhs, rhs)

  paired_lhs = lhs[: 2 * pairs].reshape(pairs, 2, rows, inner)
  paired_lhs = jnp.concatenate([paired_lhs[:, 0], paired_lhs[:, 1]], axis=1)
  paired_rhs = rhs[: 2 * pairs].reshape(pairs, 2, inner, columns)
  paired_rhs = jnp.concatenate([paired_rhs[:, 0], paired_rhs[:, 1]], axis=2)
  packed = _dot_bf16_batched(paired_lhs, paired_rhs)
  first = packed[:, :rows, :columns]
  second = packed[:, rows:, columns:]
  result = jnp.stack([first, second], axis=1).reshape(2 * pairs, rows, columns)
  if heads % 2:
    result = jnp.concatenate(
        [result, _dot_bf16_batched(lhs[-1:], rhs[-1:])], axis=0
    )
  return result


def pack_value_heads(x: jax.Array, group_size: int = 2) -> jax.Array:
  """Map [heads, rows, width] to [heads // group_size, rows, group_size *
  width].

  Adjacent value heads share a Q/K head; group_size = n_v // n_kq.

  Args:
    x: Values [heads, rows, width]; heads must be divisible by group_size.
    group_size: Number of adjacent value heads sharing one Q/K head.

  Returns:
    Packed values [heads // group_size, rows, group_size * width], with dtype
    preserved.
  """
  groups = x.reshape(
      x.shape[0] // group_size, group_size, x.shape[1], x.shape[2]
  )
  if group_size == 2:
    return jnp.concatenate([groups[:, 0], groups[:, 1]], axis=-1)
  return jnp.concatenate(
      [groups[:, index] for index in range(group_size)], axis=-1
  )


def unpack_value_heads(x: jax.Array, group_size: int = 2) -> jax.Array:
  """Invert pack_value_heads, preserving dtype and the original head order.

  Args:
    x: Packed values [groups, rows, packed_width]; width must divide by
      group_size.
    group_size: Number of adjacent value heads sharing one Q/K head.

  Returns:
    Unpacked values [groups * group_size, rows, packed_width // group_size],
    with dtype preserved.
  """
  if group_size == 2:
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.stack([first, second], axis=1).reshape(
        2 * x.shape[0], x.shape[1], x.shape[2] // 2
    )
  pieces = jnp.split(x, group_size, axis=-1)
  return jnp.stack(pieces, axis=1).reshape(
      group_size * x.shape[0], x.shape[1], x.shape[2] // group_size
  )


def _state_dot_bf16x3(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
  """Project state with BF16 hi*hi + hi*lo + lo*hi and FP32 accumulation.

  Omitting lo*lo loses FP32 agreement, especially under cancellation. Inputs
  must be finite and within BF16 range. Multiply [batch, rows, inner] by
  [batch, inner, columns] to return [batch, rows, columns] FP32.

  Args:
    lhs: Left product operand, shaped [batch, rows, inner].
    rhs: Right product operand, shaped [batch, inner, columns].

  Returns:
    FP32 approximate products [batch, rows, columns] from three BF16 terms.
  """
  lhs = lhs.astype(jnp.float32)
  rhs = rhs.astype(jnp.float32)
  lhs_hi = lhs.astype(jnp.bfloat16)
  rhs_hi = rhs.astype(jnp.bfloat16)
  lhs_lo = (lhs - lhs_hi.astype(jnp.float32)).astype(jnp.bfloat16)
  rhs_lo = (rhs - rhs_hi.astype(jnp.float32)).astype(jnp.bfloat16)

  def product(left: jax.Array, right: jax.Array) -> jax.Array:
    """Multiply one pair of batched BF16 operands with FP32 accumulation.

    Args:
      left: Left BF16 projection operand, shaped [batch, rows, inner].
      right: Right BF16 projection operand, shaped [batch, inner, columns].

    Returns:
      FP32 accumulation of the BF16 operand products.
    """
    return jax.lax.dot(
        left,
        right,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )

  correction = product(lhs_hi, rhs_lo) + product(lhs_lo, rhs_hi)
  return product(lhs_hi, rhs_hi) + correction


def _solve_column_sweep(
    n: jax.Array,
    rhs: jax.Array,
    block_size: int = 8,
) -> jax.Array:
  """Solve (I + n) x = rhs by blockwise forward substitution.

  n is strictly lower triangular [heads, chunk, chunk]; rhs is [heads, chunk,
  columns]. chunk must be divisible by min(block_size, chunk). Return the
  solution in n.dtype.

  Args:
    n: Strictly lower-triangular coefficient array [heads, chunk, chunk].
    rhs: Right-hand sides [heads, chunk, columns] for (I + n)x = rhs.
    block_size: Forward-substitution block width; must divide the chunk after
      clamping.

  Returns:
    Solution [heads, chunk, columns] in n.dtype.
  """
  out_dtype = n.dtype
  chunk = n.shape[-1]
  block_size = min(block_size, chunk)
  blocks = []

  for block in range(chunk // block_size):
    start = block * block_size
    end = start + block_size
    target = rhs[:, start:end, :]

    if block:
      previous = jnp.concatenate(blocks, axis=1)
      interaction = n[:, start:end, :start]
      target = target - _dot_bf16_head_pairs(interaction, previous)

    if block_size <= 8:
      for s in range(block_size - 1):
        target = target - (
            n[:, start:end, start + s : start + s + 1] * target[:, s : s + 1, :]
        )
    else:
      # Strict triangularity lets later updates skip the solved half.
      split = block_size // 2
      for s in range(split - 1):
        target = target - (
            n[:, start:end, start + s : start + s + 1] * target[:, s : s + 1, :]
        )
      solved = target[:, :split, :]
      tail = target[:, split:, :] - (
          n[:, start + split : end, start + split - 1 : start + split]
          * solved[:, -1:, :]
      )
      for s in range(block_size - split - 1):
        tail = tail - (
            n[:, start + split : end, start + split + s : start + split + s + 1]
            * tail[:, s : s + 1, :]
        )
      target = jnp.concatenate([solved, tail], axis=1)

    blocks.append(target.astype(out_dtype))

  return jnp.concatenate(blocks, axis=1)


def _prefix_decay(gating_log: jax.Array) -> jax.Array:
  """Prefix-sum FP32 log decays [batch, rows, aligned_heads].

  Decays must be finite and nonpositive, with zero padded rows, to avoid
  cancellation during reassociation. Row shifts preserve padded heads.

  Args:
    gating_log: FP32 log decays in the documented layout; padded rows are zero.

  Returns:
    Inclusive log-decay prefix sums with the input shape and FP32 dtype.
  """
  result = gating_log
  distance = 1
  while distance < gating_log.shape[1]:
    shifted = jnp.concatenate(
        [jnp.zeros_like(result[:, :distance]), result[:, :-distance]], axis=1
    )
    result = result + shifted
    distance *= 2
  return result


def _chunked_gdn_batched_prepare(
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    gating_log: jax.Array,
    beta: jax.Array,
    state_prev: jax.Array,
    cfg: config.GDNConfig,
    masks: tuple[jax.Array, jax.Array],
    grouped_state_scratch_ref: Any,
) -> tuple[jax.Array, jax.Array]:
  """Batch 128-row solves independently of state, then advance recurrence.

  L = I + strictly_lower(beta * decay * KK);
  U = solve(L, beta * V) - solve(L, beta * exp(g) * K) @ S.
  Never divide by exp(g), which may underflow. Requires >= 2 complete chunks,
  128-wide Q/K/V heads, and zero gates on invalid rows. Both RHSs share a
  256-column solve; state and scratch match _chunked_gdn_per_seq.

  Args:
    q_large: Normalized, scaled queries [n_kq, chunk_size, 128].
    k_large: Normalized keys [n_kq, chunk_size, 128].
    v_large: Values [n_v, chunk_size, 128].
    gating_log: FP32 log decays [1, chunk_size, aligned_num_v_heads]; padded rows
      are zero.
    beta: Update gates [1, chunk_size, aligned_num_v_heads]; padded rows are zero.
    state_prev: Previous FP32 recurrent state [n_v, 128, 128].
    cfg: Static GDN configuration defining head layout, tile sizes, and dtypes.
    masks: Inclusive and strictly lower-triangular causal masks.
    grouped_state_scratch_ref: Mutable FP32 scratch [n_kq, 128, (n_v // n_kq) *
      128].

  Returns:
    Activation-dtype output [n_v, chunk_size, 128] and FP32 final state [n_v,
    128, 128].
  """
  rows = 128
  chunks = cfg.chunk_size // rows
  nv = cfg.num_v_heads
  nk = cfg.num_kq_heads

  def split_chunks(x: jax.Array) -> jax.Array:
    """Reshape a long tile into the chunk/head layout used by batched
    preparation.

    Args:
      x: Head-major values [heads, chunks * 128, 128].

    Returns:
      Chunk-major values [chunks * heads, 128, 128].
    """
    return (
        x.reshape(x.shape[0], chunks, rows, 128)
        .transpose(1, 0, 2, 3)
        .reshape(chunks * x.shape[0], rows, 128)
    )

  q = split_chunks(q_large)
  k = split_chunks(k_large)
  v = split_chunks(v_large)
  g = _prefix_decay(gating_log.reshape(chunks, rows, -1))
  g_t = g.transpose(0, 2, 1)[:, :nv, :].reshape(chunks * nv, 1, rows)
  g = compute_gdn.fused_transpose_broadcast(g_t, src_dim=2, dst_dim=1)
  beta_t = beta.reshape(chunks, rows, -1).transpose(0, 2, 1)
  beta_t = beta_t[:, :nv, :].reshape(chunks * nv, 1, rows)
  beta_rows = compute_gdn.fused_transpose_broadcast(
      beta_t, src_dim=2, dst_dim=1
  )
  forward = jnp.exp(g)
  backward = jnp.exp(g_t[..., -1:] - g_t)
  backward = compute_gdn.fused_transpose_broadcast(
      backward, src_dim=2, dst_dim=1
  )
  qk_kk = _dot_bf16_batched(
      jnp.concatenate([q, k], axis=1), jnp.swapaxes(k, 1, 2)
  )
  qk, kk = jnp.split(qk_kk, 2, axis=1)
  kk = jnp.repeat(kk, cfg.v_per_kq_head, axis=0)
  decay = jnp.exp(g - g_t)
  lower, strict_lower = (mask[:, :rows, :rows] for mask in masks)
  n = jnp.where(strict_lower, decay, 0) * (kk * beta_rows)
  shared_k = jnp.repeat(k, cfg.v_per_kq_head, axis=0)
  rhs = jnp.concatenate(
      [v * beta_rows, shared_k * (beta_rows * forward)], axis=2
  )
  solved = _solve_column_sweep(n, rhs, block_size=8)
  value, key = jnp.split(solved, 2, axis=2)
  out_qk = jnp.repeat(qk, cfg.v_per_kq_head, axis=0)
  out_qk = out_qk * jnp.where(lower, decay, 0)

  # VMEM refs keep the loop from carrying every chunk's state in an SSA tuple.
  prepared = (
      key.reshape(chunks, nv, *key.shape[1:]),
      value.reshape(chunks, nv, *value.shape[1:]),
      k.reshape(chunks, nk, *k.shape[1:]),
      backward.reshape(chunks, nv, *backward.shape[1:]),
      forward[:, -1:].reshape(chunks, nv, *forward[:, -1:].shape[1:]),
  )

  def advance(
      prepared_refs: Any, state_ref: Any, residual_ref: Any, incoming_ref: Any
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Initialize scoped buffers and advance the prepared recurrent chunks.

    Args:
      prepared_refs: Scratch refs for solved keys/values, keys, and decay
        factors.
      state_ref: Mutable FP32 recurrent state carried through prepared
        subchunks.
      residual_ref: Scratch receiving the solved value residual of each
        recurrent subchunk.
      incoming_ref: Scratch receiving each subchunk's incoming grouped recurrent
        state.

    Returns:
      Final FP32 state, per-chunk residuals, and grouped incoming states.
    """
    for ref, values in zip(prepared_refs, prepared):
      ref[...] = values
    key_ref, value_ref, k_ref, backward_ref, last_ref = prepared_refs
    state_ref[...] = state_prev.astype(jnp.float32)

    def step(chunk: jax.Array, unused: None) -> None:
      """Record incoming state, compute the chunk residual, and advance state.

      Args:
        chunk: Zero-based recurrent subchunk index.
        unused: Unused loop carry, always None.
      """
      state = state_ref[...]
      incoming_ref[chunk] = pack_value_heads(state, cfg.v_per_kq_head)
      key_state = _state_dot_bf16x3(key_ref[chunk], state)
      residual = value_ref[chunk] - key_state
      residual_ref[chunk] = residual
      grouped_state_scratch_ref[...] = pack_value_heads(
          residual * backward_ref[chunk], cfg.v_per_kq_head
      )
      state_update = _state_dot_bf16x3(
          jnp.swapaxes(k_ref[chunk], 1, 2), grouped_state_scratch_ref[...]
      )
      state_ref[...] = state * last_ref[chunk] + unpack_value_heads(
          state_update, cfg.v_per_kq_head
      )
      return unused

    jax.lax.fori_loop(0, chunks, step, None)
    return state_ref[...], residual_ref[...], incoming_ref[...]

  state, residuals, incoming_states = pl.run_scoped(
      advance,
      tuple(pltpu.VMEM(x.shape, x.dtype) for x in prepared),
      pltpu.VMEM(state_prev.shape, jnp.float32),
      pltpu.VMEM((chunks, nv, rows, 128), jnp.float32),
      pltpu.VMEM((chunks, nk, 128, cfg.v_per_kq_head * 128), jnp.float32),
  )

  # Shared Q/K heads let us project grouped states across all chunks.
  query_state = _state_dot_bf16x3(
      q, incoming_states.reshape(chunks * nk, 128, cfg.v_per_kq_head * 128)
  )
  query_state = unpack_value_heads(query_state, cfg.v_per_kq_head) * forward
  output = query_state + _dot_bf16_batched(
      out_qk, residuals.reshape(chunks * nv, rows, 128)
  )
  output = output.reshape(chunks, nv, rows, 128).transpose(1, 0, 2, 3)
  return (
      output.reshape(nv, cfg.chunk_size, 128).astype(cfg.dtypes.act_out),
      state,
  )


def _chunked_gdn_per_seq(
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    gating_log: jax.Array,
    beta: jax.Array,
    state_prev: jax.Array,
    cfg: config.GDNConfig,
    masks: tuple[jax.Array, jax.Array],
    grouped_state_scratch_ref: Any,
) -> tuple[jax.Array, jax.Array]:
  """Advance GDN in recurrent subchunks of at most 128 rows.

  Q/K must already be normalized, with Q scaled by the head dimension. Q/K/V
  are [heads, chunk_size, 128]; gates are [1, chunk_size,
  aligned_num_v_heads]. chunk_size must be a multiple of 64; invalid rows
  need zero beta/log decay. masks are inclusive/strict causal triangles [1,
  chunk_size, chunk_size]. Grouped state scratch is [n_kq, 128, v_per_kq *
  128] FP32. Return activation-dtype output [n_v, chunk_size, 128] and FP32
  state [n_v, 128, 128].

  Args:
    q_large: Normalized and scaled query activations in the layout described
      above.
    k_large: Normalized key activations in the head/sequence layout described
      above.
    v_large: Value activations in the head/sequence layout described above.
    gating_log: FP32 log decays in the documented layout; padded rows are zero.
    beta: Per-token update gates in the layout described above; padding must be
      zero.
    state_prev: Previous FP32 recurrent state in the layout described above.
    cfg: Static GDN configuration defining head layout, tile sizes, and dtypes.
    masks: Inclusive and strictly lower-triangular causal masks.
    grouped_state_scratch_ref: Mutable FP32 scratch [n_kq, 128, (n_v // n_kq) *
      128].

  Returns:
    Activation-dtype output [n_v, chunk_size, 128] and FP32 final state [n_v,
    128, 128].
  """
  if (
      cfg.num_kq_heads == 1
      and cfg.num_v_heads == 3
      and cfg.chunk_size >= _MIN_BATCHED_PREPARE_ROWS
  ):
    return _chunked_gdn_batched_prepare(
        q_large,
        k_large,
        v_large,
        gating_log,
        beta,
        state_prev,
        cfg,
        masks,
        grouped_state_scratch_ref,
    )

  sub_chunk_size = sub_chunk_rows(
      cfg.chunk_size, cfg.num_kq_heads, cfg.num_v_heads
  )
  lower, strict_lower = masks
  lower = lower[:, :sub_chunk_size, :sub_chunk_size]
  strict_lower = strict_lower[:, :sub_chunk_size, :sub_chunk_size]
  state = state_prev.astype(jnp.float32)
  outputs = []

  # Small Q/K layouts amortize wider solve panels across value heads.
  solve_block_size = 16 if cfg.num_kq_heads <= 4 and cfg.num_v_heads >= 6 else 8

  for start in range(0, cfg.chunk_size, sub_chunk_size):
    end = start + sub_chunk_size
    q_sub = q_large[:, start:end, :]
    k_sub = k_large[:, start:end, :]
    v_sub = v_large[:, start:end, :]
    gating_sub = gating_log[:, start:end, :]
    beta_sub = beta[:, start:end, :]

    g = _prefix_decay(gating_sub)

    g_t = g[0].T[: cfg.num_v_heads, None, :]
    g = compute_gdn.fused_transpose_broadcast(g_t, src_dim=2, dst_dim=1)
    beta_t = beta_sub[0].T[: cfg.num_v_heads, None, :]
    beta_large = compute_gdn.fused_transpose_broadcast(
        beta_t, src_dim=2, dst_dim=1
    )

    g_forward = jnp.exp(g)
    g_last = g_forward[:, -1:]
    g_backward = jnp.exp(-(g_t - g_t[..., -1:]))

    k_t = jnp.swapaxes(k_sub, 1, 2)
    qk_kk = _dot_bf16_batched(jnp.concatenate([q_sub, k_sub], axis=1), k_t)
    qk, kk = jnp.split(qk_kk, 2, axis=1)

    beta_kk = jnp.repeat(kk, cfg.v_per_kq_head, axis=0) * beta_large
    decay = jnp.exp(g - g_t)
    n = jnp.where(strict_lower, decay, 0) * beta_kk

    grouped_state_scratch_ref[...] = pack_value_heads(state, cfg.v_per_kq_head)
    grouped_projections = _state_dot_bf16x3(
        jnp.concatenate([k_sub, q_sub], axis=1),
        grouped_state_scratch_ref[...],
    )
    projections = unpack_value_heads(grouped_projections, cfg.v_per_kq_head)
    k_S, q_S = jnp.split(projections, 2, axis=1)

    kg_S = (k_S * beta_large) * g_forward
    out_updated = q_S * g_forward
    rhs = v_sub.astype(jnp.float32) * beta_large - kg_S
    # Each value head has its own gates; within-panel updates stay FP32.
    u_ws = _solve_column_sweep(n, rhs, block_size=solve_block_size)

    gating_lower = jnp.where(lower, decay, 0)
    out_qk = jnp.repeat(qk, cfg.v_per_kq_head, axis=0) * gating_lower

    out_new = _dot_bf16_head_pairs_wide(out_qk, u_ws)
    outputs.append((out_updated + out_new).astype(cfg.dtypes.act_out))

    g_backward_rows = compute_gdn.fused_transpose_broadcast(
        g_backward, src_dim=2, dst_dim=1
    )
    grouped_state_scratch_ref[:, :sub_chunk_size, :] = pack_value_heads(
        u_ws * g_backward_rows, cfg.v_per_kq_head
    )
    grouped_state_update = _state_dot_bf16x3(
        k_t,
        grouped_state_scratch_ref[:, :sub_chunk_size, :],
    )
    state_new = unpack_value_heads(grouped_state_update, cfg.v_per_kq_head)
    state = state * g_last + state_new

  return jnp.concatenate(outputs, axis=1), state


def sub_chunk_rows(tile_rows: int, n_kq: int, n_v: int) -> int:
  """Choose recurrence rows for an activation tile.

  Use up to 128 through four Q/K and sixteen value heads; wider layouts use
  64 to limit working storage. tile_rows_supported checks divisibility.

  Args:
    tile_rows: Number of token rows in the candidate activation tile.
    n_kq: Number of query/key heads.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.

  Returns:
    Recurrent subchunk rows, bounded by tile_rows and the layout's 64/128-row
    limit.
  """
  return min(tile_rows, 128 if n_kq <= 4 and n_v <= 16 else 64)


def chunked_gdn(
    real_sizes: jax.Array,
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    b_large: jax.Array,
    a_large: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
    grouped_state_scratch_ref: Any,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
  """Mask prefill gates and advance one sequence's recurrent state.

  Q/K/V add a leading sequence axis to _chunked_gdn_per_seq's layouts;
  beta/decay inputs are [1, 1, chunk_size, aligned_num_v_heads]. Q/K must be
  normalized and appropriately scaled; padded activation rows initialized.
  cfg.seq_tile_size must be one and chunk_size a multiple of 64. Return a
  per-sequence output tuple and [1, n_v, 128, 128] FP32 state.

  Args:
    real_sizes: Per-sequence valid token counts; this fused path uses one
      sequence.
    q_large: Normalized and scaled query activations in the layout described
      above.
    k_large: Normalized key activations in the head/sequence layout described
      above.
    v_large: Value activations in the head/sequence layout described above.
    b_large: Raw beta-gate inputs [1, 1, chunk_size, aligned_num_v_heads].
    a_large: Raw decay-gate inputs [1, 1, chunk_size, aligned_num_v_heads].
    state_prev: Previous FP32 recurrent state in the layout described above.
    a_log: Log decay weights padded to aligned_num_v_heads.
    dt_bias: Timestep biases padded to aligned_num_v_heads.
    cfg: Static GDN configuration defining head layout, tile sizes, and dtypes.
    grouped_state_scratch_ref: Mutable FP32 scratch [n_kq, 128, (n_v // n_kq) *
      128].

  Returns:
    Per-sequence output tuple and stacked FP32 state [1, n_v, 128, 128].
  """
  mask_dtype = compute_gdn.get_mask_dtype(cfg.dtypes.compute)
  positions = jnp.arange(cfg.chunk_size, dtype=mask_dtype)
  rows = positions[None, :, None]
  cols = positions[None, None, :]
  masks = (rows >= cols, rows > cols)
  valid = positions.reshape(1, 1, cfg.chunk_size, 1) < (
      real_sizes.reshape(-1, 1, 1, 1).astype(mask_dtype)
  )

  # Zero padded gates so inactive rows cannot advance recurrent state.
  b_large = b_large.astype(cfg.dtypes.compute)
  a_large = a_large.astype(cfg.dtypes.compute)
  a_log = a_log.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
  dt_bias = dt_bias.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)

  beta = jax.nn.sigmoid(b_large)
  gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_large + dt_bias)
  beta = jnp.where(valid, beta, 0)
  gating_log = jnp.where(valid, gating_log, 0)

  outputs = []
  states = []
  for idx in range(cfg.seq_tile_size):
    out, state = _chunked_gdn_per_seq(
        q_large[idx],
        k_large[idx],
        v_large[idx],
        gating_log[idx],
        beta[idx],
        state_prev[idx],
        cfg,
        masks,
        grouped_state_scratch_ref,
    )
    outputs.append(out)
    states.append(state)

  return tuple(outputs), jnp.stack(states, axis=0)


def _native_qkv_slab(
    qkv_slot_ref: Any,
    token_start: jax.Array | int,
    head_start: int,
    head_end: int,
    input_offset: jax.Array,
    token_rows: int = 16,
) -> jax.Array:
  """Read 128-channel head panels without changing native token/channel tiling.

  token_start/token_rows are multiples of 16. input_offset locates the
  request within its eight-row-aligned DMA window; stacking adds an untiled
  head axis.

  Args:
    qkv_slot_ref: Private packed-QKV window for one prefill tile, including
      alignment rows.
    token_start: First tile-relative token row to load; a multiple of 16.
    head_start: First packed-head index of the slab, counting Q, K, then V
      heads.
    head_end: Exclusive packed-head index at the end of the slab.
    input_offset: Logical token offset within the private eight-row-aligned
      input window.
    token_rows: Rows to load per head; a static multiple of 16.

  Returns:
    FP32 values [head_end - head_start, token_rows, 128] at the logical token
    start.
  """

  def aligned() -> jax.Array:
    """Read aligned tokens as FP32 head-major values."""
    return jnp.stack(
        [
            qkv_slot_ref[
                pl.ds(token_start, token_rows), head * 128 : (head + 1) * 128
            ].astype(jnp.float32)
            for head in range(head_start, head_end)
        ]
    )

  def unaligned() -> jax.Array:
    """Rotate away the alignment prefix and return FP32 head-major values."""
    values = jnp.stack(
        [
            qkv_slot_ref[
                pl.ds(token_start, token_rows + 8),
                head * 128 : (head + 1) * 128,
            ].astype(jnp.float32)
            for head in range(head_start, head_end)
        ]
    )
    # Drop the neighboring request's prefix; TPU rotation needs a positive shift.
    return pltpu.roll(values, shift=token_rows + 8 - input_offset, axis=1)[
        :, :token_rows, :
    ]

  return jax.lax.cond(input_offset == 0, aligned, unaligned)


def _tail_conv_history(
    qkv_slot_ref: Any,
    previous: jax.Array,
    size: jax.Array,
    head_start: int,
    head_end: int,
    input_offset: jax.Array,
) -> jax.Array:
  """Select final history from aligned windows without dynamic sublane loads.

  previous is [heads, kernel_size - 1, 128] FP32 and size is positive. A tile
  shorter than the history retains the required prior suffix.

  Args:
    qkv_slot_ref: Private packed-QKV window for one prefill tile, including
      alignment rows.
    previous: Previous FP32 convolution history [heads, history_rows, 128].
    size: Positive number of valid token rows in the final convolution tile.
    head_start: First packed-head index of the slab, counting Q, K, then V
      heads.
    head_end: Exclusive packed-head index at the end of the slab.
    input_offset: Logical token offset within the private eight-row-aligned
      input window.

  Returns:
    FP32 final history with the same shape as previous.
  """
  history_size = previous.shape[1]
  if not history_size:
    return previous
  # Select among static slices to avoid dynamic VMEM sublane indexing.
  older = []
  for offset in range(history_size - 1):
    old = previous[:, -1:, :]
    for length in range(history_size - offset - 2, 0, -1):
      old = jnp.where(
          size == length,
          previous[:, offset + length : offset + length + 1, :],
          old,
      )
    older.append(old)
  rows = []
  for offset in range(history_size):
    token = size - history_size + offset
    safe_token = jnp.maximum(token, 0)
    start = pl.multiple_of((safe_token // 16) * 16, 16)
    values = _native_qkv_slab(
        qkv_slot_ref, start, head_start, head_end, input_offset
    )
    # TPU rotation requires a nonnegative shift.
    row = pltpu.roll(values, shift=(-safe_token) % 16, axis=1)[:, :1, :]
    if offset < history_size - 1:
      row = jnp.where(token >= 0, row, older[offset])
    rows.append(row)
  return jnp.concatenate(rows, axis=1)


def conv_silu_activation(
    result: jax.Array,
    half: jax.Array,
    *,
    cast_before_norm: bool,
    head_start: int,
    n_kq: int,
) -> jax.Array:
  """Apply SiLU, normalize Q/K, and scale Q by the head dimension.

  Slabs hold <= 16 heads; static Q/K/V boundaries select each head's
  treatment.

  Args:
    result: Convolution values [slab_heads, ..., head_dim]; current callers use
      FP32.
    half: Scalar 0.5 used by the tanh form of SiLU.
    cast_before_norm: Whether to cast activated values to FP32 before Q/K
      normalization.
    head_start: First packed-head index of the slab, counting Q, K, then V
      heads.
    n_kq: Number of query/key heads.

  Returns:
    Activated values with result's shape; Q/K are normalized and Q is scaled.
  """
  head_end = head_start + result.shape[0]
  if head_end <= n_kq:
    slab = 0
  elif n_kq <= head_start and head_end <= 2 * n_kq:
    slab = 1
  elif head_start >= 2 * n_kq:
    slab = 2
  else:
    slab = -1
  activated = result * (half + half * jnp.tanh(half * result))
  if slab >= 2:
    return activated
  if cast_before_norm:
    activated = activated.astype(jnp.float32)
  ss = jnp.sum(activated * activated, axis=-1, keepdims=True, dtype=jnp.float32)
  normalized = activated * jax.lax.rsqrt(ss + _L2_NORM_EPSILON)
  if slab == 0:
    return normalized * (result.shape[-1] ** -0.5)
  if slab == 1:
    return normalized
  # Static slices avoid reshaping a short index vector for broadcasting.
  q_end = max(0, min(n_kq - head_start, result.shape[0]))
  k_end = max(0, min(2 * n_kq - head_start, result.shape[0]))
  pieces = []
  if q_end:
    pieces.append(normalized[:q_end] * (result.shape[-1] ** -0.5))
  if k_end > q_end:
    pieces.append(normalized[q_end:k_end])
  if k_end < result.shape[0]:
    pieces.append(activated[k_end:])
  return jnp.concatenate(pieces, axis=0)


def dense_conv_silu(
    qkv_slot_ref: Any,
    real_sizes: jax.Array,
    dense_conv_ref: tuple[Any, Any | None],
    qkv_head_major_scratch_ref: Any,
    conv_state_slot_ref: Any,
    carry_conv_scratch_ref: Any,
    *,
    cfg: config.GDNConfig,
    input_offset: jax.Array,
) -> None:
  """Convolve native QKV windows directly into FP32 [heads, chunk_size, 128].

  QKV is [chunk_size + 8, width] BF16/FP32; input_offset is in [0, 8).
  Weights are [kernel_size, heads, 128] FP32. Both histories are [1, max(1,
  kernel_size - 1), 1, width]; the one-tap row is unused. Carry stays FP32;
  cache keeps its dtype. real_sizes bounds the valid prefix, and inactive QKV
  rows must be cleared before entry.

  Args:
    qkv_slot_ref: Private packed-QKV window for one prefill tile, including
      alignment rows.
    real_sizes: Per-sequence valid token counts; this fused path uses one
      sequence.
    dense_conv_ref: FP32 convolution tap and optional bias refs in head-major
      layout.
    qkv_head_major_scratch_ref: Shared FP32 head-major scratch; decode may
      borrow it before prefill.
    conv_state_slot_ref: Mutable per-request convolution history prepared for
      cache writeback.
    carry_conv_scratch_ref: Mutable FP32 convolution history carried between
      prefill tiles.
    cfg: Static GDN configuration defining head layout, tile sizes, and dtypes.
    input_offset: Logical token offset within the private eight-row-aligned
      input window.
  """
  weight_ref, bias_ref = dense_conv_ref
  half = jnp.asarray(0.5, dtype=jnp.float32)
  packed_heads = 2 * cfg.num_kq_heads + cfg.num_v_heads
  history_size = cfg.kernel_size - 1
  # Wider row panels amortize staging while preserving per-row arithmetic.
  token_rows = (
      64 if cfg.num_kq_heads in (2, 4, 8) and cfg.v_per_kq_head == 3 else 16
  )

  for head_start in range(0, packed_heads, 16):
    head_end = min(head_start + 16, packed_heads)
    weights = [
        weight_ref[tap, head_start:head_end, :][:, None, :]
        for tap in range(cfg.kernel_size)
    ]
    bias = (
        None
        if bias_ref is None
        else bias_ref[head_start:head_end, :][:, None, :]
    )
    if history_size:
      previous = jnp.stack(
          [
              carry_conv_scratch_ref[0, :, 0, head * 128 : (head + 1) * 128]
              for head in range(head_start, head_end)
          ]
      )
      history = previous

    for token_start in range(0, cfg.chunk_size, token_rows):
      values = _native_qkv_slab(
          qkv_slot_ref,
          token_start,
          head_start,
          head_end,
          input_offset,
          token_rows,
      )
      if history_size:
        window = jnp.concatenate((history, values), axis=1)
        result = window[:, :token_rows, :] * weights[0]
        for tap in range(1, history_size):
          result = result + window[:, tap : tap + token_rows, :] * weights[tap]
        result = result + values * weights[history_size]
      else:
        result = values * weights[0]
      if bias is not None:
        result = result + bias
      activated = conv_silu_activation(
          result,
          half,
          cast_before_norm=True,
          head_start=head_start,
          n_kq=cfg.num_kq_heads,
      )
      qkv_head_major_scratch_ref[
          head_start:head_end, token_start : token_start + token_rows, :
      ] = activated.astype(jnp.float32)
      if history_size:
        # History longer than this slab also needs earlier input rows.
        history = (
            values[:, -history_size:, :]
            if history_size <= token_rows
            else window[:, -history_size:, :]
        )

    if history_size:
      final = jax.lax.cond(
          real_sizes[0] == cfg.chunk_size,
          lambda: history,
          lambda: _tail_conv_history(
              qkv_slot_ref,
              previous,
              real_sizes[0],
              head_start,
              head_end,
              input_offset,
          ),
      )
      for head in range(head_start, head_end):
        channels = slice(head * 128, (head + 1) * 128)
        conv_state_slot_ref[0, :, 0, channels] = final[
            head - head_start
        ].astype(conv_state_slot_ref.dtype)
        carry_conv_scratch_ref[0, :, 0, channels] = final[head - head_start]


def _shift_output_rows(x: jax.Array, offset: jax.Array) -> jax.Array:
  """Shift [chunk_size, 128] into FP32 [chunk_size + 16, 128].

  offset must be in [0, 16) so valid rows cannot wrap.

  Args:
    x: Unshifted output for one head, shaped [chunk_size, 128].
    offset: Row offset within a 16-row output block, in [0, 16).

  Returns:
    FP32 [chunk_size + 16, 128] values shifted down by offset rows, with zero
    padding.
  """
  padded = jnp.pad(x.astype(jnp.float32), ((0, OUTPUT_ALIGNMENT), (0, 0)))
  return pltpu.roll(padded, shift=offset, axis=0)


def store_prefill_output(
    out: jax.Array,
    out_slot_ref: Any,
    output_carry_ref: Any,
    offset: jax.Array,
    flush_rows: jax.Array,
) -> None:
  """Pack head-major output and retain its incomplete aligned tail.

  out is [n_v, chunk_size, 128]; out_slot_ref is [chunk_size + 16,
  output_width] and carry is [16, output_width], both in the activation
  dtype. Drain slot DMA reads first. carry holds earlier rows below offset in
  [0, 16); flush_rows counts rows in complete 16-row blocks.

  Args:
    out: Head-major output values to pack into token-major storage.
    out_slot_ref: Mutable VMEM output tile [chunk_size + 16, n_v * d_v].
    output_carry_ref: Mutable [16, n_v * d_v] partial output block in activation
      dtype.
    offset: Row offset within a 16-row output block, in [0, 16).
    flush_rows: Number of output rows forming complete 16-row blocks.
  """
  tile_rows = out.shape[1]

  @pl.when(offset == 0)
  def aligned() -> None:
    """Store aligned prefill results and zero the spare output rows."""
    for head in range(out.shape[0]):
      col = head * out.shape[-1]
      out_slot_ref[:tile_rows, col : col + out.shape[-1]] = out[head]
    out_slot_ref[tile_rows:, :] = jnp.zeros(
        (OUTPUT_ALIGNMENT, out_slot_ref.shape[-1]),
        dtype=out_slot_ref.dtype,
    )

  @pl.when(offset != 0)
  def unaligned() -> None:
    """Shift prefill results and merge the previous partial-block carry."""
    for head in range(out.shape[0]):
      col = head * out.shape[-1]
      out_slot_ref[:, col : col + out.shape[-1]] = _shift_output_rows(
          out[head], offset
      ).astype(out_slot_ref.dtype)

    row = jax.lax.broadcasted_iota(
        jnp.int32, (OUTPUT_ALIGNMENT, out_slot_ref.shape[-1]), 0
    )
    out_slot_ref[:OUTPUT_ALIGNMENT, :] = jnp.where(
        row < offset,
        output_carry_ref[...],
        out_slot_ref[:OUTPUT_ALIGNMENT, :],
    )

  tail_start = pl.multiple_of(flush_rows, OUTPUT_ALIGNMENT)
  output_carry_ref[...] = out_slot_ref[pl.ds(tail_start, OUTPUT_ALIGNMENT), :]


def store_decode_output(
    out: jax.Array,
    output_carry_ref: Any,
    offset: jax.Array,
) -> None:
  """Insert one [n_v, 1, 128] decode output into the [16, out_width] carry.

  offset must be in [0, 16); preserve other rows in the activation dtype.

  Args:
    out: Head-major output values to pack into token-major storage.
    output_carry_ref: Mutable [16, n_v * d_v] partial output block in activation
      dtype.
    offset: Row offset within a 16-row output block, in [0, 16).
  """
  row = jax.lax.broadcasted_iota(
      jnp.int32, (OUTPUT_ALIGNMENT, out.shape[-1]), 0
  )
  selected = row == offset
  for head in range(out.shape[0]):
    col = head * out.shape[-1]
    value = out[head].astype(output_carry_ref.dtype)
    output_carry_ref[:, col : col + out.shape[-1]] = jnp.where(
        selected,
        value,
        output_carry_ref[:, col : col + out.shape[-1]],
    )
