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
"""Compute kernels for KDA (Kimi Delta Attention) linear attention.

The KDA counterpart of `compute_gdn`, sharing its tiling, state layout and
helpers. The one structural difference is the gate: KDA's is per-channel,
so decay is rank-3 over (row, col, channel) rather than a per-head scalar,
which is what drives the split intra-chunk solve in
`chunked_kda_per_seq`.
"""

import jax
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import compute_gdn
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config


def _l2_norm_f32(x: jax.Array, eps: float = 1e-6) -> jax.Array:
  """L2-normalize along the last dim, accumulating in float32.

  Deliberately not `compute_gdn.l2_norm`, which accumulates in the input
  dtype. KDA runs q/k through the per-channel gate, so a bf16 sum of
  squares here loses enough precision to show up against the fp64
  goldens.
  """
  x_f32 = x.astype(jnp.float32)
  inv_norm = jax.lax.rsqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
  return (x_f32 * inv_norm).astype(x.dtype)


def activate_gate(
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> jax.Array:
  """Turn the raw gate `a` into a per-channel log-decay.

  `a_log` and `dt_bias` arrive already reshaped to broadcast against `a`;
  the two callers work in different ranks, so they do that themselves.

  Both forms return values <= 0, which is what the chunked solve relies on
  (see `chunked_kda_per_seq`): the cumulative gate must be monotonically
  decreasing so every causal exponent `g_r - g_t` is non-positive.
  """
  a_f32 = a.astype(jnp.float32)
  if cfg.gate_lower_bound is None:
    # Unbounded: the decay can reach 0, i.e. a channel forgets entirely.
    return -jnp.exp(a_log) * jax.nn.softplus(a_f32 + dt_bias)
  # Bounded: the log-decay is floored at `gate_lower_bound`, so the
  # per-token decay never drops below exp(gate_lower_bound).
  return cfg.gate_lower_bound * jax.nn.sigmoid(
      jnp.exp(a_log) * (a_f32 + dt_bias)
  )


def chunked_kda_per_seq(
    q_large: jax.Array,  # (num_kq_heads, chunk, kq_head_dim)
    k_large: jax.Array,  # (num_kq_heads, chunk, kq_head_dim)
    v_large: jax.Array,  # (num_v_heads, chunk, v_head_dim)
    gating_log: jax.Array,  # KDA specific: (num_v_heads, chunk, kq_head_dim)
    beta: jax.Array,  # (1, chunk, num_v_heads)
    state_prev: jax.Array,  # (num_v_heads, kq_head_dim, v_head_dim)
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
  q = jnp.repeat(q_large, cfg.v_per_kq_head, axis=0)
  k = jnp.repeat(k_large, cfg.v_per_kq_head, axis=0)

  beta = compute_gdn.fused_transpose_broadcast(beta, src_dim=2, dst_dim=0)
  beta = beta[: cfg.num_v_heads]

  g_cum_sum_list = [gating_log[:, :1]]
  for row in range(1, cfg.chunk_size):
    g_cum_sum_list.append(g_cum_sum_list[-1] + gating_log[:, row : row + 1])
  g_cumsum = jnp.concat(g_cum_sum_list, axis=1)

  # The chunk is partitioned into sub-blocks of `block_size`. For each row
  # sub-block r_b we compute the attention scores (Aqk) and Householder
  # transition blocks (L) against every column block, then immediately use
  # L[r_b, :r_b] to forward-substitute that row strip of T_inv.
  #
  # Diagonal and off-diagonal blocks take different paths. The split, and
  # the reasoning below, come from `kernels/kimi_k3/chunk_kda.py`, which
  # has the longer write-up and the measurements.
  #
  # The quantity wanted is, for every causal pair (r, t),
  #   Aqk[h, r, t] = sum_k q[h, r, k] * k[h, t, k]
  #                        * exp(g_cumsum[h, r, k] - g_cumsum[h, t, k])
  # KDA's gate is per-channel, so that decay is rank-3 over (r, t, k) and
  # the sum over k cannot be folded into a plain matmul. Factoring it
  # through a per-sub-block anchor restores the matmul:
  #   exp(g_r - g_ref) * exp(g_ref - g_t)
  # but those two factors are a tiny*huge pair. Their product is
  # exp(g_r - g_t) <= 1 while individually they can each saturate float32's
  # exp range.
  #
  # Off-diagonal (c_b < r_b), g_ref separates the rows from the columns:
  # g_cumsum[t] >= g_ref for every column and g_r <= g_ref for every row, so
  # both exponents are provably <= 0, neither can overflow, and underflow to
  # 0 is the right answer -- a fully decayed contribution. That block is the
  # factored GEMM on the MXU.
  #
  # On the diagonal the rows and columns are the same tokens, so g_ref
  # cannot sit between them and one factor has to carry a positive exponent.
  # Kimi-Linear's gate is unbounded and reaches -1736 for a *single* token,
  # so one factor goes to 0 and the other to inf: 0 * inf = NaN. On real
  # layer-0 weights that was 26 of 32 heads NaN, reaching the sampler as
  # non-finite logits. Those blocks take exact pairwise differences
  # g_r - g_t on the VPU, which stays <= 0 for causal pairs because
  # g_cumsum is monotonically decreasing.
  #
  # Hence the precondition `activate_gate` upholds and `fused_conv1d_gdn`
  # validates: the activated gate is <= 0, i.e. `gate_lower_bound` is
  # negative when given.
  #
  # TODO: a bounded gate could take the factored path on the
  # diagonal too, collapsing the rank-4 [H, BC, BC, K] VPU contraction into
  # one more GEMM. The largest exponent that produces is the spread of
  # g_cumsum across one sub-block, |lower_bound| * (block_size - 1): 75 at
  # -5.0 and block_size=16, against float32's exp overflow at ~88.7. The
  # headroom is thin and scales with block_size -- 32 gives 155 and
  # overflows even bounded -- so it needs a guard on
  # `triangular_block_size` as well as on the bound.
  block_size = min(cfg.triangular_block_size, cfg.chunk_size)
  num_blocks = cfg.chunk_size // block_size

  identity_chunk = jnp.eye(cfg.chunk_size, dtype=jnp.float32)
  causal_mask_block = (
      jnp.arange(block_size)[:, None] >= jnp.arange(block_size)[None, :]
  )

  aqk_row_strips = []
  t_inv_row_strips = []

  for r_b in range(num_blocks):
    r_start, r_end = r_b * block_size, (r_b + 1) * block_size

    q_row = q[:, r_start:r_end]
    k_row = k[:, r_start:r_end]
    beta_row = beta[:, r_start:r_end]
    g_row = g_cumsum[:, r_start:r_end]
    # Sub-block anchor: row 0 of the current sub-block.
    g_ref = g_row[:, :1, :]

    aqk_col_blocks = []
    L_interaction_blocks = []

    for c_b in range(num_blocks):
      c_start, c_end = c_b * block_size, (c_b + 1) * block_size
      k_col = k[:, c_start:c_end]
      g_col = g_cumsum[:, c_start:c_end]

      if c_b < r_b:
        # Off-diagonal: factored through g_ref, GEMM on the MXU.
        q_scaled = q_row * jnp.exp(g_row - g_ref)
        k_beta_scaled = (k_row * beta_row) * jnp.exp(g_row - g_ref)
        k_col_scaled = k_col * jnp.exp(g_ref - g_col)

        # [H, 2*BC, K] @ [H, BC, K]^T -> [H, 2*BC, BC], giving
        # both Aqk[r_b, c_b] and L[r_b, c_b] in one instruction.
        qk_scaled_merged = jnp.concat([q_scaled, k_beta_scaled], axis=1)
        gemm_out = jax.lax.dot(
            qk_scaled_merged,
            k_col_scaled,
            dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        b_aqk, b_L = jnp.split(gemm_out, 2, axis=1)

        aqk_col_blocks.append(b_aqk)
        L_interaction_blocks.append(b_L)

      elif c_b == r_b:
        # Diagonal: exact pairwise differences on the VPU. [H, BC,
        # BC, K].
        delta_g = g_row[:, :, None, :] - g_col[:, None, :, :]
        decay_diag = jnp.exp(jnp.minimum(delta_g, 0.0))
        k_col_decayed = k_col[:, None, :, :] * decay_diag

        # Exact channel contraction: sum_k (q[r] * k[c] * decay).
        b_aqk = jnp.sum(q_row[:, :, None, :] * k_col_decayed, axis=-1)
        b_aqk = jnp.where(causal_mask_block[None, :, :], b_aqk, 0.0)
        aqk_col_blocks.append(b_aqk)

      else:
        b_zero = jnp.zeros(
            (cfg.num_v_heads, block_size, block_size), dtype=jnp.float32
        )
        aqk_col_blocks.append(b_zero)

    # concat into: [H, BC, chunk_size]
    aqk_row_strips.append(jnp.concat(aqk_col_blocks, axis=2))

    # Solve T_inv for the Current Row Sub-Block r_b
    # target to invert: (I + StrictTril(L)) @ T_inv = I
    # for row block r_b:
    #   T_inv[r_b, :] = (I[r_b, :] - L[r_b, :r_b] @ T_inv[:r_b, :])
    target_rhs = jnp.broadcast_to(
        identity_chunk[None, r_start:r_end, :],
        (cfg.num_v_heads, block_size, cfg.chunk_size),
    )

    if r_b > 0:
      # Subtract the already-solved rows: L_past @ T_inv_past.
      L_past = jnp.concat(L_interaction_blocks, axis=2)  # [H, BC, r_start]
      T_inv_past = jnp.concat(
          t_inv_row_strips, axis=1
      )  # [H, r_start, chunk_size]
      prev_contribution = jax.lax.dot(
          L_past,
          T_inv_past,
          dimension_numbers=(((2,), (1,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
      )
      target_rhs = target_rhs - prev_contribution

    # Local row-by-row forward substitution within the diagonal sub-block
    x_local_rows = []
    for i in range(block_size):
      rhs_row = target_rhs[:, i, :]
      if i == 0:
        x_row = rhs_row
      else:
        # Pairwise channel decay against strictly preceding
        # tokens in this sub-block.
        delta_g_i = g_row[:, i : i + 1, :] - g_row[:, :i, :]
        k_decayed_prev = k_row[:, :i, :] * jnp.exp(jnp.minimum(delta_g_i, 0.0))

        # Transition row: beta[i] * k[i] @ (k[:i] * decay)^T
        L_row_i = jnp.sum(
            (k_row[:, i : i + 1, :] * beta_row[:, i : i + 1, :])
            * k_decayed_prev,
            axis=-1,
        )  # [H, i]

        # x[i] = rhs[i] - sum_{j < i} (L[i, j] * x[j])
        solved_so_far = jnp.stack(x_local_rows, axis=1)  # [H, i, chunk_size]
        x_row = rhs_row - jnp.sum(L_row_i[..., None] * solved_so_far, axis=1)

      x_local_rows.append(x_row)

    t_inv_row_strips.append(jnp.stack(x_local_rows, axis=1))

  Aqk = jnp.concat(aqk_row_strips, axis=1)  # [H, chunk_size, chunk_size]
  T_inv = jnp.concat(t_inv_row_strips, axis=1)  # [H, chunk_size, chunk_size]

  v_beta = v_large * beta
  k_beta_gating = (k * beta) * jnp.exp(g_cumsum)

  merged_v_k = jnp.concat([v_beta, k_beta_gating], axis=-1)
  merged_uw = jax.lax.dot(
      T_inv,
      merged_v_k,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )

  u, w = jnp.split(merged_uw, [cfg.v_head_dim], axis=-1)

  q_gated = q * jnp.exp(g_cumsum)
  wq = jnp.concat([w, q_gated], axis=1)
  ws_and_out = jax.lax.dot_general(
      wq,
      state_prev,
      (((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  ws, out_updated = jnp.split(ws_and_out, 2, axis=1)

  u_ws = u - ws

  g_last = g_cumsum[:, -1:, :]
  k_gating_last = k * jnp.exp(g_last - g_cumsum)
  state_new = jax.lax.dot_general(
      k_gating_last,
      u_ws,
      (((1,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  state_updated = state_prev * jnp.exp(g_last.swapaxes(1, 2))
  state = state_updated + state_new

  out_new = jax.lax.dot(
      Aqk,
      u_ws,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  out = (out_updated + out_new).astype(cfg.dtypes.compute)

  return out, state


def chunked_kda(
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
) -> tuple[jax.Array, jax.Array]:
  mask_dtype = compute_gdn.get_mask_dtype(cfg.dtypes.compute)
  iota = jax.lax.broadcasted_iota(
      mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1), 2
  )
  mask = iota < real_sizes.reshape(-1, 1, 1, 1).astype(mask_dtype)

  # (seqs, num_kq_heads, chunk, kq_head_dim)
  q_large = jnp.where(mask, q_large.astype(jnp.float32), 0.0)
  k_large = jnp.where(mask, k_large.astype(jnp.float32), 0.0)
  # (seqs, num_v_heads, chunk, v_head_dim)
  v_large = jnp.where(mask, v_large.astype(jnp.float32), 0.0)

  q_large = _l2_norm_f32(q_large)
  q_scale = cfg.kq_head_dim**-0.5
  q_large *= q_scale
  k_large = _l2_norm_f32(k_large)

  # (seqs, 1, chunk, num_v_heads)
  beta = jax.nn.sigmoid(b_large.astype(jnp.float32))
  beta = jnp.where(mask, beta, 0.0)

  a_log_kda = a_log.reshape(1, -1, 1, 1).astype(jnp.float32)
  dt_bias_kda = dt_bias.reshape(1, cfg.num_v_heads, 1, cfg.kq_head_dim).astype(
      jnp.float32
  )
  gating_log = activate_gate(a_large, a_log_kda, dt_bias_kda, cfg)
  gating_log = jnp.where(mask, gating_log, 0)

  out_list = []
  state_list = []
  for idx in range(cfg.seq_tile_size):
    out, state = chunked_kda_per_seq(
        q_large[idx],
        k_large[idx],
        v_large[idx],
        gating_log[idx],
        beta[idx],
        state_prev[idx],
        cfg,
    )
    out_list.append(out.swapaxes(0, 1))
    state_list.append(state)
  out = jnp.stack(out_list, axis=0)
  # The caller expects one state per window position. KDA rejects
  # speculative decoding, so `window_size` is always 1 and the only
  # checkpoint is the end-of-tile state.
  state = jnp.stack(state_list, axis=0)[:, jnp.newaxis]
  return out, state


def recurrent_kda_per_seq(
    q_curr: jax.Array,  # (num_kq_heads, kq_head_dim)
    k_curr: jax.Array,  # (num_kq_heads, kq_head_dim)
    v_curr: jax.Array,  # (num_v_heads, v_head_dim)
    gating_curr: jax.Array,  # (num_v_heads, kq_head_dim)
    beta_curr: jax.Array,  # (num_v_heads, 1)
    state: jax.Array,  # (num_v_heads, kq_head_dim, v_head_dim)
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
  # Repeat Q and K for GQA
  q_heads = jnp.repeat(q_curr, cfgs.v_per_kq_head, axis=0)[
      :, None, :
  ]  # (num_v_heads, 1, kq_head_dim)
  k_heads = jnp.repeat(k_curr, cfgs.v_per_kq_head, axis=0)[
      :, None, :
  ]  # (num_v_heads, 1, kq_head_dim)
  k_heads_t = compute_gdn.fused_transpose_broadcast(
      k_heads, src_dim=2, dst_dim=1
  )  # (num_v_heads, kq_head_dim, 1)

  v_heads = v_curr[:, None, :]  # (num_v_heads, 1, v_head_dim)
  beta_heads = beta_curr[:, None]  # (num_v_heads, 1, 1)
  gating_decay = gating_curr[..., None]  # (num_v_heads, kq_head_dim, 1)

  state_updated = state * gating_decay

  contract_dk = (((2,), (1,)), ((0,), (0,)))
  v_updated = jax.lax.dot(
      k_heads,
      state_updated,
      dimension_numbers=contract_dk,
      preferred_element_type=jnp.float32,
  ).astype(cfgs.dtypes.compute)

  v_diff = v_heads - v_updated
  v_new = beta_heads * v_diff

  state_new = k_heads_t * v_new
  state = state_updated + state_new

  out = jax.lax.dot(
      q_heads,
      state,
      dimension_numbers=contract_dk,
      preferred_element_type=jnp.float32,
  ).astype(cfgs.dtypes.compute)

  return out[:, 0, :], state


def recurrent_kda(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
  mask_dtype = compute_gdn.get_mask_dtype(cfg.dtypes.compute)
  valid_seq_mask = (real_sizes[:, None, None] > 0).astype(mask_dtype)

  # squeeze unnecessary axis
  q = q_compact[:, :, 0, 0, :].astype(cfg.dtypes.compute)
  k = k_compact[:, :, 0, 0, :].astype(cfg.dtypes.compute)
  v = v_compact[:, :, 0, 0, :].astype(cfg.dtypes.compute)
  a = a_compact[:, :, 0, 0, :].astype(jnp.float32)

  # Beta: [seqs, num_v_heads, 1]
  b = b_compact[:, 0, 0, 0, : cfg.num_v_heads, None].astype(jnp.float32)

  q = jnp.where(valid_seq_mask, q, 0.0)
  k = jnp.where(valid_seq_mask, k, 0.0)
  v = jnp.where(valid_seq_mask, v, 0.0)

  q = _l2_norm_f32(q) * (cfg.kq_head_dim**-0.5)
  k = _l2_norm_f32(k)

  # Delta rule step-size beta in [0, 1]
  beta = jnp.where(valid_seq_mask, jax.nn.sigmoid(b), 0.0)

  a_log_kda = a_log.reshape(1, cfg.num_v_heads, 1).astype(jnp.float32)
  dt_bias_kda = dt_bias.reshape(1, cfg.num_v_heads, cfg.kq_head_dim).astype(
      jnp.float32
  )
  gating_log = activate_gate(a, a_log_kda, dt_bias_kda, cfg)
  gating_log = jnp.where(valid_seq_mask, gating_log, 0.0)
  gating_decay = jnp.exp(gating_log)  # [seqs, num_v_heads, kq_head_dim]

  out_list = []
  new_state_list = []
  for idx in range(cfg.seq_tile_size):
    out, state = recurrent_kda_per_seq(
        q[idx],
        k[idx],
        v[idx],
        gating_decay[idx],
        beta[idx],
        state_prev[idx],
        cfg,
    )
    out_list.append(out[None, :])
    new_state_list.append(state)

  out = jnp.stack(out_list, axis=0)
  # One state per window position; see `chunked_kda`.
  new_recurrent_state = jnp.stack(new_state_list, axis=0)[:, jnp.newaxis]
  return out, new_recurrent_state
