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
which is what drives the split intra-chunk solve in `chunked_kda_per_seq`.
"""

import jax
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import compute_gdn
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config


def _l2_norm_f32(x: jax.Array, eps: float = 1e-6) -> jax.Array:
  """L2-normalize along the last dim, accumulating in float32."""
  # Deliberately not `compute_gdn.l2_norm`, which accumulates in the input
  # dtype. KDA runs q/k through the per-channel gate, so a bf16 sum of
  # squares here loses enough precision to show up against the fp64
  # goldens.
  x_f32 = x.astype(jnp.float32)
  inv_norm = jax.lax.rsqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
  return (x_f32 * inv_norm).astype(x.dtype)


def activate_gate(
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> jax.Array:
  """Turn the raw gate `a` into a per-channel log-decay."""
  # `a_log` and `dt_bias` arrive already reshaped to broadcast against `a`;
  # the two callers work in different ranks, so they do that themselves.
  # Both forms return values <= 0, which is what the chunked solve relies on
  # (see `chunked_kda_per_seq`): the cumulative gate must be monotonically
  # decreasing so every causal exponent `g_r - g_t` is non-positive.
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
  """Per-sequence chunked KDA recurrence with sub-block intra-chunk solve."""
  q = jnp.repeat(q_large, cfg.v_per_kq_head, axis=0)
  k = jnp.repeat(k_large, cfg.v_per_kq_head, axis=0)

  beta = compute_gdn.fused_transpose_broadcast(beta, src_dim=2, dst_dim=0)
  beta = beta[: cfg.num_v_heads]

  g_cum_sum_list = [gating_log[:, :1]]
  for row in range(1, cfg.chunk_size):
    g_cum_sum_list.append(g_cum_sum_list[-1] + gating_log[:, row : row + 1])
  g_cumsum = jnp.concat(g_cum_sum_list, axis=1)

  # The chunk is partitioned into sub-blocks of `block_size`. For each row
  # sub-block row_block we compute the attention scores (Aqk) and Householder
  # transition blocks (L) against every column block, then immediately use
  # L[row_block, :row_block] to forward-substitute that row strip of T_inv.
  #
  # Off-diagonal (col_block < row_block), g_ref separates the rows from the columns:
  # g_cumsum[t] >= g_ref for every column and g_r <= g_ref for every row, so
  # both exponents are provably <= 0, neither can overflow, and underflow to
  # 0 is a fully decayed contribution.
  #
  # On the diagonal the rows and columns are the same tokens, so g_ref
  # cannot sit between them and one factor would carry a positive exponent
  # that can overflow float32 exp when paired with an unbounded gate. Those
  # blocks take exact pairwise differences g_r - g_t, which stay <= 0 for
  # causal pairs because g_cumsum is monotonically decreasing.
  block_size = min(cfg.triangular_block_size, cfg.chunk_size)
  num_blocks = cfg.chunk_size // block_size

  identity_chunk = jnp.eye(cfg.chunk_size, dtype=jnp.float32)
  causal_mask_block = (
      jnp.arange(block_size)[:, None] >= jnp.arange(block_size)[None, :]
  )

  aqk_row_strips = []
  t_inv_row_strips = []

  for row_block in range(num_blocks):
    r_start, r_end = row_block * block_size, (row_block + 1) * block_size

    q_row = q[:, r_start:r_end]
    k_row = k[:, r_start:r_end]
    beta_row = beta[:, r_start:r_end]
    g_row = g_cumsum[:, r_start:r_end]
    # Sub-block anchor: row 0 of the current sub-block.
    g_ref = g_row[:, :1, :]

    aqk_col_blocks = []
    l_interaction_blocks = []

    for col_block in range(num_blocks):
      c_start, c_end = col_block * block_size, (col_block + 1) * block_size
      k_col = k[:, c_start:c_end]
      g_col = g_cumsum[:, c_start:c_end]

      if col_block < row_block:
        # Off-diagonal: factored through g_ref as a batched matmul.
        q_scaled = q_row * jnp.exp(g_row - g_ref)
        k_beta_scaled = (k_row * beta_row) * jnp.exp(g_row - g_ref)
        k_col_scaled = k_col * jnp.exp(g_ref - g_col)

        # [H, 2*BC, K] @ [H, BC, K]^T -> [H, 2*BC, BC], giving
        # both Aqk[row_block, col_block] and L[row_block, col_block] in one contraction.
        qk_scaled_merged = jnp.concat([q_scaled, k_beta_scaled], axis=1)
        gemm_out = jax.lax.dot(
            qk_scaled_merged,
            k_col_scaled,
            dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        b_aqk, b_l = jnp.split(gemm_out, 2, axis=1)

        aqk_col_blocks.append(b_aqk)
        l_interaction_blocks.append(b_l)

      elif col_block == row_block:
        # Diagonal: exact pairwise differences [H, BC, BC, K].
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

    # Solve T_inv for the Current Row Sub-Block row_block
    # target to invert: (I + StrictTril(L)) @ T_inv = I
    # for row block row_block:
    #   T_inv[row_block, :] = (I[row_block, :] - L[row_block, :row_block] @ T_inv[:row_block, :])
    target_rhs = jnp.broadcast_to(
        identity_chunk[None, r_start:r_end, :],
        (cfg.num_v_heads, block_size, cfg.chunk_size),
    )

    if row_block > 0:
      # Subtract the already-solved rows: L_past @ T_inv_past.
      l_past = jnp.concat(l_interaction_blocks, axis=2)  # [H, BC, r_start]
      t_inv_past = jnp.concat(
          t_inv_row_strips, axis=1
      )  # [H, r_start, chunk_size]
      prev_contribution = jax.lax.dot(
          l_past,
          t_inv_past,
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
        l_row_i = jnp.sum(
            (k_row[:, i : i + 1, :] * beta_row[:, i : i + 1, :])
            * k_decayed_prev,
            axis=-1,
        )  # [H, i]

        # x[i] = rhs[i] - sum_{j < i} (L[i, j] * x[j])
        solved_so_far = jnp.stack(x_local_rows, axis=1)  # [H, i, chunk_size]
        x_row = rhs_row - jnp.sum(l_row_i[..., None] * solved_so_far, axis=1)

      x_local_rows.append(x_row)

    t_inv_row_strips.append(jnp.stack(x_local_rows, axis=1))

  aqk = jnp.concat(aqk_row_strips, axis=1)  # [H, chunk_size, chunk_size]
  t_inv = jnp.concat(t_inv_row_strips, axis=1)  # [H, chunk_size, chunk_size]

  v_beta = v_large * beta
  k_beta_gating = (k * beta) * jnp.exp(g_cumsum)

  merged_v_k = jnp.concat([v_beta, k_beta_gating], axis=-1)
  merged_uw = jax.lax.dot(
      t_inv,
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
      aqk,
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
  """Chunked KDA over a tile of sequences."""
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
  # Chunked KDA is only selected for a one-checkpoint window; speculative
  # verify uses the recurrent scan below.
  state = jnp.stack(state_list, axis=0)[:, jnp.newaxis]
  return out, state


def recurrent_kda_per_seq(
    q: jax.Array,  # (num_kq_heads, chunk, kq_head_dim)
    k: jax.Array,  # (num_kq_heads, chunk, kq_head_dim)
    v: jax.Array,  # (num_v_heads, chunk, v_head_dim)
    gating_decay: jax.Array,  # (num_v_heads, chunk, kq_head_dim)
    beta: jax.Array,  # (num_v_heads, chunk, 1)
    state: jax.Array,  # (num_v_heads, kq_head_dim, v_head_dim)
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
  """Run the token recurrence and retain one state per verify position."""
  # BATCHED speculative decode sets `chunk_size == window_size` so every
  # post-token state is retained. The ordinary decode case is the degenerate
  # one-token window. Invalid ragged tail rows have q/k/v/beta zeroed and a
  # unit decay, therefore their output is zero and their state is unchanged;
  # memory_ref only writes checkpoints belonging to real rows.
  out_list = []
  state_list = []
  contract_dk = (((2,), (1,)), ((0,), (0,)))

  for c_idx in range(cfgs.chunk_size):
    # Repeat Q and K for GQA.
    q_heads = jnp.repeat(q[:, c_idx], cfgs.v_per_kq_head, axis=0)[
        :, None, :
    ]  # (num_v_heads, 1, kq_head_dim)
    k_heads = jnp.repeat(k[:, c_idx], cfgs.v_per_kq_head, axis=0)[
        :, None, :
    ]  # (num_v_heads, 1, kq_head_dim)
    k_heads_t = compute_gdn.fused_transpose_broadcast(
        k_heads, src_dim=2, dst_dim=1
    )  # (num_v_heads, kq_head_dim, 1)

    v_heads = v[:, c_idx : c_idx + 1]
    beta_heads = beta[:, c_idx : c_idx + 1]
    decay = gating_decay[:, c_idx, :, None]

    state_updated = state * decay
    v_updated = jax.lax.dot(
        k_heads,
        state_updated,
        dimension_numbers=contract_dk,
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)

    v_new = beta_heads * (v_heads - v_updated)
    state = state_updated + k_heads_t * v_new

    out = jax.lax.dot(
        q_heads,
        state,
        dimension_numbers=contract_dk,
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)
    out_list.append(out[:, 0, :])

    if c_idx >= cfgs.chunk_size - cfgs.window_size:
      state_list.append(state)

  return jnp.stack(out_list, axis=0), jnp.stack(state_list, axis=0)


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
  """Token-recurrent KDA scan over a tile of sequences."""
  mask_dtype = compute_gdn.get_mask_dtype(cfg.dtypes.compute)
  iota = jax.lax.broadcasted_iota(
      mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1, 1), 2
  )
  mask = iota < real_sizes.reshape(-1, 1, 1, 1, 1).astype(mask_dtype)

  q = jnp.where(mask, q_compact.astype(cfg.dtypes.compute), 0.0)
  k = jnp.where(mask, k_compact.astype(cfg.dtypes.compute), 0.0)
  v = jnp.where(mask, v_compact.astype(cfg.dtypes.compute), 0.0)

  q = _l2_norm_f32(q) * (cfg.kq_head_dim**-0.5)
  k = _l2_norm_f32(k)

  # Delta rule step-size beta in [0, 1]
  beta = jnp.where(mask, jax.nn.sigmoid(b_compact.astype(jnp.float32)), 0.0)
  if cfg.window_size > 1:
    # Match `_activate_beta` feeding the one-token decode kernel: its
    # sigmoid result is rounded to the model activation dtype before the
    # recurrent update. Return to fp32 here so the fused scan itself keeps
    # its existing compute dtype and layout.
    beta = beta.astype(cfg.dtypes.act_out).astype(jnp.float32)
  beta = compute_gdn.fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)
  beta = beta[:, : cfg.num_v_heads, :, 0, :]

  a_log_kda = a_log.reshape(1, cfg.num_v_heads, 1, 1, 1).astype(jnp.float32)
  dt_bias_kda = dt_bias.reshape(
      1, cfg.num_v_heads, 1, 1, cfg.kq_head_dim
  ).astype(jnp.float32)
  gating_log = activate_gate(a_compact, a_log_kda, dt_bias_kda, cfg)
  # Invalid ragged tail tokens must leave the state untouched.
  gating_log = jnp.where(mask, gating_log, 0.0)
  gating_decay = jnp.exp(gating_log)[:, :, :, 0, :]

  # Drop the compact layout's singleton dimension before the token scan.
  q = q[:, :, :, 0, :]
  k = k[:, :, :, 0, :]
  v = v[:, :, :, 0, :]

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
    out_list.append(out)
    new_state_list.append(state)

  out = jnp.stack(out_list, axis=0)
  new_recurrent_state = jnp.stack(new_state_list, axis=0)
  return out, new_recurrent_state
