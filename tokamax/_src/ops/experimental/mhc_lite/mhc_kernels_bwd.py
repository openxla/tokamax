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
"""Low-level Pallas backward kernels and custom VJP rules for mHC-lite."""

import functools
import math
from typing import Any
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.mhc_lite import common


def _read_streams_2d(
    ref,
    s_slice: slice,
    streams: int,
    embedding: int,
) -> list[jax.Array]:
  """Reads `streams` feature bands of shape `(sub_tokens, embedding)` in f32."""
  return [
      ref[s_slice, k * embedding : (k + 1) * embedding].astype(jnp.float32)
      for k in range(streams)
  ]


def _fold_to_num_lanes(
    x: jax.Array, embedding: int, num_lanes: int
) -> jax.Array:
  """Folds `(sub_tokens, embedding)` into `(sub_tokens, num_lanes)` in bf16.

  Accumulates `embedding // num_lanes` strided `num_lanes`-lane bands on the VPU
  with zero cross-lane shuffles:
    `S_tilde[t, c] = sum_{b=0}^{D/L - 1} x[t, L*b + c]` for `c in [0, L)`.

  Args:
    x: Input 2D array of shape `(sub_tokens, embedding)`.
    embedding: Feature dimension (multiple of `num_lanes`).
    num_lanes: Number of TPU vector lanes.

  Returns:
    Folded `bfloat16` array of shape `(sub_tokens, num_lanes)`.
  """
  acc = x[:, 0:num_lanes]
  for col in range(num_lanes, embedding, num_lanes):
    acc = acc + x[:, col : col + num_lanes]
  return acc.astype(jnp.bfloat16)


def _post_bwd_2d_mxu_core(
    streams: int,
    embedding: int,
    sub_tokens: int,
    num_subtiles: int,
    x_ref,
    layer_output_ref,
    h_post_ref,
    residual_ref,
    d_x_ref,
    d_layer_output_ref,
    d_h_post_ref,
    d_residual_ref,
    dy_sub_fn,
) -> None:
  """Shared 2D+MXU post-branch backward core for standalone and fused ops.

  Differentiates the post-branch forward mixing equation:
    `Y[t, j, e] = h_post[t, j] * L[t, e] + sum_k H_res[t, j, k] * X[t, k, e]`
  using VPU subtile loops for activation cotangents (`dL`, `dX`) and a two-stage
  reduction for gate/residual cotangents (`dh_post`, `dH_res`):
    - Stage A (VPU lane fold, `D -> num_lanes` per stream): folds elementwise
      products across `D // num_lanes` bands into `S_tilde_post` and
      `S_tilde_res^(j)` without cross-lane VPU shuffles.
    - Stage B (MXU `num_lanes -> 1` systolic contraction): contracts folded
      bands with binary selector matrices `U_post` and `U_res^(j)`:
        `dh_post[t, j] = sum_e L[t, e] * dY[t, j, e]`
        `              = (S_tilde_post @ U_post)[t, j]`
        `dH_res[t, j, k] = sum_e dY[t, j, e] * X[t, k, e]`
        `                = sum_j (S_tilde_res^(j) @ U_res^(j))`
  """
  num_lanes = pltpu.get_tpu_info().num_lanes
  with jax.named_scope("build_mxu_selector_matrices"):
    folded_cols = streams * num_lanes
    row_band_post = (
        jax.lax.broadcasted_iota(jnp.int32, (folded_cols, streams), 0)
        // num_lanes
    )
    u_post_bf16 = (
        row_band_post == jax.lax.broadcasted_iota(
            jnp.int32, (folded_cols, streams), 1
        )
    ).astype(jnp.bfloat16)

    row_band_res = (
        jax.lax.broadcasted_iota(jnp.int32, (folded_cols, streams * streams), 0)
        // num_lanes
    )
    col_id_res = jax.lax.broadcasted_iota(
        jnp.int32, (folded_cols, streams * streams), 1
    )
    u_res_mats = [
        (
            (col_id_res // streams == j)
            & (row_band_res == (col_id_res % streams))
        ).astype(jnp.bfloat16)
        for j in range(streams)
    ]

  s_post_rows = []
  s_res_rows_by_j = [[] for _ in range(streams)]

  with jax.named_scope("vpu_subtile_cotangents_and_fold"):
    for sub_idx in range(num_subtiles):
      with jax.named_scope(f"subtile_{sub_idx}"):
        s_slice = slice(None)
        if num_subtiles > 1:
          s_slice = slice(sub_idx * sub_tokens, (sub_idx + 1) * sub_tokens)
        with jax.named_scope("load_subtile_inputs"):
          dy_sub = dy_sub_fn(s_slice)
          l_sub = layer_output_ref[s_slice, :].astype(jnp.float32)
          h_post_sub = h_post_ref[s_slice, :]
          res_sub = residual_ref[s_slice, :].astype(jnp.float32)

        # Step 1 (VPU): dL[t, e] = sum_j h_post[t, j] * dY[t, j, e]
        with jax.named_scope("d_layer_output_vpu"):
          dl_sub = h_post_sub[:, 0:1] * dy_sub[0]
          for j in range(1, streams):
            dl_sub = dl_sub + h_post_sub[:, j : j + 1] * dy_sub[j]
          d_layer_output_ref[s_slice, :] = dl_sub.astype(
              d_layer_output_ref.dtype
          )

        # Step 2 (VPU): dX[t, k, e] = sum_j H_res[t, j, k] * dY[t, j, e]
        with jax.named_scope("d_x_vpu"):
          for k in range(streams):
            dx_k = res_sub[:, k : k + 1] * dy_sub[0]
            for j in range(1, streams):
              dx_k = (
                  dx_k
                  + res_sub[:, j * streams + k : j * streams + k + 1]
                  * dy_sub[j]
              )
            d_x_ref[s_slice, k * embedding : (k + 1) * embedding] = dx_k.astype(
                d_x_ref.dtype
            )

        # Stage A (VPU lane fold, D -> num_lanes per stream):
        # S_tilde_post[t, j*L + c] = sum_b (L * dY_j)[t, L*b + c]
        # S_tilde_res^(j)[t, k*L + c] = sum_b (dY_j * X_k)[t, L*b + c]
        with jax.named_scope("fold_d_h_post_128_lanes"):
          s_post_sub_parts = [
              _fold_to_num_lanes(l_sub * dy_sub[j], embedding, num_lanes)
              for j in range(streams)
          ]
          s_post_rows.append(jnp.concatenate(s_post_sub_parts, axis=1))

        with jax.named_scope("fold_d_residual_128_lanes"):
          s_res_sub_by_j = [[] for _ in range(streams)]
          for x_k in _read_streams_2d(x_ref, s_slice, streams, embedding):
            for j in range(streams):
              s_res_sub_by_j[j].append(
                  _fold_to_num_lanes(dy_sub[j] * x_k, embedding, num_lanes)
              )
          for j in range(streams):
            s_res_rows_by_j[j].append(
                jnp.concatenate(s_res_sub_by_j[j], axis=1)
            )

  # Stage B (MXU num_lanes -> 1 systolic contraction):
  # dh_post = S_tilde_post @ U_post
  # dH_res = sum_j (S_tilde_res^(j) @ U_res^(j))
  with jax.named_scope("mxu_systolic_gate_reductions"):

    def _mxu_reduce(rows: list[jax.Array], u_mat: jax.Array) -> jax.Array:
      stacked = rows[0] if num_subtiles == 1 else jnp.concatenate(rows, axis=0)
      return jnp.dot(stacked, u_mat, preferred_element_type=jnp.float32)

    with jax.named_scope("mxu_d_h_post"):
      d_h_post_ref[...] = _mxu_reduce(s_post_rows, u_post_bf16)
    with jax.named_scope("mxu_d_residual"):
      d_res_all = _mxu_reduce(s_res_rows_by_j[0], u_res_mats[0])
      for j in range(1, streams):
        d_res_all = d_res_all + _mxu_reduce(s_res_rows_by_j[j], u_res_mats[j])
      d_residual_ref[...] = d_res_all.astype(d_residual_ref.dtype)


def _post_apply_bwd_2d_mxu_body(
    streams: int,
    embedding: int,
    sub_tokens: int,
    num_subtiles: int,
    x_ref,
    layer_output_ref,
    h_post_ref,
    residual_ref,
    d_output_ref,
    d_x_ref,
    d_layer_output_ref,
    d_h_post_ref,
    d_residual_ref,
) -> None:
  """Pallas 1D-grid 2D+MXU pipeline body for post-branch backward pass."""

  def _dy_sub_from_ref(s_slice: slice) -> list[jax.Array]:
    return _read_streams_2d(d_output_ref, s_slice, streams, embedding)

  _post_bwd_2d_mxu_core(
      streams,
      embedding,
      sub_tokens,
      num_subtiles,
      x_ref,
      layer_output_ref,
      h_post_ref,
      residual_ref,
      d_x_ref,
      d_layer_output_ref,
      d_h_post_ref,
      d_residual_ref,
      _dy_sub_from_ref,
  )


def _zero_pre_bwd_accumulators(d_phi_ref, d_ab_ref) -> None:
  """Zeros out chunk accumulators at step 0."""
  d_phi_ref[...] = jnp.zeros_like(d_phi_ref)
  d_ab_ref[...] = jnp.zeros_like(d_ab_ref)


def _pre_bwd_core(
    streams: int,
    embedding: int,
    rms_epsilon: float,
    block_tokens: int,
    step: jax.Array,
    x_ref,
    phi_ref,
    gate_ab_ref,
    permutations_ref,
    d_layer_input_ref,
    d_h_post_ref,
    d_residual_ref,
    d_phi_ref,
    d_ab_ref,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Shared pre-branch backward core accumulating parameter gradients."""
  # Differentiates the pre-branch forward equations (with folded weight matrix
  # Phi = diag(gamma) [W_pre, W_post, W_res] of shape (F, C) where F = S * D
  # and C = 2S + P = 32):
  #   mu_t = (1/F) sum_f X[t, f]^2, r_t = (mu_t + eps_rms)^(-1/2)
  #   R_t = X_t @ Phi, Z_hat_t = r_t * R_t, Z_t = alpha * Z_hat_t + b
  #   h_pre = sigmoid(Z[:, :S]) + eps_pre, h_post = 2 * sigmoid(Z[:, S:2S])
  #   w_res = softmax(Z[:, 2S:]), H_res = w_res @ P
  #   L_in[t, e] = sum_i h_pre[t, i] * X[t, i, e]
  with jax.named_scope("init_chunk_accumulators"):
    pl.when(step == 0)(
        functools.partial(_zero_pre_bwd_accumulators, d_phi_ref, d_ab_ref)
    )

  with jax.named_scope("load_params_and_inputs"):
    flattened_size = streams * embedding
    phi_value = phi_ref[...]
    phi_cols = phi_value.shape[1]
    num_permutations = phi_cols - 2 * streams
    gate_ab_value = gate_ab_ref[...]
    permutations_value = permutations_ref[...]
    d_layer_input_f32 = d_layer_input_ref[...].astype(jnp.float32)
    alpha_row = gate_ab_value[0:1, :]
    bias_row = gate_ab_value[1:2, :]
    x_value = x_ref[...]

  # Step 1: Overlapped MXU projection R = X @ Phi, pre-gate cotangent
  # dh_pre[t, i] = sum_e X[t, i, e] * dL_in[t, e], and sum_f X[t, f]^2.
  with jax.named_scope("mxu_proj_and_vpu_dot_sqr_reduction"):
    with jax.named_scope("mxu_gate_proj"):
      raw_projected = jnp.dot(
          x_value,
          phi_value,
          preferred_element_type=jnp.float32,
      )
    with jax.named_scope("vpu_dh_pre_and_sum_squares"):
      d_h_pre_parts = []
      mean_square_sum = jnp.zeros((block_tokens, 1), dtype=jnp.float32)
      for i in range(streams):
        feature_slice = slice(i * embedding, (i + 1) * embedding)
        x_stream_f32 = x_value[:, feature_slice].astype(jnp.float32)
        d_h_pre_parts.append(
            jnp.sum(x_stream_f32 * d_layer_input_f32, axis=-1, keepdims=True)
        )
        mean_square_sum = mean_square_sum + jnp.sum(
            jnp.square(x_stream_f32), axis=-1, keepdims=True
        )
      d_h_pre = jnp.concatenate(d_h_pre_parts, axis=-1)

  # Step 2: Recompute r_t = (mu_t + eps_rms)^(-1/2), Z_hat_t = r_t * R_t,
  # and Z_t = alpha * Z_hat_t + b.
  with jax.named_scope("remat_rms_norm_and_logits"):
    mean_square = mean_square_sum * (1.0 / flattened_size)
    inv_rms = jax.lax.rsqrt(mean_square + rms_epsilon)
    projected = raw_projected * inv_rms
    logits = projected * alpha_row + bias_row

  # Step 3: 32-lane aligned logit gradients dZ:
  # - Sigmoid gates (0..2S-1): dZ_gates = [dh_pre, 2 * dh_post] * g * (1 - g)
  #   where g = sigmoid(Z[:, :2S]).
  with jax.named_scope("sigmoid_gates_bwd"):
    gates = jax.nn.sigmoid(logits[:, : 2 * streams])
    d_logits_gates = (
        jnp.concatenate((d_h_pre, 2.0 * d_h_post_ref[...]), axis=-1)
        * gates
        * (1.0 - gates)
    )

  # - Softmax weights (2S..31): mask lanes 0..2S-1 with -inf to keep the 24
  #   permutation lanes at native indices 2S..31 (avoiding cross-lane vector
  #   rotations), then compute dw_res = dH_res @ P^T and
  #   dZ_res = w_res * (dw_res - sum_q dw_res[t, q] * w_res[t, q]).
  with jax.named_scope("softmax_bvn_residual_bwd"):
    col_idx = jax.lax.broadcasted_iota(jnp.int32, logits.shape, dimension=1)
    res_logits = jnp.where(col_idx >= 2 * streams, logits, -jnp.inf)
    weights_res = jax.nn.softmax(res_logits, axis=-1)
    d_weights_res = jnp.dot(
        d_residual_ref[...].astype(jnp.float32),
        permutations_value.T,
        preferred_element_type=jnp.float32,
    )
    d_logits_res = weights_res * (
        d_weights_res
        - jnp.sum(d_weights_res * weights_res, axis=-1, keepdims=True)
    )

  # Step 4: Accumulate bias db += sum_t dZ_t and alpha
  # d_alpha += sum_t dZ_t * Z_hat_t (kept at leading dim num_sublanes for VREG
  # sublane alignment).
  with jax.named_scope("accum_d_alpha_and_d_bias"):
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    d_logits = (
        jnp.pad(d_logits_gates, ((0, 0), (0, num_permutations))) + d_logits_res
    )
    d_ab_tokens = jnp.concatenate((d_logits, d_logits * projected), axis=-1)
    d_ab_ref[...] = d_ab_ref[...] + jnp.sum(
        d_ab_tokens.reshape(
            block_tokens // num_sublanes, num_sublanes, 2 * phi_cols
        ),
        axis=0,
    )

  # Step 5: Folded weight & RMSNorm scale gradients:
  # dZ_hat = alpha * dZ, dR = r_t * dZ_hat, dPhi += X^T @ dR,
  # scale_x = -(r_t^2 / F) * sum_c dZ_hat[t, c] * Z_hat[t, c].
  with jax.named_scope("accum_d_phi_and_scale_x"):
    d_projected = d_logits * alpha_row
    d_raw_projected = d_projected * inv_rms
    with jax.named_scope("mxu_accum_d_phi"):
      d_phi_ref[...] = d_phi_ref[...] + jnp.dot(
          x_value.T,
          d_raw_projected,
          preferred_element_type=jnp.float32,
      )
    with jax.named_scope("vpu_rmsnorm_scale_x"):
      scale_x = (
          -jnp.sum(d_projected * projected, axis=-1, keepdims=True)
          * (inv_rms * inv_rms)
          * (1.0 / flattened_size)
      )
  return phi_value, d_raw_projected, scale_x


def _pre_bwd_dx_streams(
    streams: int,
    embedding: int,
    phi_value: jax.Array,
    d_raw_projected: jax.Array,
    scale_x: jax.Array,
    x_ref,
    h_pre_ref,
    d_layer_input_ref,
    d_x_acc_ref,
    out_dtype: jax.typing.DTypeLike,
) -> list[jax.Array]:
  """Computes per-stream `dX` tiles of shape `(block_tokens, embedding)`."""
  h_pre_value = h_pre_ref[...]
  d_layer_input_f32 = d_layer_input_ref[...].astype(jnp.float32)
  dx_streams = []
  for i in range(streams):
    f_slice = slice(i * embedding, (i + 1) * embedding)
    with jax.named_scope("mxu_dx_projected_stream"):
      dx_proj_i = jnp.dot(
          d_raw_projected,
          phi_value[f_slice, :].T,
          preferred_element_type=jnp.float32,
      )
    dx_streams.append(
        (
            dx_proj_i
            + d_x_acc_ref[:, f_slice]
            + h_pre_value[:, i : i + 1] * d_layer_input_f32
            + x_ref[:, f_slice] * scale_x
        ).astype(out_dtype)
    )
  return dx_streams


def _pre_bwd_fused_body(
    streams: int,
    embedding: int,
    rms_epsilon: float,
    block_tokens: int,
    step: pltpu.PipelineStep,
    x_ref,
    phi_ref,
    gate_ab_ref,
    permutations_ref,
    h_pre_ref,
    d_layer_input_ref,
    d_h_post_ref,
    d_residual_ref,
    d_x_acc_ref,
    d_x_ref,
    d_phi_ref,
    d_ab_ref,
) -> None:
  """Pallas pipeline body for fused 2D parallel-chunk pre-branch backward.

  Combines the 4 input gradient terms per stream `i`:
    `dX[t, i, e] = dX_proj[t, i, e] + dX_acc[t, i, e]`
    `            + h_pre[t, i] * dL_in[t, e] + scale_x[t] * X[t, i, e]`
  """
  with jax.named_scope("pre_bwd_core"):
    phi_value, d_raw_projected, scale_x = _pre_bwd_core(
        streams,
        embedding,
        rms_epsilon,
        block_tokens,
        step.index[1],
        x_ref,
        phi_ref,
        gate_ab_ref,
        permutations_ref,
        d_layer_input_ref,
        d_h_post_ref,
        d_residual_ref,
        d_phi_ref,
        d_ab_ref,
    )
  with jax.named_scope("combine_dx_streams"):
    dx_streams = _pre_bwd_dx_streams(
        streams,
        embedding,
        phi_value,
        d_raw_projected,
        scale_x,
        x_ref,
        h_pre_ref,
        d_layer_input_ref,
        d_x_acc_ref,
        d_x_ref.dtype,
    )
    for i, dx_i in enumerate(dx_streams):
      d_x_ref[:, i * embedding : (i + 1) * embedding] = dx_i


def _chunk_token_index_map(
    blocks_per_chunk: int, num_trailing_dims: int, chunk, step
):
  """Index map for 2D (chunk, step) token tiling."""
  return (chunk * blocks_per_chunk + step,) + (0,) * num_trailing_dims


def _zero_2d_index_map(chunk, step):
  """Index map returning (0, 0) across all (chunk, step) grid points."""
  del chunk, step
  return (0, 0)


def _chunk_accum_index_map(chunk, step):
  """Index map selecting the per-chunk accumulator slice (chunk, 0, 0)."""
  del step
  return (chunk, 0, 0)


def _chunk_token_spec(
    shape: tuple[int, ...], block_tokens: int, blocks_per_chunk: int
) -> pl.BlockSpec:
  """Builds a 2D chunk-step BlockSpec along the leading token axis."""
  return pl.BlockSpec(
      (block_tokens,) + shape[1:],
      functools.partial(
          _chunk_token_index_map, blocks_per_chunk, len(shape) - 1
      ),
  )


def _prepare_bwd_params(
    weights: common.MhcWeights,
    permutations: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Packs `gate_ab` and transposed `permutations_padded`."""
  with jax.named_scope("prepare_bwd_params"):
    num_permutations, streams, _ = permutations.shape
    with jax.named_scope("pack_gate_alpha_bias"):
      alpha_vector = jnp.concatenate(
          (
              jnp.broadcast_to(weights.alpha_pre, (streams,)),
              jnp.broadcast_to(weights.alpha_post, (streams,)),
              jnp.broadcast_to(weights.alpha_res, (num_permutations,)),
          ),
          axis=0,
      )
      bias_vector = jnp.concatenate(
          (weights.b_pre, weights.b_post, weights.b_res),
          axis=0,
      )
      gate_ab = jnp.stack(
          (alpha_vector, bias_vector), axis=0, dtype=jnp.float32
      )
    with jax.named_scope("pad_permutations_32_lanes"):
      permutations_padded = jnp.pad(
          permutations.astype(jnp.float32)
          .swapaxes(-1, -2)
          .reshape(num_permutations, streams * streams),
          ((2 * streams, 0), (0, 0)),
      )
    return gate_ab, permutations_padded


def _finalize_weight_grads(
    d_phi_chunks: jax.Array,
    d_ab_chunks: jax.Array,
    weights: common.MhcWeights,
    streams: int,
) -> common.MhcWeights:
  """Reduces chunk accumulators and backpropagates `d_phi` into `MhcWeights`."""
  with jax.named_scope("finalize_weight_grads"):
    with jax.named_scope("reduce_chunk_accumulators"):
      d_phi = jnp.sum(d_phi_chunks, axis=0)
      d_ab_sum = jnp.sum(d_ab_chunks, axis=(0, 1))
      phi_cols = d_phi.shape[-1]
      d_bias = d_ab_sum[:phi_cols]
      d_alpha = d_ab_sum[phi_cols:]
    with jax.named_scope("unfold_norm_scale_vjp"):
      _, phi_vjp = jax.vjp(
          common.fold_norm_scale,
          weights.norm_scale,
          weights.w_pre,
          weights.w_post,
          weights.w_res,
      )
      d_norm_scale, d_w_pre, d_w_post, d_w_res = phi_vjp(d_phi)
    with jax.named_scope("pack_weight_grads"):
      grads = common.MhcWeights(
          norm_scale=d_norm_scale,
          w_pre=d_w_pre,
          b_pre=d_bias[:streams],
          alpha_pre=jnp.sum(d_alpha[:streams], keepdims=True),
          w_post=d_w_post,
          b_post=d_bias[streams : 2 * streams],
          alpha_post=jnp.sum(d_alpha[streams : 2 * streams], keepdims=True),
          w_res=d_w_res,
          b_res=d_bias[2 * streams :],
          alpha_res=jnp.sum(d_alpha[2 * streams :], keepdims=True),
      )
      return jax.tree.map(lambda g, w: g.astype(w.dtype), grads, weights)


def _run_bwd_chunk_pipeline(
    scope_name: str,
    body_fn,
    tokens: int,
    block_tokens: int,
    bwd_num_chunks: int,
    phi: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    token_args: tuple[jax.Array, ...],
    token_out_shapes_dtypes: tuple[
        tuple[tuple[int, ...], jax.typing.DTypeLike], ...
    ],
    cost_estimate: pl.CostEstimate,
    *,
    vmem_limit_bytes: int,
    interpret: bool,
) -> tuple[tuple[jax.Array, ...], common.MhcWeights]:
  """Emits and executes a 2D chunk-parallel Pallas backward kernel."""
  common.validate_token_block_size(tokens, block_tokens, name="bwd_block_size")
  num_sublanes = common.get_tpu_info().num_sublanes
  num_blocks = tokens // block_tokens
  num_chunks = math.gcd(num_blocks, bwd_num_chunks)
  blocks_per_chunk = num_blocks // num_chunks
  gate_ab, permutations_padded = _prepare_bwd_params(weights, permutations)
  flattened_size, phi_cols = phi.shape
  streams = permutations.shape[-1]
  param_args = (phi, gate_ab, permutations_padded)
  in_args = (token_args[0], *param_args, *token_args[1:])
  in_specs = (
      _chunk_token_spec(token_args[0].shape, block_tokens, blocks_per_chunk),
      *(pl.BlockSpec(p.shape, _zero_2d_index_map) for p in param_args),
      *(
          _chunk_token_spec(arg.shape, block_tokens, blocks_per_chunk)
          for arg in token_args[1:]
      ),
  )
  out_specs = tuple(
      _chunk_token_spec(shape, block_tokens, blocks_per_chunk)
      for shape, _ in token_out_shapes_dtypes
  ) + (
      pl.BlockSpec((None, flattened_size, phi_cols), _chunk_accum_index_map),
      pl.BlockSpec((None, num_sublanes, 2 * phi_cols), _chunk_accum_index_map),
  )
  out_shape = tuple(
      jax.ShapeDtypeStruct(shape, dtype)
      for shape, dtype in token_out_shapes_dtypes
  ) + (
      jax.ShapeDtypeStruct((num_chunks, flattened_size, phi_cols), jnp.float32),
      jax.ShapeDtypeStruct(
          (num_chunks, num_sublanes, 2 * phi_cols), jnp.float32
      ),
  )
  kernel_main = pltpu.emit_pipeline(
      body_fn,
      grid=(num_chunks, blocks_per_chunk),
      in_specs=in_specs,
      out_specs=out_specs,
      dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
      _explicit_indices=True,
  )
  with common.tpu_mesh_context():
    *token_outs, d_phi_chunks, d_ab_chunks = pl.pallas_call(
        kernel_main,
        name=scope_name,
        out_shape=out_shape,
        in_specs=(common.HBM_SPEC,) * len(in_args),
        out_specs=(common.HBM_SPEC,) * len(out_shape),
        cost_estimate=cost_estimate,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )(*in_args)
  d_weights = _finalize_weight_grads(
      d_phi_chunks, d_ab_chunks, weights, streams
  )
  return tuple(token_outs), d_weights


def _post_and_pre_bwd_fused_body(
    streams: int,
    embedding: int,
    rms_epsilon: float,
    block_tokens: int,
    sub_tokens: int,
    num_subtiles: int,
    step: pltpu.PipelineStep,
    x_next_ref,
    phi_ref,
    gate_ab_ref,
    permutations_ref,
    next_h_pre_ref,
    d_next_layer_input_ref,
    d_next_h_post_ref,
    d_next_residual_ref,
    d_x_next_acc_ref,
    x_ref,
    layer_output_ref,
    h_post_ref,
    residual_ref,
    d_x_ref,
    d_layer_output_ref,
    d_h_post_ref,
    d_residual_ref,
    d_phi_ref,
    d_ab_ref,
) -> None:
  """Pallas pipeline body for fused Stage-2 pre_bwd + Stage-1 post_bwd.

  Since Stage-2 input is Stage-1 output (`X^(2) = Y^(1)`), the upstream
  cotangent for Stage-1 post_bwd (`dY^(1)[t, j, e] = dX^(2)[t, j, e]`) is
  materialized per stream in `bfloat16` before `_post_bwd_2d_mxu_core` so all
  Stage-2 `float32` temporaries end their live ranges early and avoid a
  full-tensor HBM round-trip.
  """
  with jax.named_scope("stage2_pre_bwd_core"):
    phi_value, d_raw_projected, scale_x_next = _pre_bwd_core(
        streams,
        embedding,
        rms_epsilon,
        block_tokens,
        step.index[1],
        x_next_ref,
        phi_ref,
        gate_ab_ref,
        permutations_ref,
        d_next_layer_input_ref,
        d_next_h_post_ref,
        d_next_residual_ref,
        d_phi_ref,
        d_ab_ref,
    )
  with jax.named_scope("materialize_stage1_dy_streams"):
    dy_streams = _pre_bwd_dx_streams(
        streams,
        embedding,
        phi_value,
        d_raw_projected,
        scale_x_next,
        x_next_ref,
        next_h_pre_ref,
        d_next_layer_input_ref,
        d_x_next_acc_ref,
        x_next_ref.dtype,
    )

  def _dy_sub_from_pre_bwd(s_slice: slice) -> list[jax.Array]:
    return [dy_j[s_slice, :].astype(jnp.float32) for dy_j in dy_streams]

  with jax.named_scope("stage1_post_bwd_core"):
    _post_bwd_2d_mxu_core(
        streams,
        embedding,
        sub_tokens,
        num_subtiles,
        x_ref,
        layer_output_ref,
        h_post_ref,
        residual_ref,
        d_x_ref,
        d_layer_output_ref,
        d_h_post_ref,
        d_residual_ref,
        _dy_sub_from_pre_bwd,
    )


def pre_bwd(
    config: Any,
    residuals: tuple[
        tuple[jax.Array, jax.Array],
        tuple[jax.Array, common.MhcWeights, jax.Array],
    ],
    cotangents: tuple[jax.Array, common.MHCContext],
    *,
    rms_epsilon: float | None = None,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, common.MhcWeights, None]:
  """Custom-VJP backward rule for `pre`."""
  with jax.named_scope("mhc_pre_bwd"):
    (phi, h_pre), (x, weights, permutations) = residuals
    d_layer_input, d_context = cotangents
    tokens, streams = h_pre.shape
    embedding = d_layer_input.shape[-1]
    config, rms_eps, _, vmem_limit, interpret_flag = (
        common.prepare_kernel_config(
            config,
            tokens,
            streams,
            embedding,
            permutations.shape[0],
            x.dtype,
            is_bwd=True,
            rms_epsilon=rms_epsilon,
            vmem_limit_bytes=vmem_limit_bytes,
            interpret=interpret,
        )
    )
    flattened_size = streams * embedding
    block_tokens = config.bwd_block_size
    x_in = x.reshape(tokens, flattened_size)

    (d_x_out,), d_weights = _run_bwd_chunk_pipeline(
        "mhc_pre_bwd_fused",
        functools.partial(
            _pre_bwd_fused_body, streams, embedding, rms_eps, block_tokens
        ),
        tokens,
        block_tokens,
        config.bwd_num_chunks,
        phi,
        weights,
        permutations,
        token_args=(
            x_in,
            h_pre,
            d_layer_input.reshape(tokens, embedding),
            d_context.h_post.reshape(tokens, streams),
            d_context.residual.reshape(tokens, streams * streams),
            d_context.x.reshape(tokens, flattened_size),
        ),
        token_out_shapes_dtypes=((x_in.shape, x.dtype),),
        cost_estimate=common.pre_cost(
            tokens, streams, embedding, permutations.shape[0], is_bwd=True
        ),
        vmem_limit_bytes=vmem_limit,
        interpret=interpret_flag,
    )
    return d_x_out.reshape(x.shape), d_weights, None


def post_bwd(
    config: Any,
    saved: tuple[jax.Array, common.MHCContext],
    d_output: jax.Array,
    *,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, common.MHCContext]:
  """Custom-VJP backward rule for `post`."""
  layer_output, context = saved
  x, h_post, residual = context.x, context.h_post, context.residual
  embedding = layer_output.shape[-1]
  tokens = layer_output.size // embedding
  streams = h_post.shape[-1]
  config, _, _, vmem_limit, interpret_flag = common.prepare_kernel_config(
      config,
      tokens,
      streams,
      embedding,
      24,
      x.dtype,
      is_bwd=True,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  block_tokens = config.bwd_block_size
  common.validate_token_block_size(tokens, block_tokens, name="bwd_block_size")
  sub_tokens = math.gcd(block_tokens, config.bwd_sub_tokens)
  in_args = (
      x.reshape(tokens, streams * embedding),
      layer_output.reshape(tokens, embedding),
      h_post.reshape(tokens, streams),
      residual.reshape(tokens, streams * streams),
      d_output.reshape(tokens, streams * embedding),
  )
  in_specs = tuple(
      common.token_block_spec(arg.shape, block_tokens) for arg in in_args
  )
  out_args = in_args[:4]
  kernel_main = pltpu.emit_pipeline(
      lambda unused_step, *refs: _post_apply_bwd_2d_mxu_body(
          streams,
          embedding,
          sub_tokens,
          block_tokens // sub_tokens,
          *refs,
      ),
      grid=(tokens // block_tokens,),
      in_specs=in_specs,
      out_specs=in_specs[:4],
      dimension_semantics=(pltpu.PARALLEL,),
      _explicit_indices=True,
  )
  with common.tpu_mesh_context():
    d_x, d_layer_output, d_h_post, d_residual = pl.pallas_call(
        kernel_main,
        name="mhc_post_apply_bwd",
        out_shape=tuple(
            jax.ShapeDtypeStruct(arg.shape, arg.dtype) for arg in out_args
        ),
        in_specs=(common.HBM_SPEC,) * len(in_args),
        out_specs=(common.HBM_SPEC,) * len(out_args),
        cost_estimate=common.post_cost(tokens, streams, embedding, is_bwd=True),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit,
        ),
        interpret=interpret_flag,
    )(*in_args)
  return (
      d_layer_output.reshape(layer_output.shape),
      common.MHCContext(
          x=d_x.reshape(x.shape),
          h_post=d_h_post.reshape(h_post.shape),
          residual=d_residual.reshape(residual.shape),
          implementation=context.implementation,
      ),
  )


def post_and_pre_bwd(
    config: Any,
    residuals: tuple[
        tuple[jax.Array, jax.Array],
        tuple[jax.Array, common.MHCContext],
        tuple[jax.Array, common.MhcWeights, jax.Array],
    ],
    cotangents: tuple[jax.Array, common.MHCContext],
    *,
    rms_epsilon: float | None = None,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> tuple[
    jax.Array,
    common.MHCContext,
    common.MhcWeights,
    None,
]:
  """Custom-VJP backward rule for `post_and_pre`."""
  with jax.named_scope("mhc_post_and_pre_bwd"):
    (
        (phi, next_h_pre),
        (layer_output, context),
        (x_next, next_weights, permutations),
    ) = residuals
    x, h_post, residual = context.x, context.h_post, context.residual
    embedding = layer_output.shape[-1]
    tokens = layer_output.size // embedding
    streams = h_post.shape[-1]
    config, rms_eps, _, vmem_limit, interpret_flag = (
        common.prepare_kernel_config(
            config,
            tokens,
            streams,
            embedding,
            permutations.shape[0],
            x.dtype,
            is_bwd=True,
            rms_epsilon=rms_epsilon,
            vmem_limit_bytes=vmem_limit_bytes,
            interpret=interpret,
        )
    )
    flattened_size = streams * embedding
    d_next_layer_input, d_next_context = cotangents

    block_tokens = config.bwd_block_size
    sub_tokens = math.gcd(block_tokens, config.bwd_sub_tokens)
    num_subtiles = block_tokens // sub_tokens
    x_in = x.reshape(tokens, flattened_size)
    x_next_in = x_next.reshape(tokens, flattened_size)

    (
        d_x_out,
        d_layer_output_flat,
        d_h_post_out,
        d_residual_out,
    ), d_next_weights = _run_bwd_chunk_pipeline(
        "mhc_post_and_pre_bwd_fused",
        functools.partial(
            _post_and_pre_bwd_fused_body,
            streams,
            embedding,
            rms_eps,
            block_tokens,
            sub_tokens,
            num_subtiles,
        ),
        tokens,
        block_tokens,
        config.bwd_num_chunks,
        phi,
        next_weights,
        permutations,
        token_args=(
            x_next_in,
            next_h_pre,
            d_next_layer_input.reshape(tokens, embedding),
            d_next_context.h_post.reshape(tokens, streams),
            d_next_context.residual.reshape(tokens, streams * streams),
            d_next_context.x.reshape(tokens, flattened_size),
            x_in,
            layer_output.reshape(tokens, embedding),
            h_post.reshape(tokens, streams),
            residual.reshape(tokens, streams * streams),
        ),
        token_out_shapes_dtypes=(
            (x_in.shape, x.dtype),
            ((tokens, embedding), layer_output.dtype),
            ((tokens, streams), h_post.dtype),
            ((tokens, streams * streams), residual.dtype),
        ),
        cost_estimate=common.post_and_pre_cost(
            tokens, streams, embedding, permutations.shape[0], is_bwd=True
        ),
        vmem_limit_bytes=vmem_limit,
        interpret=interpret_flag,
    )
    return (
        d_layer_output_flat.reshape(layer_output.shape),
        common.MHCContext(
            x=d_x_out.reshape(x.shape),
            h_post=d_h_post_out.reshape(h_post.shape),
            residual=d_residual_out.reshape(residual.shape),
            implementation=context.implementation,
        ),
        d_next_weights,
        None,
    )
