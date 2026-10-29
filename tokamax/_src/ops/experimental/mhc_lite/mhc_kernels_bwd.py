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


def _fold_to_128_lanes(x: jax.Array, embedding: int) -> jax.Array:
  """Folds `(sub_tokens, embedding)` into `(sub_tokens, 128)` in bf16.

  Accumulates `embedding // 128` strided 128-lane bands on the VPU with zero
  cross-lane shuffles:
    `S_tilde[t, c] = sum_{b=0}^{D/128 - 1} x[t, 128*b + c]` for `c in [0, 128)`.
  """
  acc = x[:, 0:128]
  for col in range(128, embedding, 128):
    acc = acc + x[:, col : col + 128]
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
    - Stage A (VPU 128-lane fold, `D -> 128` per stream): folds elementwise
      products across `D // 128` bands into `S_tilde_post` and `S_tilde_res^(j)`
      without cross-lane VPU shuffles.
    - Stage B (MXU `128 -> 1` systolic contraction): contracts folded bands
      with binary selector matrices `U_post` and `U_res^(j)`:
        `dh_post[t, j] = sum_e L[t, e] * dY[t, j, e]`
        `              = (S_tilde_post @ U_post)[t, j]`
        `dH_res[t, j, k] = sum_e dY[t, j, e] * X[t, k, e]`
        `                = sum_j (S_tilde_res^(j) @ U_res^(j))`
  """
  folded_cols = streams * 128
  row_band_post = (
      jax.lax.broadcasted_iota(jnp.int32, (folded_cols, streams), 0) // 128
  )
  u_post_bf16 = (
      row_band_post
      == jax.lax.broadcasted_iota(jnp.int32, (folded_cols, streams), 1)
  ).astype(jnp.bfloat16)

  row_band_res = (
      jax.lax.broadcasted_iota(jnp.int32, (folded_cols, streams * streams), 0)
      // 128
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

  for sub_idx in range(num_subtiles):
    s_slice = slice(sub_idx * sub_tokens, (sub_idx + 1) * sub_tokens)
    dy_sub = dy_sub_fn(s_slice)
    l_sub = layer_output_ref[s_slice, :].astype(jnp.float32)
    h_post_sub = h_post_ref[s_slice, :]
    res_sub = residual_ref[s_slice, :].astype(jnp.float32)

    # Step 1 (VPU): dL[t, e] = sum_j h_post[t, j] * dY[t, j, e]
    dl_sub = h_post_sub[:, 0:1] * dy_sub[0]
    for j in range(1, streams):
      dl_sub = dl_sub + h_post_sub[:, j : j + 1] * dy_sub[j]
    d_layer_output_ref[s_slice, :] = dl_sub.astype(d_layer_output_ref.dtype)

    # Step 2 (VPU): dX[t, k, e] = sum_j H_res[t, j, k] * dY[t, j, e]
    for k in range(streams):
      dx_k = res_sub[:, k : k + 1] * dy_sub[0]
      for j in range(1, streams):
        dx_k = (
            dx_k + res_sub[:, j * streams + k : j * streams + k + 1] * dy_sub[j]
        )
      d_x_ref[s_slice, k * embedding : (k + 1) * embedding] = dx_k.astype(
          d_x_ref.dtype
      )

    # Stage A (VPU 128-lane fold, D -> 128 per stream):
    # S_tilde_post[t, j*128 + c] = sum_b (L * dY_j)[t, 128*b + c]
    # S_tilde_res^(j)[t, k*128 + c] = sum_b (dY_j * X_k)[t, 128*b + c]
    s_post_sub_parts = [
        _fold_to_128_lanes(l_sub * dy_sub[j], embedding) for j in range(streams)
    ]
    s_post_rows.append(jnp.concatenate(s_post_sub_parts, axis=1))

    s_res_sub_by_j = [[] for _ in range(streams)]
    for x_k in _read_streams_2d(x_ref, s_slice, streams, embedding):
      for j in range(streams):
        s_res_sub_by_j[j].append(_fold_to_128_lanes(dy_sub[j] * x_k, embedding))
    for j in range(streams):
      s_res_rows_by_j[j].append(jnp.concatenate(s_res_sub_by_j[j], axis=1))

  # Stage B (MXU 128 -> 1 systolic contraction):
  # dh_post = S_tilde_post @ U_post
  # dH_res = sum_j (S_tilde_res^(j) @ U_res^(j))
  d_h_post_all = jnp.dot(
      jnp.concatenate(s_post_rows, axis=0),
      u_post_bf16,
      preferred_element_type=jnp.float32,
  )
  d_res_all = jnp.dot(
      jnp.concatenate(s_res_rows_by_j[0], axis=0),
      u_res_mats[0],
      preferred_element_type=jnp.float32,
  )
  for j in range(1, streams):
    d_res_all = d_res_all + jnp.dot(
        jnp.concatenate(s_res_rows_by_j[j], axis=0),
        u_res_mats[j],
        preferred_element_type=jnp.float32,
    )

  d_h_post_ref[...] = d_h_post_all
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
    config: common.MhcKernelConfig,
    block_tokens: int,
    step: jax.Array,
    x_ref,
    phi_ref,
    gate_ab_ref,
    permutations_ref,
    h_pre_ref,
    d_layer_input_ref,
    d_h_post_ref,
    d_residual_ref,
    d_phi_ref,
    d_ab_ref,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Shared pre-branch backward core accumulating parameter gradients.

  Differentiates the pre-branch forward equations (with folded weight matrix
  `Phi = diag(gamma) [W_pre, W_post, W_res]` of shape `(F, C)` where `F = S * D`
  and `C = 2S + P = 32`):
    `mu_t = (1/F) sum_f X[t, f]^2`, `r_t = (mu_t + eps_rms)^(-1/2)`
    `R_t = X_t @ Phi`, `Z_hat_t = r_t * R_t`, `Z_t = alpha * Z_hat_t + b`
    `h_pre = sigmoid(Z[:, :S]) + eps_pre`, `h_post = 2 * sigmoid(Z[:, S:2S])`
    `w_res = softmax(Z[:, 2S:])`, `H_res = w_res @ P`
    `L_in[t, e] = sum_i h_pre[t, i] * X[t, i, e]`
  """
  pl.when(step == 0)(
      functools.partial(_zero_pre_bwd_accumulators, d_phi_ref, d_ab_ref)
  )

  flattened_size = streams * embedding
  phi_value = phi_ref[...]
  phi_cols = phi_value.shape[1]
  num_permutations = phi_cols - 2 * streams
  gate_ab_value = gate_ab_ref[...]
  permutations_value = permutations_ref[...]
  h_pre_value = h_pre_ref[...]
  d_layer_input_f32 = d_layer_input_ref[...].astype(jnp.float32)
  alpha_row = gate_ab_value[0:1, :]
  bias_row = gate_ab_value[1:2, :]

  # Step 1: Overlapped MXU projection R = X @ Phi, pre-gate cotangent
  # dh_pre[t, i] = sum_e X[t, i, e] * dL_in[t, e], and sum_f X[t, f]^2.
  x_value = x_ref[...]
  raw_projected = jnp.dot(
      x_value,
      phi_value,
      preferred_element_type=jnp.float32,
  )
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
  mean_square = mean_square_sum * (1.0 / flattened_size)
  inv_rms = jax.lax.rsqrt(mean_square + config.rms_epsilon)
  projected = raw_projected * inv_rms
  logits = projected * alpha_row + bias_row

  # Step 3: 32-lane aligned logit gradients dZ:
  # - Sigmoid gates (0..2S-1): dZ_gates = [dh_pre, 2 * dh_post] * g * (1 - g)
  #   where g = sigmoid(Z[:, :2S]).
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
  # d_alpha += sum_t dZ_t * Z_hat_t (kept at leading dim 8 for VREG sublane
  # alignment).
  d_logits = (
      jnp.pad(d_logits_gates, ((0, 0), (0, num_permutations))) + d_logits_res
  )
  d_ab_tokens = jnp.concatenate((d_logits, d_logits * projected), axis=-1)
  d_ab_ref[...] = d_ab_ref[...] + jnp.sum(
      d_ab_tokens.reshape(block_tokens // 8, 8, 2 * phi_cols), axis=0
  )

  # Step 5: Folded weight & RMSNorm input gradients:
  # dZ_hat = alpha * dZ, dR = r_t * dZ_hat, dPhi += X^T @ dR,
  # scale_x = -(r_t^2 / F) * sum_c dZ_hat[t, c] * Z_hat[t, c],
  # dX_proj = dR @ Phi^T.
  d_projected = d_logits * alpha_row
  d_raw_projected = d_projected * inv_rms
  d_phi_ref[...] = d_phi_ref[...] + jnp.dot(
      x_value.T,
      d_raw_projected,
      preferred_element_type=jnp.float32,
  )

  scale_x = (
      -jnp.sum(d_projected * projected, axis=-1, keepdims=True)
      * (inv_rms * inv_rms)
      * (1.0 / flattened_size)
  )
  d_x_projected = jnp.dot(
      d_raw_projected,
      phi_value.T,
      preferred_element_type=jnp.float32,
  )
  return x_value, h_pre_value, d_layer_input_f32, scale_x, d_x_projected


def _pre_bwd_fused_body(
    streams: int,
    embedding: int,
    config: common.MhcKernelConfig,
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
  x_value, h_pre_value, d_layer_input_f32, scale_x, d_x_projected = (
      _pre_bwd_core(
          streams,
          embedding,
          config,
          block_tokens,
          step.index[1],
          x_ref,
          phi_ref,
          gate_ab_ref,
          permutations_ref,
          h_pre_ref,
          d_layer_input_ref,
          d_h_post_ref,
          d_residual_ref,
          d_phi_ref,
          d_ab_ref,
      )
  )
  # 4-term input gradient sum per stream i:
  # dX[t, i, e] = (
  #     dX_proj[t, i, e] + dX_acc[t, i, e]
  #     + h_pre[t, i] * dL_in[t, e] + scale_x[t] * X[t, i, e]
  # )
  for i in range(streams):
    feature_slice = slice(i * embedding, (i + 1) * embedding)
    d_x_ref[:, feature_slice] = (
        d_x_projected[:, feature_slice]
        + d_x_acc_ref[:, feature_slice]
        + h_pre_value[:, i : i + 1] * d_layer_input_f32
        + x_value[:, feature_slice] * scale_x
    ).astype(d_x_ref.dtype)


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
    gate_ab = jnp.stack((alpha_vector, bias_vector), axis=0, dtype=jnp.float32)
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
    d_phi = jnp.sum(d_phi_chunks, axis=0)
    d_ab_sum = jnp.sum(d_ab_chunks, axis=(0, 1))
    phi_cols = d_phi.shape[-1]
    d_bias = d_ab_sum[:phi_cols]
    d_alpha = d_ab_sum[phi_cols:]
    _, phi_vjp = jax.vjp(
        common.fold_norm_scale,
        weights.norm_scale,
        weights.w_pre,
        weights.w_post,
        weights.w_res,
    )
    d_norm_scale, d_w_pre, d_w_post, d_w_res = phi_vjp(d_phi)
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
    phi: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    token_args: tuple[jax.Array, ...],
    token_out_shapes_dtypes: tuple[tuple[tuple[int, ...], jnp.dtype], ...],
    cost_estimate: pl.CostEstimate,
    config: common.MhcKernelConfig,
) -> tuple[tuple[jax.Array, ...], common.MhcWeights]:
  """Emits and executes a 2D chunk-parallel Pallas backward kernel."""
  with jax.named_scope(scope_name):
    common.validate_token_block_size(
        tokens, block_tokens, name="bwd_block_size"
    )
    num_blocks = tokens // block_tokens
    num_chunks = math.gcd(num_blocks, 4)
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
        pl.BlockSpec((None, 8, 2 * phi_cols), _chunk_accum_index_map),
    )
    out_shape = tuple(
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in token_out_shapes_dtypes
    ) + (
        jax.ShapeDtypeStruct(
            (num_chunks, flattened_size, phi_cols), jnp.float32
        ),
        jax.ShapeDtypeStruct((num_chunks, 8, 2 * phi_cols), jnp.float32),
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
          out_shape=out_shape,
          in_specs=(common.HBM_SPEC,) * len(in_args),
          out_specs=(common.HBM_SPEC,) * len(out_shape),
          cost_estimate=cost_estimate,
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=config.vmem_limit_bytes,
          ),
          interpret=config.interpret,
      )(*in_args)
    d_weights = _finalize_weight_grads(
        d_phi_chunks, d_ab_chunks, weights, streams
    )
    return tuple(token_outs), d_weights


def _post_and_pre_bwd_fused_body(
    streams: int,
    embedding: int,
    config: common.MhcKernelConfig,
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
  materialized on the fly per `sub_tokens` subtile in VMEM from the 4
  `_pre_bwd_core` terms and fed directly into `_post_bwd_2d_mxu_core`,
  avoiding a full-tensor HBM round-trip.
  """
  (
      x_next_value,
      next_h_pre_value,
      _,
      scale_x_next,
      d_x_next_projected,
  ) = _pre_bwd_core(
      streams,
      embedding,
      config,
      block_tokens,
      step.index[1],
      x_next_ref,
      phi_ref,
      gate_ab_ref,
      permutations_ref,
      next_h_pre_ref,
      d_next_layer_input_ref,
      d_next_h_post_ref,
      d_next_residual_ref,
      d_phi_ref,
      d_ab_ref,
  )

  # Materialize dY^(1)[t, j, e] = dX^(2)[t, j, e] on the fly per subtile in
  # VMEM from the 4 _pre_bwd_core terms. Quantizing through x_next_ref.dtype
  # (bf16) via .astype(x_next_ref.dtype).astype(jnp.float32) matches non-fused
  # HBM boundary semantics and halves VMEM spill traffic.
  def _dy_sub_from_pre_bwd(s_slice: slice) -> list[jax.Array]:
    scale_sub = scale_x_next[s_slice, :]
    next_h_pre_sub = next_h_pre_value[s_slice, :]
    d_next_li_sub = d_next_layer_input_ref[s_slice, :].astype(jnp.float32)
    x_next_sub_2d = x_next_value[s_slice, :]
    dy_sub = []
    for j in range(streams):
      f_slice = slice(j * embedding, (j + 1) * embedding)
      dy_j = (
          d_x_next_projected[s_slice, f_slice]
          + d_x_next_acc_ref[s_slice, f_slice]
          + next_h_pre_sub[:, j : j + 1] * d_next_li_sub
          + x_next_sub_2d[:, f_slice] * scale_sub
      ).astype(x_next_ref.dtype)
      dy_sub.append(dy_j.astype(jnp.float32))
    return dy_sub

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
    config: common.MhcKernelConfig,
    residuals: tuple[
        tuple[jax.Array, jax.Array],
        tuple[jax.Array, common.MhcWeights, jax.Array],
    ],
    cotangents: tuple[jax.Array, common.MHCContext],
) -> tuple[jax.Array, common.MhcWeights, None]:
  """Custom-VJP backward rule for `pre`."""
  (phi, h_pre), (x, weights, permutations) = residuals
  d_layer_input, d_context = cotangents
  tokens, streams = h_pre.shape
  embedding = d_layer_input.shape[-1]
  flattened_size = streams * embedding
  block_tokens = config.bwd_block_size
  x_in = x.reshape(tokens, flattened_size)

  (d_x_out,), d_weights = _run_bwd_chunk_pipeline(
      "mhc_pre_bwd_fused",
      functools.partial(
          _pre_bwd_fused_body, streams, embedding, config, block_tokens
      ),
      tokens,
      block_tokens,
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
      config=config,
  )
  return d_x_out.reshape(x.shape), d_weights, None


def post_bwd(
    config: common.MhcKernelConfig,
    saved: tuple[jax.Array, common.MHCContext],
    d_output: jax.Array,
) -> tuple[jax.Array, common.MHCContext]:
  """Custom-VJP backward rule for `post`."""
  with jax.named_scope("mhc_post_apply_bwd"):
    layer_output, context = saved
    x, h_post, residual = context.x, context.h_post, context.residual
    embedding = layer_output.shape[-1]
    tokens = layer_output.size // embedding
    streams = h_post.shape[-1]
    flattened_size = streams * embedding
    block_tokens = config.bwd_block_size
    common.validate_token_block_size(
        tokens, block_tokens, name="bwd_block_size"
    )
    sub_tokens = math.gcd(block_tokens, 32)
    num_subtiles = block_tokens // sub_tokens

    x_in = x.reshape(tokens, flattened_size)
    layer_output_in = layer_output.reshape(tokens, embedding)
    h_post_in = h_post.reshape(tokens, streams)
    residual_in = residual.reshape(tokens, streams * streams)
    d_output_in = d_output.reshape(tokens, flattened_size)

    spec_x_2d = common.token_block_spec((tokens, flattened_size), block_tokens)
    spec_layer_output = common.token_block_spec(
        (tokens, embedding), block_tokens
    )
    spec_h_post = common.token_block_spec((tokens, streams), block_tokens)
    spec_residual_2d = common.token_block_spec(
        (tokens, streams * streams), block_tokens
    )
    kernel_main = pltpu.emit_pipeline(
        lambda unused_step, *refs: _post_apply_bwd_2d_mxu_body(
            streams,
            embedding,
            sub_tokens,
            num_subtiles,
            *refs,
        ),
        grid=(tokens // block_tokens,),
        in_specs=(
            spec_x_2d,
            spec_layer_output,
            spec_h_post,
            spec_residual_2d,
            spec_x_2d,
        ),
        out_specs=(
            spec_x_2d,
            spec_layer_output,
            spec_h_post,
            spec_residual_2d,
        ),
        dimension_semantics=(pltpu.PARALLEL,),
        _explicit_indices=True,
    )
    with common.tpu_mesh_context():
      d_x, d_layer_output, d_h_post, d_residual = pl.pallas_call(
          kernel_main,
          out_shape=(
              jax.ShapeDtypeStruct((tokens, flattened_size), x.dtype),
              jax.ShapeDtypeStruct((tokens, embedding), layer_output.dtype),
              jax.ShapeDtypeStruct((tokens, streams), h_post.dtype),
              jax.ShapeDtypeStruct((tokens, streams * streams), residual.dtype),
          ),
          in_specs=(common.HBM_SPEC,) * 5,
          out_specs=(common.HBM_SPEC,) * 4,
          cost_estimate=common.post_cost(
              tokens, streams, embedding, is_bwd=True
          ),
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=config.vmem_limit_bytes,
          ),
          interpret=config.interpret,
      )(x_in, layer_output_in, h_post_in, residual_in, d_output_in)
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
    config: common.MhcKernelConfig,
    residuals: tuple[
        tuple[jax.Array, jax.Array],
        tuple[jax.Array, common.MHCContext],
        tuple[jax.Array, common.MhcWeights, jax.Array],
    ],
    cotangents: tuple[jax.Array, common.MHCContext],
) -> tuple[
    jax.Array,
    common.MHCContext,
    common.MhcWeights,
    None,
]:
  """Custom-VJP backward rule for `post_and_pre`."""
  (
      (phi, next_h_pre),
      (layer_output, context),
      (x_next, next_weights, permutations),
  ) = residuals
  x, h_post, residual = context.x, context.h_post, context.residual
  embedding = layer_output.shape[-1]
  tokens = layer_output.size // embedding
  streams = h_post.shape[-1]
  flattened_size = streams * embedding
  d_next_layer_input, d_next_context = cotangents

  block_tokens = config.bwd_block_size
  sub_tokens = math.gcd(block_tokens, 32)
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
          config,
          block_tokens,
          sub_tokens,
          num_subtiles,
      ),
      tokens,
      block_tokens,
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
      config=config,
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
