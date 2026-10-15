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


def _post_apply_bwd_2d_mxu_body(
    streams: int,
    embedding: int,
    quarter_emb: int,
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
  # Backward pass for Y = H_res @ X + h_post * L:
  #   1. dL[t, e]       = sum_j h_post[t, j] * dY[t, j, e]    (VPU pointwise)
  #   2. dX[t, k, e]    = sum_j H_res[t, j, k] * dY[t, j, e]  (VPU pointwise)
  #   3. dh_post[t, j]  = sum_e L[t, e] * dY[t, j, e]         (VPU 4-fold + MXU)
  #   4. dH_res[t, j,k] = sum_e dY[t, j, e] * X[t, k, e]      (VPU 4-fold + MXU)
  #
  # Block-diagonal 0/1 selector matrices for Stage B (128 -> 1 on 128x128 MXU):
  #   u_post_bf16[128*j : 128*(j+1), j] = 1
  #   u_res_mats[j][128*k : 128*(k+1), 4*j + k] = 1
  row_band_post = (
      jax.lax.broadcasted_iota(jnp.int32, (embedding, streams), 0)
      // quarter_emb
  )
  u_post_bf16 = (
      row_band_post
      == jax.lax.broadcasted_iota(jnp.int32, (embedding, streams), 1)
  ).astype(jnp.bfloat16)

  row_band_res = (
      jax.lax.broadcasted_iota(jnp.int32, (embedding, streams * streams), 0)
      // quarter_emb
  )
  col_id_res = jax.lax.broadcasted_iota(
      jnp.int32, (embedding, streams * streams), 1
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
    dy_sub = [
        d_output_ref[s_slice, j * embedding : (j + 1) * embedding].astype(
            jnp.float32
        )
        for j in range(streams)
    ]
    l_sub = layer_output_ref[s_slice, :].astype(jnp.float32)
    h_post_sub = h_post_ref[s_slice, :].astype(jnp.float32)
    res_sub = residual_ref[s_slice, :].astype(jnp.bfloat16).astype(jnp.float32)

    # 1. dL[t, e] = sum_j h_post[t, j] * dY[t, j, e]
    dl_sub = h_post_sub[:, 0:1] * dy_sub[0]
    for j in range(1, streams):
      dl_sub = dl_sub + h_post_sub[:, j : j + 1] * dy_sub[j]
    d_layer_output_ref[s_slice, :] = dl_sub.astype(d_layer_output_ref.dtype)

    # 2. dX[t, k, e] = sum_j H_res[t, j, k] * dY[t, j, e]
    for k in range(streams):
      dx_k = res_sub[:, k : k + 1] * dy_sub[0]
      for j in range(1, streams):
        dx_k = (
            dx_k + res_sub[:, j * streams + k : j * streams + k + 1] * dy_sub[j]
        )
      d_x_ref[s_slice, k * embedding : (k + 1) * embedding] = dx_k.astype(
          d_x_ref.dtype
      )

    # Stage A (VPU 4x lane-fold, 512 -> 128 lanes):
    #   S[t, r] = P[t, r] + P[t, 128+r] + P[t, 256+r] + P[t, 384+r]
    s_post_sub_parts = []
    for j in range(streams):
      p_post_j = l_sub * dy_sub[j]
      s_post_sub_parts.append(
          (
              p_post_j[:, 0:quarter_emb]
              + p_post_j[:, quarter_emb : 2 * quarter_emb]
              + p_post_j[:, 2 * quarter_emb : 3 * quarter_emb]
              + p_post_j[:, 3 * quarter_emb : 4 * quarter_emb]
          ).astype(jnp.bfloat16)
      )
    s_post_rows.append(jnp.concatenate(s_post_sub_parts, axis=1))

    s_res_sub_by_j = [[] for _ in range(streams)]
    for k in range(streams):
      x_k = x_ref[s_slice, k * embedding : (k + 1) * embedding].astype(
          jnp.float32
      )
      for j in range(streams):
        p_jk = dy_sub[j] * x_k
        s_res_sub_by_j[j].append(
            (
                p_jk[:, 0:quarter_emb]
                + p_jk[:, quarter_emb : 2 * quarter_emb]
                + p_jk[:, 2 * quarter_emb : 3 * quarter_emb]
                + p_jk[:, 3 * quarter_emb : 4 * quarter_emb]
            ).astype(jnp.bfloat16)
        )
    for j in range(streams):
      s_res_rows_by_j[j].append(jnp.concatenate(s_res_sub_by_j[j], axis=1))

  # Stage B (Systolic MXU 128 -> 1 reduction): (256, 512) @ selector_matrix
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

  d_h_post_ref[...] = d_h_post_all.astype(d_h_post_ref.dtype)
  d_residual_ref[...] = d_res_all.astype(d_residual_ref.dtype)


def _zero_post_bwd_reductions(d_h_post_ref, d_residual_ref) -> None:
  """Zeros out reduction accumulators on the first feature block."""
  d_h_post_ref[...] = jnp.zeros_like(d_h_post_ref)
  d_residual_ref[...] = jnp.zeros_like(d_residual_ref)


def _round_d_residual(d_residual_ref) -> None:
  """Rounds accumulated d_residual to bfloat16 on the final feature block."""
  d_residual_ref[...] = (
      d_residual_ref[...].astype(jnp.bfloat16).astype(d_residual_ref.dtype)
  )


def _post_apply_bwd_feature_tiled_body(
    feature_blocks: int,
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
  """Pallas 2D feature-tiled pipeline body for post-branch backward pass."""
  feature_block = pl.program_id(1)
  d_out_f32 = d_output_ref[...].astype(jnp.float32)

  d_x = jnp.einsum(
      "tjk,tjd->tkd",
      residual_ref[...].astype(jnp.bfloat16),
      d_out_f32,
      preferred_element_type=jnp.float32,
  )
  d_layer_output = jnp.sum(h_post_ref[...][:, :, None] * d_out_f32, axis=1)
  d_x_ref[...] = d_x.astype(d_x_ref.dtype)
  d_layer_output_ref[...] = d_layer_output.astype(d_layer_output_ref.dtype)

  d_h_post = jnp.sum(layer_output_ref[...][:, None, :] * d_out_f32, axis=-1)
  d_residual = jnp.einsum(
      "tjd,tkd->tjk",
      d_out_f32,
      x_ref[...],
      preferred_element_type=jnp.float32,
  )

  pl.when(feature_block == 0)(
      functools.partial(_zero_post_bwd_reductions, d_h_post_ref, d_residual_ref)
  )
  d_h_post_ref[...] += d_h_post
  d_residual_ref[...] += d_residual
  pl.when(feature_block == feature_blocks - 1)(
      functools.partial(_round_d_residual, d_residual_ref)
  )


def _post_apply_bwd(
    x: jax.Array,
    layer_output: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
    d_output: jax.Array,
    config: common.MhcKernelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Builds the Pallas pipeline for the post-branch backward pass."""
  tokens, streams = h_post.shape
  embedding = layer_output.shape[1]
  flattened_size = streams * embedding
  feature_block_size = min(embedding, config.bwd_feature_block_size)
  feature_blocks = embedding // feature_block_size
  dims = common.MhcDims(tokens=tokens, streams=streams, embedding=embedding)

  if feature_blocks == 1:
    block_tokens = config.bwd_block_size
    sub_tokens = min(block_tokens, 32)
    num_subtiles = block_tokens // sub_tokens
    quarter_emb = embedding // streams

    x_2d = x.reshape(tokens, flattened_size)
    d_output_2d = d_output.reshape(tokens, flattened_size)
    residual_2d = residual.reshape(tokens, streams * streams)

    spec_x_2d = common.token_block_spec((tokens, flattened_size), block_tokens)
    spec_layer_output = common.token_block_spec(
        (tokens, embedding), block_tokens
    )
    spec_h_post = common.token_block_spec((tokens, streams), block_tokens)
    spec_residual_2d = common.token_block_spec(
        (tokens, streams * streams), block_tokens
    )

    kernel_main = pltpu.emit_pipeline(
        functools.partial(
            _post_apply_bwd_2d_mxu_body,
            streams,
            embedding,
            quarter_emb,
            sub_tokens,
            num_subtiles,
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
    )

    with jax.named_scope("mhc_post_apply_bwd"):
      with common.tpu_mesh_context():
        d_x_2d, d_layer_output, d_h_post, d_residual = pl.pallas_call(
            kernel_main,
            out_shape=(
                jax.ShapeDtypeStruct((tokens, flattened_size), x.dtype),
                jax.ShapeDtypeStruct((tokens, embedding), layer_output.dtype),
                jax.ShapeDtypeStruct((tokens, streams), h_post.dtype),
                jax.ShapeDtypeStruct(
                    (tokens, streams * streams), residual.dtype
                ),
            ),
            in_specs=(common.HBM_SPEC,) * 5,
            out_specs=(common.HBM_SPEC,) * 4,
            cost_estimate=dims.post_apply_bwd_cost(),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=config.vmem_limit_bytes,
            ),
            interpret=config.interpret,
        )(
            x_2d,
            layer_output,
            h_post,
            residual_2d,
            d_output_2d,
        )
    return (
        d_x_2d.reshape(x.shape),
        d_layer_output,
        d_h_post,
        d_residual.reshape(residual.shape),
    )

  x_3d = x.reshape(tokens, streams, embedding)
  d_output_3d = d_output.reshape(tokens, streams, embedding)
  residual_3d = residual.reshape(tokens, streams, streams)

  spec_x = common.feature_tiled_block_spec(
      (tokens, streams, embedding),
      config.bwd_block_size,
      feature_block_size,
      tiled_feature=True,
  )
  spec_layer_output = common.feature_tiled_block_spec(
      (tokens, embedding),
      config.bwd_block_size,
      feature_block_size,
      tiled_feature=True,
  )
  spec_h_post = common.feature_tiled_block_spec(
      (tokens, streams),
      config.bwd_block_size,
      feature_block_size,
      tiled_feature=False,
  )
  spec_residual = common.feature_tiled_block_spec(
      (tokens, streams, streams),
      config.bwd_block_size,
      feature_block_size,
      tiled_feature=False,
  )

  kernel_main = pltpu.emit_pipeline(
      functools.partial(_post_apply_bwd_feature_tiled_body, feature_blocks),
      grid=(tokens // config.bwd_block_size, feature_blocks),
      in_specs=(
          spec_x,
          spec_layer_output,
          spec_h_post,
          spec_residual,
          spec_x,
      ),
      out_specs=(
          spec_x,
          spec_layer_output,
          spec_h_post,
          spec_residual,
      ),
      dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
  )

  with jax.named_scope("mhc_post_apply_bwd"):
    with common.tpu_mesh_context():
      d_x_3d, d_layer_output, d_h_post, d_residual = pl.pallas_call(
          kernel_main,
          out_shape=(
              jax.ShapeDtypeStruct((tokens, streams, embedding), x.dtype),
              jax.ShapeDtypeStruct((tokens, embedding), layer_output.dtype),
              jax.ShapeDtypeStruct((tokens, streams), h_post.dtype),
              jax.ShapeDtypeStruct((tokens, streams, streams), jnp.float32),
          ),
          in_specs=(common.HBM_SPEC,) * 5,
          out_specs=(common.HBM_SPEC,) * 4,
          cost_estimate=dims.post_apply_bwd_cost(),
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=config.vmem_limit_bytes,
          ),
          interpret=config.interpret,
      )(x_3d, layer_output, h_post, residual_3d, d_output_3d)
  return (
      d_x_3d.reshape(x.shape),
      d_layer_output,
      d_h_post,
      d_residual.reshape(residual.shape).astype(residual.dtype),
  )


def _zero_pre_bwd_accumulators(d_phi_ref, d_ab_ref) -> None:
  """Zeros out chunk accumulators at step 0."""
  d_phi_ref[...] = jnp.zeros_like(d_phi_ref)
  d_ab_ref[...] = jnp.zeros_like(d_ab_ref)


def _pre_bwd_fused_body(
    dims: common.MhcDims,
    config: common.MhcKernelConfig,
    block_tokens: int,
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
  """Pallas pipeline body for fused 2D parallel-chunk pre-branch backward."""
  step = pl.program_id(1)
  pl.when(step == 0)(
      functools.partial(_zero_pre_bwd_accumulators, d_phi_ref, d_ab_ref)
  )

  streams = dims.streams
  embedding = dims.embedding
  phi_value = phi_ref[...]
  gate_ab_value = gate_ab_ref[...]
  permutations_value = permutations_ref[...]
  h_pre_value = h_pre_ref[...].astype(jnp.float32)
  d_layer_input_f32 = d_layer_input_ref[...].astype(jnp.float32)
  alpha_row = gate_ab_value[0:1, :]
  bias_row = gate_ab_value[1:2, :]

  # 1. Pre-gate cotangent: dh_pre[t, i] = sum_e x[t, i, e] * d_layer_input[t, e]
  d_h_pre_parts = []
  mean_square_sum = jnp.zeros((block_tokens, 1), dtype=jnp.float32)
  for i in range(streams):
    feature_slice = slice(i * embedding, (i + 1) * embedding)
    x_stream_f32 = x_ref[:, feature_slice].astype(jnp.float32)
    d_h_pre_parts.append(
        jnp.sum(x_stream_f32 * d_layer_input_f32, axis=-1, keepdims=True)
    )
    mean_square_sum = mean_square_sum + jnp.sum(
        jnp.square(x_stream_f32), axis=-1, keepdims=True
    )
  d_h_pre = jnp.concatenate(d_h_pre_parts, axis=-1)

  # 2. Recompute normalized logits: alpha * ((x @ phi) * inv_rms) + bias
  x_value = x_ref[...]
  raw_projected = jnp.dot(
      x_value,
      phi_value,
      preferred_element_type=jnp.float32,
  )
  mean_square = mean_square_sum * (1.0 / dims.flattened_size)
  inv_rms = jax.lax.rsqrt(mean_square + config.rms_epsilon)
  projected = raw_projected * inv_rms
  logits = projected * alpha_row + bias_row

  # 3. Logit gradients through sigmoid and softmax:
  #   dz_pre  = dh_pre * s_pre * (1 - s_pre)
  #   dz_post = dh_post * 2 * s_post * (1 - s_post)
  #   dz_res  = w_res * (d_w_res - sum(d_w_res * w_res))
  sigmoid_pre = jax.nn.sigmoid(logits[:, dims.pre_slice])
  d_logits_pre = d_h_pre * sigmoid_pre * (1.0 - sigmoid_pre)

  sigmoid_post = jax.nn.sigmoid(logits[:, dims.post_slice])
  d_logits_post = d_h_post_ref[...] * (
      2.0 * sigmoid_post * (1.0 - sigmoid_post)
  )

  weights_res = jax.nn.softmax(logits[:, dims.res_slice], axis=-1)
  d_weights_res = jnp.dot(
      d_residual_ref[...].astype(jnp.float32),
      permutations_value.T,
      preferred_element_type=jnp.float32,
  )
  d_logits_res = weights_res * (
      d_weights_res
      - jnp.sum(d_weights_res * weights_res, axis=-1, keepdims=True)
  )

  # 4. Accumulate gate bias (sum_t dz) and alpha (sum_t dz * proj) grads:
  d_logits = jnp.concatenate(
      (d_logits_pre, d_logits_post, d_logits_res), axis=-1
  )
  d_ab_tokens = jnp.concatenate((d_logits, d_logits * projected), axis=-1)
  d_ab_ref[...] = d_ab_ref[...] + jnp.sum(
      d_ab_tokens.reshape(block_tokens // 8, 8, 2 * dims.phi_cols), axis=0
  )

  # 5. Projection weight & RMSNorm input gradients:
  #   d_phi += x^T @ (dz * alpha * inv_rms)
  d_projected = d_logits * alpha_row
  d_raw_projected = d_projected * inv_rms
  d_phi_ref[...] = d_phi_ref[...] + jnp.dot(
      x_value.T,
      d_raw_projected,
      preferred_element_type=jnp.float32,
  )

  #   dX_total = dX_post + h_pre * dL_in + d_raw_proj @ phi^T + x * scale_x
  scale_x = (
      -jnp.sum(d_projected * projected, axis=-1, keepdims=True)
      * (inv_rms * inv_rms)
      * (1.0 / dims.flattened_size)
  )
  d_x_projected = jnp.dot(
      d_raw_projected,
      phi_value.T,
      preferred_element_type=jnp.float32,
  )
  for i in range(streams):
    feature_slice = slice(i * embedding, (i + 1) * embedding)
    d_x_ref[:, feature_slice] = (
        d_x_acc_ref[:, feature_slice].astype(jnp.float32)
        + h_pre_value[:, i : i + 1] * d_layer_input_f32
        + d_x_projected[:, feature_slice]
        + x_ref[:, feature_slice].astype(jnp.float32) * scale_x
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


def _pre_bwd_fused(
    x_2d: jax.Array,
    h_pre: jax.Array,
    d_layer_input: jax.Array,
    coeff_params: common.MhcCoeffParams,
    permutations: jax.Array,
    d_h_post: jax.Array,
    d_residual: jax.Array,
    d_x_acc_2d: jax.Array,
    config: common.MhcKernelConfig,
) -> tuple[jax.Array, common.MhcCoeffParams]:
  """Builds the fused 2D parallel-chunk Pallas pipeline for pre-branch backward."""
  tokens, streams = h_pre.shape
  embedding = d_layer_input.shape[1]
  dims = common.MhcDims(
      tokens=tokens,
      streams=streams,
      embedding=embedding,
      num_permutations=permutations.shape[0],
  )
  block_tokens = config.bwd_block_size
  num_blocks = tokens // block_tokens
  num_chunks = math.gcd(num_blocks, 4)
  blocks_per_chunk = num_blocks // num_chunks

  phi_bf16 = coeff_params.phi.astype(jnp.bfloat16)
  alpha_vector = jnp.concatenate(
      (
          jnp.broadcast_to(
              coeff_params.alpha_pre.astype(jnp.float32), (streams,)
          ),
          jnp.broadcast_to(
              coeff_params.alpha_post.astype(jnp.float32), (streams,)
          ),
          jnp.broadcast_to(
              coeff_params.alpha_res.astype(jnp.float32),
              (dims.num_permutations,),
          ),
      ),
      axis=0,
  )
  bias_vector = jnp.concatenate(
      (
          coeff_params.b_pre.astype(jnp.float32),
          coeff_params.b_post.astype(jnp.float32),
          coeff_params.b_res.astype(jnp.float32),
      ),
      axis=0,
  )
  gate_ab = jnp.stack((alpha_vector, bias_vector), axis=0)
  permutations_f32 = (
      permutations.swapaxes(-1, -2)
      .reshape(dims.num_permutations, streams * streams)
      .astype(jnp.float32)
  )

  spec_x_2d = _chunk_token_spec(
      (tokens, dims.flattened_size), block_tokens, blocks_per_chunk
  )
  spec_phi = pl.BlockSpec(phi_bf16.shape, _zero_2d_index_map)
  spec_gate_ab = pl.BlockSpec(gate_ab.shape, _zero_2d_index_map)
  spec_permutations = pl.BlockSpec(permutations_f32.shape, _zero_2d_index_map)
  spec_h_pre = _chunk_token_spec(h_pre.shape, block_tokens, blocks_per_chunk)
  spec_d_layer_input = _chunk_token_spec(
      d_layer_input.shape, block_tokens, blocks_per_chunk
  )
  spec_d_h_post = _chunk_token_spec(
      d_h_post.shape, block_tokens, blocks_per_chunk
  )
  spec_d_residual = _chunk_token_spec(
      d_residual.shape, block_tokens, blocks_per_chunk
  )
  spec_d_phi = pl.BlockSpec(
      (None, dims.flattened_size, dims.phi_cols),
      _chunk_accum_index_map,
  )
  spec_d_ab = pl.BlockSpec((None, 8, 2 * dims.phi_cols), _chunk_accum_index_map)

  kernel_main = pltpu.emit_pipeline(
      functools.partial(_pre_bwd_fused_body, dims, config, block_tokens),
      grid=(num_chunks, blocks_per_chunk),
      in_specs=(
          spec_x_2d,
          spec_phi,
          spec_gate_ab,
          spec_permutations,
          spec_h_pre,
          spec_d_layer_input,
          spec_d_h_post,
          spec_d_residual,
          spec_x_2d,
      ),
      out_specs=(
          spec_x_2d,
          spec_d_phi,
          spec_d_ab,
      ),
      dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
  )

  with jax.named_scope("mhc_pre_bwd_fused"):
    with common.tpu_mesh_context():
      d_x_2d, d_phi_chunks, d_ab_chunks = pl.pallas_call(
          kernel_main,
          out_shape=(
              jax.ShapeDtypeStruct((tokens, dims.flattened_size), x_2d.dtype),
              jax.ShapeDtypeStruct(
                  (num_chunks, dims.flattened_size, dims.phi_cols),
                  jnp.float32,
              ),
              jax.ShapeDtypeStruct(
                  (num_chunks, 8, 2 * dims.phi_cols), jnp.float32
              ),
          ),
          in_specs=(common.HBM_SPEC,) * 9,
          out_specs=(common.HBM_SPEC,) * 3,
          cost_estimate=dims.pre_bwd_cost(),
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=config.vmem_limit_bytes,
          ),
          interpret=config.interpret,
      )(
          x_2d,
          phi_bf16,
          gate_ab,
          permutations_f32,
          h_pre,
          d_layer_input,
          d_h_post,
          d_residual,
          d_x_acc_2d,
      )

  d_phi = jnp.sum(d_phi_chunks, axis=0)
  d_ab_sum = jnp.sum(d_ab_chunks, axis=(0, 1))
  d_bias_vector = d_ab_sum[: dims.phi_cols]
  d_alpha_vector = d_ab_sum[dims.phi_cols :]
  d_coeff_grads = common.MhcCoeffParams(
      phi=d_phi,
      alpha_pre=jnp.sum(d_alpha_vector[dims.pre_slice], keepdims=True),
      b_pre=d_bias_vector[dims.pre_slice],
      alpha_post=jnp.sum(d_alpha_vector[dims.post_slice], keepdims=True),
      b_post=d_bias_vector[dims.post_slice],
      alpha_res=jnp.sum(d_alpha_vector[dims.res_slice], keepdims=True),
      b_res=d_bias_vector[dims.res_slice],
  )
  return d_x_2d, d_coeff_grads


def pre_bwd(
    residuals: tuple[jax.Array, jax.Array],
    cotangents: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    config: common.MhcKernelConfig,
) -> tuple[jax.Array, common.MhcWeights]:
  """Computes pre-branch gradients with fused pre-apply and coeff backward."""
  phi, h_pre = residuals
  d_layer_input, d_x_acc, d_h_post, d_residual = cotangents
  batch, sequence, embedding = d_layer_input.shape
  tokens = batch * sequence
  streams = h_pre.shape[-1]
  flattened_size = streams * embedding

  x_2d = x.reshape(tokens, flattened_size)
  d_x_acc_2d = d_x_acc.reshape(tokens, flattened_size)
  d_h_post_flat = d_h_post.reshape(tokens, streams)
  d_residual_flat = d_residual.reshape(tokens, streams * streams)
  d_layer_input_flat = d_layer_input.reshape(tokens, embedding)

  coeff_params = common.MhcCoeffParams(
      phi=phi,
      alpha_pre=weights.alpha_pre,
      b_pre=weights.b_pre,
      alpha_post=weights.alpha_post,
      b_post=weights.b_post,
      alpha_res=weights.alpha_res,
      b_res=weights.b_res,
  )
  d_x_2d, d_coeff_grads = _pre_bwd_fused(
      x_2d,
      h_pre,
      d_layer_input_flat,
      coeff_params,
      permutations,
      d_h_post_flat,
      d_residual_flat,
      d_x_acc_2d,
      config=config,
  )
  _, phi_vjp = jax.vjp(
      common.fold_norm_scale,
      weights.norm_scale,
      weights.w_pre,
      weights.w_post,
      weights.w_res,
  )
  d_norm_scale, d_w_pre, d_w_post, d_w_res = phi_vjp(
      d_coeff_grads.phi.astype(phi.dtype)
  )
  d_weights = common.MhcWeights(
      norm_scale=d_norm_scale,
      w_pre=d_w_pre,
      b_pre=d_coeff_grads.b_pre.astype(weights.b_pre.dtype),
      alpha_pre=d_coeff_grads.alpha_pre.astype(weights.alpha_pre.dtype),
      w_post=d_w_post,
      b_post=d_coeff_grads.b_post.astype(weights.b_post.dtype),
      alpha_post=d_coeff_grads.alpha_post.astype(weights.alpha_post.dtype),
      w_res=d_w_res,
      b_res=d_coeff_grads.b_res.astype(weights.b_res.dtype),
      alpha_res=d_coeff_grads.alpha_res.astype(weights.alpha_res.dtype),
  )
  return d_x_2d.reshape(batch, sequence, streams, embedding), d_weights


def post_bwd(
    cotangent: jax.Array,
    layer_output: jax.Array,
    x: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
    config: common.MhcKernelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Computes post-branch gradients."""
  batch, sequence, embedding = layer_output.shape
  tokens = batch * sequence
  streams = h_post.shape[-1]
  flattened_size = streams * embedding
  d_x, d_layer_output, d_h_post, d_residual = _post_apply_bwd(
      x.reshape(tokens, flattened_size),
      layer_output.reshape(tokens, embedding),
      h_post.reshape(tokens, streams),
      residual.reshape(tokens, streams * streams),
      cotangent.reshape(tokens, flattened_size),
      config=config,
  )
  return (
      d_layer_output.reshape(batch, sequence, embedding),
      d_x.reshape(x.shape),
      d_h_post.reshape(h_post.shape),
      d_residual.reshape(residual.shape),
  )


def pre_op_bwd(
    config: common.MhcKernelConfig,
    residuals: tuple[
        tuple[jax.Array, jax.Array],
        tuple[jax.Array, common.MhcWeights, jax.Array],
    ],
    cotangents: tuple[jax.Array, common.KernelContext],
) -> tuple[jax.Array, None, common.MhcWeights]:
  """Custom-VJP backward rule for the low-level pre-branch entry point."""
  saved, (x, weights, permutations) = residuals
  d_layer_input, (d_x, d_h_post, d_residual) = cotangents
  d_x_out, d_weights = pre_bwd(
      saved,
      (d_layer_input, d_x, d_h_post, d_residual),
      x,
      weights,
      permutations,
      config=config,
  )
  return d_x_out, None, d_weights


def post_op_bwd(
    config: common.MhcKernelConfig,
    saved: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    d_output: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Custom-VJP backward rule for the low-level post-branch entry point."""
  layer_output, x, h_post, residual = saved
  d_layer_output, d_x, d_h_post, d_residual = post_bwd(
      d_output,
      layer_output,
      x,
      h_post,
      residual,
      config=config,
  )
  return d_layer_output, d_x, d_h_post, d_residual
