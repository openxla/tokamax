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

import math
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.mhc_lite import common


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
    x_2d = x.reshape(tokens, flattened_size)
    d_output_2d = d_output.reshape(tokens, flattened_size)

    def pipeline_body_1d(
        x_ref,
        layer_output_ref,
        h_post_ref,
        residual_ref,
        d_output_ref,
        d_x_ref,
        d_layer_output_ref,
        d_h_post_ref,
        d_residual_ref,
    ):
      d_output_val = d_output_ref[...]
      d_out_streams = [
          d_output_val[:, s * embedding : (s + 1) * embedding].astype(
              jnp.float32
          )
          for s in range(streams)
      ]
      h_post_f32 = h_post_ref[...].astype(jnp.float32)
      d_layer_output = h_post_f32[:, 0:1] * d_out_streams[0]
      for s in range(1, streams):
        d_layer_output = (
            d_layer_output + h_post_f32[:, s : s + 1] * d_out_streams[s]
        )
      d_layer_output_ref[...] = d_layer_output.astype(d_layer_output_ref.dtype)

      res_2d = (
          residual_ref[...]
          .astype(jnp.bfloat16)
          .astype(jnp.float32)
          .reshape(-1, streams * streams)
      )
      for k in range(streams):
        sl = slice(k * embedding, (k + 1) * embedding)
        d_x_k = res_2d[:, k * streams : k * streams + 1] * d_out_streams[0]
        for j in range(1, streams):
          d_x_k = (
              d_x_k
              + res_2d[:, k * streams + j : k * streams + j + 1]
              * d_out_streams[j]
          )
        d_x_ref[:, sl] = d_x_k.astype(d_x_ref.dtype)

      layer_output_f32 = layer_output_ref[...].astype(jnp.float32)
      d_h_post_ref[...] = jnp.concatenate(
          [
              jnp.sum(
                  layer_output_f32 * d_out_streams[s], axis=-1, keepdims=True
              )
              for s in range(streams)
          ],
          axis=-1,
      ).astype(d_h_post_ref.dtype)

      d_res_streams = []
      for k in range(streams):
        sl = slice(k * embedding, (k + 1) * embedding)
        x_k = x_ref[:, sl].astype(jnp.float32)
        for j in range(streams):
          d_res_streams.append(
              jnp.sum(x_k * d_out_streams[j], axis=-1, keepdims=True)
          )
      d_residual_ref[...] = (
          jnp.concatenate(d_res_streams, axis=-1)
          .reshape(-1, streams, streams)
          .astype(jnp.bfloat16)
          .astype(d_residual_ref.dtype)
      )

    spec_x_2d = common.token_block_spec(
        (tokens, flattened_size), config.bwd_block_size
    )
    spec_layer_output = common.token_block_spec(
        (tokens, embedding), config.bwd_block_size
    )
    spec_h_post = common.token_block_spec(
        (tokens, streams), config.bwd_block_size
    )
    spec_residual = common.token_block_spec(
        (tokens, streams, streams), config.bwd_block_size
    )

    kernel_main = pltpu.emit_pipeline(
        pipeline_body_1d,
        grid=(tokens // config.bwd_block_size,),
        in_specs=(
            spec_x_2d,
            spec_layer_output,
            spec_h_post,
            spec_residual,
            spec_x_2d,
        ),
        out_specs=(
            spec_x_2d,
            spec_layer_output,
            spec_h_post,
            spec_residual,
        ),
        dimension_semantics=common.PARALLEL_DIMENSION_SEMANTICS,
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
                    (tokens, streams, streams), residual.dtype
                ),
            ),
            in_specs=common.hbm_specs(5),
            out_specs=common.hbm_specs(4),
            cost_estimate=dims.post_apply_bwd_cost(),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=config.vmem_limit_bytes,
            ),
            interpret=config.interpret,
        )(x_2d, layer_output, h_post, residual, d_output_2d)
    return d_x_2d.reshape(x.shape), d_layer_output, d_h_post, d_residual

  x_3d = x.reshape(tokens, streams, embedding)
  d_output_3d = d_output.reshape(tokens, streams, embedding)

  def pipeline_body_2d(
      x_ref,
      layer_output_ref,
      h_post_ref,
      residual_ref,
      d_output_ref,
      d_x_ref,
      d_layer_output_ref,
      d_h_post_ref,
      d_residual_ref,
  ):
    feature_block = pl.program_id(1)

    d_x, d_layer_output = common.post_apply_bwd_pointwise(
        d_output_ref[...],
        h_post_ref[...],
        residual_ref[...],
    )
    d_x_ref[...] = d_x.astype(d_x_ref.dtype)
    d_layer_output_ref[...] = d_layer_output.astype(d_layer_output_ref.dtype)

    d_h_post, d_residual = common.post_apply_bwd_reductions(
        d_output_ref[...],
        layer_output_ref[...],
        x_ref[...],
    )

    @pl.when(feature_block == 0)
    def initialize_reductions():
      d_h_post_ref[...] = jnp.zeros_like(d_h_post_ref)
      d_residual_ref[...] = jnp.zeros_like(d_residual_ref)

    d_h_post_ref[...] += d_h_post
    d_residual_ref[...] += d_residual

    @pl.when(feature_block == feature_blocks - 1)
    def round_d_residual():
      d_residual_ref[...] = (
          d_residual_ref[...].astype(jnp.bfloat16).astype(d_residual_ref.dtype)
      )

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
      pipeline_body_2d,
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
      dimension_semantics=common.POST_BWD_DIMENSION_SEMANTICS,
  )

  with jax.named_scope("mhc_post_apply_bwd"):
    with common.tpu_mesh_context():
      d_x_3d, d_layer_output, d_h_post, d_residual = pl.pallas_call(
          kernel_main,
          out_shape=(
              jax.ShapeDtypeStruct((tokens, streams, embedding), x.dtype),
              jax.ShapeDtypeStruct((tokens, embedding), layer_output.dtype),
              jax.ShapeDtypeStruct((tokens, streams), h_post.dtype),
              jax.ShapeDtypeStruct((tokens, streams, streams), residual.dtype),
          ),
          in_specs=common.hbm_specs(5),
          out_specs=common.hbm_specs(4),
          cost_estimate=dims.post_apply_bwd_cost(),
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=config.vmem_limit_bytes,
          ),
          interpret=config.interpret,
      )(x_3d, layer_output, h_post, residual, d_output_3d)
  return d_x_3d.reshape(x.shape), d_layer_output, d_h_post, d_residual


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
) -> tuple[jax.Array, common.MhcCoeffGradients]:
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
  alpha_vec = jnp.concatenate(
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
  b_vec = jnp.concatenate(
      (
          coeff_params.b_pre.astype(jnp.float32),
          coeff_params.b_post.astype(jnp.float32),
          coeff_params.b_res.astype(jnp.float32),
      ),
      axis=0,
  )
  gate_ab = jnp.stack((alpha_vec, b_vec), axis=0)
  perms_f32 = permutations.reshape(
      dims.num_permutations, streams * streams
  ).astype(jnp.float32)

  def pipeline_body(
      x_ref,
      phi_ref,
      gate_ab_ref,
      perms_ref,
      h_pre_ref,
      d_layer_input_ref,
      d_h_post_ref,
      d_res_ref,
      d_x_acc_ref,
      d_x_ref,
      d_phi_ref,
      d_ab_ref,
  ):
    step = pl.program_id(1)

    @pl.when(step == 0)
    def _zero_acc():
      d_phi_ref[...] = jnp.zeros_like(d_phi_ref)
      d_ab_ref[...] = jnp.zeros_like(d_ab_ref)

    phi_val = phi_ref[...]
    gate_ab_val = gate_ab_ref[...]
    perms_val = perms_ref[...]
    h_pre_val = h_pre_ref[...].astype(jnp.float32)
    d_li_f32 = d_layer_input_ref[...].astype(jnp.float32)
    alpha_v = gate_ab_val[0:1, :]
    b_v = gate_ab_val[1:2, :]

    d_h_pre_parts = []
    mean_sq_acc = jnp.zeros((block_tokens, 1), dtype=jnp.float32)
    for s in range(streams):
      sl = slice(s * embedding, (s + 1) * embedding)
      x_s_f32 = x_ref[:, sl].astype(jnp.float32)
      d_h_pre_parts.append(jnp.sum(x_s_f32 * d_li_f32, axis=-1, keepdims=True))
      mean_sq_acc = mean_sq_acc + jnp.sum(
          jnp.square(x_s_f32), axis=-1, keepdims=True
      )
    d_h_pre = jnp.concatenate(d_h_pre_parts, axis=-1)

    x_val = x_ref[...]
    raw_proj = jnp.dot(
        x_val,
        phi_val,
        preferred_element_type=jnp.float32,
    )
    mean_sq = mean_sq_acc * (1.0 / dims.flattened_size)
    rstd = jax.lax.rsqrt(mean_sq + config.rms_epsilon)
    projected = raw_proj * rstd
    z = projected * alpha_v + b_v

    s_pre = jax.nn.sigmoid(z[:, dims.pre_slice])
    d_z_pre = d_h_pre * s_pre * (1.0 - s_pre)

    s_post = jax.nn.sigmoid(z[:, dims.post_slice])
    d_z_post = d_h_post_ref[...] * (2.0 * s_post * (1.0 - s_post))

    w_res = jax.nn.softmax(z[:, dims.res_slice], axis=-1)
    d_w_res = jnp.dot(
        d_res_ref[...].reshape(block_tokens, streams * streams),
        perms_val.T,
        preferred_element_type=jnp.float32,
    )
    d_z_res = w_res * (
        d_w_res - jnp.sum(d_w_res * w_res, axis=-1, keepdims=True)
    )

    d_z = jnp.concatenate((d_z_pre, d_z_post, d_z_res), axis=-1)
    d_ab_tokens = jnp.concatenate((d_z, d_z * projected), axis=-1)
    d_ab_ref[...] = d_ab_ref[...] + jnp.sum(
        d_ab_tokens.reshape(block_tokens // 8, 8, 2 * dims.phi_cols), axis=0
    )

    d_projected = d_z * alpha_v
    d_raw_proj = d_projected * rstd
    d_phi_ref[...] = d_phi_ref[...] + jnp.dot(
        x_val.T,
        d_raw_proj,
        preferred_element_type=jnp.float32,
    )

    scale_x = (
        -jnp.sum(d_projected * projected, axis=-1, keepdims=True)
        * (rstd * rstd)
        * (1.0 / dims.flattened_size)
    )
    d_x_proj = jnp.dot(
        d_raw_proj,
        phi_val.T,
        preferred_element_type=jnp.float32,
    )
    for s in range(streams):
      sl = slice(s * embedding, (s + 1) * embedding)
      d_x_ref[:, sl] = (
          d_x_acc_ref[:, sl].astype(jnp.float32)
          + h_pre_val[:, s : s + 1] * d_li_f32
          + d_x_proj[:, sl]
          + x_ref[:, sl].astype(jnp.float32) * scale_x
      ).astype(d_x_ref.dtype)

  def _chunk_token_spec(shape: tuple[int, ...]) -> pl.BlockSpec:
    return pl.BlockSpec(
        (block_tokens,) + shape[1:],
        lambda c, s: (c * blocks_per_chunk + s,) + tuple(0 for _ in shape[1:]),
    )

  spec_x_2d = _chunk_token_spec((tokens, dims.flattened_size))
  spec_phi = pl.BlockSpec(phi_bf16.shape, lambda c, s: (0, 0))
  spec_gate_ab = pl.BlockSpec(gate_ab.shape, lambda c, s: (0, 0))
  spec_perms = pl.BlockSpec(perms_f32.shape, lambda c, s: (0, 0))
  spec_h_pre = _chunk_token_spec(h_pre.shape)
  spec_d_layer_input = _chunk_token_spec(d_layer_input.shape)
  spec_d_h_post = _chunk_token_spec(d_h_post.shape)
  spec_d_res = _chunk_token_spec(d_residual.shape)
  spec_d_phi = pl.BlockSpec(
      (None, dims.flattened_size, dims.phi_cols), lambda c, s: (c, 0, 0)
  )
  spec_d_ab = pl.BlockSpec((None, 8, 2 * dims.phi_cols), lambda c, s: (c, 0, 0))

  kernel_main = pltpu.emit_pipeline(
      pipeline_body,
      grid=(num_chunks, blocks_per_chunk),
      in_specs=(
          spec_x_2d,
          spec_phi,
          spec_gate_ab,
          spec_perms,
          spec_h_pre,
          spec_d_layer_input,
          spec_d_h_post,
          spec_d_res,
          spec_x_2d,
      ),
      out_specs=(
          spec_x_2d,
          spec_d_phi,
          spec_d_ab,
      ),
      dimension_semantics=common.POST_BWD_DIMENSION_SEMANTICS,
  )

  coeff_cost = dims.coeff_bwd_cost()
  pre_apply_cost = dims.pre_apply_bwd_cost()
  total_cost = pl.CostEstimate(
      flops=coeff_cost.flops + pre_apply_cost.flops,
      transcendentals=coeff_cost.transcendentals
      + pre_apply_cost.transcendentals,
      bytes_accessed=coeff_cost.bytes_accessed,
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
          in_specs=common.hbm_specs(9),
          out_specs=common.hbm_specs(3),
          cost_estimate=total_cost,
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=config.vmem_limit_bytes,
          ),
          interpret=config.interpret,
      )(
          x_2d,
          phi_bf16,
          gate_ab,
          perms_f32,
          h_pre,
          d_layer_input,
          d_h_post,
          d_residual,
          d_x_acc_2d,
      )

  d_phi = jnp.sum(d_phi_chunks, axis=0)
  d_ab_sum = jnp.sum(d_ab_chunks, axis=(0, 1))
  d_b_vec = d_ab_sum[: dims.phi_cols]
  d_alpha_vec = d_ab_sum[dims.phi_cols :]
  d_coeff_grads = common.MhcCoeffGradients(
      phi=d_phi,
      alpha_pre=jnp.sum(d_alpha_vec[dims.pre_slice], keepdims=True),
      b_pre=d_b_vec[dims.pre_slice],
      alpha_post=jnp.sum(d_alpha_vec[dims.post_slice], keepdims=True),
      b_post=d_b_vec[dims.post_slice],
      alpha_res=jnp.sum(d_alpha_vec[dims.res_slice], keepdims=True),
      b_res=d_b_vec[dims.res_slice],
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
  d_residual_flat = d_residual.reshape(tokens, streams, streams)
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
      residual.reshape(tokens, streams, streams),
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
    permutations: jax.Array,
    residuals: tuple[
        tuple[jax.Array, jax.Array], tuple[jax.Array, common.MhcWeights]
    ],
    cotangents: tuple[jax.Array, common.KernelContext],
) -> tuple[jax.Array, common.MhcWeights]:
  """Custom-VJP backward rule for the low-level pre-branch entry point."""
  saved, (x, weights) = residuals
  d_layer_input, (d_x, d_h_post, d_residual) = cotangents
  return pre_bwd(
      saved,
      (d_layer_input, d_x, d_h_post, d_residual),
      x,
      weights,
      permutations,
      config=config,
  )


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
