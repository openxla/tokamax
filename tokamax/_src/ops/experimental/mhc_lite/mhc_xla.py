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
"""Reference JAX-only (XLA) implementation for mHC-lite."""

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.mhc_lite import common


def _gate_logits(
    normalized: jax.Array,
    w: jax.Array,
    alpha: jax.Array,
    b: jax.Array,
) -> jax.Array:
  """Computes float32 gate logits `alpha * (normalized @ w) + b`."""
  with jax.named_scope("gate_logits"):
    return alpha * jnp.dot(normalized, w) + b


def pre_xla(
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    *,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
) -> tuple[jax.Array, common.MHCContext]:
  """Reference JAX-only implementation of the mHC-lite pre-branch stage."""
  with jax.named_scope("mhc_pre_xla"):
    if x.ndim == 2:
      tokens, flattened_size = x.shape
      streams = permutations.shape[1]
      embedding = flattened_size // streams
      layer_input_shape = (tokens, embedding)
    elif x.ndim == 4:
      batch, sequence, streams, embedding = x.shape
      tokens = batch * sequence
      flattened_size = streams * embedding
      layer_input_shape = (batch, sequence, embedding)
    else:
      raise common.UnsupportedInputError(
          "Expected x to have shape (batch, sequence, streams, embedding) or"
          f" (tokens, streams * embedding); got {x.shape}."
      )
    dtype = jnp.promote_types(x.dtype, jnp.float32)
    with jax.named_scope("rms_norm"):
      x_f32 = x.reshape(tokens, streams, embedding).astype(dtype)
      flat_f32 = x_f32.reshape(tokens, flattened_size)
      inv_rms = jax.lax.rsqrt(
          jnp.mean(jnp.square(flat_f32), axis=-1, keepdims=True)
          + config.rms_epsilon
      )
      normalized = flat_f32 * inv_rms * weights.norm_scale

    with jax.named_scope("h_pre_gate"):
      h_pre = (
          jax.nn.sigmoid(
              _gate_logits(
                  normalized, weights.w_pre, weights.alpha_pre, weights.b_pre
              )
          )
          + config.pre_mapping_epsilon
      )
    with jax.named_scope("stream_collapse"):
      layer_input = (
          jnp.sum(h_pre[:, :, None] * x_f32, axis=1)
          .astype(x.dtype)
          .reshape(layer_input_shape)
      )

    with jax.named_scope("h_post_gate"):
      h_post = 2.0 * jax.nn.sigmoid(
          _gate_logits(
              normalized, weights.w_post, weights.alpha_post, weights.b_post
          )
      )
    with jax.named_scope("residual_bvn_mix"):
      weights_res = jax.nn.softmax(
          _gate_logits(
              normalized, weights.w_res, weights.alpha_res, weights.b_res
          ),
          axis=-1,
      )
      residual = jnp.einsum("tp,pkj->tjk", weights_res, permutations)

    return layer_input, common.MHCContext(
        x=x,
        h_post=h_post,
        residual=residual,
        implementation="xla",
    )


def post_xla(
    layer_output: jax.Array,
    context: common.MHCContext,
) -> jax.Array:
  """Reference JAX-only implementation of the mHC-lite post-branch stage."""
  with jax.named_scope("mhc_post_xla"):
    embedding = layer_output.shape[-1]
    streams = context.h_post.shape[-1]
    tokens = (
        layer_output.shape[0]
        if layer_output.ndim == 2
        else layer_output.shape[0] * layer_output.shape[1]
    )
    out_shape = (
        (tokens, streams * embedding)
        if layer_output.ndim == 2
        else (*layer_output.shape[:2], streams, embedding)
    )
    dtype = jnp.promote_types(context.x.dtype, jnp.float32)
    x_f32 = context.x.reshape(tokens, streams, embedding).astype(dtype)
    layer_output_f32 = layer_output.reshape(tokens, embedding).astype(dtype)
    residual = context.residual.reshape(tokens, streams, streams)
    with jax.named_scope("residual_stream_mix"):
      residual_mix = jnp.einsum("tjk,tkd->tjd", residual, x_f32)
    with jax.named_scope("post_gate_scale"):
      post_mix = context.h_post[:, :, None] * layer_output_f32[:, None, :]
    with jax.named_scope("combine_post_output"):
      return (
          (residual_mix + post_mix).astype(context.x.dtype).reshape(out_shape)
      )


def post_and_pre_xla(
    layer_output: jax.Array,
    context: common.MHCContext,
    next_weights: common.MhcWeights,
    permutations: jax.Array,
    *,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
) -> tuple[jax.Array, common.MHCContext]:
  """Reference JAX-only implementation of fused Stage-1 post and Stage-2 pre."""
  with jax.named_scope("mhc_post_and_pre_xla"):
    x_next = post_xla(layer_output, context)
    return pre_xla(x_next, next_weights, permutations, config=config)
