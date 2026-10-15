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


def pre_xla(
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    *,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
) -> tuple[jax.Array, common.MHCContext]:
  """Reference JAX-only implementation of the mHC-lite pre-branch stage."""
  if x.ndim != 4:
    raise common.UnsupportedInputError(
        "Expected x to have shape (batch, sequence, streams, embedding); got"
        f" {x.shape}."
    )
  batch, sequence, streams, embedding = x.shape
  tokens = batch * sequence
  flattened_size = streams * embedding
  permutation_count = permutations.shape[0]
  x_flat = x.reshape(tokens, streams, embedding)
  flattened_f32 = x_flat.reshape(tokens, flattened_size).astype(jnp.float32)
  normalized = (
      flattened_f32
      * jax.lax.rsqrt(
          jnp.mean(jnp.square(flattened_f32), axis=-1, keepdims=True)
          + config.rms_epsilon
      )
      * weights.norm_scale.astype(jnp.float32)
  ).astype(x.dtype)

  h_pre = (
      jax.nn.sigmoid(
          weights.alpha_pre.astype(jnp.float32)
          * jnp.dot(
              normalized,
              weights.w_pre,
              preferred_element_type=jnp.float32,
          )
          + weights.b_pre.astype(jnp.float32)
      )
      + config.pre_mapping_epsilon
  )
  layer_input = (
      jnp.sum(
          h_pre[:, :, None] * x_flat.astype(jnp.float32),
          axis=1,
      )
      .astype(x.dtype)
      .reshape(batch, sequence, embedding)
  )

  h_post = 2.0 * jax.nn.sigmoid(
      weights.alpha_post.astype(jnp.float32)
      * jnp.dot(
          normalized,
          weights.w_post,
          preferred_element_type=jnp.float32,
      )
      + weights.b_post.astype(jnp.float32)
  )
  weights_res = jax.nn.softmax(
      weights.alpha_res.astype(jnp.float32)
      * jnp.dot(
          normalized,
          weights.w_res,
          preferred_element_type=jnp.float32,
      )
      + weights.b_res.astype(jnp.float32),
      axis=-1,
  )
  residual = jnp.dot(
      weights_res,
      permutations.swapaxes(-1, -2)
      .reshape(permutation_count, streams * streams)
      .astype(jnp.float32),
  ).reshape(tokens, streams, streams)

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
  batch, sequence, streams, embedding = context.x.shape
  tokens = batch * sequence
  x_flat = context.x.reshape(tokens, streams, embedding)
  layer_output_flat = layer_output.reshape(tokens, embedding)
  residual = context.residual.reshape(tokens, streams, streams)
  residual_mix = jnp.einsum(
      "tjk,tkd->tjd",
      residual.astype(context.x.dtype),
      x_flat,
      preferred_element_type=jnp.float32,
  )
  post_mix = (
      context.h_post.astype(jnp.float32)[:, :, None]
      * layer_output_flat.astype(jnp.float32)[:, None, :]
  )
  return (
      (residual_mix + post_mix)
      .astype(context.x.dtype)
      .reshape(batch, sequence, streams, embedding)
  )
