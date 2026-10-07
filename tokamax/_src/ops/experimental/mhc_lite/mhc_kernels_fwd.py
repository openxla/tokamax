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
"""Low-level Pallas forward kernels and custom VJP functions for mHC-lite."""

import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.mhc_lite import common
from tokamax._src.ops.experimental.mhc_lite import mhc_kernels_bwd


def _vpu_block(
    x_block: jax.Array,
    projected_block: jax.Array,
    params: common.MhcCoeffParams,
    permutations_value: jax.Array,
    dims: common.MhcDims,
    config: common.MhcKernelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Computes RMSNorm scaling, routing gates, BvN residual, and stream mix."""
  # RMSNorm scaling: projected = (x_block @ phi) / sqrt(mean(x^2) + eps)
  flattened_f32 = x_block.astype(jnp.float32)
  mean_square = jnp.mean(jnp.square(flattened_f32), axis=-1, keepdims=True)
  inv_rms = jax.lax.rsqrt(mean_square + config.rms_epsilon)
  projected = projected_block * inv_rms

  # Pre/post gates:
  #   h_pre  = sigmoid(alpha_pre * z_pre + b_pre) + eps
  #   h_post = 2 * sigmoid(alpha_post * z_post + b_post)
  h_pre = (
      jax.nn.sigmoid(
          params.alpha_pre.astype(jnp.float32) * projected[:, dims.pre_slice]
          + params.b_pre.astype(jnp.float32)
      )
      + config.pre_mapping_epsilon
  )
  h_post = 2.0 * jax.nn.sigmoid(
      params.alpha_post.astype(jnp.float32) * projected[:, dims.post_slice]
      + params.b_post.astype(jnp.float32)
  )

  # Doubly-stochastic residual matrix:
  #   H_res = softmax(alpha_res * z_res + b_res) @ P
  res_weights = jax.nn.softmax(
      params.alpha_res.astype(jnp.float32) * projected[:, dims.res_slice]
      + params.b_res.astype(jnp.float32),
      axis=-1,
  )
  residual = jnp.dot(
      res_weights.astype(jnp.bfloat16),
      permutations_value.astype(jnp.bfloat16),
      preferred_element_type=jnp.float32,
  ).astype(jnp.bfloat16)

  # Collapse streams into branch input:
  #   layer_input[t, e] = sum_{j=0..S-1} h_pre[t, j] * x[t, j, e]
  embedding = dims.embedding
  layer_input = h_pre[:, 0:1] * flattened_f32[:, 0:embedding]
  for i in range(1, dims.streams):
    layer_input = (
        layer_input
        + h_pre[:, i : i + 1]
        * flattened_f32[:, i * embedding : (i + 1) * embedding]
    )
  return layer_input.astype(jnp.bfloat16), h_pre, h_post, residual


def _deref(ref):
  """Reads the full array value from a Pallas VMEM reference."""
  return ref[...]


def _zero_index_map(ndim: int, *_):
  """Index map returning all-zero block coordinates of length `ndim`."""
  return (0,) * ndim


def _full_param_block_spec(param: jax.Array) -> pl.BlockSpec:
  """Creates an unsharded BlockSpec covering the full parameter shape."""
  return pl.BlockSpec(
      param.shape, functools.partial(_zero_index_map, param.ndim)
  )


def _write_pre_fwd_outputs(
    token_slice: slice,
    layer_input_ref,
    output_refs: common.MhcCoeffOutputs,
    layer_input: jax.Array,
    h_pre: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
) -> None:
  """Writes a single block of pre-forward outputs to VMEM refs."""
  layer_input_ref[token_slice, :] = layer_input
  output_refs.h_pre[token_slice, :] = h_pre.astype(output_refs.h_pre.dtype)
  output_refs.h_post[token_slice, :] = h_post.astype(output_refs.h_post.dtype)
  output_refs.residual[token_slice, :] = residual.astype(
      output_refs.residual.dtype
  )


def _pre_fwd_pipeline_body(
    dims: common.MhcDims,
    config: common.MhcKernelConfig,
    pipeline_blocks: int,
    num_params: int,
    x_ref,
    *refs,
) -> None:
  """Pallas pipeline body for fused pre-branch forward kernel."""
  ref_iter = iter(refs)
  param_refs = common.MhcCoeffParams(
      *(next(ref_iter) for _ in range(num_params))
  )
  permutations_ref = next(ref_iter)
  layer_input_ref = next(ref_iter)
  output_refs = common.MhcCoeffOutputs(*ref_iter)

  params = jax.tree.map(_deref, param_refs)
  permutations_value = permutations_ref[...]
  phi_bf16 = params.phi.astype(jnp.bfloat16)
  block_size = config.block_size

  # Double-buffered inter-block software pipeline:
  # Prologue: Compute MXU projection for Block 0 (x_block @ phi)
  x_current = x_ref[0:block_size, :]
  projected_current = jnp.dot(
      x_current, phi_bf16, preferred_element_type=jnp.float32
  )

  for i in range(pipeline_blocks - 1):
    current_slice = slice(i * block_size, (i + 1) * block_size)
    next_slice = slice((i + 1) * block_size, (i + 2) * block_size)
    # Overlap: Block i+1 MXU projection with Block i VPU operations
    x_next = x_ref[next_slice, :]
    projected_next = jnp.dot(
        x_next, phi_bf16, preferred_element_type=jnp.float32
    )
    layer_input, h_pre, h_post, residual = _vpu_block(
        x_current, projected_current, params, permutations_value, dims, config
    )
    _write_pre_fwd_outputs(
        current_slice,
        layer_input_ref,
        output_refs,
        layer_input,
        h_pre,
        h_post,
        residual,
    )
    x_current = x_next
    projected_current = projected_next

  # Epilogue: Compute Block K-1 VPU operations
  last_slice = slice(
      (pipeline_blocks - 1) * block_size, pipeline_blocks * block_size
  )
  layer_input, h_pre, h_post, residual = _vpu_block(
      x_current, projected_current, params, permutations_value, dims, config
  )
  _write_pre_fwd_outputs(
      last_slice,
      layer_input_ref,
      output_refs,
      layer_input,
      h_pre,
      h_post,
      residual,
  )


def _pre_fwd_fused(
    x: jax.Array,
    coeff_params: common.MhcCoeffParams,
    permutations: jax.Array,
    config: common.MhcKernelConfig,
) -> tuple[jax.Array, common.MhcCoeffOutputs]:
  """Fused coefficient and pre-application forward Pallas kernel."""
  tokens, flattened_size = x.shape
  streams = permutations.shape[1]
  embedding = flattened_size // streams
  dims = common.MhcDims(
      tokens=tokens,
      streams=streams,
      embedding=embedding,
      num_permutations=permutations.shape[0],
  )

  pipeline_blocks = config.fwd_pipeline_blocks
  if (
      pipeline_blocks > 1
      and tokens % (pipeline_blocks * config.block_size) == 0
  ):
    chunk_tokens = pipeline_blocks * config.block_size
  else:
    pipeline_blocks = 1
    chunk_tokens = config.block_size

  num_params = len(jax.tree.leaves(coeff_params))
  spec_x = common.token_block_spec((tokens, dims.flattened_size), chunk_tokens)
  param_specs = jax.tree.map(_full_param_block_spec, coeff_params)
  permutations_2d = permutations.swapaxes(-1, -2).reshape(
      dims.num_permutations, streams * streams
  )
  spec_layer_input = common.token_block_spec((tokens, embedding), chunk_tokens)
  output_specs = (
      spec_layer_input,
      common.token_block_spec((tokens, streams), chunk_tokens),
      common.token_block_spec((tokens, streams), chunk_tokens),
      common.token_block_spec((tokens, streams * streams), chunk_tokens),
  )

  kernel_main = pltpu.emit_pipeline(
      functools.partial(
          _pre_fwd_pipeline_body, dims, config, pipeline_blocks, num_params
      ),
      grid=(tokens // chunk_tokens,),
      in_specs=(
          spec_x,
          *jax.tree.leaves(param_specs),
          _full_param_block_spec(permutations_2d),
      ),
      out_specs=output_specs,
      dimension_semantics=(pltpu.PARALLEL,),
  )

  with jax.named_scope("mhc_pre_fwd_fused"):
    with common.tpu_mesh_context():
      layer_input, h_pre, h_post, residual = pl.pallas_call(
          kernel_main,
          out_shape=(
              jax.ShapeDtypeStruct((tokens, embedding), jnp.bfloat16),
              jax.ShapeDtypeStruct((tokens, streams), jnp.float32),
              jax.ShapeDtypeStruct((tokens, streams), jnp.float32),
              jax.ShapeDtypeStruct((tokens, streams * streams), jnp.bfloat16),
          ),
          in_specs=(common.HBM_SPEC,) * (num_params + 2),
          out_specs=(common.HBM_SPEC,) * 4,
          cost_estimate=dims.pre_fwd_cost(),
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=config.vmem_limit_bytes,
          ),
          interpret=config.interpret,
      )(
          x,
          *jax.tree.leaves(coeff_params),
          permutations_2d,
      )
    return layer_input, common.MhcCoeffOutputs(
        h_pre=h_pre, h_post=h_post, residual=residual
    )


def _post_apply_fwd_pipeline_body(
    streams: int,
    x_ref,
    layer_output_ref,
    h_post_ref,
    residual_ref,
    output_ref,
) -> None:
  """Pallas pipeline body for post-branch forward kernel."""
  # Post-branch stream mixing:
  #   Y[t, j, e] = sum_k H_res[t, j, k] * X[t, k, e] + h_post[t, j] * L[t, e]
  residual_value = residual_ref[...]
  if residual_value.ndim == 2:
    residual_value = residual_value.reshape(
        residual_value.shape[0], streams, streams
    )
  residual_mix = jnp.matmul(
      residual_value.astype(jnp.bfloat16),
      x_ref[...],
      preferred_element_type=jnp.float32,
  )
  post_mix = h_post_ref[...][:, :, None].astype(jnp.float32) * layer_output_ref[
      ...
  ][:, None, :].astype(jnp.float32)
  output_ref[...] = (residual_mix + post_mix).astype(jnp.bfloat16)


def _post_apply_fwd(
    x: jax.Array,
    layer_output: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
    config: common.MhcKernelConfig,
) -> jax.Array:
  """Builds the Pallas pipeline for the post-branch forward pass."""
  tokens, streams, embedding = x.shape
  dims = common.MhcDims(tokens=tokens, streams=streams, embedding=embedding)

  spec_x = common.token_block_spec(
      (tokens, streams, embedding), config.block_size
  )
  spec_layer_output = common.token_block_spec(
      (tokens, embedding), config.block_size
  )
  spec_h_post = common.token_block_spec((tokens, streams), config.block_size)
  spec_residual = common.token_block_spec(
      (tokens, streams * streams), config.block_size
  )

  kernel_main = pltpu.emit_pipeline(
      functools.partial(_post_apply_fwd_pipeline_body, streams),
      grid=(tokens // config.block_size,),
      in_specs=(
          spec_x,
          spec_layer_output,
          spec_h_post,
          spec_residual,
      ),
      out_specs=spec_x,
      dimension_semantics=(pltpu.PARALLEL,),
  )

  with jax.named_scope("mhc_post_apply_fwd"):
    with common.tpu_mesh_context():
      return pl.pallas_call(
          kernel_main,
          out_shape=jax.ShapeDtypeStruct(
              (tokens, streams, embedding), jnp.bfloat16
          ),
          in_specs=(common.HBM_SPEC,) * 4,
          out_specs=common.HBM_SPEC,
          cost_estimate=dims.post_apply_fwd_cost(),
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=config.vmem_limit_bytes,
          ),
          interpret=config.interpret,
      )(x, layer_output, h_post, residual)


def pre_fwd(
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    config: common.MhcKernelConfig,
) -> tuple[tuple[jax.Array, common.KernelContext], tuple[jax.Array, jax.Array]]:
  """Runs coefficient and pre-application forward kernels."""
  common.validate_inputs(x, config.block_size, permutations.shape)
  batch, sequence, streams, embedding = x.shape
  tokens = batch * sequence
  flattened_size = streams * embedding
  x_2d = x.reshape(tokens, flattened_size)

  coeff_params = weights.to_coeff_params()
  layer_input, outputs = _pre_fwd_fused(
      x_2d, coeff_params, permutations, config
  )

  context: common.KernelContext = (
      x_2d,
      outputs.h_post,
      outputs.residual,
  )
  primals_out = (layer_input.reshape(batch, sequence, embedding), context)
  saved_residuals = (coeff_params.phi, outputs.h_pre)
  return primals_out, saved_residuals


def post_fwd(
    layer_output: jax.Array,
    x: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
    config: common.MhcKernelConfig,
) -> jax.Array:
  """Runs the fused post-gate and residual-mixing forward kernel."""
  batch, sequence, embedding = layer_output.shape
  tokens = batch * sequence
  streams = h_post.shape[-1]
  output = _post_apply_fwd(
      x.reshape(tokens, streams, embedding),
      layer_output.reshape(tokens, embedding),
      h_post.reshape(tokens, streams),
      residual.reshape(tokens, streams * streams),
      config,
  )
  return output.reshape(batch, sequence, streams, embedding)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _pre_op(
    config: common.MhcKernelConfig,
    x: jax.Array,
    permutations: jax.Array,
    weights: common.MhcWeights,
) -> tuple[jax.Array, common.KernelContext]:
  """Differentiable pre-branch mHC operation."""
  (layer_input, context), _ = pre_fwd(x, weights, permutations, config)
  return layer_input, context


def _pre_op_fwd(
    config: common.MhcKernelConfig,
    x: jax.Array,
    permutations: jax.Array,
    weights: common.MhcWeights,
):
  """Custom-VJP forward rule for the pre-branch operation."""
  primals_out, saved = pre_fwd(x, weights, permutations, config)
  x_2d = primals_out[1][0]
  return primals_out, (saved, (x_2d, weights, permutations))


_pre_op.defvjp(_pre_op_fwd, mhc_kernels_bwd.pre_op_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _post_op(
    config: common.MhcKernelConfig,
    layer_output: jax.Array,
    x: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
) -> jax.Array:
  """Differentiable post-branch mHC operation."""
  return post_fwd(layer_output, x, h_post, residual, config)


def _post_op_fwd(
    config: common.MhcKernelConfig,
    layer_output: jax.Array,
    x: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
):
  """Custom-VJP forward rule for the post-branch operation."""
  output = post_fwd(layer_output, x, h_post, residual, config)
  return output, (layer_output, x, h_post, residual)


_post_op.defvjp(_post_op_fwd, mhc_kernels_bwd.post_op_bwd)


def pre(
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
) -> tuple[jax.Array, common.KernelContext]:
  """Runs the coefficient and pre-application kernels.

  Args:
    x: Input streams with shape `[batch, sequence, streams, embedding]`.
    weights: Structured weights container.
    permutations: Permutation matrices with shape `[streams!, streams,
      streams]`.
    config: Kernel tuning and compiler configuration.

  Returns:
    A pair containing the branch input and opaque kernel context.
  """
  common.validate_inputs(x, config.block_size, permutations.shape)
  common.validate_token_block_size(
      x.shape[0] * x.shape[1], config.bwd_block_size, name="bwd_block_size"
  )
  return _pre_op(config, x, permutations, weights)


def post(
    layer_output: jax.Array,
    context: common.KernelContext,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
) -> jax.Array:
  """Runs the fused post-gate and residual-mixing kernel.

  Args:
    layer_output: Wrapped branch output with shape `[batch, sequence,
      embedding]`.
    context: Opaque context returned by `pre`.
    config: Kernel tuning and compiler configuration.

  Returns:
    Mixed output streams with shape `[batch, sequence, streams, embedding]`.
  """
  x, h_post, residual = context
  tokens = layer_output.shape[0] * layer_output.shape[1]
  embedding = layer_output.shape[-1]
  feature_block_size = min(embedding, config.bwd_feature_block_size)
  if x.ndim == 4:
    common.validate_inputs(x, config.block_size)
  else:
    common.validate_token_block_size(
        tokens, config.block_size, name="block_size"
    )
  common.validate_token_block_size(
      tokens, config.bwd_block_size, name="bwd_block_size"
  )
  common.validate_feature_block_size(embedding, feature_block_size)
  return _post_op(config, layer_output, x, h_post, residual)
