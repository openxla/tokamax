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
import math
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.mhc_lite import common
from tokamax._src.ops.experimental.mhc_lite import mhc_kernels_bwd


def _zero_index_map(ndim: int, *_):
  """Index map returning all-zero block coordinates of length `ndim`."""
  return (0,) * ndim


def _full_param_block_spec(param: jax.Array) -> pl.BlockSpec:
  """Creates an unsharded BlockSpec covering the full parameter shape."""
  return pl.BlockSpec(
      param.shape, functools.partial(_zero_index_map, param.ndim)
  )


def _fwd_chunk_tokens(tokens: int, config: common.MhcKernelConfig) -> int:
  """Returns the pipelined token chunk size for forward kernels."""
  pipelined = config.fwd_pipeline_blocks * config.block_size
  if config.fwd_pipeline_blocks > 1 and tokens % pipelined == 0:
    return pipelined
  return config.block_size


def _prepare_fwd_params(
    weights: common.MhcWeights,
    permutations: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
  """Folds `phi`, pre-scales by `alpha`, concatenates biases, and pads permutations."""
  with jax.named_scope("prepare_fwd_params"):
    phi = weights.fold_norm_scale()
    num_permutations, streams, _ = permutations.shape
    permutations_2d = (
        permutations.astype(jnp.float32)
        .swapaxes(-1, -2)
        .reshape(num_permutations, streams * streams)
    )
    alpha_vec = jnp.concatenate(
        [
            jnp.broadcast_to(a, (n,))
            for a, n in (
                (weights.alpha_pre, streams),
                (weights.alpha_post, streams),
                (weights.alpha_res, num_permutations),
            )
        ],
        axis=0,
    )[None, :]
    gate_bias = jnp.concatenate(
        (weights.b_pre, weights.b_post, weights.b_res),
        axis=0,
        dtype=jnp.float32,
    )[None, :]
    phi_scaled = phi * alpha_vec
    permutations_padded = jnp.pad(
        permutations_2d, ((2 * streams, 0), (0, 0))
    )
    return phi, (phi_scaled, gate_bias, permutations_padded)


def _run_fwd_pipeline(
    body_fn,
    tokens: int,
    chunk_tokens: int,
    token_args: tuple[jax.Array, ...],
    out_shapes_dtypes: tuple[tuple[tuple[int, ...], jnp.dtype], ...],
    cost_estimate: pl.CostEstimate,
    config: common.MhcKernelConfig,
    param_args: tuple[jax.Array, ...] = (),
):
  """Emits and executes a 1D token-pipelined Pallas forward kernel."""
  in_args = token_args + param_args
  in_specs = tuple(
      common.token_block_spec(arg.shape, chunk_tokens) for arg in token_args
  ) + tuple(_full_param_block_spec(arg) for arg in param_args)
  out_specs = tuple(
      common.token_block_spec(shape, chunk_tokens)
      for shape, _ in out_shapes_dtypes
  )
  out_shape = tuple(
      jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in out_shapes_dtypes
  )
  kernel_main = pltpu.emit_pipeline(
      lambda unused_step, *refs: body_fn(*refs),
      grid=(tokens // chunk_tokens,),
      in_specs=in_specs,
      out_specs=out_specs,
      dimension_semantics=(pltpu.PARALLEL,),
      _explicit_indices=True,
  )
  with common.tpu_mesh_context():
    return pl.pallas_call(
        kernel_main,
        out_shape=out_shape,
        in_specs=(common.HBM_SPEC,) * len(in_args),
        out_specs=(common.HBM_SPEC,) * len(out_shapes_dtypes),
        cost_estimate=cost_estimate,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=config.vmem_limit_bytes,
        ),
        interpret=config.interpret,
    )(*in_args)


def _pre_fwd_pipeline_body(
    streams: int,
    embedding: int,
    config: common.MhcKernelConfig,
    x_ref,
    phi_ref,
    gate_bias_ref,
    permutations_ref,
    layer_input_ref,
    h_pre_ref,
    h_post_ref,
    residual_ref,
) -> None:
  """Pallas pipeline body for fused pre-branch forward kernel."""
  gate_bias = gate_bias_ref[...]
  permutations = permutations_ref[...]
  phi = phi_ref[...]

  # Full-chunk MXU projection overlapped with hardware bf16 sum-of-squares
  # reduction on VPU (opcode 81), sharing the bfloat16 vector registers of x.
  x_full = x_ref[...]
  raw_proj = jnp.dot(x_full, phi, preferred_element_type=jnp.float32)
  flattened_f32 = x_full.astype(jnp.float32)
  mean_square = jnp.mean(jnp.square(flattened_f32), axis=-1, keepdims=True)
  inv_rms = jax.lax.rsqrt(mean_square + config.rms_epsilon)
  logits = raw_proj * inv_rms + gate_bias

  # Fused pre/post sigmoid on lanes 0..2*S-1.
  gates = jax.nn.sigmoid(logits[:, : 2 * streams])
  h_pre = gates[:, :streams] + config.pre_mapping_epsilon
  h_post = 2.0 * gates[:, streams : 2 * streams]

  # Zero-shift 32-lane softmax + padded (32, S^2) permutation projection:
  # Masking lanes 0..2*S-1 with -inf avoids cross-lane shifts (XLU opcode 58).
  col_idx = jax.lax.broadcasted_iota(jnp.int32, logits.shape, dimension=1)
  res_logits = jnp.where(col_idx >= 2 * streams, logits, -jnp.inf)
  res_weights = jax.nn.softmax(res_logits, axis=-1)
  residual = jnp.dot(
      res_weights,
      permutations,
      preferred_element_type=jnp.float32,
  )

  h_pre_ref[...] = h_pre
  h_post_ref[...] = h_post
  residual_ref[...] = residual.astype(residual_ref.dtype)

  # Strip-mined fori_loop stream collapse over 64-token strips and 128-col
  # bands to bound live float32 vector registers and prevent VMEM register
  # spills.
  strip_tokens = math.gcd(x_full.shape[0], 64)
  num_strips = x_full.shape[0] // strip_tokens

  def _li_strip_body(strip_idx: jax.Array, _) -> None:
    t_dslice = pl.dslice(strip_idx * strip_tokens, strip_tokens)
    h_s = h_pre_ref[t_dslice, :]
    for band_col in range(0, embedding, 128):
      acc = h_s[:, 0:1] * x_ref[t_dslice, band_col : band_col + 128]
      for s in range(1, streams):
        col = s * embedding + band_col
        acc = acc + h_s[:, s : s + 1] * x_ref[t_dslice, col : col + 128]
      layer_input_ref[t_dslice, band_col : band_col + 128] = acc.astype(
          jnp.bfloat16
      )

  jax.lax.fori_loop(0, num_strips, _li_strip_body, None)


def _post_apply_fwd_pipeline_body(
    streams: int,
    embedding: int,
    sub_tokens: int,
    num_subtiles: int,
    x_ref,
    layer_output_ref,
    h_post_ref,
    residual_ref,
    output_ref,
) -> None:
  """Subtiled VPU pipeline body for 2D post-branch stream mixing."""
  for sub_idx in range(num_subtiles):
    s_slice = slice(sub_idx * sub_tokens, (sub_idx + 1) * sub_tokens)
    res_sub = residual_ref[s_slice, :].astype(jnp.float32)
    h_post_sub = h_post_ref[s_slice, :]
    l_sub = layer_output_ref[s_slice, :].astype(jnp.float32)
    x_sub = [
        x_ref[s_slice, k * embedding : (k + 1) * embedding].astype(jnp.float32)
        for k in range(streams)
    ]
    for j in range(streams):
      out_j = h_post_sub[:, j : j + 1] * l_sub
      for k in range(streams):
        out_j = (
            out_j + res_sub[:, j * streams + k : j * streams + k + 1] * x_sub[k]
        )
      output_ref[s_slice, j * embedding : (j + 1) * embedding] = out_j.astype(
          output_ref.dtype
      )


def _post_and_pre_fwd_pipeline_body(
    streams: int,
    embedding: int,
    config: common.MhcKernelConfig,
    sub_tokens: int,
    num_subtiles: int,
    x_ref,
    layer_output_ref,
    h_post_ref,
    residual_ref,
    phi_ref,
    gate_bias_ref,
    permutations_ref,
    layer_input_ref,
    x_next_out_ref,
    *out_refs,
) -> None:
  """Pallas pipeline body for fused Stage-1 post + Stage-2 pre forward kernel."""
  _post_apply_fwd_pipeline_body(
      streams,
      embedding,
      sub_tokens,
      num_subtiles,
      x_ref,
      layer_output_ref,
      h_post_ref,
      residual_ref,
      x_next_out_ref,
  )
  _pre_fwd_pipeline_body(
      streams,
      embedding,
      config,
      x_next_out_ref,
      phi_ref,
      gate_bias_ref,
      permutations_ref,
      layer_input_ref,
      *out_refs,
  )


def _pre_fwd(
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    config: common.MhcKernelConfig,
):
  """Custom-VJP forward rule for `pre`."""
  with jax.named_scope("mhc_pre_fwd_fused"):
    common.validate_inputs(x, config.block_size, permutations.shape)
    streams = permutations.shape[-1]
    embedding = weights.norm_scale.shape[0] // streams
    tokens = x.size // (streams * embedding)
    x_in = x.reshape(tokens, streams * embedding)
    chunk_tokens = _fwd_chunk_tokens(tokens, config)
    phi, param_args = _prepare_fwd_params(weights, permutations)
    layer_input, h_pre, h_post, residual = _run_fwd_pipeline(
        functools.partial(_pre_fwd_pipeline_body, streams, embedding, config),
        tokens,
        chunk_tokens,
        token_args=(x_in,),
        param_args=param_args,
        out_shapes_dtypes=(
            ((tokens, embedding), jnp.bfloat16),
            ((tokens, streams), jnp.float32),
            ((tokens, streams), jnp.float32),
            ((tokens, streams * streams), jnp.bfloat16),
        ),
        cost_estimate=common.pre_cost(
            tokens, streams, embedding, permutations.shape[0]
        ),
        config=config,
    )
    context = common.MHCContext(
        x=x_in,
        h_post=h_post,
        residual=residual,
        implementation="mosaic_tpu",
    )
    layer_input_out = (
        layer_input.reshape(*x.shape[:2], embedding)
        if x.ndim == 4
        else layer_input
    )
    return (layer_input_out, context), (
        (phi, h_pre),
        (x, weights, permutations),
    )


def _post_fwd(
    layer_output: jax.Array,
    context: common.MHCContext,
    config: common.MhcKernelConfig,
):
  """Custom-VJP forward rule for `post`."""
  with jax.named_scope("mhc_post_apply_fwd"):
    x, h_post, residual = context.x, context.h_post, context.residual
    embedding = layer_output.shape[-1]
    tokens = layer_output.size // embedding
    streams = h_post.shape[-1]
    flattened_size = streams * embedding
    x_in = x.reshape(tokens, flattened_size)
    x_check = (
        x.reshape(*layer_output.shape[:2], streams, embedding)
        if layer_output.ndim == 3
        else x_in
    )
    common.validate_inputs(x_check, config.block_size)

    block_tokens = _fwd_chunk_tokens(tokens, config)
    sub_tokens = math.gcd(block_tokens, 16)
    (output,) = _run_fwd_pipeline(
        functools.partial(
            _post_apply_fwd_pipeline_body,
            streams,
            embedding,
            sub_tokens,
            block_tokens // sub_tokens,
        ),
        tokens,
        block_tokens,
        token_args=(
            x_in,
            layer_output.reshape(tokens, embedding),
            h_post.reshape(tokens, streams),
            residual.reshape(tokens, streams * streams),
        ),
        out_shapes_dtypes=((x_in.shape, jnp.bfloat16),),
        cost_estimate=common.post_cost(tokens, streams, embedding),
        config=config,
    )
    out_shape = (
        (*layer_output.shape[:2], streams, embedding)
        if layer_output.ndim == 3
        else (tokens, flattened_size)
    )
    return output.reshape(out_shape), (layer_output, context)


def _post_and_pre_fwd(
    layer_output: jax.Array,
    context: common.MHCContext,
    next_weights: common.MhcWeights,
    permutations: jax.Array,
    config: common.MhcKernelConfig,
):
  """Custom-VJP forward rule for `post_and_pre`."""
  with jax.named_scope("mhc_post_and_pre_fwd"):
    x, h_post, residual = context.x, context.h_post, context.residual
    embedding = layer_output.shape[-1]
    tokens = layer_output.size // embedding
    streams = h_post.shape[-1]
    flattened_size = streams * embedding
    x_in = x.reshape(tokens, flattened_size)
    x_check = (
        x.reshape(*layer_output.shape[:2], streams, embedding)
        if layer_output.ndim == 3
        else x_in
    )
    common.validate_inputs(x_check, config.block_size, permutations.shape)

    chunk_tokens = _fwd_chunk_tokens(tokens, config)
    sub_tokens = math.gcd(chunk_tokens, 16)
    phi, param_args = _prepare_fwd_params(next_weights, permutations)
    next_layer_input, x_next, next_h_pre, next_h_post, next_residual = (
        _run_fwd_pipeline(
            functools.partial(
                _post_and_pre_fwd_pipeline_body,
                streams,
                embedding,
                config,
                sub_tokens,
                chunk_tokens // sub_tokens,
            ),
            tokens,
            chunk_tokens,
            token_args=(
                x_in,
                layer_output.reshape(tokens, embedding),
                h_post.reshape(tokens, streams),
                residual.reshape(tokens, streams * streams),
            ),
            param_args=param_args,
            out_shapes_dtypes=(
                ((tokens, embedding), jnp.bfloat16),
                (x_in.shape, jnp.bfloat16),
                ((tokens, streams), jnp.float32),
                ((tokens, streams), jnp.float32),
                ((tokens, streams * streams), jnp.bfloat16),
            ),
            cost_estimate=common.post_and_pre_cost(
                tokens, streams, embedding, permutations.shape[0]
            ),
            config=config,
        )
    )
    next_context = common.MHCContext(
        x=x_next,
        h_post=next_h_post,
        residual=next_residual,
        implementation="mosaic_tpu",
    )
    return (next_layer_input.reshape(layer_output.shape), next_context), (
        (phi, next_h_pre),
        (layer_output, context),
        (x_next, next_weights, permutations),
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def pre(
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
) -> tuple[jax.Array, common.MHCContext]:
  """Runs the coefficient and pre-application kernels."""
  primals_out, _ = _pre_fwd(x, weights, permutations, config)
  return primals_out


pre.defvjp(_pre_fwd, mhc_kernels_bwd.pre_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def post(
    layer_output: jax.Array,
    context: common.MHCContext,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
) -> jax.Array:
  """Runs the fused post-gate and residual-mixing kernel."""
  output, _ = _post_fwd(layer_output, context, config)
  return output


post.defvjp(_post_fwd, mhc_kernels_bwd.post_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def post_and_pre(
    layer_output: jax.Array,
    context: common.MHCContext,
    next_weights: common.MhcWeights,
    permutations: jax.Array,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
) -> tuple[jax.Array, common.MHCContext]:
  """Runs the fused Stage-1 post and Stage-2 pre kernel."""
  primals_out, _ = _post_and_pre_fwd(
      layer_output, context, next_weights, permutations, config
  )
  return primals_out


post_and_pre.defvjp(_post_and_pre_fwd, mhc_kernels_bwd.post_and_pre_bwd)
