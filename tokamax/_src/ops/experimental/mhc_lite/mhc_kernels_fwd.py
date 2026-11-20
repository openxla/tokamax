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
"""Low-level Pallas forward kernels for mHC-lite."""

import functools
import math
from typing import Any
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.mhc_lite import common


def _full_param_block_spec(param: jax.Array) -> pl.BlockSpec:
  """Creates an unsharded BlockSpec covering the full parameter shape."""
  return pl.BlockSpec(param.shape, lambda *_: (0,) * param.ndim)


def _prepare_fwd_params(
    weights: common.MhcWeights,
    permutations: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
  """Folds `phi`, pre-scales by `alpha`, concatenates biases, and pads permutations."""
  with jax.named_scope("prepare_fwd_params"):
    phi = weights.fold_norm_scale()
    num_permutations, streams, _ = permutations.shape
    with jax.named_scope("scale_phi_by_alpha"):
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
      phi_scaled = phi * alpha_vec
    with jax.named_scope("pack_gate_bias"):
      gate_bias = jnp.concatenate(
          (weights.b_pre, weights.b_post, weights.b_res),
          axis=0,
          dtype=jnp.float32,
      )[None, :]
    with jax.named_scope("pad_permutations_32_lanes"):
      permutations_2d = (
          permutations.astype(jnp.float32)
          .swapaxes(-1, -2)
          .reshape(num_permutations, streams * streams)
      )
      permutations_padded = jnp.pad(permutations_2d, ((2 * streams, 0), (0, 0)))
    return phi, (phi_scaled, gate_bias, permutations_padded)


def _run_fwd_pipeline(
    scope_name: str,
    body_fn,
    tokens: int,
    chunk_tokens: int,
    token_args: tuple[jax.Array, ...],
    out_shapes_dtypes: tuple[
        tuple[tuple[int, ...], jax.typing.DTypeLike], ...
    ],
    cost_estimate: pl.CostEstimate,
    *,
    vmem_limit_bytes: int,
    interpret: bool,
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
        name=scope_name,
        out_shape=out_shape,
        in_specs=(common.HBM_SPEC,) * len(in_args),
        out_specs=(common.HBM_SPEC,) * len(out_shapes_dtypes),
        cost_estimate=cost_estimate,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )(*in_args)


def _pre_fwd_pipeline_body(
    streams: int,
    embedding: int,
    rms_epsilon: float,
    pre_mapping_epsilon: float,
    fwd_strip_tokens: int,
    fwd_band_width: int,
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
  with jax.named_scope("load_params_and_x"):
    gate_bias = gate_bias_ref[...]
    permutations = permutations_ref[...]
    phi = phi_ref[...]
    x_full = x_ref[...]

  # Full-chunk MXU projection overlapped with hardware bf16 sum-of-squares
  # reduction on VPU (opcode 81), sharing the bfloat16 vector registers of x.
  with jax.named_scope("mxu_proj_and_rms_norm"):
    with jax.named_scope("mxu_gate_proj"):
      raw_proj = jnp.dot(x_full, phi, preferred_element_type=jnp.float32)
    with jax.named_scope("vpu_rms_norm"):
      flattened_f32 = x_full.astype(jnp.float32)
      mean_square = jnp.mean(jnp.square(flattened_f32), axis=-1, keepdims=True)
      inv_rms = jax.lax.rsqrt(mean_square + rms_epsilon)
      logits = raw_proj * inv_rms + gate_bias

  # Fused pre/post sigmoid on lanes 0..2*S-1.
  with jax.named_scope("sigmoid_pre_post_gates"):
    gates = jax.nn.sigmoid(logits[:, : 2 * streams])
    h_pre = gates[:, :streams] + pre_mapping_epsilon
    h_post = 2.0 * gates[:, streams : 2 * streams]

  # Zero-shift 32-lane softmax + padded (32, S^2) permutation projection:
  # Masking lanes 0..2*S-1 with -inf avoids cross-lane shifts (XLU opcode 58).
  with jax.named_scope("softmax_bvn_residual_mix"):
    col_idx = jax.lax.broadcasted_iota(jnp.int32, logits.shape, dimension=1)
    res_logits = jnp.where(col_idx >= 2 * streams, logits, -jnp.inf)
    res_weights = jax.nn.softmax(res_logits, axis=-1)
    residual = jnp.dot(
        res_weights,
        permutations,
        preferred_element_type=jnp.float32,
    )

  with jax.named_scope("store_gates_and_residual"):
    h_pre_ref[...] = h_pre
    h_post_ref[...] = h_post
    residual_ref[...] = residual.astype(residual_ref.dtype)

  # Strip-mined stream collapse over token strips and column bands to bound
  # live float32 vector registers and prevent VMEM register spills.
  with jax.named_scope("strip_mined_stream_collapse"):
    strip_tokens = math.gcd(x_full.shape[0], fwd_strip_tokens)
    num_strips = x_full.shape[0] // strip_tokens
    band_width = math.gcd(embedding, fwd_band_width)

    def _collapse_bands(t_slice, h_strip) -> None:
      with jax.named_scope("stream_collapse_strip"):
        for band_col in range(0, embedding, band_width):
          band_end = band_col + band_width
          acc = h_strip[:, 0:1] * x_ref[t_slice, band_col:band_end]
          for k in range(1, streams):
            col = k * embedding + band_col
            acc = (
                acc
                + h_strip[:, k : k + 1] * x_ref[t_slice, col : col + band_width]
            )
          layer_input_ref[t_slice, band_col:band_end] = acc.astype(jnp.bfloat16)

    if num_strips == 1:
      _collapse_bands(slice(None), h_pre)
    else:
      jax.lax.fori_loop(
          0,
          num_strips,
          lambda strip_idx, _: _collapse_bands(
              pl.dslice(strip_idx * strip_tokens, strip_tokens),
              h_pre_ref[pl.dslice(strip_idx * strip_tokens, strip_tokens), :],
          ),
          None,
      )


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
  with jax.named_scope("post_stream_mix_subtiles"):
    for sub_idx in range(num_subtiles):
      with jax.named_scope(f"subtile_{sub_idx}"):
        s_slice = slice(None)
        if num_subtiles > 1:
          s_slice = slice(sub_idx * sub_tokens, (sub_idx + 1) * sub_tokens)
        with jax.named_scope("load_subtile_inputs"):
          res_sub = residual_ref[s_slice, :].astype(jnp.float32)
          h_post_sub = h_post_ref[s_slice, :]
          l_sub = layer_output_ref[s_slice, :].astype(jnp.float32)
          x_sub = [
              x_ref[s_slice, k * embedding : (k + 1) * embedding].astype(
                  jnp.float32
              )
              for k in range(streams)
          ]
        with jax.named_scope("mix_and_store_streams"):
          for j in range(streams):
            out_j = h_post_sub[:, j : j + 1] * l_sub
            for k in range(streams):
              out_j = (
                  out_j
                  + res_sub[:, j * streams + k : j * streams + k + 1] * x_sub[k]
              )
            output_ref[s_slice, j * embedding : (j + 1) * embedding] = (
                out_j.astype(output_ref.dtype)
            )


def _post_and_pre_fwd_pipeline_body(
    streams: int,
    embedding: int,
    rms_epsilon: float,
    pre_mapping_epsilon: float,
    fwd_strip_tokens: int,
    fwd_band_width: int,
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
  with jax.named_scope("stage1_post_apply_fwd"):
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
  with jax.named_scope("stage2_pre_fwd"):
    _pre_fwd_pipeline_body(
        streams,
        embedding,
        rms_epsilon,
        pre_mapping_epsilon,
        fwd_strip_tokens,
        fwd_band_width,
        x_next_out_ref,
        phi_ref,
        gate_bias_ref,
        permutations_ref,
        layer_input_ref,
        *out_refs,
    )


def pre_fwd(
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    config: Any,
    *,
    rms_epsilon: float | None = None,
    pre_mapping_epsilon: float | None = None,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
):
  """Forward execution for `pre`, returning outputs and backward residuals."""
  streams = permutations.shape[-1]
  embedding = weights.norm_scale.shape[0] // streams
  tokens = x.size // (streams * embedding)
  config, rms_eps, pre_eps, vmem_limit, interpret_flag = (
      common.prepare_kernel_config(
          config,
          tokens,
          streams,
          embedding,
          permutations.shape[0],
          x.dtype,
          rms_epsilon=rms_epsilon,
          pre_mapping_epsilon=pre_mapping_epsilon,
          vmem_limit_bytes=vmem_limit_bytes,
          interpret=interpret,
      )
  )
  common.validate_inputs(x, config.block_size, permutations.shape)
  x_in = x.reshape(tokens, streams * embedding)
  chunk_tokens = common.chunk_tokens(
      tokens, config.block_size, config.fwd_pipeline_blocks
  )
  phi, param_args = _prepare_fwd_params(weights, permutations)
  layer_input, h_pre, h_post, residual = _run_fwd_pipeline(
      "mhc_pre_fwd_fused",
      functools.partial(
          _pre_fwd_pipeline_body,
          streams,
          embedding,
          rms_eps,
          pre_eps,
          config.fwd_strip_tokens,
          config.fwd_band_width,
      ),
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
      vmem_limit_bytes=vmem_limit,
      interpret=interpret_flag,
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


def _prepare_post_fwd(
    layer_output: jax.Array,
    context: common.MHCContext,
    config: Any,
    permutations_shape: tuple[int, ...] | None = None,
    *,
    rms_epsilon: float | None = None,
    pre_mapping_epsilon: float | None = None,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
):
  """Resolves config, validates shapes, and packs post-branch forward args."""
  x, h_post, residual = context.x, context.h_post, context.residual
  embedding = layer_output.shape[-1]
  tokens = layer_output.size // embedding
  streams = h_post.shape[-1]
  num_perms = permutations_shape[0] if permutations_shape else 24
  config, rms_eps, pre_eps, vmem_limit, interpret_flag = (
      common.prepare_kernel_config(
          config,
          tokens,
          streams,
          embedding,
          num_perms,
          x.dtype,
          rms_epsilon=rms_epsilon,
          pre_mapping_epsilon=pre_mapping_epsilon,
          vmem_limit_bytes=vmem_limit_bytes,
          interpret=interpret,
      )
  )
  x_in = x.reshape(tokens, streams * embedding)
  x_check = x_in
  if layer_output.ndim == 3:
    x_check = x.reshape(*layer_output.shape[:2], streams, embedding)
  common.validate_inputs(x_check, config.block_size, permutations_shape)
  chunk = common.chunk_tokens(
      tokens, config.block_size, config.fwd_pipeline_blocks
  )
  sub_tokens = math.gcd(chunk, config.fwd_sub_tokens)
  token_args = (
      x_in,
      layer_output.reshape(tokens, embedding),
      h_post.reshape(tokens, streams),
      residual.reshape(tokens, streams * streams),
  )
  return (
      config,
      rms_eps,
      pre_eps,
      vmem_limit,
      interpret_flag,
      tokens,
      streams,
      embedding,
      chunk,
      sub_tokens,
      token_args,
  )


def post_fwd(
    layer_output: jax.Array,
    context: common.MHCContext,
    config: Any,
    *,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
):
  """Forward execution for `post`, returning output and backward residuals."""
  (
      _,
      _,
      _,
      vmem_limit,
      interpret_flag,
      tokens,
      streams,
      embedding,
      chunk,
      sub_tokens,
      token_args,
  ) = _prepare_post_fwd(
      layer_output,
      context,
      config,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  (output,) = _run_fwd_pipeline(
      "mhc_post_apply_fwd",
      functools.partial(
          _post_apply_fwd_pipeline_body,
          streams,
          embedding,
          sub_tokens,
          chunk // sub_tokens,
      ),
      tokens,
      chunk,
      token_args=token_args,
      out_shapes_dtypes=((token_args[0].shape, jnp.bfloat16),),
      cost_estimate=common.post_cost(tokens, streams, embedding),
      vmem_limit_bytes=vmem_limit,
      interpret=interpret_flag,
  )
  out_shape = token_args[0].shape
  if layer_output.ndim == 3:
    out_shape = (*layer_output.shape[:2], streams, embedding)
  return output.reshape(out_shape), (layer_output, context)


def post_and_pre_fwd(
    layer_output: jax.Array,
    context: common.MHCContext,
    next_weights: common.MhcWeights,
    permutations: jax.Array,
    config: Any,
    *,
    rms_epsilon: float | None = None,
    pre_mapping_epsilon: float | None = None,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
):
  """Forward execution for `post_and_pre`, returning outputs and residuals."""
  (
      config,
      rms_eps,
      pre_eps,
      vmem_limit,
      interpret_flag,
      tokens,
      streams,
      embedding,
      chunk,
      sub_tokens,
      token_args,
  ) = _prepare_post_fwd(
      layer_output,
      context,
      config,
      permutations.shape,
      rms_epsilon=rms_epsilon,
      pre_mapping_epsilon=pre_mapping_epsilon,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  phi, param_args = _prepare_fwd_params(next_weights, permutations)
  next_layer_input, x_next, next_h_pre, next_h_post, next_residual = (
      _run_fwd_pipeline(
          "mhc_post_and_pre_fwd",
          functools.partial(
              _post_and_pre_fwd_pipeline_body,
              streams,
              embedding,
              rms_eps,
              pre_eps,
              config.fwd_strip_tokens,
              config.fwd_band_width,
              sub_tokens,
              chunk // sub_tokens,
          ),
          tokens,
          chunk,
          token_args=token_args,
          param_args=param_args,
          out_shapes_dtypes=(
              ((tokens, embedding), jnp.bfloat16),
              (token_args[0].shape, jnp.bfloat16),
              ((tokens, streams), jnp.float32),
              ((tokens, streams), jnp.float32),
              ((tokens, streams * streams), jnp.bfloat16),
          ),
          cost_estimate=common.post_and_pre_cost(
              tokens, streams, embedding, permutations.shape[0]
          ),
          vmem_limit_bytes=vmem_limit,
          interpret=interpret_flag,
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


# Backwards-compatible aliases kept for lift_kernels/tokamax/converter.py.
_pre_fwd = pre_fwd
_post_fwd = post_fwd
_post_and_pre_fwd = post_and_pre_fwd
