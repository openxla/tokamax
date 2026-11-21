# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Experimental Kimi Delta Attention base implementation."""

from typing import Any, TypeAlias, TypeVar

import jax
from jax.experimental import checkify
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.kda import reference
from tokamax._src.ops.kda.cp_utils import (
    ContextParallelMetadata,
    ContextParallelMetadataArg,
)
from typing_extensions import override

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray

# Dimensions.
#   B: batch size
#   H: number of heads
#   T: sequence length
#   N: number of sequences per batch element (varlen / packed inputs)
#   K: key/query head dimension
#   V: value head dimension
#   GW: gate channel width (1 for scalar GDN gate, K for per-channel KDA gate)

_Config = TypeVar("_Config")
_Key = TypeVar("_Key")
Output: TypeAlias = tuple[jax.Array, jax.Array | None]
Residuals: TypeAlias = Any


def _validate_beta(beta: jax.Array) -> None:
  """Checks that the post-activation delta-rule rate is in [0, 1]."""
  valid = jnp.all((beta >= 0) & (beta <= 1))
  message = "`beta` must contain finite values in [0, 1]."
  checkify.debug_check(valid, message)
  if not isinstance(beta, jax.core.Tracer):
    checkify.check(valid, message)


def _validate_gate_args(
    *,
    use_gate_in_kernel: bool,
    a_log: jax.Array | AbstractArray | None,
    delta_time_bias: jax.Array | AbstractArray | None,
    heads: int,
    gate_dim: int,
    lower_bound: float | None,
):
  if not use_gate_in_kernel:
    return
  if a_log is None:
    raise ValueError("`a_log` must be provided when `use_gate_in_kernel=True`.")
  if a_log.shape != (heads,):
    raise ValueError(f"`a_log` shape {a_log.shape} must be {(heads,)}.")
  if delta_time_bias is not None and delta_time_bias.shape != (
      heads * gate_dim,
  ):
    raise ValueError(
        f"`delta_time_bias` shape {delta_time_bias.shape} must be"
        f" {(heads * gate_dim,)}."
    )
  if lower_bound is not None and not (-5 <= lower_bound < 0):
    raise ValueError(f"`lower_bound` must be in [-5, 0), got {lower_bound}.")


class KimiDeltaAttention(op.Op[Any, Output, Residuals, _Config, _Key]):
  """Kimi Delta Attention reference implementation.

  The public contract is head-first: inputs are `[H, B, T, D]`, varlen
  segment IDs are `[B, T]`, and recurrent state is `[B, N, H, K, V]`.
  """

  supports_symbolic_shapes = False

  @jaxtyping.jaxtyped
  @override
  def bind(
      self,
      query: Float[Array | AbstractArray, "H B T K"],
      key: Float[Array | AbstractArray, "H B T K"],
      value: Float[Array | AbstractArray, "H B T V"],
      gate: Float[Array | AbstractArray, "H B T GW"],
      beta: Float[Array | AbstractArray, "H B T"],
      *,
      a_log: Float[Array | AbstractArray, "H"] | None = None,
      delta_time_bias: Float[Array | AbstractArray, "H*GW"] | None = None,
      scale: float | None = None,
      initial_state: Float[Array | AbstractArray, "B N H K V"] | None = None,
      output_final_state: bool = False,
      use_qk_l2norm: bool = False,
      use_gate_in_kernel: bool = False,
      per_channel_gate: bool = True,
      chunk_size: int | None = None,
      segment_ids: Int[Array | np.ndarray | AbstractArray, "B T"] | None = None,
      lower_bound: float | None = None,
      context_parallel_metadata: ContextParallelMetadata | None = None,
      max_num_segments: int | None = None,
      residual_checkpoint_name: str | None = None,
      return_residuals: bool = False,
      conv_weight_q: Float[Array | AbstractArray, "H_Q W K"] | None = None,
      conv_weight_k: Float[Array | AbstractArray, "H_K W K"] | None = None,
      conv_weight_v: Float[Array | AbstractArray, "H W V"] | None = None,
      conv_bias_q: Float[Array | AbstractArray, "H_Q K"] | None = None,
      conv_bias_k: Float[Array | AbstractArray, "H_K K"] | None = None,
      conv_bias_v: Float[Array | AbstractArray, "H V"] | None = None,
      use_conv1d_in_kernel: bool = False,
  ) -> op.BoundArguments:
    """Binds KDA arguments and validates semantic constraints."""
    heads, _, _, key_dim = query.shape

    if initial_state is not None:
      state_count = initial_state.shape[1]
      if state_count <= 0:
        raise ValueError(
            "`initial_state` must contain at least one recurrent state."
        )
      if max_num_segments is None:
        max_num_segments = state_count
      elif max_num_segments != state_count:
        raise ValueError(
            "`max_num_segments` must match the `initial_state` segment "
            "dimension; got "
            f"max_num_segments={max_num_segments}, "
            f"num_segments={state_count}."
        )
    if max_num_segments is not None and max_num_segments <= 0:
      raise ValueError(
          f"`max_num_segments` must be positive, got {max_num_segments}."
      )
    if (
        segment_ids is not None
        and initial_state is None
        and max_num_segments is None
    ):
      raise ValueError(
          "`max_num_segments` is required when `segment_ids` is provided "
          "without `initial_state`."
      )
    gate_width = gate.shape[-1]
    if per_channel_gate:
      if gate_width != key_dim:
        raise ValueError(
            f"`gate` last dimension {gate_width} must equal the key dimension"
            f" {key_dim} when `per_channel_gate=True`."
        )
    elif gate_width != 1:
      raise ValueError(
          f"`gate` last dimension {gate_width} must be 1 (one value per head"
          " and token) when `per_channel_gate=False`."
      )
    if use_conv1d_in_kernel and (
        conv_weight_q is None or conv_weight_k is None or conv_weight_v is None
    ):
      raise ValueError(
          "`conv_weight_q`, `conv_weight_k` and `conv_weight_v` must all be"
          " provided when `use_conv1d_in_kernel=True`."
      )

    _validate_gate_args(
        use_gate_in_kernel=use_gate_in_kernel,
        a_log=a_log,
        delta_time_bias=delta_time_bias,
        heads=heads,
        gate_dim=gate_width,
        lower_bound=lower_bound,
    )

    if scale is None:
      scale = key_dim**-0.5

    return super().bind(
        query=query,
        key=key,
        value=value,
        gate=gate,
        beta=beta,
        a_log=a_log,
        delta_time_bias=delta_time_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        per_channel_gate=per_channel_gate,
        chunk_size=chunk_size,
        segment_ids=segment_ids,
        lower_bound=lower_bound,
        context_parallel_metadata=context_parallel_metadata,
        max_num_segments=max_num_segments,
        residual_checkpoint_name=residual_checkpoint_name,
        return_residuals=return_residuals,
        conv_weight_q=conv_weight_q,
        conv_weight_k=conv_weight_k,
        conv_weight_v=conv_weight_v,
        conv_bias_q=conv_bias_q,
        conv_bias_k=conv_bias_k,
        conv_bias_v=conv_bias_v,
        use_conv1d_in_kernel=use_conv1d_in_kernel,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      query: Float[Array, "H B T K"],
      key: Float[Array, "H B T K"],
      value: Float[Array, "H B T V"],
      gate: Float[Array, "H B T GW"],
      beta: Float[Array, "H B T"],
      *,
      a_log: Float[Array, "H"] | None,
      delta_time_bias: Float[Array, "H*GW"] | None,
      scale: float,
      initial_state: Float[Array, "B N H K V"] | None,
      output_final_state: bool,
      use_qk_l2norm: bool,
      use_gate_in_kernel: bool,
      per_channel_gate: bool,
      chunk_size: int | None,
      segment_ids: Int[Array, "B T"] | None,
      lower_bound: float | None,
      context_parallel_metadata: ContextParallelMetadataArg,
      max_num_segments: int | None,
      residual_checkpoint_name: str | None = None,
      return_residuals: bool,
      conv_weight_q: Float[Array, "H_Q W K"] | None = None,
      conv_weight_k: Float[Array, "H_K W K"] | None = None,
      conv_weight_v: Float[Array, "H W V"] | None = None,
      conv_bias_q: Float[Array, "H_Q K"] | None = None,
      conv_bias_k: Float[Array, "H_K K"] | None = None,
      conv_bias_v: Float[Array, "H V"] | None = None,
      use_conv1d_in_kernel: bool = False,
      config: _Config,
  ) -> tuple[Output, Residuals]:
    """Dispatches to the pure JAX KDA reference implementation."""
    del config, return_residuals, per_channel_gate, chunk_size
    del residual_checkpoint_name
    _validate_beta(beta)
    output = reference.kimi_delta_attention(
        query,
        key,
        value,
        gate,
        beta,
        a_log=a_log,
        delta_time_bias=delta_time_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        lower_bound=lower_bound,
        context_parallel_metadata=context_parallel_metadata,
        max_num_segments=max_num_segments,
        conv_weight_q=conv_weight_q,
        conv_weight_k=conv_weight_k,
        conv_weight_v=conv_weight_v,
        conv_bias_q=conv_bias_q,
        conv_bias_k=conv_bias_k,
        conv_bias_v=conv_bias_v,
        use_conv1d_in_kernel=use_conv1d_in_kernel,
    )
    return output, None


class GatedDeltaNet(op.Op[Any, Output, Residuals, _Config, _Key]):
  """Gated Delta Net reference implementation.

  Uses natural token-major layouts (`[B, T, H_QK, K]` / `[B, T, H_V, V]` or
  fused `[B, T, 2 * H_QK * K + H_V * V]`) and supports grouped query/key heads
  (`H_V % H_QK == 0`).
  """

  supports_symbolic_shapes = False

  @jaxtyping.jaxtyped
  @override
  def bind(
      self,
      query: Float[Array | AbstractArray, "B T H_QK K"] | None = None,
      key: Float[Array | AbstractArray, "B T H_QK K"] | None = None,
      value: Float[Array | AbstractArray, "B T H_V V"] | None = None,
      gate: Float[Array | AbstractArray, "B T H_V"] | None = None,
      beta: Float[Array | AbstractArray, "B T H_V"] | None = None,
      *,
      qkv: Float[Array | AbstractArray, "B T C_QKV"] | None = None,
      num_k_heads: int | None = None,
      num_v_heads: int | None = None,
      head_k_dim: int | None = None,
      head_v_dim: int | None = None,
      conv_weight: Float[Array | AbstractArray, "W C_QKV"] | None = None,
      conv_bias: Float[Array | AbstractArray, "C_QKV"] | None = None,
      a_log: Float[Array | AbstractArray, "H_V"] | None = None,
      delta_time_bias: Float[Array | AbstractArray, "H_V"] | None = None,
      scale: float | None = None,
      initial_state: Float[Array | AbstractArray, "B H_V K V"] | None = None,
      output_final_state: bool = False,
      use_qk_l2norm: bool = True,
      use_gate_in_kernel: bool = False,
      use_beta_sigmoid_in_kernel: bool = False,
      chunk_size: int | None = None,
      segment_ids: Int[Array | np.ndarray | AbstractArray, "B T"] | None = None,
      context_parallel_metadata: ContextParallelMetadata | None = None,
      residual_checkpoint_name: str | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds GDN arguments and validates semantic constraints."""
    if gate is None or beta is None:
      raise ValueError("`gate` and `beta` must both be provided.")
    if gate.ndim != 3:
      raise ValueError(
          f"`gate` must be rank 3 `[B, T, H_V]`; got {gate.shape}."
      )

    has_qkv = qkv is not None
    has_split = query is not None or key is not None or value is not None
    if has_qkv and has_split:
      raise ValueError(
          "Provide either `qkv` or `(query, key, value)`, not both."
      )
    if not has_qkv and (query is None or key is None or value is None):
      raise ValueError(
          "`query`, `key`, and `value` must all be provided when `qkv` is None."
      )

    if has_qkv:
      assert qkv is not None
      if qkv.ndim != 3:
        raise ValueError(
            f"`qkv` must be rank 3 `[B, T, C_QKV]`; got {qkv.shape}."
        )
      batch, seq_len, c_qkv = qkv.shape
      if num_k_heads is None or head_k_dim is None:
        raise ValueError(
            "`num_k_heads` and `head_k_dim` must be provided when `qkv` is"
            " used."
        )
      if num_v_heads is None:
        num_v_heads = gate.shape[-1]
      elif num_v_heads != gate.shape[-1]:
        raise ValueError(
            f"`num_v_heads` ({num_v_heads}) must match `gate.shape[-1]`"
            f" ({gate.shape[-1]})."
        )
      rem_v = c_qkv - 2 * num_k_heads * head_k_dim
      if head_v_dim is None:
        if num_v_heads <= 0 or rem_v <= 0 or rem_v % num_v_heads != 0:
          raise ValueError(
              f"Cannot infer positive `head_v_dim` from qkv.shape={qkv.shape},"
              f" num_k_heads={num_k_heads}, head_k_dim={head_k_dim},"
              f" num_v_heads={num_v_heads}."
          )
        head_v_dim = rem_v // num_v_heads
      expected_c = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
      if c_qkv != expected_c:
        raise ValueError(
            f"`qkv.shape[-1]` ({c_qkv}) must equal"
            f" 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim"
            f" ({expected_c})."
        )
    else:
      assert query is not None and key is not None and value is not None
      if query.ndim != 4:
        raise ValueError(
            f"`query` must be rank 4 `[B, T, H_QK, K]`; got {query.shape}."
        )
      if key.shape != query.shape:
        raise ValueError(
            f"`key.shape` {key.shape} must match `query.shape` {query.shape}."
        )
      batch, seq_len, q_heads, k_dim = query.shape
      if value.ndim != 4 or value.shape[:2] != (batch, seq_len):
        raise ValueError(
            f"`value.shape` {value.shape} must have leading `(B, T)`"
            f" {(batch, seq_len)}."
        )
      v_heads = value.shape[2]
      v_dim = value.shape[3]
      if num_k_heads is not None and num_k_heads != q_heads:
        raise ValueError(
            f"`num_k_heads` ({num_k_heads}) must match `query.shape[2]`"
            f" ({q_heads})."
        )
      if num_v_heads is not None and num_v_heads != v_heads:
        raise ValueError(
            f"`num_v_heads` ({num_v_heads}) must match `value.shape[2]`"
            f" ({v_heads})."
        )
      if head_k_dim is not None and head_k_dim != k_dim:
        raise ValueError(
            f"`head_k_dim` ({head_k_dim}) must match `query.shape[3]`"
            f" ({k_dim})."
        )
      if head_v_dim is not None and head_v_dim != v_dim:
        raise ValueError(
            f"`head_v_dim` ({head_v_dim}) must match `value.shape[3]`"
            f" ({v_dim})."
        )
      num_k_heads = q_heads
      num_v_heads = v_heads
      head_k_dim = k_dim
      head_v_dim = v_dim
      expected_c = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

    if head_k_dim <= 0 or head_v_dim <= 0:
      raise ValueError(
          "`head_k_dim` and `head_v_dim` must be positive; got"
          f" head_k_dim={head_k_dim}, head_v_dim={head_v_dim}."
      )
    if num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads != 0:
      raise ValueError(
          f"`num_v_heads` ({num_v_heads}) must be a positive multiple of"
          f" `num_k_heads` ({num_k_heads})."
      )
    if gate.shape != (batch, seq_len, num_v_heads):
      raise ValueError(
          f"`gate.shape` {gate.shape} must be {(batch, seq_len, num_v_heads)}."
      )
    if beta.shape != (batch, seq_len, num_v_heads):
      raise ValueError(
          f"`beta.shape` {beta.shape} must be {(batch, seq_len, num_v_heads)}."
      )
    if conv_bias is not None and conv_weight is None:
      raise ValueError(
          "`conv_weight` must be provided when `conv_bias` is provided."
      )
    if conv_weight is not None and (
        conv_weight.ndim != 2
        or conv_weight.shape[0] < 1
        or conv_weight.shape[1] != expected_c
    ):
      raise ValueError(
          "`conv_weight` must have shape `[W, C_QKV]` with `W >= 1` and"
          f" `C_QKV == {expected_c}`; got {conv_weight.shape}."
      )
    if conv_bias is not None and (
        conv_bias.ndim != 1 or conv_bias.shape != (expected_c,)
    ):
      raise ValueError(
          f"`conv_bias` must have shape {(expected_c,)}; got"
          f" {conv_bias.shape}."
      )
    if segment_ids is not None and (
        segment_ids.ndim != 2 or segment_ids.shape != (batch, seq_len)
    ):
      raise ValueError(
          f"`segment_ids` must have shape {(batch, seq_len)}; got"
          f" {segment_ids.shape}."
      )
    if initial_state is not None and initial_state.shape != (
        batch,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    ):
      raise ValueError(
          f"`initial_state.shape` {initial_state.shape} must be"
          f" {(batch, num_v_heads, head_k_dim, head_v_dim)}."
      )
    if not use_gate_in_kernel and (
        a_log is not None or delta_time_bias is not None
    ):
      raise ValueError(
          "`use_gate_in_kernel=True` is required when `a_log` or"
          " `delta_time_bias` is provided."
      )
    _validate_gate_args(
        use_gate_in_kernel=use_gate_in_kernel,
        a_log=a_log,
        delta_time_bias=delta_time_bias,
        heads=num_v_heads,
        gate_dim=1,
        lower_bound=None,
    )

    if scale is None:
      scale = head_k_dim**-0.5

    return super().bind(
        query=query,
        key=key,
        value=value,
        gate=gate,
        beta=beta,
        qkv=qkv,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        delta_time_bias=delta_time_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        chunk_size=chunk_size,
        segment_ids=segment_ids,
        context_parallel_metadata=context_parallel_metadata,
        residual_checkpoint_name=residual_checkpoint_name,
        return_residuals=return_residuals,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      query: Float[Array, "B T H_QK K"] | None,
      key: Float[Array, "B T H_QK K"] | None,
      value: Float[Array, "B T H_V V"] | None,
      gate: Float[Array, "B T H_V"],
      beta: Float[Array, "B T H_V"],
      *,
      qkv: Float[Array, "B T C_QKV"] | None,
      num_k_heads: int,
      num_v_heads: int,
      head_k_dim: int,
      head_v_dim: int,
      conv_weight: Float[Array, "W C_QKV"] | None,
      conv_bias: Float[Array, "C_QKV"] | None,
      a_log: Float[Array, "H_V"] | None,
      delta_time_bias: Float[Array, "H_V"] | None,
      scale: float,
      initial_state: Float[Array, "B H_V K V"] | None,
      output_final_state: bool,
      use_qk_l2norm: bool,
      use_gate_in_kernel: bool,
      use_beta_sigmoid_in_kernel: bool,
      chunk_size: int | None,
      segment_ids: Int[Array, "B T"] | None,
      context_parallel_metadata: ContextParallelMetadataArg,
      residual_checkpoint_name: str | None = None,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[Output, Residuals]:
    """Dispatches to the pure JAX GDN reference implementation."""
    del config, return_residuals, chunk_size
    del residual_checkpoint_name
    if not use_beta_sigmoid_in_kernel:
      _validate_beta(beta)
    output = reference.gated_delta_net(
        query,
        key,
        value,
        gate,
        beta,
        qkv=qkv,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        delta_time_bias=delta_time_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        segment_ids=segment_ids,
        context_parallel_metadata=context_parallel_metadata,
    )
    return output, None
