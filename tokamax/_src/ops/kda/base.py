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
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.kda import reference
from tokamax._src.ops.kda.cp_utils import (
    ContextParallelMetadata,
    ContextParallelMetadataArg,
)
from typing_extensions import override

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray

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
    key_dim: int,
    lower_bound: float | None,
):
  if not use_gate_in_kernel:
    return
  if a_log is None:
    raise ValueError("`a_log` must be provided when `use_gate_in_kernel=True`.")
  if a_log.shape != (heads,):
    raise ValueError(f"`a_log` shape {a_log.shape} must be {(heads,)}.")
  if delta_time_bias is not None and delta_time_bias.shape != (heads * key_dim,):
    raise ValueError(
        f"`delta_time_bias` shape {delta_time_bias.shape} must be {(heads * key_dim,)}."
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
      delta_time_bias: Float[Array | AbstractArray, "H*K"] | None = None,
      scale: float | None = None,
      initial_state: Float[Array | AbstractArray, "B N H K V"] | None = None,
      output_final_state: bool = False,
      use_qk_l2norm: bool = False,
      use_gate_in_kernel: bool = False,
      per_channel_gate: bool = True,
      chunk_size: int | None = None,
      segment_ids: Int[Array | AbstractArray, "B T"] | None = None,
      lower_bound: float | None = None,
      context_parallel_metadata: ContextParallelMetadata | None = None,
      max_num_segments: int | None = None,
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
    if not per_channel_gate and use_gate_in_kernel:
      raise ValueError(
          "`use_gate_in_kernel=True` needs a per-channel gate:"
          " `delta_time_bias` is per key channel."
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
        key_dim=key_dim,
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
      delta_time_bias: Float[Array, "H*K"] | None,
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
