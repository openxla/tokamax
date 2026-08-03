# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.kda import reference
from tokamax._src.ops.experimental.kda.cp_utils import (
    CPContext,
    CPContextArg,
)
from typing_extensions import override


_Config = TypeVar("_Config")
_Key = TypeVar("_Key")
Output: TypeAlias = tuple[jax.Array, jax.Array | None]
Residuals: TypeAlias = Any


def _check_array_rank(x: jax.Array, rank: int, name: str):
  if x.ndim != rank:
    raise ValueError(f"`{name}` must be rank {rank}, got shape {x.shape}.")


def _validate_gate_args(
    *,
    use_gate_in_kernel: bool,
    A_log: jax.Array | None,
    dt_bias: jax.Array | None,
    heads: int,
    key_dim: int,
    safe_gate: bool,
    lower_bound: float | None,
):
  if not use_gate_in_kernel:
    return
  if A_log is None:
    raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True`.")
  if A_log.shape != (heads,):
    raise ValueError(f"`A_log` shape {A_log.shape} must be {(heads,)}.")
  if dt_bias is not None and dt_bias.shape != (heads * key_dim,):
    raise ValueError(
        f"`dt_bias` shape {dt_bias.shape} must be {(heads * key_dim,)}."
    )
  if safe_gate and lower_bound is None:
    raise ValueError(
        "`lower_bound` must be specified when `safe_gate=True` and "
        "`use_gate_in_kernel=True`."
    )
  if lower_bound is not None and not (-5 <= lower_bound < 0):
    raise ValueError(f"`lower_bound` must be in [-5, 0), got {lower_bound}.")


class KimiDeltaAttention(op.Op[Any, Output, Residuals, _Config, _Key]):
  """Kimi Delta Attention reference implementation.

  The public contract is head-first: inputs are `[H, B, T, D]`, varlen
  segment IDs are `[B, T]`, and recurrent state is `[B, N, H, K, V]`.
  """

  supports_symbolic_shapes = False

  @override
  def bind(
      self,
      query: Float[Array, "H B T K"],
      key: Float[Array, "H B T K"],
      value: Float[Array, "H B T V"],
      gate: Float[Array, "H B T K"],
      beta: Float[Array, "H B T"],
      *,
      A_log: Float[Array, "H"] | None = None,
      dt_bias: Float[Array, "H*K"] | None = None,
      scale: float | None = None,
      initial_state: Float[Array, "B N H K V"] | None = None,
      output_final_state: bool = False,
      use_qk_l2norm_in_kernel: bool = False,
      use_gate_in_kernel: bool = False,
      segment_ids: Int[Array, "B T"] | None = None,
      safe_gate: bool = True,
      lower_bound: float | None = None,
      disable_recompute: bool = True,
      cp_context: CPContext | None = None,
      N_max: int | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds KDA arguments and validates the reference contract."""
    _check_array_rank(query, 4, "query")
    heads, batch, seq_len, key_dim = query.shape
    value_dim = value.shape[-1]

    if key.shape != query.shape:
      raise ValueError(
          f"`key` shape {key.shape} must match `query` shape {query.shape}."
      )
    if gate.shape != query.shape:
      raise ValueError(
          f"`gate` shape {gate.shape} must match `query` shape {query.shape}."
      )
    if value.shape != (heads, batch, seq_len, value_dim):
      raise ValueError(
          f"`value` shape {value.shape} must be "
          f"{(heads, batch, seq_len, value_dim)}."
      )
    if beta.shape != (heads, batch, seq_len):
      raise ValueError(
          f"`beta` shape {beta.shape} must be {(heads, batch, seq_len)}."
      )
    if initial_state is not None:
      expected_tail = (heads, key_dim, value_dim)
      if initial_state.ndim != 5 or initial_state.shape[0] != batch:
        raise ValueError(
            "`initial_state` must have shape [B, N, H, K, V]; got "
            f"{initial_state.shape}."
        )
      if initial_state.shape[2:] != expected_tail:
        raise ValueError(
            "`initial_state` trailing dimensions must be "
            f"{expected_tail}; got {initial_state.shape[2:]}."
        )
      state_count = initial_state.shape[1]
      if state_count <= 0:
        raise ValueError(
            "`initial_state` must contain at least one recurrent state."
        )
      if N_max is None:
        N_max = state_count
      elif N_max != state_count:
        raise ValueError(
            "`N_max` must match the `initial_state` segment dimension; got "
            f"N_max={N_max}, N={state_count}."
        )
    if segment_ids is not None and segment_ids.shape != (batch, seq_len):
      raise ValueError(
          f"`segment_ids` shape {segment_ids.shape} must be {(batch, seq_len)}."
      )
    if N_max is not None and N_max <= 0:
      raise ValueError(f"`N_max` must be positive, got {N_max}.")
    if segment_ids is not None and initial_state is None and N_max is None:
      raise ValueError(
          "`N_max` is required when `segment_ids` is provided without "
          "`initial_state`."
      )
    _validate_gate_args(
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        heads=heads,
        key_dim=key_dim,
        safe_gate=safe_gate,
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
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        disable_recompute=disable_recompute,
        cp_context=cp_context,
        N_max=N_max,
        return_residuals=return_residuals,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      query: Float[Array, "H B T K"],
      key: Float[Array, "H B T K"],
      value: Float[Array, "H B T V"],
      gate: Float[Array, "H B T K"],
      beta: Float[Array, "H B T"],
      *,
      A_log: Float[Array, "H"] | None,
      dt_bias: Float[Array, "H*K"] | None,
      scale: float,
      initial_state: Float[Array, "B N H K V"] | None,
      output_final_state: bool,
      use_qk_l2norm_in_kernel: bool,
      use_gate_in_kernel: bool,
      segment_ids: Int[Array, "B T"] | None,
      safe_gate: bool,
      lower_bound: float | None,
      disable_recompute: bool,
      cp_context: CPContextArg,
      N_max: int | None,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[Output, Residuals]:
    """Dispatches to the pure JAX KDA reference implementation."""
    del config, return_residuals, safe_gate, disable_recompute
    output = reference.kimi_delta_attention(
        query,
        key,
        value,
        gate,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        lower_bound=lower_bound,
        cp_context=cp_context,
        N_max=N_max,
    )
    return output, None
