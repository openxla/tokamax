# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.kda.cp_utils import (
    CPContext,
    CPContextArg,
)
from typing_extensions import override


_Config = TypeVar("_Config")
_Key = TypeVar("_Key")
Output: TypeAlias = tuple[jax.Array, jax.Array | None]
Residuals: TypeAlias = Any


def _accumulator_dtype(dtype: jax.typing.DTypeLike) -> jnp.dtype:
  dtype = jnp.dtype(dtype)
  return jnp.float64 if dtype == jnp.float64 else jnp.float32


def _check_array_rank(x: jax.Array, rank: int, name: str):
  if x.ndim != rank:
    raise ValueError(f"`{name}` must be rank {rank}, got shape {x.shape}.")


def _l2_normalize(x: jax.Array, acc_dtype: jnp.dtype) -> jax.Array:
  x_f = x.astype(acc_dtype)
  rstd = jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1) + 1e-6)
  return x_f * rstd[..., None]


def _activate_gate(
    g: jax.Array,
    *,
    A_log: jax.Array | None,
    dt_bias: jax.Array | None,
    lower_bound: float | None,
) -> jax.Array:
  heads, _, _, key_dim = g.shape
  if A_log is None:
    raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True`.")
  g_f = g.astype(jnp.float32)
  if dt_bias is not None:
    g_f = g_f + dt_bias.astype(jnp.float32).reshape(heads, 1, 1, key_dim)
  A = jnp.exp(A_log.astype(jnp.float32)).reshape(heads, 1, 1, 1)
  if lower_bound is None:
    return -A * jax.nn.softplus(g_f)
  return lower_bound * jax.nn.sigmoid(A * g_f)


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


def _state_count(
    *,
    segment_ids: jax.Array | None,
    initial_state: jax.Array | None,
    N_max: int | None,
) -> int:
  if initial_state is not None:
    return initial_state.shape[1]
  if segment_ids is None:
    return 1
  if N_max is not None:
    return N_max
  raise ValueError(
      "`N_max` is required when `segment_ids` is provided without "
      "`initial_state`."
  )


class KimiDeltaAttention(op.Op[Any, Output, Residuals, _Config, _Key]):
  """Kimi Delta Attention reference implementation.

  The public contract is head-first: inputs are `[H, B, T, D]`, varlen
  segment IDs are `[B, T]`, and recurrent state is `[B, N, H, K, V]`.
  """

  supports_symbolic_shapes = False

  @override
  def bind(
      self,
      q: Float[Array, "H B T K"],
      k: Float[Array, "H B T K"],
      v: Float[Array, "H B T V"],
      g: Float[Array, "H B T K"],
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
      chunk_size: int = 64,
      N_max: int | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds KDA arguments and validates the reference contract."""
    _check_array_rank(q, 4, "q")
    heads, batch, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]

    if k.shape != q.shape:
      raise ValueError(f"`k` shape {k.shape} must match `q` shape {q.shape}.")
    if g.shape != q.shape:
      raise ValueError(f"`g` shape {g.shape} must match `q` shape {q.shape}.")
    if v.shape != (heads, batch, seq_len, value_dim):
      raise ValueError(
          f"`v` shape {v.shape} must be {(heads, batch, seq_len, value_dim)}."
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
    if chunk_size <= 0:
      raise ValueError(f"`chunk_size` must be positive, got {chunk_size}.")
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
        q=q,
        k=k,
        v=v,
        g=g,
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
        chunk_size=chunk_size,
        N_max=N_max,
        return_residuals=return_residuals,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "H B T K"],
      k: Float[Array, "H B T K"],
      v: Float[Array, "H B T V"],
      g: Float[Array, "H B T K"],
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
      chunk_size: int,
      N_max: int | None,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[Output, Residuals]:
    """Computes KDA with explicit Python loops."""
    del config, return_residuals, safe_gate, disable_recompute, chunk_size

    heads, batch, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]
    acc_dtype = _accumulator_dtype(q.dtype)
    output_dtype = q.dtype
    local_seq_len = seq_len

    if use_gate_in_kernel:
      g = _activate_gate(
          g, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound
      )
    if use_qk_l2norm_in_kernel:
      q_h = _l2_normalize(q, acc_dtype)
      k_h = _l2_normalize(k, acc_dtype)
    else:
      q_h = q.astype(acc_dtype)
      k_h = k.astype(acc_dtype)

    q_h = q_h * scale
    v_h = v.astype(acc_dtype)
    g_h = g.astype(acc_dtype)
    beta_h = beta.astype(acc_dtype)
    cp_enabled = cp_context is not None and getattr(
        cp_context, "is_cp_enabled", False
    )
    if cp_enabled:
      from tokamax._src.ops.experimental.kda.cp_utils import (  # pylint: disable=g-import-not-at-top
          all_gather_into_tensor,
      )

      def gather_time_axis(x, axis: int):
        x_all, _ = all_gather_into_tensor(x, cp_context.axis_name)
        return jnp.concatenate(
            [x_all[i] for i in range(x_all.shape[0])], axis=axis
        )

      q_h = gather_time_axis(q_h, 2)
      k_h = gather_time_axis(k_h, 2)
      v_h = gather_time_axis(v_h, 2)
      g_h = gather_time_axis(g_h, 2)
      beta_h = gather_time_axis(beta_h, 2)
      if segment_ids is not None:
        segment_ids = gather_time_axis(segment_ids, 1)
      seq_len = q_h.shape[2]

    num_states = _state_count(
        segment_ids=segment_ids,
        initial_state=initial_state,
        N_max=N_max,
    )

    states = jnp.zeros(
        (batch, num_states, heads, key_dim, value_dim), dtype=acc_dtype
    )
    if initial_state is not None:
      states = states + initial_state.astype(acc_dtype)

    output_h = jnp.zeros((heads, batch, seq_len, value_dim), dtype=acc_dtype)

    def step_token(h, b, t, carry):
      states, output_h = carry
      if segment_ids is None:
        state_idx = jnp.array(0, dtype=jnp.int32)
        valid = jnp.array(True)
      else:
        seg_id = segment_ids[b, t].astype(jnp.int32)
        state_idx = jnp.clip(seg_id - 1, 0, num_states - 1)
        valid = (seg_id > 0) & (seg_id <= num_states)

      previous_state = states[b, state_idx, h]
      state = previous_state * jnp.exp(g_h[h, b, t])[:, None]
      prediction = k_h[h, b, t] @ state
      residual = v_h[h, b, t] - prediction
      new_state = state + (
          beta_h[h, b, t] * k_h[h, b, t]
      )[:, None] * residual[None, :]
      out_t = q_h[h, b, t] @ new_state
      output_h = output_h.at[h, b, t].set(
          jnp.where(valid, out_t, jnp.zeros_like(out_t))
      )
      updated_state = jnp.where(valid, new_state, previous_state)
      states = states.at[b, state_idx, h].set(updated_state)
      return states, output_h

    def step_head(h, carry):
      states, output_h = carry

      def body_b(b, b_carry):
        def body_t(t, t_carry):
          return step_token(h, b, t, t_carry)

        return jax.lax.fori_loop(0, seq_len, body_t, b_carry)

      states, output_h = jax.lax.fori_loop(
          0, batch, body_b, (states, output_h)
      )
      return states, output_h

    states, output_h = jax.lax.fori_loop(
        0, heads, step_head, (states, output_h)
    )

    if cp_enabled:
      rank = jax.lax.axis_index(cp_context.axis_name)
      output_h = jax.lax.dynamic_slice_in_dim(
          output_h, rank * local_seq_len, local_seq_len, axis=2
      )

    output = output_h.astype(output_dtype)
    final_state = states if output_final_state else None
    return (output, final_state), None
