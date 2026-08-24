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
"""Pure JAX reference implementation of Kimi Delta Attention."""

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.kda.cp_utils import ContextParallelMetadataArg


def _accumulator_dtype(dtype: jax.typing.DTypeLike) -> jnp.dtype:
  return jnp.promote_types(jnp.dtype(dtype), jnp.float32)


def _l2_normalize(x: jax.Array, acc_dtype: jnp.dtype) -> jax.Array:
  x_f = x.astype(acc_dtype)
  rstd = jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1) + 1e-6)
  return x_f * rstd[..., None]


def _activate_gate(
    g: jax.Array,
    *,
    a_log: jax.Array | None,
    delta_time_bias: jax.Array | None,
    lower_bound: float | None,
) -> jax.Array:
  heads, _, _, key_dim = g.shape
  if a_log is None:
    raise ValueError("`a_log` must be provided when `use_gate_in_kernel=True`.")
  g_f = g.astype(jnp.float32)
  if delta_time_bias is not None:
    g_f = g_f + delta_time_bias.astype(jnp.float32).reshape(heads, 1, 1, key_dim)
  A = jnp.exp(a_log.astype(jnp.float32)).reshape(heads, 1, 1, 1)
  if lower_bound is None:
    return -A * jax.nn.softplus(g_f)
  return lower_bound * jax.nn.sigmoid(A * g_f)


def _state_count(
    *,
    segment_ids: jax.Array | None,
    initial_state: jax.Array | None,
    max_num_segments: int | None,
) -> int:
  if initial_state is not None:
    return initial_state.shape[1]
  if segment_ids is None:
    return 1
  if max_num_segments is not None:
    return max_num_segments
  raise ValueError(
      "`max_num_segments` is required when `segment_ids` is provided without "
      "`initial_state`."
  )


def kimi_delta_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    gate: jax.Array,
    beta: jax.Array,
    *,
    a_log: jax.Array | None,
    delta_time_bias: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    output_final_state: bool,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    segment_ids: jax.Array | None,
    lower_bound: float | None,
    context_parallel_metadata: ContextParallelMetadataArg,
    max_num_segments: int | None,
) -> tuple[jax.Array, jax.Array | None]:
  """Computes KDA with an explicit token-by-token JAX recurrence."""
  q, k, v, g = query, key, value, gate
  heads, batch, seq_len, key_dim = q.shape
  value_dim = v.shape[-1]
  acc_dtype = _accumulator_dtype(q.dtype)
  output_dtype = q.dtype
  local_seq_len = seq_len

  if use_gate_in_kernel:
    g = _activate_gate(
        g, a_log=a_log, delta_time_bias=delta_time_bias, lower_bound=lower_bound
    )
  if use_qk_l2norm:
    q_h = _l2_normalize(q, acc_dtype)
    k_h = _l2_normalize(k, acc_dtype)
  else:
    q_h = q.astype(acc_dtype)
    k_h = k.astype(acc_dtype)

  q_h = q_h * scale
  v_h = v.astype(acc_dtype)
  g_h = g.astype(acc_dtype)
  beta_h = beta.astype(acc_dtype)
  cp_enabled = context_parallel_metadata is not None and getattr(
      context_parallel_metadata, "is_cp_enabled", False
  )
  if cp_enabled:
    assert context_parallel_metadata is not None
    axis_name = context_parallel_metadata.axis_name
    from tokamax._src.ops.experimental.kda.cp_utils import (  # pylint: disable=g-import-not-at-top
        all_gather_into_tensor,
    )

    def gather_time_axis(x, axis: int):
      x_all, _ = all_gather_into_tensor(x, axis_name)
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
      max_num_segments=max_num_segments,
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

  if cp_enabled:
    # Keep the native token-by-token recurrence, but rematerialize each
    # head's large state intermediates during backward. Dot outputs are much
    # smaller than the K x V states and are retained to limit recomputation.
    step_head = jax.checkpoint(
        step_head,
        prevent_cse=False,
        policy=jax.checkpoint_policies.dots_saveable,
    )

  states, output_h = jax.lax.fori_loop(
      0, heads, step_head, (states, output_h)
  )

  if cp_enabled:
    assert context_parallel_metadata is not None
    rank = jax.lax.axis_index(context_parallel_metadata.axis_name)
    output_h = jax.lax.dynamic_slice_in_dim(
        output_h, rank * local_seq_len, local_seq_len, axis=2
    )

  output = output_h.astype(output_dtype)
  final_state = states if output_final_state else None
  return output, final_state
