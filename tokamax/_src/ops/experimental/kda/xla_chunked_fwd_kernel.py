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
"""Pure-JAX/XLA chunked Kimi Delta Attention forward implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member

from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda import common
from tokamax._src.ops.experimental.kda import utils


def _accumulator_dtype(dtype: jax.typing.DTypeLike) -> jnp.dtype:
  return jnp.promote_types(jnp.dtype(dtype), jnp.float32)


def _invert_unit_lower(matrix: jax.Array) -> jax.Array:
  """Inverts ``I + matrix`` for a strictly lower-triangular matrix."""
  size = matrix.shape[-1]
  dtype = matrix.dtype
  eye = jnp.eye(size, dtype=dtype)
  inverse = jnp.broadcast_to(eye, matrix.shape)
  columns = jnp.arange(size)

  def solve_row(row_index, value):
    coefficients = -matrix[..., row_index, :]
    coefficients = jnp.where(columns < row_index, coefficients, 0)
    row = jnp.einsum(
        "...j,...jk->...k",
        coefficients,
        value,
        precision=jax.lax.Precision.HIGHEST,
    )
    row = row.at[..., row_index].set(1)
    return value.at[..., row_index, :].set(row)

  return jax.lax.fori_loop(1, size, solve_row, inverse)


def _chunk_kda_fwd_intra(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    gate_cumsum: jax.Array,
    beta: jax.Array,
    *,
    scale: float,
    chunk_size: int,
    safe_gate: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Builds the intra-chunk matrices and WY representation."""
  heads, batch, seq_len, key_dim = query.shape
  value_dim = value.shape[-1]
  num_chunks = seq_len // chunk_size
  block_size = min(16, chunk_size)
  num_blocks = chunk_size // block_size
  acc_dtype = _accumulator_dtype(query.dtype)

  # Flatten heads, batches and sequence chunks into one batch of independent
  # triangular systems. This keeps the Python loops dependent on chunk_size,
  # rather than on the sequence length.
  def chunked(x):
    return x.reshape(heads, batch, num_chunks, chunk_size, *x.shape[3:]).reshape(
        -1, chunk_size, *x.shape[3:]
    )

  q_chunks = chunked(query).astype(acc_dtype)
  k_chunks = chunked(key).astype(acc_dtype)
  v_chunks = chunked(value).astype(acc_dtype)
  g_chunks = chunked(gate_cumsum).astype(acc_dtype)
  beta_chunks = chunked(beta).astype(acc_dtype)
  flat_chunks = q_chunks.shape[0]

  attention = jnp.zeros((flat_chunks, chunk_size, chunk_size), dtype=acc_dtype)
  key_system = jnp.zeros_like(attention)

  for row_block in range(num_blocks):
    row_start = row_block * block_size
    row_end = row_start + block_size
    q_row = q_chunks[:, row_start:row_end]
    k_row = k_chunks[:, row_start:row_end]
    g_row = g_chunks[:, row_start:row_end]
    beta_row = beta_chunks[:, row_start:row_end]
    reference_index = block_size // 2 if safe_gate else 0
    gate_reference = g_row[:, reference_index : reference_index + 1]

    q_gated = q_row * jnp.exp2(g_row - gate_reference)
    k_gated = k_row * jnp.exp2(g_row - gate_reference)
    k_inverse_gated = k_row * jnp.exp2(gate_reference - g_row)

    diagonal_attention = jnp.einsum(
        "nik,njk->nij",
        q_gated,
        k_inverse_gated,
        precision=jax.lax.Precision.HIGHEST,
    ) * jnp.asarray(scale, dtype=acc_dtype)
    diagonal_system = (
        jnp.einsum(
            "nik,njk->nij",
            k_gated,
            k_inverse_gated,
            precision=jax.lax.Precision.HIGHEST,
        )
        * beta_row[..., None]
    )
    causal = jnp.tril(jnp.ones((block_size, block_size), dtype=jnp.bool_))
    strict_lower = jnp.tril(jnp.ones((block_size, block_size), dtype=jnp.bool_), k=-1)
    diagonal_attention = jnp.where(causal, diagonal_attention, 0)
    diagonal_system = jnp.where(strict_lower, diagonal_system, 0)
    attention = attention.at[:, row_start:row_end, row_start:row_end].set(
        diagonal_attention
    )
    key_system = key_system.at[:, row_start:row_end, row_start:row_end].set(
        diagonal_system
    )

    for column_block in range(row_block):
      column_start = column_block * block_size
      column_end = column_start + block_size
      k_column = k_chunks[:, column_start:column_end]
      g_column = g_chunks[:, column_start:column_end]
      k_column_inverse_gated = k_column * jnp.exp2(gate_reference - g_column)
      off_diagonal_attention = jnp.einsum(
          "nik,njk->nij",
          q_gated,
          k_column_inverse_gated,
          precision=jax.lax.Precision.HIGHEST,
      ) * jnp.asarray(scale, dtype=acc_dtype)
      off_diagonal_system = (
          jnp.einsum(
              "nik,njk->nij",
              k_gated,
              k_column_inverse_gated,
              precision=jax.lax.Precision.HIGHEST,
          )
          * beta_row[..., None]
      )
      attention = attention.at[:, row_start:row_end, column_start:column_end].set(
          off_diagonal_attention
      )
      key_system = key_system.at[:, row_start:row_end, column_start:column_end].set(
          off_diagonal_system
      )

  inverse = _invert_unit_lower(key_system)
  beta_value = v_chunks * beta_chunks[..., None]
  effective_value = jnp.einsum(
      "nij,njv->niv",
      inverse,
      beta_value,
      precision=jax.lax.Precision.HIGHEST,
  )
  beta_gated_key = k_chunks * beta_chunks[..., None] * jnp.exp2(g_chunks)
  effective_key = jnp.einsum(
      "nij,njk->nik",
      inverse,
      beta_gated_key,
      precision=jax.lax.Precision.HIGHEST,
  )
  final_gate = g_chunks[:, -1:, :]
  normalized_key = k_chunks * jnp.exp2(final_gate - g_chunks)

  def restore(x, feature_dim):
    return x.reshape(heads, batch, num_chunks, chunk_size, feature_dim).reshape(
        heads, batch, seq_len, feature_dim
    )

  attention = attention.reshape(
      heads, batch, num_chunks, chunk_size, chunk_size
  ).reshape(heads, batch, seq_len, chunk_size)
  return (
      restore(effective_key, key_dim).astype(key.dtype),
      restore(effective_value, value_dim).astype(value.dtype),
      restore(normalized_key, key_dim).astype(key.dtype),
      attention.astype(query.dtype),
      inverse.reshape(heads, batch, num_chunks, chunk_size, chunk_size)
      .reshape(heads, batch, seq_len, chunk_size)
      .astype(query.dtype),
  )


def _chunk_state_recurrence(
    normalized_key: jax.Array,
    effective_key: jax.Array,
    effective_value: jax.Array,
    gate_cumsum: jax.Array,
    initial_state: jax.Array,
    *,
    aligned_cu_seqlens: jax.Array | None,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Propagates state between chunks and resets it at segment boundaries."""
  heads, batch, seq_len, key_dim = normalized_key.shape
  value_dim = effective_value.shape[-1]
  num_chunks = seq_len // chunk_size
  acc_dtype = _accumulator_dtype(normalized_key.dtype)

  def scan_chunks(x):
    x = x.reshape(heads, batch, num_chunks, chunk_size, *x.shape[3:])
    return jnp.moveaxis(x, 2, 0)

  key_chunks = scan_chunks(normalized_key).astype(acc_dtype)
  write_chunks = scan_chunks(effective_key).astype(acc_dtype)
  value_chunks = scan_chunks(effective_value).astype(acc_dtype)
  gate_chunks = scan_chunks(gate_cumsum).astype(acc_dtype)
  zero_state = jnp.zeros((batch, heads, key_dim, value_dim), dtype=acc_dtype)
  # Preserve initial states for unused/padded segment slots, matching the
  # public recurrent reference contract.
  final_states = initial_state.astype(acc_dtype)
  batch_indices = jnp.arange(batch, dtype=jnp.int32)

  def step(carry, xs):
    state, segment_final_states = carry
    chunk_index, chunk_key, chunk_write, chunk_value, chunk_gate = xs
    if aligned_cu_seqlens is None:
      sequence_ids = jnp.zeros((batch,), dtype=jnp.int32)
      valid = jnp.ones((batch,), dtype=jnp.bool_)
      first = chunk_index == 0
      last = chunk_index == num_chunks - 1
      first = jnp.full((batch,), first)
      last = jnp.full((batch,), last)
    else:
      chunk_start = chunk_index * chunk_size
      sequence_ids = jnp.sum(chunk_start >= aligned_cu_seqlens[:, 1:], axis=1).astype(
          jnp.int32
      )
      sequence_ids = jnp.minimum(sequence_ids, initial_state.shape[1] - 1)
      valid = chunk_start < aligned_cu_seqlens[:, -1]
      sequence_start = jnp.take_along_axis(
          aligned_cu_seqlens[:, :-1], sequence_ids[:, None], axis=1
      )[:, 0]
      sequence_end = jnp.take_along_axis(
          aligned_cu_seqlens[:, 1:], sequence_ids[:, None], axis=1
      )[:, 0]
      first = valid & (chunk_start == sequence_start)
      last = valid & (chunk_start + chunk_size == sequence_end)

    selected_initial_state = initial_state[batch_indices, sequence_ids]
    state = jnp.where(first[:, None, None, None], selected_initial_state, state)
    state = jnp.where(valid[:, None, None, None], state, zero_state)
    state_in = state
    # scan inputs are [H, B, C, D].
    corrected_value = chunk_value - jnp.einsum(
        "hbck,bhkv->hbcv",
        chunk_write,
        state,
        precision=jax.lax.Precision.HIGHEST,
    )
    gate_last = chunk_gate[:, :, -1].transpose(1, 0, 2)
    state = state * jnp.exp2(gate_last[..., None])
    state = state + jnp.einsum(
        "hbck,hbcv->bhkv",
        chunk_key,
        corrected_value,
        precision=jax.lax.Precision.HIGHEST,
    )
    updated_final_states = segment_final_states.at[batch_indices, sequence_ids].set(
        state
    )
    segment_final_states = jnp.where(
        last[:, None, None, None, None],
        updated_final_states,
        segment_final_states,
    )
    return (state, segment_final_states), (state_in, corrected_value)

  chunk_indices = jnp.arange(num_chunks, dtype=jnp.int32)
  (_, final_states), (states, corrected_values) = jax.lax.scan(
      step,
      (zero_state, final_states),
      (chunk_indices, key_chunks, write_chunks, value_chunks, gate_chunks),
  )
  # [NT, B, H, K, V] -> [H, B, NT, K, V]
  states = states.transpose(2, 1, 0, 3, 4)
  # [NT, H, B, C, V] -> [H, B, T, V]
  corrected_values = corrected_values.transpose(1, 2, 0, 3, 4).reshape(
      heads, batch, seq_len, value_dim
  )
  return states, corrected_values, final_states


def _valid_aligned_tokens(
    cu_seqlens: jax.Array,
    aligned_cu_seqlens: jax.Array,
    aligned_seq_len: int,
) -> jax.Array:
  positions = jnp.arange(aligned_seq_len, dtype=jnp.int32)

  def per_batch(original, aligned):
    sequence_ids = jnp.sum(positions[:, None] >= aligned[None, 1:], axis=1)
    sequence_ids = jnp.minimum(sequence_ids, original.shape[0] - 2)
    original_lengths = original[1:] - original[:-1]
    offsets = positions - aligned[sequence_ids]
    return (positions < aligned[-1]) & (offsets < original_lengths[sequence_ids])

  return jax.vmap(per_batch)(cu_seqlens, aligned_cu_seqlens)


@jaxtyping.jaxtyped
def chunk_kda_fwd(
    query: Float[Array, "H B T K"],
    key: Float[Array, "H B T K"],
    value: Float[Array, "H B T V"],
    gate: Float[Array, "H B T K"],
    beta: Float[Array, "H B T"],
    *,
    a_log: Float[Array, "H"] | None,
    delta_time_bias: Float[Array, "H*K"] | None,
    scale: float,
    initial_state: Float[Array, "B N H K V"] | None,
    output_final_state: bool,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    segment_ids: Int[Array, "B T"] | None,
    lower_bound: float | None,
    max_num_segments: int | None,
    chunk_size: int,
    safe_gate: bool,
) -> tuple[Float[Array, "H B T V"], Float[Array, "B N H K V"] | None]:
  """Runs pure-JAX chunked KDA using the Tokamax head-first contract."""
  heads, batch, original_seq_len, key_dim = query.shape
  value_dim = value.shape[-1]
  if chunk_size != 64:
    raise NotImplementedError("`xla_chunked` currently supports chunk_size=64.")
  if segment_ids is None and original_seq_len % chunk_size:
    raise NotImplementedError(
        "`xla_chunked` requires a fixed sequence length divisible by "
        f"chunk_size; got T={original_seq_len}, chunk_size={chunk_size}."
    )

  if use_qk_l2norm:
    query, _ = utils.l2norm_fwd(query)
    key, _ = utils.l2norm_fwd(key)

  cu_seqlens = None
  aligned_cu_seqlens = None
  valid = None
  if segment_ids is not None:
    if max_num_segments is None:
      raise ValueError("`max_num_segments` is required for packed inputs.")
    cu_seqlens, _ = utils.segment_ids_to_cu_seqlens(
        segment_ids,
        initial_state=initial_state,
        max_num_segments=max_num_segments,
    )
    assert cu_seqlens is not None
    (query, key, value, gate), (beta,), aligned_cu_seqlens, _ = utils._align_seqs(  # pylint: disable=protected-access
        (query, key, value, gate),
        (beta,),
        cu_seqlens,
        chunk_size,
    )
    valid = _valid_aligned_tokens(cu_seqlens, aligned_cu_seqlens, query.shape[2])
    valid_4d = valid[None, :, :, None]
    query = jnp.where(valid_4d, query, 0)
    key = jnp.where(valid_4d, key, 0)
    value = jnp.where(valid_4d, value, 0)
    beta = jnp.where(valid[None], beta, 0)
    gate = jnp.where(valid_4d, gate, 0)

  seq_len = query.shape[2]
  if initial_state is None:
    state_count = max_num_segments if segment_ids is not None else 1
    assert state_count is not None
    initial_state = jnp.zeros(
        (batch, state_count, heads, key_dim, value_dim),
        dtype=_accumulator_dtype(query.dtype),
    )

  if use_gate_in_kernel:
    if a_log is None:
      raise ValueError("`a_log` is required when `use_gate_in_kernel=True`.")
    gate_cumsum = common.kda_gate_chunk_cumsum(
        gate,
        a_log,
        chunk_size,
        scale=common.RCP_LN2,
        delta_time_bias=delta_time_bias,
        lower_bound=lower_bound,
        valid_mask=valid,
    )
  else:
    gate_cumsum = common.chunk_local_cumsum_vector(
        gate,
        chunk_size,
        scale=common.RCP_LN2,
    )

  effective_key, effective_value, normalized_key, attention, _ = _chunk_kda_fwd_intra(
      query,
      key,
      value,
      gate_cumsum,
      beta,
      scale=scale,
      chunk_size=chunk_size,
      safe_gate=safe_gate,
  )
  states, corrected_value, final_state = _chunk_state_recurrence(
      normalized_key,
      effective_key,
      effective_value,
      gate_cumsum,
      initial_state,
      aligned_cu_seqlens=aligned_cu_seqlens,
      chunk_size=chunk_size,
  )

  num_chunks = seq_len // chunk_size
  q_chunks = query.reshape(heads, batch, num_chunks, chunk_size, key_dim)
  g_chunks = gate_cumsum.reshape(heads, batch, num_chunks, chunk_size, key_dim)
  q_gated = (q_chunks * jnp.exp2(g_chunks)).astype(query.dtype)
  output_inter = jnp.asarray(scale, jnp.float32) * jnp.einsum(
      "hbnck,hbnkv->hbncv",
      q_gated,
      states.astype(q_gated.dtype),
      precision=jax.lax.Precision.HIGHEST,
  )
  attention = attention.reshape(
      heads, batch, num_chunks, chunk_size, chunk_size
  ).astype(value.dtype)
  corrected_value_chunks = corrected_value.reshape(
      heads, batch, num_chunks, chunk_size, value_dim
  )
  output_intra = jnp.einsum(
      "hbnij,hbnjv->hbniv",
      attention,
      corrected_value_chunks,
      precision=jax.lax.Precision.HIGHEST,
  )
  output = (
      (output_inter + output_intra)
      .reshape(heads, batch, seq_len, value_dim)
      .astype(value.dtype)
  )

  if cu_seqlens is not None:
    assert aligned_cu_seqlens is not None
    output = utils._unalign_output(  # pylint: disable=protected-access
        output,
        cu_seqlens,
        aligned_cu_seqlens,
        original_seq_len,
    )
  return output, final_state if output_final_state else None


__all__ = ["chunk_kda_fwd"]
