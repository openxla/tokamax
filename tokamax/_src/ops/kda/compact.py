# Copyright 2026 Ant Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Tokamax adapter for native original-token KDA forward and residuals."""

import dataclasses

import jax
from jax.experimental import checkify
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from tokamax._src.ops.kda import compact_forward


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class CompactResiduals:
  """Original-token inputs and the matching compact chunk-domain tape."""

  q: Float[Array, "B T H K_PAD"]
  k: Float[Array, "B T H K_PAD"]
  v: Float[Array, "B T H V_PAD"]
  gate: Float[Array, "B T H K_PAD"]
  beta: Float[Array, "H B T"]
  a_log: Float[Array, "H"] | None
  bias: Float[Array, "H*K_PAD"] | None
  segment_ids: Int[Array, "B T"]
  cu_seqlens: Int[Array, "B N_CU"]
  permutation: Int[Array, "B T"] | None
  inverse_permutation: Int[Array, "B T"] | None
  ordered: jax.Array | None
  g_cumsum: Float[Array, "H B T K_PAD"]
  aqk: Float[Array, "H B T BT"]
  akk: Float[Array, "H B T BT"]
  h: Float[Array, "H B NT K_PAD V_PAD"]


def supported(
    query,
    value,
    *,
    segment_ids,
    initial_state,
    output_final_state,
    use_qk_l2norm,
    context_parallel_metadata,
    config,
):
  """Match the pinned source's saved-state, non-CP training profile."""
  cp = context_parallel_metadata
  return (
      config.compact_forward
      and query.dtype == jnp.bfloat16
      and value.dtype == jnp.bfloat16
      and config.chunk_size == 64
      and query.shape[2] % 64 == 0
      and segment_ids is not None
      and initial_state is None
      and not output_final_state
      and not use_qk_l2norm
      and not config.rematerialize_for_backward
      and config.safe_gate is not False
      and (cp is None or not cp.is_cp_enabled)
      and 0 < query.shape[-1] <= 256
      and 0 < value.shape[-1] <= 256
  )


def _canonical_segments(segment_ids):
  """Renumber already grouped labels without allocating by label magnitude."""
  previous = jnp.pad(segment_ids[:, :-1], ((0, 0), (1, 0)))
  valid = segment_ids > 0
  starts = valid & (segment_ids != previous)
  return jnp.where(valid, jnp.cumsum(starts, axis=1, dtype=jnp.int32), 0)


def _packing(segment_ids):
  """Keep packed inputs copy-free; group interleaved labels stably otherwise."""
  tokens = segment_ids.shape[1]
  identity = jnp.broadcast_to(
      jnp.arange(tokens, dtype=jnp.int32), segment_ids.shape
  )
  keys = jnp.where(
      segment_ids > 0,
      segment_ids.astype(jnp.uint32),
      jnp.asarray(jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32),
  )
  ordered = jnp.all(keys[:, 1:] >= keys[:, :-1])
  permutation = jax.lax.cond(
      ordered,
      lambda _: identity,
      lambda _: jnp.argsort(keys, axis=1, stable=True).astype(jnp.int32),
      None,
  )
  inverse = jax.lax.cond(
      ordered,
      lambda _: identity,
      lambda _: jnp.argsort(permutation, axis=1, stable=True).astype(jnp.int32),
      None,
  )
  labels = jnp.take_along_axis(segment_ids, permutation, axis=1)
  return _canonical_segments(labels), permutation, inverse, ordered


def _take_tokens(value, indices):
  return jnp.take_along_axis(
      value, indices.reshape(indices.shape + (1,) * (value.ndim - 2)), axis=1
  )


def _segment_boundaries(segment_ids, capacity):
  # At most T non-empty runs exist. Bound the physical table by local tokens,
  # independent of the caller's potentially huge logical segment capacity.
  counts = jax.vmap(lambda row: jnp.bincount(row, length=capacity + 1))(
      segment_ids
  )
  counts = counts.at[:, 0].set(0)
  return jnp.cumsum(counts, axis=1, dtype=jnp.int32)


def _check_sorted_segments(segment_ids, capacity):
  valid = segment_ids > 0
  lengths = jnp.sum(valid, axis=1)
  condition = (
      jnp.all(segment_ids >= 0)
      & jnp.all(
          valid
          == (jnp.arange(segment_ids.shape[1])[None, :] < lengths[:, None])
      )
      & jnp.all(
          (segment_ids[:, 1:] >= segment_ids[:, :-1])
          | (segment_ids[:, 1:] == 0)
      )
      & jnp.all(segment_ids <= capacity)
  )
  message = "compact_sorted_segments requires grouped positive labels and trailing zero padding"
  checkify.debug_check(condition, message)
  if not isinstance(segment_ids, jax.core.Tracer):
    checkify.check(condition, message)


def forward(
    query,
    key,
    value,
    gate,
    beta,
    *,
    a_log,
    delta_time_bias,
    scale,
    segment_ids,
    lower_bound,
    use_gate_in_kernel,
    return_residuals,
    sorted_segments,
    max_num_segments,
):
  """Compute [H,B,T,V] output and a T/64 compact tape without token alignment."""
  heads, _, tokens, key_dim = query.shape
  value_dim = value.shape[-1]
  kp, vp = ((key_dim + 127) // 128) * 128, ((value_dim + 127) // 128) * 128

  def batch_first(x, width):
    return jnp.pad(
        x.transpose(1, 2, 0, 3),
        ((0, 0), (0, 0), (0, 0), (0, width - x.shape[-1])),
    )

  q, k, g = (batch_first(x, kp) for x in (query, key, gate))
  v = batch_first(value, vp)
  bias = delta_time_bias
  if bias is not None and kp != key_dim:
    bias = jnp.pad(
        bias.reshape(heads, key_dim), ((0, 0), (0, kp - key_dim))
    ).reshape(-1)
  capacity = min(max_num_segments, tokens)
  if sorted_segments:
    _check_sorted_segments(segment_ids, max_num_segments)
    ids = (
        segment_ids
        if max_num_segments <= tokens
        else _canonical_segments(segment_ids)
    )
    permutation = inverse = ordered = None
  else:
    ids, permutation, inverse, ordered = _packing(segment_ids)

  def run(reorder):
    args = (q, k, v, g, beta.transpose(1, 2, 0))
    if reorder:
      args = tuple(_take_tokens(x, permutation) for x in args)
    result = compact_forward._chunk_kda_fwd_native_segids_impl(
        *args,
        ids,
        initial_state=None,
        output_final_state=False,
        scale=scale,
        chunk_size=64,
        store_h=return_residuals,
        disable_recompute=return_residuals,
        only_fwd=not return_residuals,
        safe_gate=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=a_log,
        dt_bias=bias,
        lower_bound=lower_bound,
        N_max=capacity,
        batch_first=True,
    )
    output = _take_tokens(result[0], inverse) if reorder else result[0]
    return output, result[2], result[3], result[4], result[10]

  result = (
      run(False)
      if sorted_segments
      else jax.lax.cond(
          ordered, lambda _: run(False), lambda _: run(True), None
      )
  )
  output = result[0][..., :value_dim].transpose(2, 0, 1, 3).astype(query.dtype)
  residuals = None
  if return_residuals:
    residuals = CompactResiduals(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        bias,
        ids,
        _segment_boundaries(ids, capacity),
        permutation,
        inverse,
        ordered,
        result[1],
        result[2],
        result[3],
        result[4],
    )
  return (output, None), residuals


def backward(
    residuals,
    dout,
    *,
    query,
    value,
    gate,
    scale,
    lower_bound,
    use_gate_in_kernel,
    a_log,
    delta_time_bias,
):
  """Consume the original-token tape with M5 and return public gradients."""
  from tokamax._src.ops.kda.compact_backward import chunk_kda_bwd_fusion
  from tokamax._src.ops.kda.compact_gate import kda_gate_bwd

  r = residuals
  heads, _, _, key_dim = query.shape
  value_dim = value.shape[-1]
  do = jnp.pad(
      dout[0].transpose(1, 2, 0, 3),
      ((0, 0), (0, 0), (0, 0), (0, r.v.shape[-1] - value_dim)),
  )
  fused_gate = (
      use_gate_in_kernel
      and lower_bound == -5.0
      and r.a_log is not None
      and r.bias is not None
  )

  def run(reorder):
    q, k, v, raw_gate, beta, dy = (
        r.q,
        r.k,
        r.v,
        r.gate,
        r.beta.transpose(1, 2, 0),
        do,
    )
    if reorder:
      q, k, v, raw_gate, beta, dy = (
          _take_tokens(x, r.permutation) for x in (q, k, v, raw_gate, beta, dy)
      )
    values = chunk_kda_bwd_fusion(
        q=q,
        k=k,
        v=v,
        beta=beta.transpose(2, 0, 1),
        Aqk=r.aqk,
        Akk=r.akk,
        g=r.g_cumsum,
        h=r.h,
        do=dy,
        dht=None,
        initial_state=None,
        scale=scale,
        segment_ids=None,
        cu_seqlens=r.cu_seqlens,
        chunk_indices=None,
        chunk_size=64,
        use_exp2=True,
        safe_gate=True,
        lower_bound=lower_bound,
        return_dh0=False,
        disable_recompute=True,
        cp_active=False,
        has_initial_state=False,
        N_MAX=r.cu_seqlens.shape[-1] - 1,
        batch_first_inputs=True,
        cast_dg_to_q_dtype=gate.dtype == query.dtype,
        original_token_segment_ids=r.segment_ids,
        original_token_segments=True,
        raw_gate=raw_gate if fused_gate else None,
        gate_A_log=r.a_log if fused_gate else None,
        gate_dt_bias=r.bias if fused_gate else None,
        fuse_gate_bwd=fused_gate,
    )
    dq, dk, dv, db, dg, _ = values[:6]
    dA, dbias = values[6:] if fused_gate else (None, None)
    if use_gate_in_kernel and not fused_gate:
      valid = r.segment_ids[:, :, None, None] > 0
      dg, dA, dbias = kda_gate_bwd(
          jnp.where(valid, raw_gate, 0),
          r.a_log,
          r.bias,
          jnp.where(valid, dg, 0),
          lower_bound,
          head_first=False,
      )
    if reorder:
      dq, dk, dv, dg = (
          _take_tokens(x, r.inverse_permutation) for x in (dq, dk, dv, dg)
      )
      db = _take_tokens(db.transpose(1, 2, 0), r.inverse_permutation).transpose(
          2, 0, 1
      )
    return dq, dk, dv, db, dg, dA, dbias

  dq, dk, dv, db, dg, dA, dbias = (
      run(False)
      if r.permutation is None
      else jax.lax.cond(
          r.ordered, lambda _: run(False), lambda _: run(True), None
      )
  )
  gradients = dict(
      query=dq[..., :key_dim].transpose(2, 0, 1, 3).astype(query.dtype),
      key=dk[..., :key_dim].transpose(2, 0, 1, 3).astype(query.dtype),
      value=dv[..., :value_dim].transpose(2, 0, 1, 3).astype(value.dtype),
      gate=dg[..., :key_dim].transpose(2, 0, 1, 3).astype(gate.dtype),
      beta=db.astype(r.beta.dtype),
  )
  if a_log is not None:
    gradients["a_log"] = jnp.zeros_like(a_log) if dA is None else dA
  if delta_time_bias is not None:
    gradients["delta_time_bias"] = (
        jnp.zeros_like(delta_time_bias)
        if dbias is None
        else dbias.reshape(heads, r.q.shape[-1])[:, :key_dim].reshape(-1)
    )
  return gradients
