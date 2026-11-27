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
"""Opt-in capacity-free all-gather CP, using the aligned and native VJPs.

Packed sequences must have globally ordered positive labels, with zero padding
allowed between records. This is the usual CP packing contract; arbitrary
interleaved labels spanning multiple ranks are not supported by this entry.
The public operator and its default dispatch remain unchanged.
"""

import dataclasses
import jax
from jax.experimental import checkify
import jax.numpy as jnp
from tokamax._src.ops.kda import compact
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic


def _native(q, k, v, g, beta, a, bias, *, ids, scale, lower_bound, raw_gate):
  options = dict(
      scale=scale,
      segment_ids=ids,
      lower_bound=lower_bound,
      use_gate_in_kernel=raw_gate,
      sorted_segments=False,
      max_num_segments=ids.shape[1],
  )

  def forward(xs, save):
    qi, ki, vi, gi, bi, ai, di = xs
    return compact.forward(
        qi,
        ki,
        vi,
        gi,
        bi,
        a_log=ai,
        delta_time_bias=di,
        return_residuals=save,
        **options,
    )

  @jax.custom_vjp
  def run(*xs):
    return forward(xs, False)[0][0]

  def fwd(*xs):
    out, tape = forward(xs, True)
    return out[0], (tape, xs)

  def bwd(res, dy):
    tape, xs = res
    qi, _, vi, gi, _, ai, di = xs
    grads = compact.backward(
        tape,
        (dy, None),
        query=qi,
        value=vi,
        gate=gi,
        a_log=ai,
        delta_time_bias=di,
        scale=scale,
        lower_bound=lower_bound,
        use_gate_in_kernel=raw_gate,
    )
    return tuple(
        None if x is None else grads[name].astype(x.dtype)
        for name, x in zip(
            (
                "query",
                "key",
                "value",
                "gate",
                "beta",
                "a_log",
                "delta_time_bias",
            ),
            xs,
            strict=True,
        )
    )

  run.defvjp(fwd, bwd)
  return run(q, k, v, g, beta, a, bias)


def kimi_delta_attention_cp_compact(
    query,
    key,
    value,
    gate,
    beta,
    *,
    segment_ids,
    context_parallel_metadata,
    a_log=None,
    delta_time_bias=None,
    scale=None,
    use_gate_in_kernel=False,
    lower_bound=None,
    staged_run_cap=64,
    config=None,
):
  """Return (output, None) for BF16 [H,1,T_local,D] training inputs.

  Local aligned capacity is bounded by staged_run_cap, independent of the
  logical segment capacity. Shards above the cap stage only the first and
  last positive label, the only sequences that can cross a rank boundary.
  Other sequences use the native original-token forward/backward. The native
  pass always runs, avoiding a conditional that retains two residual tapes.
  Initial/final states, convolution and Q/K normalization are not supported.
  """
  cp = context_parallel_metadata
  if cp is None or not cp.is_cp_enabled:
    raise ValueError("compact all-gather requires an active CP mesh")
  if query.ndim != 4 or query.shape[1] != 1 or query.shape[2] % 64:
    raise ValueError(
        "compact CP requires [H,1,T_local,D], T_local divisible by 64"
    )
  if query.dtype != jnp.bfloat16 or any(
      x.dtype != query.dtype for x in (key, value)
  ):
    raise ValueError("compact CP requires BF16 query/key/value")
  if type(staged_run_cap) is not int or staged_run_cap < 2:
    raise ValueError("staged_run_cap must be at least two")
  if segment_ids.ndim == 1:
    segment_ids = segment_ids[None]
  if segment_ids.shape != query.shape[1:3]:
    raise ValueError("segment_ids must have shape [1,T_local]")
  valid = segment_ids > 0
  # Ignore padding while checking the grouped-label contract.
  previous = jax.lax.associative_scan(jnp.maximum, segment_ids, axis=1)
  legal = jnp.all(segment_ids >= 0) & jnp.all(
      ~valid | (segment_ids >= previous)
  )
  checkify.debug_check(legal, "compact CP requires globally grouped labels")
  tokens = query.shape[2]
  positions = jnp.arange(tokens)[None]
  first_pos = jnp.min(
      jnp.where(valid, positions, tokens), axis=1, keepdims=True
  )
  last_pos = jnp.max(jnp.where(valid, positions, -1), axis=1, keepdims=True)
  first = jnp.take_along_axis(
      segment_ids, jnp.minimum(first_pos, tokens - 1), axis=1
  )
  last = jnp.take_along_axis(segment_ids, jnp.maximum(last_pos, 0), axis=1)
  starts = valid & (
      segment_ids != jnp.pad(segment_ids[:, :-1], ((0, 0), (1, 0)))
  )
  count = jnp.sum(starts)
  boundary = valid & ((segment_ids == first) | (segment_ids == last))
  select = (count <= staged_run_cap) | boundary
  staged_ids = jnp.where(select, segment_ids, 0)
  # Raw padding gates must activate to zero and contribute no parameter gradient.
  staged_gate = jnp.where(
      select[None, ..., None],
      gate,
      jnp.asarray(-1e30 if use_gate_in_kernel else 0, gate.dtype),
  )
  cfg = dataclasses.replace(
      config or mosaic.Config(), cp_local_state_indices=True
  )
  attention = mosaic.PallasMosaicTpuKimiDeltaAttention(config=cfg)
  scale = query.shape[-1] ** -0.5 if scale is None else scale
  staged = attention(
      query,
      key,
      value,
      staged_gate,
      beta,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      scale=scale,
      segment_ids=staged_ids,
      max_num_segments=staged_run_cap,
      context_parallel_metadata=cp,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
  )[0]
  local = _native(
      query,
      key,
      value,
      gate,
      beta,
      a_log,
      delta_time_bias,
      ids=segment_ids,
      scale=scale,
      lower_bound=lower_bound,
      raw_gate=use_gate_in_kernel,
  )
  return jnp.where(select[None, ..., None], staged, local), None
