# Copyright 2026 Ant Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in memory-bounded scheduling around the public Mosaic CP operator."""

import functools
import dataclasses

import jax
import jax.numpy as jnp
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic


def _uniform_segment_count(segment_ids, axis_name):
  """Choose one collective branch for every rank and every batch row."""
  ends = jnp.concatenate(
      (
          segment_ids[..., 1:] != segment_ids[..., :-1],
          jnp.ones(segment_ids.shape[:-1] + (1,), jnp.bool_),
      ),
      axis=-1,
  )
  count = jnp.max(jnp.sum(ends & (segment_ids != 0), axis=-1, dtype=jnp.int32))
  return jnp.max(jax.lax.all_gather(count, axis_name)) if axis_name else count


def _rematerialized_cond(predicate, small, full, operands):
  """Do not join both branches' differently sized residual workspaces."""

  @jax.custom_vjp
  def conditional(pred, xs):
    return jax.lax.cond(pred, small, full, xs)

  def forward(pred, xs):
    return conditional(pred, xs), (pred, xs)

  def backward(residual, cotangent):
    pred, xs = residual

    def pullback(branch, values):
      return jax.vjp(branch, values)[1](cotangent)[0]

    gradient = jax.lax.cond(
        pred,
        functools.partial(pullback, small),
        functools.partial(pullback, full),
        xs,
    )
    return None, gradient

  conditional.defvjp(forward, backward, optimize_remat=True)
  return conditional(predicate, operands)


def kimi_delta_attention_cp(
    query,
    key,
    value,
    gate,
    beta,
    *,
    segment_ids,
    context_parallel_metadata,
    max_num_segments,
    a_log=None,
    delta_time_bias=None,
    config=None,
    fast_segment_capacity=0,
    head_group_size=0,
    **kwargs,
):
  """Run training CP KDA with optional capacity buckets and sequential heads.

  Inputs/outputs use the public head-first layout [H,B,T,K/V]. Gate parameters
  are [H] and flattened [H*K]. No initial or final state is accepted by this
  training-only adapter. Existing operator defaults are unchanged.

  All ranks select the small capacity only when every rank/batch row fits it.
  Otherwise the original maximum is retained. A positive group size divides H;
  fallback head groups are checkpointed and evaluated sequentially. This bounds
  residual liveness but can add computation and collectives, so it is opt-in,
  not a universal performance or accelerator-memory-fit guarantee.
  """
  if query.ndim != 4 or any(size <= 0 for size in query.shape):
    raise ValueError("query must be nonempty [H,B,T,K]")
  heads = query.shape[0]
  if (
      context_parallel_metadata is None
      or not context_parallel_metadata.is_cp_enabled
  ):
    raise ValueError("Workspace scheduling requires an enabled CP context")
  if type(max_num_segments) is not int or max_num_segments <= 0:
    raise ValueError("max_num_segments must be positive")
  if (
      type(fast_segment_capacity) is not int
      or not 0 <= fast_segment_capacity < max_num_segments
  ):
    raise ValueError("fast capacity must be zero or below the maximum")
  if (
      type(head_group_size) is not int
      or head_group_size < 0
      or (head_group_size and heads % head_group_size)
  ):
    raise ValueError("head_group_size must be zero or divide the head count")
  if segment_ids.shape != query.shape[1:3]:
    raise ValueError("segment_ids must have rank-local shape [B,T]")
  if kwargs.get("initial_state") is not None or kwargs.get(
      "output_final_state", False
  ):
    raise ValueError("Workspace scheduling supports stateless training only")
  if kwargs.get("return_residuals", False):
    raise ValueError("Workspace scheduling does not expose private residuals")
  attention = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=dataclasses.replace(
          config or mosaic.Config(), cp_local_state_indices=True
      )
  )

  def run(capacity, group_size, operands):
    q, k, v, g, b, a, d, ids = operands

    def call(xs):
      qi, ki, vi, gi, bi, ai, di = xs
      return attention(
          qi,
          ki,
          vi,
          gi,
          bi,
          a_log=ai,
          delta_time_bias=None if di is None else di.reshape(-1),
          segment_ids=ids,
          context_parallel_metadata=context_parallel_metadata,
          # A rank-local nonempty run needs at least one token. CP exposes no
          # capacity-sized state output, so this does not narrow any legal batch.
          max_num_segments=min(capacity, query.shape[2]),
          **kwargs,
      )[0]

    if not group_size or group_size == heads:
      return call(operands[:7])
    groups = heads // group_size

    def split(x):
      return (
          None if x is None else x.reshape((groups, group_size) + x.shape[1:])
      )

    d = None if d is None else d.reshape(heads, query.shape[-1])
    output = jax.lax.map(
        jax.checkpoint(call), tuple(map(split, (q, k, v, g, b, a, d)))
    )
    return output.reshape((heads,) + output.shape[2:])

  operands = (
      query,
      key,
      value,
      gate,
      beta,
      a_log,
      delta_time_bias,
      segment_ids,
  )
  if not fast_segment_capacity:
    return run(max_num_segments, head_group_size, operands), None
  count = _uniform_segment_count(
      segment_ids, context_parallel_metadata.axis_name
  )
  output = _rematerialized_cond(
      count <= fast_segment_capacity,
      functools.partial(run, fast_segment_capacity, 0),
      functools.partial(run, max_num_segments, head_group_size),
      operands,
  )
  return output, None
