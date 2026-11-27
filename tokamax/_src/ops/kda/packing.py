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
"""Packing correction around the existing KDA custom VJP.

Canonicalization preserves Tokamax's per-label recurrence. Convolution is
performed at original token positions before gathering, so padding does not
change its lag semantics. Common canonical layouts retain the fused path.
"""

import dataclasses
from typing import Annotated, Any
import jax
import jax.numpy as jnp
import pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PackingResiduals:
  pullback: Annotated[
      Any,
      pydantic.PlainSerializer(
          lambda _: None, return_type=type(None), when_used="json"
      ),
  ]


ARRAY_NAMES = (
    "query",
    "key",
    "value",
    "gate",
    "beta",
    "a_log",
    "delta_time_bias",
    "initial_state",
    "conv_weight_q",
    "conv_weight_k",
    "conv_weight_v",
    "conv_bias_q",
    "conv_bias_k",
    "conv_bias_v",
)


def _take(x, order):
  index = order[None, ...]
  if x.ndim == 4:
    index = index[..., None]
  return jnp.take_along_axis(x, index, axis=2)


def _conv(x, weight, bias, ids):
  width = weight.shape[1]
  x_f, w_f = x.astype(jnp.float32), weight.astype(jnp.float32)
  acc = x_f * w_f[:, None, width - 1 : width, :]
  for lag in range(1, min(width, x.shape[2])):
    shifted = jnp.pad(x_f[:, :, :-lag], ((0, 0), (0, 0), (lag, 0), (0, 0)))
    same = (ids[:, lag:] == ids[:, :-lag]) & (ids[:, lag:] > 0)
    same = jnp.pad(same, ((0, 0), (lag, 0)))
    shifted = jnp.where(same[None, ..., None], shifted, 0)
    acc = acc + shifted * w_f[:, None, width - 1 - lag : width - lag, :]
  if bias is not None:
    acc = acc + bias.astype(jnp.float32)[:, None, None, :]
  return jax.nn.silu(acc).astype(x.dtype)


def _needs_packing(ids, clean, valid):
  previous = jnp.pad(clean[:, :-1], ((0, 0), (1, 0)))
  changed = valid & (clean != previous)
  return (
      jnp.any(changed & (clean != previous + 1))
      | jnp.any(valid[:, 1:] & ~valid[:, :-1])
      | jnp.any(ids != clean)
  )


def invoke(op, config, arguments):
  """Pack operands, call KDA once, then restore the public token/state ABI."""
  context = arguments["context_parallel_metadata"]
  # Cross-rank branch coordination and explicitly fused convolution retain
  # their existing implementations. The default local adapter is compositional.
  if context is not None and context.is_cp_enabled:
    return _legacy_invoke(op, config, arguments)
  convolution = arguments["use_conv1d_in_kernel"]
  capacity_hint = (
      arguments["initial_state"].shape[1]
      if arguments["initial_state"] is not None
      else arguments["max_num_segments"]
  )
  short_decode = (
      arguments["query"].shape[2] <= config.chunk_size
      and capacity_hint is not None
      and capacity_hint > 1
  )
  optimized = any(
      getattr(config, name)
      for name in (
          "fuse_forward",
          "fuse_backward",
          "fuse_rematerialization",
          "packed_forward",
          "packed_backward",
          "packed_gradients",
          "packed_output",
      )
  )
  if (
      optimized
      or (convolution and not short_decode)
      or (not convolution and not arguments["return_residuals"])
  ):
    return _legacy_invoke(op, config, arguments)

  original = dict(arguments)
  save = original.pop("return_residuals")
  ids = original["segment_ids"]
  batch, tokens = original["query"].shape[1:3]
  if ids.ndim == 1:
    ids = jnp.broadcast_to(ids[None], (batch, tokens))
  if ids.shape != (batch, tokens):
    raise ValueError("segment_ids must have shape [B,T] or [T]")
  h0 = original["initial_state"]
  capacity = h0.shape[1] if h0 is not None else original["max_num_segments"]
  if capacity is None or capacity <= 0:
    raise ValueError("max_num_segments must be positive")
  if h0 is not None and original["max_num_segments"] not in (None, capacity):
    raise ValueError("initial_state state count must match max_num_segments")
  physical = min(capacity, tokens)
  valid = (ids > 0) & (ids <= capacity)
  clean = jnp.where(valid, ids, 0)
  keys = jnp.where(
      valid,
      ids.astype(jnp.uint32),
      jnp.asarray(jnp.iinfo(jnp.uint32).max, jnp.uint32),
  )
  needs = _needs_packing(ids, clean, valid)
  force = physical != capacity
  rows = jnp.arange(batch)[:, None]
  iota = jnp.broadcast_to(jnp.arange(tokens, dtype=jnp.int32), (batch, tokens))
  inner = type(op)(
      config=dataclasses.replace(config, canonicalize_packed_inputs=False),
      bypass_device_check=op.bypass_device_check,
  )
  static = {k: v for k, v in original.items() if k not in ARRAY_NAMES}

  def composed(*values):
    params = dict(zip(ARRAY_NAMES, values, strict=True))
    q, k, v = (params[n] for n in ("query", "key", "value"))
    options = {**static, **params}
    if options["use_conv1d_in_kernel"]:
      # Evaluate only the original T-token domain, before any row movement.
      # This keeps physical-lag semantics and avoids convolution on aligned pads.
      q, k, v = (
          _conv(x, params["conv_weight_" + n], params["conv_bias_" + n], ids)
          for n, x in zip(("q", "k", "v"), (q, k, v), strict=True)
      )
      options["use_conv1d_in_kernel"] = False
      for n in ("q", "k", "v"):
        options["conv_weight_" + n] = options["conv_bias_" + n] = None
    operands = (
        q,
        k,
        v,
        params["gate"],
        params["beta"],
        params["initial_state"],
    )

    def maps(_):
      order = jnp.argsort(keys, axis=1, stable=True).astype(jnp.int32)
      inverse = jnp.zeros_like(order).at[rows, order].set(iota)
      labels_per_token = jnp.take_along_axis(clean, order, axis=1)
      positive = labels_per_token > 0
      starts = positive & (
          labels_per_token
          != jnp.pad(labels_per_token[:, :-1], ((0, 0), (1, 0)))
      )
      canonical = jnp.where(positive, jnp.cumsum(starts, axis=1), 0)
      positions = jax.vmap(
          lambda row: jnp.nonzero(row, size=physical, fill_value=tokens)[0]
      )(starts)
      labels = jnp.where(
          positions < tokens,
          jnp.take_along_axis(
              labels_per_token, jnp.minimum(positions, tokens - 1), axis=1
          ),
          0,
      )
      return order, canonical, labels, inverse

    def identity(_):
      labels = jnp.broadcast_to(
          jnp.arange(1, physical + 1, dtype=ids.dtype), (batch, physical)
      )
      return iota, ids, labels, iota

    order, canonical, labels, inverse = (
        maps(None) if force else jax.lax.cond(needs, maps, identity, None)
    )
    state = operands[5]
    if state is not None:

      def gather_state(x):
        gathered = jnp.take_along_axis(
            x, jnp.maximum(labels - 1, 0)[..., None, None, None], axis=1
        )
        return jnp.where((labels > 0)[..., None, None, None], gathered, 0)

      state = (
          gather_state(state)
          if force
          else jax.lax.cond(needs, gather_state, lambda x: x, state)
      )
    prepared = (*operands[:5], state)
    options["_packing_order"] = jnp.stack((order, inverse))
    options.update(
        zip(
            ("query", "key", "value", "gate", "beta", "initial_state"),
            prepared,
            strict=True,
        )
    )
    options.update(segment_ids=canonical, max_num_segments=physical)
    # One core custom VJP, without a second Op-level batching/format boundary.
    core_static = {
        k: v
        for k, v in options.items()
        if k not in ARRAY_NAMES and k not in ("segment_ids", "_packing_order")
    }

    def run(meta, xs, save):
      kwargs = {
          **core_static,
          **dict(zip(ARRAY_NAMES, xs, strict=True)),
          "segment_ids": meta[0],
          "_packing_order": meta[1],
      }
      return inner._fwd(**kwargs, return_residuals=save, config=inner.config)

    @jax.custom_vjp
    def core(meta, *xs):
      return run(meta, xs, False)[0]

    def fwd(meta, *xs):
      out, tape = run(meta, xs, True)
      return out, (meta, xs, out, tape)

    def bwd(res, cot):
      meta, xs, out, tape = res
      kwargs = {
          **core_static,
          **dict(zip(ARRAY_NAMES, xs, strict=True)),
          "segment_ids": meta[0],
          "_packing_order": meta[1],
      }
      grads, _ = inner.vjp._fwd(
          tape, out, cot, **kwargs, return_residuals=False, config=inner.config
      )
      return (
          None,
          *(
              None if x is None else grads.get(name, jnp.zeros_like(x))
              for name, x in zip(ARRAY_NAMES, xs, strict=True)
          ),
      )

    core.defvjp(fwd, bwd, optimize_remat=True)
    core_values = tuple(options[n] for n in ARRAY_NAMES)
    y, state = core((canonical, options.get("_packing_order")), *core_values)
    # Varlen unalignment already emits exact zeros for all invalid rows.
    if state is not None:

      def restore(updated):
        logical = (
            jnp.zeros((batch, capacity) + updated.shape[2:], updated.dtype)
            if params["initial_state"] is None
            else params["initial_state"].astype(updated.dtype)
        )
        dest = jnp.where(labels > 0, labels - 1, capacity)
        return logical.at[rows, dest].set(updated, mode="drop")

      state = (
          restore(state)
          if force
          else jax.lax.cond(needs, restore, lambda x: x, state)
      )
    return y, state

  values = tuple(original[n] for n in ARRAY_NAMES)
  if save:
    output, pullback = jax.vjp(composed, *values)
    return output, PackingResiduals(pullback)
  return composed(*values), None


def _legacy_invoke(op, config, arguments):
  """Return corrected output and a compositional, shape-safe pullback tape."""
  original = dict(arguments)
  return_residuals = original.pop("return_residuals")
  ids = original["segment_ids"]
  if ids.ndim == 1:
    ids = jnp.broadcast_to(ids[None], original["query"].shape[1:3])
  if ids.ndim != 2:
    raise ValueError("segment_ids must have shape [B, T] or [T]")
  batch, tokens = ids.shape
  context = original["context_parallel_metadata"]
  cp = context is not None and context.is_cp_enabled
  h0 = original["initial_state"]
  capacity = original["max_num_segments"]
  if h0 is not None:
    if h0.ndim != 5:
      raise ValueError("Packed initial_state must have shape [B,N,H,K,V]")
    if capacity is not None and capacity != h0.shape[1]:
      raise ValueError("initial_state state count must match max_num_segments")
    capacity = h0.shape[1]
  if capacity is None:
    raise ValueError(
        "max_num_segments is required for packed inputs without state"
    )
  if capacity <= 0:
    raise ValueError("max_num_segments must be positive")
  if cp and (h0 is not None or original["output_final_state"]):
    raise ValueError("CP does not support initial/final states")
  valid = ids > 0
  if not cp:
    valid = valid & (ids <= capacity)
  clean_ids = jnp.where(valid, ids, 0)
  iota = jnp.broadcast_to(jnp.arange(tokens, dtype=jnp.int32), ids.shape)
  if cp:
    needs = jnp.any(valid[:, 1:] & ~valid[:, :-1]) | jnp.any(ids < 0)
    needs = jax.lax.pmax(needs.astype(jnp.int32), context.axis_name) != 0
  else:
    keys = jnp.where(
        valid,
        ids.astype(jnp.uint32),
        jnp.asarray(jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32),
    )
    starts = valid & (clean_ids != jnp.pad(clean_ids[:, :-1], ((0, 0), (1, 0))))
    dense = jnp.where(valid, jnp.cumsum(starts, axis=1), 0)
    needs = jnp.any(keys[:, 1:] < keys[:, :-1]) | jnp.any(clean_ids != dense)
    needs = needs | jnp.any(ids != clean_ids)
  inner_cfg = dataclasses.replace(config, canonicalize_packed_inputs=False)
  inner = type(op)(config=inner_cfg, bypass_device_check=op.bypass_device_check)
  if not isinstance(needs, jax.core.Tracer) and not bool(needs):
    return (
        inner(**original, return_residuals=return_residuals)
        if return_residuals
        else (inner(**original), None)
    )
  static = {k: v for k, v in original.items() if k not in ARRAY_NAMES}

  def composed(*values):
    def unchanged(xs):
      options = {**static, **dict(zip(ARRAY_NAMES, xs, strict=True))}
      y, state = inner(**options)
      return jnp.where(valid[None, ..., None], y, 0), state

    def corrected(xs):
      params = dict(zip(ARRAY_NAMES, xs, strict=True))
      opts = {**static, **params}
      q, k, v = (opts[n] for n in ("query", "key", "value"))
      if opts["use_conv1d_in_kernel"]:
        q, k, v = (
            _conv(x, opts["conv_weight_" + n], opts["conv_bias_" + n], ids)
            for n, x in zip(("q", "k", "v"), (q, k, v), strict=True)
        )
        opts["use_conv1d_in_kernel"] = False
        for n in ("q", "k", "v"):
          opts["conv_weight_" + n] = None
          opts["conv_bias_" + n] = None
      if cp:
        count = jnp.sum(valid, axis=1, keepdims=True)
        inverse = jnp.where(
            valid,
            jnp.cumsum(valid, axis=1) - 1,
            count + jnp.cumsum(~valid, axis=1) - 1,
        ).astype(jnp.int32)
        order = (
            jnp.zeros_like(inverse)
            .at[jnp.arange(batch)[:, None], inverse]
            .set(iota)
        )
      else:
        order = jnp.argsort(keys, axis=1, stable=True).astype(jnp.int32)
        inverse = (
            jnp.zeros_like(order)
            .at[jnp.arange(batch)[:, None], order]
            .set(iota)
        )
      packed_ids = jnp.take_along_axis(clean_ids, order, axis=1)
      for n, x in zip(
          ("query", "key", "value", "gate", "beta"),
          (q, k, v, opts["gate"], opts["beta"]),
          strict=True,
      ):
        opts[n] = _take(x, order)
      if not cp:
        positive = packed_ids > 0
        starts = positive & (
            packed_ids != jnp.pad(packed_ids[:, :-1], ((0, 0), (1, 0)))
        )
        opts["segment_ids"] = jnp.where(positive, jnp.cumsum(starts, axis=1), 0)
        physical = min(capacity, tokens)
        opts["max_num_segments"] = physical
        positions = jax.vmap(
            lambda row: jnp.nonzero(row, size=physical, fill_value=tokens)[0]
        )(starts)
        labels = jnp.where(
            positions < tokens,
            jnp.take_along_axis(
                packed_ids, jnp.minimum(positions, tokens - 1), axis=1
            ),
            0,
        )
        if params["initial_state"] is not None:
          state_index = jnp.maximum(labels - 1, 0)[..., None, None, None]
          opts["initial_state"] = jnp.take_along_axis(
              params["initial_state"], state_index, axis=1
          )
          opts["initial_state"] = jnp.where(
              (labels > 0)[..., None, None, None], opts["initial_state"], 0
          )
      else:
        opts["segment_ids"] = packed_ids
      y, state = inner(**opts)
      y = _take(y, inverse)
      y = jnp.where(valid[None, ..., None], y, 0)
      if state is not None:
        logical = (
            jnp.zeros((batch, capacity) + state.shape[2:], state.dtype)
            if params["initial_state"] is None
            else params["initial_state"].astype(state.dtype)
        )
        dest = jnp.where(labels > 0, labels - 1, capacity)
        state = logical.at[jnp.arange(batch)[:, None], dest].set(
            state, mode="drop"
        )
      return y, state

    return jax.lax.cond(needs, corrected, unchanged, values)

  values = tuple(original[n] for n in ARRAY_NAMES)
  if return_residuals:
    output, pullback = jax.vjp(composed, *values)
    return output, PackingResiduals(pullback)
  return composed(*values), None


def backward(residuals, cotangent, segment_ids):
  grads = residuals.pullback(cotangent)
  result = {
      name: value
      for name, value in zip(ARRAY_NAMES, grads, strict=True)
      if value is not None
  }
  result["segment_ids"] = jnp.zeros_like(segment_ids)
  return result, None
