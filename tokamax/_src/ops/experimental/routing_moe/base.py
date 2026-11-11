# Copyright 2026 Google LLC
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
"""Base operator interface for Tokamax Routing MoE."""

from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops import op

type ArrayLikeSpec = jax.Array | jax.ShapeDtypeStruct | jax.core.ShapedArray | np.ndarray


def _apply_act_fn(acc: jax.Array, act_fn: str) -> jax.Array:
  """Applies gated activation to `[..., gate | up]` columns."""
  gate, up = jnp.split(acc, 2, axis=-1)
  match act_fn:
    case "silu":
      return jax.nn.silu(gate) * up
    case "gelu":
      return jax.nn.gelu(gate) * up
    case "gelu_tanh":
      return jax.nn.gelu(gate, approximate=True) * up
    case "swigluoai":
      gate = jnp.clip(gate, max=7.0)
      up = jnp.clip(up, min=-7.0, max=7.0)
      return (up + 1.0) * gate * jax.nn.sigmoid(1.702 * gate)
    case "silu_and_mul_with_clamp":
      gate = jnp.clip(gate, max=10.0)
      up = jnp.clip(up, min=-10.0, max=10.0)
      return jax.nn.silu(gate) * up
    case _:
      raise NotImplementedError(f"Unsupported activation function: {act_fn}")


def _dequantize_one(w: jax.Array, scale: jax.Array | None) -> jax.Array:
  """Dequantizes one expert weight `[in, out]` with `[blocks, 1, out]` scale."""
  w_f32 = w.astype(jnp.float32)
  if scale is None:
    return w_f32
  num_blocks, _, out_cols = scale.shape
  return (
      w_f32.reshape(num_blocks, -1, out_cols) * scale.astype(jnp.float32)
  ).reshape(-1, out_cols)


def routing_moe_reference(
    x: jax.Array,
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w1_bias: jax.Array | None = None,
    w2_bias: jax.Array | None = None,
    experts_start: jax.Array | None = None,
    *,
    global_num_experts: int | None = None,
    act_fn: str = "silu",
    skip_padded_tokens: bool = False,
) -> jax.Array:
  """Computes single-shard or EP-shard routed MoE output in pure JAX."""
  del global_num_experts
  if w1_bias is not None or w2_bias is not None:
    raise NotImplementedError("Bias is currently not supported.")

  num_tokens, _ = topk_weights.shape
  hidden = x.shape[1]
  local_experts = w1.shape[0]
  start = (
      jnp.int32(0)
      if experts_start is None
      else experts_start.reshape(-1)[0].astype(jnp.int32)
  )
  weights = topk_weights.astype(jnp.float32)
  indices = topk_indices.astype(jnp.int32)
  if skip_padded_tokens:
    indices = jnp.where(weights != 0, indices, -1)
  local_ids = indices - start
  x32 = x.astype(jnp.float32)

  def _expert_step(e: jax.Array, acc_out: jax.Array) -> jax.Array:
    pick_w = jnp.sum(jnp.where(local_ids == e, weights, 0.0), axis=-1)[:, None]
    w1e = _dequantize_one(w1[e], None if w1_scale is None else w1_scale[e])
    w2e = _dequantize_one(w2[e], None if w2_scale is None else w2_scale[e])
    gate_up = jnp.matmul(x32, w1e, preferred_element_type=jnp.float32)
    inter = _apply_act_fn(gate_up, act_fn).astype(x.dtype).astype(jnp.float32)
    row = jnp.matmul(inter, w2e, preferred_element_type=jnp.float32)
    return acc_out + jnp.where(pick_w != 0, row, 0.0) * pick_w

  out = jax.lax.fori_loop(
      0,
      local_experts,
      _expert_step,
      jnp.zeros((num_tokens, hidden), dtype=jnp.float32),
  )
  return out.astype(x.dtype)


class RoutingMoe[C](op.Op[Any, jax.Array, None, C, Any]):
  """Base operator interface for pre-routed single-shard / EP-shard MoE."""

  def bind(
      self,
      x: ArrayLikeSpec,
      topk_weights: ArrayLikeSpec,
      topk_indices: ArrayLikeSpec,
      w1: ArrayLikeSpec,
      w2: ArrayLikeSpec,
      w1_scale: ArrayLikeSpec | None = None,
      w2_scale: ArrayLikeSpec | None = None,
      w1_bias: ArrayLikeSpec | None = None,
      w2_bias: ArrayLikeSpec | None = None,
      experts_start: ArrayLikeSpec | None = None,
      *,
      global_num_experts: int | None = None,
      act_fn: str = "silu",
      skip_padded_tokens: bool = False,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Validates and binds arguments for RoutingMoe."""
    if x.ndim != 2:
      raise ValueError(f"x must be rank 2 (T, H), got {x.shape}.")
    if topk_weights.ndim != 2 or topk_indices.shape != topk_weights.shape:
      raise ValueError(
          f"topk_weights {topk_weights.shape} and topk_indices"
          f" {topk_indices.shape} must have matching rank-2 (T, K) shapes."
      )
    if topk_weights.shape[0] != x.shape[0]:
      raise ValueError(
          f"Token count mismatch: x has {x.shape[0]}, topk_weights has"
          f" {topk_weights.shape[0]}."
      )
    return super().bind(
        x=x,
        topk_weights=topk_weights,
        topk_indices=topk_indices,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        experts_start=experts_start,
        global_num_experts=global_num_experts,
        act_fn=act_fn,
        skip_padded_tokens=skip_padded_tokens,
        return_residuals=return_residuals,
    )

  @override
  def _fwd(
      self,
      x: jax.Array,
      topk_weights: jax.Array,
      topk_indices: jax.Array,
      w1: jax.Array,
      w2: jax.Array,
      w1_scale: jax.Array | None = None,
      w2_scale: jax.Array | None = None,
      w1_bias: jax.Array | None = None,
      w2_bias: jax.Array | None = None,
      experts_start: jax.Array | None = None,
      *,
      global_num_experts: int | None = None,
      act_fn: str = "silu",
      skip_padded_tokens: bool = False,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[jax.Array, None]:
    del return_residuals, config
    out = routing_moe_reference(
        x=x,
        topk_weights=topk_weights,
        topk_indices=topk_indices,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        experts_start=experts_start,
        global_num_experts=global_num_experts,
        act_fn=act_fn,
        skip_padded_tokens=skip_padded_tokens,
    )
    return out, None
