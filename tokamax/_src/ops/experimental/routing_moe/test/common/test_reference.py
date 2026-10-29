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
"""Pure-XLA references and unfused baselines for routing_moe tests."""

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.test.common import test_inputs
from tokamax._src.ops.experimental.routing_moe.test.common import test_shapes


def topk_reduce(
    rows: jax.Array, weights: jax.Array, out_dtype: jax.typing.DTypeLike
) -> jax.Array:
  """Computes the weighted sum of each token's `top_k` rows."""
  num_tokens, topk = weights.shape
  w = weights.astype(jnp.float32).reshape(-1, 1)
  x = jnp.where(w != 0, rows.astype(jnp.float32), 0.0) * w
  return x.reshape(num_tokens, topk, -1).sum(axis=1).astype(out_dtype)


def local_pick_weights(
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    expert_lo: jax.Array | int,
    num_local_experts: int,
) -> jax.Array:
  """Zeros out `topk_weights` for picks served by other EP shards."""
  local = jnp.logical_and(
      topk_indices >= expert_lo, topk_indices < expert_lo + num_local_experts
  )
  return jnp.where(local, topk_weights, 0)


def xla_output_reduce_unpack(
    rows_compact: jax.Array,
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    group_offset: jax.Array,
    num_local_experts: int,
    *,
    out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> jax.Array:
  """Unpacks and reduces this shard's routed rows in pure XLA."""
  w_local = local_pick_weights(
      topk_weights, topk_indices, group_offset, num_local_experts
  )
  rows = rows_compact.reshape(layout_util.to_large_shape(rows_compact.shape))
  return topk_reduce(rows, w_local, out_dtype)


def _apply_act_fn(acc: jax.Array, fuse_act: str | None) -> jax.Array:
  """Applies a gated activation to `[..., gate | up]` columns."""
  if fuse_act is None:
    return acc
  gate, up = jnp.split(acc, 2, axis=-1)
  match fuse_act:
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
      raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def _dequantize(
    w: jax.Array, scale: jax.Array | None, dtype: jax.typing.DTypeLike
) -> jax.Array:
  """Dequantizes `w` with `[groups, num_blocks, 1, out]` scales to `dtype`."""
  if scale is None:
    return w.astype(dtype)
  g, num_blocks, _, n = scale.shape
  w_f32 = w.astype(jnp.float32).reshape(g, num_blocks, -1, n)
  return (w_f32 * scale.astype(jnp.float32)).reshape(g, -1, n).astype(dtype)


def _ragged_matmul(
    x: jax.Array,
    w: arg.SingleWeightWrapper[jax.Array],
    group_sizes: jax.Array,
) -> jax.Array:
  """Computes grouped x @ w + bias in float32."""
  acc = jax.lax.ragged_dot(
      x,
      _dequantize(w.weight, w.scale, x.dtype),
      group_sizes,
      preferred_element_type=jnp.float32,
  )
  if w.bias is not None:
    ones = jnp.ones((x.shape[0], 1), dtype=jnp.float32)
    acc += jax.lax.ragged_dot(
        ones,
        w.bias.astype(jnp.float32),
        group_sizes,
        preferred_element_type=jnp.float32,
    )
  return acc


@jax.jit(static_argnames=("fuse_act", "out_dtype"))
def reference_gmm(
    x_sorted: jax.Array,
    w: arg.SingleWeightWrapper[jax.Array],
    group_sizes: jax.Array,
    *,
    w2: arg.SingleWeightWrapper[jax.Array] | None = None,
    fuse_act: str | None = None,
    out_dtype: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Computes grouped matmul on sorted rows in pure XLA."""
  act = _apply_act_fn(_ragged_matmul(x_sorted, w, group_sizes), fuse_act)
  if w2 is not None:
    act = _ragged_matmul(act.astype(x_sorted.dtype), w2, group_sizes)
  return act.astype(x_sorted.dtype if out_dtype is None else out_dtype)


@jax.jit
def unfused_gather_gmm(
    x: jax.Array,
    w: arg.SingleWeightWrapper[jax.Array],
    group_sizes: jax.Array,
    gather_indices: jax.Array,
) -> jax.Array:
  """Gathers rows and runs a grouped matmul as an unfused XLA baseline."""
  return reference_gmm(x[gather_indices], w, group_sizes, fuse_act="silu")


@jax.jit
def unfused_gmm_scatter(
    x_sorted: jax.Array,
    w: arg.SingleWeightWrapper[jax.Array],
    group_sizes: jax.Array,
    sorted_to_flat: jax.Array,
) -> jax.Array:
  """Runs a grouped matmul and scatters rows as an unfused XLA baseline."""
  out = reference_gmm(x_sorted, w, group_sizes)
  mask = jnp.arange(out.shape[0]) < group_sizes.sum()
  out = jnp.where(mask[:, None], out, 0)
  return jnp.zeros_like(out).at[sorted_to_flat].set(out)


def compute_topk(
    moe: test_inputs.MoEInputs,
    shape: test_shapes.MoEShape,
    *,
    fold_shared_expert: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Computes top-k weights and indices, appending a shared expert if folded."""
  weights, indices = test_inputs.top_k(
      moe.gating_output, shape.top_k, renormalize=shape.renormalize
  )
  if not fold_shared_expert:
    return weights, indices

  assert moe.w_shared_gate is not None
  shared_logits = jnp.matmul(
      moe.hidden_states,
      moe.w_shared_gate,
      preferred_element_type=jnp.float32,
  )
  shared_weight = jax.nn.sigmoid(shared_logits).astype(weights.dtype)
  shared_index = jnp.full_like(indices[:, :1], moe.gating_output.shape[-1])
  return (
      jnp.concatenate([weights, shared_weight], axis=-1),
      jnp.concatenate([indices, shared_index], axis=-1),
  )


def _matmul(x: jax.Array, w: jax.Array, scale: jax.Array | None) -> jax.Array:
  """Computes `x @ w[0]` in `float32`, applying sub-channel `scale` if set."""
  if scale is None:
    return jnp.matmul(x, w[0], preferred_element_type=jnp.float32)
  num_blocks = scale.shape[1]
  x_blocks = x.reshape(x.shape[0], num_blocks, -1).swapaxes(0, 1)
  w_blocks = w.reshape(num_blocks, -1, w.shape[-1])
  acc = jnp.matmul(x_blocks, w_blocks, preferred_element_type=jnp.float32)
  return (acc * scale[0]).sum(axis=0)


def shared_expert(
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w_gate: jax.Array,
    *,
    activation: str,
) -> jax.Array:
  """Computes the dense shared-expert gated MLP output in pure XLA."""
  gate_up = _matmul(x, w1, w1_scale)
  act = _apply_act_fn(gate_up, activation).astype(x.dtype)
  down = _matmul(act, w2, w2_scale).astype(x.dtype)
  gate_logits = jnp.matmul(x, w_gate, preferred_element_type=jnp.float32)
  return down * jax.nn.sigmoid(gate_logits).astype(x.dtype)


@jax.jit(static_argnames=("shape", "fold_shared_expert"))
def reference_moe(
    moe: test_inputs.MoEInputs,
    shape: test_shapes.MoEShape,
    *,
    fold_shared_expert: bool = False,
) -> jax.Array:
  """Computes the full MoE layer output in pure XLA."""
  topk_weights, topk_indices = compute_topk(
      moe, shape, fold_shared_expert=fold_shared_expert
  )
  num_experts = moe.w1.shape[0]
  topk = topk_weights.shape[-1]
  flat = topk_indices.flatten()
  _, sorted_to_flat = jax.lax.sort_key_val(
      flat, jnp.arange(flat.size, dtype=jnp.int32), is_stable=True
  )
  group_sizes = jnp.bincount(flat, length=num_experts)

  x_sorted = moe.hidden_states[sorted_to_flat // topk]
  w1 = arg.make_weight_wrapper(
      moe.w1, moe.w1_scale, moe.w1_bias, fused_gate_up=True
  )
  w2 = arg.make_weight_wrapper(moe.w2, moe.w2_scale, moe.w2_bias)
  out_sorted = reference_gmm(
      x_sorted, w1, group_sizes, w2=w2, fuse_act=shape.activation
  )
  rows = jnp.zeros_like(out_sorted).at[sorted_to_flat].set(out_sorted)
  dtype = moe.hidden_states.dtype
  out = topk_reduce(rows, topk_weights.astype(dtype), dtype)
  if moe.w1_shared is not None:
    assert moe.w2_shared is not None and moe.w_shared_gate is not None
    out += shared_expert(
        moe.hidden_states,
        moe.w1_shared,
        moe.w2_shared,
        moe.w1_shared_scale,
        moe.w2_shared_scale,
        moe.w_shared_gate,
        activation=shape.activation,
    )
  return out
