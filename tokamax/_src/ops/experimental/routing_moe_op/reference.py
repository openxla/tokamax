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
"""Reference JAX implementation of the routing MoE operator."""

import functools
import jax
import jax.numpy as jnp
from tokamax._src.ops.gmm_v2 import gmm_v2


def _validate_scale(weight: jax.Array, scale: jax.Array) -> None:
  """Validates `[E, num_blocks, 1, N]` scale against `[E, K, N]` weight."""
  if scale.ndim != 4:
    raise NotImplementedError(
        f"Unsupported weight scale shape: {scale.shape}. Expected 4D"
        " [num_experts, num_blocks, 1, out_features]."
    )
  num_experts, num_blocks, single_dim, out_dim = scale.shape
  if weight.shape[0] != num_experts:
    raise ValueError(
        f"Weight experts {weight.shape[0]} does not match scale experts"
        f" {num_experts}."
    )
  if weight.shape[1] % num_blocks != 0:
    raise ValueError(
        f"{num_blocks} scale blocks do not divide contraction dimension"
        f" {weight.shape[1]}."
    )
  if single_dim != 1 or out_dim != weight.shape[2]:
    raise ValueError(
        f"Scale shape {scale.shape} is incompatible with weight shape"
        f" {weight.shape}."
    )


def _scaled_ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    sorted_experts: jax.Array,
    scale: jax.Array | None,
) -> jax.Array:
  """Computes `ragged_dot(lhs, rhs)` with optional per-channel/block scales."""
  # Keep FP8/bf16 weights in bf16 (FP8 e4m3 -> bf16 is exact) to avoid
  # materializing f32 weight copies across all experts.
  rhs_op = rhs.astype(jnp.bfloat16) if rhs.dtype.itemsize <= 2 else rhs
  lhs_op = lhs.astype(rhs_op.dtype)
  if scale is None:
    return jax.lax.ragged_dot(
        lhs_op, rhs_op, group_sizes, preferred_element_type=jnp.float32
    )
  _validate_scale(rhs, scale)
  num_blocks = scale.shape[1]
  block_size = rhs.shape[1] // num_blocks
  row_scales = scale[sorted_experts, :, 0, :].astype(jnp.float32)
  acc = jnp.zeros((lhs_op.shape[0], rhs.shape[2]), dtype=jnp.float32)
  for block_idx in range(num_blocks):
    start = block_idx * block_size
    end = start + block_size
    block_out = jax.lax.ragged_dot(
        lhs_op[:, start:end],
        rhs_op[:, start:end, :],
        group_sizes,
        preferred_element_type=jnp.float32,
    )
    acc = acc + block_out * row_scales[:, block_idx, :]
  return acc


def _apply_activation(accumulator: jax.Array, activation: str) -> jax.Array:
  """Applies gated activation to `[..., gate | up]` columns."""
  gate, up = jnp.split(accumulator, 2, axis=-1)
  match activation:
    case "silu":
      return jax.nn.silu(gate) * up
    case "gelu":
      return jax.nn.gelu(gate) * up
    case "gelu_tanh":
      return jax.nn.gelu(gate, approximate=True) * up
    case "swigluoai":
      return gmm_v2.swigluoai(gate, up)
    case "silu_and_mul_with_clamp":
      return gmm_v2.silu_and_mul_with_clamp(gate, up)
    case _:
      raise NotImplementedError(
          f"Unsupported activation function: {activation}"
      )


@functools.partial(
    jax.jit,
    static_argnames=(
        "mesh",
        "activation",
        "tp_axis",
        "ep_axis",
        "use_sp",
        "reduce_output",
    ),
)
def routing_moe_reference(
    hidden_states: jax.Array,
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w1_bias: jax.Array | None = None,
    w2_bias: jax.Array | None = None,
    mesh: jax.sharding.Mesh | None = None,
    activation: str = "silu",
    tp_axis: str | None = None,
    ep_axis: str | None = None,
    use_sp: bool = False,
    reduce_output: bool = True,
) -> jax.Array:
  """Unsharded XLA reference for routing_moe using jax.lax.ragged_dot."""
  del mesh, ep_axis, use_sp
  if tp_axis is not None:
    raise NotImplementedError(
        f"Unsharded XLA reference does not support tp_axis={tp_axis!r}."
    )
  if not reduce_output:
    raise NotImplementedError(
        "Unsharded reference only supports reduce_output=True."
    )

  num_experts = w1.shape[0]
  num_tokens, topk = topk_indices.shape
  hidden_size = hidden_states.shape[1]
  num_routed = num_tokens * topk

  flat_experts = topk_indices.reshape(-1).astype(jnp.int32)
  group_sizes = jnp.bincount(flat_experts, length=num_experts)
  sorted_experts, perm = jax.lax.sort_key_val(
      flat_experts, jnp.arange(num_routed, dtype=jnp.int32)
  )

  sorted_inputs = hidden_states[perm // topk]
  acc1 = _scaled_ragged_dot(
      sorted_inputs, w1, group_sizes, sorted_experts, w1_scale
  )
  if w1_bias is not None:
    acc1 = acc1 + w1_bias[sorted_experts, 0, :].astype(jnp.float32)

  act = _apply_activation(acc1, activation)
  acc2 = _scaled_ragged_dot(act, w2, group_sizes, sorted_experts, w2_scale)
  unsorted_out = (
      jnp.zeros((num_routed, hidden_size), dtype=jnp.float32).at[perm].set(acc2)
  )
  # Added after the unsort: fusing this gather with the `w2_scale` gather
  # makes XLA:TPU in libtpu 0.0.50 (JAX 0.12.0) read the scale as the bias.
  if w2_bias is not None:
    unsorted_out = unsorted_out + w2_bias[flat_experts, 0, :].astype(
        jnp.float32
    )
  weights = topk_weights.astype(jnp.float32)
  weighted = (
      unsorted_out.reshape(num_tokens, topk, hidden_size) * weights[:, :, None]
  )
  return jnp.sum(weighted, axis=1).astype(hidden_states.dtype)
