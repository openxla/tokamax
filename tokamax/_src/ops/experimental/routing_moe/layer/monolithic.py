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

"""Monolithic MoE entry point that computes routing and invokes modular MoE."""

import jax
from jax import sharding
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan_lib
from tokamax._src.ops.experimental.routing_moe.layer import modular
from tokamax._src.ops.experimental.routing_moe.layer import plan as moe_plan_lib

Mesh = sharding.Mesh


def apply_scoring_fn(scoring_fn: str, x: jax.Array) -> jax.Array:
  """Applies the router scoring function to gating logits."""
  match scoring_fn:
    case "softmax":
      return jax.nn.softmax(x, axis=-1)
    case "sigmoid":
      return jax.nn.sigmoid(x)
    case _:
      raise NotImplementedError(
          f"FusedMoE does not support {scoring_fn} scoring function"
      )


@jax.jit(
    static_argnames=(
        "topk",
        "renormalize",
        "mesh",
        "activation",
        "scoring_fn",
        "tp_axis",
        "ep_axis",
        "use_sp",
        "reduce_output",
        "moe_plan",
        "gmm_plan",
    )
)
def gated_routing_moe(
    hidden_states: jax.Array,
    gating_output: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    topk: int,
    renormalize: bool,
    mesh: Mesh,
    activation: str,
    scoring_fn: str,
    tp_axis: str | None,
    ep_axis: str | None,
    use_sp: bool = False,
    reduce_output: bool = True,
    moe_plan: moe_plan_lib.RoutingMoEPlan | None = None,
    gmm_plan: gmm_plan_lib.RoutingGmmPlan | None = None,
    shard_idx: jax.Array | None = None,
) -> jax.Array:
  """Route tokens in hidden_states into each experts based on routing.

  Args:
    hidden_states: [num_tokens, hidden_size]
    gating_output: routing information of tokens [num_tokens, num_experts]
    w1: first moe weights [num_experts, hidden_size, intermediate_size * 2]
    w2: second moe weights [num_experts, intermediate_size, hidden_size]
    w1_scale: w1 scale [num_experts, num_blocks, 1, intermediate_size * 2]
    w2_scale: w2 scale [num_experts, num_blocks, 1, hidden_size]
    w1_bias: optional bias of w1 [num_experts, 1, intermediate_size * 2]
    w2_bias: optional bias of w2 [num_experts, 1, hidden_size]
    topk: number of experts to choose per token.
    renormalize: normalize gating_output.
    mesh: mesh to perform moe.
    activation: activation function to perform on the output of w1.
    scoring_fn: scoring function to apply on gating_output.
    tp_axis: tensor parallelism axis name.
    ep_axis: expert parallelism axis name.
    use_sp: use sequence parallelism.
    reduce_output: reduce output across devices.
    moe_plan: optional MoE layer execution plan.
    gmm_plan: optional GMM kernel execution plan.
    shard_idx: optional int32 [num_shards] that holds each shard's index.

  Returns:
    Output of moe operation. [num_tokens, hidden_size] if reduce_output=True,
      else [num_shards * num_tokens, hidden_size].
  """
  topk_weights = apply_scoring_fn(scoring_fn, gating_output)
  topk_weights, topk_indices = jax.lax.top_k(topk_weights, k=topk)
  if renormalize:
    topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)

  return modular.routing_moe(
      hidden_states=hidden_states,
      topk_weights=topk_weights,
      topk_indices=topk_indices,
      w1=w1,
      w2=w2,
      w1_scale=w1_scale,
      w2_scale=w2_scale,
      w1_bias=w1_bias,
      w2_bias=w2_bias,
      mesh=mesh,
      activation=activation,
      tp_axis=tp_axis,
      ep_axis=ep_axis,
      use_sp=use_sp,
      reduce_output=reduce_output,
      moe_plan=moe_plan,
      gmm_plan=gmm_plan,
      shard_idx=shard_idx,
  )
