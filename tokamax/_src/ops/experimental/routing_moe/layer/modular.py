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

"""Modular MoE entry point accepting pre-computed top-k weights and indices."""

import functools

import jax
from jax import numpy as jnp
from jax import sharding
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.gmm import kernel as gmm
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan_lib
from tokamax._src.ops.experimental.routing_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.routing_moe.io_relayout import output_reduce
from tokamax._src.ops.experimental.routing_moe.layer import config
from tokamax._src.ops.experimental.routing_moe.layer import plan as moe_plan_lib

P = sharding.PartitionSpec
NamedSharding = sharding.NamedSharding
Mesh = sharding.Mesh


def _routing_moe_local(
    x: jax.Array,
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    ep_shard_idx: jax.Array | None,
    *,
    cfg: config.RoutingMoEConfig,
    gmm_plan: gmm_plan_lib.RoutingGmmPlan | None = None,
) -> jax.Array:
  """MPMD MoE function that invokes io kernels and routing gmm."""

  local_num_experts = w1.shape[0]
  topk = topk_indices.shape[-1]

  # Step 1: Book keep sharding related logictics.
  if cfg.ep_axis is None:
    global_num_experts = local_num_experts
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    num_ep_shards = jax.lax.axis_size(cfg.ep_axis)
    global_num_experts = local_num_experts * num_ep_shards
    if ep_shard_idx is None:
      ep_shard_idx = jax.lax.axis_index(cfg.ep_axis)
    elif not jnp.isscalar(ep_shard_idx):
      ep_shard_idx = ep_shard_idx[0]
    group_offset = jnp.array(
        [ep_shard_idx * local_num_experts], dtype=jnp.int32
    )
    if w2_bias is not None:
      w2_bias = jnp.where(ep_shard_idx == 0, w2_bias, 0)
    if cfg.plan.remote_scatter and not cfg.use_sp:
      num_local_tokens = topk_weights.shape[0] // num_ep_shards
      topk_weights = jax.lax.dynamic_slice_in_dim(
          topk_weights,
          ep_shard_idx * num_local_tokens,
          num_local_tokens,
          axis=0,
      )

  if cfg.use_sp:
    # Step 2: If input is not replicated (SP), perform all-gather.
    if not cfg.plan.remote_scatter:
      topk_weights = jax.lax.all_gather(
          topk_weights, axis_name=cfg.reduction_axis, axis=0, tiled=True
      )
    topk_indices = jax.lax.all_gather(
        topk_indices, axis_name=cfg.reduction_axis, axis=0, tiled=True
    )
    x = jax.lax.all_gather(x, axis_name=cfg.reduction_axis, axis=0, tiled=True)

  # Step 3: Preprocess inputs for routing gmm.
  # NOTE: all-gather only supports large layout. Therefore, performing
  # repacking triggers both input and output relayout. Thus, we need to perform
  # repacking after all-gather.
  # TODO: Support all-gather with compact layout.
  x_compact = input_pack.input_pack(x)

  # Step 4: Sort indices and compute expert group sizes.
  # NOTE: Unlike topk, index sorting has to be done after all-gather to ensure
  # all shards have the same information.
  topk_indices_flat = topk_indices.flatten()
  # One stable sort gives both the order of the routed rows and, as its sorted
  # keys, the group of each sorted row (`row_group`), so the latter is free.
  row_group, topk_argsort_indices = jax.lax.sort_key_val(
      topk_indices_flat,
      jnp.arange(topk_indices_flat.size, dtype=jnp.int32),
      is_stable=True,
  )
  group_sizes = jax.nn.one_hot(
      topk_indices_flat, global_num_experts, dtype=jnp.int32
  ).sum(axis=0)

  # Step 5: Invoke GMM Gate and Up.
  gmm_res = gmm.routing_gmm(
      x=x_compact,
      w=arg.make_weight_wrapper(w1, w1_scale, w1_bias, fused_gate_up=True),
      w2=arg.make_weight_wrapper(w2, w2_scale, w2_bias),
      group_sizes=group_sizes,
      gather=arg.make_routing_wrapper(topk_argsort_indices // topk),
      scatter=arg.make_routing_wrapper(
          topk_argsort_indices,
          remote_axis=cfg.ep_axis if cfg.plan.remote_scatter else None,
      ),
      fuse_act=cfg.activation,
      group_offset=group_offset,
      row_group=row_group,
      custom_plan=gmm_plan,
  )

  # Step 6: Postprocess outputs from routing gmm. With remote_scatter or
  # without EP sharding, every row in gmm_res is valid on this shard.
  if cfg.plan.remote_scatter or cfg.ep_axis is None:
    out = output_reduce.output_reduce_unpack(gmm_res, topk_weights)
  else:
    out = output_reduce.output_reduce_unpack(
        gmm_res,
        topk_weights,
        topk_indices,
        group_offset,
        local_num_experts,
    )

  # Step 7: Combine across devices.
  match cfg.output_collective:
    case config.OutputCollective.NONE:
      return out
    case config.OutputCollective.ALL_GATHER:
      return jax.lax.all_gather(
          out, axis_name=cfg.reduction_axis, axis=0, tiled=True
      )
    case config.OutputCollective.REDUCE_SCATTER:
      return jax.lax.psum_scatter(
          out, axis_name=cfg.reduction_axis, scatter_dimension=0, tiled=True
      )
    case config.OutputCollective.ALL_REDUCE:
      return jax.lax.psum(out, axis_name=cfg.reduction_axis)


@jax.jit(
    static_argnames=(
        "mesh",
        "activation",
        "tp_axis",
        "ep_axis",
        "use_sp",
        "reduce_output",
        "moe_plan",
        "gmm_plan",
    )
)
def routing_moe(
    hidden_states: jax.Array,
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    mesh: Mesh,
    activation: str,
    tp_axis: str | None,
    ep_axis: str | None,
    use_sp: bool = False,
    reduce_output: bool = True,
    moe_plan: moe_plan_lib.RoutingMoEPlan | None = None,
    gmm_plan: gmm_plan_lib.RoutingGmmPlan | None = None,
    ep_shard_idx: jax.Array | None = None,
) -> jax.Array:
  """Route tokens in hidden_states into each experts based on routing.

  Args:
    hidden_states: [num_tokens, hidden_size]
    topk_weights: routing weights of selected experts [num_tokens, topk]
    topk_indices: expert indices of selected experts [num_tokens, topk]
    w1: first moe weights [num_experts, hidden_size, intermediate_size * 2]
    w2: second moe weights [num_experts, intermediate_size, hidden_size]
    w1_scale: w1 scale [num_experts, num_blocks, 1, intermediate_size * 2]
    w2_scale: w2 scale [num_experts, num_blocks, 1, hidden_size]
    w1_bias: optional bias of w1 [num_experts, 1, intermediate_size * 2]
    w2_bias: optional bias of w2 [num_experts, 1, hidden_size]
    mesh: mesh to perform moe.
    activation: activation function to perform on the output of w1.
    tp_axis: tensor parallelism axis name.
    ep_axis: expert parallelism axis name.
    use_sp: use sequence parallelism.
    reduce_output: reduce output across devices.
    moe_plan: optional MoE layer execution plan.
    gmm_plan: optional GMM kernel execution plan.
    ep_shard_idx: optional int32 [num_shards] that holds each ep_shard's index.

  Returns:
    Output of moe operation. [num_tokens, hidden_size] if reduce_output=True,
    else [num_shards * num_tokens, hidden_size].
  """
  if w1_bias is not None or w2_bias is not None:
    raise NotImplementedError("Bias is currently not supported.")

  resolved_plan = moe_plan_lib.ResolvedMoEPlan.create(moe_plan)
  cfg = config.RoutingMoEConfig.create(
      tp_axis=tp_axis,
      ep_axis=ep_axis,
      use_sp=use_sp,
      reduce_output=reduce_output,
      activation=activation,
      plan=resolved_plan,
      mesh=mesh,
  )

  w1_spec = P(ep_axis, None, tp_axis)
  w2_spec = P(ep_axis, tp_axis, None)
  w1_scale_spec = None if w1_scale is None else P(ep_axis, None, None, tp_axis)
  w2_scale_spec = None
  if w2_scale is not None:
    block_axis = None if w2_scale.shape[1] == 1 else tp_axis
    w2_scale_spec = P(ep_axis, block_axis, None, None)
  w1_bias_spec = None if w1_bias is None else P(ep_axis, None, tp_axis)
  w2_bias_spec = None if w2_bias is None else P(ep_axis, None, None)
  ep_shard_idx_spec = None if ep_shard_idx is None else P(ep_axis)

  return jax.shard_map(
      functools.partial(
          _routing_moe_local,
          cfg=cfg,
          gmm_plan=gmm_plan,
      ),
      mesh=mesh,
      in_specs=(
          cfg.act_spec,
          cfg.act_spec,
          cfg.act_spec,
          w1_spec,
          w2_spec,
          w1_scale_spec,
          w2_scale_spec,
          w1_bias_spec,
          w2_bias_spec,
          ep_shard_idx_spec,
      ),
      out_specs=(cfg.out_spec),
      check_vma=False,
  )(
      hidden_states,
      topk_weights,
      topk_indices,
      w1,
      w2,
      w1_scale,
      w2_scale,
      w1_bias,
      w2_bias,
      ep_shard_idx,
  )
