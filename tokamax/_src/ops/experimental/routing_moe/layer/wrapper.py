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

"""Wrapper module that combines gmm and io relayout kenrels to compute moe."""

import functools

import jax
from jax import numpy as jnp
from jax import sharding
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.gmm import kernel as gmm
from tokamax._src.ops.experimental.routing_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.routing_moe.io_relayout import output_reduce

P = sharding.PartitionSpec
NamedSharding = sharding.NamedSharding
Mesh = sharding.Mesh


def apply_scoring_fn(scoring_fn: str, x: jax.Array) -> jax.Array:
  match scoring_fn:
    case "softmax":
      return jax.nn.softmax(x, axis=-1)
    case "sigmoid":
      return jax.nn.sigmoid(x)
    case _:
      raise NotImplementedError(
          f"FusedMoE does not support {scoring_fn} scoring function"
      )


def routing_moe_local(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    gating_output: jax.Array,
    *,
    activation: str,
    scoring_fn: str,
    topk: int,
    renormalize: bool,
    ep_axis: str | None,
    tp_axis: str | None,
    use_sp: bool,
    remote_combine: bool,
) -> jax.Array:
  """MPMD MoE function that invokes io kernels and routing gmm."""

  local_num_experts = w1.shape[0]
  global_num_experts = gating_output.shape[-1]

  # Step 1: Book keep sharding related logictics.
  if tp_axis is not None and ep_axis is not None:
    raise NotImplementedError("Hybrid parallelism is not yet supported.")
  elif tp_axis is not None:
    reduction_axis = tp_axis
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    reduction_axis = ep_axis
    shard_idx = jax.lax.axis_index(ep_axis)
    group_offset = jnp.array([shard_idx * local_num_experts], dtype=jnp.int32)
    if w2_bias is not None:
      w2_bias = jnp.where(shard_idx == 0, w2_bias, 0)

  # Step 2: Perform local scoring and topk.
  topk_weights = apply_scoring_fn(scoring_fn, gating_output)
  topk_weights, topk_indices = jax.lax.top_k(topk_weights, k=topk)
  if renormalize:
    topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)

  if use_sp:
    # Step 3.0: If input is not replicated (SP), perform all-gather.
    # NOTE: This should be done AFTER topk scoring and reshape to ensure they
    # are done cheaper sharded manner.
    topk_weights = jax.lax.all_gather(
        topk_weights, axis_name=reduction_axis, axis=0, tiled=True
    )
    topk_indices = jax.lax.all_gather(
        topk_indices, axis_name=reduction_axis, axis=0, tiled=True
    )
    x = jax.lax.all_gather(x, axis_name=reduction_axis, axis=0, tiled=True)

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
  topk_argsort_indices = jnp.argsort(topk_indices_flat)
  group_sizes = jnp.bincount(topk_indices_flat, length=global_num_experts)
  # Step 5: Invoke GMM Gate and Up.
  gmm1_res = gmm.routing_gmm(
      x=x_compact,
      w=arg.make_weight_wrapper(w1, w1_scale, w1_bias, fused_gate_up=True),
      group_sizes=group_sizes,
      gather=arg.make_routing_wrapper(topk_argsort_indices // topk),
      fuse_act=activation,
      group_offset=group_offset,
  )

  # Step 6: Invoke GMM Down. With remote_combine, each row is DMAed to the
  # device that owns its token, so every device ends up with all topk rows of
  # its own token shard.
  gmm2_res = gmm.routing_gmm(
      x=gmm1_res,
      w=arg.make_weight_wrapper(w2, w2_scale, w2_bias),
      group_sizes=group_sizes,
      scatter=arg.make_routing_wrapper(
          topk_argsort_indices,
          remote_axis=reduction_axis if remote_combine else None,
      ),
      group_offset=group_offset,
  )

  # Step 7: Postprocess outputs from routing gmm. With remote_combine, the rows
  # are already this device's token shard and are reduced as such.
  out = output_reduce.output_reduce_unpack(
      gmm2_res,
      topk_weights,
      topk_indices,
      group_offset,
      w1.shape[0],
  )

  # Step 8: Combine across devices. A remote combine already left each device
  # with its own tokens' output.
  if remote_combine:
    if use_sp:
      return out
    return jax.lax.all_gather(out, axis_name=reduction_axis, axis=0, tiled=True)
  if use_sp:
    out = jax.lax.psum_scatter(
        out, axis_name=reduction_axis, scatter_dimension=0, tiled=True
    )
  else:
    out = jax.lax.psum(out, axis_name=reduction_axis)
  return out


@jax.jit(
    static_argnames=(
        "topk",
        "renormalize",
        "mesh",
        "ep_axis",
        "tp_axis",
        "use_sp",
        "remote_combine",
        "activation",
        "scoring_fn",
    )
)
def routing_moe(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    gating_output: jax.Array,
    topk: int,
    renormalize: bool,
    mesh: Mesh,
    activation: str,
    scoring_fn: str,
    tp_axis: str | None,
    ep_axis: str | None,
    use_sp: bool = False,
    remote_combine: bool = False,
) -> jax.Array:
  """Route tokens in hidden_states into each experts based on routing.

  Args:
      hidden_states: [num_tokens, hidden_size]
      w1: first moe weights [num_experts, hidden_size, intermediate_size * 2]
      w2: second moe weights [num_experts, intermediate_size, hidden_size]
      w1_scale: w1 scale [num_experts, num_blocks, 1, intermediate_size * 2]
      w2_scale: w2 scale [num_experts, num_blocks, 1, hidden_size]
      w1_bias: optional bias of w1 [num_experts, 1, intermediate_size * 2]
      w2_bias: optional bias of w2 [num_experts, 1, hidden_size]
      gating_output: routing information of tokens [num_tokens, num_experts]
      topk: number of experts to choose per token.
      renormalize: normalize gating_output.
      mesh: mesh to perform moe.
      activation: activation function to perform on the output of w1.
      scoring_fn: scoring function to apply on gating_output.
      tp_axis: tensor parallelism axis name.
      ep_axis: expert parallelism axis name.
      use_sp: use sequence parallelism.
      remote_combine: with ep_axis, DMA each output row straight to the device
        that owns its token instead of reducing across devices.

  Returns:
      Output of moe operation [num_tokens, hidden_size]
  """
  if use_sp:
    if tp_axis is not None and ep_axis is not None:
      raise NotImplementedError("Hybrid parallelism is not yet supported.")
    elif ep_axis is not None:
      act_spec = P(ep_axis)
    else:
      act_spec = P(tp_axis)
  else:
    act_spec = P(None)

  if w1_bias is not None or w2_bias is not None:
    raise NotImplementedError("Bias is currently not supported.")
  if remote_combine and ep_axis is None:
    raise ValueError("remote_combine requires ep_axis.")

  w1_spec = P(ep_axis, None, tp_axis)
  w1_scale_spec = None if w1_scale is None else P(ep_axis, None, None, tp_axis)
  w1_bias_spec = None if w1_bias is None else P(ep_axis, None, tp_axis)
  w2_spec = P(ep_axis, tp_axis, None)
  w2_scale_spec = None if w2_scale is None else P(ep_axis, None, None, None)
  w2_bias_spec = None if w2_bias is None else P(ep_axis, None, None)

  return jax.shard_map(
      functools.partial(
          routing_moe_local,
          activation=activation,
          scoring_fn=scoring_fn,
          topk=topk,
          renormalize=renormalize,
          ep_axis=ep_axis,
          tp_axis=tp_axis,
          use_sp=use_sp,
          remote_combine=remote_combine,
      ),
      mesh=mesh,
      in_specs=(
          act_spec,
          w1_spec,
          w1_scale_spec,
          w1_bias_spec,
          w2_spec,
          w2_scale_spec,
          w2_bias_spec,
          act_spec,
      ),
      out_specs=(act_spec),
      check_vma=False,
  )(hidden_states, w1, w1_scale, w1_bias, w2, w2_scale, w2_bias, gating_output)
