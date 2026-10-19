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

"""Reference MoE implementation based on gmm_v2."""

import functools

import jax
from jax import sharding
import jax.numpy as jnp
from tokamax._src.ops.gmm_v2 import gmm_v2
from tokamax._src.ops.ragged_gather.pallas_mosaic_tpu_kernel import ragged_gather_pallas
from tokamax._src.ops.ragged_gather_reduce.pallas_mosaic_tpu_kernel import ragged_gather_reduce

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


def gmm_wrapper(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    fuse_act: str | None = None,
) -> jax.Array:
  return gmm_v2.gmm_v2(
      lhs=lhs,
      rhs=rhs,
      rhs_scale=rhs_scale,
      rhs_bias=rhs_bias,
      group_sizes=group_sizes,
      group_offset=group_offset[0],
      # ragged_gather_reduce multiplies other shards' rows by 0 instead of
      # skipping them, so a NaN left in uninitialized rows would survive.
      zero_initialize=True,
      fuse_act=fuse_act,
  )


def valid_rows_mask(
    batch_size: int,
    group_sizes: jax.Array,
    group_start: jax.Array,
    group_end: jax.Array,
) -> jax.Array:
  """Mask indicating rows processed by current shard."""
  group_sizes_sum = jnp.cumulative_sum(group_sizes, include_initial=True)

  token_start = group_sizes_sum[group_start]
  token_end = group_sizes_sum[group_end]

  index = jnp.arange(batch_size)
  return jnp.where(
      jnp.logical_and(token_start <= index, index < token_end), True, False
  )


def moe_gmm_local(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    topk_argsort_revert_indices: jax.Array,
    topk_weights: jax.Array,
    *,
    activation: str,
    topk: int,
    ep_axis: str | None = None,
    tp_axis: str | None = None,
    use_sp: bool = False,
) -> jax.Array:
  """Main MoE logic on a local shard can run in TP or EP mode."""
  if tp_axis is not None and ep_axis is not None:
    raise NotImplementedError("Hybrid parallelism is not yet supported.")

  # GMM1 computes x @ (W_up | W_gate) together and activation, output is
  # [tokens, padded_intermediate_size]
  gmm1_res = gmm_wrapper(
      x,
      w1,
      w1_scale,
      w1_bias,
      group_sizes,
      group_offset,
      fuse_act=activation,
  )

  # When the parallelism is TP since w2_bias is not sharded, we should only
  # apply bias once, not applying to every shard. So we set w2_bias to 0 to all
  # shards other than shard 0. For EP, it is not needed since bias is sharded on
  # leading expert axis.
  if tp_axis is not None and w2_bias is not None:
    shard_id = jax.lax.axis_index(tp_axis).sum()
    w2_bias = jnp.where(shard_id == 0, w2_bias, 0)
  gmm1_res = gmm1_res[:, : w2.shape[1]]  # trim to hidden size if padded
  gmm2_res = gmm_wrapper(
      gmm1_res, w2, w2_scale, w2_bias, group_sizes, group_offset
  )

  batch_size = gmm2_res.shape[0]
  local_group_size = w1.shape[0]

  if local_group_size < group_sizes.size:
    mask = valid_rows_mask(
        gmm1_res.shape[0],
        group_sizes,
        group_offset,
        group_offset + local_group_size,
    )[topk_argsort_revert_indices].reshape(-1, topk, 1)
  else:
    mask = jnp.full((batch_size,), True).reshape(-1, topk, 1)

  reduction_axis = tp_axis if tp_axis is not None else ep_axis

  out = ragged_gather_reduce(
      gmm2_res,
      topk_argsort_revert_indices,
      topk_weights.reshape(-1),
      mask.reshape(-1),
      topk,
  )
  if use_sp:
    out = jax.lax.psum_scatter(
        out, axis_name=reduction_axis, scatter_dimension=0, tiled=True
    )
  else:
    out = jax.lax.psum(out, axis_name=reduction_axis)
  return out


def tensor_parallel_gmm(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    topk_argsort_revert_indices: jax.Array,
    topk_weights: jax.Array,
    *,
    activation: str,
    topk: int,
    mesh: Mesh,
    tp_axis: str | None,
    use_sp: bool = False,
) -> jax.Array:
  """Tensor-parallel GMM execution across the mesh."""
  data_p_spec = P(None)
  out_p_spec = P(tp_axis) if use_sp else data_p_spec
  group_offset = jnp.array([0])

  w1_spec = P(None, None, tp_axis)
  w2_spec = P(None, tp_axis, None)

  w1_scale_spec = None if w1_scale is None else P(None, None, None, tp_axis)
  w1_bias_spec = None if w1_bias is None else P(None, None, tp_axis)

  num_blocks = 1 if w2_scale is None else w2_scale.shape[1]
  w2_scale_spec = None if num_blocks == 1 else P(None, tp_axis, None, None)
  w2_bias_spec = None if w2_bias is None else P(None, None, None)

  return jax.shard_map(
      functools.partial(
          moe_gmm_local,
          activation=activation,
          topk=topk,
          tp_axis=tp_axis,
          use_sp=use_sp,
      ),
      mesh=mesh,
      in_specs=(
          data_p_spec,
          w1_spec,
          w1_scale_spec,
          w1_bias_spec,
          w2_spec,
          w2_scale_spec,
          w2_bias_spec,
          data_p_spec,
          P(),
          data_p_spec,
          data_p_spec,
      ),
      out_specs=(out_p_spec),
      check_vma=False,
  )(
      x,
      w1,
      w1_scale,
      w1_bias,
      w2,
      w2_scale,
      w2_bias,
      group_sizes,
      group_offset,
      topk_argsort_revert_indices,
      topk_weights,
  )


def expert_parallel_gmm(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    topk_argsort_revert_indices: jax.Array,
    topk_weights: jax.Array,
    *,
    activation: str,
    topk: int,
    mesh: Mesh,
    ep_axis: str,
    use_sp: bool = False,
) -> jax.Array:
  """Expert-parallel GMM execution across the mesh."""
  ep_size = mesh.shape[ep_axis]
  ep_p_spec = P(ep_axis)
  data_p_spec = P(None)
  out_p_spec = P(ep_axis) if use_sp else data_p_spec
  num_experts = w1.shape[0]
  num_experts_per_shard = num_experts // ep_size
  group_offset = jnp.arange(0, num_experts, num_experts_per_shard)

  w1_scale_spec = None if w1_scale is None else ep_p_spec
  w1_bias_spec = None if w1_bias is None else ep_p_spec
  w2_scale_spec = None if w2_scale is None else ep_p_spec
  w2_bias_spec = None if w2_bias is None else ep_p_spec

  return jax.shard_map(
      functools.partial(
          moe_gmm_local,
          activation=activation,
          topk=topk,
          ep_axis=ep_axis,
          use_sp=use_sp,
      ),
      mesh=mesh,
      in_specs=(
          data_p_spec,
          ep_p_spec,
          w1_scale_spec,
          w1_bias_spec,
          ep_p_spec,
          w2_scale_spec,
          w2_bias_spec,
          data_p_spec,
          ep_p_spec,
          data_p_spec,
          data_p_spec,
      ),
      out_specs=(out_p_spec),
      check_vma=False,
  )(
      x,
      w1,
      w1_scale,
      w1_bias,
      w2,
      w2_scale,
      w2_bias,
      group_sizes,
      group_offset,
      topk_argsort_revert_indices,
      topk_weights,
  )


@jax.jit(
    static_argnames=(
        "topk",
        "renormalize",
        "mesh",
        "ep_axis",
        "tp_axis",
        "use_sp",
        "activation",
        "scoring_fn",
    )
)
def reference_moe(
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

  Returns:
      Output of moe operation [num_tokens, hidden_size]
  """
  if tp_axis is not None and ep_axis is not None:
    raise NotImplementedError("Hybrid parallelism is not yet supported.")

  num_tokens, hidden_size = hidden_states.shape
  global_num_experts, padded_hidden_size, _ = w1.shape
  dtype = hidden_states.dtype

  assert (num_tokens * topk) % 16 == 0, (
      "The kernel requires num_tokens * topk to be a multiple of "
      f"16 but got {num_tokens}*{topk}={num_tokens*topk}"
  )

  assert gating_output.shape == (num_tokens, global_num_experts)

  topk_weights = apply_scoring_fn(scoring_fn, gating_output)
  topk_weights, topk_indices = jax.lax.top_k(topk_weights, k=topk)
  if renormalize:
    topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
  topk_weights = topk_weights.astype(dtype)
  # All-gather topk weights for attention dp
  topk_weights = jax.lax.with_sharding_constraint(
      topk_weights,
      NamedSharding(mesh, P(None, None)),
  )

  def _process_tokens_locally(hidden_states_local, topk_indices_local):
    num_tokens_local = hidden_states_local.shape[0]
    topk_indices_flat = topk_indices_local.flatten()
    topk_argsort_indices = jnp.argsort(topk_indices_flat)
    token_indices = jnp.arange(num_tokens_local, dtype=jnp.int32).repeat(topk)
    token_indices_sorted = token_indices[topk_argsort_indices]
    # Below one_hot is equivalent to jnp.bincount(topk_indices_flat,
    # length=global_num_experts) but is more performant.
    group_sizes_local = jax.nn.one_hot(
        topk_indices_flat, global_num_experts, dtype=jnp.int32
    ).sum(axis=0)
    topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)

    if ep_axis is not None:
      num_ep_shard = mesh.shape[ep_axis]
      local_num_experts = global_num_experts // num_ep_shard
      shard_idx = jax.lax.axis_index(ep_axis)

      experts_start = shard_idx * local_num_experts
      experts_end = experts_start + local_num_experts
    else:
      experts_start = 0
      experts_end = global_num_experts

    group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)
    shard_output_start = group_offsets[experts_start]
    shard_output_end = group_offsets[experts_end]
    x = ragged_gather_pallas(
        hidden_states_local,
        token_indices_sorted,
        shard_output_start,
        shard_output_end,
    )

    return x, group_sizes_local, topk_argsort_revert_indices

  x, group_sizes, topk_argsort_revert_indices = jax.shard_map(
      _process_tokens_locally,
      mesh=mesh,
      in_specs=(
          P(None, None),
          P(None, None),
      ),
      out_specs=(
          P(None),
          P(None),
          P(None),
      ),
      check_vma=False,
  )(hidden_states, topk_indices)

  x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))

  if ep_axis is not None:
    x = expert_parallel_gmm(
        x,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        topk_argsort_revert_indices,
        topk_weights,
        activation=activation,
        topk=topk,
        mesh=mesh,
        ep_axis=ep_axis,
        use_sp=use_sp,
    )
  else:
    x = tensor_parallel_gmm(
        x,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        topk_argsort_revert_indices,
        topk_weights,
        activation=activation,
        topk=topk,
        mesh=mesh,
        tp_axis=tp_axis,
        use_sp=use_sp,
    )

  return x[:num_tokens, :hidden_size]
