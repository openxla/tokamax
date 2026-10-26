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
"""Runner for routed and shared MoE layers in tests and benchmarks."""

import functools

import jax
from jax import sharding
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.layer import modular
from tokamax._src.ops.experimental.routing_moe.layer import monolithic
from tokamax._src.ops.experimental.routing_moe.layer import plan as moe_plan_lib
from tokamax._src.ops.experimental.routing_moe.test import moe_test_reference

P = sharding.PartitionSpec
NamedSharding = sharding.NamedSharding
Mesh = sharding.Mesh


def _apply_act_fn(acc: jax.Array, fuse_act: str | None) -> jax.Array:
  """Optionally applies a fused activation function to the accumulator."""
  if fuse_act is None:
    return acc

  acc_gate, acc_up = jnp.split(acc, 2, axis=-1)
  match fuse_act:
    case "silu":
      return jax.nn.silu(acc_gate) * acc_up
    case "gelu":
      return jax.nn.gelu(acc_gate) * acc_up
    case "gelu_tanh":
      return jax.nn.gelu(acc_gate, approximate=True) * acc_up
    case _:
      raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def _quantized_matmul(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array | None,
) -> jax.Array:
  """Computes a dynamically quantized matmul for a single-expert weight."""
  num_groups = w.shape[0]
  assert num_groups == 1, f"Shared expert has group size of {num_groups}"
  w = w[0]
  if w_scale is None:
    return jnp.matmul(x, w, preferred_element_type=jnp.float32).astype(x.dtype)
  assert w_scale.shape[0] == num_groups
  num_blocks = w_scale.shape[1]
  # Maybe we can use gmm_v2 for this?
  if num_blocks > 1:
    raise NotImplementedError(
        f"Only supports single quant block. Got {num_blocks}"
    )

  q_dtype = w.dtype
  is_float = jnp.issubdtype(q_dtype, jnp.floating)
  dtype_max = float(
      jnp.finfo(q_dtype).max if is_float else jnp.iinfo(q_dtype).max
  )
  acc_dtype = jnp.float32 if is_float else jnp.int32
  x_scale = jnp.max(jnp.abs(x), axis=-1, keepdims=True) / dtype_max
  x_scale = jnp.where(x_scale == 0, 1.0, x_scale)
  x_q = (x / x_scale).astype(q_dtype)
  out = jnp.matmul(x_q, w, preferred_element_type=acc_dtype).astype(jnp.float32)
  return (out * x_scale.astype(jnp.float32) * w_scale[0, 0]).astype(x.dtype)


def _shared_expert(
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w_gate: jax.Array,
    *,
    mesh: Mesh,
    reduction_axis: str | None,
    reduce_output: bool,
    use_sp: bool,
    activation: str,
) -> jax.Array:
  """Computes the unreduced per-shard output of the TP-sharded shared expert."""
  act_spec = P(reduction_axis, None) if use_sp else P(None, None)

  # IFF output is reduced and not sp sharded, it is fully replicated.
  if reduce_output and not use_sp:
    out_spec = P(None, None)
  else:
    # Output holds either partial sum or batch sharded along reduction axis.
    out_spec = P(reduction_axis, None)

  w1_scale_spec = (
      None if w1_scale is None else P(None, None, None, reduction_axis)
  )
  w2_scale_spec = None if w2_scale is None else P(None, None, None, None)

  def _local_fn(x, w1, w2, w1_scale, w2_scale, w_gate):
    if use_sp:
      x = jax.lax.all_gather(x, axis_name=reduction_axis, axis=0, tiled=True)
    out_up = _quantized_matmul(x, w1, w1_scale)
    hidden = _apply_act_fn(out_up, activation).astype(x.dtype)
    down = _quantized_matmul(hidden, w2, w2_scale)
    gate_logits = jnp.matmul(x, w_gate, preferred_element_type=jnp.float32)
    # It seems like basically every model in existance use sigmoid?
    gate_score = jax.nn.sigmoid(gate_logits).astype(x.dtype)
    out = down * gate_score

    if reduce_output:
      if use_sp:
        out = jax.lax.psum_scatter(
            out, axis_name=reduction_axis, scatter_dimension=0, tiled=True
        )
      else:
        out = jax.lax.psum(out, axis_name=reduction_axis)
    return out

  return jax.shard_map(
      _local_fn,
      mesh=mesh,
      in_specs=(
          act_spec,
          P(None, None, reduction_axis),
          P(None, reduction_axis, None),
          w1_scale_spec,
          w2_scale_spec,
          P(None, None),
      ),
      out_specs=out_spec,
      check_vma=False,
  )(x, w1, w2, w1_scale, w2_scale, w_gate)


def _fold_shared_expert(
    hidden_states: jax.Array,
    gating_output: jax.Array,
    w_shared_gate: jax.Array,
    *,
    topk: int,
    renormalize: bool,
    scoring_fn: str,
) -> tuple[jax.Array, jax.Array]:
  """Appends the shared expert to the top-k routing weights and indices."""
  num_experts = gating_output.shape[-1]
  routed_weights = monolithic.apply_scoring_fn(scoring_fn, gating_output)
  routed_weights, routed_indices = jax.lax.top_k(routed_weights, k=topk)
  if renormalize:
    routed_weights /= routed_weights.sum(axis=-1, keepdims=True)

  shared_logits = jnp.matmul(
      hidden_states, w_shared_gate, preferred_element_type=jnp.float32
  )
  shared_weight = jax.nn.sigmoid(shared_logits).astype(routed_weights.dtype)
  shared_indices = jnp.full_like(routed_indices[:, :1], num_experts)
  topk_weights = jnp.concatenate([routed_weights, shared_weight], axis=-1)
  topk_indices = jnp.concatenate([routed_indices, shared_indices], axis=-1)
  return topk_weights, topk_indices


@functools.partial(
    jax.jit,
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
        "remote_scatter",
        "use_reference",
        "fold_shared_expert",
    ),
)
def run_moe(
    hidden_states: jax.Array,
    gating_output: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w1_bias: jax.Array | None = None,
    w2_bias: jax.Array | None = None,
    w1_shared: jax.Array | None = None,
    w2_shared: jax.Array | None = None,
    w1_shared_scale: jax.Array | None = None,
    w2_shared_scale: jax.Array | None = None,
    w_shared_gate: jax.Array | None = None,
    *,
    topk: int,
    renormalize: bool,
    mesh: Mesh,
    activation: str,
    scoring_fn: str,
    tp_axis: str | None,
    ep_axis: str | None,
    use_sp: bool = False,
    reduce_output: bool = True,
    remote_scatter: bool = False,
    use_reference: bool = False,
    fold_shared_expert: bool = False,
) -> jax.Array:
  """Runs routed and optional shared experts with optional outer reduction."""
  common_kwargs = dict(
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
  )
  moe_plan = moe_plan_lib.RoutingMoEPlan(remote_scatter=remote_scatter)

  if fold_shared_expert:
    assert w_shared_gate is not None
    assert w1_shared is None and w2_shared is None
    assert w1.shape[0] == w2.shape[0] == gating_output.shape[-1] + 1
    topk_weights, topk_indices = _fold_shared_expert(
        hidden_states,
        gating_output,
        w_shared_gate,
        topk=topk,
        renormalize=renormalize,
        scoring_fn=scoring_fn,
    )
    if use_reference:
      out = moe_test_reference.reference_moe(
          hidden_states,
          topk_weights,
          topk_indices,
          **common_kwargs,
      )
    else:
      out = modular.routing_moe(
          hidden_states,
          topk_weights,
          topk_indices,
          moe_plan=moe_plan,
          **common_kwargs,
      )
  elif use_reference:
    out = moe_test_reference.gated_reference_moe(
        hidden_states,
        gating_output,
        topk=topk,
        renormalize=renormalize,
        scoring_fn=scoring_fn,
        **common_kwargs,
    )
  else:
    out = monolithic.gated_routing_moe(
        hidden_states,
        gating_output,
        topk=topk,
        renormalize=renormalize,
        scoring_fn=scoring_fn,
        moe_plan=moe_plan,
        **common_kwargs,
    )

  reduction_axis = tp_axis if tp_axis is not None else ep_axis
  if w1_shared is not None:
    assert w2_shared is not None and w_shared_gate is not None
    out += _shared_expert(
        hidden_states,
        w1_shared,
        w2_shared,
        w1_shared_scale,
        w2_shared_scale,
        w_shared_gate,
        mesh=mesh,
        reduction_axis=reduction_axis,
        use_sp=use_sp,
        activation=activation,
        reduce_output=reduce_output,
    )

  if not reduce_output:
    if use_sp:
      reduce_fn = functools.partial(
          jax.lax.psum_scatter,
          axis_name=reduction_axis,
          scatter_dimension=0,
          tiled=True,
      )
      out_spec = P(reduction_axis)
    else:
      reduce_fn = functools.partial(jax.lax.psum, axis_name=reduction_axis)
      out_spec = P(None)
    out = jax.shard_map(
        reduce_fn,
        mesh=mesh,
        in_specs=P(reduction_axis),
        out_specs=out_spec,
    )(out)
  return out
