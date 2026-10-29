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
"""Per-shard input preparation and sharded runner for routing_gmm tests."""

import dataclasses
from typing import Self

import jax
from jax import sharding
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.gmm import kernel
from tokamax._src.ops.experimental.routing_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases
from tokamax._src.ops.experimental.routing_moe.test.common import test_inputs
from tokamax._src.ops.experimental.routing_moe.test.common import test_util

P = sharding.PartitionSpec


def _local_weight(
    w: jax.Array,
    *,
    group_offset: int,
    num_local_experts: int,
    quantize: bool,
    fused_gate_up: bool = False,
    quant_block_size: int | None = None,
) -> arg.SingleWeightWrapper[jax.Array]:
  """Slices this shard's experts from `w` and optionally quantizes to float8."""
  w = w[group_offset : group_offset + num_local_experts]
  scale = None
  if quantize:
    w, scale = test_inputs.sub_channel_quantize(
        w, jnp.float8_e4m3fn, quant_block_size
    )
  return arg.make_weight_wrapper(w, scale, fused_gate_up=fused_gate_up)


@dataclasses.dataclass(frozen=True)
class GmmCase:
  """Inputs of a gate/up then down routing_gmm pair on one shard.

  Attributes:
    x: `[num_tokens, size_k]` activations in token order.
    w1: `[num_local_experts, size_k, size_n]` fused gate/up weights.
    w2: `[num_local_experts, size_n // 2, size_k]` down weights.
    routing: Routing of the rows, restricted to this shard's experts.
  """

  x: jax.Array
  w1: arg.SingleWeightWrapper[jax.Array]
  w2: arg.SingleWeightWrapper[jax.Array]
  routing: test_inputs.Routing

  @property
  def x_sorted(self) -> jax.Array:
    """`x` permuted into sorted-row order."""
    return self.x[self.routing.gather_indices]

  @property
  def x_compact(self) -> arg.ActivationWrapper[jax.Array]:
    """`x` in the compact layout."""
    return input_pack.input_pack(self.x)

  @property
  def x_sorted_compact(self) -> arg.ActivationWrapper[jax.Array]:
    """`x_sorted` in the compact layout."""
    return input_pack.input_pack(self.x_sorted)

  @classmethod
  def make_case(
      cls,
      case: test_cases.Case,
      *,
      quant_w1: bool = False,
      quant_w2: bool = False,
      group_offset: int = 0,
      num_skipped: int = 0,
      balanced_ep: int | None = None,
  ) -> Self:
    """Builds the layer inputs of `case` for the shard at `group_offset`."""
    test_util.check_gmm_supported(case)
    shape = dataclasses.replace(
        case.shape, intermediate_size=case.local_intermediate_size
    )
    quant_shape = shape.w_dtype is not None
    layer_inputs = test_inputs.MoEInputs.generate(
        shape,
        case.num_tokens,
        active_expert_ratio=case.active_expert_ratio,
        num_skipped=num_skipped,
        force_expert=None if num_skipped or balanced_ep else group_offset,
    )
    if balanced_ep is not None:
      topk_indices = test_inputs.balanced_topk_indices(
          jax.random.key(1),
          case.num_tokens,
          shape.top_k,
          shape.num_experts,
          balanced_ep,
      )
    else:
      _, topk_indices = test_inputs.top_k(
          layer_inputs.gating_output, shape.top_k
      )
    local = dict(
        group_offset=group_offset,
        num_local_experts=case.num_local_experts,
    )
    return cls(
        x=layer_inputs.hidden_states,
        w1=_local_weight(
            layer_inputs.w1,
            quantize=quant_w1 or quant_shape,
            fused_gate_up=True,
            **local,
            quant_block_size=shape.w_quant_block_size,
        ),
        w2=_local_weight(
            layer_inputs.w2,
            quantize=quant_w2 or quant_shape,
            **local,
            quant_block_size=shape.w_quant_block_size,
        ),
        routing=test_inputs.Routing.make_routing(
            topk_indices, shape.num_experts, **local
        ),
    )


def to_large_layout(out: arg.ActivationWrapper[jax.Array]) -> jax.Array:
  """Reshapes `out` to the `[rows, cols]` layout."""
  return out.data.reshape(out.spec.large_shape)


@jax.jit(static_argnames=("mesh", "topk", "remote_scatter", "all_to_all"))
def sharded_gmm_unpermute(
    x: arg.ActivationWrapper[jax.Array],
    w: jax.Array,
    group_sizes: jax.Array,
    scatter_indices: jax.Array,
    *,
    mesh: sharding.Mesh,
    topk: int | None = None,
    remote_scatter: bool = True,
    all_to_all: bool = False,
) -> jax.Array:
  """Runs EP-sharded `routing_gmm` scatter, with fused or unfused all-to-all."""
  axis = test_cases.EP_AXIS
  ep_size = mesh.shape[axis]
  num_experts = w.shape[0]
  group_offset = jnp.arange(0, num_experts, num_experts // ep_size)
  remote_axis = axis if remote_scatter else None

  def unpermute(
      x: arg.ActivationWrapper[jax.Array],
      w: jax.Array,
      group_sizes: jax.Array,
      offset: jax.Array,
      scatter_indices: jax.Array,
  ) -> jax.Array:
    out = kernel.routing_gmm(
        x,
        arg.make_weight_wrapper(w),
        group_sizes,
        group_offset=offset,
        scatter=arg.make_routing_wrapper(
            scatter_indices, remote_axis=remote_axis
        ),
    ).data
    if remote_scatter or not all_to_all:
      return out
    assert topk is not None
    num_tokens, topk_per_shard = x.spec.row_size // topk, topk // ep_size
    row_shape = out.shape[1:]
    out = out[: num_tokens * topk_per_shard]
    out = out.reshape((num_tokens, topk_per_shard) + row_shape)
    out = jax.lax.all_to_all(out, axis, split_axis=0, concat_axis=1, tiled=True)
    return out.reshape((-1,) + row_shape)

  return jax.shard_map(
      unpermute,
      mesh=mesh,
      in_specs=(P(), P(axis), P(), P(axis), P()),
      out_specs=P(axis),
      check_vma=False,
  )(x, w, group_sizes, group_offset, scatter_indices)
