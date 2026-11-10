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
"""Routing MoE base operator class."""

from typing import Any, TypeVar
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.routing_moe_op import reference
from typing_extensions import override

_Config = TypeVar("_Config")


class RoutingMoe(op.Op[Any, jax.Array, None, _Config, Any]):
  """Base operator interface for Tokamax Routing MoE."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      hidden_states: Float[Array, "T H"],
      topk_weights: Float[Array, "T K"],
      topk_indices: Int[Array, "T K"],
      w1: Array,
      w2: Array,
      w1_scale: Array | None = None,
      w2_scale: Array | None = None,
      w1_bias: Array | None = None,
      w2_bias: Array | None = None,
      mesh: jax.sharding.Mesh | None = None,
      activation: str = "silu",
      tp_axis: str | None = None,
      ep_axis: str | None = None,
      use_sp: bool = False,
      reduce_output: bool = True,
      *,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds arguments for Routing MoE operator."""
    if topk_weights.shape != topk_indices.shape:
      raise ValueError(
          f"topk_weights.shape {topk_weights.shape} does not match"
          f" topk_indices.shape {topk_indices.shape}."
      )
    if not jnp.issubdtype(topk_indices.dtype, jnp.integer):
      raise TypeError(
          f"topk_indices must have integer dtype, got {topk_indices.dtype}."
      )
    if topk_weights.shape[0] != hidden_states.shape[0]:
      raise ValueError(
          f"topk_weights tokens {topk_weights.shape[0]} does not match"
          f" hidden_states tokens {hidden_states.shape[0]}."
      )
    return super().bind(
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
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      hidden_states: Float[Array, "T H"],
      topk_weights: Float[Array, "T K"],
      topk_indices: Int[Array, "T K"],
      w1: Array,
      w2: Array,
      w1_scale: Array | None = None,
      w2_scale: Array | None = None,
      w1_bias: Array | None = None,
      w2_bias: Array | None = None,
      mesh: jax.sharding.Mesh | None = None,
      activation: str = "silu",
      tp_axis: str | None = None,
      ep_axis: str | None = None,
      use_sp: bool = False,
      reduce_output: bool = True,
      *,
      return_residuals: bool = False,
      config: _Config | None = None,
  ) -> tuple[jax.Array, None]:
    del config, return_residuals
    if tp_axis is not None:
      raise NotImplementedError(
          f"XLA RoutingMoe does not support tp_axis={tp_axis!r}."
      )
    return reference.routing_moe_reference(
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
    ), None
