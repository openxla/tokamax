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
"""Base class for PyTorch interfaces to Tokamax Routing MoE operators."""

from typing import Any, TypeVar, override
import jax
from tokamax._src.ops.experimental.routing_moe import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch

_Config = TypeVar("_Config")


class _RoutingMoe(torch_op.TorchOp[_Config]):
  """Base class for Tokamax Routing MoE PyTorch operators."""

  def __init__(self) -> None:
    super().__init__()
    self.jax_op_name = "base_routing_moe"
    self.op_impl_jax = jax_base.RoutingMoe()
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      x: torch.Tensor,
      *args: Any,
      **kwargs: Any,
  ) -> torch.Tensor:
    """Meta implementation preserving symbolic token dimensions."""
    del args, kwargs
    return torch.empty_like(x)

  @override
  def __call__(
      self,
      x: torch.Tensor,
      topk_weights: torch.Tensor,
      topk_indices: torch.Tensor,
      w1: torch.Tensor,
      w2: torch.Tensor,
      w1_scale: torch.Tensor | None = None,
      w2_scale: torch.Tensor | None = None,
      w1_bias: torch.Tensor | None = None,
      w2_bias: torch.Tensor | None = None,
      experts_start: torch.Tensor | None = None,
      *,
      global_num_experts: int | None = None,
      act_fn: str = "silu",
      skip_padded_tokens: bool = False,
      config: _Config | None = None,
  ) -> torch.Tensor:
    del config
    assert self._torch_tokamax_op is not None, "Forward op not registered."
    return self._torch_tokamax_op(
        x,
        topk_weights,
        topk_indices,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        experts_start,
        global_num_experts=global_num_experts,
        act_fn=act_fn,
        skip_padded_tokens=skip_padded_tokens,
    )

  @override
  def op_impl_call(
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
      global_num_experts: int | None = None,
      act_fn: str = "silu",
      skip_padded_tokens: bool = False,
  ) -> jax.Array:
    assert self.op_impl_jax is not None, "Forward class not set."
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        topk_weights,
        topk_indices,
        w1,
        w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        experts_start=experts_start,
        global_num_experts=global_num_experts,
        act_fn=act_fn,
        skip_padded_tokens=skip_padded_tokens,
        config=None,
    )
    return out


RoutingMoe = _RoutingMoe()  # pylint: disable=invalid-name
