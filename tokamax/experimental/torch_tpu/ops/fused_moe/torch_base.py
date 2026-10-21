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
"""Base class for PyTorch interfaces to Tokamax Fused MoE operators."""

from typing import Any, TypeVar, override
import jax
from tokamax._src.ops.experimental.fused_moe import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch

_Config = TypeVar("_Config")


class _FusedMoe(torch_op.TorchOp[_Config]):
  """Base class for Fused MoE operators."""

  def __init__(self) -> None:
    super().__init__()
    self.jax_op_name = "base_fused_moe"
    self.op_impl_jax = jax_base.FusedMoe()
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      x: torch.Tensor,
      *args: Any,
      **kwargs: Any,
  ) -> torch.Tensor:
    """Meta/fake implementation for symbolic shape tracing in torch.compile."""
    del args, kwargs
    return torch.empty_like(x)

  @override
  def __call__(
      self,
      x: torch.Tensor,
      w1: torch.Tensor,
      w2: torch.Tensor,
      gating: torch.Tensor,
      w1_scale: torch.Tensor | None = None,
      w2_scale: torch.Tensor | None = None,
      w1_bias: torch.Tensor | None = None,
      w2_bias: torch.Tensor | None = None,
      *,
      topk: int = 2,
      renormalize: bool = True,
      act_fn: str = "silu",
      config: _Config | None = None,
  ) -> torch.Tensor:
    del config
    assert self._torch_tokamax_op is not None, "Forward op not registered."
    return self._torch_tokamax_op(
        x,
        w1,
        w2,
        gating,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
    )

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      w1: jax.Array,
      w2: jax.Array,
      gating: jax.Array,
      w1_scale: jax.Array | None = None,
      w2_scale: jax.Array | None = None,
      w1_bias: jax.Array | None = None,
      w2_bias: jax.Array | None = None,
      topk: int = 2,
      renormalize: bool = True,
      act_fn: str = "silu",
  ) -> jax.Array:
    assert self.op_impl_jax is not None, "Forward class not set."
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        w1,
        w2,
        gating,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
        config=None,
    )
    return out


FusedMoe = _FusedMoe()  # pylint: disable=invalid-name
