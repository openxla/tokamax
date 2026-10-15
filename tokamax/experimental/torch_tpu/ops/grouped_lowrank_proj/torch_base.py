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
"""GroupedLowrankProj PyTorch Op API using reference implementation."""

from typing import Any, TypeVar, override

import jax
from tokamax._src.ops.grouped_lowrank_proj import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch

_Config = TypeVar("_Config")


class _GroupedLowrankProj(torch_op.TorchOp[_Config]):
  """GroupedLowrankProj PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.GroupedLowrankProj()
    self.jax_op_name = "base_grouped_lowrank_proj"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      x: torch.Tensor,
      positions: torch.Tensor,
      cos_sin_cache: torch.Tensor,
      wo_a: torch.Tensor,
      wo_a_scale: torch.Tensor,
      inverse: bool = True,
      quantize_activations: bool = True,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> torch.Tensor:
    del positions, cos_sin_cache, wo_a_scale, inverse
    del quantize_activations, return_residuals, config
    return x.new_empty((x.shape[0], wo_a.shape[1]))

  @override
  def __call__(
      self,
      x: torch.Tensor,
      positions: torch.Tensor,
      cos_sin_cache: torch.Tensor,
      wo_a: torch.Tensor,
      wo_a_scale: torch.Tensor,
      inverse: bool = True,
      quantize_activations: bool = True,
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
  ) -> torch.Tensor:
    del configs  # Unused.
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        x,
        positions,
        cos_sin_cache,
        wo_a,
        wo_a_scale,
        inverse=inverse,
        quantize_activations=quantize_activations,
        return_residuals=return_residuals,
        config=None,
    )

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      positions: jax.Array,
      cos_sin_cache: jax.Array,
      wo_a: jax.Array,
      wo_a_scale: jax.Array,
      inverse: bool = True,
      quantize_activations: bool = True,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    del config  # Unused.
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        positions,
        cos_sin_cache,
        wo_a,
        wo_a_scale,
        inverse=inverse,
        quantize_activations=quantize_activations,
        return_residuals=return_residuals,
        config=None,
    )
    return out


# Singleton instance of GroupedLowrankProj.
GroupedLowrankProj = _GroupedLowrankProj[Any]()
