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
"""Base class for PyTorch interfaces to Tokamax Ragged Gather Reduce operators."""

from typing import TypeVar, override
import jax
from tokamax._src.ops.ragged_gather_reduce import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch

_Config = TypeVar("_Config")


class _RaggedGatherReduce(torch_op.TorchOp[_Config]):
  """Base class for Ragged Gather Reduce operators."""

  def __init__(self) -> None:
    super().__init__()
    self.jax_op_name = "base_ragged_gather_reduce"
    self.op_impl_jax = jax_base.RaggedGatherReduce()
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      x: torch.Tensor,
      indices: torch.Tensor,
      topk_weights: torch.Tensor,
      valid_rows_mask: torch.Tensor,
      reduce_group_size: int,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> torch.Tensor:
    del topk_weights, valid_rows_mask, return_residuals, config
    return torch.empty(
        (indices.shape[0] // reduce_group_size, x.shape[-1]),
        dtype=x.dtype,
        device=x.device,
    )

  @override
  def __call__(
      self,
      x: torch.Tensor,
      indices: torch.Tensor,
      topk_weights: torch.Tensor,
      valid_rows_mask: torch.Tensor,
      reduce_group_size: int,
      return_residuals: bool = False,
      config: _Config | None = None,
  ) -> torch.Tensor:
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. self.op_impl_jax was not set in the"
        " constructor."
    )
    return self._torch_tokamax_op(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
        return_residuals=return_residuals,
        config=self.deconstruct_config(config),
    )

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      indices: jax.Array,
      topk_weights: jax.Array,
      valid_rows_mask: jax.Array,
      reduce_group_size: int,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    assert self.op_impl_jax is not None, "Forward class not set."
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
        return_residuals=return_residuals,
        config=kernel_config,
    )
    return out


RaggedGatherReduce = _RaggedGatherReduce()  # pylint: disable=invalid-name
