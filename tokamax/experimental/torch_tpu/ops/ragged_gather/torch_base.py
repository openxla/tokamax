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
"""Ragged Gather PyTorch Op API using reference implementation."""

from typing import Any, TypeVar
import jax
from tokamax._src.ops.ragged_gather import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch
from typing_extensions import override

_Config = TypeVar("_Config")


class _RaggedGather(torch_op.TorchOp[_Config]):
  """Ragged Gather PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.RaggedGather()
    self.jax_op_name = "base_ragged_gather"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      x: torch.Tensor,
      indices: torch.Tensor,
      start: torch.Tensor,
      end: torch.Tensor,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> torch.Tensor:
    del start, end, return_residuals, config
    return x.new_empty((indices.shape[0], *x.shape[1:]))

  @override
  def __call__(
      self,
      x: torch.Tensor,
      indices: torch.Tensor,
      start: torch.Tensor,
      end: torch.Tensor,
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
  ) -> torch.Tensor:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        x,
        indices,
        start,
        end,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      indices: jax.Array,
      start: jax.Array,
      end: jax.Array,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    del config
    assert self.op_impl_jax is not None, "Forward class not set."
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        indices,
        start,
        end,
        return_residuals=return_residuals,
        config=None,
    )
    return out


# Singleton instance of RaggedGather.
RaggedGather = _RaggedGather[Any]()
