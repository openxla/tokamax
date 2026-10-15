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
"""Base class for PyTorch interfaces to Tokamax ProjAndSaveState operators."""

from collections.abc import Sequence
from typing import Any, TypeVar, override

import jax
from tokamax._src.ops.proj_and_save_state import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch

_Config = TypeVar("_Config")


class _ProjAndSaveState(torch_op.TorchOp[_Config]):
  """ProjAndSaveState PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.ProjAndSaveState()
    self.jax_op_name = "base_proj_and_save_state"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      hidden_states: torch.Tensor,
      wkv_wgate: torch.Tensor,
      ape: torch.Tensor,
      positions: torch.Tensor,
      slot_mapping: torch.Tensor,
      cache: torch.Tensor,
      compress_ratio: int,
      return_residuals: bool = False,
      config: Sequence[int] | None = None,
  ) -> torch.Tensor:
    """Meta/fake implementation for symbolic shape tracing in torch.compile."""
    del hidden_states, wkv_wgate, ape, positions, slot_mapping
    del compress_ratio, return_residuals, config
    return torch.empty_like(cache)

  @override
  def __call__(
      self,
      hidden_states: torch.Tensor,
      wkv_wgate: torch.Tensor,
      ape: torch.Tensor,
      positions: torch.Tensor,
      slot_mapping: torch.Tensor,
      cache: torch.Tensor,
      compress_ratio: int,
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
  ) -> torch.Tensor:
    del configs  # Unused.
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        hidden_states,
        wkv_wgate,
        ape,
        positions,
        slot_mapping,
        cache,
        compress_ratio=compress_ratio,
        return_residuals=return_residuals,
        config=None,
    )

  @override
  def op_impl_call(
      self,
      hidden_states: jax.Array,
      wkv_wgate: jax.Array,
      ape: jax.Array,
      positions: jax.Array,
      slot_mapping: jax.Array,
      cache: jax.Array,
      compress_ratio: int,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        hidden_states,
        wkv_wgate,
        ape,
        positions,
        slot_mapping,
        cache,
        compress_ratio=compress_ratio,
        return_residuals=return_residuals,
        config=None,
    )
    return out


# Singleton instance of ProjAndSaveState.
ProjAndSaveState = _ProjAndSaveState[Any]()  # pylint: disable=invalid-name
