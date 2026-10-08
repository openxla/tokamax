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
"""CSA Gather PyTorch Op API using reference implementation."""

from typing import Any, TypeVar
import jax
from tokamax._src.ops.experimental.tpu.csa_gather import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch
from typing_extensions import override

_Config = TypeVar("_Config")


class _CsaGather(torch_op.TorchOp[_Config]):
  """CSA Gather PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.CsaGather()
    self.jax_op_name = "base_csa_gather"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      nope_cache: torch.Tensor,
      rope_cache: torch.Tensor,
      indices: torch.Tensor,
      num_valid_indices: torch.Tensor | None = None,
      top_k: int = 1024,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del num_valid_indices, top_k, return_residuals, config
    n = indices.shape[0]
    return (
        nope_cache.new_empty((n, 128)),
        rope_cache.new_empty((n // 4, 128)),
    )

  @override
  def __call__(
      self,
      nope_cache: torch.Tensor,
      rope_cache: torch.Tensor,
      indices: torch.Tensor,
      num_valid_indices: torch.Tensor | None = None,
      top_k: int = 1024,
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del configs  # Unused.
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        nope_cache,
        rope_cache,
        indices,
        num_valid_indices,
        top_k=top_k,
        return_residuals=return_residuals,
        config=None,
    )

  @override
  def op_impl_call(
      self,
      nope_cache: jax.Array,
      rope_cache: jax.Array,
      indices: jax.Array,
      num_valid_indices: jax.Array | None = None,
      top_k: int = 1024,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    (nope_out, rope_out), _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        nope_cache,
        rope_cache,
        indices,
        num_valid_indices,
        top_k=top_k,
        return_residuals=return_residuals,
        config=None,
    )
    return nope_out, rope_out


# Singleton instance of CsaGather.
CsaGather = _CsaGather[Any]()
