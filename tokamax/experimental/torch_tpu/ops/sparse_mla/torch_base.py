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
"""Base class for PyTorch interfaces to Tokamax SparseMLA operators."""

from collections.abc import Sequence
from typing import Any, TypeVar, override

import jax
from tokamax._src.ops.sparse_mla import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch

_Config = TypeVar("_Config")


class _SparseMla(torch_op.TorchOp[_Config]):
  """SparseMLA PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.SparseMla()
    self.jax_op_name = "base_sparse_mla"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      q: torch.Tensor,
      cache_kv_nope: torch.Tensor,
      cache_kv_rope: torch.Tensor,
      topk_indices: torch.Tensor,
      page_indices: torch.Tensor,
      cu_q_lens: torch.Tensor,
      distribution: torch.Tensor,
      attention_sinks: torch.Tensor,
      swa_accumution: torch.Tensor,
      swa_l: torch.Tensor,
      swa_m: torch.Tensor,
      sm_scale: float = 1.0,
      return_residuals: bool = False,
      config: Sequence[int] | None = None,
  ) -> torch.Tensor:
    """Meta/fake implementation for symbolic shape tracing in torch.compile."""
    del cache_kv_nope, cache_kv_rope, topk_indices, page_indices, cu_q_lens
    del distribution, attention_sinks, swa_accumution, swa_l, swa_m
    del sm_scale, return_residuals, config
    return q.new_empty(q.shape)

  @override
  def __call__(
      self,
      q: torch.Tensor,
      cache_kv_nope: torch.Tensor,
      cache_kv_rope: torch.Tensor,
      topk_indices: torch.Tensor,
      page_indices: torch.Tensor,
      cu_q_lens: torch.Tensor,
      distribution: torch.Tensor,
      attention_sinks: torch.Tensor,
      swa_accumution: torch.Tensor,
      swa_l: torch.Tensor,
      swa_m: torch.Tensor,
      sm_scale: float = 1.0,
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
  ) -> torch.Tensor:
    del configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        q,
        cache_kv_nope,
        cache_kv_rope,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m,
        sm_scale=sm_scale,
        return_residuals=return_residuals,
        config=None,
    )

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      cache_kv_nope: jax.Array,
      cache_kv_rope: jax.Array,
      topk_indices: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
      attention_sinks: jax.Array,
      swa_accumution: jax.Array,
      swa_l: jax.Array,
      swa_m: jax.Array,
      sm_scale: float = 1.0,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        q,
        cache_kv_nope,
        cache_kv_rope,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m,
        sm_scale=sm_scale,
        return_residuals=return_residuals,
        config=None,
    )
    return out


# Singleton instance of SparseMla.
SparseMla = _SparseMla[Any]()  # pylint: disable=invalid-name
