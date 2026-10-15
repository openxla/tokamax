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
"""Base class for PyTorch interfaces to Tokamax MaskedDenseMLA operators."""

from collections.abc import Sequence
from typing import Any, TypeVar, override

import jax
from tokamax._src.ops.masked_dense_mla import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch

_Config = TypeVar("_Config")


class _MaskedDenseMla(torch_op.TorchOp[_Config]):
  """Masked-Dense Multi-Head Latent Attention PyTorch Op using reference impl."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.MaskedDenseMla()
    self.jax_op_name = "base_masked_dense_mla"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      q: torch.Tensor,
      cache_kv_nope: torch.Tensor,
      cache_kv_rope: torch.Tensor,
      kv_lens: torch.Tensor,
      topk_indices: torch.Tensor,
      page_indices: torch.Tensor,
      cu_q_lens: torch.Tensor,
      distribution: torch.Tensor,
      sm_scale: float = 1.0,
      k_scale: float = 1.0,
      mask_value: float | None = None,
      max_kv_len: int | None = None,
      chunk_prefill_size: int | None = None,
      sequence_start: torch.Tensor | None = None,
      return_residuals: bool = False,
      config: Sequence[int] | None = None,
  ) -> torch.Tensor:
    """Meta/fake implementation for symbolic shape tracing in torch.compile."""
    del cache_kv_nope, cache_kv_rope, kv_lens, topk_indices, page_indices
    del cu_q_lens, distribution, sm_scale, k_scale, mask_value, max_kv_len
    del chunk_prefill_size, sequence_start, return_residuals, config
    return torch.empty_like(q)

  @override
  def __call__(
      self,
      q: torch.Tensor,
      cache_kv_nope: torch.Tensor,
      cache_kv_rope: torch.Tensor,
      kv_lens: torch.Tensor,
      topk_indices: torch.Tensor,
      page_indices: torch.Tensor,
      cu_q_lens: torch.Tensor,
      distribution: torch.Tensor,
      sm_scale: float = 1.0,
      k_scale: float = 1.0,
      mask_value: float | None = None,
      max_kv_len: int | None = None,
      chunk_prefill_size: int | None = None,
      sequence_start: torch.Tensor | None = None,
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
  ) -> torch.Tensor:
    del configs  # Unused.
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        q,
        cache_kv_nope,
        cache_kv_rope,
        kv_lens,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        k_scale=k_scale,
        mask_value=mask_value,
        max_kv_len=max_kv_len,
        chunk_prefill_size=chunk_prefill_size,
        sequence_start=sequence_start,
        return_residuals=return_residuals,
        config=None,
    )

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      cache_kv_nope: jax.Array,
      cache_kv_rope: jax.Array,
      kv_lens: jax.Array,
      topk_indices: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
      sm_scale: float = 1.0,
      k_scale: float = 1.0,
      mask_value: float | None = None,
      max_kv_len: int | None = None,
      chunk_prefill_size: int | None = None,
      sequence_start: jax.Array | None = None,
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
        kv_lens,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        k_scale=k_scale,
        mask_value=mask_value,
        max_kv_len=max_kv_len,
        chunk_prefill_size=chunk_prefill_size,
        sequence_start=sequence_start,
        return_residuals=return_residuals,
        config=None,
    )
    return out


# Singleton instance of MaskedDenseMla.
MaskedDenseMla = _MaskedDenseMla[Any]()  # pylint: disable=invalid-name
