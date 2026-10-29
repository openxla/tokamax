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
"""Base class for PyTorch interfaces to Tokamax MLA-SWA operators."""

from collections.abc import Sequence
from typing import Any, TypeVar, override

import jax
from tokamax._src.ops.experimental.mla_swa import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch

_Config = TypeVar("_Config")


class _MlaSwa(torch_op.TorchOp[_Config]):
  """Multi-Head Latent Sliding Window Attention PyTorch Op using reference impl."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.MlaSwa()
    self.jax_op_name = "base_mla_swa"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      q: torch.Tensor,
      new_kv: torch.Tensor,
      cache_kv: torch.Tensor,
      kv_lens: torch.Tensor,
      page_indices: torch.Tensor,
      cu_q_lens: torch.Tensor,
      distribution: torch.Tensor,
      attention_sinks: torch.Tensor,
      sliding_window: int,
      logical_page_size: int,
      sm_scale: float = 1.0,
      chunk_prefill_size: int | None = None,
      unnormalized_output: bool = False,
      non_causal_block: bool = False,
      return_residuals: bool = False,
      config: Sequence[int] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Meta/fake implementation for symbolic shape tracing in torch.compile."""
    del new_kv, kv_lens, page_indices, cu_q_lens, distribution
    del attention_sinks, sliding_window, logical_page_size, sm_scale
    del chunk_prefill_size, unnormalized_output, non_causal_block
    del return_residuals, config
    l_m_shape = (q.shape[0], q.shape[1])
    return (
        torch.empty_like(q),
        torch.empty_like(cache_kv),
        q.new_empty(l_m_shape, dtype=torch.float32),
        q.new_empty(l_m_shape, dtype=torch.float32),
    )

  @override
  def __call__(
      self,
      q: torch.Tensor,
      new_kv: torch.Tensor,
      cache_kv: torch.Tensor,
      kv_lens: torch.Tensor,
      page_indices: torch.Tensor,
      cu_q_lens: torch.Tensor,
      distribution: torch.Tensor,
      attention_sinks: torch.Tensor,
      sliding_window: int,
      logical_page_size: int,
      sm_scale: float = 1.0,
      chunk_prefill_size: int | None = None,
      unnormalized_output: bool = False,
      non_causal_block: bool = False,
      return_residuals: bool = False,
      config: _Config | None = None,
      configs: tuple[Any, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del config, configs  # Unused.
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        q,
        new_kv,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        sm_scale=sm_scale,
        chunk_prefill_size=chunk_prefill_size,
        unnormalized_output=unnormalized_output,
        non_causal_block=non_causal_block,
        return_residuals=return_residuals,
        config=None,
    )

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      new_kv: jax.Array,
      cache_kv: jax.Array,
      kv_lens: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
      attention_sinks: jax.Array,
      sliding_window: int,
      logical_page_size: int,
      sm_scale: float = 1.0,
      chunk_prefill_size: int | None = None,
      unnormalized_output: bool = False,
      non_causal_block: bool = False,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    (out, updated_cache_kv, out_l, out_m), _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        q,
        new_kv,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        sm_scale=sm_scale,
        chunk_prefill_size=chunk_prefill_size,
        unnormalized_output=unnormalized_output,
        non_causal_block=non_causal_block,
        return_residuals=return_residuals,
        config=None,
    )
    return out, updated_cache_kv, out_l, out_m


# Singleton instance of MlaSwa.
MlaSwa = _MlaSwa[Any]()  # pylint: disable=invalid-name
