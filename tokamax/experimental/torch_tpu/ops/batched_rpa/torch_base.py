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
"""Base class for PyTorch interfaces to Tokamax Batched RPA operators."""

from collections.abc import Sequence
from typing import Any, TypeVar, override
import jax
from tokamax._src.ops.experimental.batched_rpa import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch
import torch_tpu
import torch_tpu._internal.pallas.pallas

_Config = TypeVar("_Config")


class _BatchedRpa(torch_op.TorchOp[_Config]):
  """Base class for Batched RPA operators."""

  def __init__(self):
    super().__init__()
    self.jax_op_name = "base_batched_rpa"
    self.op_impl_jax = jax_base.BatchedRpa()
    self.is_vjp = False

  @override
  def __call__(
      self,
      queries: torch.Tensor,
      keys: torch.Tensor,
      values: torch.Tensor,
      kv_cache: torch.Tensor,
      kv_lens: torch.Tensor,
      page_indices: torch.Tensor,
      cu_q_lens: torch.Tensor,
      distribution: torch.Tensor,
      *,
      use_causal_mask: bool = True,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      out_dtype: Any = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      return_residuals: bool = False,
      config: _Config | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    self.configs = (config, None)
    assert self._torch_tokamax_op is not None, "Forward op not registered."
    return self._torch_tokamax_op(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=use_causal_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        return_residuals=return_residuals,
    )

  @override
  def op_impl_call(
      self,
      queries: jax.Array,
      keys: jax.Array,
      values: jax.Array,
      kv_cache: jax.Array,
      kv_lens: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
      *,
      use_causal_mask: bool = True,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      out_dtype: Any = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      return_residuals: bool = False,
  ) -> tuple[jax.Array, jax.Array]:
    assert self.op_impl_jax is not None, "Forward class not set."
    (out, kv_cache), _ = self.op_impl_jax._fwd(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=use_causal_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        out_dtype=out_dtype,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        return_residuals=return_residuals,
        config=None,
    )
    return out, kv_cache


BatchedRpa = _BatchedRpa()
