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
from tokamax._src.ops.experimental.batched_rpa import types as jax_types
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
      cp_rank: torch.Tensor | None = None,
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
      chunk_prefill_size: int | None = None,
      decode_block_sizes: jax_types.BlockSizes | None = None,
      prefill_block_sizes: jax_types.BlockSizes | None = None,
      vmem_limit_bytes: int | None = None,
      debug_mode: bool = False,
      skip_kv_update: bool = True,
      kv_layout: (
          jax_types.KVLayout | str
      ) = jax_types.KVLayout.HEAD_ALONG_SUBLANE,
      decode_query_size: int = 1,
      cp_group_size: int | None = None,
      attention_scope: (
          jax_types.AttentionScope | str
      ) = jax_types.AttentionScope.FULL,
      return_lse: bool = False,
      return_residuals: bool = False,
      config: _Config | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    self.configs = (config, None)
    assert self._torch_tokamax_op is not None, "Forward op not registered."

    # Store the inputs that cannot go through jax_op.
    self.out_dtype = out_dtype
    self.decode_block_sizes = decode_block_sizes
    self.prefill_block_sizes = prefill_block_sizes
    self.kv_layout = kv_layout
    self.attention_scope = attention_scope

    return self._torch_tokamax_op(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        cp_rank,
        use_causal_mask=use_causal_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
        skip_kv_update=skip_kv_update,
        decode_query_size=decode_query_size,
        cp_group_size=cp_group_size,
        return_lse=return_lse,
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
      cp_rank: jax.Array | None = None,
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
      chunk_prefill_size: int | None = None,
      vmem_limit_bytes: int | None = None,
      debug_mode: bool = False,
      skip_kv_update: bool = True,
      decode_query_size: int = 1,
      cp_group_size: int | None = None,
      return_lse: bool = False,
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
        cp_rank=cp_rank,
        use_causal_mask=use_causal_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        out_dtype=out_dtype,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        decode_query_size=decode_query_size,
        skip_kv_update=skip_kv_update,
        kv_layout=self.kv_layout,
        cp_group_size=cp_group_size,
        attention_scope=self.attention_scope,
        return_lse=return_lse,
        decode_block_sizes=self.decode_block_sizes,
        prefill_block_sizes=self.prefill_block_sizes,
        return_residuals=return_residuals,
        config=None,
    )
    return out, kv_cache


BatchedRpa = _BatchedRpa()
