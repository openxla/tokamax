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
import dataclasses
from typing import Any, TypeVar, override
import jax
from tokamax._src.ops.experimental.batched_rpa import base as jax_base
from tokamax._src.ops.experimental.batched_rpa.kernel import configs as jax_types
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops import torch_utils
import torch
import torch_tpu

_Config = TypeVar("_Config")


def to_block_sizes(
    block_sizes: jax_types.BlockSizes | Sequence[int] | None,
) -> jax_types.BlockSizes | None:
  """Converts a tuple/sequence of ints or BlockSizes to BlockSizes."""
  if block_sizes is None or isinstance(block_sizes, jax_types.BlockSizes):
    return block_sizes
  return jax_types.BlockSizes(*block_sizes)


def block_sizes_to_tuple(
    block_sizes: jax_types.BlockSizes | Sequence[int] | None,
) -> tuple[int, ...] | None:
  """Converts BlockSizes to a tuple of ints for static_argnums."""
  if block_sizes is None:
    return None
  if isinstance(block_sizes, jax_types.BlockSizes):
    return dataclasses.astuple(block_sizes)
  return tuple(block_sizes)


class _BatchedRpa(torch_op.TorchOp[_Config]):
  """Base class for Batched RPA operators."""

  def __init__(self):
    super().__init__()
    self.jax_op_name = "base_batched_rpa"
    self.op_impl_jax = jax_base.BatchedRpa()
    self.is_vjp = False

  @override
  def get_bound_args(self, *args: Any, **kwargs: Any) -> Any:
    # `get_bound_args` is called both from `op_impl_call_config_setup` (where
    # `decode_block_sizes` and `prefill_block_sizes` have been serialized to
    # int tuples and `out_dtype` to a string for `torch_tpu.jax_op`) and
    # directly by users (e.g. `torch_utils.get_configs`, where `out_dtype` may
    # be a `torch.dtype` and block sizes may be sequences). Normalize them to
    # `jax_types.BlockSizes` and `jnp.dtype` so `op_impl_jax.bind` validation in
    # `super().get_bound_args` receives the JAX types expected by `_fwd`.
    norm_kwargs = dict(kwargs)
    if "decode_block_sizes" in norm_kwargs:
      norm_kwargs["decode_block_sizes"] = to_block_sizes(
          norm_kwargs["decode_block_sizes"]
      )
    if "prefill_block_sizes" in norm_kwargs:
      norm_kwargs["prefill_block_sizes"] = to_block_sizes(
          norm_kwargs["prefill_block_sizes"]
      )
    if "out_dtype" in norm_kwargs:
      norm_kwargs["out_dtype"] = torch_utils.str_to_jax_dtype(
          norm_kwargs["out_dtype"]
      )
    return super().get_bound_args(*args, **norm_kwargs)

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
      decode_block_sizes: jax_types.BlockSizes | Sequence[int] | None = None,
      prefill_block_sizes: jax_types.BlockSizes | Sequence[int] | None = None,
      vmem_limit_bytes: int | None = None,
      debug_mode: bool = False,
      skip_kv_update: bool = False,
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
    del config
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
        cp_rank,
        use_causal_mask=use_causal_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        out_dtype=torch_utils.dtype_to_str(out_dtype),
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        decode_block_sizes=block_sizes_to_tuple(decode_block_sizes),
        prefill_block_sizes=block_sizes_to_tuple(prefill_block_sizes),
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
        skip_kv_update=skip_kv_update,
        kv_layout=str(kv_layout),
        decode_query_size=decode_query_size,
        cp_group_size=cp_group_size,
        attention_scope=str(attention_scope),
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
      out_dtype: str | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      chunk_prefill_size: int | None = None,
      decode_block_sizes: tuple[int, ...] | None = None,
      prefill_block_sizes: tuple[int, ...] | None = None,
      vmem_limit_bytes: int | None = None,
      debug_mode: bool = False,
      skip_kv_update: bool = False,
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
        out_dtype=torch_utils.str_to_jax_dtype(out_dtype),
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        decode_query_size=decode_query_size,
        skip_kv_update=skip_kv_update,
        kv_layout=kv_layout,
        cp_group_size=cp_group_size,
        attention_scope=attention_scope,
        return_lse=return_lse,
        decode_block_sizes=to_block_sizes(decode_block_sizes),
        prefill_block_sizes=to_block_sizes(prefill_block_sizes),
        return_residuals=return_residuals,
        config=None,
    )
    return out, kv_cache


BatchedRpa = _BatchedRpa()
