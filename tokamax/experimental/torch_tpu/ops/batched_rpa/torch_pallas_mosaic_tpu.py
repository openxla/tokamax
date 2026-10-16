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
"""Tokamax operator wrapper for Pallas Mosaic TPU Batched RPA."""

from collections.abc import Sequence
import dataclasses
from typing import Any, override

import jax
from tokamax._src.ops.batched_rpa.kernel import configs as jax_types
import tokamax._src.ops.batched_rpa.pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.batched_rpa import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config
_KV_LAYOUTS = tuple(jax_types.KVLayout.__members__)

# Index of `kv_cache` in `op_impl_call`'s positional arguments. The kernel
# writes the new keys and values into the cache and returns it, so the input
# buffer is dead on return; donating it saves a copy of the whole pool per call.
_KV_CACHE_ARGNUM = 3


class _PallasMosaicTpuBatchedRpa(
    torch_base._BatchedRpa[Config]  # pylint: disable=protected-access
):
  """Tokamax operator wrapper for Pallas Mosaic TPU Batched RPA."""

  def __init__(self):
    torch_op.TorchOp.__init__(self)
    self.jax_op_name = "torch_tpu_pallas_mosaic_tpu_batched_rpa"
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuBatchedRpa()
    self.is_vjp = False
    # TODO: support donate_argnames.
    self.donate_argnums = (_KV_CACHE_ARGNUM,)
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | None
  ) -> tuple[int, ...] | None:
    """Deconstructs a Config object into a 7-int tuple for jax_op."""
    if config is None:
      return None
    assert len(dataclasses.fields(config)) == 7, (
        f"Expected 7 fields in Config, got {len(dataclasses.fields(config))}."
    )
    return (
        config.prefill_bq_sz,
        config.prefill_bq_c_sz,
        config.bkv_sz,
        config.n_buffer,
        config.decode_batch_size,
        config.prefill_batch_size,
        _KV_LAYOUTS.index(config.kv_layout.upper()),
    )

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Rebuilds the deconstructed 7-int tuple back into a Config."""
    config = config_parts[0]
    assert config is not None, "Forward config not set."
    return Config(*config[:6], kv_layout=_KV_LAYOUTS[config[6]])

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
      kv_layout: str = "head_along_sublane",
      decode_query_size: int = 1,
      cp_group_size: int | None = None,
      attention_scope: str = "full",
      return_lse: bool = False,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    assert (
        self.op_impl_jax is not None
    ), "Forward class not set. self.op_impl_jax was not set in the constructor."
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
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
        out_dtype=torch_utils.to_jax_dtype(out_dtype),
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        decode_query_size=decode_query_size,
        skip_kv_update=skip_kv_update,
        kv_layout=jax_types.KVLayout(kv_layout),
        cp_group_size=cp_group_size,
        cp_rank=cp_rank,
        attention_scope=jax_types.AttentionScope(attention_scope),
        return_lse=return_lse,
        decode_block_sizes=torch_base.to_block_sizes(decode_block_sizes),
        prefill_block_sizes=torch_base.to_block_sizes(prefill_block_sizes),
        return_residuals=return_residuals,
        config=kernel_config,
    )
    return out, kv_cache

  def _fake_impl(
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
      kv_layout: str = "head_along_sublane",
      decode_query_size: int = 1,
      cp_group_size: int | None = None,
      attention_scope: str = "full",
      return_lse: bool = False,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Meta shapes for the registered op.

    jax_op's default fake re-exports the JAX function, which rejects a
    `torch.SymInt` token count, and serving stacks compile once for a range of
    batch sizes. The output is query-shaped and the cache keeps its shape.
    """
    if return_lse:
      raise NotImplementedError(
          "return_lse adds an output and is not supported through the torch"
          " operator."
      )
    jax_dtype = torch_utils.to_jax_dtype(out_dtype)
    if jax_dtype is None:
      return torch.empty_like(queries), torch.empty_like(kv_cache)
    dtype = getattr(torch, jax_dtype.name)
    return torch.empty_like(queries, dtype=dtype), torch.empty_like(kv_cache)

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
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      chunk_prefill_size: int | None = None,
      decode_block_sizes: jax_types.BlockSizes | Sequence[int] | None = None,
      prefill_block_sizes: jax_types.BlockSizes | Sequence[int] | None = None,
      vmem_limit_bytes: int | None = None,
      debug_mode: bool = False,
      out_dtype: Any = None,
      skip_kv_update: bool = False,
      kv_layout: str = "head_along_sublane",
      decode_query_size: int = 1,
      cp_group_size: int | None = None,
      attention_scope: str = "full",
      return_lse: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:

    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. self.op_impl_jax was not set in the"
        " constructor."
    )

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
        decode_block_sizes=torch_base.block_sizes_to_tuple(decode_block_sizes),
        prefill_block_sizes=torch_base.block_sizes_to_tuple(
            prefill_block_sizes
        ),
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
        skip_kv_update=skip_kv_update,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
        cp_group_size=cp_group_size,
        attention_scope=attention_scope,
        return_lse=return_lse,
        return_residuals=return_residuals,
        config=self.deconstruct_config(config),
    )


PallasMosaicTpuBatchedRpa = _PallasMosaicTpuBatchedRpa()
