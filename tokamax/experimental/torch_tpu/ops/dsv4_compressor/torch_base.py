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
"""Base class for PyTorch interfaces to Tokamax DSv4 Compressor operators."""

from collections.abc import Sequence
from typing import Any, TypeVar, override

import jax
from tokamax._src.ops.dsv4_compressor import base as jax_base
from tokamax._src.ops.dsv4_compressor import reference as jax_reference
from tokamax.experimental.torch_tpu.ops import torch_op
import torch

_Config = TypeVar("_Config")


class _Dsv4Compressor(torch_op.TorchOp[_Config]):
  """DSv4 Compressor PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.Dsv4Compressor()
    self.jax_op_name = "base_dsv4_compressor"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      cache: torch.Tensor,
      positions: torch.Tensor,
      block_table: torch.Tensor,
      token_to_req_indices: torch.Tensor,
      kv_slot_mapping: torch.Tensor,
      rms_weight: torch.Tensor,
      cos_sin_cache: torch.Tensor,
      block_table_stride: int,
      state_block_size: int,
      compress_ratio: int,
      overlap: bool,
      state_cache: torch.Tensor | None = None,
      rope_cache: torch.Tensor | None = None,
      quant_block: int = jax_base.CSA_QUANT_BLOCK,
      rms_eps: float = 1e-6,
      return_residuals: bool = False,
      config: Sequence[int] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Meta/fake implementation for symbolic shape tracing in torch.compile."""
    del positions, block_table, token_to_req_indices, kv_slot_mapping
    del cos_sin_cache, block_table_stride, state_block_size, compress_ratio
    del state_cache, quant_block, rms_eps, return_residuals, config
    has_rope_cache = (
        overlap and rms_weight.shape[0] != jax_reference.INDEXER_HEAD_DIM
    )
    out_rope_cache = (
        torch.empty_like(rope_cache)
        if has_rope_cache and rope_cache is not None
        else None
    )
    return torch.empty_like(cache), out_rope_cache

  @override
  def __call__(
      self,
      cache: torch.Tensor,
      positions: torch.Tensor,
      block_table: torch.Tensor,
      token_to_req_indices: torch.Tensor,
      kv_slot_mapping: torch.Tensor,
      rms_weight: torch.Tensor,
      cos_sin_cache: torch.Tensor,
      block_table_stride: int,
      state_block_size: int,
      compress_ratio: int,
      overlap: bool,
      state_cache: torch.Tensor | None = None,
      rope_cache: torch.Tensor | None = None,
      quant_block: int = jax_base.CSA_QUANT_BLOCK,
      rms_eps: float = 1e-6,
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    del configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        cache,
        positions,
        block_table,
        token_to_req_indices,
        kv_slot_mapping,
        rms_weight,
        cos_sin_cache,
        block_table_stride=block_table_stride,
        state_block_size=state_block_size,
        compress_ratio=compress_ratio,
        overlap=overlap,
        state_cache=state_cache,
        rope_cache=rope_cache,
        quant_block=quant_block,
        rms_eps=rms_eps,
        return_residuals=return_residuals,
        config=None,
    )

  @override
  def op_impl_call(
      self,
      cache: jax.Array,
      positions: jax.Array,
      block_table: jax.Array,
      token_to_req_indices: jax.Array,
      kv_slot_mapping: jax.Array,
      rms_weight: jax.Array,
      cos_sin_cache: jax.Array,
      block_table_stride: int,
      state_block_size: int,
      compress_ratio: int,
      overlap: bool,
      state_cache: jax.Array | None = None,
      rope_cache: jax.Array | None = None,
      quant_block: int = jax_base.CSA_QUANT_BLOCK,
      rms_eps: float = 1e-6,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array | None]:
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    (out_cache, out_rope_cache), _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        cache,
        positions,
        block_table,
        token_to_req_indices,
        kv_slot_mapping,
        rms_weight,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=block_table_stride,
        state_block_size=state_block_size,
        compress_ratio=compress_ratio,
        overlap=overlap,
        state_cache=state_cache,
        rope_cache=rope_cache,
        quant_block=quant_block,
        rms_eps=rms_eps,
        return_residuals=return_residuals,
        config=None,
    )
    return out_cache, out_rope_cache


# Singleton instance of Dsv4Compressor.
Dsv4Compressor = _Dsv4Compressor[Any]()  # pylint: disable=invalid-name
