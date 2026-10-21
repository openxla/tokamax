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
"""Base class for PyTorch interfaces to Tokamax Lightning Indexer operators."""

from collections.abc import Sequence
from typing import Any, TypeVar, override

import jax
from tokamax._src.ops.lightning_indexer import base as jax_base
from tokamax._src.ops.lightning_indexer.kernel import config as kernel_config
from tokamax.experimental.torch_tpu.ops import torch_op
import torch

_Config = TypeVar("_Config")


class _LightningIndexer(torch_op.TorchOp[_Config]):
  """Lightning Indexer PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.LightningIndexer()
    self.jax_op_name = "base_lightning_indexer"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      q: torch.Tensor,
      indexer_weights: torch.Tensor,
      cache_kv: torch.Tensor,
      seq_lens: torch.Tensor,
      page_indices: torch.Tensor,
      cu_q_lens: torch.Tensor,
      distribution: torch.Tensor,
      k: int,
      compression_ratio: int = 1,
      kv_layout: str = "head_along_sublane",
      cp_size: int = 1,
      cp_rank: int = 0,
      interleave_size: int = 1,
      return_scores: bool = False,
      return_residuals: bool = False,
      config: Sequence[int] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Meta/fake implementation for symbolic shape tracing in torch.compile."""
    del indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens
    del distribution, compression_ratio, kv_layout, cp_size, cp_rank
    del interleave_size, return_scores, return_residuals, config
    out_shape = (q.shape[0], k)
    return (
        q.new_empty(out_shape, dtype=torch.int32),
        q.new_empty(out_shape, dtype=torch.int32),
    )

  @override
  def __call__(
      self,
      q: torch.Tensor,
      indexer_weights: torch.Tensor,
      cache_kv: torch.Tensor,
      seq_lens: torch.Tensor,
      page_indices: torch.Tensor,
      cu_q_lens: torch.Tensor,
      distribution: torch.Tensor,
      k: int,
      compression_ratio: int = 1,
      kv_layout: str = "head_along_sublane",
      cp_size: int = 1,
      cp_rank: torch.Tensor | int = 0,
      interleave_size: int = 1,
      return_scores: bool = False,
      return_residuals: bool = False,
      config: _Config | None = None,
      configs: tuple[Any, Any] | None = None,
  ) -> Any:
    del config, configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    if isinstance(cp_rank, torch.Tensor):
      cp_rank = int(cp_rank.item())
    top_indices, top_scores = self._torch_tokamax_op(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=k,
        compression_ratio=compression_ratio,
        kv_layout=kv_layout,
        cp_size=cp_size,
        cp_rank=cp_rank,
        interleave_size=interleave_size,
        return_scores=return_scores,
        return_residuals=return_residuals,
        config=None,
    )
    return (top_indices, top_scores) if return_scores else top_indices

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      indexer_weights: jax.Array,
      cache_kv: jax.Array,
      seq_lens: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
      k: int,
      compression_ratio: int = 1,
      kv_layout: str = "head_along_sublane",
      cp_size: int = 1,
      cp_rank: int = 0,
      interleave_size: int = 1,
      return_scores: bool = False,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=k,
        compression_ratio=compression_ratio,
        kv_layout=kernel_config.KVLayout(kv_layout),
        cp_size=cp_size,
        cp_rank=cp_rank,
        interleave_size=interleave_size,
        return_scores=return_scores,
        return_residuals=return_residuals,
        config=None,
    )
    if return_scores:
      return out
    return out, out


# Singleton instance of LightningIndexer.
LightningIndexer = _LightningIndexer[Any]()  # pylint: disable=invalid-name
