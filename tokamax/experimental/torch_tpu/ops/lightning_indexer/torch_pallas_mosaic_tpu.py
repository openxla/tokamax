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
"""Tokamax operator wrapper for Pallas Mosaic TPU Lightning Indexer."""

from collections.abc import Sequence
import dataclasses
from typing import Any, override

import jax
from tokamax._src.ops.experimental.lightning_indexer import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax._src.ops.experimental.lightning_indexer.kernel import config as kernel_config
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.lightning_indexer import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config
KVLayout = kernel_config.KVLayout


def _to_triple(val: Sequence[int] | int) -> tuple[int, int, int]:
  """Normalizes an int or 3-element sequence into a 3-tuple of ints."""
  if isinstance(val, int) and not isinstance(val, bool):
    return (int(val), int(val), int(val))
  if isinstance(val, Sequence):
    vals = tuple(int(x) for x in val)
    if len(vals) == 3:
      return (vals[0], vals[1], vals[2])
  raise ValueError(f"Expected 3-tuple or int, got {val!r}.")


class _PallasMosaicTpuLightningIndexer(
    torch_base._LightningIndexer[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax Lightning Indexer Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuLightningIndexer()
    self.jax_op_name = "pallas_mosaic_tpu_lightning_indexer"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | None
  ) -> tuple[int, ...] | None:
    """Deconstructs a Config object into a flat 13-int tuple for jax_op."""
    if config is None:
      return None
    assert len(dataclasses.fields(config)) == 7, (
        f"Expected 7 fields in Config, got {len(dataclasses.fields(config))}."
    )
    kv_pages = _to_triple(config.num_kv_pages_per_block)
    queries = _to_triple(config.num_queries_per_block)
    buf_cnt = _to_triple(config.buffer_count)
    vmem_limit = int(config.vmem_limit_bytes)
    decode_bs = int(config.decode_req_batch_size)
    early_exit = int(bool(config.enable_early_exit))
    chunk_tokens = (
        -1 if config.chunk_tokens is None else int(config.chunk_tokens)
    )
    return (
        *kv_pages,
        *queries,
        *buf_cnt,
        vmem_limit,
        decode_bs,
        early_exit,
        chunk_tokens,
    )

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Rebuilds the deconstructed 13-int tuple back into a Config."""
    if not config_parts or config_parts[0] is None:
      raise ValueError("Forward config cannot be None.")
    flat = config_parts[0]
    if isinstance(flat, Config):
      return flat
    assert len(flat) == 13, f"Invalid config: {flat!r}"
    return Config(
        num_kv_pages_per_block=(flat[0], flat[1], flat[2]),
        num_queries_per_block=(flat[3], flat[4], flat[5]),
        buffer_count=(flat[6], flat[7], flat[8]),
        vmem_limit_bytes=int(flat[9]),
        decode_req_batch_size=int(flat[10]),
        enable_early_exit=bool(flat[11]),
        chunk_tokens=None if flat[12] <= 0 else int(flat[12]),
    )

  @override
  def op_impl_call_config_setup(
      self, *args: Any, config: Any = None, **kwargs: Any
  ) -> tuple[int, ...] | None:
    if config is None:
      config = self.deconstruct_config(
          self.get_bound_args(*args, **kwargs).get_config(
              check_autotuning_cache=False,
          )
      )
    return config

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
      kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
      cp_size: int = 1,
      cp_rank: torch.Tensor | int = 0,
      interleave_size: int = 1,
      return_scores: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
      configs: tuple[Config | None, Any] | None = None,
  ) -> Any:
    fwd_config = config if configs is None else configs[0]
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
        kv_layout=torch_base.kv_layout_to_str(kv_layout),
        cp_size=cp_size,
        cp_rank=cp_rank,
        interleave_size=interleave_size,
        return_scores=return_scores,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
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
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert config is not None, "Forward config not set."
    kernel_cfg = self.reconstruct_config(config)
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
        kv_layout=KVLayout.parse(kv_layout),
        cp_size=cp_size,
        cp_rank=cp_rank,
        interleave_size=interleave_size,
        return_scores=return_scores,
        return_residuals=return_residuals,
        config=kernel_cfg,
    )
    if return_scores:
      return out
    return out, out


# Singleton instance of Pallas Mosaic TPU LightningIndexer.
PallasMosaicTpuLightningIndexer = _PallasMosaicTpuLightningIndexer()  # pylint: disable=invalid-name
PallasTpuLightningIndexer = PallasMosaicTpuLightningIndexer  # pylint: disable=invalid-name
