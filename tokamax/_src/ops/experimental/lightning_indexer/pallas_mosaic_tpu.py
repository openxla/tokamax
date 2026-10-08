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
"""Pallas Mosaic TPU operator wrapper for Lightning Indexer."""

import dataclasses
import itertools
from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int, UInt8  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.lightning_indexer import base
from tokamax._src.ops.experimental.lightning_indexer.kernel import config as kernel_config
from tokamax._src.ops.experimental.lightning_indexer.kernel import streamindex_topk as kernel_lib

KVLayout = kernel_config.KVLayout


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Lightning Indexer Pallas TPU kernel."""

  num_kv_pages_per_block: tuple[int, int, int] | int = (1, 1, 1)
  num_queries_per_block: tuple[int, int, int] | int = (1, 16, 16)
  buffer_count: tuple[int, int, int] | int = kernel_config.DEFAULT_BUFFER_COUNT
  vmem_limit_bytes: Annotated[int, pydantic.Field(gt=0)] = (
      kernel_lib.DEFAULT_VMEM_LIMIT_BYTES
  )
  decode_req_batch_size: Annotated[int, pydantic.Field(gt=0)] = 4
  enable_early_exit: bool = False
  chunk_tokens: Annotated[int, pydantic.Field(gt=0)] | None = None


@dataclasses.dataclass(frozen=True)
class PallasTpuLightningIndexer(base.LightningIndexer[Config]):
  """Tokamax operator wrapper for Pallas Mosaic TPU Lightning Indexer kernel."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array, "T H D"],
      indexer_weights: Float[Array, "T H"],
      cache_kv: UInt8[Array, "P _ 4 _"],
      seq_lens: Int[Array, "B"],
      page_indices: Int[Array, "_"],
      cu_q_lens: Int[Array, "_"],
      distribution: Int[Array, "3"],
      *,
      k: int,
      compression_ratio: int = 1,
      kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
      cp_size: int = 1,
      cp_rank: Int[Array, ""] | int = 0,
      interleave_size: int = 1,
      return_scores: bool = False,
      return_residuals: bool = False,
      config: Config,
  ) -> tuple[jax.Array | tuple[jax.Array, jax.Array], None]:
    del return_residuals
    out = kernel_lib.streamindex_topk(
        q=q,
        indexer_weights=indexer_weights,
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        k=k,
        compression_ratio=compression_ratio,
        num_kv_pages_per_block=config.num_kv_pages_per_block,
        num_queries_per_block=config.num_queries_per_block,
        buffer_count=config.buffer_count,
        vmem_limit_bytes=config.vmem_limit_bytes,
        decode_req_batch_size=config.decode_req_batch_size,
        enable_early_exit=config.enable_early_exit,
        kv_layout=kv_layout,
        chunk_tokens=config.chunk_tokens,
        cp_size=cp_size,
        cp_rank=cp_rank,
        interleave_size=interleave_size,
        return_scores=return_scores,
    )
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(
        num_kv_pages_per_block=(1, 1, 1),
        num_queries_per_block=(1, 16, 16),
        buffer_count=kernel_config.DEFAULT_BUFFER_COUNT,
        vmem_limit_bytes=kernel_lib.DEFAULT_VMEM_LIMIT_BYTES,
        decode_req_batch_size=4,
        enable_early_exit=False,
        chunk_tokens=None,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    del ba
    kv_pages = [1, 2]
    queries = [8, 16, 32]
    decode_batch_sizes = [1, 4]
    configs = set()
    for bkv_p, bq, decode_bs in itertools.product(
        kv_pages,
        queries,
        decode_batch_sizes,
    ):
      configs.add(
          Config(
              num_kv_pages_per_block=(bkv_p, bkv_p, bkv_p),
              num_queries_per_block=(1, bq, bq),
              decode_req_batch_size=decode_bs,
          )
      )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 7
