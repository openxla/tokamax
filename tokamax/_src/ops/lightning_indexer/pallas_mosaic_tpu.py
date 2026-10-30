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
import math
from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int, UInt8  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import mosaic_tpu
from tokamax._src.ops import op
from tokamax._src.ops.lightning_indexer import base
from tokamax._src.ops.lightning_indexer.kernel import config as kernel_config
from tokamax._src.ops.lightning_indexer.kernel import streamindex_topk as kernel_lib
from tokamax._src.ops.lightning_indexer.kernel import tc_topk

KVLayout = kernel_config.KVLayout


def _tc_topk_shape_params(ba: op.BoundArguments) -> tuple[int, int, int, int]:
  """Returns `(num_tokens, max_num_seqs, pages_per_seq, page_size)`.

  `page_size` is the number of compressed KV positions per cache page.

  Args:
    ba: Bound arguments of the Lightning Indexer op.
  """
  q = ba.arguments["q"]
  cache_kv = ba.arguments["cache_kv"]
  seq_lens = ba.arguments["seq_lens"]
  page_indices = ba.arguments["page_indices"]
  kv_layout = KVLayout.parse(
      ba.arguments.get("kv_layout", KVLayout.HEAD_ALONG_SUBLANE)
  )
  max_num_seqs = seq_lens.shape[0]
  pages_per_seq = max(1, page_indices.shape[0] // max_num_seqs)
  if kv_layout == KVLayout.SEQ_ALONG_LANE:
    page_size = cache_kv.shape[-1]
  else:
    page_size = cache_kv.shape[1] * cache_kv.shape[2]
  return q.shape[0], max_num_seqs, pages_per_seq, page_size


def _tc_kv_pages(max_pages: int, pages_per_seq: int, page_size: int) -> int:
  """KV pages per block: `<= max_pages` when possible, 128-lane aligned."""
  pages_per_lane_tile = 128 // math.gcd(page_size, 128)
  pages = min(max_pages, pages_per_seq)
  return max(1, -(-pages // pages_per_lane_tile)) * pages_per_lane_tile


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
      uniform_q_len: int | None = None,
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
        uniform_q_len=uniform_q_len,
    )
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    if not tc_topk.has_full_sparsecore_topk():
      num_tokens, max_num_seqs, pages_per_seq, page_size = (
          _tc_topk_shape_params(ba)
      )
      if page_size <= 256:
        max_dec_p, max_pre_p = 4, 8
      elif page_size <= 512:
        max_dec_p, max_pre_p = 2, 4
      else:
        max_dec_p, max_pre_p = 2, 2
      bkv_p_dec = _tc_kv_pages(max_dec_p, pages_per_seq, page_size)
      bkv_p_pre = _tc_kv_pages(max_pre_p, pages_per_seq, page_size)
      bq = 128 if num_tokens >= 64 else (32 if num_tokens >= 32 else 16)
      decode_bs = 8 if max_num_seqs >= 8 else (4 if max_num_seqs >= 4 else 1)
      return Config(
          num_kv_pages_per_block=(bkv_p_dec, bkv_p_pre, bkv_p_pre),
          num_queries_per_block=(1, bq, bq),
          buffer_count=kernel_config.DEFAULT_BUFFER_COUNT,
          vmem_limit_bytes=mosaic_tpu.default_tc_vmem_limit_bytes(),
          decode_req_batch_size=decode_bs,
          enable_early_exit=False,
          chunk_tokens=None,
      )
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
    if not tc_topk.has_full_sparsecore_topk():
      num_tokens, max_num_seqs, pages_per_seq, page_size = (
          _tc_topk_shape_params(ba)
      )
      vmem_limit = mosaic_tpu.default_tc_vmem_limit_bytes()
      max_pre_p = 8 if page_size <= 256 else 4
      kv_pages = sorted({
          _tc_kv_pages(p, pages_per_seq, page_size)
          for p in (1, 2, 4, max_pre_p)
      })
      queries = [8, 16, 32, 64, 128] if num_tokens >= 64 else [8, 16, 32]
      decode_batch_sizes = sorted(
          {1} | {d for d in (4, 8) if d <= max_num_seqs}
      )
      configs = {self._get_heuristics_config(ba)}
      for bkv_p, bq, decode_bs in itertools.product(
          kv_pages, queries, decode_batch_sizes
      ):
        configs.add(
            Config(
                num_kv_pages_per_block=(bkv_p, bkv_p, bkv_p),
                num_queries_per_block=(1, bq, bq),
                vmem_limit_bytes=vmem_limit,
                decode_req_batch_size=decode_bs,
            )
        )
      return configs

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
