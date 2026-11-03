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


# TensorCore top-k heuristic. Sizes are in compressed KV positions per sequence.
#
# A query block whose whole compressed KV fits in one KV block takes the
# single-block top-k path (bitonic sort of VMEM-resident scores), which is about
# 3x faster than the streaming top-k path: with `k / num_positions >= 1 / 256`
# the streaming path flushes its candidates about once per `k` positions.
_TC_DECODE_MAX_SINGLE_BLOCK_POSITIONS = 16384
_TC_DECODE_STREAM_BLOCK_POSITIONS = 16384
_TC_BATCHED_DECODE_STREAM_BLOCK_POSITIONS = 2048
_TC_QBLOCK_MAX_SINGLE_BLOCK_POSITIONS = 8192
_TC_QBLOCK_STREAM_BLOCK_POSITIONS = 8192
# Upper bound on the bytes of one KV block (all sequences of a decode batch).
# The pipeline keeps `buffer_count` such blocks in VMEM, which may use at most
# half of the VMEM limit.
_TC_MAX_KV_BLOCK_BYTES = 24 * 1024 * 1024
# Decode batches of fewer than 5 sequences split each sequence across sublanes,
# which always takes the streaming top-k path.
_TC_MIN_DECODE_BATCH = 8
_TC_MAX_DECODE_BATCH = 16
# Upper bound on decode batch size times KV positions per block.
_TC_MAX_DECODE_STEP_POSITIONS = 65536


def _tc_page_bytes(ba: op.BoundArguments) -> int:
  """Returns the bytes of one compressed KV cache page."""
  cache_kv = ba.arguments["cache_kv"]
  return math.prod(cache_kv.shape[1:]) * cache_kv.dtype.itemsize


def _tc_block_pages(
    *,
    pages_per_seq: int,
    page_size: int,
    page_bytes: int,
    seqs_per_block: int,
    max_single_block_positions: int,
    stream_block_positions: int,
    vmem_limit_bytes: int,
) -> int:
  """Returns KV pages per block for `seqs_per_block` sequences per block."""
  max_block_bytes = min(
      _TC_MAX_KV_BLOCK_BYTES,
      vmem_limit_bytes // (2 * max(kernel_config.DEFAULT_BUFFER_COUNT)),
  )
  max_pages = max(1, max_block_bytes // (page_bytes * seqs_per_block))
  if (
      pages_per_seq * page_size <= max_single_block_positions
      and pages_per_seq <= max_pages
  ):
    return _tc_kv_pages(pages_per_seq, pages_per_seq, page_size)
  pages = min(max(1, stream_block_positions // page_size), max_pages)
  # Round down to a power of two so that blocks tile the page table evenly.
  return _tc_kv_pages(1 << (pages.bit_length() - 1), pages_per_seq, page_size)


def _tc_decode_batch_size(max_num_seqs: int, num_q_heads: int) -> int:
  """Returns the decode batch size the kernel uses for `max_num_seqs`."""
  if max_num_seqs < 4:
    return 1
  if max_num_seqs < _TC_MIN_DECODE_BATCH:
    return 4
  # The kernel caps the batch at 8 for <= 32 sequences with >= 64 heads.
  if num_q_heads >= 64 and max_num_seqs <= 32:
    return _TC_MIN_DECODE_BATCH
  return min(_TC_MAX_DECODE_BATCH, 1 << (max_num_seqs.bit_length() - 1))


def _tc_heuristics_config(ba: op.BoundArguments) -> Config:
  """Returns the heuristic config for the TensorCore top-k kernel."""
  num_tokens, max_num_seqs, pages_per_seq, page_size = _tc_topk_shape_params(
      ba
  )
  page_bytes = _tc_page_bytes(ba)
  vmem_limit_bytes = mosaic_tpu.default_tc_vmem_limit_bytes()
  decode_bs = _tc_decode_batch_size(max_num_seqs, ba.arguments["q"].shape[1])

  def decode_pages(seqs_per_block: int) -> int:
    return _tc_block_pages(
        pages_per_seq=pages_per_seq,
        page_size=page_size,
        page_bytes=page_bytes,
        seqs_per_block=seqs_per_block,
        max_single_block_positions=_TC_DECODE_MAX_SINGLE_BLOCK_POSITIONS,
        stream_block_positions=(
            _TC_BATCHED_DECODE_STREAM_BLOCK_POSITIONS
            if seqs_per_block >= _TC_MIN_DECODE_BATCH
            else _TC_DECODE_STREAM_BLOCK_POSITIONS
        ),
        vmem_limit_bytes=vmem_limit_bytes,
    )

  bkv_p_dec = decode_pages(seqs_per_block=decode_bs)
  # A smaller decode batch whose KV fits one block beats a larger one that
  # streams.
  while decode_bs > _TC_MIN_DECODE_BATCH and bkv_p_dec < pages_per_seq:
    smaller_bkv_p = decode_pages(seqs_per_block=decode_bs // 2)
    if smaller_bkv_p < pages_per_seq:
      break
    decode_bs //= 2
    bkv_p_dec = smaller_bkv_p
  # Each decode step unrolls the scoring of all sequences of the batch; keep
  # the per-step score count bounded to avoid register spills.
  while (
      decode_bs > _TC_MIN_DECODE_BATCH
      and decode_bs * bkv_p_dec * page_size > _TC_MAX_DECODE_STEP_POSITIONS
  ):
    decode_bs //= 2
    bkv_p_dec = decode_pages(decode_bs)
  bkv_p_pre = _tc_block_pages(
      pages_per_seq=pages_per_seq,
      page_size=page_size,
      page_bytes=page_bytes,
      seqs_per_block=1,
      max_single_block_positions=_TC_QBLOCK_MAX_SINGLE_BLOCK_POSITIONS,
      stream_block_positions=_TC_QBLOCK_STREAM_BLOCK_POSITIONS,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  # With several sequences, a query block of >= 64 sends every sequence
  # (decode ones included) through one pass with that block size; a small block
  # keeps decode sequences on the batched decode pass unless nearly all tokens
  # are prefill tokens.
  if num_tokens >= 64 and (
      max_num_seqs == 1 or num_tokens >= 256 * max_num_seqs
  ):
    bq = 128
  elif num_tokens >= 32:
    bq = 32
  else:
    bq = 16
  return Config(
      num_kv_pages_per_block=(bkv_p_dec, bkv_p_pre, bkv_p_pre),
      num_queries_per_block=(1, bq, bq),
      buffer_count=kernel_config.DEFAULT_BUFFER_COUNT,
      vmem_limit_bytes=vmem_limit_bytes,
      decode_req_batch_size=decode_bs,
      enable_early_exit=False,
      chunk_tokens=None,
  )


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
      return _tc_heuristics_config(ba)
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
