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
"""fused_out_proj's static decisions, all made once per call (create)."""

import dataclasses
import math
from typing import NamedTuple

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import memory
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import tiling


def quant_block(model: model_config.AttentionConfig) -> int:
  """o_proj's fp8 activation quant block: >= 2D, divides Hq * D, <= 1024."""
  pair = model.q_packing * model.head_dim
  for target in (model.act_quant_block, 1024):
    block = math.gcd(model.q_dim, target)
    if block >= pair and block % pair == 0:
      return block
  return model.q_dim


@dataclasses.dataclass(frozen=True)
class FusedOutProjConfig:
  """Everything fused_out_proj decides before tracing the kernel."""

  model: model_config.AttentionConfig
  rpa_tokens: int  # T_rpa: batched_rpa's (maybe padded) token count.
  hidden_size: int  # K: o_proj's output columns.
  tile: tiling.TileConfig
  quant_block: int  # fp8: columns per activation quant block (quant_block()).
  # o_proj columns per MXU call. Always a full MXU width, even when it doesn't
  # divide K (the last block is then shorter): narrower blocks halve MXU use
  # (gpt-oss, K = 2880, was 1.85x slower than XLA's matmul at 128 wide).
  col_block: int
  # (first column, width) of each o_proj block, in compute order. w_chunks is
  # built from it, so tile_body reads W in the order it arrives.
  col_blocks: memory.ColumnChunks
  w_chunks: memory.ColumnChunks  # W's DMA chunks.
  # One tile: tile_body waits for each weight chunk just before reading it, so
  # the DMAs overlap compute. More tiles: the pipeline body runs once per tile,
  # so its waits would need a per-tile branch (a scheduling barrier); instead
  # outer_pipeline waits for every chunk once, before the pipeline. Measured:
  # with one tile, overlapping was 7-16% faster than waiting up front; with
  # more tiles, per-tile waits under pl.when were 6-36% slower.
  overlap_weight_dma: bool
  vmem_limit_bytes: int  # Tiles and scratch, plus W's VMEM copy.

  @classmethod
  def create(
      cls,
      model: model_config.AttentionConfig,
      o: jax.Array,
      gate: jax.Array | None,
      weights: model_config.OutProjWeights,
      *,
      num_tokens: int | None = None,
      tile_size: int | None = None,
  ) -> "FusedOutProjConfig":
    """Makes every decision, then asserts it is one fused_out_proj implements."""
    if num_tokens is None:
      num_tokens = gate.shape[0] if gate is not None else o.shape[1]
    hidden_size = weights.w.shape[1]
    tile = tiling.TileConfig.create(
        num_tokens=num_tokens,
        fp8=model.fp8,
        resident_bytes=sum(memory.nbytes(w) for w in jax.tree.leaves(weights)),
        bytes_per_row=tiling.total_bytes(bytes_per_row(model, hidden_size)),
        tile_size=tile_size,
    )
    col_block = pltpu.get_tpu_info().mxu_column_size
    col_blocks = tuple(
        (col, min(col_block, hidden_size - col))
        for col in range(0, hidden_size, col_block)
    )
    cfg = cls(
        model=model,
        rpa_tokens=o.shape[1],
        hidden_size=hidden_size,
        tile=tile,
        quant_block=quant_block(model),
        col_block=col_block,
        col_blocks=col_blocks,
        w_chunks=memory.stream_chunks(
            col_blocks, tile.policy.stream_blocks_per_chunk * col_block
        ),
        overlap_weight_dma=tile.num_tiles == 1,
        vmem_limit_bytes=tile.vmem_limit_bytes + memory.nbytes(weights.w),
    )
    cfg._check(o, gate, weights)
    return cfg

  def _check(
      self,
      o: jax.Array,
      gate: jax.Array | None,
      weights: model_config.OutProjWeights,
  ) -> None:
    """Asserts the decisions are ones fused_out_proj implements."""
    m = self.model
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert m.q_dtype == jnp.bfloat16, "only bf16 o is implemented"
    assert (gate is not None) == m.gate
    assert self.rpa_tokens >= self.num_tokens
    assert o.shape == self.o_shape
    assert weights.w.shape == (m.q_dim, self.hidden_size)
    assert m.head_dim % num_lanes == 0 or 2 * m.head_dim == num_lanes
    assert self.hidden_size % num_lanes == 0 or not m.fp8

  @property
  def num_tokens(self) -> int:  # T.
    return self.tile.num_tokens

  @property
  def o_shape(self) -> tuple[int, ...]:
    """batched_rpa's raw output: (Hkv, T_rpa, q_per_kv / q_packing, ...)."""
    m = self.model
    return (
        m.num_kv_heads,
        self.rpa_tokens,
        m.q_per_kv // m.q_packing,
        m.q_packing,
        layout_util.align(m.head_dim, pltpu.get_tpu_info().num_lanes),
    )

  @property
  def o_tile_tokens(self) -> int:
    """Tokens per o block.

    With a single tile and a padded o (T_rpa > T), the block keeps o's full
    T_rpa extent so o can stay in VMEM, aliased, with no slice.
    """
    return self.rpa_tokens if self.tile.num_tiles == 1 else self.tile.tile_size

  @property
  def kernel_name(self) -> str:
    """e.g. fused_out_proj_fp8_t256."""
    precision = "fp8" if self.model.fp8 else "bf16"
    return f"fused_out_proj_{precision}_t{self.tile.tile_size}"

  def metadata(self) -> dict[str, str]:
    """The decisions above, serialized into the HLO for debugging."""
    return {
        **self.tile.metadata(),
        "rpa_tokens": str(self.rpa_tokens),
        "quant_block": str(self.quant_block),
        "col_block": str(self.col_block),
        "w_chunks": str(self.w_chunks),
        "overlap_weight_dma": str(self.overlap_weight_dma),
        "vmem_limit_bytes": str(self.vmem_limit_bytes),
    }


class RowBytes(NamedTuple):
  """fused_out_proj's VMEM bytes per token row; pipelined blocks double-buffered."""

  o: int  # Input block.
  gate: int  # Input block.
  out: int  # Output block.
  x_q: int  # Scratch.
  x_scale: int = 0  # Scratch, fp8 only.


def bytes_per_row(
    model: model_config.AttentionConfig, hidden_size: int
) -> RowBytes:
  """fused_out_proj's VMEM bytes per token row."""
  num_lanes = pltpu.get_tpu_info().num_lanes
  d_pad = layout_util.align(model.head_dim, num_lanes)
  q_dim_pad = layout_util.align(model.q_dim, num_lanes)
  k_pad = layout_util.align(hidden_size, num_lanes)
  bf16 = 2
  return RowBytes(
      o=2 * model.num_q_heads * d_pad * bf16,
      gate=2 * q_dim_pad * bf16 if model.gate else 0,
      out=2 * k_pad * bf16,
      x_q=q_dim_pad * (1 if model.fp8 else bf16),
      x_scale=(
          model.q_dim // quant_block(model) * num_lanes * 4 if model.fp8 else 0
      ),
  )
