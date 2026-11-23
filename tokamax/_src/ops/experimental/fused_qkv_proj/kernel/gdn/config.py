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
"""fused_gdn's static decisions, all made once per call by FusedGdnConfig.create."""

import dataclasses
from typing import NamedTuple

import jax
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import memory
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import tiling


@dataclasses.dataclass(frozen=True)
class FusedGdnConfig:
  """Everything fused_gdn decides before tracing the kernel."""

  model: model_config.GdnConfig
  hidden_size: int  # K.
  tile: tiling.TileConfig
  col_block: int  # in_proj_qkvz columns per MXU call: one head pair.
  # (first column, width) of each in_proj_qkvz block, in compute order.
  # w_chunks is built from it, so tile_body reads W in the order it arrives.
  col_blocks: memory.ColumnChunks
  w_chunks: memory.ColumnChunks  # in_proj_qkvz's DMA chunks.
  ba_chunks: memory.ColumnChunks  # in_proj_ba's DMA chunk: all of it.
  # One tile: tile_body waits for each weight chunk just before reading it, so
  # the DMAs overlap compute. More tiles: the pipeline body runs once per tile,
  # so its waits would need a per-tile branch (a scheduling barrier); instead
  # outer_pipeline waits for every chunk once, before the pipeline. Measured:
  # with one tile, overlapping was 7-16% faster than waiting up front; with
  # more tiles, per-tile waits under pl.when were 6-36% slower.
  overlap_weight_dma: bool
  ba_rows: int
  vmem_limit_bytes: int  # Tiles and scratch, plus the weights' VMEM copies.

  @classmethod
  def create(
      cls,
      model: model_config.GdnConfig,
      x: jax.Array,
      residual: jax.Array | None,
      weights: model_config.GdnWeights,
      *,
      tile_size: int | None = None,
  ) -> "FusedGdnConfig":
    """Makes every decision, then asserts it is one fused_gdn implements."""
    num_tokens, hidden_size = x.shape
    tile = tiling.TileConfig.create(
        num_tokens=num_tokens,
        fp8=model.fp8,
        resident_bytes=sum(memory.nbytes(w) for w in jax.tree.leaves(weights)),
        bytes_per_row=tiling.total_bytes(bytes_per_row(model, hidden_size)),
        tile_size=tile_size,
    )
    col_block = quantized_matmul.head_column_block(
        model.head_dim, model.num_cols
    )
    col_blocks = tuple(
        (col, col_block) for col in range(0, model.num_cols, col_block)
    )
    two_groups = 2 * tile.matmul_rows
    cfg = cls(
        model=model,
        hidden_size=hidden_size,
        tile=tile,
        col_block=col_block,
        col_blocks=col_blocks,
        w_chunks=memory.stream_chunks(
            col_blocks, tile.policy.stream_blocks_per_chunk * col_block
        ),
        ba_chunks=((0, weights.ba_w.shape[1]),),
        overlap_weight_dma=tile.num_tiles == 1,
        ba_rows=two_groups
        if tile.tile_size % two_groups == 0
        else tile.matmul_rows,
        vmem_limit_bytes=tile.vmem_limit_bytes
        + memory.nbytes(weights.w)
        + memory.nbytes(weights.ba_w),
    )
    cfg._check(residual)
    return cfg

  def _check(self, residual: jax.Array | None) -> None:
    """Asserts the decisions are ones fused_gdn implements."""
    m = self.model
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert m.fp8, "fused_gdn is fp8-only"
    assert (residual is not None) == m.residual
    assert self.num_tokens % num_lanes == 0, "pad T to a multiple of 128"
    assert m.head_dim == num_lanes
    assert m.num_v_heads % 2 == 0 and m.num_v_heads <= num_lanes // 2
    assert m.qkv_cols % num_lanes == 0
    assert self.hidden_size % m.act_quant_block == 0
    # No block straddles mixed_qkv and z, and each z block is one head pair.
    assert m.z_col % self.col_block == 0 and self.col_block == 2 * m.head_dim

  @property
  def num_tokens(self) -> int:  # T.
    return self.tile.num_tokens

  @property
  def kernel_name(self) -> str:
    """e.g. fused_qkv_gdn_t256."""
    return f"fused_qkv_gdn_t{self.tile.tile_size}"

  def metadata(self) -> dict[str, str]:
    """The decisions above, serialized into the HLO for debugging."""
    return {
        **self.tile.metadata(),
        "col_block": str(self.col_block),
        "ba_rows": str(self.ba_rows),
        "w_chunks": str(self.w_chunks),
        "overlap_weight_dma": str(self.overlap_weight_dma),
        "norm_dtype": str(self.model.norm_dtype),
        "vmem_limit_bytes": str(self.vmem_limit_bytes),
    }


class RowBytes(NamedTuple):
  """fused_gdn's VMEM bytes per token row; the outputs are double-buffered."""

  inputs: tiling.InputRowBytes
  normed: int  # Scratch: bf16 h * w, in_proj_ba's input.
  mixed_qkv: int  # Output block.
  z: int  # Output block.
  b: int  # Output block.
  a: int  # Output block.


def bytes_per_row(model: model_config.GdnConfig, hidden_size: int) -> RowBytes:
  """fused_gdn's VMEM bytes per token row."""
  num_lanes = pltpu.get_tpu_info().num_lanes
  return RowBytes(
      inputs=tiling.input_bytes_per_row(model, hidden_size),
      normed=layout_util.align(hidden_size, num_lanes) * 2,
      mixed_qkv=2 * model.qkv_cols * 4,
      z=2 * model.value_dim * 2,
      b=2 * num_lanes * 4,
      a=2 * num_lanes * 4,
  )
