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
"""fused_qkv's static decisions, all made once per call by FusedQkvConfig.create.

Which model needs which variant: models.py presets (TP omitted), and the two
flags derived here.

                       qwen3_coder qwen3_5   gemma4_sliding gemma4_full gpt_oss
  fp8                  yes         yes       yes            yes         -
  head_dim             128         256       256            512         64
  gate                 -           yes       -              -           -
  v_from_k             -           -         -              yes         -
  bias                 -           -         -              -           yes
  residual             yes         yes       -              -           yes
  input norm           RMS         1+w       RMS            RMS         RMS
  q/k norm             RMS         1+w       RMS            RMS         -
  v norm               -           -         noscale        noscale     -
  RoPE dims (NeoX)     128         64 mrope  256            128 of 512  64
  lane_rope            yes         yes       -              -           yes
  two_heads_per_vreg   -           -         -              -           yes

Both KV layouts are used: HEAD_ALONG_SUBLANE (Gemma 4), SEQ_ALONG_LANE (Qwen).
"""

import dataclasses
from typing import NamedTuple

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import configs as rpa_configs
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import memory
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import rope as rope_lib
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import tiling


@dataclasses.dataclass(frozen=True)
class FusedQkvConfig:
  """Everything fused_qkv decides before tracing the kernel."""

  model: model_config.AttentionConfig
  hidden_size: int  # K.
  kv_layout: rpa_configs.KVLayout  # batched_rpa KV layout to write.
  page_size: int  # batched_rpa page size; SEQ_ALONG_LANE pads KV to it.
  tile: tiling.TileConfig
  col_block: int  # Projection columns per MXU call.
  # (first column, width) of each projection block, in compute order: K/V
  # first, so the K/V epilogue can start while the Q/gate dots run. w_chunks
  # is built from it, so tile_body reads W in the order it arrives.
  col_blocks: memory.ColumnChunks
  w_chunks: memory.ColumnChunks  # W's DMA chunks, in col_blocks order.
  # One tile: tile_body waits for each weight chunk just before reading it, so
  # the DMAs overlap compute. More tiles: the pipeline body runs once per tile,
  # so its waits would need a per-tile branch (a scheduling barrier); instead
  # outer_pipeline waits for every chunk once, before the pipeline. Measured:
  # with one tile, overlapping was 7-16% faster than waiting up front; with
  # more tiles, per-tile waits under pl.when were 6-36% slower.
  overlap_weight_dma: bool
  # D = 64: heads 2i and 2i + 1 share one 128-lane vreg through the epilogue.
  two_heads_per_vreg: bool
  lane_rope: bool  # RoPE as one lane gather + FMA (rope.LaneRope).
  vmem_limit_bytes: int  # Tiles and scratch, plus W's VMEM copy.

  @classmethod
  def create(
      cls,
      model: model_config.AttentionConfig,
      x: jax.Array | jax.ShapeDtypeStruct,  # Only its shape is used.
      residual: jax.Array | jax.ShapeDtypeStruct | None,
      weights: model_config.GqaWeights,
      *,
      kv_layout: rpa_configs.KVLayout,
      page_size: int,
      tile_size: int | None = None,
  ) -> "FusedQkvConfig":
    """Makes every decision, then asserts it is one fused_qkv implements."""
    num_tokens, hidden_size = x.shape
    tile = tiling.TileConfig.create(
        num_tokens=num_tokens,
        fp8=model.fp8,
        resident_bytes=sum(memory.nbytes(w) for w in jax.tree.leaves(weights)),
        bytes_per_row=tiling.total_bytes(
            bytes_per_row(model, kv_layout, hidden_size)
        ),
        tile_size=tile_size,
    )
    col_block = quantized_matmul.head_column_block(
        model.head_dim, model.num_cols
    )
    col_blocks = tuple(
        (block * col_block, col_block)
        for block in _kv_blocks_first(model, col_block)
    )
    two_heads = _two_heads_per_vreg(model)
    num_lanes = pltpu.get_tpu_info().num_lanes
    cfg = cls(
        model=model,
        hidden_size=hidden_size,
        kv_layout=kv_layout,
        page_size=page_size,
        tile=tile,
        col_block=col_block,
        col_blocks=col_blocks,
        w_chunks=memory.stream_chunks(
            col_blocks, tile.policy.stream_blocks_per_chunk * col_block
        ),
        overlap_weight_dma=tile.num_tiles == 1,
        two_heads_per_vreg=two_heads,
        # LaneRope rotates whole 128-lane vregs: one head, or two D = 64 heads.
        lane_rope=rope_lib.lane_rope_eligible(model.rope, model.head_dim)
        and (model.head_dim % num_lanes == 0 or two_heads),
        vmem_limit_bytes=tile.vmem_limit_bytes + memory.nbytes(weights.w),
    )
    cfg._check(residual, weights)
    return cfg

  def _check(
      self,
      residual: jax.Array | jax.ShapeDtypeStruct | None,
      weights: model_config.GqaWeights,
  ) -> None:
    """Asserts the decisions are ones fused_qkv implements."""
    m = self.model
    tpu_info = pltpu.get_tpu_info()
    num_lanes, num_sublanes = tpu_info.num_lanes, tpu_info.num_sublanes
    assert m.q_dtype == jnp.bfloat16, "only bf16 Q is implemented"
    assert m.kv_dtype == quantized_matmul.F8, "only fp8 KV is implemented"
    assert (residual is not None) == m.residual
    assert weights.w.shape == (self.hidden_size, m.num_cols)
    assert m.q_per_kv % m.q_packing == 0
    assert m.head_dim % (num_sublanes * m.kv_packing) == 0
    assert m.num_cols % max(m.head_dim, num_lanes) == 0
    assert not m.gate or m.head_dim % num_lanes == 0
    assert m.kv_col % self.col_block == 0
    if m.fp8:
      assert self.hidden_size % m.act_quant_block == 0
      assert self.col_block % num_lanes == 0
    if self.two_heads_per_vreg:  # compute.kv_head_pairs applies no norms.
      assert m.q_norm == m.k_norm == m.v_norm == model_config.Norm.NONE

  # Derived sizes.

  @property
  def num_tokens(self) -> int:  # T.
    return self.tile.num_tokens

  @property
  def defers_inv_rms(self) -> bool:
    """fp8 with an input norm: 1/rms is applied to the projection output."""
    return self.model.fp8 and self.model.has_input_norm

  @property
  def seq_along_lane(self) -> bool:
    return self.kv_layout == rpa_configs.KVLayout.SEQ_ALONG_LANE

  @property
  def kv_folds(self) -> int:
    """HEAD_ALONG_SUBLANE: 128-lane slices each K/V head is split into."""
    return kv_folds(self.model, self.kv_layout)

  @property
  def kv_block_lanes(self) -> int:
    """SEQ_ALONG_LANE: tokens per KV block, padded to whole lanes."""
    return layout_util.align(
        self.tile.tile_size, pltpu.get_tpu_info().num_lanes
    )

  @property
  def kv_tokens(self) -> int:
    """SEQ_ALONG_LANE: KV tokens, zero-padded to whole lanes and pages.

    batched_rpa reads new KV in page- and lane-aligned token spans.
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    return layout_util.align(self.num_tokens, max(num_lanes, self.page_size))

  @property
  def kv_pad_starts(self) -> tuple[int, ...]:
    """SEQ_ALONG_LANE: first token of each lane chunk past the last KV block.

    The pipeline's blocks never reach these chunks, so the kernel zeroes them
    with DMAs. (Lanes past T inside a short lone block are zeroed by the
    kernel's stores.)
    """
    if not self.seq_along_lane:
      return ()
    num_lanes = pltpu.get_tpu_info().num_lanes
    first = self.tile.num_tiles * self.kv_block_lanes
    return tuple(range(first, self.kv_tokens, num_lanes))

  # Names for profiles.

  @property
  def kernel_name(self) -> str:
    """e.g. fused_qkv_fp8_snh_gate_t256."""
    m = self.model
    parts = ["fused_qkv", "fp8" if m.fp8 else "bf16", self.kv_layout.symbol]
    if m.gate:
      parts.append("gate")
    parts.append(f"t{self.tile.tile_size}")
    return "_".join(parts)

  def metadata(self) -> dict[str, str]:
    """The decisions above, serialized into the HLO for debugging."""
    return {
        **self.tile.metadata(),
        "col_block": str(self.col_block),
        "col_blocks": str(self.col_blocks),
        "w_chunks": str(self.w_chunks),
        "overlap_weight_dma": str(self.overlap_weight_dma),
        "kv_layout": self.kv_layout.name,
        "kv_folds": str(self.kv_folds),
        "page_size": str(self.page_size),
        "two_heads_per_vreg": str(self.two_heads_per_vreg),
        "lane_rope": str(self.lane_rope),
        "vmem_limit_bytes": str(self.vmem_limit_bytes),
    }


def _kv_blocks_first(
    model: model_config.AttentionConfig, col_block: int
) -> tuple[int, ...]:
  num_blocks = model.num_cols // col_block
  first_kv_block = model.kv_col // col_block
  return (*range(first_kv_block, num_blocks), *range(first_kv_block))


def _two_heads_per_vreg(model: model_config.AttentionConfig) -> bool:
  """Whether D = 64 heads 2i and 2i + 1 go through the epilogue in one vreg."""
  # Every epilogue step must then handle both heads at once: no per-head
  # norms, gates or v_from_k, and RoPE must be LaneRope.
  return (
      2 * model.head_dim == pltpu.get_tpu_info().num_lanes
      and model.num_kv_heads % 2 == 0  # K/V heads pair up.
      and model.q_per_kv % 2 == 0  # Q heads pair up within each KV group.
      and model.q_packing == 2  # Q stores both heads in one u32.
      and not model.gate  # Gate columns sit between Q heads.
      and not model.v_from_k  # V is derived from K one head at a time.
      and model.q_norm == model.k_norm == model.v_norm == model_config.Norm.NONE
      and (
          model.rope is None
          or rope_lib.lane_rope_eligible(model.rope, model.head_dim)
      )
  )


class RowBytes(NamedTuple):
  """fused_qkv's VMEM bytes per token row; pipelined blocks double-buffered."""

  inputs: tiling.InputRowBytes
  rope: int  # cos and sin input blocks.
  q: int  # Output block.
  kv: int  # Output block.
  gate: int  # Output block.
  qkv: int  # Scratch: the projection output.


def kv_folds(
    model: model_config.AttentionConfig, kv_layout: rpa_configs.KVLayout
) -> int:
  """batched_rpa's lane folds of a K/V head (1 for SEQ_ALONG_LANE)."""
  return rpa_configs.num_lane_folds_in_head_dim(
      model.head_dim,
      jnp.dtype(model.kv_dtype),
      kv_layout,
      pltpu.get_tpu_info().num_lanes,
  )


def bytes_per_row(
    model: model_config.AttentionConfig,
    kv_layout: rpa_configs.KVLayout,
    hidden_size: int,
) -> RowBytes:
  """fused_qkv's VMEM bytes per token row."""
  num_lanes = pltpu.get_tpu_info().num_lanes
  padded_head_dim = layout_util.align(model.head_dim, num_lanes)
  bf16, kv_bytes = 2, jnp.dtype(model.kv_dtype).itemsize
  rope_bytes = 0
  if model.rope is not None:
    rope_bytes = 2 * layout_util.align(model.rope.n_pairs, num_lanes) * bf16
  match kv_layout:
    case rpa_configs.KVLayout.SEQ_ALONG_LANE:
      kv = 2 * model.num_kv_heads * model.head_dim * kv_bytes
    case rpa_configs.KVLayout.HEAD_ALONG_SUBLANE:
      folds = kv_folds(model, kv_layout)
      kv = (
          layout_util.align(2 * model.num_kv_heads * folds, model.kv_packing)
          * layout_util.align(model.head_dim // folds, num_lanes)
          * kv_bytes
      )
  return RowBytes(
      inputs=tiling.input_bytes_per_row(model, hidden_size),
      rope=2 * rope_bytes,
      q=2 * model.num_q_heads * padded_head_dim * bf16,
      kv=2 * kv,
      gate=2 * model.q_dim * bf16 if model.gate else 0,
      qkv=model.num_cols * bf16,
  )
