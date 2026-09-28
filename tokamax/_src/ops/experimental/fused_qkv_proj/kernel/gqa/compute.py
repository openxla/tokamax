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
"""fused_qkv's math for Steps 0, 2 and 3 of pipeline.tile_body.

Steps 1 (one quantized_matmul.project_block call per column block) and 4 (a
copy) are short enough to stay inline in tile_body. Each step here reads and
writes the VMEM refs itself, at the same points a single fused loop would:
every value is stored as soon as it is computed, so few vregs stay live.
"""

from collections.abc import Callable, Iterator
import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, BFloat16, Float, Float8e4m3fn, UInt32  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import input_prep
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import norm
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import rope as rope_lib
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import config as config_lib
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import memory_ref

Rope = Callable[[Float[Array, "tile width"]], Float[Array, "tile width"]]


# Step 0: input prep.


def prep_chunk_fp8(
    cfg: config_lib.FusedQkvConfig,
    inputs: memory_ref.Inputs[jax.Ref],  # Tile blocks.
    outputs: memory_ref.Outputs[jax.Ref],  # Tile blocks.
    pinned: model_config.PinnedGqaWeights[jax.Ref],  # VMEM refs.
    scratch: memory_ref.Scratch[jax.Ref],
    rows: pl.Slice,
) -> None:
  """Residual add, deferred input norm and sub-channel fp8 quant into x_q."""
  m = cfg.model
  assert scratch.x_scale is not None  # fp8 scratch.
  residual = None
  if m.residual:
    assert inputs.residual is not None and outputs.residual is not None
    residual = input_prep.Residual(inputs.residual, outputs.residual)
  input_norm = None
  if m.has_input_norm:
    assert pinned.input_norm_w is not None and scratch.row_scale is not None
    input_norm = input_prep.InputNorm(pinned.input_norm_w, scratch.row_scale)
  input_prep.add_norm_quantize(
      rows,
      act_quant_block=m.act_quant_block,
      eps=m.eps,
      x_ref=inputs.x,
      residual=residual,
      input_norm=input_norm,
      x_q_ref=scratch.x_q,
      x_scale_ref=scratch.x_scale,
  )


def prep_chunk_bf16(
    cfg: config_lib.FusedQkvConfig,
    inputs: memory_ref.Inputs[jax.Ref],
    outputs: memory_ref.Outputs[jax.Ref],
    pinned: model_config.PinnedGqaWeights[jax.Ref],
    scratch: memory_ref.Scratch[jax.Ref],
    rows: pl.Slice,
) -> None:
  """Residual add and input norm of `rows` into x_q, in f32 as in vLLM.

  Two passes over column blocks, the f32 sum of squares and then the normed
  rows (redoing the add), keep only a few vregs live.

  Args:
    cfg: the kernel's config.
    inputs: this tile's input blocks.
    outputs: this tile's output blocks.
    pinned: the pinned weights, in VMEM.
    scratch: the kernel's VMEM scratch.
    rows: the rows to prepare.
  """
  m = cfg.model
  block_width = cfg.tile.norm_cols
  col_blocks = [
      pl.ds(start, min(block_width, cfg.hidden_size - start))
      for start in range(0, cfg.hidden_size, block_width)
  ]

  def add_residual(cols: pl.Slice) -> Float[Array, "rows cols"]:
    h = inputs.x[rows, cols].astype(jnp.float32)
    if m.residual:
      assert inputs.residual is not None
      h = h + inputs.residual[rows, cols].astype(jnp.float32)
    return h

  inv_rms = None
  if m.has_input_norm:
    sum_sq = jnp.zeros((cfg.tile.prep_rows, 1), jnp.float32)
    for cols in col_blocks:
      sum_sq = sum_sq + norm.sum_sq(add_residual(cols))
    inv_rms = norm.inv_rms_from_sum_sq(sum_sq, cfg.hidden_size, m.eps)
  norm_w = pinned.input_norm_w
  for cols in col_blocks:
    h = add_residual(cols)
    if m.residual:
      assert outputs.residual is not None
      outputs.residual[rows, cols] = h.astype(outputs.residual.dtype)
    scratch.x_q[rows, cols] = norm.scale_norm(
        m.input_norm,
        h,
        inv_rms,
        None if norm_w is None else norm_w[:, cols],
        scratch.x_q.dtype,
    )


# Steps 2 and 3: the epilogue over the tile's projection output.


def make_rope(
    cfg: config_lib.FusedQkvConfig,
    cos: BFloat16[Array, "tile n_pairs"] | None,
    sin: BFloat16[Array, "tile n_pairs"] | None,
) -> Rope:
  """RoPE of one head (or one two-head vreg) read from qkv."""
  if cfg.lane_rope:
    assert cfg.model.rope is not None and cos is not None and sin is not None
    return rope_lib.LaneRope.expand(
        cfg.model.rope,
        cfg.model.head_dim,
        cos,
        sin,
        jnp.bfloat16,
        precision=cfg.model.rope_precision,
    )
  return functools.partial(
      rope_lib.rope,
      cfg.model.rope,
      cos=cos,
      sin=sin,
      precision=cfg.model.rope_precision,
  )


def kv_heads(
    cfg: config_lib.FusedQkvConfig,
    qkv: jax.Ref,
    pinned: model_config.PinnedGqaWeights[jax.Ref],
    rope: Rope,
) -> Iterator[Float8e4m3fn[Array, "..."]]:
  """Yields fp8 K/V heads in cache order: k_0, v_0, k_1, v_1, ...

  Each head is k/v norm, RoPE (K only) and the static fp8 cast. SEQ_ALONG_LANE
  heads are transposed to (D, tile), tokens on lanes; HEAD_ALONG_SUBLANE heads
  stay (tile, D). Yielding one head at a time lets the caller store it before
  the next is computed.

  Args:
    cfg: the kernel's config.
    qkv: (tile, N) bf16 projection output.
    pinned: the pinned weights, in VMEM.
    rope: the tile's RoPE (make_rope).

  Yields:
    One fp8 head per KV slot.
  """
  m = cfg.model
  for kv_head in range(m.num_kv_heads):
    k = norm.norm(
        m.k_norm,
        _read(qkv, m.k_col(kv_head), m.head_dim),
        None if pinned.k_norm_w is None else pinned.k_norm_w[...],
        m.eps,
    )
    v = norm.norm(
        m.v_norm, _read(qkv, m.v_col(kv_head), m.head_dim), None, m.eps
    )
    k = _to_kv_layout(cfg, rope(k))
    v = _to_kv_layout(cfg, v)
    yield quantized_matmul.quantize_fp8_per_tensor(k, m.k_scale)
    yield quantized_matmul.quantize_fp8_per_tensor(v, m.v_scale)


def packed_q_heads(
    cfg: config_lib.FusedQkvConfig,
    qkv: jax.Ref,
    pinned: model_config.PinnedGqaWeights[jax.Ref],
    rope: Rope,
    kv_head: int,
    group: int,
) -> UInt32[Array, "tile Dp"]:
  """Q heads of `group` (of kv_head), normed, RoPE'd and packed per u32.

  A group is the q_packing heads batched_rpa stores in one u32 (2 for bf16).

  Args:
    cfg: the kernel's config.
    qkv: (tile, N) bf16 projection output.
    pinned: the pinned weights, in VMEM.
    rope: the tile's RoPE (make_rope).
    kv_head: the KV head whose query heads these are.
    group: which group of kv_head's query heads.

  Returns:
    (tile, Dp) u32.
  """
  m = cfg.model
  first_q_head = kv_head * m.q_per_kv + group * m.q_packing
  return layout_util.pack_heads([
      rope(
          norm.norm(
              m.q_norm,
              _read(qkv, m.q_col(first_q_head + i), m.head_dim),
              None if pinned.q_norm_w is None else pinned.q_norm_w[...],
              m.eps,
          )
      )
      for i in range(m.q_packing)
  ])


# D = 64 (cfg.two_heads_per_vreg, gpt-oss): heads 2i and 2i + 1 go through the
# epilogue together in one 128-lane vreg. Config checks there are no per-head
# norms and that RoPE is LaneRope, which rotates both heads at once. Same
# signatures as above.


def kv_head_pairs(
    cfg: config_lib.FusedQkvConfig,
    qkv: jax.Ref,  # (tile, N) bf16
    pinned: model_config.PinnedGqaWeights[
        jax.Ref
    ],  # Unused: no per-head norms.
    rope: Rope,
) -> Iterator[Float8e4m3fn[Array, "..."]]:
  """kv_heads for D = 64: RoPE (K only) and the fp8 cast, two heads per read."""
  del pinned
  m = cfg.model
  for kv_head in range(0, m.num_kv_heads, 2):
    k = _read(qkv, m.k_col(kv_head), 2 * m.head_dim)
    v = _read(qkv, m.v_col(kv_head), 2 * m.head_dim)
    k_heads = _split_pair(cfg, _to_kv_layout(cfg, rope(k)))
    v_heads = _split_pair(cfg, _to_kv_layout(cfg, v))
    for k_head, v_head in zip(k_heads, v_heads):
      yield quantized_matmul.quantize_fp8_per_tensor(k_head, m.k_scale)
      yield quantized_matmul.quantize_fp8_per_tensor(v_head, m.v_scale)


def packed_q_head_pair(
    cfg: config_lib.FusedQkvConfig,
    qkv: jax.Ref,  # (tile, N) bf16
    pinned: model_config.PinnedGqaWeights[
        jax.Ref
    ],  # Unused: no per-head norms.
    rope: Rope,
    kv_head: int,
    group: int,
) -> UInt32[Array, "tile lanes"]:
  """packed_q_heads for D = 64: both heads of the group in one vreg."""
  del pinned
  m = cfg.model
  first_q_head = kv_head * m.q_per_kv + group * m.q_packing
  q = rope(_read(qkv, m.q_col(first_q_head), 2 * m.head_dim))
  return layout_util.pack_lane_pair(q, m.head_dim)


def _read(qkv: jax.Ref, col: int, width: int) -> BFloat16[Array, "tile width"]:
  """qkv's columns [col, col + width): one head, or a D = 64 head pair."""
  return qkv[:, pl.ds(col, width)]


def _to_kv_layout(
    cfg: config_lib.FusedQkvConfig, y: Float[Array, "tile width"]
) -> Float[Array, "..."]:
  """f32; SEQ_ALONG_LANE: (tile, width) -> (width, tile), tokens on lanes."""
  y = y.astype(jnp.float32)
  return y.T if cfg.seq_along_lane else y


def _split_pair(
    cfg: config_lib.FusedQkvConfig, y: Float[Array, "..."]
) -> list[Float[Array, "..."]]:
  """A D = 64 head pair, processed in one 128-lane vreg, as two heads to store."""
  head_dim = cfg.model.head_dim
  if cfg.seq_along_lane:
    # y is (2D, tile): head 0 is rows 0:D, head 1 is rows D:2D.
    return [y[:head_dim], y[head_dim:]]
  # y is (tile, 2D): head 0 is lanes 0:D, head 1 is lanes D:2D.
  return [y[:, :head_dim], pltpu.roll(y, head_dim, 1)[:, :head_dim]]
