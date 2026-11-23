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
"""fused_qkv's kernel body: streams W into VMEM, then runs tile_body.

For each token tile, tile_body computes:

  Step 0: Add the residual, apply the input norm, and (fp8) quantize:
            h = x + residual
            fp8:  x_q = fp8(bf16(h * w_norm) / x_scale), one x_scale per row
                  per quant block. 1/rms(h) is one number per row, so it is
                  saved as row_scale and multiplied into the matmul outputs.
            bf16: x_q = bf16(norm(h)), with the norm in f32.
  Step 1: Multiply by W = [q (+ gate) | k | v] in f32 and undo every scale.
            fp8:  qkv = sum_k(x_scale_k * (x_q_k @ W_k)) * w_scale * row_scale
                        + bias
            bf16: qkv = x_q @ W + bias
  Step 2: Normalize K and V per head, apply RoPE to K, and cast to fp8.
            k = fp8(rope(k_norm(k)) / k_scale),  v = fp8(v_norm(v) / v_scale)
  Step 3: Normalize Q per head, apply RoPE, and pack q_packing heads per u32.
            q = rope(q_norm(q))
  Step 4: Copy the gate columns out.
"""

from collections.abc import Callable, Iterator
import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import memory
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import compute
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import config as config_lib
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import memory_ref


def outer_pipeline(
    inputs: memory_ref.Inputs[jax.Ref],  # HBM, or VMEM with a single tile.
    w_hbm: jax.Ref,  # (K, N) W, left in HBM.
    pinned: model_config.PinnedGqaWeights[jax.Ref],  # VMEM refs.
    outputs: memory_ref.Outputs[jax.Ref],  # HBM.
    scratch: memory_ref.Scratch[jax.Ref],
    *,
    cfg: config_lib.FusedQkvConfig,
) -> None:
  """Streams W into VMEM and runs tile_body over the token tiles."""
  w = memory.StreamedWeight(w_hbm, scratch.w_vmem, scratch.w_sems, cfg.w_chunks)
  w.start()
  if not cfg.overlap_weight_dma:
    w.wait_all()  # tile_body's wait_for calls then emit nothing.
  kv_pad = None
  if cfg.kv_pad_starts:
    assert scratch.kv_pad_zeros is not None
    assert scratch.kv_pad_sem is not None
    kv_pad = memory_ref.KvPadZeroing(
        cfg, outputs.kv, scratch.kv_pad_zeros, scratch.kv_pad_sem
    )
    kv_pad.start()
  pltpu.emit_pipeline(
      functools.partial(tile_body, cfg=cfg, w=w),
      grid=(cfg.tile.num_tiles,),
      in_specs=(memory_ref.Inputs.block_specs(cfg),),
      out_specs=(memory_ref.Outputs.block_specs(cfg),),
  )(inputs, outputs, scratches=(pinned, scratch))
  if kv_pad is not None:
    kv_pad.wait()


def tile_body(
    inputs: memory_ref.Inputs[jax.Ref],  # This tile's VMEM blocks.
    outputs: memory_ref.Outputs[jax.Ref],  # This tile's VMEM blocks.
    pinned: model_config.PinnedGqaWeights[jax.Ref],  # VMEM refs.
    scratch: memory_ref.Scratch[jax.Ref],
    *,
    cfg: config_lib.FusedQkvConfig,
    w: memory.StreamedWeight,  # Its VMEM copy is scratch.w_vmem.
) -> None:
  """One token tile: Steps 0-4 of the module docstring."""
  cos = sin = None
  if cfg.model.rope is not None:
    assert inputs.cos is not None and inputs.sin is not None
    cos, sin = inputs.cos[...], inputs.sin[...]
  # Steps 0 and 1, interleaved per group of matmul rows.
  _prep_and_project(inputs, outputs, pinned, scratch, cfg=cfg, w=w)
  rope = compute.make_rope(cfg, cos, sin)
  if cfg.two_heads_per_vreg:  # D = 64: two heads per vreg (compute.py).
    kv_heads, q_heads = compute.kv_head_pairs, compute.packed_q_head_pair
  else:
    kv_heads, q_heads = compute.kv_heads, compute.packed_q_heads
  # Step 2.
  _store_kv(cfg, outputs.kv, kv_heads(cfg, scratch.qkv, pinned, rope))
  # Step 3.
  _store_q(
      cfg, outputs.q, functools.partial(q_heads, cfg, scratch.qkv, pinned, rope)
  )
  # Step 4.
  if cfg.model.gate:
    assert outputs.gate is not None
    _copy_gate(cfg, outputs.gate, scratch.qkv)


def _prep_and_project(
    inputs: memory_ref.Inputs[jax.Ref],
    outputs: memory_ref.Outputs[jax.Ref],
    pinned: model_config.PinnedGqaWeights[jax.Ref],
    scratch: memory_ref.Scratch[jax.Ref],
    *,
    cfg: config_lib.FusedQkvConfig,
    w: memory.StreamedWeight,
) -> None:
  """Steps 0 and 1: per matmul_rows group, prep its rows, then project them."""
  prep_chunk = (
      compute.prep_chunk_fp8 if cfg.model.fp8 else compute.prep_chunk_bf16
  )
  row_scale = scratch.row_scale if cfg.defers_inv_rms else None
  for first_row in cfg.tile.row_groups():
    rows = pl.ds(first_row, cfg.tile.matmul_rows)
    # Step 0.
    for chunk in cfg.tile.prep_chunks(first_row):
      prep_chunk(cfg, inputs, outputs, pinned, scratch, chunk)
    # Step 1.
    for first_col, width in cfg.col_blocks:
      cols = pl.ds(first_col, width)
      w.wait_for(first_col)
      acc = quantized_matmul.project_block(
          rows,
          cols,
          x_q_ref=scratch.x_q,
          w_ref=scratch.w_vmem,
          x_scale_ref=scratch.x_scale,
          row_scale_ref=row_scale,
          w_scale_ref=pinned.w_scale,
          bias_ref=pinned.bias,
      )
      scratch.qkv[rows, cols] = acc.astype(scratch.qkv.dtype)


def _store_kv(
    cfg: config_lib.FusedQkvConfig,
    kv_ref: jax.Ref,  # outputs.kv
    heads: Iterator[jax.Array],  # fp8 heads in cache order.
) -> None:
  """Step 2's stores, in cfg.kv_layout."""
  m = cfg.model
  if cfg.seq_along_lane:
    layout_util.store_kv_seq_along_lane(
        kv_ref, heads, m.head_dim, m.kv_packing, cfg.tile.tile_size
    )
  else:
    layout_util.store_kv_head_along_sublane(
        kv_ref, heads, m.kv_packing, cfg.kv_folds
    )


def _store_q(
    cfg: config_lib.FusedQkvConfig,
    q_ref: jax.Ref,  # outputs.q
    q_heads: Callable[[int, int], jax.Array],  # (kv_head, group) -> packed u32
) -> None:
  """Step 3's stores: each kv_head's groups of packed Q heads."""
  m = cfg.model
  for kv_head in range(m.num_kv_heads):
    for group in range(m.q_per_kv // m.q_packing):
      layout_util.strided_store_heads(
          q_ref.at[kv_head], group, q_heads(kv_head, group)
      )


def _copy_gate(
    cfg: config_lib.FusedQkvConfig,
    gate_ref: jax.Ref,  # outputs.gate
    qkv: jax.Ref,  # scratch.qkv
) -> None:
  """Step 4: copy each Q head's gate columns out of the projection output."""
  m = cfg.model
  for q_head in range(m.num_q_heads):
    gate_ref[:, pl.ds(q_head * m.head_dim, m.head_dim)] = qkv[
        :, pl.ds(m.gate_col(q_head), m.head_dim)
    ]
