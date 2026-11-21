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
"""fused_out_proj's kernel body: streams W into VMEM, then runs tile_body.

For each token tile, tile_body computes:

  Step 0: Unpack o's packed query heads, apply the optional gate, and (fp8)
          quantize:
            x = o * sigmoid(gate)
            fp8:  x_q = fp8(x / x_scale), one x_scale per row per quant block
            bf16: x_q = x
  Step 1: Multiply by W_o in f32 and undo every scale.
            fp8:  out = sum_k(x_scale_k * (x_q_k @ W_k)) * w_scale + bias
            bf16: out = x_q @ W + bias
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import memory
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import compute
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import config as config_lib
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import memory_ref


def outer_pipeline(
    inputs: memory_ref.Inputs[jax.Ref],  # HBM, or VMEM with a single tile.
    w_hbm: jax.Ref,  # (Hq * D, K) W, left in HBM.
    pinned: model_config.PinnedOutProjWeights[jax.Ref],  # VMEM refs.
    outputs: memory_ref.Outputs[jax.Ref],  # HBM.
    scratch: memory_ref.Scratch[jax.Ref],
    *,
    cfg: config_lib.FusedOutProjConfig,
) -> None:
  """Streams W into VMEM and runs tile_body over the token tiles."""
  w = memory.StreamedWeight(w_hbm, scratch.w_vmem, scratch.w_sems, cfg.w_chunks)
  w.start()
  if not cfg.overlap_weight_dma:
    w.wait_all()  # tile_body's wait_for calls then emit nothing.
  pltpu.emit_pipeline(
      functools.partial(tile_body, cfg=cfg, w=w),
      grid=(cfg.tile.num_tiles,),
      in_specs=(memory_ref.Inputs.block_specs(cfg),),
      out_specs=(memory_ref.Outputs.block_specs(cfg),),
  )(inputs, outputs, scratches=(pinned, scratch))


def tile_body(
    inputs: memory_ref.Inputs[jax.Ref],  # This tile's VMEM blocks.
    outputs: memory_ref.Outputs[jax.Ref],  # This tile's VMEM blocks.
    pinned: model_config.PinnedOutProjWeights[jax.Ref],  # VMEM refs.
    scratch: memory_ref.Scratch[jax.Ref],
    *,
    cfg: config_lib.FusedOutProjConfig,
    w: memory.StreamedWeight,  # Its VMEM copy is scratch.w_vmem.
) -> None:
  """One token tile: Steps 0-1 of the module docstring."""
  prep_chunk = (
      compute.prep_chunk_fp8 if cfg.model.fp8 else compute.prep_chunk_bf16
  )
  for first_row in cfg.tile.row_groups():
    rows = pl.ds(first_row, cfg.tile.matmul_rows)
    # Step 0.
    for chunk in cfg.tile.prep_chunks(first_row):
      prep_chunk(cfg, inputs, scratch, chunk)
    # Step 1.
    for col, width in cfg.col_blocks:
      cols = pl.ds(col, width)
      w.wait_for(col)
      acc = quantized_matmul.project_block(
          rows,
          cols,
          x_q_ref=scratch.x_q,
          w_ref=scratch.w_vmem,
          x_scale_ref=scratch.x_scale,
          w_scale_ref=pinned.w_scale,
          bias_ref=pinned.bias,
      )
      outputs.out[rows, cols] = acc.astype(outputs.out.dtype)
