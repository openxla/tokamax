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
"""fused_gdn's kernel body: streams the weights into VMEM, then runs tile_body.

For each token tile, tile_body computes:

  Step 0: Add the residual, apply the norm weight, and quantize to fp8:
            h      = x + residual
            normed = bf16(h * w_norm)
            x_q    = fp8(normed / x_scale), one x_scale per row per quant
                     block
          The norm's division by rms(h) is deferred: 1/rms(h) is one number
          per row, so it is saved as row_scale and multiplied into the matmul
          outputs.
  Step 1: Multiply by in_proj_qkvz in f32 and undo every scale.
            [mixed_qkv | z] = sum_k(x_scale_k * (x_q_k @ W_qkvz_k))
                              * w_scale * row_scale
  Step 2: Multiply the unquantized normed by in_proj_ba (bf16).
            [b | a] = (normed @ W_ba) * row_scale
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import memory
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import norm
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gdn import compute
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gdn import config as config_lib
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gdn import memory_ref


def outer_pipeline(
    inputs: memory_ref.Inputs[jax.Ref],  # HBM, or VMEM with a single tile.
    w_hbm: jax.Ref,  # (K, N) in_proj_qkvz, left in HBM.
    ba_hbm: jax.Ref,  # (K, lanes) in_proj_ba, left in HBM.
    pinned: model_config.PinnedGdnWeights[jax.Ref],  # VMEM refs.
    outputs: memory_ref.Outputs[jax.Ref],  # HBM.
    scratch: memory_ref.Scratch[jax.Ref],
    *,
    cfg: config_lib.FusedGdnConfig,
) -> None:
  """Streams both projection weights into VMEM; runs tile_body per tile."""
  w = memory.StreamedWeight(w_hbm, scratch.w_vmem, scratch.w_sems, cfg.w_chunks)
  ba_w = memory.StreamedWeight(
      ba_hbm, scratch.ba_vmem, scratch.ba_sems, cfg.ba_chunks
  )
  w.start()
  ba_w.start()
  if not cfg.overlap_weight_dma:
    # tile_body's wait_for calls then emit nothing.
    w.wait_all()
    ba_w.wait_all()
  pltpu.emit_pipeline(
      functools.partial(tile_body, cfg=cfg, w=w, ba_w=ba_w),
      grid=(cfg.tile.num_tiles,),
      in_specs=(memory_ref.Inputs.block_specs(cfg),),
      out_specs=(memory_ref.Outputs.block_specs(cfg),),
  )(inputs, outputs, scratches=(pinned, scratch))


def tile_body(
    inputs: memory_ref.Inputs[jax.Ref],  # This tile's VMEM blocks.
    outputs: memory_ref.Outputs[jax.Ref],  # This tile's VMEM blocks.
    pinned: model_config.PinnedGdnWeights[jax.Ref],  # VMEM refs.
    scratch: memory_ref.Scratch[jax.Ref],
    *,
    cfg: config_lib.FusedGdnConfig,
    w: memory.StreamedWeight,  # Its VMEM copy is scratch.w_vmem.
    ba_w: memory.StreamedWeight,  # Its VMEM copy is scratch.ba_vmem.
) -> None:
  """One token tile: Steps 0-2 of the module docstring."""
  has_input_norm = cfg.model.has_input_norm
  row_scale = scratch.row_scale if has_input_norm else None

  for first_row in cfg.tile.row_groups():
    rows = pl.ds(first_row, cfg.tile.matmul_rows)
    # Step 0.
    for chunk in cfg.tile.prep_chunks(first_row):
      compute.prep_chunk(cfg, inputs, outputs, pinned, scratch, chunk)
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
          row_scale_ref=row_scale,
          w_scale_ref=pinned.w_scale,
      )
      compute.store_qkvz_block(cfg, outputs, first_row, rows, col, acc)

  # Step 2.
  ba_w.wait_for(0)
  for first_row in range(0, cfg.tile.tile_size, cfg.ba_rows):
    rows = pl.ds(first_row, cfg.ba_rows)
    ba = jnp.dot(  # (ba_rows, lanes)
        scratch.normed[rows],
        scratch.ba_vmem[...],
        preferred_element_type=jnp.float32,
    )
    if has_input_norm:
      ba = norm.apply_row_scale(ba, scratch.row_scale[rows])
    compute.store_ba(outputs, first_row, ba)
