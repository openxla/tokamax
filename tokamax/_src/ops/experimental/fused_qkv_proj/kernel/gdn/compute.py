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
"""fused_gdn's math for pipeline.tile_body: Step 0, and Steps 1 and 2's stores.

Step 2's dot is short enough to stay inline in tile_body. Each step here reads
and writes the VMEM refs itself, at the same points a single
fused loop would: every value is stored as soon as it is computed, so few vregs
stay live.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float32  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import input_prep
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gdn import config as config_lib
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gdn import memory_ref


def prep_chunk(
    cfg: config_lib.FusedGdnConfig,
    inputs: memory_ref.Inputs[jax.Ref],
    outputs: memory_ref.Outputs[jax.Ref],
    pinned: model_config.PinnedGdnWeights[jax.Ref],
    scratch: memory_ref.Scratch[jax.Ref],
    rows: pl.Slice,
) -> None:
  """Step 0 on `rows`: residual add, deferred input norm, sub-channel fp8 quant.

  As fused_qkv's, but h * norm weight also goes to `normed` (in_proj_ba's
  unquantized input), and 1/rms is applied to both projections' outputs.

  Args:
    cfg: the kernel's config.
    inputs: this tile's input blocks.
    outputs: this tile's output blocks.
    pinned: the pinned weights, in VMEM.
    scratch: the kernel's VMEM scratch.
    rows: the rows to prepare.
  """
  m = cfg.model
  residual = None
  if m.residual:
    assert inputs.residual is not None and outputs.residual is not None
    residual = input_prep.Residual(inputs.residual, outputs.residual)
  input_norm = None
  if m.has_input_norm:
    assert pinned.norm_w is not None and scratch.row_scale is not None
    input_norm = input_prep.InputNorm(pinned.norm_w, scratch.row_scale)
  input_prep.add_norm_quantize(
      rows,
      act_quant_block=m.act_quant_block,
      eps=m.eps,
      x_ref=inputs.x,
      residual=residual,
      input_norm=input_norm,
      x_q_ref=scratch.x_q,
      x_scale_ref=scratch.x_scale,
      normed_ref=scratch.normed,
      f32_norm=jnp.dtype(m.norm_dtype) == jnp.float32,
  )


def store_qkvz_block(
    cfg: config_lib.FusedGdnConfig,
    outputs: memory_ref.Outputs[jax.Ref],  # Tile blocks.
    first_row: int,
    rows: pl.Slice,
    col: int,
    acc: Float32[Array, "rows cols"],
) -> None:
  """Step 1's output: an in_proj_qkvz block goes to mixed_qkv or z."""
  if col < cfg.model.z_col:
    # mixed_qkv is compact: each token's lane chunks are consecutive rows.
    layout_util.strided_store(outputs.mixed_qkv, first_row, 1, acc, col=col)
  else:
    pair = (col - cfg.model.z_col) // (2 * cfg.model.head_dim)
    layout_util.store_head_pair(outputs.z, rows, pair, acc)


def store_ba(
    outputs: memory_ref.Outputs[jax.Ref],
    first_row: int,
    ba: Float32[Array, "rows lanes"],
) -> None:
  """Step 2's output: b is lanes 0:64 of ba and a lanes 64:128.

  Each is stored at lane 0 with the other half zeroed.

  Args:
    outputs: this tile's output blocks.
    first_row: ba's first row in the tile.
    ba: (rows, lanes) f32 in_proj_ba output.
  """
  half = pltpu.get_tpu_info().num_lanes // 2
  low = jax.lax.broadcasted_iota(jnp.int32, ba.shape, 1) < half
  layout_util.strided_store(outputs.b, first_row, 1, jnp.where(low, ba, 0.0))
  a = pltpu.roll(ba, shift=half, axis=1)
  layout_util.strided_store(outputs.a, first_row, 1, jnp.where(low, a, 0.0))
