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
"""Step 0 shared by fused_qkv (fp8) and fused_gdn: add, input norm, fp8 quant.

The norm is deferred (norm.py, FlashNorm): rows are multiplied by the norm
weight and their squares summed while x streams through, and 1/rms is saved as
a per-row scale for the projection outputs. The quant is sub-channel: one
scale per row and act_quant_block of columns. The math is bf16, two values per
lane, except the f32 sum of squares and, with f32_norm, h * w.
"""

from typing import NamedTuple

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import norm
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul


class Residual(NamedTuple):
  """The residual add: x + in_ref, also stored to out_ref."""

  in_ref: jax.Ref  # (tile, K) bf16
  out_ref: jax.Ref  # (tile, K) bf16


class InputNorm(NamedTuple):
  """The deferred input norm's weight and its per-row 1/rms output."""

  w_ref: jax.Ref  # layout_util.broadcast_row's layout.
  row_scale_ref: jax.Ref  # (tile, lanes) f32


def add_norm_quantize(
    rows: pl.Slice,
    *,
    act_quant_block: int,
    eps: float,
    x_ref: jax.Ref,
    residual: Residual | None,
    input_norm: InputNorm | None,
    x_q_ref: jax.Ref,
    x_scale_ref: jax.Ref,
    normed_ref: jax.Ref | None = None,
    f32_norm: bool = False,
) -> None:
  """Residual add, norm weight and sub-channel fp8 quant of `rows` into x_q.

  Scales are stored broadcast across lanes, so the matmul reads them as plain
  vregs.

  Args:
    rows: the tile rows to process (one fp8 sublane tile).
    act_quant_block: columns per fp8 quant block; one scale per row and block.
    eps: the input norm's epsilon.
    x_ref: (tile, K) bf16 decoder hidden states.
    residual: the residual to add to x and where to store x + residual, or None
      for no residual add.
    input_norm: the norm weight and the deferred 1/rms output, or None for no
      input norm.
    x_q_ref: (tile, K) fp8 output, the quantized h * w.
    x_scale_ref: (K / block, tile, lanes) f32 output, x_q's per-block scales.
    normed_ref: optional (tile, K) bf16 output for h * w before the quant
      (fused_gdn's in_proj_ba input).
    f32_norm: compute h * w in f32, rounded to bf16 once (vLLM's precision).
  """
  hidden_size = x_ref.shape[1]
  num_rows = rows.size
  assert isinstance(num_rows, int)
  sum_sq = jnp.zeros((num_rows, 1), jnp.float32)
  for block in range(hidden_size // act_quant_block):
    cols = pl.ds(block * act_quant_block, act_quant_block)

    # Residual add.
    h = x_ref[rows, cols]
    if residual is not None:
      h = h + residual.in_ref[rows, cols]
      residual.out_ref[rows, cols] = h

    # Norm weight; 1/rms is applied later, from the sum of squares.
    normed = h
    if input_norm is not None:
      h_f32 = h.astype(jnp.float32)
      sum_sq = sum_sq + norm.sum_sq(h_f32)
      w_dtype = jnp.float32 if f32_norm else h.dtype
      norm_w = layout_util.broadcast_row(
          input_norm.w_ref, block, num_rows, w_dtype
      )
      if f32_norm:
        normed = (h_f32 * norm_w).astype(h.dtype)
      else:
        normed = h * norm_w

    # Quantize.
    if normed_ref is not None:
      normed_ref[rows, cols] = normed
    x_q, x_scale = quantized_matmul.quantize_block(normed)
    x_q_ref[rows, cols] = x_q
    x_scale_ref[block, rows] = layout_util.broadcast_lanes(x_scale)

  # Deferred 1/rms, one per row.
  if input_norm is not None:
    inv_rms = norm.inv_rms_from_sum_sq(sum_sq, hidden_size, eps)
    input_norm.row_scale_ref[rows] = layout_util.broadcast_lanes(inv_rms)
