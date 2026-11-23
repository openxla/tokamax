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
"""The projections' numerics: fp8 quant of x, then x @ w dequantized in f32.

The kernels quantize x one act_quant_block of columns at a time, with one
scale per row and block (sub-channel). quantize_block makes those scales and
scaled_dot undoes them: each block's partial dot is scaled before the partial
dots are summed. Scales are stored broadcast across lanes. project_block is
one column block of a projection, fp8 or bf16, with every scale and the bias.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, BFloat16, Float, Float32, Float8e4m3fn  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import norm

F8 = jnp.float8_e4m3fn
F8_MAX = float(jnp.finfo(F8).max)


def quantize_fp8_per_token(
    x: Float[Array, "... K"],
) -> tuple[Float8e4m3fn[Array, "... K"], Float32[Array, "... 1"]]:
  """Dynamic per-row fp8 quant, x ≈ x_q * scale.

  vllm_torchtpu quantized_matmul/util.py: quantize_block(x, axis=-1).

  Args:
    x: (..., K).

  Returns:
    (x_q, scale): (..., K) fp8 and (..., 1) f32.
  """
  scale = jnp.max(jnp.abs(x), axis=-1, keepdims=True).astype(jnp.float32)
  scale = scale / F8_MAX
  scale = jnp.where(scale == 0, 1.0, scale)
  x_q = (x.astype(jnp.float32) / scale).clip(-F8_MAX, F8_MAX).astype(F8)
  return x_q, scale


def quantize_fp8_per_tensor(
    x: Float[Array, "*dims"], scale: float
) -> Float8e4m3fn[Array, "*dims"]:
  """fp8 cast with a fixed, calibrated per-tensor scale (the KV cache).

  vllm_torchtpu quantization.py: static_per_tensor_quantize_tensor.

  Args:
    x: any shape.
    scale: the calibrated scale, e.g. k_scale.

  Returns:
    x / scale in fp8, clipped to fp8's range.
  """
  return (x.astype(jnp.float32) / scale).clip(-F8_MAX, F8_MAX).astype(F8)


# In-kernel.


def quantize_block(
    x: BFloat16[Array, "rows block"],
) -> tuple[Float8e4m3fn[Array, "rows block"], Float32[Array, "rows 1"]]:
  """Sub-channel fp8 quant of one block: (x * bf16(1 / scale), scale).

  The multiply is bf16, two values per lane; vLLM divides in f32.

  Args:
    x: (rows, block) bf16, one activation quant block.

  Returns:
    (x_q, scale): (rows, block) fp8 and (rows, 1) f32.
  """
  amax = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(jnp.float32)
  scale = amax / F8_MAX
  scale = jnp.where(scale == 0, 1.0, scale)
  x_q = x * (1.0 / scale).astype(x.dtype)
  return x_q.clip(-F8_MAX, F8_MAX).astype(F8), scale


def scaled_dot(
    x_q_ref: jax.Ref,  # (tile, K) fp8
    x_scale_ref: jax.Ref,  # (K / block, tile, lanes) f32, lane-broadcast
    w_ref: jax.Ref,  # (K, N) fp8
    rows: pl.Slice,
    cols: pl.Slice,
) -> Float32[Array, "rows cols"]:
  """x[rows] @ w[:, cols] in f32, rescaled per block by x's scale only."""
  num_blocks = x_scale_ref.shape[0]
  block_width = x_q_ref.shape[1] // num_blocks
  acc = jnp.zeros((rows.size, cols.size), jnp.float32)
  width = cols.size
  assert isinstance(width, int)
  for block in range(num_blocks):
    k = pl.ds(block * block_width, block_width)  # This block's slice of K.
    acc += layout_util.repeat_lanes(x_scale_ref[block, rows], width) * jnp.dot(
        x_q_ref[rows, k], w_ref[k, cols], preferred_element_type=jnp.float32
    )
  return acc


# Projections.


def head_column_block(head_dim: int, num_cols: int) -> int:
  """Columns per MXU call: whole heads, at least one full MXU width."""
  tpu_info = pltpu.get_tpu_info()
  block = max(head_dim, tpu_info.mxu_column_size)
  if num_cols % block:
    block = max(head_dim, tpu_info.num_lanes)
  return block


def project_block(
    rows: pl.Slice,
    cols: pl.Slice,
    *,
    x_q_ref: jax.Ref,
    w_ref: jax.Ref,
    x_scale_ref: jax.Ref | None = None,
    row_scale_ref: jax.Ref | None = None,
    w_scale_ref: jax.Ref | None = None,
    bias_ref: jax.Ref | None = None,
) -> Float32[Array, "rows cols"]:
  """x[rows] @ w[:, cols] in f32, times every scale, plus the bias.

  fp8: each activation quant block has its own per-row scale, so the partial
  dots are scaled before they are summed (scaled_dot). Refs that are None are
  skipped.

  Args:
    rows: rows of x.
    cols: columns of w.
    x_q_ref: (tile, K) fp8 or bf16.
    w_ref: (K, N) fp8 or bf16.
    x_scale_ref: fp8: (K / block, tile, lanes) f32; None for bf16.
    row_scale_ref: (tile, lanes) f32 deferred 1/rms; None without one.
    w_scale_ref: (1, N) f32, or None.
    bias_ref: (1, N), or None.

  Returns:
    (rows, cols) f32.
  """
  if x_scale_ref is not None:
    acc = scaled_dot(x_q_ref, x_scale_ref, w_ref, rows, cols)
    if row_scale_ref is not None:
      acc = norm.apply_row_scale(acc, row_scale_ref[rows])
  else:
    acc = jnp.dot(
        x_q_ref[rows], w_ref[:, cols], preferred_element_type=jnp.float32
    )
  if w_scale_ref is not None:
    acc = acc * w_scale_ref[:, cols]
  if bias_ref is not None:
    acc = acc + bias_ref[:, cols].astype(jnp.float32)
  return acc
