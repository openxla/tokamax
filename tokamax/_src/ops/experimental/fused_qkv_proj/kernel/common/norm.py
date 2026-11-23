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
"""RMS norms: vLLM's references and the kernels' deferred input norm.

The kernels split the input norm in two (FlashNorm,
https://arxiv.org/abs/2407.09577). While x streams through VMEM they only
multiply by the norm weight and sum the squares. 1/rms is one scalar per row,
so it commutes with the projection, (r * u) @ W = r * (u @ W), and is applied
to the f32 matmul output instead. With fp8 the quant never needs it: the
block scales absorb any per-row factor.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Float32, UInt32  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config


def norm(
    kind: model_config.Norm,
    x: Float[Array, "... K"],
    w: Float[Array, "1 K"] | None,
    eps: float,
    dtype: jax.typing.DTypeLike | None = None,
) -> Float[Array, "... K"]:
  """vLLM RMSNorm / GemmaRMSNorm forward_native, in dtype (default x's)."""
  dtype = dtype or x.dtype
  if kind == model_config.Norm.NONE:
    return x.astype(dtype)
  x_f32 = x.astype(jnp.float32)
  inv_rms = jax.lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
  return scale_norm(kind, x_f32, inv_rms, w, dtype)


def scale_norm(
    kind: model_config.Norm,
    x_f32: Float32[Array, "... K"],
    inv_rms: Float32[Array, "... 1"] | None,
    w: Float[Array, "1 K"] | None,
    dtype: jax.typing.DTypeLike,
) -> Float[Array, "... K"]:
  """norm() of x_f32, given its 1 / rms (inv_rms; unused for Norm.NONE)."""
  if kind == model_config.Norm.NONE:
    return x_f32.astype(dtype)
  assert inv_rms is not None
  normed = x_f32 * inv_rms
  match kind:
    case model_config.Norm.RMS:
      assert w is not None
      return (normed.astype(w.dtype) * w).astype(dtype)
    case model_config.Norm.RMS_ONE_PLUS_W:
      assert w is not None
      return (normed * (w.astype(jnp.float32) + 1.0)).astype(dtype)
    case model_config.Norm.RMS_NOSCALE:
      return normed.astype(dtype)
    case _:
      raise ValueError(kind)


def add_norm(
    kind: model_config.Norm,
    x: Float[Array, "T K"],
    residual: Float[Array, "T K"] | None,
    w: Float[Array, "1 K"] | None,
    eps: float,
) -> tuple[Float[Array, "T K"], Float[Array, "T K"] | None]:
  """Decoder input norm: (normed x, x + residual or None); the add is f32."""
  if residual is None:
    return norm(kind, x, w, eps), None
  h_f32 = x.astype(jnp.float32) + residual.astype(jnp.float32)
  return norm(kind, h_f32, w, eps, x.dtype), h_f32.astype(x.dtype)


# Weight layouts, prepared once at weight load time.


def _multiplier(
    kind: model_config.Norm, w: Float[Array, "1 K"]
) -> Float32[Array, "1 K"]:
  w = w.astype(jnp.float32)
  if kind == model_config.Norm.RMS_ONE_PLUS_W:
    w = w + 1.0
  return w


def packed_bf16_weight(
    kind: model_config.Norm, w: Float[Array, "1 K"], block: int
) -> UInt32[Array, "K/block 1 block"]:
  """Norm weight pre-chopped into quant blocks, as bf16 in both u32 halves.

  Both halves, so one stride-0 load (layout_util.broadcast_row) broadcasts it
  over a whole bf16 vreg.

  Args:
    kind: the norm; RMS_ONE_PLUS_W adds its 1 here.
    w: (1, K) as loaded.
    block: act_quant_block.

  Returns:
    (K / block, 1, block) u32.
  """
  bits = jax.lax.bitcast_convert_type(
      _multiplier(kind, w).astype(jnp.bfloat16), jnp.uint16
  )
  bits = bits.astype(jnp.uint32)
  return ((bits << 16) | bits).reshape(-1, 1, block)


def f32_weight(
    kind: model_config.Norm, w: Float[Array, "1 K"], block: int
) -> Float32[Array, "K/block 1 block"]:
  """Norm weight pre-chopped into quant blocks, in f32 (for broadcast_row)."""
  return _multiplier(kind, w).reshape(-1, 1, block)


# In-kernel.


def sum_sq(h: Float32[Array, "rows cols"]) -> Float32[Array, "rows 1"]:
  """Per-row sum of squares."""
  return jnp.sum(h * h, axis=1, keepdims=True)


def inv_rms_from_sum_sq(
    sum_squares: Float32[Array, "rows 1"], num_cols: int, eps: float
) -> Float32[Array, "rows 1"]:
  """1 / rms from the sum of squares over num_cols columns."""
  return jax.lax.rsqrt(sum_squares / num_cols + eps)


def apply_row_scale(
    acc: Float32[Array, "rows N"], row_scale: Float32[Array, "rows lanes"]
) -> Float32[Array, "rows N"]:
  """acc times the lane-broadcast per-row 1/rms."""
  return acc * layout_util.repeat_lanes(row_scale, acc.shape[1])
