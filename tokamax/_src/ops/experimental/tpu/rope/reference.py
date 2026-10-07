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
"""Pure JAX reference implementation of the DeepSeek-V4 RoPE.

DeepSeek-V4 rotates the *trailing* `rotary_dim` channels of each head with the
GPT-J (interleaved) rotation: channel pair `(2k, 2k + 1)` is rotated by
frequency `k`. The `cos_sin_cache` holds one `[cos | sin]` row per position,
`rotary_dim // 2` columns each.

Three variants are provided, matching the Pallas kernels:

*   `rope`: the rotation alone.
*   `qnorm_rope`: a per-head RMSNorm (no weight) over the whole head, then the
    rotation.
*   `rope_quant`: the rotation in float32, then per-row dynamic quantization
    (one scale per row over the channel axis).
"""

import jax
import jax.numpy as jnp


def rope(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    inverse: bool = False,
) -> jax.Array:
  """Reference implementation of `DeepseekV4ScalingRotaryEmbedding.forward_native`.

  GPT-J interleaved rotation over the *trailing* `rotary_dim` channels.

  Args:
    x: `[num_tokens, head_dim]` or `[num_tokens, num_heads, head_dim]`.
    positions: `[num_tokens]` int RoPE position of each token.
    cos_sin_cache: `[max_position, rotary_dim]` packed `[cos | sin]` rows.
    inverse: Whether to negate `sin`, i.e. apply the transposed rotation.

  Returns:
    `x` with the trailing `rotary_dim` channels rotated, in `x.dtype`.
  """
  rotary_dim = cos_sin_cache.shape[1]
  orig_dtype = x.dtype
  xf = jnp.asarray(x).astype(jnp.float32)

  cos, sin = jnp.split(
      jnp.asarray(cos_sin_cache)[positions].astype(jnp.float32), 2, axis=-1
  )
  cos = jnp.repeat(cos, 2, axis=-1)  # repeat_interleave
  sin = jnp.repeat(sin, 2, axis=-1)
  if inverse:
    sin = -sin
  if xf.ndim == 3:  # [num_tokens, num_heads, head_dim]: broadcast over heads
    cos = cos[:, None, :]
    sin = sin[:, None, :]

  x_pass, x_rot = xf[..., :-rotary_dim], xf[..., -rotary_dim:]
  # rotate_gptj: [-x1, x0, -x3, x2, ...]
  rotated = jnp.stack([-x_rot[..., 1::2], x_rot[..., 0::2]], axis=-1).reshape(
      x_rot.shape
  )
  out_rot = x_rot * cos + rotated * sin
  return jnp.concatenate([x_pass, out_rot], axis=-1).astype(orig_dtype)


def quantize(
    y: jax.Array, quant_dtype: jax.typing.DTypeLike
) -> tuple[jax.Array, jax.Array]:
  """Reference implementation of `quantization.quantize_tensor(..., axis=-1)`.

  Args:
    y: The values to quantize.
    quant_dtype: The floating-point quantized dtype.

  Returns:
    `(q, scale)`: `q` is `y / scale` in `quant_dtype`, and `scale` is the
    float32 per-row scale `max(|y|) / max(quant_dtype)`, `y.shape[:-1]`.
  """
  dtype_max = float(jnp.finfo(quant_dtype).max)
  yf = jnp.asarray(y).astype(jnp.float32)
  abs_max = jnp.max(jnp.abs(yf), axis=-1, keepdims=True)
  scale = abs_max / dtype_max
  # quantize_tensor leaves an all-zero row at NaN; the kernel returns zeros.
  scale_inv = jnp.where(abs_max == 0.0, 0.0, 1.0 / scale)
  return (yf * scale_inv).astype(quant_dtype), jnp.squeeze(scale, -1)


def qnorm_rope(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    eps: float,
    inverse: bool = False,
) -> jax.Array:
  """Reference implementation of `DeepseekV4Attention.qnorm_rope`.

  Per-head RMSNorm (no weight) over the whole `head_dim`, in float32, then the
  same RoPE `rope` applies.

  Args:
    x: `[num_tokens, num_heads, head_dim]`.
    positions: `[num_tokens]` int RoPE position of each token.
    cos_sin_cache: `[max_position, rotary_dim]` packed `[cos | sin]` rows.
    eps: The RMSNorm epsilon.
    inverse: Whether to negate `sin`, i.e. apply the transposed rotation.

  Returns:
    The normalized and rotated `x`, in `x.dtype`.
  """
  xf = jnp.asarray(x).astype(jnp.float32)
  rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
  return rope(xf * rms, positions, cos_sin_cache, inverse=inverse).astype(
      x.dtype
  )


def rope_quant(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    inverse: bool = False,
    quant_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array]:
  """RoPE in float32 followed by per-row quantization.

  The kernel quantizes the float32 rotation directly -- it never rounds to
  `x.dtype` in between -- so neither does the reference.

  Args:
    x: `[num_tokens, head_dim]` or `[num_tokens, num_heads, head_dim]`.
    positions: `[num_tokens]` int RoPE position of each token.
    cos_sin_cache: `[max_position, rotary_dim]` packed `[cos | sin]` rows.
    inverse: Whether to negate `sin`, i.e. apply the transposed rotation.
    quant_dtype: The floating-point quantized dtype.

  Returns:
    `(q, scales)`, see `quantize`.
  """
  return quantize(
      rope(x.astype(jnp.float32), positions, cos_sin_cache, inverse=inverse),
      quant_dtype,
  )
