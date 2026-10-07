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
"""Base class for the DeepSeek-V4 RoPE.

See `reference` for the rotation and the three variants (`mode`s).
"""

from typing import Any, Literal, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.rope import reference

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray | np.ndarray

type Mode = Literal["rope", "qnorm_rope", "rope_quant"]
MODES: tuple[str, ...] = ("rope", "qnorm_rope", "rope_quant")

# Width of a lane block; the kernels work on whole lane blocks of the head.
LANE = 128

type RopeOutput = jax.Array | tuple[jax.Array, jax.Array]


class Rope[C](op.Op[Any, RopeOutput, None, C, Any]):
  """Tokamax operator for the DeepSeek-V4 RoPE.

  One op covers the three DeepSeek-V4 RoPE variants, selected by `mode`:

  *   `"rope"`: rotates the trailing `rotary_dim` channels of each head and
      returns an array of the shape and dtype of `x`.
  *   `"qnorm_rope"`: applies a per-head RMSNorm (no weight, epsilon `eps`)
      over the whole head first. `x` must be rank 3.
  *   `"rope_quant"`: quantizes the float32 rotation per row to `quant_dtype`
      and returns `(q, scales)`. `head_dim` must be 128.

  This XLA implementation does not modify `x`. The Pallas implementation
  donates `x` in the `"rope"` and `"qnorm_rope"` modes (see
  `pallas_mosaic_tpu.PallasTpuRope`).
  """

  @jaxtyping.jaxtyped
  def bind(
      self,
      x: Float[Array | AbstractArray, "N *dims"],
      positions: Shaped[Array | AbstractArray, "*P"],
      cos_sin_cache: Float[Array | AbstractArray, "max_pos rotary_dim"],
      *,
      mode: Mode = "rope",
      inverse: bool = False,
      eps: float = 1e-6,
      quant_dtype: jax.typing.DTypeLike | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Validates and canonicalizes the arguments.

    Args:
      x: `(num_tokens, head_dim)` or `(num_tokens, num_heads, head_dim)`
        floating-point values. `head_dim` is a multiple of 128.
      positions: `(num_tokens,)` int32 RoPE position of each token, in `[0,
        max_pos)`. The Pallas kernels do not bounds check them.
      cos_sin_cache: `(max_pos, rotary_dim)` float32 packed `[cos | sin]` rows;
        `rotary_dim` is even and at most 128.
      mode: The variant, one of `"rope"`, `"qnorm_rope"` and `"rope_quant"`.
      inverse: Whether to negate `sin`, i.e. apply the transposed rotation.
      eps: The RMSNorm epsilon (`"qnorm_rope"` only).
      quant_dtype: The floating-point quantized dtype (`"rope_quant"` only).
        Defaults to `float8_e4m3fn`. Canonicalized to `None` in other modes.
      return_residuals: Unused; the op has no residuals.

    Returns:
      The bound arguments.

    Raises:
      ValueError: If the arguments are invalid for `mode`.
    """
    if mode not in MODES:
      raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
    if x.ndim not in (2, 3):
      raise ValueError(f"x must be rank 2 or 3, got shape {x.shape}.")
    num_tokens, head_dim = x.shape[0], x.shape[-1]
    if head_dim <= 0 or head_dim % LANE:
      raise ValueError(
          f"head_dim must be a positive multiple of {LANE}, got {head_dim}."
      )
    if positions.shape != (num_tokens,) or positions.dtype != jnp.int32:
      raise ValueError(
          f"positions must be int32 of shape ({num_tokens},), got"
          f" {positions.dtype} {positions.shape}."
      )
    if cos_sin_cache.dtype != jnp.float32:
      raise ValueError(
          f"cos_sin_cache must be float32, got {cos_sin_cache.dtype}."
      )
    rotary_dim = cos_sin_cache.shape[1]
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > LANE:
      raise ValueError(
          f"rotary_dim must be even and in (0, {LANE}], got {rotary_dim}."
      )
    if mode == "qnorm_rope" and x.ndim != 3:
      raise ValueError(f"qnorm_rope requires a rank 3 x, got shape {x.shape}.")
    if mode == "rope_quant":
      if head_dim != LANE:
        raise ValueError(
            f"rope_quant requires head_dim == {LANE}, got {head_dim}."
        )
      quant_dtype = jnp.dtype(
          jnp.float8_e4m3fn if quant_dtype is None else quant_dtype
      )
      if not jnp.issubdtype(quant_dtype, jnp.floating):
        raise ValueError(
            f"quant_dtype must be a floating dtype, got {quant_dtype}."
        )
    else:
      quant_dtype = None
    return super().bind(
        x=x,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        mode=mode,
        inverse=inverse,
        eps=eps,
        quant_dtype=quant_dtype,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Float[Array, "N *dims"],
      positions: Int[Array, "N"],
      cos_sin_cache: Float[Array, "max_pos rotary_dim"],
      *,
      mode: Mode = "rope",
      inverse: bool = False,
      eps: float = 1e-6,
      quant_dtype: jax.typing.DTypeLike | None = None,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[RopeOutput, None]:
    """Applies the DeepSeek-V4 RoPE variant selected by `mode`.

    Runs the matching function of `reference`.

    Args:
      x: `(num_tokens, head_dim)` or `(num_tokens, num_heads, head_dim)`.
      positions: `(num_tokens,)` int32 RoPE position of each token.
      cos_sin_cache: `(max_pos, rotary_dim)` float32 packed `[cos | sin]` rows.
      mode: The variant, one of `"rope"`, `"qnorm_rope"` and `"rope_quant"`.
      inverse: Whether to negate `sin`, i.e. apply the transposed rotation.
      eps: The RMSNorm epsilon (`"qnorm_rope"` only).
      quant_dtype: The quantized dtype (`"rope_quant"` only), else `None`.
      return_residuals: Unused; the op has no residuals.
      config: Unused.

    Returns:
      `(out, None)`. `out` is the rotated `x` (same shape and dtype) for
      `"rope"` and `"qnorm_rope"`, and `(q, scales)` for `"rope_quant"`, with
      `q` of `x.shape` in `quant_dtype` and float32 `scales` of `x.shape[:-1]`.
    """
    del config  # Unused.
    match mode:
      case "rope":
        out = reference.rope(x, positions, cos_sin_cache, inverse=inverse)
      case "qnorm_rope":
        out = reference.qnorm_rope(
            x, positions, cos_sin_cache, eps, inverse=inverse
        )
      case "rope_quant":
        assert quant_dtype is not None  # Canonicalized by `bind`.
        out = reference.rope_quant(
            x,
            positions,
            cos_sin_cache,
            inverse=inverse,
            quant_dtype=quant_dtype,
        )
      case _:
        raise ValueError(f"Unknown mode: {mode!r}.")
    return out, None
