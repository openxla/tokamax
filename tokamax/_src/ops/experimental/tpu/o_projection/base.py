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
"""Base class for the DeepSeek-V4 fused reverse-RoPE `wo_a` projection.

See `reference` for the computation and layouts.
"""

from typing import Any, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Shaped  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.o_projection import reference

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray | np.ndarray

# Heads per `wo_a` group. DeepSeek-V4 uses 8, and the kernel relies on it.
HEADS_PER_GROUP = 8
# Width of a lane block; `head_dim` is a whole number of them.
LANE = 128


class OProjection[C](op.Op[Any, jax.Array, None, C, Any]):
  """Tokamax operator for the DeepSeek-V4 fused reverse-RoPE `wo_a` projection.

  Applies the RoPE (by default the inverse rotation) to the trailing
  `rotary_dim` channels of each head of the attention output, then projects
  each group of 8 heads with its block of the fp8 `wo_a`.
  """

  @jaxtyping.jaxtyped
  def bind(
      self,
      x: Shaped[Array | AbstractArray, "T *dims"],
      positions: Shaped[Array | AbstractArray, "*P"],
      cos_sin_cache: Float[Array | AbstractArray, "max_pos rotary_dim"],
      wo_a: Shaped[Array | AbstractArray, "D GR"],
      wo_a_scale: Shaped[Array | AbstractArray, "*S"],
      *,
      inverse: bool = True,
      quantize_activations: bool = True,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Validates the arguments.

    Args:
      x: `(T, G * 8, head_dim)` bf16 attention output; `head_dim` is a multiple
        of 128.
      positions: `(T,)` int32 RoPE position of each token, in `[0, max_pos)`.
        The Pallas kernels do not bounds check them.
      cos_sin_cache: `(max_pos, rotary_dim)` float32 packed `[cos | sin]` rows;
        `rotary_dim` is even and at most 128.
      wo_a: `(8 * head_dim, G * R)` float8_e4m3fn projection weights.
      wo_a_scale: `(G * R,)` float32 per-column weight scales.
      inverse: Whether to negate `sin`, i.e. apply the transposed rotation.
      quantize_activations: Whether to quantize the activations to fp8 per token
        and group before the projection.
      return_residuals: Unused; the op has no residuals.

    Returns:
      The bound arguments.

    Raises:
      ValueError: If the arguments are invalid.
    """
    if x.ndim != 3 or x.dtype != jnp.bfloat16:
      raise ValueError(
          f"x must be a rank 3 bf16 array, got {x.dtype} {x.shape}."
      )
    num_tokens, num_heads, head_dim = x.shape
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
    if wo_a.dtype != jnp.float8_e4m3fn:
      raise ValueError(f"wo_a must be float8_e4m3fn, got {wo_a.dtype}.")
    reduction, out_features = wo_a.shape
    if reduction != HEADS_PER_GROUP * head_dim:
      raise ValueError(
          "heads_per_group (wo_a.shape[0] / head_dim) must be"
          f" {HEADS_PER_GROUP}, got wo_a {wo_a.shape} and head_dim {head_dim}."
      )
    if num_heads % HEADS_PER_GROUP:
      raise ValueError(
          f"num_heads must be a multiple of {HEADS_PER_GROUP}, got {num_heads}."
      )
    num_groups = num_heads // HEADS_PER_GROUP
    if out_features % num_groups:
      raise ValueError(
          f"wo_a.shape[1] ({out_features}) must be a multiple of the number of"
          f" groups ({num_groups})."
      )
    if wo_a_scale.shape != (out_features,) or wo_a_scale.dtype != jnp.float32:
      raise ValueError(
          f"wo_a_scale must be float32 of shape ({out_features},), got"
          f" {wo_a_scale.dtype} {wo_a_scale.shape}."
      )
    return super().bind(
        x=x,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        wo_a=wo_a,
        wo_a_scale=wo_a_scale,
        inverse=inverse,
        quantize_activations=quantize_activations,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Float[Array, "T H head_dim"],
      positions: Shaped[Array, "T"],
      cos_sin_cache: Float[Array, "max_pos rotary_dim"],
      wo_a: Shaped[Array, "D GR"],
      wo_a_scale: Float[Array, "GR"],
      *,
      inverse: bool = True,
      quantize_activations: bool = True,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[jax.Array, None]:
    """Applies the RoPE to `x`, then projects it with `wo_a`.

    Runs `reference.o_projection`.

    Args:
      x: `(T, G * 8, head_dim)` bf16 attention output.
      positions: `(T,)` int32 RoPE position of each token.
      cos_sin_cache: `(max_pos, rotary_dim)` float32 packed `[cos | sin]` rows.
      wo_a: `(8 * head_dim, G * R)` float8_e4m3fn projection weights.
      wo_a_scale: `(G * R,)` float32 per-column weight scales.
      inverse: Whether to negate `sin`, i.e. apply the transposed rotation.
      quantize_activations: Whether to quantize the activations to fp8 per token
        and group before the projection.
      return_residuals: Unused; the op has no residuals.
      config: Unused.

    Returns:
      `(out, None)`, where `out` is the `(T, G * R)` bf16 projection.
    """
    del config  # Unused.
    return (
        reference.o_projection(
            x,
            positions,
            cos_sin_cache,
            wo_a,
            wo_a_scale,
            inverse=inverse,
            quantize_activations=quantize_activations,
        ),
        None,
    )
