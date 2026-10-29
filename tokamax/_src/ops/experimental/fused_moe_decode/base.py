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
"""Base operator class for Fused MoE Decode."""

from typing import Any, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.fused_moe import reference

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray | np.ndarray

_Q_BLOCK = 512
_LANE = 128


def _reshape(
    x: Array | AbstractArray, new_shape: tuple[int, ...]
) -> Array | AbstractArray:
  """Reshapes a concrete or abstract array to `new_shape`."""
  if x.shape == new_shape:
    return x
  if isinstance(x, jax.ShapeDtypeStruct):
    return jax.ShapeDtypeStruct(new_shape, x.dtype)
  if isinstance(x, jax.core.ShapedArray):
    return jax.core.ShapedArray(new_shape, x.dtype)
  return x.reshape(new_shape)


class FusedMoeDecode[C](op.Op[Any, jax.Array, None, C, Any]):
  """Base operator interface for Tokamax Fused MoE Decode."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      rank: Int[Array | AbstractArray, "*#r"],
      hidden: Float[Array | AbstractArray, "tokens H"],
      w1: Shaped[Array | AbstractArray, "E H two_i"],
      w2: Shaped[Array | AbstractArray, "E i H"],
      w1_scale: Float[Array | AbstractArray, "E *#s_w1 two_i"],
      w2_scale: Float[Array | AbstractArray, "E *#s_w2 H"],
      shared_w1: Shaped[Array | AbstractArray, "H two_is"],
      shared_w1_scale: Float[Array | AbstractArray, "*#s_sw1 two_is"],
      shared_w2: Shaped[Array | AbstractArray, "i_s H"],
      shared_w2_scale: Float[Array | AbstractArray, "*#s_sw2 H"],
      shared_gate_weight: Float[Array | AbstractArray, "H *#s_sg"] | None,
      topk_weights: Float[Array | AbstractArray, "tokens k"],
      topk_ids: Int[Array | AbstractArray, "tokens k"],
      *,
      n_ranks: int = 1,
      expert_start: Int[Array | AbstractArray, "*#es"] | int | None = None,
      scatter_rows: bool = False,
      routed_scale: float = 1.0,
      axis_name: str | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Validates and binds arguments for the Fused MoE Decode operator."""
    tokens, h = hidden.shape
    if h <= 0 or h % _Q_BLOCK:
      raise ValueError(
          f"hidden size {h} must be a positive multiple of {_Q_BLOCK}."
      )
    if n_ranks <= 0 or h % n_ranks or (h // n_ranks) % _LANE:
      raise ValueError(
          f"h // n_ranks must be a multiple of {_LANE}, got h={h} and"
          f" n_ranks={n_ranks}."
      )
    if scatter_rows and tokens % n_ranks:
      raise ValueError(
          f"scatter_rows needs tokens % n_ranks == 0; got tokens={tokens},"
          f" n_ranks={n_ranks}."
      )
    experts, i, h_w2 = w2.shape
    if h_w2 != h or w1.shape != (experts, h, 2 * i):
      raise ValueError(
          f"Incompatible w1 {w1.shape} and w2 {w2.shape} shapes for h={h}."
      )
    i_s, h_sw2 = shared_w2.shape
    if h_sw2 != h or shared_w1.shape != (h, 2 * i_s):
      raise ValueError(
          f"Incompatible shared_w1 {shared_w1.shape} and shared_w2"
          f" {shared_w2.shape} shapes for h={h}."
      )
    if i <= 0 or i % _LANE or i_s <= 0 or i_s % _LANE:
      raise ValueError(
          f"Intermediate sizes i={i} and i_s={i_s} must be positive multiples"
          f" of {_LANE}."
      )
    if w1_scale.shape not in (
        (experts, 2 * i),
        (experts, 1, 2 * i),
        (experts, 1, 1, 2 * i),
    ):
      raise ValueError(
          f"Incompatible w1_scale shape {w1_scale.shape} for w1 {w1.shape}."
      )
    w1_scale = _reshape(w1_scale, (experts, 1, 1, 2 * i))
    if w2_scale.shape not in (
        (experts, h),
        (experts, 1, h),
        (experts, 1, 1, h),
    ):
      raise ValueError(
          f"Incompatible w2_scale shape {w2_scale.shape} for w2 {w2.shape}."
      )
    w2_scale = _reshape(w2_scale, (experts, 1, 1, h))
    if shared_w1_scale.shape not in ((2 * i_s,), (1, 2 * i_s)):
      raise ValueError(
          f"Incompatible shared_w1_scale shape {shared_w1_scale.shape} for"
          f" shared_w1 {shared_w1.shape}."
      )
    shared_w1_scale = _reshape(shared_w1_scale, (1, 2 * i_s))
    if shared_w2_scale.shape not in ((h,), (1, h)):
      raise ValueError(
          f"Incompatible shared_w2_scale shape {shared_w2_scale.shape} for"
          f" shared_w2 {shared_w2.shape}."
      )
    shared_w2_scale = _reshape(shared_w2_scale, (1, h))
    if shared_gate_weight is not None:
      if shared_gate_weight.shape not in ((h,), (h, 1)):
        raise ValueError(
            "Incompatible shared_gate_weight shape"
            f" {shared_gate_weight.shape} for h={h}."
        )
      shared_gate_weight = _reshape(shared_gate_weight, (h, 1))
    if isinstance(expert_start, int):
      expert_start = jnp.asarray(expert_start, dtype=jnp.int32)
    return super().bind(
        rank=rank,
        hidden=hidden,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        shared_w1=shared_w1,
        shared_w1_scale=shared_w1_scale,
        shared_w2=shared_w2,
        shared_w2_scale=shared_w2_scale,
        shared_gate_weight=shared_gate_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        n_ranks=n_ranks,
        expert_start=expert_start,
        scatter_rows=scatter_rows,
        routed_scale=routed_scale,
        axis_name=axis_name,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      rank: Int[Array, "*#r"],
      hidden: Float[Array, "tokens H"],
      w1: Shaped[Array, "E H two_i"],
      w2: Shaped[Array, "E i H"],
      w1_scale: Float[Array, "E 1 1 two_i"],
      w2_scale: Float[Array, "E 1 1 H"],
      shared_w1: Shaped[Array, "H two_is"],
      shared_w1_scale: Float[Array, "1 two_is"],
      shared_w2: Shaped[Array, "i_s H"],
      shared_w2_scale: Float[Array, "1 H"],
      shared_gate_weight: Float[Array, "H 1"] | None,
      topk_weights: Float[Array, "tokens k"],
      topk_ids: Int[Array, "tokens k"],
      *,
      n_ranks: int = 1,
      expert_start: Int[Array, "*#es"] | None = None,
      scatter_rows: bool = False,
      routed_scale: float = 1.0,
      axis_name: str | None = None,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[jax.Array, None]:
    del config, return_residuals
    return (
        reference.fused_moe_reference(
            x=hidden,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_w1=shared_w1,
            shared_w1_scale=shared_w1_scale,
            shared_w2=shared_w2,
            shared_w2_scale=shared_w2_scale,
            shared_gate_weight=shared_gate_weight,
            rank=rank,
            n_ranks=n_ranks,
            expert_start=expert_start,
            scatter_rows=scatter_rows,
            routed_scale=routed_scale,
            axis_name=axis_name,
        ),
        None,
    )
