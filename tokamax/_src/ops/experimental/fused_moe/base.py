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
"""Fused MoE base operator class."""

from typing import Any, TypeVar
import jax
from jaxtyping import Array, Float
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.fused_moe import reference
from typing_extensions import override

_Config = TypeVar("_Config")


class FusedMoe(op.Op[Any, jax.Array, None, _Config, Any]):
  """Base operator interface for Tokamax Fused MoE."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      x: Float[Array, "T H"],
      w1: Array,
      w2: Array,
      gating: Float[Array, "T E"],
      w1_scale: Array | None = None,
      w2_scale: Array | None = None,
      w1_bias: Array | None = None,
      w2_bias: Array | None = None,
      *,
      topk: int = 2,
      renormalize: bool = True,
      act_fn: str = "silu",
  ) -> op.BoundArguments:
    """Binds arguments for Fused MoE operator."""
    return super().bind(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Float[Array, "T H"],
      w1: Array,
      w2: Array,
      gating: Float[Array, "T E"],
      w1_scale: Array | None = None,
      w2_scale: Array | None = None,
      w1_bias: Array | None = None,
      w2_bias: Array | None = None,
      *,
      topk: int = 2,
      renormalize: bool = True,
      act_fn: str = "silu",
      config: _Config | None = None,
  ) -> tuple[jax.Array, None]:
    del config
    return reference.fused_moe_reference(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
    ), None
