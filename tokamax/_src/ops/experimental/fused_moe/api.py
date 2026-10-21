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
"""Public functional and operator API for Fused MoE."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
from jaxtyping import Array, Float
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.fused_moe import base

Implementation = Literal["xla", "mosaic_tpu"]

_IMPLEMENTATIONS: dict[str, base.FusedMoe] = dict(xla=base.FusedMoe())
_DEFAULT_IMPLEMENTATIONS: Final[tuple[Implementation, ...]] = ("xla",)

try:
  from tokamax._src.ops.experimental.fused_moe import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasMosaicTpuFusedMoe()
  _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, base.FusedMoe]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def fused_moe(
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
    mesh: jax.sharding.Mesh | None = None,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ) = None,
) -> jax.Array:
  """Fused Mixture of Experts (MoE) layer.

  Computes the forward pass of an MoE layer combining gate/up projection,
  activation, down projection, and weighted accumulation.

  Args:
    x: Input activations of shape `[tokens, hidden]`.
    w1: Expert weights for gate/up projection of shape `[experts, hidden, 2 *
      inter]`.
    w2: Expert weights for down projection of shape `[experts, inter, hidden]`.
    gating: Router gating logits of shape `[tokens, experts]`.
    w1_scale: Optional weight scale for w1 (per-channel or per-block).
    w2_scale: Optional weight scale for w2 (per-channel or per-block).
    w1_bias: Optional bias for w1 of shape `[experts, 1, 2 * inter]`.
    w2_bias: Optional bias for w2 of shape `[experts, 1, hidden]`.
    topk: Number of experts to select per token. Defaults to 2.
    renormalize: Whether to renormalize top-k gating weights to sum to 1.
    act_fn: Activation function to apply between projections (e.g. "silu").
    mesh: Optional device mesh for expert-parallel sharded execution (used by
      Mosaic TPU).
    implementation: Which implementation to use ("xla", "mosaic_tpu", or a
      sequence of fallbacks).

  Returns:
    Output activations of shape `[tokens, hidden]`.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  base_kwargs: dict[str, Any] = dict(
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

  errors = []
  for impl in implementation:
    if isinstance(impl, str):
      impl_key = impl
      if impl_key not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation: {impl}. You may need to add a dependency"
            " on the corresponding backend."
        )
      op_impl = IMPLEMENTATIONS[impl_key]
    else:
      op_impl = impl
      impl_key = None

    kwargs = base_kwargs
    if mesh is not None and impl_key == "mosaic_tpu":
      kwargs = {**base_kwargs, "mesh": mesh}

    try:
      return op_impl(**kwargs)
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
