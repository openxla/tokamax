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
"""Public functional and operator API for Routing MoE."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
from jaxtyping import Array
from jaxtyping import Float
from tokamax._src.ops.experimental.routing_moe import base

Implementation = Literal["xla", "mosaic_tpu"]

# Implementations that run expert-parallel over a mesh, and so take `mesh`.
_MESH_IMPLEMENTATIONS: Final[frozenset[str]] = frozenset(("mosaic_tpu",))

_IMPLEMENTATIONS: dict[str, base.RoutingMoe] = dict(xla=base.RoutingMoe())

try:
  from tokamax._src.ops.experimental.routing_moe import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
  )
except ImportError:
  pass

_DEFAULT_IMPLEMENTATIONS: Final[tuple[Implementation, ...]] = (
    ("mosaic_tpu", "xla") if "mosaic_tpu" in _IMPLEMENTATIONS else ("xla",)
)

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, base.RoutingMoe]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def routing_moe(
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
    scoring_fn: str = "softmax",
    mesh: jax.sharding.Mesh | None = None,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ) = None,
) -> jax.Array:
  """Routing Mixture of Experts (MoE) layer.

  Computes the forward pass of an MoE layer combining router scoring, top-k
  selection, pipelined gather/scatter grouped matrix multiplication, activation,
  down projection, and weighted accumulation.

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
    scoring_fn: Router scoring function ("softmax" or "sigmoid").
    mesh: Optional device mesh for expert-parallel sharded execution (used by
      Mosaic TPU).
    implementation: Which implementation to use ("xla", "mosaic_tpu", or a
      sequence of fallbacks).

  Returns:
    Output activations of shape `[tokens, hidden]`.

  Raises:
    ValueError: If `implementation` is empty or names an unknown backend.
    NotImplementedError: If the only implementation requested can't handle
      the arguments.
    ExceptionGroup: If every implementation in a sequence of fallbacks raises
      `NotImplementedError`.
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
      scoring_fn=scoring_fn,
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
    # Op instances passed in directly may be mesh-based, and every routing MoE
    # op accepts `mesh`, so they get it too. Other callables don't.
    if mesh is not None and (
        impl_key in _MESH_IMPLEMENTATIONS
        or (impl_key is None and isinstance(op_impl, base.RoutingMoe))
    ):
      kwargs = {**base_kwargs, "mesh": mesh}

    try:
      return op_impl(**kwargs)
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
