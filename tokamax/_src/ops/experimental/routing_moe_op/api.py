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
from jaxtyping import Int
from tokamax._src.ops.experimental.routing_moe_op import base

Implementation = Literal["xla", "mosaic_tpu"]

_IMPLEMENTATIONS: dict[str, base.RoutingMoe] = dict(xla=base.RoutingMoe())

try:
  from tokamax._src.ops.experimental.routing_moe_op import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
except ImportError:
  pass

_DEFAULT_IMPLEMENTATIONS: Final[tuple[Implementation, ...]] = (
    ("mosaic_tpu", "xla") if "mosaic_tpu" in _IMPLEMENTATIONS else ("xla",)
)

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, base.RoutingMoe]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS

_MESH_IMPLEMENTATIONS: Final[frozenset[str]] = frozenset({"mosaic_tpu"})


def _resolve_implementation(
    impl: Implementation | Callable[..., jax.Array],
) -> tuple[Callable[..., jax.Array], bool]:
  if isinstance(impl, str):
    if impl not in IMPLEMENTATIONS:
      raise ValueError(
          f"Unknown implementation: {impl}. You may need to add a dependency"
          " on the corresponding backend."
      )
    return IMPLEMENTATIONS[impl], impl in _MESH_IMPLEMENTATIONS
  return impl, isinstance(impl, base.RoutingMoe)


def routing_moe(
    hidden_states: Float[Array, "T H"],
    topk_weights: Float[Array, "T K"],
    topk_indices: Int[Array, "T K"],
    w1: Array,
    w2: Array,
    w1_scale: Array | None = None,
    w2_scale: Array | None = None,
    w1_bias: Array | None = None,
    w2_bias: Array | None = None,
    mesh: jax.sharding.Mesh | None = None,
    activation: str = "silu",
    tp_axis: str | None = None,
    ep_axis: str | None = None,
    use_sp: bool = False,
    reduce_output: bool = True,
    *,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ) = None,
) -> jax.Array:
  """Routes input activations through an MoE layer using top-k gating.

  When `use_sp` is True, inputs and outputs are sharded along sequence
  parallelism; `reduce_output` controls whether cross-device reduction runs.

  Args:
    hidden_states: Input activations `[tokens, hidden]`.
    topk_weights: Gating weights `[tokens, topk]`.
    topk_indices: Routed expert indices `[tokens, topk]`.
    w1: Expert weights for gate/up projection `[experts, hidden, 2 * inter]`.
    w2: Expert weights for down projection `[experts, inter, hidden]`.
    w1_scale: Optional weight scale for w1.
    w2_scale: Optional weight scale for w2.
    w1_bias: Optional bias for w1.
    w2_bias: Optional bias for w2.
    mesh: Device mesh for expert-parallel execution.
    activation: Activation function between projections (e.g. "silu").
    tp_axis: Tensor parallelism axis name.
    ep_axis: Expert parallelism axis name.
    use_sp: Whether tokens are sharded over the reduction axis.
    reduce_output: Whether to reduce output across devices.
    implementation: Backend implementation name or sequence of fallbacks.

  Returns:
    Output activations of shape `[tokens, hidden]`.

  Raises:
    ValueError: If `implementation` is empty or unknown.
    NotImplementedError: If an implementation cannot handle the arguments.
    ExceptionGroup: If all implementations in the fallback sequence fail.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  base_kwargs: dict[str, Any] = dict(
      hidden_states=hidden_states,
      topk_weights=topk_weights,
      topk_indices=topk_indices,
      w1=w1,
      w2=w2,
      w1_scale=w1_scale,
      w2_scale=w2_scale,
      w1_bias=w1_bias,
      w2_bias=w2_bias,
      activation=activation,
      tp_axis=tp_axis,
      ep_axis=ep_axis,
      use_sp=use_sp,
      reduce_output=reduce_output,
  )

  errors = []
  for impl in implementation:
    op_impl, accepts_mesh = _resolve_implementation(impl)
    kwargs = (
        {**base_kwargs, "mesh": mesh}
        if mesh is not None and accepts_mesh
        else base_kwargs
    )
    try:
      return op_impl(**kwargs)
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
