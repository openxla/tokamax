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
import warnings

import immutabledict
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from tokamax._src.ops.experimental.routing_moe_op import base

Implementation = Literal["xla", "mosaic_tpu"]

_MESH_IMPLEMENTATIONS: Final[frozenset[str]] = frozenset(("mosaic_tpu",))

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


class RoutingMoeFromLogitsWarning(UserWarning):
  """Warning emitted when using the convenience routing_moe_from_logits API."""


_WARNED_ROUTING_MOE_FROM_LOGITS = False


def _reset_routing_moe_from_logits_warning() -> None:
  """Resets the warning flag for tests."""
  global _WARNED_ROUTING_MOE_FROM_LOGITS
  _WARNED_ROUTING_MOE_FROM_LOGITS = False


def _apply_scoring_fn(scoring_fn: str, gating_output: jax.Array) -> jax.Array:
  """Applies router scoring function to gating logits."""
  match scoring_fn:
    case "softmax":
      return jax.nn.softmax(gating_output, axis=-1)
    case "sigmoid":
      return jax.nn.sigmoid(gating_output)
    case _:
      raise NotImplementedError(
          f"routing_topk does not support scoring_fn={scoring_fn!r}."
      )


def routing_topk(
    gating_output: Float[Array, "T E"],
    topk: int,
    *,
    renormalize: bool,
    scoring_fn: str = "softmax",
) -> tuple[jax.Array, jax.Array]:
  """Computes top-k expert routing weights and indices from router logits.

  Args:
    gating_output: Router gating logits of shape `[tokens, experts]`.
    topk: Number of experts to choose per token.
    renormalize: Whether to normalize top-k weights to sum to 1.
    scoring_fn: Scoring function to apply to logits ("softmax" or "sigmoid").

  Returns:
    A tuple of `(topk_weights, topk_indices)`.
  """
  scores = _apply_scoring_fn(scoring_fn, gating_output)
  topk_weights, topk_indices = jax.lax.top_k(scores, k=topk)
  if renormalize:
    topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
  return topk_weights, topk_indices.astype(jnp.int32)


def _normalize_implementations(
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ),
) -> Sequence[Implementation | Callable[..., jax.Array]]:
  """Returns a non-empty sequence of implementations to try in order."""
  if implementation is None:
    return _DEFAULT_IMPLEMENTATIONS
  if isinstance(implementation, str):
    return (implementation,)
  if not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")
  return implementation


def _call_implementation(
    impl: Implementation | Callable[..., jax.Array],
    base_kwargs: dict[str, Any],
    mesh: jax.sharding.Mesh | None,
) -> jax.Array:
  """Resolves and invokes a single `routing_moe` implementation."""
  if isinstance(impl, str):
    if impl not in IMPLEMENTATIONS:
      raise ValueError(
          f"Unknown implementation: {impl}. You may need to add a dependency"
          " on the corresponding backend."
      )
    op_impl = IMPLEMENTATIONS[impl]
    uses_mesh = impl in _MESH_IMPLEMENTATIONS
  else:
    op_impl = impl
    uses_mesh = isinstance(op_impl, base.RoutingMoe)

  kwargs = (
      {**base_kwargs, "mesh": mesh}
      if (mesh is not None and uses_mesh)
      else base_kwargs
  )
  return op_impl(**kwargs)


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

  Routing inputs (topk_weights, topk_indices) can be computed from logits
  using `routing_topk`.

  Args:
    hidden_states: Input activations of shape `[tokens, hidden]`.
    topk_weights: Routing weights of selected experts `[tokens, topk]`.
    topk_indices: Expert indices of selected experts `[tokens, topk]`.
    w1: Expert weights for gate/up projection `[experts, hidden, 2 * inter]`.
    w2: Expert weights for down projection `[experts, inter, hidden]`.
    w1_scale: Optional weight scale for w1 `[experts, num_blocks, 1, 2*inter]`.
    w2_scale: Optional weight scale for w2 `[experts, num_blocks, 1, hidden]`.
    w1_bias: Optional bias for w1 `[experts, 1, 2 * inter]`.
    w2_bias: Optional bias for w2 `[experts, 1, hidden]`.
    mesh: Optional device mesh for distributed execution.
    activation: Activation function to apply between projections (e.g. "silu").
    tp_axis: Tensor parallelism axis name.
    ep_axis: Expert parallelism axis name.
    use_sp: Whether tokens are sharded over the reduction axis.
    reduce_output: Whether to reduce output across devices.
    implementation: Implementation name or fallback sequence to use.

  Returns:
    Output activations of shape `[tokens, hidden]` if `reduce_output=True`,
    else `[num_shards * tokens, hidden]` (unreduced partial sums).

  Raises:
    ValueError: If `implementation` is empty or names an unknown backend.
    NotImplementedError: If implementation cannot handle the arguments.
    ExceptionGroup: If all implementations in fallback sequence fail.
  """
  impls = _normalize_implementations(implementation)
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
  for impl in impls:
    try:
      return _call_implementation(impl, base_kwargs, mesh)
    except NotImplementedError as e:
      if len(impls) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)


def routing_moe_from_logits(
    hidden_states: Float[Array, "T H"],
    gating_output: Float[Array, "T E"],
    w1: Array,
    w2: Array,
    w1_scale: Array | None = None,
    w2_scale: Array | None = None,
    w1_bias: Array | None = None,
    w2_bias: Array | None = None,
    *,
    topk: int,
    renormalize: bool,
    mesh: jax.sharding.Mesh | None = None,
    activation: str = "silu",
    scoring_fn: str = "softmax",
    tp_axis: str | None = None,
    ep_axis: str | None = None,
    use_sp: bool = False,
    reduce_output: bool = True,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ) = None,
) -> jax.Array:
  """Routing MoE from router logits; prefer `routing_moe` with precomputed top-k.

  Prefer:
    topk_weights, topk_indices = routing_topk(
        gating_output, topk, renormalize=renormalize
    )
    out = routing_moe(hidden_states, topk_weights, topk_indices, w1, w2, ...)

  Args:
    hidden_states: Input activations of shape `[tokens, hidden]`.
    gating_output: Router gating logits of shape `[tokens, experts]`.
    w1: Expert weights for gate/up projection `[experts, hidden, 2 * inter]`.
    w2: Expert weights for down projection `[experts, inter, hidden]`.
    w1_scale: Optional weight scale for w1 `[experts, num_blocks, 1, 2*inter]`.
    w2_scale: Optional weight scale for w2 `[experts, num_blocks, 1, hidden]`.
    w1_bias: Optional bias for w1 `[experts, 1, 2 * inter]`.
    w2_bias: Optional bias for w2 `[experts, 1, hidden]`.
    topk: Number of experts to choose per token.
    renormalize: Whether to normalize top-k weights to sum to 1.
    mesh: Optional device mesh for distributed execution.
    activation: Activation function to apply between projections (e.g. "silu").
    scoring_fn: Scoring function to apply to logits ("softmax" or "sigmoid").
    tp_axis: Tensor parallelism axis name.
    ep_axis: Expert parallelism axis name.
    use_sp: Whether tokens are sharded over the reduction axis.
    reduce_output: Whether to reduce output across devices.
    implementation: Implementation name or fallback sequence to use.

  Returns:
    Output activations of shape `[tokens, hidden]` if `reduce_output=True`,
    else `[num_shards * tokens, hidden]` (unreduced partial sums).
  """
  global _WARNED_ROUTING_MOE_FROM_LOGITS
  if not _WARNED_ROUTING_MOE_FROM_LOGITS:
    _WARNED_ROUTING_MOE_FROM_LOGITS = True
    warnings.warn(
        "routing_moe_from_logits is provided for convenience; prefer"
        " computing routing with routing_topk and calling routing_moe"
        " directly. Silence this warning with"
        ' warnings.filterwarnings("ignore",'
        " category=RoutingMoeFromLogitsWarning).",
        category=RoutingMoeFromLogitsWarning,
        stacklevel=2,
    )

  topk_weights, topk_indices = routing_topk(
      gating_output=gating_output,
      topk=topk,
      renormalize=renormalize,
      scoring_fn=scoring_fn,
  )
  return routing_moe(
      hidden_states=hidden_states,
      topk_weights=topk_weights,
      topk_indices=topk_indices,
      w1=w1,
      w2=w2,
      w1_scale=w1_scale,
      w2_scale=w2_scale,
      w1_bias=w1_bias,
      w2_bias=w2_bias,
      mesh=mesh,
      activation=activation,
      tp_axis=tp_axis,
      ep_axis=ep_axis,
      use_sp=use_sp,
      reduce_output=reduce_output,
      implementation=implementation,
  )

