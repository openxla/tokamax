# Copyright 2026 Google LLC. All Rights Reserved.
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

"""Raven Gated Slot Attention (GSA) public API."""

# Dimensions.
#   B      batch size
#   T      sequence length
#   H      number of attention heads
#   K      key/query head dimension
#   M      number of memory slots
#   V      value head dimension

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, TypeAlias
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.raven import base

Implementation: TypeAlias = Literal["mosaic_tpu", "xla"]

IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {
    "xla": base.RavenGSA(),
}
_DEFAULT_IMPLEMENTATION: tuple[str, ...] = ("xla",)

try:
  from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top

  IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuRavenGSA()
  _DEFAULT_IMPLEMENTATION = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass

IMPLEMENTATIONS: Final[dict[str, Callable[..., Any]]] = IMPLEMENTATIONS  # type: ignore[misc]


@jaxtyping.jaxtyped
def raven_gsa(
    q: Float[Array, "B T H K"],
    k: Float[Array, "B T H K"],
    s: Float[Array, "B T H M"],
    g: Float[Array, "B T H M"],
    v: Float[Array, "B T H V"] | None = None,
    *,
    chunk_size: int = 64,
    scale: float | None = None,
    initial_state: base.State = None,
    output_final_state: bool = False,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., Any]]
        | None
    ) = None,
) -> base.Output:
  """Executes the Raven Gated Slot Attention (GSA) operator.

  Args:
    q: Query tensor of shape ``[B, T, H, K]``.
    k: Key tensor of shape ``[B, T, H, K]``.
    s: Slot write weight tensor of shape ``[B, T, H, M]``.
    g: Log-space slot retention gate tensor of shape ``[B, T, H, M]``.
    v: Optional value tensor of shape ``[B, T, H, V]``. When omitted, only
      Stage 1 slot logits ``[B, T, H, M]`` are computed. When provided, runs
      the full two-stage GSA recurrence (Stage 1 -> slot softmax -> Stage 2).
    chunk_size: Intra-chunk tile size ``tt`` dividing ``T``.
    scale: Optional multiplicative scaling factor applied to ``q``.
    initial_state: Optional initial recurrent state ``[B, H, K, M]`` (for
      Stage 1) or ``(s1_0, s2_0)`` (for full two-stage GSA).
    output_final_state: Whether to return the final recurrent state.
    implementation: Backend implementation name (``"mosaic_tpu"`` or ``"xla"``)
      or sequence of fallback implementations.

  Returns:
    A tuple ``(output, final_state)``.
  """
  if implementation is None:
    implementations: Sequence[str | Callable[..., Any]] = (
        _DEFAULT_IMPLEMENTATION
    )
  elif isinstance(implementation, str) or callable(implementation):
    implementations = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")
  else:
    implementations = tuple(implementation)

  errors: list[Exception] = []
  for impl in implementations:
    if isinstance(impl, str):
      if impl not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown implementation: {impl}")
      fn = IMPLEMENTATIONS[impl]
    else:
      fn = impl

    try:
      return fn(
          q,
          k,
          s,
          g,
          v,
          chunk_size=chunk_size,
          scale=scale,
          initial_state=initial_state,
          output_final_state=output_final_state,
      )
    except NotImplementedError as e:
      if len(implementations) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
