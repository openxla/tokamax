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
"""Experimental Gated Delta Net 2 (GDN-2) API."""

# Dimensions.
#   B   batch size
#   T   sequence length
#   H   number of attention heads
#   K   key head dimension
#   V   value head dimension

from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

from jaxtyping import (  # pylint: disable=g-multiple-import,g-importing-member
    Array,
    Float,
)
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.gdn2 import base

Implementation: TypeAlias = Literal["xla", "mosaic"]

IMPLEMENTATIONS = {
    "xla": base.GatedDeltaNet2(),
}

try:
  from tokamax._src.ops.experimental.gdn2 import (
      pallas_mosaic_tpu,  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]
  )

  IMPLEMENTATIONS["mosaic"] = (
      pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet2()
  )
except ImportError:
  pass

_DEFAULT_IMPLEMENTATIONS: Final[Sequence[Implementation]] = (
    ("mosaic", "xla") if "mosaic" in IMPLEMENTATIONS else ("xla",)
)


@jaxtyping.jaxtyped
def gated_delta_net_2(
    q: Float[Array, "B T H K"],
    k: Float[Array, "B T H K"],
    v: Float[Array, "B T H V"],
    g: Float[Array, "B T H K"],
    b: Float[Array, "B T H K"],
    w: Float[Array, "B T H V"],
    *,
    chunk_size: int = 64,
    scale: float | None = None,
    initial_state: Float[Array, "B H K V"] | None = None,
    output_final_state: bool = False,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> tuple[Float[Array, "B T H V"], Float[Array, "B H K V"] | None]:
  """Gated Delta Net 2 (GDN-2).

  GDN-2 is a linear recurrent delta-rule operator with input-dependent erase
  gating (`b`) and write gating (`w`).

  Args:
    q: Query tensor with shape `[B, T, H, K]`.
    k: Key tensor with shape `[B, T, H, K]`.
    v: Value tensor with shape `[B, T, H, V]`.
    g: Per-channel log-space decay tensor with shape `[B, T, H, K]`.
    b: Per-token, per-channel erase gate tensor with shape `[B, T, H, K]`.
    w: Per-token, per-channel write gate tensor with shape `[B, T, H, V]`.
    chunk_size: Chunk size for chunked execution. Defaults to 64.
    scale: Query scale. Defaults to `K ** -0.5`.
    initial_state: Optional initial recurrent state, shape `[B, H, K, V]`.
    output_final_state: Whether to return the final recurrent state.
    implementation: The implementation to use. By default, the Mosaic TPU
      implementation is attempted first when available, with XLA as a fallback.
      `"xla"` evaluates the recurrent reference implementation. `"mosaic"` uses
      the experimental Pallas/Mosaic TPU kernel. A sequence tries
      implementations in order, falling back when an implementation raises
      `NotImplementedError`.

  Returns:
    A pair `(output, final_state)`. The output has shape `[B, T, H, V]`.
    The final state has shape `[B, H, K, V]` when requested, otherwise `None`.
  """
  if implementation is None:
    implementations: Sequence[Implementation] = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementations = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")
  else:
    implementations = tuple(implementation)

  errors = []
  for impl in implementations:
    if impl not in IMPLEMENTATIONS:
      raise ValueError(f"Unknown implementation: {impl}")

    try:
      return IMPLEMENTATIONS[impl](
          q,
          k,
          v,
          g,
          b,
          w,
          chunk_size=chunk_size,
          scale=scale,
          initial_state=initial_state,
          output_final_state=output_final_state,
      )
    except NotImplementedError as e:
      if len(implementations) == 1:
        raise
      errors.append(e)

  if len(errors) == 1:
    raise errors[0]
  raise RuntimeError(f"all implementations failed: {errors}")
