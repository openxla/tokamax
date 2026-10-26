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
"""Experimental RWKV-7 (DPLR Delta Rule) API."""

from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

from jaxtyping import (  # pylint: disable=g-multiple-import,g-importing-member
    Array,
    Float,
)
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.rwkv7 import base

Implementation: TypeAlias = Literal["xla", "mosaic_tpu", "mosaic"]

IMPLEMENTATIONS = {
    "xla": base.RWKV7(),
}

try:
  from tokamax._src.ops.experimental.rwkv7 import (
      pallas_mosaic_tpu,  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]
  )

  IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuRWKV7()
except ImportError:
  pass

_DEFAULT_IMPLEMENTATIONS: Final[Sequence[Implementation]] = (
    ("mosaic_tpu", "xla") if "mosaic_tpu" in IMPLEMENTATIONS else ("xla",)
)


@jaxtyping.jaxtyped
def rwkv7_delta_rule(
    q: Float[Array, "B T H K"],
    k: Float[Array, "B T H K"],
    v: Float[Array, "B T H V"],
    alpha: Float[Array, "B T H K"],
    beta: Float[Array, "B T H K"],
    gk: Float[Array, "B T H K"],
    *,
    chunk_size: int = 64,
    scale: float | None = None,
    initial_state: Float[Array, "B H K V"] | None = None,
    output_final_state: bool = False,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> tuple[Float[Array, "B T H V"], Float[Array, "B H K V"] | None]:
  """RWKV-7 Diagonal Plus Low-Rank (DPLR) Delta Rule.

  Args:
    q: Query tensor with shape `[B, T, H, K]`.
    k: Key tensor with shape `[B, T, H, K]`.
    v: Value tensor with shape `[B, T, H, V]`.
    alpha: Low-rank removal vector tensor with shape `[B, T, H, K]`.
    beta: Low-rank replacement vector tensor with shape `[B, T, H, K]`.
    gk: Log-space diagonal decay tensor with shape `[B, T, H, K]`.
    chunk_size: Chunk size for chunked execution. Defaults to 64.
    scale: Query scale. Defaults to `K ** -0.5`.
    initial_state: Optional initial recurrent state, shape `[B, H, K, V]`.
    output_final_state: Whether to return the final recurrent state.
    implementation: The implementation to use (`"xla"`, `"mosaic_tpu"`, or
      `"mosaic"`).

  Returns:
    A pair `(output, final_state)`. The output has shape `[B, T, H, V]`.
    The final state has shape `[B, H, K, V]` when requested, otherwise `None`.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  errors = []
  for impl in implementation:
    if impl == "mosaic":
      impl = "mosaic_tpu"
    if impl not in IMPLEMENTATIONS:
      raise ValueError(f"Unknown implementation: {impl}")

    try:
      return IMPLEMENTATIONS[impl](
          q,
          k,
          v,
          alpha,
          beta,
          gk,
          chunk_size=chunk_size,
          scale=scale,
          initial_state=initial_state,
          output_final_state=output_final_state,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  if len(errors) == 1:
    raise errors[0]
  raise RuntimeError(f"all implementations failed: {errors}")
