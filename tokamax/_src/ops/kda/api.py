# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Kimi Delta Attention API."""

from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

import jax
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.kda import base


Implementation: TypeAlias = Literal["xla"]

IMPLEMENTATIONS = dict(xla=base.KimiDeltaAttention())
_DEFAULT_IMPLEMENTATIONS: Final[Sequence[Implementation]] = ("xla",)


@jaxtyping.jaxtyped
def kimi_delta_attention(
    q: Float[Array, "B T H K"],
    k: Float[Array, "B T H K"],
    v: Float[Array, "B T H V"],
    g: Float[Array, "B T H K"],
    beta: Float[Array, "B T H"],
    *,
    scale: float | None = None,
    initial_state: Float[Array, "B H K V"] | None = None,
    output_final_state: bool = False,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> tuple[Float[Array, "B T H V"], Float[Array, "B H K V"] | None]:
  """Kimi Delta Attention.

  Kimi Delta Attention is a recurrent linear attention module. This portable
  XLA implementation evaluates the dense recurrence and serves as the reference
  contract for future chunk-wise implementations.

  Args:
    q: Query tensor with shape `[B, T, H, K]`.
    k: Key tensor with shape `[B, T, H, K]`.
    v: Value tensor with shape `[B, T, H, V]`.
    g: Per-channel gate tensor in log space, shape `[B, T, H, K]`.
    beta: Per-token delta-rule learning-rate tensor, shape `[B, T, H]`.
    scale: Query scale. Defaults to `K ** -0.5`.
    initial_state: Optional initial recurrent state, shape `[B, H, K, V]`.
    output_final_state: Whether to return the final recurrent state.
    implementation: The implementation to use. Only `"xla"` is currently
      supported.

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

  if tuple(implementation) != ("xla",):
    raise NotImplementedError("Only XLA implementation is supported.")

  return IMPLEMENTATIONS["xla"](
      q=q,
      k=k,
      v=v,
      g=g,
      beta=beta,
      scale=scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
  )

