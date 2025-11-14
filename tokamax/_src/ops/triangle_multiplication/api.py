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
"""Triangle Multiplication API."""

from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

import jax
from jaxtyping import Array, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.triangle_multiplication import base


Implementation: TypeAlias = Literal["xla"]

IMPLEMENTATIONS = dict(xla=base.TriangleMultiplication())


_DEFAULT_IMPLEMENTATIONS: Final[Sequence[Implementation]] = ("xla",)


@jaxtyping.jaxtyped
def triangle_multiplication(
    x: Float[Array, "N N C"],
    mask: Bool[Array, "N N"],
    projection_in_weights: Float[Array, "C 2 H"],
    gate_in_weights: Float[Array, "C 2 H"],
    projection_out_weights: Float[Array, "H D"],
    gate_out_weights: Float[Array, "C D"],
    layernorm_in_scale: Float[Array, "C"],
    layernorm_in_offset: Float[Array, "C"],
    layernorm_out_scale: Float[Array, "H"],
    layernorm_out_offset: Float[Array, "H"],
    triangle_type: Literal["incoming", "outgoing"],
    *,
    precision: jax.lax.PrecisionLike = None,
    epsilon: float = 1e-6,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> Float[Array, "N N D"]:
  """Triangle multiplicative update.

  Implements Supplementary Algorithm 11 and 12 of 'Highly accurate protein
  structure prediction with AlphaFold', Jumper et. al. 2021.

  Args:
    x: The input array of shape `[N, N, C]`.
    mask: A boolean mask of shape `[N, N]`.
    projection_in_weights: Weights for projection layer `[C 2 H]`.
    gate_in_weights: Weights for gate layer `[C 2 H]`.
    projection_out_weights: Weights for the output projection layer `[H, D]`.
    gate_out_weights: Weights for the output gate layer `[C, D]`.
    layernorm_in_scale: Scale for the input layer normalization `[C]`.
    layernorm_in_offset: Offset for the input layer normalization `[C]`.
    layernorm_out_scale: Scale for the output layer normalization `[H]`.
    layernorm_out_offset: Offset for the output layer normalization `[H]`.
    triangle_type: The type of triangle multiplication, either "incoming" or
      "outgoing".
    precision: Specifies the matrix multiplication precision. Default is `None`,
      which means the default precision for the backend.
    epsilon: Epsilon value added to the denominator to avoid division by zero.
      Default is 1e-6.
    implementation: The implementation to use. By default, `None` is used, which
      will automatically select the best available backend, and is guaranteed to
      work on all platforms. If a sequence is passed, the first implementation
      that doesn't raise a `NotImplementedError` is used.

  Returns:
    The normalized array with the same shape as the input `x`.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS

  if not isinstance(implementation, (tuple, list)):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  if tuple(implementation) != ("xla",):
    raise NotImplementedError("Only XLA implementation is supported.")

  impl = IMPLEMENTATIONS["xla"]
  return impl(
      x=x,
      mask=mask,
      projection_in_weights=projection_in_weights,
      gate_in_weights=gate_in_weights,
      projection_out_weights=projection_out_weights,
      gate_out_weights=gate_out_weights,
      layernorm_in_scale=layernorm_in_scale,
      layernorm_in_offset=layernorm_in_offset,
      layernorm_out_scale=layernorm_out_scale,
      layernorm_out_offset=layernorm_out_offset,
      triangle_type=triangle_type,
      precision=precision,
      epsilon=epsilon,
  )
