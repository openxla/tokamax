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
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src import shape as shape_lib
from tokamax._src.ops.gated_linear_unit import api as glu_api
from tokamax._src.ops.normalization import api as norm_api


Implementation: TypeAlias = Literal["xla"]


_DEFAULT_IMPLEMENTATIONS: Final[Sequence[Implementation]] = ("xla",)


@jaxtyping.jaxtyped
def triangle_multiplication(
    x: Float[Array, "N N C"],
    mask: Bool[Array, "N N"],
    gate_projection_weights: Float[Array, "C 2 H 2"],
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
    gate_projection_weights: Fused weights for gate and projection layers
      `[C, 2, H, 2]`.
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

  mask = mask[..., None]

  c_dim = x.shape[-1]
  h_dim = gate_projection_weights.shape[2]

  gate_projection_weights = gate_projection_weights.reshape(c_dim, 2, -1)

  left_act = norm_api.layer_norm(
      x,
      scale=layernorm_in_scale,
      offset=layernorm_in_offset,
      epsilon=epsilon,
      axis=-1,
      implementation=implementation,
  )

  proj_act = glu_api.gated_linear_unit(
      left_act,
      gate_projection_weights,
      activation=jax.nn.sigmoid,
      precision=precision,
      implementation=implementation,
  )
  proj_act = mask * proj_act

  proj_act = shape_lib.einshape("ij(dc)->dcij", d=2, c=h_dim)(proj_act)
  left_proj_act, right_proj_act = proj_act

  equation = "cik,cjk->ijc" if triangle_type == "outgoing" else "ckj,cki->ijc"
  act = jnp.einsum(equation, left_proj_act, right_proj_act, precision=precision)

  act = norm_api.layer_norm(
      act,
      scale=layernorm_out_scale,
      offset=layernorm_out_offset,
      epsilon=epsilon,
      axis=-1,
      implementation=implementation,
  )

  act = jnp.dot(act, projection_out_weights, precision=precision)

  gate_values = jnp.dot(left_act, gate_out_weights, precision=precision)
  act *= jax.nn.sigmoid(gate_values)

  return act
