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
"""Triangle multiplication op."""

import types
from typing import Any, Literal, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src import precision as precision_lib
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.gated_linear_unit import base as glu_base
from tokamax._src.ops.normalization import base as norm_base
from typing_extensions import override


_Config = TypeVar('_Config')
_Key = TypeVar('_Key')
Residuals: TypeAlias = types.NoneType
CanonicalPrecision = precision_lib.CanonicalPrecision


class TriangleMultiplication(op.Op[Any, jax.Array, Residuals, _Config, _Key]):
  """Triangle multiplicative update."""

  @override
  def bind(
      self,
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
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds the arguments for the triangle multiplication function."""
    if return_residuals:
      raise NotImplementedError("`return_residuals=True` is not supported.")
    return super().bind(
        x=x,
        mask=mask,
        gate_projection_weights=gate_projection_weights,
        projection_out_weights=projection_out_weights,
        gate_out_weights=gate_out_weights,
        layernorm_in_scale=layernorm_in_scale,
        layernorm_in_offset=layernorm_in_offset,
        layernorm_out_scale=layernorm_out_scale,
        layernorm_out_offset=layernorm_out_offset,
        triangle_type=triangle_type,
        precision=precision_lib.canonicalize_precision(precision),
        epsilon=epsilon,
        return_residuals=return_residuals,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
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
      precision: CanonicalPrecision,
      epsilon: float,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[Float[Array, "N N D"], Residuals]:
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
      precision: Specifies the matrix multiplication precision.
      epsilon: Epsilon value added to the denominator to avoid division by zero.
      return_residuals: If True, returns residuals.
      config: The op config.

    Returns:
      The result of triangle multiplication of shape `[N, N, D]`.
    """
    del config, return_residuals  # Unused.
    mask = mask[..., None]

    c_dim = x.shape[-1]
    h_dim = gate_projection_weights.shape[2]

    gate_projection_weights = gate_projection_weights.reshape(c_dim, 2, -1)

    left_act = norm_base.Normalization()(
        x,
        scale=layernorm_in_scale,
        offset=layernorm_in_offset,
        epsilon=epsilon,
        axis=-1,
    )

    proj_act = glu_base.GatedLinearUnit()(
        left_act,
        gate_projection_weights,
        activation=jax.nn.sigmoid,
        precision=precision,
    )
    proj_act = mask * proj_act

    proj_act = shape_lib.einshape("ij(dc)->dcij", d=2, c=h_dim)(proj_act)
    left_proj_act, right_proj_act = proj_act

    equation = "cik,cjk->ijc" if triangle_type == "outgoing" else "ckj,cki->ijc"
    act = jnp.einsum(
        equation, left_proj_act, right_proj_act, precision=precision
    )

    act = norm_base.Normalization()(
        act,
        scale=layernorm_out_scale,
        offset=layernorm_out_offset,
        epsilon=epsilon,
        axis=-1,
    )

    act = jnp.dot(act, projection_out_weights, precision=precision)

    gate_values = jnp.dot(left_act, gate_out_weights, precision=precision)
    act *= jax.nn.sigmoid(gate_values)

    return act, None
