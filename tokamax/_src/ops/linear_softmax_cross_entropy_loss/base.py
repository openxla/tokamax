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

import dataclasses
from typing import Any, Final, Literal, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Integer, Real  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src import precision as precision_lib
from tokamax._src.ops import op
from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference

type AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray
CanonicalPrecision = precision_lib.CanonicalPrecision
DEFAULT_PRECISION: Final[CanonicalPrecision] = (
    jax.lax.Precision.DEFAULT,
    jax.lax.Precision.DEFAULT,
)

# Residuals are X, Labels, W and LSE
type Residuals = tuple[Real[Array, "B"],]
type AbstractResiduals = tuple[Real[Array | AbstractArray, "B"],]


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinearSoftmaxCrossEntropyLoss[C](
    op.Op[Any, jax.Array, Residuals, C, None]
):
  """Linear Softmax Cross-Entropy Loss tokamax Op API using reference impl."""

  def __post_init__(self):
    object.__setattr__(
        self, "vjp", LinearSoftmaxCrossEntropyLossVjp(config=self.config)
    )

  @override
  def bind(
      self,
      x: Real[Array | AbstractArray, "B H"],
      labels: Integer[Array | AbstractArray, "B"],
      w: Real[Array | AbstractArray, "H V"],
      *,
      reduction: Literal["sum", "mean", "none"] = "sum",
      precision: jax.lax.PrecisionLike = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    return super().bind(
        x=x,
        labels=labels,
        w=w,
        reduction=reduction,
        precision=precision_lib.canonicalize_precision(precision),
        return_residuals=return_residuals,
    )

  def _fwd(
      self,
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean", "none"] = "sum",
      precision: CanonicalPrecision = DEFAULT_PRECISION,
      config: C,
      return_residuals: bool,
  ) -> tuple[jax.Array, Residuals]:
    loss, lse = reference.linear_softmax_cross_entropy_loss_fwd_reference(
        x, labels, w, reduction=reduction, precision=precision
    )

    return loss, (lse,)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinearSoftmaxCrossEntropyLossVjp[C](op.Op[Any, Any, None, C, None]):
  """Linear Softmax Cross-Entropy Loss VJP wrapper in tokamax Op API."""

  @override
  def bind(
      self,
      residuals: AbstractResiduals,
      out: Real[Array | AbstractArray, "*"],
      dout: Real[Array | AbstractArray, "*"],
      x: Real[Array | AbstractArray, "B H"],
      labels: Integer[Array | AbstractArray, "B"],
      w: Real[Array | AbstractArray, "H V"],
      *,
      reduction: Literal["sum", "mean", "none"] = "sum",
      precision: jax.lax.PrecisionLike = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    return super().bind(
        residuals=residuals,
        out=out,
        dout=dout,
        x=x,
        labels=labels,
        w=w,
        reduction=reduction,
        precision=precision_lib.canonicalize_precision(precision),
        return_residuals=return_residuals,
    )

  def _fwd(
      self,
      residuals: Residuals,
      out: jax.Array,  # pylint: disable=unused-argument
      dout: Real[Array, "*"],
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean", "none"] = "sum",
      precision: CanonicalPrecision = DEFAULT_PRECISION,
      config: C,
      return_residuals: bool,
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
    """Computes Linear Softmax Cross-Entropy Loss VJP `(dx, dlabels, dw)`."""
    del out

    (lse,) = residuals

    x_grad, w_grad = reference.linear_softmax_cross_entropy_loss_bwd_reference(
        dout,
        lse,
        x,
        labels,
        w,
        reduction=reduction,
        precision=precision,
    )
    labels_grad = jnp.zeros_like(labels)
    return (x_grad, labels_grad, w_grad), None
