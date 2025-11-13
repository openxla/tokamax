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

from dataclasses import dataclass
from typing import Any
from typing import Literal, TypeVar
import jax
import jax.numpy as jnp
from jaxtyping import Array, Integer, Real
from tokamax._src.ops import op
from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference

# Residuals are X, Labels, W and LSE
_Config = TypeVar("_Config")
Residuals = tuple[Real[Array, "B"],]


@dataclass(frozen=True, kw_only=True)
class LinearSoftmaxCrossEntropyLoss(
    op.Op[Any, jax.Array, Residuals, _Config, None]
):
  """Linear Softmax Cross-Entropy Loss tokamax Op API using reference impl."""

  def __post_init__(self):
    object.__setattr__(
        self, "vjp", LinearSoftmaxCrossEntropyLossVjp(config=self.config)
    )

  def _fwd(
      self,
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean"] = "sum",
      config: _Config,
      return_residuals: bool,
  ) -> tuple[jax.Array, Residuals]:
    loss, lse = reference.linear_softmax_cross_entropy_loss_fwd_reference(
        x, labels, w, reduction=reduction
    )

    return loss, (lse,)


@dataclass(frozen=True, kw_only=True)
class LinearSoftmaxCrossEntropyLossVjp(op.Op[Any, Any, None, _Config, None]):
  """Linear Softmax Cross-Entropy Loss VJP wrapper in tokamax Op API."""

  def _fwd(
      self,
      residuals: Residuals,
      out: jax.Array,  # pylint: disable=unused-argument
      dout: Real[Array, ""],
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean"] = "sum",
      config: _Config,
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
    )
    labels_grad = jnp.zeros_like(labels)
    return (x_grad, labels_grad, w_grad), None
