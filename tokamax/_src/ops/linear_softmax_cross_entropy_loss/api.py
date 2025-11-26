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

"""The Linear Softmax Cross-Entropy Loss Op API."""

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias
import jax
from jaxtyping import Array, Integer, Real, Scalar  # pylint: disable=g-multiple-import, g-importing-member
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base


Implementation: TypeAlias = Literal["mosaic_tpu", "xla"]

IMPLEMENTATIONS = dict(xla=base.LinearSoftmaxCrossEntropyLoss())
_DEFAULT_IMPLEMENTATION = ("xla",)

try:
  from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss()
  )

  _DEFAULT_IMPLEMENTATION = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass


def linear_softmax_cross_entropy_loss(
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    weights: Real[Array, "H V"],
    *,
    reduction: Literal["sum", "mean"] = "sum",
    precision: jax.lax.PrecisionLike = None,
    implementation: (
        Implementation
        | Sequence[Implementation | Callable[..., jax.Array]]
        | None
    ) = None,
) -> Real[Scalar, ""]:
  """The linear softmax cross-entropy loss op.

  The Linear Softmax Cross-Entropy Loss Op is a tokamax Op that performs a
  linear projection and cross entropy loss calculation
  `loss = -reduction(labels * log(softmax(X@W)))`
  where reduction is either sum or mean.
  This op uses the regular (unsafe) Cross-Entropy loss function
  (Like `optax.softmax_cross_entropy()`) so the logits `X@W` cannot be `-inf`

  Args:
    x: The last layer output in the dimension of (B, H) where B is batch size
      and H is the hidden dimension.
    labels: The ground truth labels index in the dimension of (B,).
    weights: The linear projection weight matrix in the dimension of (H, V)
      where V is the dimension of the output logits aka vocabulary size.
    reduction: The reduction method for the cross entropy loss. Can be set to
      "sum" or "mean" explicitly.
    precision: The precision used for jax.lax.dot_general for the linear
      projection and gradient calculation.
    implementation: By default "None" will be used to pick the best available
      backend. Can be set to "xla" or "mosaic_tpu" explicitly. The "mosaic_tpu"
      implementation is memory efficient and has almost 0 additional buffer
      overhead while the "xla" implementation needs to materialize the full
      logits

  Returns:
    The Cross-Entropy loss

  Raises:
    NotImplementedError: If the implementation is not supported.
    ExceptionGroup: If all implementations failed.
  """

  if precision is not None:
    # TODO: Add support for precision customization.
    raise NotImplementedError(
        "Customization of precision is currently not supported."
    )

  if implementation is not None:
    if implementation in IMPLEMENTATIONS:
      loss = IMPLEMENTATIONS[implementation](
          x,
          labels,
          weights,
          reduction=reduction,
      )
      return loss
    else:
      raise ValueError(f"Unsupported implementation: {implementation}")

  # Find out the best impelmentation based on the hardware.
  errors = []
  for impl in IMPLEMENTATIONS:
    try:
      loss = IMPLEMENTATIONS[impl](
          x,
          labels,
          weights,
          reduction=reduction,
      )
      return loss
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
