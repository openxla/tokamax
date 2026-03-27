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


Implementation: TypeAlias = Literal["mosaic_gpu", "mosaic_tpu", "triton", "xla"]

IMPLEMENTATIONS = dict(xla=base.LinearSoftmaxCrossEntropyLoss())
_DEFAULT_IMPLEMENTATION = ("xla",)

try:
  from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_triton  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["triton"] = (
      pallas_triton.PallasTritonLinearSoftmaxCrossEntropyLoss()
  )

  _DEFAULT_IMPLEMENTATION = ("triton",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass

try:
  from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss()
  )

  _DEFAULT_IMPLEMENTATION = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass

try:
  from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_gpu  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["mosaic_gpu"] = (
      pallas_mosaic_gpu.PallasMosaicGpuLinearSoftmaxCrossEntropyLoss()
  )

  # mosaic_gpu is NOT added to _DEFAULT_IMPLEMENTATION. Its forward is at XLA
  # parity but its backward is ~3× slower (chunked scan over V vs two full-width
  # cuBLAS matmuls). The benefit is memory: the (B, V) logit matrix is never
  # materialised. Use implementation='mosaic_gpu' explicitly when the logit
  # matrix would OOM the device.
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
      backend. Can be set to "xla", "mosaic_tpu", "triton", or "mosaic_gpu"
      explicitly. The default selection order is mosaic_tpu → triton → xla,
      with each backend skipped if unavailable on the current device.
      "mosaic_gpu" is available on H100+ (SM90) but is not in the default
      chain: its forward is at XLA parity but its backward is ~3× slower due
      to chunked-scan accumulation. Use implementation='mosaic_gpu' explicitly
      when the (B, V) logit matrix would OOM the device — that is the intended
      use case. "mosaic_tpu" and "triton" are memory-efficient and avoid
      materialising the full logit matrix.

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

  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATION

  if not isinstance(implementation, (tuple, list)):
    implementation = (implementation,)

  errors = []
  for impl in implementation:
    if impl not in IMPLEMENTATIONS:
      raise ValueError(f"Unsupported implementation: {impl}")
    try:
      loss = IMPLEMENTATIONS[impl](
          x,
          labels,
          weights,
          reduction=reduction,
      )
      return loss
    except NotImplementedError as e:
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
