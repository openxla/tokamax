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
"""Gated linear unit API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, TypeAlias

from absl import logging
import immutabledict
import jax
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import gpu_utils
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit.base import FusedWeights, UnfusedWeights  # pylint: disable=g-importing-member,g-multiple-import

# TODO: Add Pallas-Mosaic-GPU implementation.
Implementation: TypeAlias = Literal['triton', 'xla']

IMPLEMENTATIONS = dict(xla=base.GatedLinearUnit())
_DEFAULT_IMPLEMENTATION = ('xla',)

try:
  from tokamax._src.ops.gated_linear_unit import pallas_triton  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS['triton'] = pallas_triton.PallasTritonGatedLinearUnit()
  _DEFAULT_IMPLEMENTATION = ('triton',) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass


IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(IMPLEMENTATIONS)
)


def gated_linear_unit(
    x: Float[Array, '*B M K'],
    weights: FusedWeights | UnfusedWeights,
    *,
    activation: Callable[[jax.Array], jax.Array] | None = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> Float[Array, '*B M N']:
  """Applies a gated linear unit (https://arxiv.org/abs/1612.08083).

  Computes `activation(x @ w_gate) * (x @ w_up)`.

  This is SwiGLU when `activation=jax.nn.swish`, GEGLU when
  `activation=jax.nn.gelu`, REGLU when `activation=jax.nn.relu`, and GLU when
  `activation=jax.nn.sigmoid` (https://arxiv.org/abs/2002.05202).

  Args:
    x: the input array.
    weights: the weights for the linear transformations. Can be a single fused
      array of shape `(K, 2, N)`, or a tuple of unfused arrays, each of shape
      `(K, N)`.
    activation: optional activation function.
    precision: specifies the matrix multiplication precision. Either `None`
      (default), which means the default precision for the backend, or
      `jax.lax.Precision` or `jax.lax.DotAlgorithmPreset` to specify the
      precision.
    implementation: if `None` (default), an implementation is automatically
      chosen and will work on any platform. 'xla' will use an XLA only
      implementation and work on any platform, and 'triton' will use a fused
      Triton GPU kernel. Only a subset of data types, shapes and GPUs are
      supported by 'triton', with an exception thrown if the input falls outside
      of these supported cases.

  Raises:
    ExceptionGroup: if all implementations fail. This will contain the errors
    from each implementation.

  Returns:
    The output array.
  """

  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATION

  if not isinstance(implementation, (tuple, list)):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError('`implementation` must not be an empty sequence.')

  errors = []
  fn = base.GatedLinearUnit()
  for impl in implementation:
    if impl == 'triton' and not gpu_utils.has_triton_support():
      continue
    if isinstance(impl, str):
      if impl not in IMPLEMENTATIONS:
        raise ValueError(f'Unknown implementation: {impl}')
      fn = IMPLEMENTATIONS[impl]

    try:
      return fn(x, weights, activation=activation, precision=precision)
    except NotImplementedError as e:
      logging.error('Failed to run implementation: %s', e)
      errors.append(e)

  raise ExceptionGroup('all implementations failed', errors)
