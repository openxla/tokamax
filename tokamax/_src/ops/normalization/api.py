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
"""Layer Normalization API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, TypeAlias

from absl import logging
import immutabledict
import jax
from tokamax._src import triton
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import pallas_triton as pl_norm


Implementation: TypeAlias = Literal['xla', 'triton']

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(
        xla=base.Normalization(),
        triton=pl_norm.PallasTritonNormalization(),
    )
)


_DEFAULT_IMPLEMENTATION = ('triton', 'xla')


def layer_norm(
    x: jax.Array,
    scale: jax.Array | None,
    offset: jax.Array | None,
    *,
    axis: int = -1,
    epsilon: float = 1e-06,
    scale_offset: float = 0.0,
    subtract_mean: bool = True,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> jax.Array:
  """Normalization layer.

  Implements LayerNorm (https://arxiv.org/abs/1607.06450), and RMSNorm
  (https://arxiv.org/abs/1910.07467) if `subtract_mean=False`.

  FP16/BF16 inputs will first be upcast to FP32, and all computations will be
  done in FP32. The result will downcasted to the input data type.

  Arguments:
    x: The array to be normalized.
    scale: An optional one-dimensional array of length `x.shape[axis]`.
    offset: An optional one-dimensional array of length `x.shape[axis]`.
    axis: The axis along which to normalize. Default is `-1`.
    epsilon: Epsilon value added to the denominator to avoid division by zero.
      Default is 1e-6.
    scale_offset: An offset added to the scale factors before scaling. Default
      is 0.0.
    subtract_mean: If `True`, use standard variance calculation. If `False`,
      assume mean is zero (i.e. RMS norm). Default is `True`.
    implementation: The implementation to use. If `None` (default), an
      implementation is automatically chosen that will work on all platforms.
      'xla' will use an XLA only implementation and works on any platform, and
      'triton' will use a Triton GPU kernel. A sequence of implementations can
      be passed, in which case all implementations will be attempted and the
      first successful result will be returned.

  Raises:
    ExceptionGroup: if all implementations fail with their error messages.

  Returns:
    The normalized array with the same shape as the input `x`.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATION

  if not isinstance(implementation, (tuple, list)):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError('`implementation` must not be an empty sequence.')

  # TODO: switch to using the offline autotuned result for the
  # the None automatic case.

  errors = []
  fn = base.Normalization()
  for impl in implementation:
    if isinstance(impl, str):
      if impl == 'triton' and not triton.has_triton_support():
        errors.append(
            NotImplementedError(
                'Triton not supported on this platform. Please use XLA'
                ' implementation.'
            )
        )
        continue
      if impl not in IMPLEMENTATIONS:
        raise ValueError(f'Unknown implementation: {impl}')
      fn = IMPLEMENTATIONS[impl]

    try:
      return fn(
          x=x,
          scale=scale,
          offset=offset,
          axis=axis,
          epsilon=epsilon,
          scale_offset=scale_offset,
          subtract_mean=subtract_mean,
      )
    except NotImplementedError as e:
      logging.error('Failed to run implementation: %s', e)
      errors.append(e)

  raise ExceptionGroup('all implementations failed', errors)
