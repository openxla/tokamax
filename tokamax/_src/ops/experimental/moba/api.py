# Copyright 2026 Google LLC. All Rights Reserved.
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

"""Mixture of Block Attention (MoBA) API."""

# Dimensions.
#   B      batch
#   H      number of attention heads
#   T      sequence length
#   D      head dimension

from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.moba import base

Implementation: TypeAlias = Literal["xla", "mosaic"]

IMPLEMENTATIONS = dict(
    xla=base.MixtureOfBlockAttention(),
)

try:
  from tokamax._src.ops.experimental.moba import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top

  IMPLEMENTATIONS["mosaic"] = (
      pallas_mosaic_tpu.PallasMosaicTpuMixtureOfBlockAttention()
  )
except ImportError:
  pass

_DEFAULT_IMPLEMENTATIONS: Final[Sequence[Implementation]] = (
    ("mosaic", "xla") if "mosaic" in IMPLEMENTATIONS else ("xla",)
)


@jaxtyping.jaxtyped
def mixture_of_block_attention(
    q: Float[Array, "B H T D"],
    k: Float[Array, "B H T D"],
    v: Float[Array, "B H T D"],
    *,
    topk: int = 4,
    chunk_size: int = 256,
    scale: float | None = None,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> Float[Array, "B H T D"]:
  """Computes Mixture of Block Attention (MoBA).

  Args:
    q: Query tensor with shape `[B, H, T, D]`.
    k: Key tensor with shape `[B, H, T, D]`.
    v: Value tensor with shape `[B, H, T, D]`.
    topk: Number of historical blocks to route to per query block.
    chunk_size: Sequence chunk size per block. Sequence length must be a
      multiple of chunk_size.
    scale: Softmax scale factor. Defaults to `head_dim ** -0.5`.
    implementation: Which implementation(s) to use, e.g. `"mosaic"`,
      `"xla"`. If None, attempts TPU kernel first with fallback to XLA.

  Returns:
    Output tensor with shape `[B, H, T, D]`.

  Raises:
    ValueError: If shapes are invalid or implementation is unknown.
    NotImplementedError: If implementation is not supported on the active
      device.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  errors = []
  for impl in implementation:
    if impl not in IMPLEMENTATIONS:
      raise ValueError(f"Unknown implementation: {impl}")

    try:
      return IMPLEMENTATIONS[impl](
          q,
          k,
          v,
          topk=topk,
          chunk_size=chunk_size,
          scale=scale,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
