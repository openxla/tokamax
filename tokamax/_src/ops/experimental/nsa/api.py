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

"""Native Sparse Attention (NSA) API."""

# Dimensions.
#   B  batch size
#   H  number of attention heads
#   T  sequence length
#   D  head dimension

from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.nsa import base

Implementation: TypeAlias = Literal["xla", "mosaic", "mosaic_tpu"]

IMPLEMENTATIONS = dict(
    xla=base.NativeSparseAttention(),
)

try:
  from tokamax._src.ops.experimental.nsa import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top

  IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasTpuNativeSparseAttention()
  )
except ImportError:
  pass

_DEFAULT_IMPLEMENTATIONS: Final[Sequence[Implementation]] = (
    ("mosaic_tpu", "xla") if "mosaic_tpu" in IMPLEMENTATIONS else ("xla",)
)


@jaxtyping.jaxtyped
def native_sparse_attention(
    q: Float[Array, "B H T D"],
    k: Float[Array, "B H T D"],
    v: Float[Array, "B H T D"],
    g_cmp: Float[Array, "B H T D"] | None = None,
    g_slc: Float[Array, "B H T D"] | None = None,
    g_swa: Float[Array, "B H T D"] | None = None,
    *,
    chunk_size: int = 256,
    topk: int = 4,
    window: int = 128,
    cmp_block_size: int = 64,
    scale: float | None = None,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> Float[Array, "B H T D"]:
  """Computes Native Sparse Attention (NSA).

  Args:
    q: Query tensor with shape `[B, H, T, D]`.
    k: Key tensor with shape `[B, H, T, D]`.
    v: Value tensor with shape `[B, H, T, D]`.
    g_cmp: Optional compression branch gate with shape `[B, H, T, D]`.
    g_slc: Optional selection branch gate with shape `[B, H, T, D]`.
    g_swa: Optional sliding-window branch gate with shape `[B, H, T, D]`.
    chunk_size: Sequence chunk size per block. Sequence length must be a
      multiple of `chunk_size`.
    topk: Number of historical blocks to route to per query block.
    window: Local causal sliding window size.
    cmp_block_size: Block size for coarse compression branch. Sequence length
      must be a multiple of `cmp_block_size`.
    scale: Softmax scale factor. Defaults to `head_dim ** -0.5`.
    implementation: Which implementation(s) to use, e.g. `"mosaic_tpu"`,
      `"xla"`. If None, attempts TPU kernel first with fallback to XLA.

  Returns:
    Output tensor with shape `[B, H, T, D]`.

  Raises:
    ValueError: If shapes are invalid or implementation is unknown.
    NotImplementedError: If implementation is not supported on the active
      device.
  """
  if implementation is None:
    implementations: Sequence[Implementation] = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementations = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")
  else:
    implementations = implementation

  errors = []
  for impl in implementations:
    if impl == "mosaic":
      impl = "mosaic_tpu"
    if impl not in IMPLEMENTATIONS:
      raise ValueError(f"Unknown implementation: {impl}")

    try:
      return IMPLEMENTATIONS[impl](
          q,
          k,
          v,
          g_cmp,
          g_slc,
          g_swa,
          chunk_size=chunk_size,
          topk=topk,
          window=window,
          cmp_block_size=cmp_block_size,
          scale=scale,
      )
    except NotImplementedError as e:
      if len(implementations) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
