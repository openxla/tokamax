# Copyright 2026 Google LLC
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
"""DeepSeek-V4 fused reverse-RoPE `wo_a` output projection API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
from tokamax._src.ops.experimental.tpu.o_projection import base

type Implementation = Literal["xla", "mosaic_tpu"]

_IMPLEMENTATIONS = dict(xla=base.OProjection())
_DEFAULT_IMPLEMENTATIONS = ("xla",)

try:
  from tokamax._src.ops.experimental.tpu.o_projection import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuOProjection()
  _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def o_projection(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    wo_a: jax.Array,
    wo_a_scale: jax.Array,
    *,
    inverse: bool = True,
    quantize_activations: bool = True,
    implementation: (
        Implementation | Sequence[Implementation | Callable[..., Any]] | None
    ) = None,
) -> jax.Array:
  """DeepSeek-V4's fused reverse-RoPE `wo_a` output projection.

  DeepSeek-V4's attention output `x` is `(T, G * 8, head_dim)`: `G` groups of 8
  heads. This undoes the RoPE on the trailing `rotary_dim` channels of each head
  (with `inverse=True`), then projects each group's `8 * head_dim` features
  with its own `(8 * head_dim, R)` block of the fp8 `wo_a`, scaled per output
  column by `wo_a_scale`. It stops before `wo_b`.

  Args:
    x: `(T, G * 8, head_dim)` bf16 attention output; `head_dim` is a multiple of
      128.
    positions: `(T,)` int32 RoPE position of each token, in `[0, max_pos)`. The
      Pallas kernels do not bounds check them.
    cos_sin_cache: `(max_pos, rotary_dim)` float32 packed `[cos | sin]` rows;
      `rotary_dim` is even and at most 128.
    wo_a: `(8 * head_dim, G * R)` float8_e4m3fn projection weights.
    wo_a_scale: `(G * R,)` float32 per-column weight scales.
    inverse: Whether to negate `sin`, i.e. apply the transposed rotation.
      Defaults to `True`, which undoes the RoPE applied to the values.
    quantize_activations: Whether to quantize the roped activations to fp8 (one
      scale per token and group) so both matmul operands are fp8.
    implementation: The implementation to use. By default, `None` is used, which
      selects the best available backend. If a sequence is passed, the first
      implementation that doesn't raise a `NotImplementedError` is used.

  Returns:
    The `(T, G * R)` bf16 projection.

  Raises:
    ExceptionGroup: If multiple implementations are provided and all of them
      raise `NotImplementedError`.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  errors = []
  for impl in implementation:
    if isinstance(impl, str):
      if impl not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation: {impl}. You may need to add a dependency"
            " on the corresponding backend."
        )
      impl = IMPLEMENTATIONS[impl]

    try:
      return impl(
          x,
          positions,
          cos_sin_cache,
          wo_a,
          wo_a_scale,
          inverse=inverse,
          quantize_activations=quantize_activations,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
