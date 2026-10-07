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
"""DeepSeek-V4 RoPE API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
from tokamax._src.ops.experimental.tpu.rope import base

type Implementation = Literal["xla", "mosaic_tpu"]

_IMPLEMENTATIONS = dict(xla=base.Rope())
_DEFAULT_IMPLEMENTATIONS = ("xla",)

try:
  from tokamax._src.ops.experimental.tpu.rope import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuRope()
  _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def rope(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    mode: base.Mode = "rope",
    inverse: bool = False,
    eps: float = 1e-6,
    quant_dtype: jax.typing.DTypeLike | None = None,
    implementation: (
        Implementation | Sequence[Implementation | Callable[..., Any]] | None
    ) = None,
) -> base.RopeOutput:
  """Applies the DeepSeek-V4 RoPE to the trailing `rotary_dim` channels of `x`.

  DeepSeek-V4 rotates the trailing `rotary_dim` channels of each head with the
  GPT-J (interleaved) rotation, reading each token's `[cos | sin]` row from
  `cos_sin_cache` at its position. `mode` selects the variant:

  *   `"rope"`: the rotation alone.
  *   `"qnorm_rope"`: a per-head RMSNorm (no weight) over the whole head, then
      the rotation. `x` must be rank 3.
  *   `"rope_quant"`: the rotation, then per-row dynamic quantization to
      `quant_dtype`. `head_dim` must be 128.

  The Pallas implementation (`"mosaic_tpu"`) donates `x` in the `"rope"` and
  `"qnorm_rope"` modes, as upstream does: the kernel writes the result into the
  buffer of `x`, so `x` must not be used after the call. Under an outer
  `jax.jit` that does not donate `x`, XLA copies it instead.

  Args:
    x: `(num_tokens, head_dim)` or `(num_tokens, num_heads, head_dim)`
      floating-point values. `head_dim` is a multiple of 128.
    positions: `(num_tokens,)` int32 RoPE position of each token, in `[0,
      max_pos)`. The Pallas kernels do not bounds check them.
    cos_sin_cache: `(max_pos, rotary_dim)` float32 packed `[cos | sin]` rows;
      `rotary_dim` is even and at most 128.
    mode: The variant, one of `"rope"`, `"qnorm_rope"` and `"rope_quant"`.
    inverse: Whether to negate `sin`, i.e. apply the transposed rotation.
    eps: The RMSNorm epsilon (`"qnorm_rope"` only).
    quant_dtype: The floating-point quantized dtype (`"rope_quant"` only).
      Defaults to `float8_e4m3fn`.
    implementation: The implementation to use. By default, `None` is used, which
      selects the best available backend. If a sequence is passed, the first
      implementation that doesn't raise a `NotImplementedError` is used.

  Returns:
    For `"rope"` and `"qnorm_rope"`, the result, with the shape and dtype of
    `x`. For `"rope_quant"`, `(q, scales)`: `q` of `x.shape` in `quant_dtype`
    and float32 `scales` of `x.shape[:-1]`, with `q * scales[..., None]`
    approximating the rotation.

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
          mode=mode,
          inverse=inverse,
          eps=eps,
          quant_dtype=quant_dtype,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
