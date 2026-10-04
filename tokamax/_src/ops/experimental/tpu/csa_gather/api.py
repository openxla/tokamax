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
"""CSA Gather API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.csa_gather import base

type Implementation = Literal["xla", "mosaic", "mosaic_tpu"]

_IMPLEMENTATIONS = dict(xla=base.CsaGather())
_DEFAULT_IMPLEMENTATIONS = ("xla",)

try:
  from tokamax._src.ops.experimental.tpu.csa_gather import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuCsaGather()
  _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def csa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    num_valid_indices: jax.Array | int | None = None,
    *,
    top_k: int = 1024,
    implementation: (
        Implementation | Sequence[Implementation | Callable[..., Any]] | None
    ) = None,
) -> tuple[jax.Array, jax.Array]:
  """Gathers DeepSeek-V4 CSA NoPE and RoPE cache rows for the given tokens.

  See `base` for the cache and output layouts.

  Args:
    nope_cache: `(num_pages, page_size, 128)` int32 NoPE cache.
    rope_cache: `(num_pages, page_size // 4, 128)` int32 RoPE cache.
    indices: `(N,)` int32 token indices into the caches. `N` is a multiple of
      `top_k`.
    num_valid_indices: Optional scalar or `(1,)` int32 number of valid leading
      indices, a multiple of `top_k`. Output rows past it are unspecified.
    top_k: The consumer's row block (the attention kernel's top-k), a multiple
      of 128.
    implementation: The implementation to use. By default, `None` is used, which
      selects the best available backend. If a sequence is passed, the first
      implementation that doesn't raise a `NotImplementedError` is used.

  Returns:
    A tuple `(nope_out, rope_out)` of shapes `(N, 128)` and `(N // 4, 128)`,
    both int32.

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

  if isinstance(num_valid_indices, int):
    num_valid_indices = jnp.asarray(num_valid_indices, jnp.int32)

  errors = []
  for impl in implementation:
    if isinstance(impl, str):
      if impl == "mosaic":
        impl = "mosaic_tpu"
      if impl not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation: {impl}. You may need to add a dependency"
            " on the corresponding backend."
        )
      impl = IMPLEMENTATIONS[impl]

    try:
      return impl(
          nope_cache, rope_cache, indices, num_valid_indices, top_k=top_k
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
