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
"""Splash Attention Op API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal
import immutabledict
import jax
from jaxtyping import Array, Bool, Float
from tokamax._src.ops.splash_attention import base
from tokamax._src.ops.splash_attention import reference
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib

type Implementation = Literal["mosaic_tpu", "base"]

_IMPLEMENTATIONS = dict(
    base=base.SplashAttention(),
)
_DEFAULT_IMPLEMENTATIONS = ("base",)

try:
  from tokamax._src.ops.splash_attention import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasMosaicTpuSplashAttention()
  )
  _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

IMPLEMENTATIONS: Final[
    immutabledict.immutabledict[str, base.SplashAttention[Any]]
] = immutabledict.immutabledict(_IMPLEMENTATIONS)
del _IMPLEMENTATIONS


def splash_attention(
    q: Float[Array, "*B H T D"],
    k: Float[Array, "*B h t D"] | Float[Array, "*B t D"],
    v: Float[Array, "*B h t d"] | Float[Array, "*B t d"],
    mask: base.Mask | mask_lib.Mask | Bool[Array, "*#B T t"] | None = None,
    segment_ids: base.SegmentIds | None = None,
    sinks: Float[Array, "*#B H"] | None = None,
    *,
    is_mqa: bool = False,
    mask_value: float = reference.DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    dropout_rate: float = 0.0,
    prng_key: jax.Array | None = None,
    implementation: Implementation | Sequence[Implementation] | None = None,
    q_sharding: jax.sharding.NamedSharding | None = None,
    k_sharding: jax.sharding.NamedSharding | None = None,
) -> Float[Array, "*B H T d"]:
  """Splash Attention.

  Args:
    q: Query array.
    k: Key array.
    v: Value array.
    mask: Optional attention mask (base.Mask, splash mask, boolean array, etc.).
    segment_ids: Optional document segment IDs.
    sinks: Optional attention sinks tensor.
    is_mqa: Whether to use Multi-Query Attention format.
    mask_value: Additive mask value for masked-out positions (defaults to -1e30)
    attn_logits_soft_cap: Optional logits soft cap.
    dropout_rate: Dropout probability in [0, 1).
    prng_key: PRNG key for generating dropout masks when dropout_rate > 0.
    implementation: The implementation to use ('mosaic_tpu', 'base', or None for
      automatic selection). If a sequence is passed, the first implementation
      that doesn't raise a NotImplementedError is used.
    q_sharding: NamedSharding for q. Sharding information for the other tensors
      is inferred from this if k_sharding is not provided.
    k_sharding: NamedSharding for k and v.

  Returns:
    Attention output array.

  Raises:
    ExceptionGroup: If multiple implementations are provided and all of them
      raise NotImplementedError.
  """
  if implementation is None:
    implementations: Sequence[Any] = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str) or callable(implementation):
    implementations = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")
  else:
    implementations = tuple(implementation)

  errors = []
  for impl in implementations:
    if isinstance(impl, str):
      if impl not in IMPLEMENTATIONS:
        raise ValueError(f"Unsupported implementation: {impl}")
      impl = IMPLEMENTATIONS[impl]

    try:
      return impl(
          q,
          k,
          v,
          mask=mask,
          segment_ids=segment_ids,
          sinks=sinks,
          is_mqa=is_mqa,
          mask_value=mask_value,
          attn_logits_soft_cap=attn_logits_soft_cap,
          dropout_rate=dropout_rate,
          prng_key=prng_key,
          q_sharding=q_sharding,
          k_sharding=k_sharding,
      )
    except NotImplementedError as e:
      if len(implementations) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("All implementations failed.", errors)
