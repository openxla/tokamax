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
"""Multi-Head Latent Attention API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.mla import base

type Implementation = Literal["xla", "mosaic", "mosaic_tpu"]

_IMPLEMENTATIONS = dict(xla=base.MultiHeadLatentAttention())
_DEFAULT_IMPLEMENTATIONS: tuple[Implementation, ...] = ("xla",)  # pylint: disable=invalid-name

try:
  from tokamax._src.ops.experimental.mla import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasTpuMultiHeadLatentAttention()
  )
  _DEFAULT_IMPLEMENTATIONS = ("mosaic",) + _DEFAULT_IMPLEMENTATIONS  # pylint: disable=invalid-name
except ImportError:
  pass

IMPLEMENTATIONS: Final[
    immutabledict.immutabledict[str, base.MultiHeadLatentAttention]
] = immutabledict.immutabledict(_IMPLEMENTATIONS)
del _IMPLEMENTATIONS


def multi_head_latent_attention(
    ql_nope: Float[Array, "H T L"],
    q_pe: Float[Array, "T H R"],
    new_kv_c: Float[Array, "T L"],
    new_k_pe: Float[Array, "T R"],
    cache_kv: Float[Array, "P K S"],
    kv_lens: Int[Array, "B"],
    page_indices: Int[Array, "N"],
    cu_q_lens: Int[Array, "B+1"],
    distribution: Int[Array, "3"],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    s_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    schedules: base.OpaqueMetadata = None,
    debug_mode: bool = False,
    implementation: (
        Implementation
        | Callable[..., Any]
        | Sequence[Implementation | Callable[..., Any]]
        | None
    ) = None,
) -> tuple[jax.Array, jax.Array]:
  """Computes Multi-Head Latent Attention.

  Args:
    ql_nope: Query latent without positional encoding, `[heads, tokens,
      lkv_dim]`.
    q_pe: Query positional encoding, `[tokens, heads, r_dim]`.
    new_kv_c: New latent KV, `[tokens, lkv_dim]`.
    new_k_pe: New key positional encoding, `[tokens, r_dim]`.
    cache_kv: Paged KV cache, `[pages, kv_dim, page_size]`.
    kv_lens: Per-sequence KV length, including new tokens.
    page_indices: Flattened `[num_seqs * pages_per_seq]` page table.
    cu_q_lens: Cumulative query lengths, `[num_seqs + 1]`.
    distribution: `[d, p, n]` splitting sequences into decode, prefill, and
      mixed bands.
    sm_scale: Softmax scale.
    sliding_window: Optional sliding window in tokens.
    soft_cap: Optional logit soft cap.
    mask_value: Optional value for masked logits.
    q_scale: Optional query dequantization scale.
    k_scale: Optional key dequantization scale.
    v_scale: Optional value dequantization scale.
    s_dtype: Logits dtype.
    schedules: Optional precomputed metadata.
    debug_mode: Whether to log debugging information.
    implementation: The implementation to use. By default, None is used, which
      selects the best available backend. If a sequence is passed, the first
      implementation that doesn't raise a NotImplementedError is used.

  Returns:
    Tuple (output, updated_cache_kv).

  Raises:
    ExceptionGroup: If multiple implementations are provided and all of them
      raise NotImplementedError.
  """
  impls: Sequence[Implementation | Callable[..., Any]]
  if implementation is None:
    impls = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str) or callable(implementation):
    impls = (implementation,)
  elif not implementation:
    raise ValueError("implementation must not be an empty sequence.")
  else:
    impls = tuple(implementation)

  errors = []
  for impl in impls:
    if isinstance(impl, str):
      if impl == "mosaic":
        impl = "mosaic_tpu"
      if impl not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation: {impl}. You may need to add a dependency"
            " on the corresponding backend."
        )
      impl_fn: Callable[..., Any] = IMPLEMENTATIONS[impl]
    else:
      impl_fn = impl

    try:
      return impl_fn(
          ql_nope,
          q_pe,
          new_kv_c,
          new_k_pe,
          cache_kv,
          kv_lens,
          page_indices,
          cu_q_lens,
          distribution,
          sm_scale=sm_scale,
          sliding_window=sliding_window,
          soft_cap=soft_cap,
          mask_value=mask_value,
          q_scale=q_scale,
          k_scale=k_scale,
          v_scale=v_scale,
          s_dtype=s_dtype,
          schedules=schedules,
          debug_mode=debug_mode,
      )
    except NotImplementedError as e:
      if len(impls) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("All implementations failed", errors)
