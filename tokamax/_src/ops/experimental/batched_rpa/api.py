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
"""Public functional and operator API for Batched Ragged Paged Attention."""

from typing import Any, Literal, overload
import immutabledict
import jax
from tokamax._src.ops.experimental.batched_rpa import base

Implementation = Literal["mosaic_tpu", "reference"]
KVLayout = Literal["HEAD_ALONG_SUBLANE", "SEQ_ALONG_LANE"]
AttentionScope = Literal["FULL", "CACHE_ONLY", "NEW_TOKENS_ONLY"]

_IMPLEMENTATIONS: dict[str, base.BatchedRpa] = dict(reference=base.BatchedRpa())

try:
  from tokamax._src.ops.experimental.batched_rpa import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuBatchedRpa()
except ImportError:
  pass

IMPLEMENTATIONS: immutabledict.immutabledict[str, base.BatchedRpa] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)


@overload
def batched_ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    use_causal_mask: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    out_dtype: Any = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    decode_query_size: int = 1,
    skip_kv_update: bool = True,
    kv_layout: KVLayout | str | Any | None = None,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: AttentionScope | str | Any = "FULL",
    return_lse: Literal[False] = False,
    decode_block_sizes: Any | None = None,
    prefill_block_sizes: Any | None = None,
    implementation: Implementation | None = None,
) -> tuple[jax.Array, jax.Array]:
  ...


@overload
def batched_ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    use_causal_mask: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    out_dtype: Any = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    decode_query_size: int = 1,
    skip_kv_update: bool = True,
    kv_layout: KVLayout | str | Any | None = None,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: AttentionScope | str | Any = "FULL",
    return_lse: Literal[True],
    decode_block_sizes: Any | None = None,
    prefill_block_sizes: Any | None = None,
    implementation: Implementation | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  ...


@overload
def batched_ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    use_causal_mask: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    out_dtype: Any = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    decode_query_size: int = 1,
    skip_kv_update: bool = True,
    kv_layout: KVLayout | str | Any | None = None,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: AttentionScope | str | Any = "FULL",
    return_lse: bool = False,
    decode_block_sizes: Any | None = None,
    prefill_block_sizes: Any | None = None,
    implementation: Implementation | None = None,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
  ...


def batched_ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    use_causal_mask: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    out_dtype: Any = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    decode_query_size: int = 1,
    skip_kv_update: bool = True,
    kv_layout: KVLayout | str | Any | None = None,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: AttentionScope | str | Any = "FULL",
    return_lse: bool = False,
    decode_block_sizes: Any | None = None,
    prefill_block_sizes: Any | None = None,
    implementation: Implementation | None = None,
) -> Any:
  """Computes batched ragged paged attention using the selected implementation."""
  if implementation is None:
    op_impl = _IMPLEMENTATIONS.get("mosaic_tpu", _IMPLEMENTATIONS["reference"])
  else:
    op_impl = _IMPLEMENTATIONS[implementation]

  return op_impl(
      queries=queries,
      keys=keys,
      values=values,
      kv_cache=kv_cache,
      kv_lens=kv_lens,
      page_indices=page_indices,
      cu_q_lens=cu_q_lens,
      distribution=distribution,
      use_causal_mask=use_causal_mask,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      out_dtype=out_dtype,
      q_scale=q_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      decode_query_size=decode_query_size,
      skip_kv_update=skip_kv_update,
      kv_layout=kv_layout,
      cp_group_size=cp_group_size,
      cp_rank=cp_rank,
      attention_scope=attention_scope,
      return_lse=return_lse,
      decode_block_sizes=decode_block_sizes,
      prefill_block_sizes=prefill_block_sizes,
  )
