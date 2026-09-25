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

from typing import Any, Literal
import immutabledict
import jax
from tokamax._src.ops.experimental.batched_rpa import base
from tokamax._src.ops.experimental.batched_rpa import types

Implementation = Literal["mosaic_tpu", "reference"]
AttentionScope = types.AttentionScope
BlockSizes = types.BlockSizes
KVLayout = types.KVLayout
RpaCase = types.RpaCase

_IMPLEMENTATIONS: dict[str, base.BatchedRpa] = dict(reference=base.BatchedRpa())

try:
  from tokamax._src.ops.experimental.batched_rpa import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuBatchedRpa()
except ImportError:
  pass

IMPLEMENTATIONS: immutabledict.immutabledict[str, base.BatchedRpa] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)


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
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    decode_block_sizes: BlockSizes | None = None,
    prefill_block_sizes: BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: Any = None,
    use_causal_mask: bool = True,
    skip_kv_update: bool = False,
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
    decode_query_size: int = 1,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: AttentionScope | str = AttentionScope.FULL,
    return_lse: bool = False,
    implementation: Implementation | None = None,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
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
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      q_scale=q_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      chunk_prefill_size=chunk_prefill_size,
      decode_block_sizes=decode_block_sizes,
      prefill_block_sizes=prefill_block_sizes,
      vmem_limit_bytes=vmem_limit_bytes,
      debug_mode=debug_mode,
      out_dtype=out_dtype,
      use_causal_mask=use_causal_mask,
      skip_kv_update=skip_kv_update,
      kv_layout=kv_layout,
      decode_query_size=decode_query_size,
      cp_group_size=cp_group_size,
      cp_rank=cp_rank,
      attention_scope=attention_scope,
      return_lse=return_lse,
  )


ragged_paged_attention = batched_ragged_paged_attention
