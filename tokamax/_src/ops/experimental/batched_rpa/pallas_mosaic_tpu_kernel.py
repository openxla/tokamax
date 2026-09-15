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
"""Pallas Mosaic TPU low-level kernel wrapper for batched ragged paged attention."""

from typing import Any
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.batched_rpa.kernel import configs as rpa_configs
from tokamax._src.ops.experimental.batched_rpa.kernel import wrapper as rpa_wrapper


def batched_rpa_mosaic_tpu_kernel(
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
    decode_block_sizes: rpa_configs.BlockSizes | None = None,
    prefill_block_sizes: rpa_configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    kv_layout: rpa_configs.KVLayout = rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
    decode_query_size: int = 1,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: rpa_configs.AttentionScope = rpa_configs.AttentionScope.FULL,
    return_lse: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
  """Executes the Pallas Mosaic TPU kernel for batched ragged paged attention."""
  if decode_block_sizes is not None:
    assert (
        decode_block_sizes.bq_sz == 1
    ), f"Decode bq_sz must be 1 for decode fast-paths, got {decode_block_sizes.bq_sz}"

  return rpa_wrapper.ragged_paged_attention(
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
      out_dtype=out_dtype,
      use_causal_mask=use_causal_mask,
      q_scale=q_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      decode_block_sizes=decode_block_sizes,
      prefill_block_sizes=prefill_block_sizes,
      vmem_limit_bytes=vmem_limit_bytes,
      kv_layout=kv_layout,
      decode_query_size=decode_query_size,
      cp_group_size=cp_group_size,
      cp_rank=cp_rank,
      attention_scope=attention_scope,
      return_lse=return_lse,
  )
