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

import warnings
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

DECODE_KERNEL_PREFIX: str = "RPAd"
MIXED_KERNEL_PREFIX: str = "RPAm"


_IMPLEMENTATIONS: dict[str, base.BatchedRpa] = dict(reference=base.BatchedRpa())
_MOSAIC_TPU_IMPORT_ERROR: Exception | None = None

try:
  from tokamax._src.ops.experimental.batched_rpa import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuBatchedRpa()
except Exception as e:  # pylint: disable=broad-exception-caught
  _MOSAIC_TPU_IMPORT_ERROR = e

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
    use_per_token_scale: bool = False,
    per_token_scale_dtype: jax.typing.DTypeLike | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    dynamic_k_scale: jax.Array | None = None,
    dynamic_v_scale: jax.Array | None = None,
    chunk_prefill_size: int | None = None,
    decode_block_sizes: BlockSizes | None = None,
    prefill_block_sizes: BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jax.typing.DTypeLike | None = None,
    use_causal_mask: bool = True,
    skip_kv_update: bool = False,
    update_kv_cache: bool | None = None,
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
    decode_query_size: int = 1,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: AttentionScope | str = AttentionScope.FULL,
    return_lse: bool = False,
    implementation: Implementation | None = None,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
  """Computes batched ragged paged attention using the selected implementation."""
  if update_kv_cache is not None:
    skip_kv_update = not update_kv_cache
  if implementation is None:
    if "mosaic_tpu" in _IMPLEMENTATIONS:
      op_impl = _IMPLEMENTATIONS["mosaic_tpu"]
    elif jax.default_backend() == "tpu" or any(
        d.platform == "tpu" for d in jax.devices()
    ):
      raise RuntimeError(
          "Pallas Mosaic TPU implementation for batched_ragged_paged_attention "
          f"failed to load on TPU backend: {_MOSAIC_TPU_IMPORT_ERROR}. "
          "Refusing to silently fall back to slow reference kernel."
      ) from _MOSAIC_TPU_IMPORT_ERROR
    else:
      warnings.warn(
          f"Pallas Mosaic TPU implementation not available ({_MOSAIC_TPU_IMPORT_ERROR}); "
          "falling back to reference implementation.",
          stacklevel=2,
      )
      op_impl = _IMPLEMENTATIONS["reference"]
  elif implementation in _IMPLEMENTATIONS:
    op_impl = _IMPLEMENTATIONS[implementation]
  else:
    raise ValueError(
        f"Unsupported implementation '{implementation}'. Available: {list(_IMPLEMENTATIONS.keys())}. "
        f"Import error: {_MOSAIC_TPU_IMPORT_ERROR}"
    )

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
      use_per_token_scale=use_per_token_scale,
      per_token_scale_dtype=per_token_scale_dtype,
      q_scale=q_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      dynamic_k_scale=dynamic_k_scale,
      dynamic_v_scale=dynamic_v_scale,
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
