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
"""SparseCore Pallas scatter of MLA latents into the paged KV cache."""

import jax
from tokamax._src.ops.experimental.masked_dense_mla import kv_cache_utils
from tokamax._src.ops.experimental.masked_dense_mla import scatter_utils


def scatter(
    cache: jax.Array,
    values: jax.Array,
    spec: kv_cache_utils.SparseMLAKVCacheSpec,
    seq_lens: jax.Array | None = None,
    block_tables: jax.Array | None = None,
    query_start_loc: jax.Array | None = None,
    dst_rows: jax.Array | None = None,
) -> jax.Array:
  """Packs `values` for `spec`'s layout and scatters one row per token.

  Args:
    cache: paged KV cache in HBM, allocated as `spec.shape` / `spec.jax_dtype`.
    values: [num_tokens, unpadded head_dim] fp8 latents -- `kv_c_normed` for a
      NOPE cache, `k_pe` for a ROPE one.
    spec: layout descriptor the cache was allocated from.
    seq_lens: per-sequence KV length including this step's tokens.
    block_tables: flattened per-sequence-padded page table.
    query_start_loc: cumulative new-token counts, [num_seqs + 1].
    dst_rows: precomputed destination row per token; derived from the three
      arrays above when None.

  Returns:
    The updated cache, same shape and dtype as `cache`.
  """
  assert cache.shape == spec.shape and cache.dtype == spec.jax_dtype, (
      f"cache {cache.shape} {cache.dtype} does not match spec"
      f" {spec.shape} {spec.jax_dtype}"
  )
  if dst_rows is None:
    assert (
        seq_lens is not None
        and block_tables is not None
        and query_start_loc is not None
    ), "pass dst_rows or the addressing arrays"
    dst_rows = kv_cache_utils.get_dst_rows(
        num_tokens=values.shape[0],
        seq_lens=seq_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        page_size=spec.page_size,
    )
  if spec.layout is kv_cache_utils.KVCacheLayout.TENSORCORE:
    src = kv_cache_utils.as_token_bytes(values, spec.token_bytes)
  else:
    src = kv_cache_utils.pack_tokens(values, spec.token_bytes)
  return scatter_utils.scatter_rows(
      cache,
      src,
      dst_rows,
      name=f"scatter_{spec.cache_type.value}_{spec.layout.value}_cache",
  )
