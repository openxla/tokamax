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
"""Stitch utilities for Batched MLA v3 (Transposed KV Cache).

Handles rolling and merging unpaged new KV tokens into contiguous VMEM blocks
for SEQ_ALONG_LANE memory layout.
"""

from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import utils


def _stitch_decode_lane(
    vmem_u32_ref: jax.Array,
    bkv_sz_cache: jax.Array,
    cache_end: jax.Array,
    new_tok_offset: jax.Array,
    v_len: int,
    *,
    cfgs: configs.MlaConfigs,
):
  """O(1) Decode Path: Target exactly the VREG containing the stitch boundary."""
  del v_len
  num_lanes = utils.get_tpu_num_lanes()

  # Destination: VREG chunk index and lane offset for new token insertion.
  dst_chunk_idx = bkv_sz_cache // num_lanes
  dst_rel = bkv_sz_cache % num_lanes

  # Source: VREG chunk index and lane offset of fetched token in VMEM.
  src_tok_idx = cache_end + new_tok_offset
  src_chunk_idx = src_tok_idx // num_lanes
  src_rel = src_tok_idx % num_lanes

  # Slice directly along lane axis in 3D view: [sublanes, 128]
  dst_vreg = vmem_u32_ref[
      :, pl.ds(pl.multiple_of(dst_chunk_idx * num_lanes, num_lanes), num_lanes)
  ]
  src_vreg = vmem_u32_ref[
      :, pl.ds(pl.multiple_of(src_chunk_idx * num_lanes, num_lanes), num_lanes)
  ]

  rolled_src_vreg = pltpu.roll(src_vreg, dst_rel - src_rel, axis=1)

  lane_idx = jax.lax.broadcasted_iota(jnp.int32, dst_vreg.shape, 1)
  merged_dst_vreg = jax.lax.select(
      lane_idx == dst_rel,
      rolled_src_vreg,
      jnp.where(lane_idx < dst_rel, dst_vreg, 0),
  )

  return dst_chunk_idx, merged_dst_vreg


def _stitch_prefill_lane(
    vmem_u32_ref: jax.Array,
    bkv_sz_cache: jax.Array,
    cache_end: jax.Array,
    new_tok_offset: jax.Array,
    v_len: int,
    *,
    cfgs: configs.MlaConfigs,
):
  """Prefill path: roll the staged new tokens into place at `bkv_sz_cache`."""
  vmem_u32 = vmem_u32_ref[...]
  shift = (bkv_sz_cache - (cache_end + new_tok_offset)) % v_len
  rolled = pltpu.roll(vmem_u32, shift, axis=1)[:, : cfgs.bkv_sz]
  lane = jax.lax.broadcasted_iota(jnp.int32, rolled.shape, 1)
  return jax.lax.select(
      lane >= bkv_sz_cache, rolled, vmem_u32[:, : cfgs.bkv_sz]
  )


def store_new_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    stitch_result: Any,
    *,
    cfgs: configs.MlaConfigs,
):
  """Stores the result of stitch_new_kv_lane back into memory."""
  vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)

  if cfgs.one_new_token:
    # Only the boundary tile changes. Cache tiles are fetched only up to
    # `align_to(bkv_sz_cache, tile)`, i.e. through this one, and the tile after
    # it holds the fetched new-KV tile, which is finite HBM input.
    dst_chunk_idx, merged_dst_vreg = stitch_result
    num_lanes = utils.get_tpu_num_lanes()
    chunk_slice = pl.ds(
        pl.multiple_of(dst_chunk_idx * num_lanes, num_lanes), num_lanes
    )
    vmem_u32_ref[:, chunk_slice] = merged_dst_vreg
  else:
    vmem_u32_ref[:, : cfgs.bkv_sz] = stitch_result


def stitch_new_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    bkv_sz_frm_cache: jax.Array,
    new_kv_len_start: jax.Array,
    *,
    cfgs: configs.MlaConfigs,
):
  """Fetches and computes stitched KV tokens (separated to avoid RAW hazards).

  Expects vmem_ref shape: [batch, aligned_kv_dim, cfgs.kv_vmem_lanes]
  """
  bkv_sz_cache = bkv_sz_frm_cache.astype(jnp.int32)
  # The new-KV fetch is widened down to a lane tile and lands at `cache_end`,
  # the end of the cached tiles.
  new_tok_offset = new_kv_len_start.astype(jnp.int32) % cfgs.kv_token_align
  cache_end = utils.align_to(bkv_sz_cache, cfgs.kv_token_align)

  v_len = cfgs.kv_vmem_lanes
  vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)

  if cfgs.one_new_token:
    return _stitch_decode_lane(
        vmem_u32_ref,
        bkv_sz_cache,
        cache_end,
        new_tok_offset,
        v_len,
        cfgs=cfgs,
    )
  else:
    return _stitch_prefill_lane(
        vmem_u32_ref,
        bkv_sz_cache,
        cache_end,
        new_tok_offset,
        v_len,
        cfgs=cfgs,
    )
