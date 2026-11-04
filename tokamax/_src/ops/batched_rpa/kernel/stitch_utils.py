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
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tokamax._src.ops.batched_rpa.kernel import configs, utils


def _stitch_decode_lane(
    vmem_u32_ref: jax.Array,
    bkv_sz_cache: jax.Array,
    cache_end: jax.Array,
    new_tok_offset: jax.Array,
    bkv_sz_new: jax.Array,
    v_len: int,
    *,
    cfgs: configs.RpaConfigs,
):
    """O(1) Decode Path: Target only the VREGs containing the stitch boundary.

    The new tokens are contiguous in both the source and the destination, so a
    single lane rotation places all of them. Requires bq_sz <= num_lanes, which
    bounds the run to two VREG columns.
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    lanes_per_col = v_len // num_lanes
    strided_vmem_ref = vmem_u32_ref.reshape(-1, num_lanes)
    outer_dim = strided_vmem_ref.shape[0] // lanes_per_col
    max_col = lanes_per_col - 1

    # Destination: VREG row (`dst_chunk_idx`) and lane offset (`dst_rel`) for new
    # token insertion.
    dst_chunk_idx = bkv_sz_cache // num_lanes
    dst_rel = bkv_sz_cache % num_lanes

    # Source: first fetched new token in VMEM.
    src_tok_idx = cache_end + new_tok_offset

    lane_idx = lax.broadcasted_iota(jnp.int32, (outer_dim, num_lanes), 1)

    if cfgs.block.bq_sz == 1:
        # A single token needs neither a second destination column nor the
        # wrap-around source column: its source column is unambiguous.
        dst_vreg = strided_vmem_ref[pl.ds(dst_chunk_idx, outer_dim, lanes_per_col)]
        src_vreg = strided_vmem_ref[
            pl.ds(src_tok_idx // num_lanes, outer_dim, lanes_per_col)
        ]
        rolled = pltpu.roll(src_vreg, dst_rel - src_tok_idx % num_lanes, axis=1)
        merged_dst_vreg = lax.select(
            lane_idx == dst_rel, rolled, jnp.where(lane_idx < dst_rel, dst_vreg, 0)
        )
        return [dst_chunk_idx], outer_dim, lanes_per_col, [merged_dst_vreg]

    # Lanes below `shift` wrap in from the preceding source column.
    delta = bkv_sz_cache - src_tok_idx
    shift = lax.rem(lax.rem(delta, num_lanes) + num_lanes, num_lanes)
    src_base = (delta - shift) // num_lanes

    def rolled_src(j: int) -> jax.Array:
        col = jnp.clip(dst_chunk_idx + j - src_base - 1, 0, max_col)
        return pltpu.roll(
            strided_vmem_ref[pl.ds(col, outer_dim, lanes_per_col)], shift, axis=1
        )

    n_cols = pl.cdiv(num_lanes - 1 + cfgs.block.bq_sz, num_lanes)
    rolled = [rolled_src(j) for j in range(n_cols + 1)]

    cols = []
    merged = []
    for k in range(n_cols):
        col = jnp.minimum(dst_chunk_idx + k, max_col)
        src_vreg = jnp.where(lane_idx >= shift, rolled[k + 1], rolled[k])
        dst_vreg = strided_vmem_ref[pl.ds(col, outer_dim, lanes_per_col)]
        rel = k * num_lanes + lane_idx - dst_rel
        cols.append(col)
        merged.append(
            jnp.where(rel < 0, dst_vreg, jnp.where(rel < bkv_sz_new, src_vreg, 0))
        )

    return cols, outer_dim, lanes_per_col, merged


def _stitch_prefill_lane(
    vmem_u32_ref: jax.Array,
    bkv_sz_cache: jax.Array,
    cache_end: jax.Array,
    new_tok_offset: jax.Array,
    v_len: int,
    *,
    cfgs: configs.RpaConfigs,
):
    """O(N) Prefill Path: Roll the entire new tokens buffer into place."""
    total_head_words = (
        cfgs.model.num_kv_heads * 2 * cfgs.aligned_kv_head_dim // cfgs.serve.packing_kv
    )
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    words_per_sublane = total_head_words // num_sublanes
    vmem_u32_reshaped = vmem_u32_ref.reshape(words_per_sublane, num_sublanes, v_len)

    roll_shift = (bkv_sz_cache - (cache_end + new_tok_offset)) % v_len
    rolled_u32 = pltpu.roll(vmem_u32_reshaped[...], roll_shift, axis=2)

    lane_idx = jax.lax.broadcasted_iota(
        jnp.int32, rolled_u32[..., : cfgs.bkv_sz].shape, 2
    )
    merged_cache_u32 = jax.lax.select(
        lane_idx >= bkv_sz_cache,
        rolled_u32[..., : cfgs.bkv_sz],
        vmem_u32_reshaped[..., : cfgs.bkv_sz],
    )

    return merged_cache_u32


def store_new_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    stitch_result,
    *,
    cfgs: configs.RpaConfigs,
):
    """Stores the result of stitch_new_kv_lane back into memory."""
    v_len = cfgs.kv_vmem_lanes
    vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)

    if cfgs.block.bq_sz <= pltpu.get_tpu_info().num_lanes:
        cols, outer_dim, lanes_per_col, merged = stitch_result
        num_lanes = pltpu.get_tpu_info().num_lanes
        strided_vmem_ref = vmem_u32_ref.reshape(-1, num_lanes)

        for col, merged_vreg in zip(cols, merged):
            strided_vmem_ref[pl.ds(col, outer_dim, lanes_per_col)] = merged_vreg

    else:
        merged_cache_u32 = stitch_result
        total_head_words = (
            cfgs.model.num_kv_heads
            * 2
            * cfgs.aligned_kv_head_dim
            // cfgs.serve.packing_kv
        )
        num_sublanes = pltpu.get_tpu_info().num_sublanes
        words_per_sublane = total_head_words // num_sublanes
        vmem_u32_reshaped = vmem_u32_ref.reshape(words_per_sublane, num_sublanes, v_len)

        # Store the fully stitched sequence back.
        vmem_u32_reshaped[..., : cfgs.bkv_sz] = merged_cache_u32


def stitch_new_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    bkv_sz_frm_cache: jax.Array,
    new_kv_len_start: jax.Array,
    bkv_sz_frm_new: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
):
    """Fetches and computes stitched KV tokens (separated to avoid RAW hazards).

    Expects vmem_ref shape: [batch, 2*kv, head_dim / packing, packing,
    cfgs.kv_vmem_lanes]
    """
    bkv_sz_cache = bkv_sz_frm_cache.astype(jnp.int32)
    new_tok_offset = new_kv_len_start.astype(jnp.int32) % cfgs.kv_token_align
    cache_end = utils.align_to(bkv_sz_cache, cfgs.kv_token_align)

    vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)

    # Up to num_lanes new kv tokens span at most two 128-lane registers, so they
    # can be rolled in place instead of rolling the entire bkv_sz.
    if cfgs.block.bq_sz <= pltpu.get_tpu_info().num_lanes:
        return _stitch_decode_lane(
            vmem_u32_ref,
            bkv_sz_cache,
            cache_end,
            new_tok_offset,
            bkv_sz_frm_new.astype(jnp.int32),
            cfgs.kv_vmem_lanes,
            cfgs=cfgs,
        )
    else:
        return _stitch_prefill_lane(
            vmem_u32_ref,
            bkv_sz_cache,
            cache_end,
            new_tok_offset,
            cfgs.kv_vmem_lanes,
            cfgs=cfgs,
        )
