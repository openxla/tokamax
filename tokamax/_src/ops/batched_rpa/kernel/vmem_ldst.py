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
"""VMEM <-> VREG boundary: scratch, stitching, Q/KV loads, and O/LSE stores."""

from collections.abc import Sequence
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import attention
from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import utils


class SoftmaxScratch:
  """VMEM scratch allocation and initialization for online softmax."""

  @staticmethod
  def scratch_shapes(cfgs: configs.RpaConfigs) -> tuple[Any, Any, Any]:
    dtype = cfgs.layout.softmax_dtype
    return (
        pltpu.VMEM(cfgs.lm_scratch_shape, dtype=dtype),
        pltpu.VMEM(cfgs.lm_scratch_shape, dtype=dtype),
        pltpu.VMEM(cfgs.acc_scratch_shape, dtype=dtype),
    )

  @staticmethod
  def init(
      m_ref: jax.Ref,
      l_ref: jax.Ref,
      acc_ref: jax.Ref,
      attention_sink_ref: jax.Ref | None,
      *,
      cfgs: configs.RpaConfigs,
  ) -> None:
    """Initializes `m`, `l`, and `acc` VMEM scratches before execution."""
    sink = None if attention_sink_ref is None else attention_sink_ref[...]
    m_ref[...], l_ref[...], acc_ref[...] = attention.init_softmax_values(
        sink, cfgs=cfgs
    )


def _stitch_decode_lane(
    vmem_u32_ref: Any,
    stitch_dst: jax.Array,
    stitch_src: jax.Array,
    stitch_len: jax.Array,
    v_len: int,
    *,
    cfgs: configs.RpaConfigs,
):
  """Rolls up to `num_lanes` new KV tokens in place for DECODE."""
  num_lanes = cfgs.layout.num_lanes
  lanes_per_col = v_len // num_lanes
  strided_vmem_ref = vmem_u32_ref.reshape(-1, num_lanes)
  outer_dim = strided_vmem_ref.shape[0] // lanes_per_col

  dst_chunk_idx = stitch_dst // num_lanes
  dst_rel = stitch_dst % num_lanes

  src_chunk_idx = stitch_src // num_lanes
  src_rel = stitch_src % num_lanes

  lane_idx = lax.broadcasted_iota(jnp.int32, (outer_dim, num_lanes), 1)
  shift = lax.rem(lax.rem(dst_rel - src_rel, num_lanes) + num_lanes, num_lanes)

  src_vreg = strided_vmem_ref[pl.ds(src_chunk_idx, outer_dim, lanes_per_col)]
  if cfgs.block.bq_sz > 1:
    src_next_vreg = strided_vmem_ref[
        pl.ds(src_chunk_idx + 1, outer_dim, lanes_per_col)
    ]
    src_vreg = jnp.where(lane_idx >= src_rel, src_vreg, src_next_vreg)

  src_vreg = pltpu.roll(src_vreg, shift, axis=1)

  n_cols = pl.cdiv(num_lanes - 1 + cfgs.block.bq_sz, num_lanes)
  cols = []
  merged = []
  for k in range(n_cols):
    col = dst_chunk_idx + k
    dst_vreg = strided_vmem_ref[pl.ds(col, outer_dim, lanes_per_col)]
    rel = k * num_lanes + lane_idx - dst_rel
    cols.append(col)
    merged.append(
        jnp.where(rel < 0, dst_vreg, jnp.where(rel < stitch_len, src_vreg, 0))
    )

  return cols, outer_dim, lanes_per_col, merged


def _stitch_prefill_lane(
    vmem_u32_ref: Any,
    stitch_dst: jax.Array,
    stitch_src: jax.Array,
    v_len: int,
    *,
    cfgs: configs.RpaConfigs,
):
  """Rolls the full staged new-KV buffer into `stitch_dst` for MIXED/PREFILL."""
  total_head_words = (
      cfgs.model.num_kv_heads
      * 2
      * cfgs.aligned_kv_head_dim
      // cfgs.serve.packing_kv
  )
  num_sublanes = cfgs.layout.num_sublanes
  words_per_sublane = total_head_words // num_sublanes
  vmem_u32_reshaped = vmem_u32_ref.reshape(
      words_per_sublane, num_sublanes, v_len
  )

  roll_shift = (stitch_dst - stitch_src) % v_len
  rolled_u32 = pltpu.roll(vmem_u32_reshaped[...], roll_shift, axis=2)

  lane_idx = jax.lax.broadcasted_iota(
      jnp.int32, rolled_u32[..., : cfgs.bkv_sz].shape, 2
  )
  return jax.lax.select(
      lane_idx >= stitch_dst,
      rolled_u32[..., : cfgs.bkv_sz],
      vmem_u32_reshaped[..., : cfgs.bkv_sz],
  )


def _store_stitched_lane(
    vmem_ref: Any,
    b_idx: int,
    stitch_result,
    *,
    cfgs: configs.RpaConfigs,
) -> None:
  """Writes the stitched KV VREG result back to VMEM for batch lane `b_idx`."""
  v_len = cfgs.kv_vmem_lanes
  vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)
  num_lanes = cfgs.layout.num_lanes

  if cfgs.mode == configs.RpaCase.DECODE and cfgs.block.bq_sz <= num_lanes:
    cols, outer_dim, lanes_per_col, merged = stitch_result
    strided_vmem_ref = vmem_u32_ref.reshape(-1, num_lanes)
    for col, merged_vreg in zip(cols, merged):
      strided_vmem_ref[pl.ds(col, outer_dim, lanes_per_col)] = merged_vreg
  else:
    total_head_words = (
        cfgs.model.num_kv_heads
        * 2
        * cfgs.aligned_kv_head_dim
        // cfgs.serve.packing_kv
    )
    num_sublanes = cfgs.layout.num_sublanes
    words_per_sublane = total_head_words // num_sublanes
    vmem_u32_reshaped = vmem_u32_ref.reshape(
        words_per_sublane, num_sublanes, v_len
    )
    vmem_u32_reshaped[..., : cfgs.bkv_sz] = stitch_result


def stitch_kv_along_lane(
    kv_vref: Any,
    stitch_dst: Sequence[jax.Array],
    stitch_src: Sequence[jax.Array],
    stitch_len: Sequence[jax.Array],
    *,
    cfgs: configs.RpaConfigs,
) -> None:
  """Stitches staged new KV tokens into the cached prefix across all batch lanes.

  Loads and rolls all batch lanes before storing any result back to `kv_vref` to
  avoid VMEM RAW hazards across batch slots.

  Args:
    kv_vref: VMEM KV slot ref `[B, 2 * H_kv, D // pack, pack, kv_vmem_lanes]`.
    stitch_dst: Per-lane destination token index (`bkv_sz_cache`).
    stitch_src: Per-lane staged new-KV source token index in `kv_vref`.
    stitch_len: Per-lane count of valid new KV tokens to merge.
    cfgs: Kernel configuration.
  """
  v_len = cfgs.kv_vmem_lanes
  num_lanes = cfgs.layout.num_lanes
  is_decode_small_q = (
      cfgs.mode == configs.RpaCase.DECODE and cfgs.block.bq_sz <= num_lanes
  )

  stitch_results = []
  for b_idx in range(cfgs.batch_size):
    vmem_u32_ref = kv_vref.at[b_idx].bitcast(jnp.uint32)
    if is_decode_small_q:
      res = _stitch_decode_lane(
          vmem_u32_ref,
          stitch_dst[b_idx],
          stitch_src[b_idx],
          stitch_len[b_idx],
          v_len,
          cfgs=cfgs,
      )
    else:
      res = _stitch_prefill_lane(
          vmem_u32_ref,
          stitch_dst[b_idx],
          stitch_src[b_idx],
          v_len,
          cfgs=cfgs,
      )
    stitch_results.append(res)

  for b_idx in range(cfgs.batch_size):
    _store_stitched_lane(kv_vref, b_idx, stitch_results[b_idx], cfgs=cfgs)


def load_q_tile(q_vref: Any, *, cfgs: configs.RpaConfigs) -> jax.Array:
  """Loads Q from VMEM into VREGs `[B, H_kv, bq_sz * G_aligned, D_compute]`."""
  q_p = cfgs.aligned_num_q_heads_per_kv_head // cfgs.serve.packing_q
  q_ref = q_vref.bitcast(jnp.uint32).reshape(-1, cfgs.aligned_q_head_dim)
  q_loaded = utils.strided_load(
      q_ref,
      0,
      cfgs.batch_size * cfgs.model.num_kv_heads * cfgs.bq_sz * q_p,
      1,
      dtype=cfgs.serve.dtype_q,
  )
  q = q_loaded.reshape(
      cfgs.batch_size,
      cfgs.model.num_kv_heads,
      cfgs.bq_sz * cfgs.aligned_num_q_heads_per_kv_head,
      cfgs.aligned_q_head_dim,
  )
  return q[..., : cfgs.compute_kv_head_dim]


def _extract_per_token_scales(
    k: jax.Array,
    v: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array | float | None,
    jax.Array | float | None,
]:
  """Slices trailing per-token scale channels from `SEQ_ALONG_LANE` K and V."""
  if not cfgs.serve.per_token_scale:
    return k, v, cfgs.serve.scale_k, cfgs.serve.scale_v

  scale_dtype = cfgs.serve.per_token_scale_dtype
  assert scale_dtype is not None
  b, num_heads = k.shape[0], k.shape[1]
  # Use unpacked `k.dtype` (e.g. fp4 after uint8 bitcast) to count channels.
  scale_bits = jax.dtypes.itemsize_bits(scale_dtype)
  kv_bits = jax.dtypes.itemsize_bits(k.dtype)
  scale_channels = max(1, scale_bits // kv_bits)

  k_scale_slice = k[
      :, :, cfgs.model.head_dim : cfgs.model.head_dim + scale_channels, :
  ]
  v_scale_slice = v[
      :, :, cfgs.model.head_dim : cfgs.model.head_dim + scale_channels, :
  ]

  if scale_channels > 1:
    k_scale = pltpu.bitcast(k_scale_slice, scale_dtype)
    v_scale = pltpu.bitcast(v_scale_slice, scale_dtype)
  else:
    k_scale = k_scale_slice.astype(scale_dtype)
    v_scale = v_scale_slice.astype(scale_dtype)

  k_scale = k_scale.reshape(b, num_heads, cfgs.bkv_sz).astype(jnp.float32)
  k_scale = k_scale[:, :, jnp.newaxis, :]
  v_scale = v_scale.reshape(b, num_heads, cfgs.bkv_sz)

  k = k[:, :, : cfgs.compute_kv_head_dim, :]
  v = v[:, :, : cfgs.compute_kv_head_dim, :]
  return k, v, k_scale, v_scale


def load_kv_seq_along_lane(
    kv_in_vref: Any, *, cfgs: configs.RpaConfigs
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array | float | None,
    jax.Array | float | None,
]:
  """Loads `SEQ_ALONG_LANE` KV into `[B, H_kv, D_compute, bkv_sz]` + scales."""
  k_b = []
  v_b = []
  pack_dim = cfgs.aligned_kv_head_dim // cfgs.serve.packing_kv
  target_shape = (-1, cfgs.kv_vmem_lanes)

  for b_idx in range(cfgs.batch_size):
    ks = []
    vs = []
    for kv_head in range(cfgs.model.num_kv_heads):
      k_head_ref = (
          kv_in_vref.at[b_idx, kv_head * 2]
          .bitcast(jnp.uint32)
          .reshape(target_shape)
      )
      v_head_ref = (
          kv_in_vref.at[b_idx, kv_head * 2 + 1]
          .bitcast(jnp.uint32)
          .reshape(target_shape)
      )

      k_head_u32 = utils.strided_load(k_head_ref, 0, pack_dim, 1)
      v_head_u32 = utils.strided_load(v_head_ref, 0, pack_dim, 1)

      if cfgs.serve.dtype_kv == jnp.uint8:
        fp4 = jnp.float4_e2m1fn
        k_head = pltpu.bitcast(k_head_u32, fp4)[:, : cfgs.bkv_sz]
        v_head = pltpu.bitcast(v_head_u32, fp4)[:, : cfgs.bkv_sz]
        ks.append(k_head)
        vs.append(v_head)
      else:
        k_head = pltpu.bitcast(k_head_u32, cfgs.serve.dtype_kv)[
            :, : cfgs.bkv_sz
        ]
        v_head = pltpu.bitcast(v_head_u32, cfgs.serve.dtype_kv)[
            :, : cfgs.bkv_sz
        ]
        ks.append(k_head.reshape(cfgs.aligned_kv_head_dim, cfgs.bkv_sz))
        vs.append(v_head.reshape(cfgs.aligned_kv_head_dim, cfgs.bkv_sz))

    k_b.append(jnp.stack(ks, axis=0))
    v_b.append(jnp.stack(vs, axis=0))

  k = jnp.stack(k_b, axis=0)
  v = jnp.stack(v_b, axis=0)
  return _extract_per_token_scales(k, v, cfgs=cfgs)


def _strided_load_bkv(
    kv_in_vref: Any,
    b_idx: int,
    start: int,
    *,
    cfgs: configs.RpaConfigs,
) -> list[tuple[jax.Array, jax.Array]]:
  """Loads interleaved `(k, v)` head pairs from `HEAD_ALONG_SUBLANE` VMEM."""
  assert start % cfgs.serve.packing_kv == 0
  start //= cfgs.serve.packing_kv
  kv_u32_ref = kv_in_vref.at[b_idx].bitcast(jnp.uint32)
  kv_ref = kv_u32_ref.reshape(-1, cfgs.aligned_kv_head_dim)
  kv_ref = pltpu.annotate(kv_ref, no_hazard_no_deps=True)

  if cfgs.serve.packing_kv == 1:
    k = utils.strided_load(
        kv_ref,
        start,
        cfgs.bkv_sz * cfgs.bkv_stride,
        cfgs.bkv_stride,
        dtype=cfgs.serve.dtype_kv,
    )
    v = utils.strided_load(
        kv_ref,
        start + 1,
        cfgs.bkv_sz * cfgs.bkv_stride,
        cfgs.bkv_stride,
        dtype=cfgs.serve.dtype_kv,
    )
    return [(k, v)]

  kv = utils.strided_load(
      kv_ref, start, cfgs.bkv_sz * cfgs.bkv_stride, cfgs.bkv_stride
  )
  bitwidth = jax.dtypes.itemsize_bits(cfgs.serve.dtype_kv)
  return utils.convert_to_target_bitwidth(
      kv, target_bitwidth=bitwidth, kv_dtype=cfgs.serve.dtype_kv
  )


def load_kv_head_along_sublane(
    kv_in_vref: Any, *, cfgs: configs.RpaConfigs
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array | float | None,
    jax.Array | float | None,
]:
  """Loads `HEAD_ALONG_SUBLANE` KV into `[B, H_kv, bkv_sz, D_compute]`."""
  k_b = []
  v_b = []
  heads_per_load = pl.cdiv(cfgs.serve.packing_kv, 2)
  num_kv_heads = cfgs.model.num_kv_heads
  num_kv_chunks = num_kv_heads * cfgs.num_lane_folds_in_head_dim

  for b_idx in range(cfgs.batch_size):
    ks = []
    vs = []
    for kv_chunk_start in range(0, num_kv_chunks, heads_per_load):
      bkv_lst = _strided_load_bkv(
          kv_in_vref, b_idx, kv_chunk_start * 2, cfgs=cfgs
      )
      ks.append(jnp.stack([k for k, _ in bkv_lst], axis=0))
      vs.append(jnp.stack([v for _, v in bkv_lst], axis=0))
    k = jnp.concat(ks, axis=0).reshape(
        -1, cfgs.bkv_sz, cfgs.aligned_kv_head_dim
    )
    v = jnp.concat(vs, axis=0).reshape(
        -1, cfgs.bkv_sz, cfgs.aligned_kv_head_dim
    )
    k = k[:num_kv_chunks]
    v = v[:num_kv_chunks]
    if cfgs.num_lane_folds_in_head_dim > 1:
      chunk_starts = range(0, num_kv_chunks, num_kv_heads)
      k = jnp.concat([k[i : i + num_kv_heads] for i in chunk_starts], axis=-1)
      v = jnp.concat([v[i : i + num_kv_heads] for i in chunk_starts], axis=-1)
    k_b.append(k)
    v_b.append(v)

  k = jnp.stack(k_b, axis=0)
  v = jnp.stack(v_b, axis=0)
  return k, v, cfgs.serve.scale_k, cfgs.serve.scale_v


def load_kv_tile(kv_in_vref: Any, *, cfgs: configs.RpaConfigs) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array | float | None,
    jax.Array | float | None,
]:
  """Dispatches KV VMEM -> VREG load to the active physical layout helper."""
  if cfgs.is_seq_along_lane:
    return load_kv_seq_along_lane(kv_in_vref, cfgs=cfgs)
  return load_kv_head_along_sublane(kv_in_vref, cfgs=cfgs)


def store_out_lane(
    o_vref: Any,
    b_idx: int | jax.Array,
    out: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
) -> None:
  """Pads and stores normalized output `out` into `o_vref.at[b_idx]`."""
  o_u32_vref = o_vref.at[b_idx].bitcast(jnp.uint32)
  out_ref = o_u32_vref.reshape(-1, cfgs.aligned_q_head_dim)
  pad_width = [[0, 0] for _ in range(out.ndim)]
  pad_width[-1][-1] = cfgs.aligned_q_head_dim - cfgs.compute_kv_head_dim
  out_padded = jnp.pad(out, pad_width, constant_values=0)
  out_u32 = pltpu.bitcast(out_padded, out_ref.dtype).reshape(out_ref.shape)
  utils.strided_store(out_ref, 0, out_ref.shape[0], 1, out_u32)


def store_lse_lane(
    lse_vref: Any,
    b_idx: int | jax.Array,
    lse_val: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
) -> None:
  """Bitcasts and stores per-token LSE `lse_val` into `lse_vref.at[b_idx]`."""
  lse_u32 = lse_vref.at[b_idx].bitcast(jnp.uint32)
  lse_ref = lse_u32.reshape(-1, lse_vref.shape[-1])
  lse_packed = pltpu.bitcast(
      lse_val.astype(cfgs.layout.softmax_dtype), jnp.uint32
  ).reshape(lse_ref.shape)
  utils.strided_store(lse_ref, 0, lse_ref.shape[0], 1, lse_packed)
