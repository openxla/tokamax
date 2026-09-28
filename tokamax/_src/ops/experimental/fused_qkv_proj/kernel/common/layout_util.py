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
"""Mosaic layouts: how in-kernel values land in tiled refs.

Lane-broadcast scales: one f32 per row, stored as (rows, lanes) so that it
loads as plain vregs.
Strided row stores: view a 32-bit ref as rows of 128 lanes and write every
k-th row, a sublane-to-major relayout with no shuffles.
"""

from collections.abc import Iterable
from typing import cast

import jax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, BFloat16, Float, Float32, Shaped, UInt32  # pylint: disable=g-multiple-import,g-importing-member


def _transformable(ref: jax.Ref | hijax.TransformedRef) -> hijax.TransformedRef:
  """`ref` itself, typed as a ref with .reshape / .bitcast (a no-op cast)."""
  return cast(hijax.TransformedRef, ref)


def align(x: int, a: int) -> int:
  """x rounded up to a multiple of a."""
  return -(-x // a) * a


# Lane-broadcast scales.


def broadcast_lanes(
    col: Float32[Array, "rows 1"],
) -> Float32[Array, "rows lanes"]:
  """The stored form of a per-row scale."""
  return jnp.broadcast_to(col, (col.shape[0], pltpu.get_tpu_info().num_lanes))


def repeat_lanes(
    scale: Float32[Array, "rows lanes"], width: int
) -> Float32[Array, "rows width"]:
  """A lane-broadcast scale, repeated to `width` columns."""
  return jnp.concatenate(
      [scale] * (width // pltpu.get_tpu_info().num_lanes), axis=1
  )


def broadcast_row(
    ref: jax.Ref,
    block: int,
    num_rows: int,
    dtype: jax.typing.DTypeLike,
) -> Float[Array, "num_rows block"]:
  """Row `block` of ref, broadcast to num_rows rows.

  A stride-0 load replicates the stored row on all sublanes, one vld per vreg.
  A packed u32 row bitcasts to twice as many bf16 rows.

  Args:
    ref: (num_blocks, 1, block) u32 with a bf16 value in both halves, or f32.
    block: which stored row to load.
    num_rows: rows of the result.
    dtype: the result's dtype; bf16 bitcasts the u32 row.

  Returns:
    (num_rows, block) in dtype.
  """
  num_sublanes = pltpu.get_tpu_info().num_sublanes
  w = ref[block, pl.ds(0, num_sublanes, stride=0), :]
  if w.dtype != dtype:
    w = pltpu.bitcast(w, dtype)  # (2 * num_sublanes, block)
  return jnp.concatenate([w] * (num_rows // w.shape[0]), axis=0)


# Strided row loads and stores.


def strided_load(
    ref: jax.Ref,
    start: int,
    stride: int,
    rows: int,
    width: int | None = None,
) -> Shaped[Array, "rows width"]:
  """ref[start::stride][:rows, :width]; the load half of strided_store.

  Args:
    ref: (..., cols) 32-bit ref, viewed as rows of cols.
    start: first row.
    stride: rows between loaded rows.
    rows: rows to load.
    width: columns to load, a multiple of 128; default all of them.

  Returns:
    (rows, width), ref's dtype.
  """
  num_lanes = pltpu.get_tpu_info().num_lanes
  width = width or ref.shape[-1]
  folds = ref.shape[-1] // num_lanes
  flat = _transformable(ref).reshape(-1, num_lanes)
  chunks = [
      flat[pl.ds(start * folds + chunk, rows, stride=stride * folds)]
      for chunk in range(width // num_lanes)
  ]
  return jnp.concatenate(chunks, axis=1)


def strided_store(
    ref: jax.Ref,
    start: int,
    stride: int,
    value: Shaped[Array, "n width"],
    col: int = 0,
) -> None:
  """ref[start::stride][:n, col:col + width] = value.

  Mosaic tiles the ref's rows as (1, 128): each row's cols / 128 lane chunks
  are consecutive 128-lane rows of the flat ref. So each lane chunk of value is
  one strided row store.

  Args:
    ref: (..., cols) 32-bit ref, viewed as rows of cols.
    start: first row.
    stride: rows between stored rows.
    value: (n, width) rows to store; width is a multiple of 128.
    col: first column, a multiple of 128.
  """
  num_lanes = pltpu.get_tpu_info().num_lanes
  folds = ref.shape[-1] // num_lanes
  flat = _transformable(ref).reshape(-1, num_lanes)
  for chunk in range(value.shape[1] // num_lanes):
    rows = pl.ds(
        start * folds + col // num_lanes + chunk,
        value.shape[0],
        stride=stride * folds,
    )
    flat[rows] = value[:, chunk * num_lanes : (chunk + 1) * num_lanes]


def pack_heads(heads: list[Shaped[Array, "tile D"]]) -> UInt32[Array, "tile D"]:
  """Packs the packing (32 // bits) heads of a group into one u32 per lane.

  The heads are different query (or K/V) heads that batched_rpa stores
  together: 2 bf16 or 4 fp8 per u32. heads[i] goes in bits
  [i * bits, (i + 1) * bits), matching batched_rpa's (..., packing, D) layout
  bitcast to u32, where packing index 0 is the low half. For bf16:

    u32 lane l = [ heads[1][l] (bits 31:16) | heads[0][l] (bits 15:0) ]

  Swapping the order would make batched_rpa read head 1 as head 0.

  Args:
    heads: the group's heads, each (tile, D), all one 16- or 8-bit dtype.

  Returns:
    (tile, D) u32.
  """
  bits = jax.dtypes.itemsize_bits(heads[0].dtype)
  packed = None
  for i, y in enumerate(heads):
    y = jax.lax.bitcast_convert_type(y, jnp.dtype(f"uint{bits}"))
    y = y.astype(jnp.uint32) << (i * bits)
    packed = y if packed is None else packed | y
  assert packed is not None
  return packed


def unpack_heads(
    packed: UInt32[Array, "rows D"], dtype: jax.typing.DTypeLike = jnp.bfloat16
) -> list[Shaped[Array, "rows D"]]:
  """Unpacks the packing (32 // bits) heads in each u32; inverse of pack_heads.

  Args:
    packed: u32 lanes, each holding one value of every head in the group.
    dtype: the heads' dtype (bf16 or fp8), which sets bits.

  Returns:
    The heads, heads[i] from bits [i * bits, (i + 1) * bits).
  """
  bits = jax.dtypes.itemsize_bits(dtype)
  mask = jnp.uint32((1 << bits) - 1)
  heads = []
  for i in range(32 // bits):
    y = ((packed >> (i * bits)) & mask).astype(jnp.dtype(f"uint{bits}"))
    heads.append(jax.lax.bitcast_convert_type(y, dtype))
  return heads


def strided_load_heads(
    ref: jax.Ref,
    rows: pl.Slice,
    group: int,
    head_dim: int | None = None,
) -> UInt32[Array, "rows width"]:
  """Loads one group of packed heads from ref; inverse of strided_store_heads.

  Args:
    ref: (tile, num_groups, packing, align(D, 128)) bf16 or fp8.
    rows: the tokens to load.
    group: which group of packing heads.
    head_dim: D, to load only align(D, 128) lanes; default all lanes.

  Returns:
    (rows, width) u32, the group's heads packed as in pack_heads.
  """
  _, num_groups, _, padded_head_dim = ref.shape
  width = (
      padded_head_dim
      if head_dim is None
      else align(head_dim, pltpu.get_tpu_info().num_lanes)
  )
  u32_ref = _transformable(ref).bitcast(jnp.uint32)
  num_rows = rows.size
  assert isinstance(num_rows, int)
  return strided_load(
      u32_ref,
      start=rows.start * num_groups + group,
      stride=num_groups,
      rows=num_rows,
      width=width,
  )


def strided_store_heads(
    ref: jax.Ref,
    group: int,
    packed: UInt32[Array, "tile D"],
) -> None:
  """Stores one group of packed heads into ref.

  A group is the packing (32 // bits) heads in each u32: 2 bf16 or 4 fp8. Per
  token the groups are back to back, so group `group` is every num_groups-th
  row. D is zero-padded to the ref's lanes.

  Args:
    ref: (tile, num_groups, packing, align(D, 128)) bf16 or fp8.
    group: which group of packing heads.
    packed: the group, from pack_heads.
  """
  tile_size, num_groups, _, padded_head_dim = ref.shape
  if packed.shape[-1] < padded_head_dim:
    pad = jnp.zeros((tile_size, padded_head_dim - packed.shape[-1]), jnp.uint32)
    packed = jnp.concatenate([packed, pad], -1)
  # (tile, num_groups, 1, align(D, 128))
  u32_ref = _transformable(ref).bitcast(jnp.uint32)
  strided_store(u32_ref, start=group, stride=num_groups, value=packed)


def store_head_pair(
    ref: jax.Ref,
    rows: pl.Slice,
    pair: int,
    value: Float32[Array, "rows 2*D"],
) -> None:
  """ref[rows, 2 * pair : 2 * pair + 2] = value, heads side by side.

  Mosaic tiles (tile, H, D) bf16 with H <= 8 as (H, 128)(2, 1), one tile per
  token: heads 2g and 2g + 1 share u32 row g. So the pair packs into one u32
  array that goes to every (H / 2)-th u32 row.

  Args:
    ref: (tile, H, D) bf16, H <= 8.
    rows: the tokens to store.
    pair: which head pair (heads 2 * pair and 2 * pair + 1).
    value: (rows, 2D) f32, the two heads side by side.
  """
  _, num_heads, head_dim = ref.shape
  heads = [
      value[:, :head_dim].astype(ref.dtype),
      value[:, head_dim:].astype(ref.dtype),
  ]
  # (tile * H / 2, D)
  flat = _transformable(ref).bitcast(jnp.uint32).reshape(-1, head_dim)
  groups = num_heads // 2
  flat[pl.ds(rows.start * groups + pair, rows.size, stride=groups)] = (
      pack_heads(heads)
  )


# D = 64: two heads in one 128-lane vreg.


def pack_lane_pair(
    pair: Float[Array, "rows lanes"], head_dim: int
) -> UInt32[Array, "rows lanes"]:
  """[head_0 | head_1] in lanes 0:2D -> both packed into lanes 0:D.

  Each u32 lane l < D holds bf16 head_0[l] in its low bits and head_1[l] in
  its high bits (pack_heads order); lanes >= D are zero. Rolling head_1's
  f32 bits down by D lanes lines them up with head_0's.

  Args:
    pair: (rows, lanes), head_0 in lanes 0:D and head_1 in lanes D:2D.
    head_dim: D.

  Returns:
    (rows, lanes) u32.
  """
  bits = jax.lax.bitcast_convert_type(pair.astype(jnp.float32), jnp.uint32)
  lane = jax.lax.broadcasted_iota(jnp.int32, bits.shape, 1)
  return jnp.where(
      lane < head_dim, (bits >> 16) | pltpu.roll(bits, head_dim, 1), 0
  )


def unpack_lane_pair(
    packed: UInt32[Array, "rows lanes"], head_dim: int
) -> BFloat16[Array, "rows lanes"]:
  """Inverse of pack_lane_pair: [head_0 | head_1] in one bf16 vreg."""
  lo = (packed & jnp.uint32(0xFFFF)) << 16
  hi = pltpu.roll(packed & jnp.uint32(0xFFFF0000), head_dim, 1)
  lane = jax.lax.broadcasted_iota(jnp.int32, packed.shape, 1)
  bits = jnp.where(lane < head_dim, lo, hi)
  return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)


# batched_rpa's new-KV layouts.


def store_kv_seq_along_lane(
    kv_ref: jax.Ref,
    heads: Iterable[Shaped[Array, "D tile"]],
    head_dim: int,
    kv_packing: int,
    tile_size: int,
) -> None:
  """Stores fp8 heads [k_0, v_0, k_1, ...], each (D, tile), tokens on lanes.

  Each head has its own slab; kv_packing sublanes pack into one u32:
  (D / kv_packing, kv_packing, kv_lanes) == (D / kv_packing, kv_lanes) u32.
  A short lone tile's block runs into batched_rpa's token pad, which must be
  zero.

  Args:
    kv_ref: (2 Hkv, D / kv_packing, kv_packing, kv_lanes) fp8.
    heads: the fp8 heads in cache order, each (D, tile).
    head_dim: D.
    kv_packing: fp8 values per u32 (4).
    tile_size: tokens in the tile; lanes past it are zeroed.
  """
  kv_lanes = kv_ref.shape[-1]
  for head_index, head in enumerate(heads):
    slab = kv_ref.at[head_index].bitcast(jnp.uint32)
    slab = slab.reshape(head_dim // kv_packing, kv_lanes)
    if kv_lanes != tile_size:
      slab[...] = jnp.zeros(slab.shape, jnp.uint32)
    slab[:, pl.ds(0, tile_size)] = pltpu.bitcast(head, jnp.uint32)


def store_kv_head_along_sublane(
    kv_ref: jax.Ref,
    heads: Iterable[Shaped[Array, "tile D"]],
    kv_packing: int,
    folds: int = 1,
) -> None:
  """Stores fp8 heads [k_0, v_0, k_1, ...], each (tile, D).

  Each head is split into `folds` lane slices; slice f of head h is row
  f * len(heads) + h. kv_packing rows share each u32, so they are packed in
  groups, the last zero-padded.

  Args:
    kv_ref: (tile, groups, kv_packing, align(D / folds, 128)) fp8.
    heads: the fp8 heads in cache order, each (tile, D).
    kv_packing: fp8 values per u32 (4).
    folds: batched_rpa's lane folds per head; D / folds is 128-aligned.
  """
  heads = list(heads)
  width = heads[0].shape[-1] // folds
  heads = [
      h[:, f * width : (f + 1) * width] for f in range(folds) for h in heads
  ]
  heads += [jnp.zeros_like(heads[0])] * (-len(heads) % kv_packing)
  for group_index in range(len(heads) // kv_packing):
    group = heads[group_index * kv_packing : (group_index + 1) * kv_packing]
    strided_store_heads(kv_ref, group_index, pack_heads(group))
