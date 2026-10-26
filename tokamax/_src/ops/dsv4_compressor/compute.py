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
"""In-kernel compute helpers of the compress-and-store Pallas kernel."""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.dsv4_compressor import csa_cache_layout


def interleaved_rope_vector(x, cos_val_32, sin_val_32):
  """Applies interleaved Rotary Position Embedding (RoPE) to a vector."""
  # x: (tile_n, 1, 128)
  # cos_val_32: (tile_n, 1, 32)
  # sin_val_32: (tile_n, 1, 32)
  tile_n = x.shape[0]

  # We work with 2D tensors for gather to keep dimensions simple.
  x_2d = jnp.squeeze(x, axis=1)  # (tile_n, 128)

  # swap adjacent pairs: [1, 0, 3, 2, ...]
  iota = jnp.arange(128)
  swap_indices = jnp.bitwise_xor(iota, 1)  # (128,)
  swap_coords = jnp.broadcast_to(swap_indices, (tile_n, 128))[
      :, :, None
  ]  # (tile_n, 128, 1)

  gather_dn = jax.lax.GatherDimensionNumbers(
      offset_dims=(),
      collapsed_slice_dims=(1,),
      start_index_map=(1,),
      operand_batching_dims=(0,),
      start_indices_batching_dims=(0,),
  )

  x_swapped_2d = jax.lax.gather(
      x_2d,
      swap_coords,
      dimension_numbers=gather_dn,
      slice_sizes=(1, 1),
      unique_indices=True,
      mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )

  # cos_val_32/sin_val_32 are (tile_n, 1, 32). Squeeze to (tile_n, 32)
  cos_32 = jnp.squeeze(cos_val_32, axis=1).astype(x.dtype)
  sin_32 = jnp.squeeze(sin_val_32, axis=1).astype(x.dtype)

  ones_32 = jnp.ones((tile_n, 32), dtype=x.dtype)
  zeros_32 = jnp.zeros((tile_n, 32), dtype=x.dtype)

  # pad to pairs: (tile_n, 64)
  cos_pairs = jnp.concatenate([ones_32, cos_32], axis=-1)
  sin_pairs = jnp.concatenate([zeros_32, sin_32], axis=-1)

  # duplicate indices: [0, 0, 1, 1, 2, 2, ..., 63, 63]
  dup_indices = iota // 2
  dup_coords = jnp.broadcast_to(dup_indices, (tile_n, 128))[:, :, None]

  cos_dup_2d = jax.lax.gather(
      cos_pairs,
      dup_coords,
      dimension_numbers=gather_dn,
      slice_sizes=(1, 1),
      unique_indices=False,
      mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )

  sin_dup_2d = jax.lax.gather(
      sin_pairs,
      dup_coords,
      dimension_numbers=gather_dn,
      slice_sizes=(1, 1),
      unique_indices=False,
      mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )

  # alternate sin: [-1, 1, -1, 1, ...]
  alt_mask = ((iota % 2) * 2 - 1).astype(x.dtype)[None, :]  # (1, 128)
  sin_alt_2d = sin_dup_2d * alt_mask

  out_2d = x_2d * cos_dup_2d + x_swapped_2d * sin_alt_2d

  return out_2d[:, None, :]


def quantize_fp8_tiled(x, block_size):
  """Block fp8 quantization with e8m0 scales over `block_size` lanes."""
  # x: (tile_n, S, 128) f32
  fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
  _, _, width = x.shape  # (tile_n, S, 128) f32
  num_blocks = width // block_size

  qs = []
  scales = []
  for b in range(num_blocks):
    start = b * block_size
    end = start + block_size
    x_block = x[:, :, start:end]  # (tile_n, S, block_size) f32

    amax = jnp.clip(
        jnp.max(jnp.abs(x_block), axis=-1, keepdims=True), 1e-4, None
    )  # (tile_n, S, 1) f32

    log2_val = jnp.log2(amax / fp8_max)  # f32
    scale = jnp.exp2(jnp.ceil(log2_val))  # (tile_n, S, 1) f32

    q_block = (x_block * (1.0 / scale)).astype(
        jnp.float8_e4m3fn
    )  # (tile_n, S, block_size) fp8

    qs.append(q_block)
    scales.append(scale)

  q = jnp.concatenate(qs, axis=-1)  # (tile_n, S, 128) fp8
  scale = jnp.concatenate(scales, axis=-1)  # (tile_n, S, num_blocks) f32

  # Bitcast workaround directly on f32 to extract exponent
  # f32 exponent is at bits 23-30. Shift right by 23 to align it.
  scale_u32 = pltpu.bitcast(scale, jnp.uint32)
  scale_exp = scale_u32 >> 23
  scale_u8 = scale_exp.astype(jnp.uint8)
  scale_f8 = pltpu.bitcast(scale_u8, jnp.float8_e8m0fnu)

  return q, scale_f8


def quantize_fp8_lane_periodic(x, period, nope_dim):
  """Quantize with one e8m0 scale per lane of a `period`-lane cycle.

  Group j is `{d < nope_dim : d % period == j}`, so the scale a stored
  value needs is a function of its lane alone. The attention kernel then
  reads the scales as a tile it can rotate into place instead of expanding
  them with a one-hot matmul.

  Args:
    x: (tile_n, S, 128) f32, of which the first `nope_dim` flattened dims are
      the ones the record keeps.
    period: The lane period of the scale groups.
    nope_dim: The number of leading dims that the record keeps.

  Returns:
    (tile_n, S, 128) fp8 values and (tile_n, 1, period) e8m0 scales.
  """
  fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
  _, _, width = x.shape
  assert width == 2 * period, (width, period)

  dim = jax.lax.broadcasted_iota(
      jnp.int32, x.shape, 1
  ) * width + jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
  # Dims past the record's end are dropped by the packer; keeping them out
  # of the amax stops them widening a group they are not stored in.
  a = jnp.where(dim < nope_dim, jnp.abs(x), 0.0)
  folded = jnp.maximum(a[:, :, :period], a[:, :, period:])
  amax = jnp.clip(jnp.max(folded, axis=1, keepdims=True), 1e-4, None)
  scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))  # (tile_n, 1, period)

  q = (x * (1.0 / jnp.concatenate([scale, scale], axis=-1))).astype(
      jnp.float8_e4m3fn
  )
  scale_u8 = (pltpu.bitcast(scale, jnp.uint32) >> 23).astype(jnp.uint8)
  return q, pltpu.bitcast(scale_u8, jnp.float8_e8m0fnu)


def pack_nope_lane_periodic(
    q, scale, nope_dim, nope_width_bytes, last_dim_size
):
  """Pack `nope_dim` fp8 values followed by the e8m0 scales."""
  tile_n = q.shape[0]
  q_nope = pltpu.bitcast(q, jnp.uint8).reshape(tile_n, -1)[:, :nope_dim]
  scale_bytes = pltpu.bitcast(scale, jnp.uint8).reshape(tile_n, -1)
  record = jnp.concatenate([q_nope, scale_bytes], axis=1)
  assert record.shape[1] == nope_width_bytes, record.shape
  return record.reshape(tile_n, -1, last_dim_size)


def pack_nope_tiled(
    q, scale, nope_dim, block_size, nope_width_bytes=512, last_dim_size=128
):
  """Pack `nope_dim` fp8 values, their block scales and zero padding."""
  # q: (tile_n, S, 128) fp8
  # scale: (tile_n, S, num_blocks) e8m0 (uint8 bitcasted)
  tile_n, _, _ = q.shape

  # Bitcast to uint8
  q_bytes = pltpu.bitcast(q, jnp.uint8)
  scale_bytes = pltpu.bitcast(scale, jnp.uint8)

  # Flat representations
  q_flat = q_bytes.reshape(tile_n, -1)  # (tile_n, S * 128)
  scale_flat = scale_bytes.reshape(tile_n, -1)  # (tile_n, S * num_blocks)

  # Select NOPE parts
  q_nope = q_flat[:, :nope_dim] if nope_dim < q_flat.shape[1] else q_flat

  nope_blocks = (nope_dim + block_size - 1) // block_size
  if nope_blocks < scale_flat.shape[1]:
    scale_nope = scale_flat[:, :nope_blocks]
  else:
    scale_nope = scale_flat

  # Pad with zeros
  pad_size = nope_width_bytes - (nope_dim + nope_blocks)
  zeros = jnp.zeros((tile_n, pad_size), dtype=jnp.uint8)

  nope_record_padded = jnp.concatenate(
      [q_nope, scale_nope, zeros], axis=1
  )  # (tile_n, 512)
  return nope_record_padded.reshape(tile_n, -1, last_dim_size)


def rope_words_tiled(rope_slot_ropped, rope_head_dim):
  """RoPE channels as cache words: low half bf16 `k`, high half `k + 32`.

  Args:
    rope_slot_ropped: (tile_n, 1, 128) f32 holding the channels in its last
      `rope_head_dim` lanes.
    rope_head_dim: The RoPE dimension.

  Returns:
    (tile_n, 1, 128) int32 with word `k` at lane `128 - rope_head_dim + k`;
    the lanes around it are junk.
  """
  bits = pltpu.bitcast(rope_slot_ropped.astype(jnp.bfloat16), jnp.uint16)
  bits = bits.astype(jnp.int32)
  half = rope_head_dim // 2
  # Lane c picks up channel c + half as its high half.
  return bits | (pltpu.roll(bits, bits.shape[-1] - half, 2) << 16)


def merge_rope_row(
    row,  # (1, 128) int32, the RoPE row as read from HBM
    kv_slots,  # (tile_n,) int, token index of every token in the tile
    words,  # rope_words_tiled(...)
    rope_lane,  # int, lane of RoPE word 0 in `words`
    n,  # int, the token whose row this is
):
  """Writes every tile token whose RoPE lands in token n's row into it.

  Token `j` of a row owns words `[32 j, 32 j + 32)` (see
  `csa_cache_layout`), so one rotation puts a token's words in its window.

  Args:
    row: (1, 128) int32, the RoPE row as read from HBM.
    kv_slots: (tile_n,) int, token index of every token in the tile.
    words: The `rope_words_tiled` words of every token in the tile.
    rope_lane: Lane of RoPE word 0 in `words`.
    n: The token whose row this is.

  Returns:
    The merged (1, 128) int32 row.
  """
  per_row = csa_cache_layout.ROPE_TOKENS_PER_ROW
  width = csa_cache_layout.ROPE_WORDS
  lanes = row.shape[-1]
  window = jax.lax.broadcasted_iota(jnp.int32, row.shape, 1) // width
  curr_slot = kv_slots[n]
  curr_row = curr_slot // per_row

  for i in range(kv_slots.shape[0]):
    slot_i = kv_slots[i]
    hit = (slot_i >= 0) & (curr_slot >= 0) & (slot_i // per_row == curr_row)
    pos_i = slot_i % per_row
    for j in range(per_row):
      placed = pltpu.roll(words[i], (j * width - rope_lane) % lanes, 1)
      row = jnp.where(hit & (pos_i == j) & (window == j), placed, row)
  return row


def merge_slot_updates(
    slots_val,  # (pack_factor, physical_slot_size) uint8 (the row)
    kv_slots,  # (tile_n,) int (all slots in tile)
    val_padded,  # (tile_n, record_subslots, 128) uint8 (all values in tile)
    n,  # int (current token index)
):
  """Merges every tile token's record that lands in token n's row into it.

  Multiple KV slots are packed into a single physical 512-byte row in HBM, so
  different slots of a tile may map to the same row. Only the first token of
  the tile that updates a physical row (per `is_first_mask`) sends its DMA.

  In this function, we:
  1. Read the current value of `slot_val`
  2. Scan all other tokens in the current tile
  3. Consolidate all other updates into the single tile.

  Args:
    slots_val: (pack_factor, physical_slot_size) uint8, the row.
    kv_slots: (tile_n,) int, all slots in the tile.
    val_padded: (tile_n, record_subslots, 128) uint8, all values in the tile.
    n: The current token index.

  Returns:
    The merged (pack_factor, physical_slot_size) uint8 row.
  """
  tile_n = kv_slots.shape[0]
  curr_slot = kv_slots[n]
  pack_factor = slots_val.shape[0]

  curr_row = curr_slot // pack_factor

  slots_list = [slots_val[i] for i in range(pack_factor)]

  for i in range(tile_n):
    slot_i = kv_slots[i]
    valid_i = slot_i >= 0
    row_i = slot_i // pack_factor
    sub_idx = slot_i % pack_factor

    update_cond = valid_i & (row_i == curr_row) & (curr_slot >= 0)

    val = val_padded[i].flatten()

    for target_s in range(pack_factor):
      slots_list[target_s] = jax.lax.select(
          update_cond & (sub_idx == target_s),
          val,
          slots_list[target_s],
      )

  return jnp.stack(slots_list)


def gather_from_page_buffer(
    page_buffer,
    positions_ref,
    kv_window_u8,
    score_window_u8,
    *,
    global_idx: int | jax.Array,
    num_tokens: int,
    window: int,
    block_size: int,
    pages_to_buffer_per_token: int,
    field_rows: int,
    state_rows_per_token: int,
    overlap: bool,
    is_indexer: bool = False,
):
  """Extracts kv_window, score_window from page buffer."""
  tile_n = page_buffer.shape[0]
  if is_indexer:
    # CSA_INDEXER (layout 32x4x256)
    # page_buffer shape: (tile_n, pages_to_buffer, 32, 4, 256)
    for n in range(tile_n):
      safe_idx = jnp.minimum(global_idx + n, num_tokens - 1)
      pos = positions_ref[safe_idx]

      block_idx_curr = pos // block_size
      pos_start = pos - window + 1

      for w in range(window):
        pos_w = pos_start + w
        block_idx_w = pos_w // block_size
        p = block_idx_w - block_idx_curr + pages_to_buffer_per_token - 1

        offset_in_block = pos_w % block_size
        token_row_start = offset_in_block * 2

        kv_row = token_row_start + 0
        score_row = token_row_start + 1

        valid_w = pos_w >= 0

        @pl.when(valid_w)
        def _():
          row_kv = page_buffer[n, p, kv_row][...]  # shape (4, 256)
          row_score = page_buffer[n, p, score_row][...]  # shape (4, 256)

          # `proj_and_save_state` stores the f32 state byte-transposed
          # (u8[s, l] == byte s of f32 l), so one (4, 256) row holds
          # 256 floats across its LANES. The prev/curr halves of the
          # overlapping state are therefore lanes 0:128 and 128:256.
          if overlap:
            is_prev = w < (window // 2)
            val_kv = jax.lax.select(
                is_prev,
                row_kv[:, 0:128],
                row_kv[:, 128:256],
            )
            val_score = jax.lax.select(
                is_prev,
                row_score[:, 0:128],
                row_score[:, 128:256],
            )
          else:
            val_kv = row_kv[:, 0:128]
            val_score = row_score[:, 0:128]

          kv_window_u8[n, w, 0, :, :] = val_kv
          score_window_u8[n, w, 0, :, :] = val_score

  else:
    # Standard HCA/CSA (last dim 128)
    slots_per_part_head = kv_window_u8.shape[2]

    for n in range(tile_n):
      safe_idx = jnp.minimum(global_idx + n, num_tokens - 1)
      pos = positions_ref[safe_idx]

      block_idx_curr = pos // block_size
      pos_start = pos - window + 1

      @pl.loop(0, window)
      def body_w(w):
        pos_w = pos_start + w
        block_idx_w = pos_w // block_size
        p = block_idx_w - block_idx_curr + pages_to_buffer_per_token - 1

        slots_per_part_row = field_rows
        slots_per_token_row = state_rows_per_token
        if overlap:
          is_prev = w < (window // 2)
          kv_slot_start_row = jax.lax.select(is_prev, 0, slots_per_part_row)
          score_slot_start_row = jax.lax.select(
              is_prev, 2 * slots_per_part_row, 3 * slots_per_part_row
          )
        else:
          kv_slot_start_row = 0
          score_slot_start_row = slots_per_part_row

        offset_in_block = pos_w % block_size

        @pl.loop(0, slots_per_part_head, unroll=True)
        def gather_loop(d_idx):
          kv_src_row = (
              offset_in_block * slots_per_token_row + kv_slot_start_row + d_idx
          )
          score_src_row = (
              offset_in_block * slots_per_token_row
              + score_slot_start_row
              + d_idx
          )

          kv_window_u8[n, w, d_idx, :, :] = page_buffer[n, p, kv_src_row, :, :]
          score_window_u8[n, w, d_idx, :, :] = page_buffer[
              n, p, score_src_row, :, :
          ]


def stage_sal_columns(
    records_t_ref,  # VMEM (33, 4, padded_tokens) u8
    records: jax.Array,  # (tile_n, 256) u8
    first_token,  # scalar int: tile's starting token index (multiple of tile_n)
) -> None:
  """Transposes `records[:, :129]` into columns `[first_token, first_token + tile_n)`.

  The 132-byte SAL record of token `k` is `[128 fp8 | 1 scale | 3 zero bytes]`,
  stored down column `first_token + k` of `records_t_ref` reshaped as
  `(132, padded_tokens)`. Because `tile_n` is smaller than the 8-row sublane
  tile, `jnp.swapaxes` is not supported on `(tile_n, 256)`; instead, two
  one-hot bf16 matmuls transpose the `(tile_n, 256)` bytes into `(132, 128)`
  and place them at lanes `[col, col + tile_n)` of the 128-lane tile containing
  `first_token`. Byte values `0..255` are exact in bf16 (7-bit significand),
  and each one-hot dot has a single non-zero term, so the matmul is bit-exact.

  Args:
    records_t_ref: VMEM scratch buffer of shape `(33, 4, padded_tokens)`.
    records: Quantized records of shape `(tile_n, 256)`.
    first_token: Starting token index of the tile.
  """
  n, width = records.shape
  sublane_tile = 8
  lanes = 128
  rec_bytes = 129  # 128 fp8 key bytes + 1 e8m0 scale byte
  out_rows = records_t_ref.shape[0] * records_t_ref.shape[1]  # 33 * 4 = 132
  rows_pad = max(sublane_tile, pl.cdiv(n, sublane_tile) * sublane_tile)

  records_bf16 = jnp.pad(records, ((0, rows_pad - n), (0, 0))).astype(
      jnp.bfloat16
  )
  b = jax.lax.broadcasted_iota(jnp.int32, (width, out_rows), 0)
  d = jax.lax.broadcasted_iota(jnp.int32, (width, out_rows), 1)
  byte_pick = ((b == d) & (d < rec_bytes)).astype(jnp.bfloat16)
  # (rows_pad, width) @ (width, 132) -> (rows_pad, 132)
  picked = jnp.dot(
      records_bf16, byte_pick, preferred_element_type=jnp.float32
  ).astype(jnp.bfloat16)

  col = first_token % lanes
  r = jax.lax.broadcasted_iota(jnp.int32, (rows_pad, lanes), 0)
  l = jax.lax.broadcasted_iota(jnp.int32, (rows_pad, lanes), 1)
  lane_place = ((r < n) & (l == col + r)).astype(jnp.bfloat16)
  # (132, rows_pad) @ (rows_pad, 128) -> (132, 128)
  placed = (
      jnp.dot(picked.T, lane_place, preferred_element_type=jnp.float32)
      .astype(jnp.uint8)
      .reshape(records_t_ref.shape[0], records_t_ref.shape[1], lanes)
  )

  tile_off = pl.multiple_of((first_token // lanes) * lanes, lanes)
  cur = records_t_ref[:, :, pl.ds(tile_off, lanes)]
  lane = jax.lax.broadcasted_iota(jnp.int32, placed.shape, 2)
  records_t_ref[:, :, pl.ds(tile_off, lanes)] = jnp.where(
      (lane >= col) & (lane < col + n), placed, cur
  )


_SAL_KV_PACKING = 4
_SAL_LANES = 128


class SalTiles:
  """`SEQ_ALONG_LANE` granule ops: a granule is one 128-lane tile.

  The staged source (`k_ref`, `[sublane_groups, 4, padded_tokens]`) is already
  transposed to the cache layout, so token `t` is lane `t` of the source and
  cache row `page * page_size + slot` is lane `slot` of page `page`.
  """

  granule = _SAL_LANES

  def __init__(self, k_ref, cache_ref, rmw_ref, page_size: int):
    _, self.sublane_groups, _, _ = cache_ref.shape
    self.cache = cache_ref
    self.k = k_ref
    self.rmw = rmw_ref
    self.page_size = page_size
    self.padded_tokens = k_ref.shape[2]
    self.rec_dim = self.sublane_groups * _SAL_KV_PACKING

  def _cache_lanes(self, row, n):
    """Lanes `[row, row + n)` of the cache; `row` is 128-aligned."""
    slot = pl.multiple_of(row % self.page_size, _SAL_LANES)
    return self.cache.at[row // self.page_size, :, :, pl.ds(slot, n)]

  def _src_tile(self, src_ref, tok_aligned):
    off = pl.multiple_of(
        jnp.clip(tok_aligned, 0, self.padded_tokens - _SAL_LANES), _SAL_LANES
    )
    return pltpu.bitcast(
        src_ref[:, :, pl.ds(off, _SAL_LANES)].reshape(self.rec_dim, _SAL_LANES),
        jnp.uint32,
    )

  def granule_copy(self, row, rmw_slot, sem, *, to_hbm: bool):
    hbm = self._cache_lanes((row >> 7) << 7, _SAL_LANES)
    vmem = self.rmw.at[rmw_slot]
    src, dst = (vmem, hbm) if to_hbm else (hbm, vmem)
    return pltpu.make_async_copy(src, dst, sem)

  def blend(self, rmw_slot, row, src_tok, n):
    """Overwrites lanes `[row, row + n) & 127` of RMW tile `rmw_slot`."""
    dst_start = row & 127
    src_start = src_tok & 127
    tok_aligned = (src_tok >> 7) << 7
    delta = (dst_start - src_start) & 127
    # The `n <= 128` source lanes span at most two source tiles.
    a = pltpu.roll(self._src_tile(self.k, tok_aligned), shift=delta, axis=1)
    b = pltpu.roll(
        self._src_tile(self.k, tok_aligned + _SAL_LANES), shift=delta, axis=1
    )
    lane = jax.lax.broadcasted_iota(
        jnp.int32, (self.sublane_groups, _SAL_LANES), 1
    )
    src = jnp.where(lane < dst_start + (_SAL_LANES - src_start), a, b)
    dst = pltpu.bitcast(
        self.rmw[rmw_slot].reshape(self.rec_dim, _SAL_LANES), jnp.uint32
    )
    blended = jnp.where((lane >= dst_start) & (lane < dst_start + n), src, dst)
    self.rmw[rmw_slot] = pltpu.bitcast(blended, jnp.uint8).reshape(
        self.sublane_groups, _SAL_KV_PACKING, _SAL_LANES
    )

  def move_token(self, tok, dst_tok):
    """Copies staged token (lane) `tok` to lane `dst_tok` (`<= tok`)."""
    src = pltpu.roll(
        self._src_tile(self.k, (tok >> 7) << 7),
        shift=(dst_tok - tok) & 127,
        axis=1,
    )
    dst_off = pl.multiple_of((dst_tok >> 7) << 7, _SAL_LANES)
    lane = jax.lax.broadcasted_iota(
        jnp.int32, (self.sublane_groups, _SAL_LANES), 1
    )
    out = jnp.where(
        lane == (dst_tok & 127), src, self._src_tile(self.k, dst_off)
    )
    self.k[:, :, pl.ds(dst_off, _SAL_LANES)] = pltpu.bitcast(
        out, jnp.uint8
    ).reshape(self.sublane_groups, _SAL_KV_PACKING, _SAL_LANES)


def find_runs(
    ops: SalTiles,
    dst_ref,
    run_src_ref,
    run_dst_ref,
    *,
    num_rows: int,
    num_tokens: jax.Array | None = None,
) -> jax.Array:
  """Walks `dst_rows` once, compacting writable tokens and recording runs."""
  max_tokens = dst_ref.shape[0]

  def _step(tok, carry):
    num_runs, num_valid, next_row = carry
    if num_tokens is None:
      row = dst_ref[tok]
    else:
      row = jnp.where(
          tok < num_tokens, dst_ref[jnp.minimum(tok, max_tokens - 1)], -1
      )
    valid = (row >= 0) & (row < num_rows)
    starts = valid & ((row != next_row) | (row % ops.granule == 0))
    run_src_ref[num_runs] = num_valid
    run_dst_ref[num_runs] = row

    @pl.when(valid & (num_valid != tok))
    def _():
      ops.move_token(tok, num_valid)

    return (
        num_runs + starts.astype(jnp.int32),
        num_valid + valid.astype(jnp.int32),
        jnp.where(valid, row + 1, next_row),
    )

  zero = jnp.int32(0)
  init = (zero, zero, jnp.int32(-1))
  if num_tokens is None:
    num_runs, num_valid, _ = jax.lax.fori_loop(
        0, max_tokens, _step, init, unroll=8
    )
  else:

    def _chunk(c, carry):
      for j in range(8):
        carry = _step(c * 8 + j, carry)
      return carry

    num_runs, num_valid, _ = jax.lax.fori_loop(
        0, (num_tokens + 7) // 8, _chunk, init
    )
  run_src_ref[num_runs] = num_valid
  return num_runs


def write_runs(
    ops: SalTiles,
    num_runs: jax.Array,
    run_src_ref,
    run_dst_ref,
    rd_sem,
    wr_sem,
    *,
    wave: int,
) -> None:
  """Writes every run (each inside one granule), `wave` runs at a time."""
  g = ops.granule

  def _wave(w, _):
    first = w * wave
    count = jnp.minimum(wave, num_runs - first)

    def _for_each(fn):
      def _body(i, _):
        src = run_src_ref[first + i]
        fn(i, src, run_dst_ref[first + i], run_src_ref[first + i + 1] - src)
        return None

      jax.lax.fori_loop(0, count, _body, None)

    def _reads(action):
      def _fn(i, src, row, n):
        del src

        @pl.when(n < g)
        def _():
          action(ops.granule_copy(row, i, rd_sem, to_hbm=False))

      return _fn

    def _blend_and_start_write(i, src, row, n):
      ops.blend(i, row, src, n)
      ops.granule_copy(row, i, wr_sem, to_hbm=True).start()

    def _wait_write(i, src, row, n):
      del src, n
      ops.granule_copy(row, i, wr_sem, to_hbm=True).wait()

    _for_each(_reads(lambda c: c.start()))
    _for_each(_reads(lambda c: c.wait()))
    _for_each(_blend_and_start_write)
    _for_each(_wait_write)
    return None

  jax.lax.fori_loop(0, (num_runs + wave - 1) // wave, _wave, None)
