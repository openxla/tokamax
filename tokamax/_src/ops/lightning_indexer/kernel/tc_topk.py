# Copyright 2026 Google LLC. All Rights Reserved.
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
"""TensorCore exact Top-K, DCP merge, and non-negative compaction primitives.

Designed for TPUs without a wide 32-subcore SparseCore. Every vector operation
executes on native `(8, 128)` `int32`
TensorCore VREGs using:
  1. An order-preserving self-inverse IEEE-754 `float32 <-> int32` bijection.
  2. Collision-driven candidate staging in VMEM/VREGs (filtering cold tiles in
     a handful of VPU ops with a single scalar synchronization per KV block).
  3. 2D VREG bitonic sort and bitonic top-K merge using single-instruction
     `tpu.dynamic_gather` (`jnp.take_along_axis` on `(8, 128)`) for intra-VREG
     lane/sublane permutes and zero-permute register pairing across blocks.
  4. 8-sublane KV stream partitioning + log2(P) sublane tree reduction when
     fewer than 8 query rows are active (`B in {1, 2, 4}` decode).
"""

import functools
import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Strictly smaller than `ord(-inf) == -0x7F800001` (-2,139,095,041), so even a
# valid `-inf` score beats an empty/masked slot.
EMPTY_ORD_SENTINEL: int = -0x80000000  # int32(-2147483648)
BITS_NEG_INF: int = -0x00800000  # 0xFF800000 as signed int32 (-8388608)
# Minimum SparseCore subcores (cores * subcores) for the SparseCore top-k path.
MIN_SPARSECORE_TOPK_SUBCORES: int = 32


def has_full_sparsecore_topk() -> bool:
  """Returns True if the TPU has at least 32 SparseCore subcores.

  When False, Lightning Indexer uses the TensorCore top-k, DCP merge, and
  compaction kernels instead of SparseCore.
  """
  sc = pltpu.get_tpu_info().sparse_core
  return (
      sc is not None
      and sc.num_cores * sc.num_subcores >= MIN_SPARSECORE_TOPK_SUBCORES
  )


def next_power_of_2(x: int) -> int:
  if x <= 1:
    return 1
  return 1 << (x - 1).bit_length()


def align_to(x: int, a: int) -> int:
  return ((x + a - 1) // a) * a


def topk_capacity(k: int) -> tuple[int, int]:
  """Returns `(k_cap, num_blocks)` where `k_cap = 128 * num_blocks >= k`."""
  if k <= 0:
    raise ValueError(f"k must be positive, got {k}.")
  k_cap = max(128, next_power_of_2(k))
  return k_cap, k_cap // 128


def sublane_partition_count(bq_total: int, bkv_sz: int) -> int:
  """Number of sublanes cooperating per query row (`1, 2, 4, or 8`)."""
  if bq_total >= 5:
    return 1
  bq_pow2 = next_power_of_2(max(1, bq_total))
  max_parts = max(1, 8 // bq_pow2)
  num_tiles = max(1, bkv_sz // 128)
  return math.gcd(max_parts, num_tiles)


# --------------------------------------------------------------------------- #
# Order-preserving float32 <-> int32 bijection.
# --------------------------------------------------------------------------- #


def f32_bits_to_ord_i32(bits: jax.Array, valid_mask: jax.Array) -> jax.Array:
  """Maps raw `int32` IEEE-754 float32 bits to monotonic signed `int32` keys.

  Matches XLA `jax.lax.top_k`'s IEEE-754 `totalOrder` (`+0.0 > -0.0`, with
  `ord(+0.0) == 0` and `ord(-0.0) == -1`) and is an exact self-inverse
  bijection on all non-sentinel bit patterns.

  Args:
    bits: `int32` array of raw IEEE-754 `float32` bit patterns.
    valid_mask: Boolean mask, broadcastable to `bits`; masked-out entries map
      to `EMPTY_ORD_SENTINEL`.

  Returns:
    `int32` order keys with the same shape as `bits`.
  """
  sign_mask = lax.shift_right_arithmetic(bits, jnp.int32(31))
  ord_key = lax.bitwise_xor(
      bits, lax.bitwise_and(sign_mask, jnp.int32(0x7FFFFFFF))
  )
  return jnp.where(valid_mask, ord_key, jnp.int32(EMPTY_ORD_SENTINEL))


def f32_to_ord_i32(x: jax.Array, valid_mask: jax.Array) -> jax.Array:
  """Maps `float32` scores to monotonic signed `int32` keys."""
  bits = lax.bitcast_convert_type(x.astype(jnp.float32), jnp.int32)
  return f32_bits_to_ord_i32(bits, valid_mask)


def ord_i32_to_f32_bits(ord_key: jax.Array, is_valid: jax.Array) -> jax.Array:
  """Inverse of `f32_bits_to_ord_i32`: recovers exact `float32` bit patterns."""
  sign_mask = lax.shift_right_arithmetic(ord_key, jnp.int32(31))
  bits = lax.bitwise_xor(
      ord_key, lax.bitwise_and(sign_mask, jnp.int32(0x7FFFFFFF))
  )
  return jnp.where(is_valid, bits, jnp.int32(BITS_NEG_INF))


# --------------------------------------------------------------------------- #
# Native (8, 128) VREG bitonic sort & top-K merge primitives.
# --------------------------------------------------------------------------- #


def _lanes_2d() -> jax.Array:
  """Returns `(8, 128)` `int32` lane iota via hardware `tpu.iota`."""
  return lax.broadcasted_iota(jnp.int32, (8, 128), 1)


def _sublanes_2d() -> jax.Array:
  """Returns `(8, 128)` `int32` sublane iota via hardware `tpu.iota`."""
  return lax.broadcasted_iota(jnp.int32, (8, 128), 0)


def _gather_lanes(x: jax.Array, idx: jax.Array) -> jax.Array:
  """Single-VREG `tpu.dynamic_gather` along lane axis (`axis=-1`)."""
  return jnp.take_along_axis(
      x,
      idx,
      axis=-1,
      mode="promise_in_bounds",
      wrap_negative_indices=False,
  )


def _gather_sublanes(x: jax.Array, idx: jax.Array) -> jax.Array:
  """Single-VREG `tpu.dynamic_gather` along sublane axis (`axis=-2`)."""
  return jnp.take_along_axis(
      x,
      idx,
      axis=-2,
      mode="promise_in_bounds",
      wrap_negative_indices=False,
  )


def better_pair(
    ord_a: jax.Array,
    idx_a: jax.Array,
    ord_b: jax.Array,
    idx_b: jax.Array,
) -> jax.Array:
  """Strict `(score, -index)` comparator matching `jax.lax.top_k` tie-breaks."""
  return (ord_a > ord_b) | ((ord_a == ord_b) & (idx_a < idx_b))


def _compare_swap_lane_stage(
    ord_b: list[jax.Array],
    idx_b: list[jax.Array],
    *,
    d: int,
    stage_log: int | None,
    descending: bool,
) -> tuple[list[jax.Array], list[jax.Array]]:
  """Executes one intra-VREG (`d < 128`) bitonic stage across all `R` blocks."""
  num_blocks = len(ord_b)
  lanes = _lanes_2d()
  partner_idx = lax.bitwise_xor(lanes, jnp.int32(d))
  is_upper = lax.bitwise_and(lanes, jnp.int32(d)) != 0

  shared_want_max: jax.Array | None = None
  if stage_log is None:
    shared_want_max = ~is_upper if descending else is_upper
  elif stage_log < 7:
    parity = (
        lax.bitwise_and(
            lax.shift_right_arithmetic(lanes, jnp.int32(stage_log)),
            jnp.int32(1),
        )
        == 0
    )
    dir_desc_v = parity if descending else ~parity
    shared_want_max = lax.bitwise_xor(dir_desc_v, is_upper)

  new_ord: list[jax.Array] = []
  new_idx: list[jax.Array] = []
  for r in range(num_blocks):
    if shared_want_max is not None:
      want_max = shared_want_max
    else:
      assert stage_log is not None
      blk_parity = (((r * 128) >> stage_log) & 1) == 0
      dir_desc = blk_parity if descending else (not blk_parity)
      want_max = ~is_upper if dir_desc else is_upper

    o_self, i_self = ord_b[r], idx_b[r]
    o_part = _gather_lanes(o_self, partner_idx)
    i_part = _gather_lanes(i_self, partner_idx)

    # Distinct slots always hold distinct KV indices (or identical empty
    # sentinels where swapping is a no-op), so `self_better == ~part_better`
    # everywhere a swap changes the value: `(want_max & p) | (~want_max & ~p)`
    # simplifies to `~want_max ^ p` (1 mask XOR instead of 9 VPU mask ops).
    part_better = better_pair(o_part, i_part, o_self, i_self)
    swap = lax.bitwise_xor(~want_max, part_better)

    new_ord.append(jnp.where(swap, o_part, o_self))
    new_idx.append(jnp.where(swap, i_part, i_self))
  return new_ord, new_idx


def _compare_swap_block_stage(
    ord_b: list[jax.Array],
    idx_b: list[jax.Array],
    *,
    r_step: int,
    stage_log: int | None,
    descending: bool,
) -> tuple[list[jax.Array], list[jax.Array]]:
  """Executes one inter-VREG (`d = 128 * r_step`) stage with zero permutes."""
  num_blocks = len(ord_b)
  new_ord = list(ord_b)
  new_idx = list(idx_b)
  for r0 in range(num_blocks):
    if (r0 & r_step) != 0:
      continue
    r1 = r0 | r_step
    if stage_log is None:
      dir_desc = descending
    else:
      pos0 = r0 * 128
      parity = ((pos0 >> stage_log) & 1) == 0
      dir_desc = parity if descending else (not parity)

    o0, i0 = ord_b[r0], idx_b[r0]
    o1, i1 = ord_b[r1], idx_b[r1]
    swap = (
        better_pair(o1, i1, o0, i0) if dir_desc else better_pair(o0, i0, o1, i1)
    )
    new_ord[r0] = jnp.where(swap, o1, o0)
    new_ord[r1] = jnp.where(swap, o0, o1)
    new_idx[r0] = jnp.where(swap, i1, i0)
    new_idx[r1] = jnp.where(swap, i0, i1)
  return new_ord, new_idx


def bitonic_merge_blocks(
    ord_blocks: tuple[jax.Array, ...],
    idx_blocks: tuple[jax.Array, ...],
    *,
    descending: bool = True,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
  """Sorts a bitonic sequence held in `R` `(8, 128)` blocks per sublane."""
  num_blocks = len(ord_blocks)
  total_len = 128 * num_blocks
  total_log = (total_len - 1).bit_length()
  ord_b = list(ord_blocks)
  idx_b = list(idx_blocks)

  for j in range(total_log - 1, -1, -1):
    d = 1 << j
    if d >= 128:
      ord_b, idx_b = _compare_swap_block_stage(
          ord_b,
          idx_b,
          r_step=d // 128,
          stage_log=None,
          descending=descending,
      )
    else:
      ord_b, idx_b = _compare_swap_lane_stage(
          ord_b,
          idx_b,
          d=d,
          stage_log=None,
          descending=descending,
      )
  return tuple(ord_b), tuple(idx_b)


def bitonic_sort_blocks(
    ord_blocks: tuple[jax.Array, ...],
    idx_blocks: tuple[jax.Array, ...],
    *,
    descending: bool = True,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
  """Bitonic-sorts `128 * R` values per sublane held in `R` `(8, 128)` VREGs."""
  num_blocks = len(ord_blocks)
  total_len = 128 * num_blocks
  total_log = (total_len - 1).bit_length()
  ord_b = list(ord_blocks)
  idx_b = list(idx_blocks)

  for s in range(1, total_log + 1):
    stage_log = None if s == total_log else s
    for j in range(s - 1, -1, -1):
      d = 1 << j
      if d >= 128:
        ord_b, idx_b = _compare_swap_block_stage(
            ord_b,
            idx_b,
            r_step=d // 128,
            stage_log=stage_log,
            descending=descending,
        )
      else:
        ord_b, idx_b = _compare_swap_lane_stage(
            ord_b,
            idx_b,
            d=d,
            stage_log=stage_log,
            descending=descending,
        )
  return tuple(ord_b), tuple(idx_b)


def bitonic_topk_merge_blocks(
    top_ord: tuple[jax.Array, ...],
    top_idx: tuple[jax.Array, ...],
    cand_asc_ord: tuple[jax.Array, ...],
    cand_asc_idx: tuple[jax.Array, ...],
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
  """Returns the descending top `128 * R` of `top` (desc) and `cand` (asc)."""
  merged_ord: list[jax.Array] = []
  merged_idx: list[jax.Array] = []
  for o_t, i_t, o_c, i_c in zip(top_ord, top_idx, cand_asc_ord, cand_asc_idx):
    take_cand = better_pair(o_c, i_c, o_t, i_t)
    merged_ord.append(jnp.where(take_cand, o_c, o_t))
    merged_idx.append(jnp.where(take_cand, i_c, i_t))
  return bitonic_merge_blocks(
      tuple(merged_ord), tuple(merged_idx), descending=True
  )


def extract_row_kth_threshold(
    top_ord: tuple[jax.Array, ...],
    k: int,
) -> jax.Array:
  """Broadcasts each sublane's `(k - 1)`-th sorted key to `(8, 128)`."""
  k_blk = (k - 1) // 128
  k_lane = (k - 1) % 128
  lane_idx = jnp.full((8, 128), k_lane, dtype=jnp.int32)
  return _gather_lanes(top_ord[k_blk], lane_idx)


def reduce_sublane_partitions(
    top_ord: tuple[jax.Array, ...],
    top_idx: tuple[jax.Array, ...],
    *,
    num_partitions: int,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
  """Tree-reduces `num_partitions` sublanes per query, compacting to sublane q.

  Query `q` initially owns sublanes `[q * P, (q + 1) * P)`; on return its
  merged top-k is in sublane `q`.

  Args:
    top_ord: Per-sublane descending order keys, one `(8, 128)` block per
      128 slots.
    top_idx: Indices matching `top_ord`.
    num_partitions: Number of sublanes `P` cooperating per query.

  Returns:
    `(top_ord, top_idx)` with each query's merged top-k in sublane `q`.
  """
  if num_partitions <= 1:
    return top_ord, top_idx

  rev_lanes = lax.bitwise_xor(_lanes_2d(), jnp.int32(127))
  sublanes = _sublanes_2d()
  num_blocks = len(top_ord)
  step_p = num_partitions // 2
  while step_p >= 1:
    partner_sublanes = lax.bitwise_xor(sublanes, jnp.int32(step_p))
    cand_asc_ord = tuple(
        _gather_lanes(
            _gather_sublanes(top_ord[num_blocks - 1 - r], partner_sublanes),
            rev_lanes,
        )
        for r in range(num_blocks)
    )
    cand_asc_idx = tuple(
        _gather_lanes(
            _gather_sublanes(top_idx[num_blocks - 1 - r], partner_sublanes),
            rev_lanes,
        )
        for r in range(num_blocks)
    )
    top_ord, top_idx = bitonic_topk_merge_blocks(
        top_ord, top_idx, cand_asc_ord, cand_asc_idx
    )
    step_p //= 2

  # Sublane `q * num_partitions` holds query `q`; compact to sublane `q`.
  compact_idx = jnp.minimum(sublanes * jnp.int32(num_partitions), jnp.int32(7))
  top_ord = tuple(_gather_sublanes(o, compact_idx) for o in top_ord)
  top_idx = tuple(_gather_sublanes(i, compact_idx) for i in top_idx)
  return top_ord, top_idx


# --------------------------------------------------------------------------- #
# Collision-driven VMEM streaming accumulator.
# --------------------------------------------------------------------------- #


def init_topk_vmem(
    top_ord_ref: Any,
    top_idx_ref: Any,
    cand_ord_ref: Any,
    cand_idx_ref: Any,
    tau_ref: Any,
    *,
    num_q_groups: int,
    num_blocks: int,
) -> None:
  """Initializes streaming Top-K VMEM state for a new query block."""
  empty_ord = jnp.full((8, 128), EMPTY_ORD_SENTINEL, dtype=jnp.int32)
  empty_idx = jnp.full((8, 128), -1, dtype=jnp.int32)

  def _init_group(g, _):
    tau_ref[g] = empty_ord
    for r in range(num_blocks):
      top_ord_ref[g, r] = empty_ord
      top_idx_ref[g, r] = empty_idx
      cand_ord_ref[g, r] = empty_ord
      cand_idx_ref[g, r] = empty_idx
    return None

  if num_q_groups == 1:
    _init_group(0, None)
  else:
    lax.fori_loop(0, num_q_groups, _init_group, None)


def accumulate_subbatch_vmem(
    top_ord_ref: Any,
    top_idx_ref: Any,
    cand_ord_ref: Any,
    cand_idx_ref: Any,
    tau_ref: Any,
    *,
    q_group: jax.Array | int,
    tile_ords: tuple[jax.Array, ...],
    tile_idxs: tuple[jax.Array, ...],
    sub_step: jax.Array | int,
    k: int,
    num_blocks: int,
) -> None:
  """Streams `W` `(8, 128)` tiles (`W | num_blocks`) into VMEM Top-K state.

  Fast path (`~95%+` of steps): 1 scalar reduction across all `W` tiles, zero
  permutes, updating only the `W` target candidate buckets in place.
  Slow path (collision): tries a 64-lane XOR secondary probe when `W == 1`, and
  otherwise flushes all `num_blocks` candidate buckets into `top` via bitonic
  sort + merge, tightens `tau`, and stages the incoming tiles cleanly.

  Args:
    top_ord_ref: VMEM ref of the running descending top-k order keys.
    top_idx_ref: VMEM ref of the indices matching `top_ord_ref`.
    cand_ord_ref: VMEM ref of the staged candidate order keys.
    cand_idx_ref: VMEM ref of the indices matching `cand_ord_ref`.
    tau_ref: VMEM ref of each row's current k-th best order key.
    q_group: Query group (first index) of the refs to update.
    tile_ords: `W` incoming `(8, 128)` order-key tiles.
    tile_idxs: Indices matching `tile_ords`.
    sub_step: Streaming step, selecting the candidate buckets to target.
    k: Number of top entries to keep.
    num_blocks: Number of `(8, 128)` blocks in the top-k state.
  """
  w_len = len(tile_ords)
  assert len(tile_idxs) == w_len
  assert num_blocks % w_len == 0
  num_phases = num_blocks // w_len

  tau = tau_ref[q_group]
  if num_phases == 1:
    base_r = 0
    cand_o = [cand_ord_ref[q_group, w] for w in range(w_len)]
  else:
    phase = sub_step % num_phases
    base_r = phase * w_len
    cand_o = [cand_ord_ref[q_group, base_r + w] for w in range(w_len)]

  hots = [tile_ords[w] > tau for w in range(w_len)]
  colls = [hots[w] & (cand_o[w] > tau) for w in range(w_len)]

  hot_union = hots[0]
  coll_union = colls[0]
  for w in range(1, w_len):
    hot_union = hot_union | hots[w]
    coll_union = coll_union | colls[w]

  # Single scalar check for completely cold sub-batches: zero VMEM writes!
  @pl.when(jnp.any(hot_union))
  def _on_any_hot():
    any_coll = jnp.any(coll_union)

    @pl.when(~any_coll)
    def _fast_stage():
      for w in range(w_len):
        r_idx = w if num_phases == 1 else base_r + w
        cand_ord_ref[q_group, r_idx] = jnp.where(
            hots[w], tile_ords[w], cand_o[w]
        )
        cand_idx_ref[q_group, r_idx] = jnp.where(
            hots[w], tile_idxs[w], cand_idx_ref[q_group, r_idx]
        )

    @pl.when(any_coll)
    def _on_collision():
      def _flush_and_stage():
        cur_top_o = tuple(top_ord_ref[q_group, r] for r in range(num_blocks))
        cur_top_i = tuple(top_idx_ref[q_group, r] for r in range(num_blocks))
        cur_cand_o = tuple(cand_ord_ref[q_group, r] for r in range(num_blocks))
        cur_cand_i = tuple(cand_idx_ref[q_group, r] for r in range(num_blocks))

        cand_asc_o, cand_asc_i = bitonic_sort_blocks(
            cur_cand_o, cur_cand_i, descending=False
        )
        new_top_o, new_top_i = bitonic_topk_merge_blocks(
            cur_top_o, cur_top_i, cand_asc_o, cand_asc_i
        )
        for r in range(num_blocks):
          top_ord_ref[q_group, r] = new_top_o[r]
          top_idx_ref[q_group, r] = new_top_i[r]

        new_tau = extract_row_kth_threshold(new_top_o, k)
        tau_ref[q_group] = new_tau

        empty_o = jnp.full((8, 128), EMPTY_ORD_SENTINEL, dtype=jnp.int32)
        empty_i = jnp.full((8, 128), -1, dtype=jnp.int32)
        if num_phases == 1:
          for w in range(w_len):
            keep = tile_ords[w] > new_tau
            cand_ord_ref[q_group, w] = jnp.where(keep, tile_ords[w], empty_o)
            cand_idx_ref[q_group, w] = jnp.where(keep, tile_idxs[w], empty_i)
        else:
          for r in range(num_blocks):
            cand_ord_ref[q_group, r] = empty_o
            cand_idx_ref[q_group, r] = empty_i
          for w in range(w_len):
            r_idx = base_r + w
            keep = tile_ords[w] > new_tau
            cand_ord_ref[q_group, r_idx] = jnp.where(
                keep, tile_ords[w], empty_o
            )
            cand_idx_ref[q_group, r_idx] = jnp.where(
                keep, tile_idxs[w], empty_i
            )

      if w_len == 1:
        # 2-choice 64-lane XOR secondary probe (only paid on a collision).
        xor64 = lax.bitwise_xor(_lanes_2d(), jnp.int32(64))
        shuf_o = _gather_lanes(tile_ords[0], xor64)
        shuf_i = _gather_lanes(tile_idxs[0], xor64)
        shuf_hot = shuf_o > tau
        shuf_coll = jnp.any(shuf_hot & (cand_o[0] > tau))

        @pl.when(~shuf_coll)
        def _secondary_stage():
          r_idx = 0 if num_phases == 1 else base_r
          cand_ord_ref[q_group, r_idx] = jnp.where(shuf_hot, shuf_o, cand_o[0])
          cand_idx_ref[q_group, r_idx] = jnp.where(
              shuf_hot, shuf_i, cand_idx_ref[q_group, r_idx]
          )

        @pl.when(shuf_coll)
        def _secondary_flush():
          _flush_and_stage()

      else:
        _flush_and_stage()


def finalize_topk_vmem(
    top_ord_ref: Any,
    top_idx_ref: Any,
    cand_ord_ref: Any,
    cand_idx_ref: Any,
    tau_ref: Any,
    *,
    q_group: jax.Array | int,
    num_blocks: int,
    num_partitions: int = 1,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
  """Flushes remaining staged candidates and reduces sublane partitions."""
  tau = tau_ref[q_group]
  cand_hot = cand_ord_ref[q_group, 0] > tau
  for r in range(1, num_blocks):
    cand_hot = cand_hot | (cand_ord_ref[q_group, r] > tau)

  @pl.when(jnp.any(cand_hot))
  def _final_flush():
    cur_top_o = tuple(top_ord_ref[q_group, r] for r in range(num_blocks))
    cur_top_i = tuple(top_idx_ref[q_group, r] for r in range(num_blocks))
    cur_cand_o = tuple(cand_ord_ref[q_group, r] for r in range(num_blocks))
    cur_cand_i = tuple(cand_idx_ref[q_group, r] for r in range(num_blocks))
    cand_asc_o, cand_asc_i = bitonic_sort_blocks(
        cur_cand_o, cur_cand_i, descending=False
    )
    merged_o, merged_i = bitonic_topk_merge_blocks(
        cur_top_o, cur_top_i, cand_asc_o, cand_asc_i
    )
    for r in range(num_blocks):
      top_ord_ref[q_group, r] = merged_o[r]
      top_idx_ref[q_group, r] = merged_i[r]

  final_o = tuple(top_ord_ref[q_group, r] for r in range(num_blocks))
  final_i = tuple(top_idx_ref[q_group, r] for r in range(num_blocks))
  if num_partitions > 1:
    final_o, final_i = reduce_sublane_partitions(
        final_o,
        final_i,
        num_partitions=num_partitions,
    )
  return final_o, final_i


# --------------------------------------------------------------------------- #
# Standalone TensorCore kernels: Top-K, non-negative pack, and DCP merge.
# --------------------------------------------------------------------------- #


@functools.partial(
    jax.jit,
    static_argnames=("k", "return_scores", "tile_cols"),
)
def tensorcore_topk(
    scores: jax.Array,
    k: int,
    *,
    row_lengths: jax.Array | None = None,
    return_scores: bool = False,
    tile_cols: int = 1024,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Exact TensorCore Top-K over `scores` (`float32` or raw `int32` f32 bits).

  Args:
    scores: `[num_rows, num_cols]` array of `float32` scores or `int32` holding
      raw IEEE-754 `float32` bit patterns.
    k: Number of top entries to return per row (`k >= 1`).
    row_lengths: Optional `int32[num_rows]` causal/valid column limit per row.
    return_scores: If True, also return `int32[num_rows, k]` raw float32 score
      bits (`BITS_NEG_INF` for unfilled `-1` entries).
    tile_cols: Streaming tile width along columns (multiple of 128).

  Returns:
    `top_indices` of shape `[num_rows, k]`, or `(top_indices, top_scores_bits)`
    when `return_scores=True`.
  """
  if scores.ndim != 2:
    raise ValueError(f"scores must be 2D, got shape {scores.shape}.")
  if scores.dtype not in (jnp.float32, jnp.int32):
    raise ValueError(f"scores must be float32 or int32, got {scores.dtype}.")
  if tile_cols % 128 != 0 or tile_cols <= 0:
    raise ValueError(
        f"tile_cols ({tile_cols}) must be a positive multiple of 128."
    )

  num_rows, num_cols = scores.shape
  if num_rows == 0:
    empty_i = jnp.empty((0, k), dtype=jnp.int32)
    return (empty_i, empty_i) if return_scores else empty_i

  k_cap, num_blocks = topk_capacity(k)
  padded_rows = align_to(num_rows, 8)
  padded_cols = max(tile_cols, align_to(max(1, num_cols), tile_cols))

  if row_lengths is None:
    eff_lens = jnp.full((num_rows,), num_cols, dtype=jnp.int32)
  else:
    eff_lens = jnp.clip(row_lengths.astype(jnp.int32), 0, num_cols)
  if padded_rows > num_rows:
    eff_lens = jnp.pad(eff_lens, (0, padded_rows - num_rows), constant_values=0)

  is_bits = scores.dtype == jnp.int32
  pad_val = jnp.int32(BITS_NEG_INF) if is_bits else jnp.float32(-jnp.inf)
  if padded_rows > num_rows or padded_cols > num_cols:
    scores = jnp.pad(
        scores,
        ((0, padded_rows - num_rows), (0, padded_cols - num_cols)),
        constant_values=pad_val,
    )

  num_row_groups = padded_rows // 8
  num_col_steps = padded_cols // tile_cols
  tiles_per_step = tile_cols // 128
  subbatch_w = math.gcd(tiles_per_step, num_blocks)
  num_subbatches = tiles_per_step // subbatch_w
  out_blocks = (2 * num_blocks) if return_scores else num_blocks

  def _topk_kernel(
      eff_lens_smem,
      scores_vmem,
      out_vmem,
      top_o_ref,
      top_i_ref,
      cand_o_ref,
      cand_i_ref,
      tau_ref,
  ):
    rg = pl.program_id(0)
    cs = pl.program_id(1)

    @pl.when(cs == 0)
    def _init():
      init_topk_vmem(
          top_o_ref,
          top_i_ref,
          cand_o_ref,
          cand_i_ref,
          tau_ref,
          num_q_groups=1,
          num_blocks=num_blocks,
      )

    row_len_vec = jnp.concatenate(
        [
            jnp.full((1, 128), eff_lens_smem[rg * 8 + s], jnp.int32)
            for s in range(8)
        ],
        axis=0,
    )
    lanes = _lanes_2d()
    col_base = cs * tile_cols
    raw_block = scores_vmem[...]

    for sb in range(num_subbatches):
      t_ords = []
      t_idxs = []
      for w in range(subbatch_w):
        m = sb * subbatch_w + w
        s_tile = raw_block[:, m * 128 : (m + 1) * 128]
        idx_tile = (col_base + m * 128) + lanes
        valid = idx_tile < row_len_vec
        ord_tile = (
            f32_bits_to_ord_i32(s_tile, valid)
            if is_bits
            else f32_to_ord_i32(s_tile, valid)
        )
        t_ords.append(ord_tile)
        t_idxs.append(idx_tile)

      accumulate_subbatch_vmem(
          top_o_ref,
          top_i_ref,
          cand_o_ref,
          cand_i_ref,
          tau_ref,
          q_group=0,
          tile_ords=tuple(t_ords),
          tile_idxs=tuple(t_idxs),
          sub_step=cs * num_subbatches + sb,
          k=k,
          num_blocks=num_blocks,
      )

    @pl.when(cs == num_col_steps - 1)
    def _finish():
      fin_o, fin_i = finalize_topk_vmem(
          top_o_ref,
          top_i_ref,
          cand_o_ref,
          cand_i_ref,
          tau_ref,
          q_group=0,
          num_blocks=num_blocks,
      )
      idx_2d = jnp.concatenate(fin_i, axis=1)
      if return_scores:
        scores_2d = jnp.concatenate(
            [ord_i32_to_f32_bits(o, i >= 0) for o, i in zip(fin_o, fin_i)],
            axis=1,
        )
        combined = jnp.concatenate([idx_2d, scores_2d], axis=1)
        out_vmem[...] = combined.reshape(8, out_blocks, 128)
      else:
        out_vmem[...] = idx_2d.reshape(8, out_blocks, 128)

  vmem_shape = (1, num_blocks, 8, 128)
  raw_out = pl.pallas_call(
      _topk_kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[
              pl.BlockSpec((8, tile_cols), lambda rg, cs, _: (rg, cs)),
          ],
          out_specs=pl.BlockSpec(
              (8, out_blocks, 128), lambda rg, cs, _: (rg, 0, 0)
          ),
          grid=(num_row_groups, num_col_steps),
          scratch_shapes=[
              pltpu.VMEM(vmem_shape, jnp.int32),
              pltpu.VMEM(vmem_shape, jnp.int32),
              pltpu.VMEM(vmem_shape, jnp.int32),
              pltpu.VMEM(vmem_shape, jnp.int32),
              pltpu.VMEM((1, 8, 128), jnp.int32),
          ],
      ),
      out_shape=jax.ShapeDtypeStruct((padded_rows, out_blocks, 128), jnp.int32),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      name=f"tc_topk_k{k}_c{tile_cols}",
  )(eff_lens, scores)

  top_idx = raw_out[:num_rows, :num_blocks, :].reshape(num_rows, k_cap)[:, :k]
  if not return_scores:
    return top_idx
  top_scores = raw_out[:num_rows, num_blocks:, :].reshape(num_rows, k_cap)[
      :, :k
  ]
  return top_idx, top_scores


@jax.jit
def pack_nonnegative_tc(vals: jax.Array) -> jax.Array:
  """Left-packs each row's non-negative `int32` entries on TensorCore.

  Preserves exact original column relative order with `-1` suffix padding,
  matching `dcp_sc_compact.pack_nonnegative` without SparseCore.
  """
  if vals.ndim != 2:
    raise ValueError(f"vals must be 2D, got shape {vals.shape}.")
  if vals.dtype != jnp.int32:
    raise ValueError(f"vals must be int32, got {vals.dtype}.")

  num_rows, width = vals.shape
  if width % 16 != 0:
    raise ValueError(f"width ({width}) must be a multiple of 16")
  if num_rows == 0 or width == 0:
    return vals

  padded_rows = align_to(num_rows, 8)
  w_cap = max(128, next_power_of_2(width))
  num_blocks = w_cap // 128

  if padded_rows > num_rows or w_cap > width:
    vals = jnp.pad(
        vals,
        ((0, padded_rows - num_rows), (0, w_cap - width)),
        constant_values=-1,
    )

  def _pack_kernel(vals_vmem, out_vmem):
    v = vals_vmem[...]
    val_blocks = tuple(v[:, r * 128 : (r + 1) * 128] for r in range(num_blocks))
    key_blocks = []
    lanes = _lanes_2d()
    for r in range(num_blocks):
      rev_pos = jnp.int32(w_cap - r * 128) - lanes
      key_blocks.append(jnp.where(val_blocks[r] >= 0, rev_pos, jnp.int32(-1)))
    _, packed_vals = bitonic_sort_blocks(
        tuple(key_blocks), val_blocks, descending=True
    )
    out_vmem[...] = jnp.concatenate(packed_vals, axis=1)

  packed = pl.pallas_call(
      _pack_kernel,
      in_specs=[pl.BlockSpec((8, w_cap), lambda rg: (rg, 0))],
      out_specs=pl.BlockSpec((8, w_cap), lambda rg: (rg, 0)),
      grid=(padded_rows // 8,),
      out_shape=jax.ShapeDtypeStruct((padded_rows, w_cap), jnp.int32),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          disable_bounds_checks=True,
      ),
      name=f"tc_pack_nonneg_b{num_rows}_w{width}",
  )(vals)
  return packed[:num_rows, :width]


@functools.partial(
    jax.jit,
    static_argnames=("k", "dcp_size", "inputs_sorted"),
)
def dcp_merge_and_compact_tc(
    gathered_scores_bits: jax.Array,
    local_idxs: jax.Array,
    *,
    k: int,
    dcp_rank: jax.Array | int,
    dcp_size: int,
    inputs_sorted: bool = False,
) -> jax.Array:
  """Merges `[T, dcp_size * k]` DCP scores and left-packs owned local winners.

  Executes entirely on TensorCore in a single fused Pallas kernel:
  1. Merges the `dcp_size` per-rank `[k]` score blocks using deterministic
     `(score, -flat_col)` tie-breaking so all DCP ranks agree on global winners
     with multiplicity 1.
  2. Carries `local_idxs` only on `rank == dcp_rank` (`-1` on foreign ranks),
     avoiding cross-rank index communication and cross-VREG index gathers.
  3. Left-packs surviving `>= 0` owned local indices into a `-1`-padded prefix.
  """
  num_rows = local_idxs.shape[0]
  if num_rows == 0:
    return jnp.empty((0, k), dtype=jnp.int32)

  k_cap, num_blocks = topk_capacity(k)
  padded_rows = align_to(num_rows, 8)

  # Reshape `[T, dcp_size * k] -> [T, dcp_size, k]`, then pad to
  # `(padded_rows, dcp_size, k_cap)`.
  scores_3d = gathered_scores_bits.reshape(num_rows, dcp_size, k)
  if padded_rows > num_rows or k_cap > k:
    scores_3d = jnp.pad(
        scores_3d,
        ((0, padded_rows - num_rows), (0, 0), (0, k_cap - k)),
        constant_values=BITS_NEG_INF,
    )
    local_idxs_pad = jnp.pad(
        local_idxs,
        ((0, padded_rows - num_rows), (0, k_cap - k)),
        constant_values=-1,
    )
  else:
    local_idxs_pad = local_idxs

  scores_2d = scores_3d.reshape(padded_rows, dcp_size * k_cap)
  dcp_rank_arr = jnp.asarray([dcp_rank], dtype=jnp.int32)

  def _merge_and_pack_stage(
      ord_b: tuple[jax.Array, ...],
      col_b: tuple[jax.Array, ...],
      own_b: tuple[jax.Array, ...],
      cand_ord: tuple[jax.Array, ...],
      cand_col: tuple[jax.Array, ...],
      cand_own: tuple[jax.Array, ...],
  ):
    """Merges candidate rank chunk `(cand_ord, cand_col, cand_own)` into top."""
    total_len = 128 * num_blocks
    total_log = (total_len - 1).bit_length()

    lanes = _lanes_2d()
    if inputs_sorted:
      # Each rank's output from `tc_streamindex_topk` is already sorted
      # descending; reverse in 1 gather/block to get ascending bitonic partner.
      rev_lanes = lax.bitwise_xor(lanes, jnp.int32(127))
      c_ord = [_gather_lanes(o, rev_lanes) for o in reversed(cand_ord)]
      c_col = [_gather_lanes(c, rev_lanes) for c in reversed(cand_col)]
      c_own = [_gather_lanes(m, rev_lanes) for m in reversed(cand_own)]
    else:
      c_ord = list(cand_ord)
      c_col = list(cand_col)
      c_own = list(cand_own)
      for s in range(1, total_log + 1):
        stage_log = None if s == total_log else s
        for j in range(s - 1, -1, -1):
          d = 1 << j
          if d >= 128:
            r_step = d // 128
            for r0 in range(num_blocks):
              if (r0 & r_step) != 0:
                continue
              r1 = r0 | r_step
              dir_desc = (
                  False
                  if stage_log is None
                  else (((r0 * 128) >> stage_log) & 1) != 0
              )
              swap = (
                  better_pair(c_ord[r1], c_col[r1], c_ord[r0], c_col[r0])
                  if dir_desc
                  else better_pair(c_ord[r0], c_col[r0], c_ord[r1], c_col[r1])
              )
              o0, o1 = c_ord[r0], c_ord[r1]
              k0, k1 = c_col[r0], c_col[r1]
              m0, m1 = c_own[r0], c_own[r1]
              c_ord[r0], c_ord[r1] = jnp.where(swap, o1, o0), jnp.where(
                  swap, o0, o1
              )
              c_col[r0], c_col[r1] = jnp.where(swap, k1, k0), jnp.where(
                  swap, k0, k1
              )
              c_own[r0], c_own[r1] = jnp.where(swap, m1, m0), jnp.where(
                  swap, m0, m1
              )
          else:
            p_idx = lax.bitwise_xor(lanes, jnp.int32(d))
            is_upper = lax.bitwise_and(lanes, jnp.int32(d)) != 0
            shared_want_max: jax.Array | None = None
            if stage_log is None:
              shared_want_max = is_upper
            elif stage_log < 7:
              dir_desc_v = (
                  lax.bitwise_and(
                      lax.shift_right_arithmetic(lanes, jnp.int32(stage_log)),
                      jnp.int32(1),
                  )
                  != 0
              )
              shared_want_max = lax.bitwise_xor(dir_desc_v, is_upper)
            for r in range(num_blocks):
              if shared_want_max is not None:
                want_max = shared_want_max
              else:
                assert stage_log is not None
                dir_desc = (((r * 128) >> stage_log) & 1) != 0
                want_max = ~is_upper if dir_desc else is_upper
              os, cs_v, ms = c_ord[r], c_col[r], c_own[r]
              op = _gather_lanes(os, p_idx)
              cp = _gather_lanes(cs_v, p_idx)
              mp = _gather_lanes(ms, p_idx)
              part_better = better_pair(op, cp, os, cs_v)
              swap = lax.bitwise_xor(~want_max, part_better)
              c_ord[r] = jnp.where(swap, op, os)
              c_col[r] = jnp.where(swap, cp, cs_v)
              c_own[r] = jnp.where(swap, mp, ms)

    # Top-K bitonic merge of `top` (descending) and `c` (ascending).
    m_ord = []
    m_col = []
    m_own = []
    for r in range(num_blocks):
      take_c = better_pair(c_ord[r], c_col[r], ord_b[r], col_b[r])
      m_ord.append(jnp.where(take_c, c_ord[r], ord_b[r]))
      m_col.append(jnp.where(take_c, c_col[r], col_b[r]))
      m_own.append(jnp.where(take_c, c_own[r], own_b[r]))

    for j in range(total_log - 1, -1, -1):
      d = 1 << j
      if d >= 128:
        r_step = d // 128
        for r0 in range(num_blocks):
          if (r0 & r_step) != 0:
            continue
          r1 = r0 | r_step
          swap = better_pair(m_ord[r1], m_col[r1], m_ord[r0], m_col[r0])
          o0, o1 = m_ord[r0], m_ord[r1]
          k0, k1 = m_col[r0], m_col[r1]
          w0, w1 = m_own[r0], m_own[r1]
          m_ord[r0] = jnp.where(swap, o1, o0)
          m_ord[r1] = jnp.where(swap, o0, o1)
          m_col[r0] = jnp.where(swap, k1, k0)
          m_col[r1] = jnp.where(swap, k0, k1)
          m_own[r0] = jnp.where(swap, w1, w0)
          m_own[r1] = jnp.where(swap, w0, w1)
      else:
        p_idx = lax.bitwise_xor(lanes, jnp.int32(d))
        want_min = lax.bitwise_and(lanes, jnp.int32(d)) != 0
        for r in range(num_blocks):
          os, cs_v, ws = m_ord[r], m_col[r], m_own[r]
          op = _gather_lanes(os, p_idx)
          cp = _gather_lanes(cs_v, p_idx)
          wp = _gather_lanes(ws, p_idx)
          part_better = better_pair(op, cp, os, cs_v)
          swap = lax.bitwise_xor(want_min, part_better)
          m_ord[r] = jnp.where(swap, op, os)
          m_col[r] = jnp.where(swap, cp, cs_v)
          m_own[r] = jnp.where(swap, wp, ws)

    return tuple(m_ord), tuple(m_col), tuple(m_own)

  def _dcp_kernel(rank_smem, scores_vmem, my_idxs_vmem, out_vmem):
    my_rank = rank_smem[0]
    my_rank_vec = jnp.full((8, 128), my_rank, dtype=jnp.int32)
    s_all = scores_vmem[...]
    my_i = my_idxs_vmem[...]
    lanes = _lanes_2d()

    def _chunk_triples(r_rank: int):
      s_chunk = s_all[:, r_rank * k_cap : (r_rank + 1) * k_cap]
      is_mine = my_rank_vec == r_rank
      ords, cols, owns = [], [], []
      for b in range(num_blocks):
        pos = jnp.int32(b * 128) + lanes
        in_k = pos < jnp.int32(k)
        sb = s_chunk[:, b * 128 : (b + 1) * 128]
        valid = in_k & (sb != jnp.int32(BITS_NEG_INF))
        ords.append(f32_bits_to_ord_i32(sb, valid))
        col_v = jnp.int32(r_rank * k) + pos
        cols.append(jnp.where(valid, col_v, jnp.int32(0x7FFFFFFF)))
        ib = my_i[:, b * 128 : (b + 1) * 128]
        owns.append(jnp.where(valid & is_mine, ib, jnp.int32(-1)))
      return tuple(ords), tuple(cols), tuple(owns)

    top_o, top_c, top_m = _chunk_triples(0)
    if not inputs_sorted:
      # Sort rank 0's chunk descending first.
      empty_o = tuple(
          jnp.full((8, 128), EMPTY_ORD_SENTINEL, jnp.int32)
          for _ in range(num_blocks)
      )
      empty_c = tuple(
          jnp.full((8, 128), 0x7FFFFFFF, jnp.int32) for _ in range(num_blocks)
      )
      empty_m = tuple(
          jnp.full((8, 128), -1, jnp.int32) for _ in range(num_blocks)
      )
      top_o, top_c, top_m = _merge_and_pack_stage(
          empty_o, empty_c, empty_m, top_o, top_c, top_m
      )

    for r_rank in range(1, dcp_size):
      co, cc, cm = _chunk_triples(r_rank)
      top_o, top_c, top_m = _merge_and_pack_stage(
          top_o, top_c, top_m, co, cc, cm
      )

    # Mask positions >= k (when k < k_cap) and left-pack owned winners >= 0.
    pack_keys = []
    pack_vals = []
    for b in range(num_blocks):
      pos = jnp.int32(b * 128) + lanes
      in_k = pos < jnp.int32(k)
      valid_own = (
          in_k & (top_o[b] > jnp.int32(EMPTY_ORD_SENTINEL)) & (top_m[b] >= 0)
      )
      val_b = jnp.where(valid_own, top_m[b], jnp.int32(-1))
      rev_pos = jnp.int32(k_cap) - pos
      pack_keys.append(jnp.where(valid_own, rev_pos, jnp.int32(-1)))
      pack_vals.append(val_b)

    _, packed_vals = bitonic_sort_blocks(
        tuple(pack_keys), tuple(pack_vals), descending=True
    )
    out_vmem[...] = jnp.concatenate(packed_vals, axis=1)

  packed = pl.pallas_call(
      _dcp_kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[
              pl.BlockSpec((8, dcp_size * k_cap), lambda rg, _: (rg, 0)),
              pl.BlockSpec((8, k_cap), lambda rg, _: (rg, 0)),
          ],
          out_specs=pl.BlockSpec((8, k_cap), lambda rg, _: (rg, 0)),
          grid=(padded_rows // 8,),
      ),
      out_shape=jax.ShapeDtypeStruct((padded_rows, k_cap), jnp.int32),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          disable_bounds_checks=True,
      ),
      name=f"tc_dcp_merge_compact_d{dcp_size}_k{k}",
  )(dcp_rank_arr, scores_2d, local_idxs_pad)
  return packed[:num_rows, :k]
