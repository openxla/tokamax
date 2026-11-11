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
import itertools
import math
import operator
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
    *,
    descending: bool = True,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
  """Returns the top `128 * R` of `top` (desc) and `cand` (asc), sorted."""
  merged_ord: list[jax.Array] = []
  merged_idx: list[jax.Array] = []
  for o_t, i_t, o_c, i_c in zip(top_ord, top_idx, cand_asc_ord, cand_asc_idx):
    take_cand = better_pair(o_c, i_c, o_t, i_t)
    merged_ord.append(jnp.where(take_cand, o_c, o_t))
    merged_idx.append(jnp.where(take_cand, i_c, i_t))
  return bitonic_merge_blocks(
      tuple(merged_ord), tuple(merged_idx), descending=descending
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
    tau_floor: jax.Array | None = None,
    inclusive: bool = False,
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
    tau_floor: Optional extra per-row lower bound on the k-th best order key
      (backed by `k` entries held outside this state); the effective
      threshold is `max(tau, tau_floor)`.
    inclusive: Whether to keep entries equal to the threshold. Required when
      incoming tiles are not in increasing index order relative to the
      entries backing the threshold, so that index tie-breaks stay exact.
  """
  w_len = len(tile_ords)

  def _eff(t):
    if tau_floor is not None:
      t = jnp.maximum(t, tau_floor)
    if inclusive:
      t = jnp.where(t == EMPTY_ORD_SENTINEL, t, t - 1)
    return t

  assert len(tile_idxs) == w_len
  assert num_blocks % w_len == 0
  num_phases = num_blocks // w_len

  tau = _eff(tau_ref[q_group])
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
        new_tau = _eff(new_tau)

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
# Per-lane sorted candidate lists (VALU-only streaming filter).
# --------------------------------------------------------------------------- #

# List levels kept in registers at a time by `lane_lists_insert_vmem`.
_LANE_LEVEL_BLOCK = 4


def lane_list_depth(num_blocks: int) -> int:
  """Depth `L` of the per-`(row, lane)` sorted candidate lists.

  Each row keeps `128 * L` candidates. The per-lane `num_blocks`-th best
  bounds the row's k-th best from below, and `L` is sized so that an entry
  falling off a list above that bound is rare for typical score streams.

  Args:
    num_blocks: Number of `(8, 128)` blocks of the top-k state.

  Returns:
    The list depth, a multiple of `num_blocks`.
  """
  if num_blocks <= 2:
    return 24
  if num_blocks == 4:
    return 32
  if num_blocks == 8:
    return 48
  return 4 * num_blocks


def init_lane_lists_vmem(
    lst_ord_ref: Any,
    lst_idx_ref: Any,
    *,
    q_group: jax.Array | int,
    depth: int,
) -> None:
  """Empties the per-lane candidate lists of `q_group`."""
  empty_o = jnp.full((8, 128), EMPTY_ORD_SENTINEL, dtype=jnp.int32)
  empty_i = jnp.full((8, 128), -1, dtype=jnp.int32)
  for i in range(depth):
    lst_ord_ref[q_group, i] = empty_o
    lst_idx_ref[q_group, i] = empty_i


def lane_lists_insert_vmem(
    lst_ord_ref: Any,
    lst_idx_ref: Any,
    *,
    q_group: jax.Array | int,
    tile_ords: tuple[jax.Array, ...],
    tile_idxs: tuple[jax.Array, ...],
    depth: int,
) -> tuple[list[jax.Array], list[jax.Array]]:
  """Inserts `W` tiles into the per-`(row, lane)` descending sorted lists.

  Each `(row, lane)` slot holds a descending list of `depth` entries across
  `depth` VMEM blocks. Tiles must arrive in strictly increasing KV index order
  per slot, so a new entry only moves above strictly smaller order keys and
  ties keep the lower index first. Pure VALU: one compare and four selects
  per tile and level, with no cross-lane permutes or scalar syncs.

  Args:
    lst_ord_ref: VMEM ref `[groups, depth, 8, 128]` of list order keys.
    lst_idx_ref: VMEM ref `[groups, depth, 8, 128]` of list indices.
    q_group: Query group (first index) of the refs to update.
    tile_ords: `W` incoming `(8, 128)` order-key tiles.
    tile_idxs: Indices matching `tile_ords`.
    depth: List depth `L`.

  Returns:
    `(ords, idxs)` of the `W` entries that fell off the end of the lists.
  """
  carry_o = list(tile_ords)
  carry_i = list(tile_idxs)
  # Levels are processed in small blocks whose loads are issued before the
  # previous block's stores, so the scheduler can overlap the
  # (level, tile) wavefront instead of serializing on VMEM dependencies.
  blocks = [
      range(lo, min(lo + _LANE_LEVEL_BLOCK, depth))
      for lo in range(0, depth, _LANE_LEVEL_BLOCK)
  ]

  def _load(levels):
    return [(lst_ord_ref[q_group, l], lst_idx_ref[q_group, l]) for l in levels]

  cur = _load(blocks[0])
  for b, levels in enumerate(blocks):
    nxt = _load(blocks[b + 1]) if b + 1 < len(blocks) else []
    out = []
    for a_o, a_i in cur:
      for w in range(len(tile_ords)):
        # Lists are sorted, so `x > a` holds from the insertion point onward.
        take = tile_ords[w] > a_o
        new_o = jnp.where(take, carry_o[w], a_o)
        new_i = jnp.where(take, carry_i[w], a_i)
        carry_o[w] = jnp.where(take, a_o, carry_o[w])
        carry_i[w] = jnp.where(take, a_i, carry_i[w])
        a_o, a_i = new_o, new_i
      out.append((a_o, a_i))
    for lvl, (a_o, a_i) in zip(levels, out):
      lst_ord_ref[q_group, lvl] = a_o
      lst_idx_ref[q_group, lvl] = a_i
    cur = nxt
  return carry_o, carry_i


def lane_lists_tau_floor(
    lst_ord_ref: Any,
    *,
    q_group: jax.Array | int,
    num_blocks: int,
) -> jax.Array:
  """Per-row lower bound on the k-th best key held by the lane lists.

  Every lane of a row holds `num_blocks` entries at or above its
  `num_blocks`-th entry, so the lane-minimum of that level is backed by
  `128 * num_blocks >= k` entries.

  Args:
    lst_ord_ref: VMEM ref `[groups, depth, 8, 128]` of list order keys.
    q_group: Query group (first index) of the refs to read.
    num_blocks: Number of `(8, 128)` blocks of the top-k state.

  Returns:
    `(8, 128)` per-row bound broadcast across lanes.
  """
  level = lst_ord_ref[q_group, num_blocks - 1]
  return jnp.broadcast_to(jnp.min(level, axis=1, keepdims=True), (8, 128))


def exclusive_threshold(tau: jax.Array) -> jax.Array:
  """Returns `t` with `x > t` iff `x >= tau` (keeps the sentinel)."""
  return jnp.where(tau == EMPTY_ORD_SENTINEL, tau, tau - 1)


def finalize_lane_lists_vmem(
    top_ord_ref: Any,
    top_idx_ref: Any,
    cand_ord_ref: Any,
    cand_idx_ref: Any,
    tau_ref: Any,
    lst_ord_ref: Any,
    lst_idx_ref: Any,
    *,
    q_groups: tuple[jax.Array | int, ...],
    k: int,
    num_blocks: int,
    depth: int,
    num_partitions: int = 1,
) -> list[tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]]:
  """Merges staged candidates and the lane lists into the final Top-K.

  Lane lists are merged `num_blocks` levels at a time. Levels are sorted per
  lane, so a level group whose best entry is below the running k-th best is
  skipped. The query groups in `q_groups` are processed in lock-step so their
  latency-bound sorting networks overlap; merging a group that is not needed
  is harmless.

  Args:
    top_ord_ref: VMEM ref of the running descending top-k order keys.
    top_idx_ref: VMEM ref of the indices matching `top_ord_ref`.
    cand_ord_ref: VMEM ref of the staged candidate order keys.
    cand_idx_ref: VMEM ref of the indices matching `cand_ord_ref`.
    tau_ref: VMEM ref of each row's current k-th best order key.
    lst_ord_ref: VMEM ref `[groups, depth + num_blocks, 8, 128]` of list
      order keys; the last `num_blocks` levels are scratch.
    lst_idx_ref: VMEM ref of the list indices, shaped like `lst_ord_ref`.
    q_groups: Query groups (first index) of the refs to finalize.
    k: Number of top entries to keep.
    num_blocks: Number of `(8, 128)` blocks in the top-k state.
    depth: List depth `L` (a multiple of `num_blocks`).
    num_partitions: Number of sublanes cooperating per query.

  Returns:
    Per query group, `(ords, idxs)` final descending blocks, partition-reduced.
  """
  assert depth % num_blocks == 0

  def _merge_level_into_top(ref_o, ref_i, lvl0):
    loaded = [
        (
            tuple(top_ord_ref[g, r] for r in range(num_blocks)),
            tuple(top_idx_ref[g, r] for r in range(num_blocks)),
            tuple(ref_o[g, lvl0 + r] for r in range(num_blocks)),
            tuple(ref_i[g, lvl0 + r] for r in range(num_blocks)),
        )
        for g in q_groups
    ]
    merged = []
    for cur_o, cur_i, blk_o, blk_i in loaded:
      asc_o, asc_i = bitonic_sort_blocks(blk_o, blk_i, descending=False)
      merged.append(bitonic_topk_merge_blocks(cur_o, cur_i, asc_o, asc_i))
    for g, (m_o, m_i) in zip(q_groups, merged):
      for r in range(num_blocks):
        top_ord_ref[g, r] = m_o[r]
        top_idx_ref[g, r] = m_i[r]
      tau_ref[g] = extract_row_kth_threshold(m_o, k)

  # The staged candidates become one more level group so that a single merge
  # body serves every group.
  for g in q_groups:
    for r in range(num_blocks):
      lst_ord_ref[g, depth + r] = cand_ord_ref[g, r]
      lst_idx_ref[g, depth + r] = cand_idx_ref[g, r]

  def _merge_level_group(grp, _):
    lvl0 = grp * num_blocks
    # An entry equal to the k-th best may still win on its lower index.
    needed = functools.reduce(
        operator.or_,
        [
            (lst_ord_ref[g, lvl0 + r] >= tau_ref[g])
            & (lst_ord_ref[g, lvl0 + r] > EMPTY_ORD_SENTINEL)
            for g, r in itertools.product(q_groups, range(num_blocks))
        ],
    )
    pl.when(jnp.any(needed))(
        lambda: _merge_level_into_top(lst_ord_ref, lst_idx_ref, lvl0)
    )
    return None

  lax.fori_loop(0, depth // num_blocks + 1, _merge_level_group, None)

  results = []
  for g in q_groups:
    final_o = tuple(top_ord_ref[g, r] for r in range(num_blocks))
    final_i = tuple(top_idx_ref[g, r] for r in range(num_blocks))
    if num_partitions > 1:
      final_o, final_i = reduce_sublane_partitions(
          final_o,
          final_i,
          num_partitions=num_partitions,
      )
    results.append((final_o, final_i))
  return results


# --------------------------------------------------------------------------- #
# Threshold-select Top-K over VMEM-resident order keys.
# --------------------------------------------------------------------------- #

# Smallest order key of a valid entry (`EMPTY_ORD_SENTINEL` marks empty ones).
MIN_VALID_ORD: int = EMPTY_ORD_SENTINEL + 1
_INT32_MAX: int = 0x7FFFFFFF
# Buckets filled per pass over the keys (`2 * _FILL_BUCKETS + 1` live VREGs).
_FILL_BUCKETS: int = 8


def _row_sum(x: jax.Array) -> jax.Array:
  """Broadcasts each sublane's sum over the 128 lanes back to `(8, 128)`."""
  return jnp.broadcast_to(jnp.sum(x, axis=1, keepdims=True), x.shape)


def _count_lanes(load_tile, n_blocks, tiles_per_block: int, pred) -> jax.Array:
  """Per-lane count of `pred(key, block, tile)` over every key tile."""
  num_acc = min(4, tiles_per_block)
  zero = jnp.zeros((8, 128), jnp.int32)

  def _body(b, accs):
    accs = list(accs)
    for t in range(tiles_per_block):
      hit = pred(load_tile(b, t), b, t)
      accs[t % num_acc] = accs[t % num_acc] + jnp.where(hit, 1, 0)
    return tuple(accs)

  accs = lax.fori_loop(0, n_blocks, _body, (zero,) * num_acc)
  total = accs[0]
  for acc in accs[1:]:
    total = total + acc
  return total


def _row_reduce(x: jax.Array, op) -> jax.Array:
  """Broadcasts each sublane's `op`-reduction over the lanes to `(8, 128)`."""
  return jnp.broadcast_to(op(x, axis=1, keepdims=True), x.shape)


def _key_stats(load_tile, n_blocks, tiles_per_block: int):
  """Per-lane valid count, max key, and min valid key over every key tile."""
  lo = jnp.int32(MIN_VALID_ORD)
  init = (
      jnp.zeros((8, 128), jnp.int32),
      jnp.full((8, 128), EMPTY_ORD_SENTINEL, jnp.int32),
      jnp.full((8, 128), _INT32_MAX, jnp.int32),
  )

  def _body(b, carry):
    cnt, mx, mn = carry
    for t in range(tiles_per_block):
      x = load_tile(b, t)
      valid = x >= lo
      cnt = cnt + jnp.where(valid, 1, 0)
      mx = jnp.maximum(mx, x)
      mn = jnp.minimum(mn, jnp.where(valid, x, _INT32_MAX))
    return cnt, mx, mn

  return lax.fori_loop(0, n_blocks, _body, init)


def select_threshold(
    load_tile,
    n_blocks: jax.Array | int,
    tiles_per_block: int,
    *,
    k: int,
    slack: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Finds a per-row key threshold that keeps every Top-K candidate.

  Keeps a per-row bracket `[lo, hi]` of order keys with
  `count(key >= lo) >= min(k, num_valid)` and `count(key > hi) < k`, starting
  from the row's smallest valid and largest key, and bisects it with one count
  pass per step. It stops once every row has at most `k + slack` candidates
  `>= lo` or an empty bracket, in which case `lo` is the exact `k`-th largest
  key of the row.

  Args:
    load_tile: `(block, tile) -> (8, 128)` int32 order keys; empty entries hold
      `EMPTY_ORD_SENTINEL`.
    n_blocks: Number of key blocks to scan (may be dynamic).
    tiles_per_block: Static number of `(8, 128)` tiles per block.
    k: Number of entries to keep per row.
    slack: Extra candidates per row tolerated before stopping early.

  Returns:
    `(t, row_count, lane_count)`: the `(8, 128)` row-broadcast threshold
    (always `>= MIN_VALID_ORD`), the row count of keys `>= t`, and the
    per-lane count of keys `>= t`.
  """
  sign = jnp.int32(EMPTY_ORD_SENTINEL)
  lane_valid, lane_max, lane_min = _key_stats(
      load_tile, n_blocks, tiles_per_block
  )
  row_valid = _row_sum(lane_valid)
  # Brackets live in the biased (unsigned order) domain: `u = key ^ sign`.
  lo_u = lax.bitwise_xor(
      jnp.maximum(_row_reduce(lane_min, jnp.min), MIN_VALID_ORD), sign
  )
  hi_u = lax.bitwise_xor(_row_reduce(lane_max, jnp.max), sign)

  def _converged(lo_u, hi_u, cnt_lo):
    return (cnt_lo <= k + slack) | (lo_u == hi_u)

  def _cond(state):
    return jnp.logical_not(state[0])

  def _body(state):
    _, lo_u, hi_u, cnt_lo, lane_lo = state
    # `width = hi - lo` as an unsigned value; the probe `lo + off` with
    # `off` in `[1, width]` always shrinks the bracket.
    width = hi_u - lo_u
    off = lax.shift_right_logical(width, 1) + lax.bitwise_and(width, 1)
    active = jnp.logical_not(_converged(lo_u, hi_u, cnt_lo))
    probe_u = jnp.where(active, lo_u + off, lo_u)
    probe = lax.bitwise_xor(probe_u, sign)
    lane = _count_lanes(
        load_tile, n_blocks, tiles_per_block, lambda x, b, tt: x >= probe
    )
    row = _row_sum(lane)
    keep = active & (row >= k)
    drop = active & (row < k)
    lo_u = jnp.where(keep, probe_u, lo_u)
    cnt_lo = jnp.where(keep, row, cnt_lo)
    lane_lo = jnp.where(keep, lane, lane_lo)
    hi_u = jnp.where(drop, probe_u - 1, hi_u)
    done = jnp.all(_converged(lo_u, hi_u, cnt_lo))
    return done, lo_u, hi_u, cnt_lo, lane_lo

  done = jnp.all(_converged(lo_u, hi_u, row_valid))
  _, lo_u, _, row_cnt, lane_cnt = lax.while_loop(
      _cond, _body, (done, lo_u, hi_u, row_valid, lane_valid)
  )
  t = lax.bitwise_xor(lo_u, sign)
  return t, row_cnt, lane_cnt


def select_tie_bound(
    load_tile,
    idx_tile,
    n_blocks: jax.Array | int,
    tiles_per_block: int,
    t: jax.Array,
    *,
    k: int,
    idx_bits: int,
) -> jax.Array:
  """Per-row index bound `j` so `key > t | (key == t & idx < j)` keeps k.

  Requires `t` to be the exact `k`-th largest key of each row. Ties at `t` are
  admitted by ascending index, matching `jax.lax.top_k`.

  Args:
    load_tile: `(block, tile) -> (8, 128)` int32 order keys.
    idx_tile: `(block, tile) -> (8, 128)` int32 KV indices of the keys.
    n_blocks: Number of key blocks to scan (may be dynamic).
    tiles_per_block: Static number of `(8, 128)` tiles per block.
    t: Row-broadcast exact `k`-th largest keys.
    k: Number of entries to keep per row.
    idx_bits: Bits needed to represent every valid index.

  Returns:
    The `(8, 128)` row-broadcast exclusive index bound.
  """
  num_gt = _row_sum(
      _count_lanes(load_tile, n_blocks, tiles_per_block, lambda x, b, tt: x > t)
  )

  def _body(i, q):
    bit = jnp.int32(idx_bits - 1) - i
    cand = lax.bitwise_or(q, lax.shift_left(jnp.int32(1), bit))
    num_eq = _row_sum(
        _count_lanes(
            load_tile,
            n_blocks,
            tiles_per_block,
            lambda x, b, tt: (x == t) & (idx_tile(b, tt) < cand),
        )
    )
    return jnp.where(num_gt + num_eq < k, cand, q)

  q = lax.fori_loop(0, idx_bits, _body, jnp.zeros((8, 128), jnp.int32))
  return q + 1


def _fill_buckets(
    load_tile,
    idx_tile,
    n_blocks,
    tiles_per_block: int,
    select,
    base_rank,
    num_buckets: int,
    block_ctx=None,
) -> tuple[list[jax.Array], list[jax.Array], jax.Array]:
  """Gathers the selected keys whose per-lane rank is in a bucket window.

  Each lane keeps a shift register of `num_buckets` slots and pushes its
  selected keys (in scan order) whose rank is in
  `[base_rank, base_rank + num_buckets)`, so the buckets end up holding exactly
  that rank window (in no particular order).

  Args:
    load_tile: `(block, tile) -> (8, 128)` int32 order keys.
    idx_tile: `(block, tile) -> (8, 128)` int32 KV indices of the keys.
    n_blocks: Number of key blocks to scan (may be dynamic).
    tiles_per_block: Static number of `(8, 128)` tiles per block.
    select: `(keys, idx, ctx, tile) -> bool` mask of the candidates, where
      `ctx` is `block_ctx(block)` and `tile` the static tile number.
    base_rank: Per-lane rank of the first bucket (static or dynamic).
    num_buckets: Number of `(8, 128)` buckets to fill.
    block_ctx: Optional `block -> ctx` evaluated once per block.

  Returns:
    `(ords, idxs, count)`: bucket lists (unfilled slots are empty with index
    `-1`) and the per-lane count of selected keys.
  """
  empty_o = jnp.full((8, 128), EMPTY_ORD_SENTINEL, dtype=jnp.int32)
  empty_i = jnp.full((8, 128), -1, dtype=jnp.int32)
  static_base = isinstance(base_rank, int)

  def _body(b, carry):
    cnt, ords, idxs = carry[0], list(carry[1]), list(carry[2])
    ctx = None if block_ctx is None else block_ctx(b)
    for t in range(tiles_per_block):
      x = load_tile(b, t)
      idx = idx_tile(b, t)
      sel = select(x, idx, ctx, t)
      if static_base and base_rank == 0:
        push = sel & (cnt < num_buckets)
      else:
        rel = cnt - base_rank
        push = sel & (rel >= 0) & (rel < num_buckets)
      for r in range(num_buckets - 1, 0, -1):
        ords[r] = jnp.where(push, ords[r - 1], ords[r])
        idxs[r] = jnp.where(push, idxs[r - 1], idxs[r])
      ords[0] = jnp.where(push, x, ords[0])
      idxs[0] = jnp.where(push, idx, idxs[0])
      cnt = cnt + jnp.where(sel, 1, 0)
    return cnt, tuple(ords), tuple(idxs)

  init = (
      jnp.zeros((8, 128), jnp.int32),
      (empty_o,) * num_buckets,
      (empty_i,) * num_buckets,
  )
  cnt, ords, idxs = lax.fori_loop(0, n_blocks, _body, init)
  return list(ords), list(idxs), cnt


def _merge_tree(chunks, descending: bool = True):
  """Sorted top `128 * R` of a list of unsorted `R`-block chunks.

  The chunks are sorted independently and merged pairwise, so the sorting
  networks of different chunks can be interleaved by the scheduler. `None`
  entries are empty chunks and cost nothing.

  Args:
    chunks: `(ords, idxs)` chunks of `R` `(8, 128)` blocks each, or `None`; a
      power-of-2 number of them.
    descending: Sort order of the result.

  Returns:
    `(ords, idxs)` of the sorted top `128 * R` per sublane, or `None` if every
    chunk is empty.
  """
  if len(chunks) == 1:
    if chunks[0] is None:
      return None
    return bitonic_sort_blocks(
        tuple(chunks[0][0]), tuple(chunks[0][1]), descending=descending
    )
  half = len(chunks) // 2
  if all(c is None for c in chunks[half:]):
    return _merge_tree(chunks[:half], descending)
  if all(c is None for c in chunks[:half]):
    return _merge_tree(chunks[half:], descending)
  hi = _merge_tree(chunks[:half], True)
  lo = _merge_tree(chunks[half:], False)
  assert hi is not None and lo is not None
  hi_o, hi_i = hi
  lo_o, lo_i = lo
  return bitonic_topk_merge_blocks(
      hi_o, hi_i, lo_o, lo_i, descending=descending
  )


def _gather_into_top(
    load_tile,
    idx_tile,
    n_blocks,
    tiles_per_block: int,
    select,
    lane_cnt: jax.Array,
    top_ord_ref: Any,
    top_idx_ref: Any,
    *,
    q_group: jax.Array | int,
    num_blocks: int,
    block_ctx=None,
    max_chunks: int | None = None,
    extra_chunk=None,
) -> None:
  """Writes the sorted top `128 * num_blocks` selected keys to the top refs.

  Gathers + sorts up to a few rounds of `num_blocks` candidates per lane at
  once (the number of rounds is set by the busiest lane, `lane_cnt`), then
  merges any remaining rounds one at a time.

  Args:
    load_tile: `(block, tile) -> (8, 128)` int32 order keys.
    idx_tile: `(block, tile) -> (8, 128)` int32 KV indices of the keys.
    n_blocks: Number of key blocks to scan (may be dynamic).
    tiles_per_block: Static number of `(8, 128)` tiles per block.
    select: Candidate mask, see `_fill_buckets`.
    lane_cnt: Per-lane number of candidates.
    top_ord_ref: VMEM ref receiving the descending Top-K order keys.
    top_idx_ref: VMEM ref receiving the indices matching `top_ord_ref`.
    q_group: Query group (first index) of the refs to write.
    num_blocks: Number of `(8, 128)` blocks in the Top-K state.
    block_ctx: Optional per-block context, see `_fill_buckets`.
    max_chunks: Most rounds to gather and sort at once.
    extra_chunk: Optional unsorted `(ords, idxs)` chunk of at most
      `num_blocks` blocks to merge in as well.
  """
  if extra_chunk is not None:
    pad = num_blocks - len(extra_chunk[0])
    extra_chunk = (
        list(extra_chunk[0]) + [_empty_ord_block()] * pad,
        list(extra_chunk[1]) + [_empty_idx_block()] * pad,
    )

  def _fill_ranks(base_rank, count: int):
    ords, idxs = [], []
    for s in range(0, count, _FILL_BUCKETS):
      o, i, _ = _fill_buckets(
          load_tile,
          idx_tile,
          n_blocks,
          tiles_per_block,
          select,
          base_rank + s,
          min(_FILL_BUCKETS, count - s),
          block_ctx,
      )
      ords += o
      idxs += i
    return ords, idxs

  if max_chunks is None:
    max_chunks = max(1, min(4, 16 // num_blocks))
  num_rounds = (jnp.max(lane_cnt) + (num_blocks - 1)) // num_blocks
  chunk_options = sorted({1, min(2, max_chunks), max_chunks})
  for i, num_chunks in enumerate(chunk_options):
    conds = []
    if i > 0:
      conds.append(num_rounds > chunk_options[i - 1])
    if i < len(chunk_options) - 1:
      conds.append(num_rounds <= num_chunks)

    def _gather_rounds(num_chunks=num_chunks):
      ords, idxs = _fill_ranks(0, num_chunks * num_blocks)
      chunks = [
          (ords[c * num_blocks : (c + 1) * num_blocks],
           idxs[c * num_blocks : (c + 1) * num_blocks])
          for c in range(num_chunks)
      ]
      if extra_chunk is not None:
        chunks.append(extra_chunk)
      chunks += [None] * (next_power_of_2(len(chunks)) - len(chunks))
      top = _merge_tree(chunks)
      assert top is not None
      _write_top(top_ord_ref, top_idx_ref, q_group, *top)

    if conds:
      pl.when(functools.reduce(jnp.logical_and, conds))(_gather_rounds)
    else:
      _gather_rounds()

  def _extra_round(rnd, _):
    ords, idxs = _fill_ranks(rnd * num_blocks, num_blocks)
    _merge_into_top(top_ord_ref, top_idx_ref, q_group, num_blocks, ords, idxs)
    return None

  lax.fori_loop(max_chunks, num_rounds, _extra_round, None)


def _empty_ord_block() -> jax.Array:
  return jnp.full((8, 128), EMPTY_ORD_SENTINEL, dtype=jnp.int32)


def _empty_idx_block() -> jax.Array:
  return jnp.full((8, 128), -1, dtype=jnp.int32)


def _write_top(top_ord_ref, top_idx_ref, q_group, top_o, top_i) -> None:
  for r, (o, i) in enumerate(zip(top_o, top_i)):
    top_ord_ref[q_group, r] = o
    top_idx_ref[q_group, r] = i


def _merge_into_top(
    top_ord_ref, top_idx_ref, q_group, num_blocks: int, ords, idxs
) -> None:
  """Merges up to `num_blocks` unsorted candidate blocks into the top refs."""
  pad = num_blocks - len(ords)
  cand_o, cand_i = bitonic_sort_blocks(
      tuple(ords) + (_empty_ord_block(),) * pad,
      tuple(idxs) + (_empty_idx_block(),) * pad,
      descending=False,
  )
  cur_o = tuple(top_ord_ref[q_group, r] for r in range(num_blocks))
  cur_i = tuple(top_idx_ref[q_group, r] for r in range(num_blocks))
  _write_top(
      top_ord_ref,
      top_idx_ref,
      q_group,
      *bitonic_topk_merge_blocks(cur_o, cur_i, cand_o, cand_i),
  )


def select_topk_vmem(
    load_tile,
    idx_tile,
    n_blocks: jax.Array | int,
    tiles_per_block: int,
    top_ord_ref: Any,
    top_idx_ref: Any,
    *,
    q_group: jax.Array | int,
    k: int,
    num_blocks: int,
    idx_bits: int,
) -> None:
  """Exact per-row Top-K of VMEM-resident keys via threshold selection.

  1. `select_threshold` finds a per-row key threshold `t` keeping at least
     `k` (or every valid) candidates using only compare + count passes.
  2. If a row still has many more than `k` candidates (heavy ties at the
     exact `k`-th key), `select_tie_bound` admits the ties by ascending index
     so exactly `k` candidates remain.
  3. The candidates are gathered `num_blocks` per lane at a time and merged
     into the descending Top-K with bitonic sorts and merges, keeping
     `jax.lax.top_k`'s `(score, -index)` order.

  Args:
    load_tile: `(block, tile) -> (8, 128)` int32 order keys.
    idx_tile: `(block, tile) -> (8, 128)` int32 KV indices of the keys.
    n_blocks: Number of key blocks to scan (may be dynamic).
    tiles_per_block: Static number of `(8, 128)` tiles per block.
    top_ord_ref: VMEM ref receiving the descending Top-K order keys.
    top_idx_ref: VMEM ref receiving the indices matching `top_ord_ref`.
    q_group: Query group (first index) of the refs to write.
    k: Number of top entries to keep.
    num_blocks: Number of `(8, 128)` blocks in the Top-K state.
    idx_bits: Bits needed to represent every valid index.
  """
  k_cap = 128 * num_blocks
  t, row_cnt, lane_cnt = select_threshold(
      load_tile, n_blocks, tiles_per_block, k=k, slack=max(128, k_cap // 4)
  )
  heavy_ties = jnp.any(row_cnt > 2 * k_cap)

  def _tie_mode():
    j = select_tie_bound(
        load_tile, idx_tile, n_blocks, tiles_per_block, t, k=k,
        idx_bits=idx_bits,
    )
    j = jnp.where(row_cnt > 2 * k_cap, j, jnp.int32(_INT32_MAX))
    lanes = _count_lanes(
        load_tile,
        n_blocks,
        tiles_per_block,
        lambda x, b, tt: (x > t) | ((x == t) & (idx_tile(b, tt) < j)),
    )
    return j, lanes

  def _plain_mode():
    return jnp.full((8, 128), _INT32_MAX, jnp.int32), lane_cnt

  bound, lane_cnt = lax.cond(heavy_ties, _tie_mode, _plain_mode)
  _gather_into_top(
      load_tile,
      idx_tile,
      n_blocks,
      tiles_per_block,
      lambda x, idx, ctx, tt: (x > t) | ((x == t) & (idx < bound)),
      lane_cnt,
      top_ord_ref,
      top_idx_ref,
      q_group=q_group,
      num_blocks=num_blocks,
  )


def summarize_key_block(load_tile, tiles_per_block: int):
  """Per-lane max key of a block and the first tile holding it."""
  best = load_tile(0)
  arg = jnp.zeros((8, 128), jnp.int32)
  for t in range(1, tiles_per_block):
    x = load_tile(t)
    better = x > best
    best = jnp.where(better, x, best)
    arg = jnp.where(better, t, arg)
  return best, arg


def select_topk_two_level(
    load_tile,
    idx_tile,
    load_cell_max,
    load_cell_arg,
    n_blocks: jax.Array | int,
    tiles_per_block: int,
    top_ord_ref: Any,
    top_idx_ref: Any,
    *,
    q_group: jax.Array | int,
    k: int,
    num_blocks: int,
    idx_bits: int,
) -> None:
  """Exact per-row Top-K using per-block lane maxima ("cells") as level 1.

  Each cell is the max key of one `(row, lane, block)` (see
  `summarize_key_block`). The threshold search and most of the candidate
  gathering run on the cells, which are `tiles_per_block` times fewer than the
  keys. Thresholding the cells is a valid lower bound for the keys
  (`count(key >= t) >= count(cell >= t)`), and every candidate key is either a
  cell or a "residual" key that is not its cell's max. A single pass over the
  keys gathers the (few) residual candidates. Heavy ties fall back to
  `select_topk_vmem` on the keys.

  Args:
    load_tile: `(block, tile) -> (8, 128)` int32 order keys.
    idx_tile: `(block, tile) -> (8, 128)` int32 KV indices of the keys; must be
      `idx_tile(b, 0) + 128 * tile`.
    load_cell_max: `block -> (8, 128)` max key of each lane in the block.
    load_cell_arg: `block -> (8, 128)` tile of the first max key in the block.
    n_blocks: Number of key blocks to scan (may be dynamic).
    tiles_per_block: Static number of `(8, 128)` tiles per block.
    top_ord_ref: VMEM ref receiving the descending Top-K order keys.
    top_idx_ref: VMEM ref receiving the indices matching `top_ord_ref`.
    q_group: Query group (first index) of the refs to write.
    k: Number of top entries to keep.
    num_blocks: Number of `(8, 128)` blocks in the Top-K state.
    idx_bits: Bits needed to represent every valid index.
  """
  k_cap = 128 * num_blocks
  # Cells are cheap to scan, so search them to a tight slack: fewer candidates
  # means fewer gather rounds and sorts.
  t, row_cnt, lane_cnt = select_threshold(
      lambda b, _: load_cell_max(b),
      n_blocks,
      1,
      k=k,
      slack=max(64, k_cap // (4 if num_blocks <= 8 else 32)),
  )
  heavy_ties = jnp.any(row_cnt > 2 * k_cap)

  @pl.when(heavy_ties)
  def _exact_on_keys():
    select_topk_vmem(
        load_tile,
        idx_tile,
        n_blocks,
        tiles_per_block,
        top_ord_ref,
        top_idx_ref,
        q_group=q_group,
        k=k,
        num_blocks=num_blocks,
        idx_bits=idx_bits,
    )

  @pl.when(jnp.logical_not(heavy_ties))
  def _two_level():
    # Rows with fewer than `k` valid cells keep every valid key.
    thr = jnp.where(row_cnt < k, jnp.int32(MIN_VALID_ORD), t)

    def _residual(x, idx, arg, tt):
      del idx
      return (x >= thr) & (arg != tt)

    res_buckets = min(num_blocks, 4)

    def _fill_residuals(base_rank):
      return _fill_buckets(
          load_tile,
          idx_tile,
          n_blocks,
          tiles_per_block,
          _residual,
          base_rank,
          res_buckets,
          load_cell_arg,
      )

    res_o, res_i, res_cnt = _fill_residuals(0)
    max_res = jnp.max(res_cnt)
    many_residuals = max_res > 4 * res_buckets

    @pl.when(many_residuals)
    def _tied_cells():
      # Many keys tie with their cell max: run the exact search on the keys.
      select_topk_vmem(
          load_tile,
          idx_tile,
          n_blocks,
          tiles_per_block,
          top_ord_ref,
          top_idx_ref,
          q_group=q_group,
          k=k,
          num_blocks=num_blocks,
          idx_bits=idx_bits,
      )

    @pl.when(jnp.logical_not(many_residuals))
    def _cells_and_residuals():
      _gather_into_top(
          lambda b, _: load_cell_max(b),
          lambda b, _: idx_tile(b, 0) + load_cell_arg(b) * 128,
          n_blocks,
          1,
          lambda x, idx, ctx, tt: x >= thr,
          lane_cnt,
          top_ord_ref,
          top_idx_ref,
          q_group=q_group,
          num_blocks=num_blocks,
          max_chunks=4 if num_blocks <= 8 else 2,
          extra_chunk=(res_o, res_i),
      )

      def _extra(rnd, _):
        ords, idxs, _ = _fill_residuals(rnd * res_buckets)
        _merge_into_top(
            top_ord_ref, top_idx_ref, q_group, num_blocks, ords, idxs
        )
        return None

      lax.fori_loop(
          1, (max_res + (res_buckets - 1)) // res_buckets, _extra, None
      )


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
