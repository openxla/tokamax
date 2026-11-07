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

"""Arrival combine: weighted top-k reduction of the arrived expert rows.

`arrivals[pos]` in XLA is an offloaded gather whose output is written to HBM
at every size, so the gathered rows would make a full round trip through HBM
before the weighted sum reads them, followed by a relayout copy because
[t, lane blocks, 128] and [t, hidden] do not share a tiling.
Here each arrival row lands in VMEM as one DMA and is reduced where it lands,
so the combine moves the rows in and the tokens out and nothing else.

The row DMAs are emitted BESIDE the reduction rather than in a loop of their
own: they are scalar-unit work against a body that runs the vector unit, so
in one basic block the scheduler packs them into slots the reduction leaves
empty.

What the schedule costs is the DMA descriptor issue, not the arithmetic, so
the two things worth doing to it are shortening the scalar sequence a row
needs and keeping both DMA queues fed.

This is a variant of `fused_moe.kernel`'s combine kernel that also covers
small (decode) token counts with a single-parity schedule.
"""

import dataclasses
import threading
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_moe import host as fused_host
from tokamax._src.ops.experimental.fused_moe import kernel as fused_kernel

HIDDEN_LANE_BLOCK = fused_host.HIDDEN_LANE_BLOCK
U32_SUBLANE_TILE = fused_host.U32_SUBLANE_TILE
pow2_shift = fused_host.pow2_shift

# The pipelined schedule's constants, shared with `fused_moe.kernel`: tokens
# per staged step, tokens per output relayout group, tokens per reduction
# chunk, the smallest token count the pipelined schedule applies to, the
# buffer rotation depth, and the steps of row lead.
COMBINE_TOKENS = fused_kernel.COMBINE_TOKENS
COMBINE_GROUP = fused_kernel.COMBINE_GROUP
COMBINE_CHUNK = fused_kernel.COMBINE_CHUNK
COMBINE_MIN_TOKENS = fused_kernel.COMBINE_MIN_TOKENS
COMBINE_PARITIES = fused_kernel.COMBINE_PARITIES
COMBINE_LEAD = fused_kernel.COMBINE_LEAD


def _combine_is_pipelined(t_local, tb):
  """Whether `t_local` and `tb` satisfy the 4-parity pipelined combine schedule."""
  if t_local < COMBINE_MIN_TOKENS or tb < COMBINE_GROUP:
    return False
  steps = t_local // tb
  return (
      not t_local % tb
      and not tb % COMBINE_GROUP
      and steps > COMBINE_LEAD
      and steps >= COMBINE_PARITIES
      and not steps % COMBINE_PARITIES
  )


def combine_step_tokens(t_local):
  """Returns the tokens a combine step stages.

  Tokens a combine step stages, or 0 where this path does not apply.

  Above COMBINE_MIN_TOKENS, returns a step width `tb` that satisfies the
  4-parity pipelined schedule (`_combine_is_pipelined`). For smaller batch
  sizes (`8 <= t_local < COMBINE_MIN_TOKENS`, `t_local % 8 == 0`), returns a
  single-parity step width `tb <= COMBINE_TOKENS` dividing `t_local` so the
  Pallas combine kernel runs at decode batch sizes too (no XLA gather
  fallback at any t bucket).

  Args:
    t_local: Tokens per shard.

  Returns:
    The step width `tb`, or 0 where the Pallas combine does not apply.
  """
  if t_local < U32_SUBLANE_TILE or t_local % U32_SUBLANE_TILE:
    return 0
  tb = COMBINE_TOKENS
  while tb >= COMBINE_GROUP:
    if _combine_is_pipelined(t_local, tb):
      return tb
    tb //= 2
  tb = min(COMBINE_TOKENS, t_local)
  while tb >= U32_SUBLANE_TILE:
    if not t_local % tb:
      return tb
    tb //= 2
  return 0


@dataclasses.dataclass(frozen=True)
class Config:
  """Static configuration for the arrival combine kernel."""

  t_local: int
  topk: int
  hidden: int
  lane_blocks: int
  wire_dtype: jax.typing.DTypeLike
  out_dtype: jax.typing.DTypeLike
  tb: int
  pipelined: bool
  steps: int
  gt: int
  groups: int
  chunk_w: int
  parities: int
  lanes: int = HIDDEN_LANE_BLOCK

  @classmethod
  def create(
      cls,
      *,
      t_local: int,
      topk: int,
      hidden: int,
      lane_blocks: int,
      wire_dtype: jax.typing.DTypeLike,
      out_dtype: jax.typing.DTypeLike,
  ):
    """Derives the combine kernel's step, group and chunk sizes.

    Args:
      t_local: Tokens per shard.
      topk: Experts per token.
      hidden: Model width.
      lane_blocks: 128-lane blocks per row.
      wire_dtype: Element type of the arrived rows.
      out_dtype: Element type of the combined output.

    Returns:
      The Config.
    """
    tb = combine_step_tokens(t_local)
    pipelined = _combine_is_pipelined(t_local, tb)
    steps = t_local // tb
    gt = COMBINE_GROUP if pipelined else min(COMBINE_GROUP, tb)
    groups = tb // gt
    chunk_w = COMBINE_CHUNK if pipelined else min(COMBINE_CHUNK, tb)
    parities = COMBINE_PARITIES if pipelined else 1
    return cls(
        t_local=t_local,
        topk=topk,
        hidden=hidden,
        lane_blocks=lane_blocks,
        wire_dtype=wire_dtype,
        out_dtype=out_dtype,
        tb=tb,
        pipelined=pipelined,
        steps=steps,
        gt=gt,
        groups=groups,
        chunk_w=chunk_w,
        parities=parities,
    )

  @property
  def kernel_name(self) -> str:
    return f"dedup_moe_combine_t{self.t_local}_k{self.topk}_b{self.tb}"


def _build_combine_kernel(
    *, t_local, topk, hidden, lane_blocks, wire_dtype, out_dtype
):
  """out[i] = sum_k coef[i, k] * arrivals[pos[i * topk + k]].

  `pos` is a scalar-prefetch table of arrival rows, already clamped into the
  buffer; `coef` is the arrival row's scale folded with the router weight and
  rides SMEM beside it, so a slot's weight reaches the vector unit as a
  scalar splat. The result is [t_local, hidden] in the layer's own output
  layout.

  An arrival row is [lane blocks, 128] -- one register -- while the output
  is [tokens, hidden], which is that layout transposed: a token's hidden
  axis lies along sublanes in the row and along lanes in the output. The sum
  therefore runs in the ROW's layout and only its bf16 result is relaid out,
  so the sublane shuffle sees one bf16 token per group instead of topk
  packed-f8 rows, and the live accumulator is one token wide.

  Args:
    t_local: Tokens per shard.
    topk: Experts per token.
    hidden: Model width.
    lane_blocks: 128-lane blocks per row.
    wire_dtype: Element type of the arrived rows.
    out_dtype: Element type of the output.

  Returns:
    `fn(arrivals, coef, pos)` returning [t_local, hidden].
  """
  cfg = Config.create(
      t_local=t_local,
      topk=topk,
      hidden=hidden,
      lane_blocks=lane_blocks,
      wire_dtype=wire_dtype,
      out_dtype=out_dtype,
  )
  tb = cfg.tb
  pipelined = cfg.pipelined
  steps = cfg.steps
  gt = cfg.gt
  groups = cfg.groups
  chunk_w = cfg.chunk_w
  parities = cfg.parities
  lanes = cfg.lanes

  def kernel(
      pos_sm,
      coef_sm,
      arr_hbm,
      out_hbm,
      rows_vm,
      tile_vm,
      out_vm,
      row_sems,
      out_sems,
  ):
    def start_row(base, parity, i, k):
      # Alternate queues by slot: this kernel is one 4 KiB descriptor per
      # arrival row and an in-order queue retires descriptors at a fixed
      # rate, which for a row this small is slower than the row's own
      # bytes. Both queues signal the one semaphore the step waits on,
      # which counts bytes and not descriptors.
      pltpu.make_async_copy(
          arr_hbm.at[pos_sm[(base + i) * topk + k]],
          rows_vm.at[parity, k * tb + i],
          row_sems.at[parity],
      ).start(priority=k % 2)

    def reduce_token(base, parity, i):
      """One token's weighted sum, in the arrival rows' own layout."""
      acc: Any = None
      for k in range(topk):
        term = (
            rows_vm[parity, k * tb + i].astype(jnp.float32)
            * coef_sm[(base + i) * topk + k]
        )
        acc = term if acc is None else acc + term
      tile_vm[i] = acc.astype(out_dtype)

    if not pipelined:
      for step in range(steps):
        base = step * tb
        # Step 1: Issue arrival row DMAs for the current step.
        for i in range(tb):
          for k in range(topk):
            start_row(base, 0, i, k)
        # Step 2: Wait for arrival row DMAs to complete.
        pltpu.make_async_copy(
            rows_vm.at[0], rows_vm.at[0], row_sems.at[0]
        ).wait()

        # Step 3: Reduce top-k arrival rows for each token in chunks.
        def one_chunk_small(chunk, carry, base=base):
          for u in range(chunk_w):
            reduce_token(base, 0, chunk * chunk_w + u)
          return carry

        lax.fori_loop(0, tb // chunk_w, one_chunk_small, jnp.int32(0))
        # Step 4: Transpose reduced token groups into output VMEM layout.
        for g in range(groups):
          out_vm[0, pl.ds(g * gt, gt)] = tile_vm[pl.ds(g * gt, gt)].reshape(
              gt, hidden
          )
        # Step 5: Store output step from VMEM to HBM and wait.
        pltpu.make_async_copy(
            out_vm.at[0],
            out_hbm.at[pl.ds(step * tb, tb)],
            out_sems.at[0],
        ).start()
        pltpu.make_async_copy(out_vm.at[0], out_vm.at[0], out_sems.at[0]).wait()
      return

    def one_step(step, parity):
      # Step 1: Compute next prefetch batch offset and parity.
      # The step's rows were started a step ago; the next step's are
      # started beside this one's reduction, so their descriptor issue
      # rides in slots the vector work leaves empty. The last step
      # re-starts its own rows rather than branching -- the copy is never
      # read and is drained below.
      nxt = jnp.minimum(step + COMBINE_LEAD, steps - 1) * tb
      base = step * tb
      npar = (parity + COMBINE_LEAD) % COMBINE_PARITIES
      # Step 2: Wait for the input DMA of this parity and the output DMA of
      # the reused parity.
      pltpu.make_async_copy(
          rows_vm.at[parity], rows_vm.at[parity], row_sems.at[parity]
      ).wait()

      @pl.when(step >= COMBINE_PARITIES)
      def _():
        pltpu.make_async_copy(
            out_vm.at[parity], out_vm.at[parity], out_sems.at[parity]
        ).wait()

      # Step 3: Reduce top-k arrival rows while issuing prefetch DMAs for `nxt`.
      def one_chunk(chunk, carry):
        for u in range(COMBINE_CHUNK):
          i = chunk * COMBINE_CHUNK + u
          for k in range(topk):
            start_row(nxt, npar, i, k)
          reduce_token(base, parity, i)
        return carry

      lax.fori_loop(0, tb // COMBINE_CHUNK, one_chunk, jnp.int32(0))
      # Step 4: Relayout reduced token groups into `[gt, hidden]` output VMEM.
      for g in range(groups):
        out_vm[parity, pl.ds(g * gt, gt)] = tile_vm[pl.ds(g * gt, gt)].reshape(
            gt, hidden
        )
      # Step 5: Start async output DMA from VMEM to HBM.
      pltpu.make_async_copy(
          out_vm.at[parity],
          out_hbm.at[pl.ds(step * tb, tb)],
          out_sems.at[parity],
      ).start()

    def step_pair(pair, carry):
      for parity in range(COMBINE_PARITIES):
        one_step(pair * COMBINE_PARITIES + parity, parity)
      return carry

    # Prologue: Prefetch first COMBINE_LEAD steps of arrival rows.
    for p in range(COMBINE_LEAD):
      for i in range(tb):
        for k in range(topk):
          start_row(p * tb, p, i, k)
    # Body: Main pipelined reduction loop over 4-parity groups.
    lax.fori_loop(0, steps // COMBINE_PARITIES, step_pair, jnp.int32(0))
    # Epilogue: The last COMBINE_LEAD steps re-start the final step's rows into
    # the buffers the loop no longer reads; drain them, and the stores.
    for p in range(COMBINE_LEAD):
      pltpu.make_async_copy(rows_vm.at[p], rows_vm.at[p], row_sems.at[p]).wait()
    for p in range(COMBINE_PARITIES):
      pltpu.make_async_copy(out_vm.at[p], out_vm.at[p], out_sems.at[p]).wait()

  hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
  call = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[hbm],
          out_specs=hbm,
          scratch_shapes=(
              pltpu.VMEM((parities, topk * tb, lane_blocks, lanes), wire_dtype),
              pltpu.VMEM((tb, lane_blocks, lanes), out_dtype),
              pltpu.VMEM((parities, tb, hidden), out_dtype),
              pltpu.SemaphoreType.DMA((parities,)),
              pltpu.SemaphoreType.DMA((parities,)),
          ),
          grid=(),
      ),
      out_shape=jax.ShapeDtypeStruct((t_local, hidden), out_dtype),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=fused_host.vmem_limit(), disable_bounds_checks=True
      ),
      name=cfg.kernel_name,
  )

  def fn(arrivals, coef, pos):
    # Both tables flatten TOKEN-major, so a token's topk entries are one
    # contiguous SMEM run and its 2 * topk scalar reads share a single
    # address computation. That is also the order the plan builds them in,
    # so neither flatten is a transpose.
    if pos.shape != (t_local, topk) or coef.shape != (t_local, topk):
      raise ValueError(
          "the combine takes token-major "
          f"[{t_local}, {topk}] tables, got pos "
          f"{pos.shape} and coef {coef.shape}"
      )
    return call(pos.astype(jnp.int32).reshape(-1), coef.reshape(-1), arrivals)

  return fn


_COMBINE_CACHE = {}
_COMBINE_CACHE_LOCK = threading.Lock()


def build_combine_kernel(**kwargs):
  """Memoizing front door (as fused_kernel.kernel.build_fused_ep_moe_kernel)."""
  key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
  fn = _COMBINE_CACHE.get(key)
  if fn is None:
    with _COMBINE_CACHE_LOCK:
      fn = _COMBINE_CACHE.get(key)
      if fn is None:
        fn = _build_combine_kernel(**kwargs)
        _COMBINE_CACHE[key] = fn
  return fn


def combine_arrivals(
    arrivals, arrival_scales, pos, mirror_pos, topk_weights, out_dtype
):
  """Combine arrivals [rows, lane blocks, 128] into [t_local, hidden].

  This is the layer's combine: every routed row the fused kernel pushed to
  this shard is weighted by its router weight and summed into its token.
  Each of a token's topk selections lands as its own arrival row (the
  combine is not de-duplicated), so `pos` [t_local, topk] names one
  arrival row per selection slot.

  Two tables index two different objects, so each owns its own clamp.
  `pos` is the arrival ROW: a row of `arrivals`, and on the kernel path
  below it is a DMA address. `mirror_pos` is a position in the wire's scale
  mirror, which is [sublanes, tile_m] because a transport can only address
  the mirror a sublane at a time and a sublane holds tile_m f32 whatever is
  written in it -- so a whole source TILE's scales are one sublane and the
  mirror position is NOT the arrival row. An unquantized (bf16) wire
  carries no scales: the kernel moves no mirror then, and neither table is
  read for it.

  Wherever t_local is a multiple of 8 (every t bucket) the row gather and
  the weighted sum are one Mosaic call (combine_step_tokens): XLA
  answers `arrivals[rows]` with an offloaded gather whose output is in HBM
  at any size, so the rows would be written out and read straight back
  before the sum sees them. Any other t_local takes the fused XLA epilogue
  below.

  Both paths fold the arrival row's scale and the router weight into one
  coefficient over K * t_local values, so each of the K * t_local * hidden
  elements is multiplied once rather than twice.

  Args:
    arrivals: [rows, lane blocks, 128] arrived rows.
    arrival_scales: [sublanes, tile_m] f32 scale mirror, or None.
    pos: [t_local, topk] arrival row of each selection.
    mirror_pos: [t_local, topk] scale-mirror position of each selection.
    topk_weights: [t_local, topk] router weights.
    out_dtype: Output element type.

  Returns:
    [t_local, hidden] combined tokens.
  """
  t_local, topk = pos.shape
  recv_rows, lane_blocks, lanes = arrivals.shape
  hidden = lane_blocks * lanes
  # A gather clamps an out-of-range index to the last one; a DMA does not,
  # and a sublane index clamped on its own would pair the last sublane with
  # the wrong lane. Neither position is ever negative.
  rows = jnp.minimum(pos, recv_rows - 1)
  if arrival_scales is None or arrivals.dtype == jnp.bfloat16:
    # Unquantized wire rows: there is no row scale to apply.
    coef = topk_weights.astype(jnp.float32)
  else:
    mirror_sublanes, tile_m = arrival_scales.shape
    shift = pow2_shift(tile_m, "the mirror's lane count")
    scl = jnp.minimum(mirror_pos, mirror_sublanes * tile_m - 1)
    row_scales = arrival_scales[jnp.right_shift(scl, shift), scl & (tile_m - 1)]
    coef = row_scales * topk_weights.astype(jnp.float32)
  if combine_step_tokens(t_local):
    combine = build_combine_kernel(
        t_local=t_local,
        topk=topk,
        hidden=hidden,
        lane_blocks=lane_blocks,
        wire_dtype=arrivals.dtype,
        out_dtype=out_dtype,
    )
    return combine(arrivals, coef, rows)
  rows_fp8 = arrivals[pos.T.reshape(-1)]  # f8 [K*t, lane blocks, 128]
  coef_k = coef.T.reshape(-1)  # k-major [K*t_local]
  terms = []
  for k in range(topk):
    slot_rows = lax.slice(
        rows_fp8, (k * t_local, 0, 0), ((k + 1) * t_local, lane_blocks, lanes)
    )
    slot_coef = lax.slice(coef_k, (k * t_local,), ((k + 1) * t_local,))
    terms.append(slot_rows.astype(jnp.float32) * slot_coef[:, None, None])
  total = terms[0]
  for term in terms[1:]:
    total = total + term
  return total.astype(out_dtype).reshape(t_local, hidden)
