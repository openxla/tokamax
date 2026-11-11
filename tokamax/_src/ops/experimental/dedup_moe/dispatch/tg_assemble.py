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
"""The fused kernel's token-gather table, assembled from the sender lists.

Every source s sends every destination one list (source_plan.build_send_lists):
a 128-word header, then one entry per pair of s computing on the destination --
the receive row of the pair's token there (and, for fp8 rows, a second plane
with the token's activation-scale bits) -- local expert ascending, then pair
order, which is exactly the order of the run (expert, s) in the destination's
slab. The lists land in `tg_land` [ep * list_rows, 128] i32, source s at row
s * list_rows.

A run (expert, s) starts at an arbitrary ROWBLK-aligned slab row, and HBM
int32 DMAs need 128-aligned offsets, so the lists cannot land in
slab order. `tg_assemble` puts them there: it waits for every list on the
start kernel's `tg_sem` (byte-exact: ep headers plus the bodies, all counts
known from the exchanged routing), DMAs the landed rows into VMEM, and copies
each run to its slab offset in a zeroed VMEM image of the table -- 8-row
aligned tiles, a dynamic lane rotate, a sublane select and a masked
read-modify-write, O(runs + live / 1024) tile steps -- then clips it to
[0, T - 1] (the kernel gathers with bounds checks off) and writes it out. The
same pass writes the fp8 activation-scale slab from the second plane. Pad rows
stay 0, as in the reference plan's clipped scatter.

`impl="smem"` is the fallback for a Mosaic without dynamic lane rotates: the
same copy as a scalar loop in SMEM, for small tables only.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.dispatch import source_plan
from tokamax._src.ops.experimental.fused_moe import host as fused_host

align_up = fused_host.align_up
token_gather_window_rows = fused_host.token_gather_window_rows
vmem_limit = fused_host.vmem_limit

LANES = 128
SUBLANES = 8
# VMEM rows in front of and behind the landed lists, so the aligned 16-row
# source window of any tile stays inside the scratch (see _copy_run_vector).
_FRONT = SUBLANES
_BACK = 2 * SUBLANES
IMPLS = ("vector", "smem")
# Run-table rows (one column per run (s, local expert)).
RUN_SRC = 0  # word of the run's first entry in tg_land (plane 0)
RUN_N = 1  # entries
RUN_DST = 2  # slab row of the run's first entry on this shard
RUN_SRC1 = 3  # word of the run's first entry in plane 1 (fp8 scales), else 0


def gather_rows(ragged_stride, tile_m):
  """Returns the rows of the [rows, 128] token-gather image.

  Args:
    ragged_stride: Slab rows per shard.
    tile_m: Tile height in rows.

  Returns:
    Rows covering the slab and the kernel's aligned-window overhang, padded to
    whole (8, 128) int32 tiles.
  """
  n = ragged_stride + token_gather_window_rows(tile_m)
  return align_up(align_up(n, LANES) // LANES, SUBLANES)


def assemble_runs(
    rows_by_dest, me, *, ep, g_local, t_local, topk, nw, slab_base, run_start
):
  """Builds the run table and wait counts tg_assemble works from.

  Args:
    rows_by_dest: [ep, E] i32 exchanged per-(source, expert) pair counts.
    me: This shard's index.
    ep: Expert-parallel width.
    g_local: Experts per shard.
    t_local: Tokens per shard.
    topk: Experts per token.
    nw: Planes per list entry.
    slab_base: [g] i32 each local expert's first row in this shard's slab.
    run_start: [g, ep] i32 where source s's run starts inside each expert.

  Returns:
    (runs [4, ep * g] i32 with run r = s * g + local expert, meta [1 + ep] i32:
    rows to wait for on tg_sem, then rows landed per source).
  """
  me = jnp.asarray(me, jnp.int32)
  n = lax.dynamic_slice(
      rows_by_dest.astype(jnp.int32), (0, me * g_local), (ep, g_local)
  )  # [s, g]
  n_s = n.sum(axis=1)
  off = jnp.cumsum(n, axis=1) - n
  words = source_plan.list_row_words(t_local, topk, nw)
  src0 = (
      jnp.arange(ep, dtype=jnp.int32)[:, None] * words
      + source_plan.HDR_WORDS
      + off
  )
  if nw == 2:
    src1 = src0 + ((n_s + (LANES - 1)) // LANES * LANES)[:, None]
  else:
    src1 = jnp.zeros_like(src0)
  dst = slab_base.astype(jnp.int32)[None, :] + run_start.astype(jnp.int32).T
  runs = jnp.stack([src0, n, dst, src1]).reshape(4, ep * g_local)
  landed = source_plan.list_dma_rows(n_s, nw)
  meta = jnp.concatenate([landed.sum()[None], landed]).astype(jnp.int32)
  return runs, meta


def _sublanes_from(win, k):
  """win[k : k + 8] of a [16, 128] window for a dynamic k in [0, 8]."""
  out = win[0:SUBLANES]
  for j in range(1, SUBLANES + 1):
    out = jnp.where(k == j, win[j : j + SUBLANES], out)
  return out


def _copy_run_vector(land_vm, img_vm, src, n, dst):
  """Copies a run of flat words: img[dst + i] = land[src + i] for i < n.

  Both refs are [rows, 128] words; src is already in land_vm's padded
  coordinates.

  Destination word W = r * 128 + l reads source word W + delta. With
  delta = q * 128 + rem (0 <= rem < 128), destination row r takes lanes
  [rem, 128) of source row r + q followed by lanes [0, rem) of row r + q + 1:
  a lane rotate of both by 128 - rem and a select at lane 128 - rem. Eight
  destination rows go at a time, aligned; their nine source rows come out
  of one aligned 16-row load by a sublane select.

  Args:
    land_vm: [rows, 128] landed lists (padded).
    img_vm: [rows, 128] token-gather image.
    src: First source word.
    n: Words to copy.
    dst: First destination word.
  """
  delta = src - dst
  q = jnp.right_shift(delta, 7)  # floor division (arithmetic shift)
  rem = jnp.bitwise_and(delta, LANES - 1)
  t_first = jnp.right_shift(dst, 10)  # (dst // 128) // 8
  t_last = jnp.right_shift(dst + n - 1, 10)
  sub = lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)
  lane = lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 1)

  def tile(ti, carry):
    r0 = pl.multiple_of(ti * SUBLANES, SUBLANES)
    q0 = r0 + q  # source row of the tile's first destination row
    qa = pl.multiple_of(jnp.bitwise_and(q0, -SUBLANES), SUBLANES)
    sh = q0 - qa
    win = land_vm[pl.ds(qa, 2 * SUBLANES), :]
    a = _sublanes_from(win, sh)
    b = _sublanes_from(win, sh + 1)
    shift = LANES - rem
    vals = jnp.where(
        lane < LANES - rem, pltpu.roll(a, shift, 1), pltpu.roll(b, shift, 1)
    )
    w = (r0 + sub) * LANES + lane
    live = (w >= dst) & (w < dst + n)
    cur = img_vm[pl.ds(r0, SUBLANES), :]
    img_vm[pl.ds(r0, SUBLANES), :] = jnp.where(live, vals, cur)
    return carry

  lax.fori_loop(t_first, t_last + 1, tile, 0)


def _for_tiles(rows, body):
  """body(r0) for every aligned 8-row tile of a [rows, 128] ref."""

  def step(i, carry):
    body(pl.multiple_of(i * SUBLANES, SUBLANES))
    return carry

  lax.fori_loop(0, rows // SUBLANES, step, 0)


def _wait_lists(tg_sem, land_hbm, rows):
  """Block until `rows` [*, 128] i32 rows of lists have landed on tg_sem."""

  @pl.when(rows > 0)
  def _():
    pltpu.make_async_copy(
        land_hbm.at[pl.ds(0, rows)], land_hbm.at[pl.ds(0, rows)], tg_sem
    ).wait()


def _vector_kernel(
    runs_sm,
    meta_sm,
    land_hbm,
    *refs,
    ep,
    nw,
    n_runs,
    n_tokens,
    src_rows,
    tg_rows,
    scale_rows,
    wait,
):
  """Vector kernel body: assemble the image with lane rotates in VMEM."""
  tg_sem = None
  if wait:
    tg_sem, *refs = refs
  if nw == 2:
    tg_hbm, scale_hbm, land_vm, tg_vm, scale_vm, sems = refs
  else:
    tg_hbm, land_vm, tg_vm, sems = refs
    scale_hbm = scale_vm = None
  if wait:
    _wait_lists(tg_sem, land_hbm, meta_sm[0])
  # Only the rows that landed: ep headers plus the live bodies.
  copies = []
  for s in range(ep):
    copies.append(
        pltpu.make_async_copy(
            land_hbm.at[pl.ds(s * src_rows, meta_sm[1 + s])],
            land_vm.at[pl.ds(_FRONT + s * src_rows, meta_sm[1 + s])],
            sems.at[0],
        )
    )
  for c in copies:
    c.start()
  zero = jnp.zeros((SUBLANES, LANES), jnp.int32)

  def clear(img):
    def body(r0):
      img[pl.ds(r0, SUBLANES), :] = zero

    _for_tiles(img.shape[0], body)

  clear(tg_vm)
  if scale_vm is not None:
    clear(scale_vm)
  for c in copies:
    c.wait()

  def run(r, carry):
    n = runs_sm[RUN_N, r]

    @pl.when(n > 0)
    def _():
      src = runs_sm[RUN_SRC, r] + _FRONT * LANES
      dst = runs_sm[RUN_DST, r]
      _copy_run_vector(land_vm, tg_vm, src, n, dst)
      if scale_vm is not None:
        _copy_run_vector(
            land_vm, scale_vm, runs_sm[RUN_SRC1, r] + _FRONT * LANES, n, dst
        )

    return carry

  lax.fori_loop(0, n_runs, run, 0)

  def clip(r0):
    tg_vm[pl.ds(r0, SUBLANES), :] = jnp.clip(
        tg_vm[pl.ds(r0, SUBLANES), :], 0, n_tokens - 1
    )

  _for_tiles(tg_rows, clip)
  out = [pltpu.make_async_copy(tg_vm, tg_hbm, sems.at[1])]
  if scale_vm is not None:
    out.append(
        pltpu.make_async_copy(
            scale_vm.at[pl.ds(0, scale_rows)], scale_hbm, sems.at[1]
        )
    )
  for c in out:
    c.start()
  for c in out:
    c.wait()


def _smem_kernel(
    runs_sm,
    meta_sm,
    land_hbm,
    *refs,
    ep,
    nw,
    n_runs,
    n_tokens,
    src_rows,
    tg_rows,
    scale_rows,
    wait,
):
  """SMEM kernel body: assemble the image with a scalar copy loop."""
  tg_sem = None
  if wait:
    tg_sem, *refs = refs
  if nw == 2:
    tg_hbm, scale_hbm, land_sm, tg_sm, scale_sm, sems = refs
  else:
    tg_hbm, land_sm, tg_sm, sems = refs
    scale_hbm = scale_sm = None
  if wait:
    _wait_lists(tg_sem, land_hbm.reshape(-1, LANES), meta_sm[0])
  words = src_rows * LANES
  copies = [
      pltpu.make_async_copy(
          land_hbm.at[pl.ds(s * words, meta_sm[1 + s] * LANES)],
          land_sm.at[pl.ds(s * words, meta_sm[1 + s] * LANES)],
          sems.at[0],
      )
      for s in range(ep)
  ]
  for c in copies:
    c.start()

  def clear(img, length):
    def body(i, carry):
      img[i] = jnp.int32(0)
      return carry

    lax.fori_loop(0, length, body, 0)

  clear(tg_sm, tg_rows * LANES)
  if scale_sm is not None:
    clear(scale_sm, scale_rows * LANES)
  for c in copies:
    c.wait()

  def run(r, carry):
    src = runs_sm[RUN_SRC, r]
    src1 = runs_sm[RUN_SRC1, r]
    dst = runs_sm[RUN_DST, r]

    def entry(i, c):
      tg_sm[dst + i] = jnp.clip(land_sm[src + i], 0, n_tokens - 1)
      if scale_sm is not None:
        scale_sm[dst + i] = land_sm[src1 + i]
      return c

    lax.fori_loop(0, runs_sm[RUN_N, r], entry, 0)
    return carry

  lax.fori_loop(0, n_runs, run, 0)
  out = [pltpu.make_async_copy(tg_sm, tg_hbm, sems.at[1])]
  if scale_sm is not None:
    out.append(pltpu.make_async_copy(scale_sm, scale_hbm, sems.at[1]))
  for c in out:
    c.start()
  for c in out:
    c.wait()


def tg_assemble(
    tg_land,
    runs,
    meta,
    tg_sem=None,
    *,
    ep,
    nw,
    n_tokens,
    tg_rows,
    scale_rows=None,
    impl="vector",
):
  """Assembles the token-gather table (and the fp8 scale slab) on this shard.

  Args:
    tg_land: [ep * list_rows, 128] i32 landed lists.
    runs: Run table from assemble_runs.
    meta: Wait counts from assemble_runs.
    tg_sem: The start kernel's list semaphore, or None when the lists are
      already complete (CPU exchange emulation).
    ep: Expert-parallel width.
    nw: Planes per list entry.
    n_tokens: Tokens in the slab.
    tg_rows: Rows of the token-gather image.
    scale_rows: Rows of the scale slab, or None.
    impl: Kernel implementation to use.

  Returns:
    (token_gather [tg_rows * 128] i32, scale_slab [scale_rows, 128] i32 bit
    patterns when nw == 2, else None).

  Raises:
    ValueError: If `impl` or `nw` is invalid, or `scale_rows` is given exactly
      when nw != 2.
  """
  if impl not in IMPLS:
    raise ValueError(f"impl={impl!r} is not one of {IMPLS}")
  if nw not in (1, 2):
    raise ValueError(f"nw={nw}: one or two words per list entry")
  if (nw == 2) != (scale_rows is not None):
    raise ValueError("scale_rows is required exactly when nw == 2")
  land_rows = tg_land.shape[0]
  src_rows = land_rows // ep
  n_runs = runs.shape[1]
  wait = tg_sem is not None
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  anyspec = pl.BlockSpec(memory_space=pl.ANY)
  in_specs = [smem, smem, anyspec]
  args = [runs.astype(jnp.int32), meta.astype(jnp.int32)]
  if impl == "vector":
    args.append(tg_land)
    out_shape = [jax.ShapeDtypeStruct((tg_rows, LANES), jnp.int32)]
    scratch = [pltpu.VMEM((_FRONT + land_rows + _BACK, LANES), jnp.int32)]
    scratch.append(pltpu.VMEM((tg_rows, LANES), jnp.int32))
    if nw == 2:
      assert scale_rows is not None
      out_shape.append(jax.ShapeDtypeStruct((scale_rows, LANES), jnp.int32))
      scratch.append(
          pltpu.VMEM((align_up(scale_rows, SUBLANES), LANES), jnp.int32)
      )
    body = _vector_kernel
  else:
    args.append(tg_land.reshape(-1))
    out_shape = [jax.ShapeDtypeStruct((tg_rows * LANES,), jnp.int32)]
    scratch = [pltpu.SMEM((land_rows * LANES,), jnp.int32)]
    scratch.append(pltpu.SMEM((tg_rows * LANES,), jnp.int32))
    if nw == 2:
      assert scale_rows is not None
      out_shape.append(jax.ShapeDtypeStruct((scale_rows * LANES,), jnp.int32))
      scratch.append(pltpu.SMEM((scale_rows * LANES,), jnp.int32))
    body = _smem_kernel
  scratch.append(pltpu.SemaphoreType.DMA((2,)))
  if wait:
    in_specs.append(pl.BlockSpec(memory_space=pltpu.SEMAPHORE))
    args.append(tg_sem)
  outs = pl.pallas_call(
      functools.partial(
          body,
          ep=ep,
          nw=nw,
          n_runs=n_runs,
          n_tokens=n_tokens,
          src_rows=src_rows,
          tg_rows=tg_rows,
          scale_rows=scale_rows,
          wait=wait,
      ),
      out_shape=tuple(out_shape),
      in_specs=in_specs,
      out_specs=tuple(anyspec for _ in out_shape),
      scratch_shapes=scratch,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit()),
      name=f"tg_assemble_{impl}_ep{ep}_nw{nw}",
  )(*args)
  token_gather = outs[0].reshape(-1)
  scale = outs[1].reshape(scale_rows, LANES) if nw == 2 else None
  return token_gather, scale
