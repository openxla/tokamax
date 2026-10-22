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
"""Pallas/Mosaic TPU kernel for the fused FP8 activation gradient (dLHS).

Computes `dlhs = dout @ rhs^T` for the fused FP8 matmul, where `rhs` is the
`(k, n)` FP8 weight with one float32 scale per column, i.e. the residuals the
forward pass saved. The contraction runs over `n`, so the column scale cannot
be factored out of it: the kernel folds it into the cotangent, quantizes the
scaled cotangent to FP8 with one scale per row, runs the contraction on the FP8
MXU and applies the row scale to the result. The quantized cotangent never
leaves VMEM.

With `bk == k` and `bn == n` on a TPU with MXU accumulators (TPU7x), a
pipelined kernel hides the quantization under the contraction; see
`_pipelined_dlhs_kernel`. It is bit-identical to the blocked kernel.

The numerics are those of `base.fused_fp8_dlhs_reference`.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_fp8_matmul import base
from tokamax._src.ops.experimental.fused_fp8_matmul import pallas_mosaic_tpu_kernel as fwd_kernel

_CORE_AXIS_NAME = "core"


def _quantize_g(g_vmem: jax.Ref, sw_vmem: jax.Ref):
  """Folds the column scales into the cotangent tile and quantizes it.

  The multiply stays in the cotangent's own dtype. Promoting the `(bm, bn)`
  tile to float32 here, as an earlier version did, made a 16 MB temporary at
  `bm = 1024, bn = 4096` and most large tilings failed to compile.

  Args:
    g_vmem: The `(bm, bn)` cotangent block.
    sw_vmem: The `(bn,)` column scales.

  Returns:
    `(gq, sg)`: the FP8 tile and its `(bm, 1)` float32 row scales.
  """
  g = g_vmem[...]
  return base.quantize_rows_fp8(g * sw_vmem[...].astype(g.dtype))


def _dlhs_kernel(
    g_hbm: jax.Ref,
    w_hbm: jax.Ref,
    sw_hbm: jax.Ref,
    o_hbm: jax.Ref,
    accum_vmem: jax.Ref,
    gq_cache: jax.Ref,
    sg_cache: jax.Ref,
    *,
    bm: int,
    bk: int,
    bn: int,
):
  """Pipelined dLHS kernel body. See `fused_fp8_dlhs_pallas`.

  Args:
    g_hbm: `(m, n)` cotangent.
    w_hbm: `(k, n)` FP8 weights.
    sw_hbm: `(n,)` float32 weight scales.
    o_hbm: `(m, k)` activation gradient.
    accum_vmem: `(bm, bk)` float32 accumulator, read only when `bn < n`.
    gq_cache: `(n / bn, bm, bn)` FP8 cache of the quantized cotangent, read only
      when `bk < k`.
    sg_cache: `(n / bn, bm, 1)` float32 cache of the cotangent row scales, read
      only when `bk < k`.
    bm: Block size along `m`.
    bk: Block size along `k`.
    bn: Block size along `n`.
  """
  m, n = g_hbm.shape
  k, _ = w_hbm.shape
  grid = (pl.cdiv(m, bm), pl.cdiv(k, bk), pl.cdiv(n, bn))
  single_k = grid[1] == 1
  single_n = grid[2] == 1

  g_spec = pl.BlockSpec((bm, bn), lambda mi, ki, ni: (mi, ni))
  w_spec = pl.BlockSpec((bk, bn), lambda mi, ki, ni: (ki, ni))
  # A 1-D VMEM block for the column scales; see the forward kernel's
  # `SX_LANES` for why a `(1, bn)` block would be a bad idea.
  sw_spec = pl.BlockSpec((bn,), lambda mi, ki, ni: (ni,))
  o_spec = pl.BlockSpec((bm, bk), lambda mi, ki, ni: (mi, ki))

  def _load_gq(g_vmem: jax.Ref, sw_vmem: jax.Ref):
    """Quantizes each `(bm, bn)` cotangent tile once and reuses it across k.

    `g_spec` does not depend on the k-block index `ki`, so a naive body would
    re-quantize the identical tile `cdiv(k, bk)` times; that VPU work, not the
    FP8 dot, dominated the first version of this kernel. The cache is keyed by
    the n-block index `ni`: with `ni` innermost, the `ki == 0` sweep fills every
    entry before any `ki > 0` step reads it. With a single k block there is
    nothing to reuse and the cache is bypassed.

    Args:
      g_vmem: The `(bm, bn)` cotangent block.
      sw_vmem: The `(bn,)` column scales.

    Returns:
      `(gq, sg)`: the FP8 tile and its `(bm, 1)` float32 row scales.
    """
    if single_k:
      return _quantize_g(g_vmem, sw_vmem)

    kind = pl.program_id(1)
    nind = pl.program_id(2)

    @pl.when(kind == 0)
    def _fill():
      gq_cache[nind], sg_cache[nind] = _quantize_g(g_vmem, sw_vmem)

    return gq_cache[nind], sg_cache[nind]

  def _partial(g_vmem: jax.Ref, w_vmem: jax.Ref, sw_vmem: jax.Ref):
    gq, sg = _load_gq(g_vmem, sw_vmem)
    gw = jax.lax.dot_general(
        gq,
        w_vmem[...],
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return gw * sg  # `(bm, 1)` broadcasts along lanes for free.

  def body_single_n(g_vmem, w_vmem, sw_vmem, o_vmem):
    """Direct pass when `bn == n`: no accumulator needed."""
    o_vmem[...] = _partial(g_vmem, w_vmem, sw_vmem).astype(o_vmem.dtype)

  def body_multi_n(g_vmem, w_vmem, sw_vmem, o_vmem):
    """Pipelined reduction over n blocks when `bn < n`."""
    nind = pl.program_id(2)

    @pl.when(nind == 0)
    def _init():
      accum_vmem[...] = jnp.zeros_like(accum_vmem)

    # Each n block is quantized against its own row absmax, so the partial
    # product is already rescaled by `_partial` before it is accumulated.
    accum_vmem[...] += _partial(g_vmem, w_vmem, sw_vmem)

    @pl.when(nind == pl.num_programs(2) - 1)
    def _write():
      o_vmem[...] = accum_vmem[...].astype(o_vmem.dtype)

  pltpu.emit_pipeline(
      body_single_n if single_n else body_multi_n,
      grid=grid,
      in_specs=[g_spec, w_spec, sw_spec],
      out_specs=o_spec,
      core_axis_name=_CORE_AXIS_NAME,
      dimension_semantics=(
          pltpu.PARALLEL,
          pltpu.PARALLEL,
          pltpu.PARALLEL if single_n else pltpu.ARBITRARY,
      ),
  )(g_hbm, w_hbm, sw_hbm, o_hbm)


def _pipelined_geometry(m: int, k: int, n: int, bm: int, bk: int, bn: int):
  """Returns the pipelined kernel's geometry, or None if it does not apply.

  The pipelined kernel drives the MXUs with explicit pushes into accumulators,
  so it needs a chip that has them, the whole of `k` and `n` in one block, and
  a cotangent block whose rows fit the accumulators.
  """
  if jax.default_backend() != "tpu" or bk != k or bn != n:
    return None
  info = pltpu.get_tpu_info()
  if not info.num_accumulators:
    return None
  col = info.mxu_column_size
  # One fp8 vmatmul takes sublanes x (32-bit packing) rows of LHS.
  rows = info.num_sublanes * (4 // jnp.dtype(base.FP8_DTYPE).itemsize)
  # An (r, col) float32 accumulator takes r / sublanes accumulator entries.
  acc_rows = info.num_accumulators * info.num_sublanes
  if (
      m % bm
      or bm % rows
      or bm > acc_rows
      or n % col
      or k % (info.num_mxus * col)
  ):
    return None
  return dict(
      col=col,
      lanes=info.num_lanes,
      rows=rows,
      num_mxus=info.num_mxus,
      num_sets=max(1, min(2, acc_rows // bm)),
  )


def _pipelined_dlhs_kernel(
    g_hbm: jax.Ref,
    w_hbm: jax.Ref,
    sw_hbm: jax.Ref,
    o_hbm: jax.Ref,
    gq0: jax.Ref,
    gq1: jax.Ref,
    sg0: jax.Ref,
    sg1: jax.Ref,
    mx_vmem: jax.Ref,
    inv_vmem: jax.Ref,
    *,
    bm: int,
    col: int,
    lanes: int,
    rows: int,
    num_mxus: int,
    num_sets: int,
    eps: float = 1e-7,
):
  """dLHS with the cotangent quantization pipelined under the contraction.

  Grid step `j` quantizes cotangent block `j` into one VMEM slot while the MXUs
  contract block `j - 1` from the other, so the quantization is no longer
  serial with the matmul (it was 4-10% of the blocked kernel's time at
  production shapes).

  Overlapping the two takes three things:

  * The contraction is issued as explicit MXU work, one `rows`-row
    `matmul_acc_lhs` per `(rows, col)` accumulator, so VPU work can sit between
    individual vmatmuls. Output columns go to the MXUs in groups of
    `num_mxus * col`; the next weight tile is staged after the first row group.
  * The quantization is cut into `rows x col` slivers that need no cross-lane
    work except one reduction per row group: `A` keeps a running per-lane
    `max |g * sw|`, `F` reduces it to the row scale, `B` scales and casts. The
    slivers are spread evenly over the vmatmuls.
  * Each sliver feeds the LHS of the next vmatmul through a select that never
    fires (`isnan` of the column scales). Without that dependence the LLO
    scheduler places all of it after the last vmatmul, since nothing in this
    grid step consumes it.

  When the accumulators hold two sets, consecutive output-column groups
  alternate between them and the previous group is popped a few steps late:
  popping an accumulator right after its last vmatmul waits on the MXU.

  Args:
    g_hbm: `(m, n)` cotangent.
    w_hbm: `(k, n)` FP8 weights.
    sw_hbm: `(n,)` float32 weight scales.
    o_hbm: `(m, k)` activation gradient.
    gq0: `(bm, n)` FP8 quantized cotangent, slot 0.
    gq1: `(bm, n)` FP8 quantized cotangent, slot 1.
    sg0: `(bm, lanes)` float32 row scales of slot 0, replicated across lanes.
    sg1: `(bm, lanes)` float32 row scales of slot 1.
    mx_vmem: `(bm, lanes)` running per-lane absmax of the block being quantized.
    inv_vmem: `(bm, lanes)` inverse row scales of that block.
    bm: Block size along `m`.
    col: MXU column size.
    lanes: Vector lanes.
    rows: LHS rows per FP8 vmatmul.
    num_mxus: MXUs per core.
    num_sets: Accumulator sets.
    eps: Lower bound on the row absmax, as in `base.quantize_rows_fp8`.
  """
  m, n = g_hbm.shape
  k, _ = w_hbm.shape
  num_m = m // bm
  num_t = n // col  # contraction tiles
  num_c = k // (num_mxus * col)  # output column groups
  num_r = bm // rows  # row groups
  steps = [(c, t) for c in range(num_c) for t in range(num_t)]
  num_slots = len(steps) * num_r
  # Pop output column group c - 1 at contraction tile `pop_t` of group c, so
  # that its last vmatmul is ~32 slots old; with one set it must go first.
  pop_t = 0 if num_sets == 1 else min(num_t - 1, pl.cdiv(32, num_r) - 1)
  # Quantization slivers in dependency order, spread evenly over the slots.
  slivers = (
      [("A", t, r) for t in range(num_t) for r in range(num_r)]
      + [("F", None, r) for r in range(num_r)]
      + [("B", t, r) for t in range(num_t) for r in range(num_r)]
  )
  slivers_at = {}
  for i, s in enumerate(slivers):
    slivers_at.setdefault(i * num_slots // len(slivers), []).append(s)

  accs = [
      [
          [
              jax.empty_ref(
                  jax.ShapeDtypeStruct((rows, col), jnp.float32),
                  memory_space=pltpu.ACC(mxu),
              )
              for _ in range(num_r)
          ]
          for mxu in range(num_mxus)
      ]
      for _ in range(num_sets)
  ]

  def push(w_vmem, c, t):
    for mxu in range(num_mxus):
      a = num_mxus * c + mxu
      pltpu.matmul_push_rhs(
          w_vmem[a * col : (a + 1) * col, t * col : (t + 1) * col],
          staging_register=0,
          mxu_index=mxu,
          transpose=True,
      )

  def as_fp8_tile(x):
    """Widens a `(rows, lanes)` value to an FP8 `(rows, col)` tile."""
    x = jnp.concatenate([x] * (col // x.shape[-1]), axis=-1)
    return x.astype(base.FP8_DTYPE)

  def quantize(sliver, g_vmem, sw_vmem, gq, sg):
    """Runs one sliver; returns an FP8 `(rows, col)` value derived from it."""
    kind, t, r = sliver
    rs = slice(r * rows, (r + 1) * rows)
    if kind == "F":
      max_abs = jnp.maximum(jnp.max(mx_vmem[rs], axis=-1, keepdims=True), eps)
      scale = (max_abs / base.FP8_MAX).astype(jnp.float32)
      inv = (base.FP8_MAX / max_abs).astype(inv_vmem.dtype)
      sg[rs] = jnp.broadcast_to(scale, (rows, lanes))
      inv_vmem[rs] = inv = jnp.broadcast_to(inv, (rows, lanes))
      return as_fp8_tile(inv)
    out = []
    for h in range(col // lanes):
      cs = slice(t * col + h * lanes, t * col + (h + 1) * lanes)
      g = g_vmem[rs, cs]
      g = g * sw_vmem[cs].astype(g.dtype)  # As in `_quantize_g`.
      if kind == "A":
        out.append(jnp.abs(g))
      else:
        out.append((g * inv_vmem[rs]).astype(base.FP8_DTYPE))
        gq[rs, cs] = out[-1]
    if kind == "B":
      return jnp.concatenate(out, axis=-1)
    part = functools.reduce(jnp.maximum, out)
    if t:
      part = jnp.maximum(mx_vmem[rs], part)
    mx_vmem[rs] = part
    return as_fp8_tile(part)

  def contract(w_vmem, o_vmem, gq, sg, quantize_next, never):
    def pop(c, r):
      rs = slice(r * rows, (r + 1) * rows)
      for mxu in range(num_mxus):
        res = pltpu.matmul_pop(accs[c % num_sets][mxu][r])
        a = num_mxus * c + mxu
        for h in range(col // lanes):
          cs = slice(a * col + h * lanes, a * col + (h + 1) * lanes)
          o_vmem[rs, cs] = (res[:, h * lanes : (h + 1) * lanes] * sg[rs]).astype(
              o_vmem.dtype
          )

    push(w_vmem, *steps[0])
    slot = 0
    for i, (c, t) in enumerate(steps):
      for r in range(num_r):
        if t == pop_t and c > 0:
          pop(c - 1, r)
        lhs = gq[r * rows : (r + 1) * rows, t * col : (t + 1) * col]
        if quantize_next is not None:
          for sliver in slivers_at.get(slot, ()):
            # Never taken; only makes this vmatmul wait for the sliver.
            dep = pltpu.bitcast(quantize_next(sliver), jnp.int32)
            lhs = pltpu.bitcast(lhs, jnp.int32)
            lhs = pltpu.bitcast(jnp.where(never, dep, lhs), base.FP8_DTYPE)
        for mxu in range(num_mxus):
          pltpu.matmul_acc_lhs(
              accs[c % num_sets][mxu][r],
              lhs,
              load_staged_rhs=0 if r == 0 else None,
          )
        if r == 0 and i + 1 < len(steps):
          push(w_vmem, *steps[i + 1])
        slot += 1
    for r in range(num_r):
      pop(num_c - 1, r)

  slots = ((gq0, sg0), (gq1, sg1))

  def body(g_vmem, w_vmem, sw_vmem, o_vmem):
    j = pl.program_id(0)
    # False for finite column scales. A NaN scale makes the whole result NaN,
    # whichever LHS the select picks. Shaped like the int32 view of a tile.
    s = sw_vmem[:lanes].astype(jnp.float32)
    s = jnp.broadcast_to(s[None, :], (rows // 4, lanes))
    never = jnp.isnan(jnp.concatenate([s] * (col // lanes), axis=-1))

    def quantize_into(q):
      gq, sg = slots[q]
      return lambda sliver: quantize(sliver, g_vmem, sw_vmem, gq, sg)

    @pl.when(j == 0)
    def _first():
      for sliver in slivers:
        quantize_into(0)(sliver)

    # The slots are separate buffers selected statically, one body per parity.
    for q in range(2):

      @pl.when((j > 0) & (j < num_m) & (j % 2 == q))
      def _steady():
        gq, sg = slots[1 - q]
        contract(w_vmem, o_vmem, gq, sg, quantize_into(q), never)

    @pl.when(j == num_m)
    def _last():
      gq, sg = slots[(num_m - 1) % 2]
      contract(w_vmem, o_vmem, gq, sg, None, never)

  pltpu.emit_pipeline(
      body,
      grid=(num_m + 1,),
      in_specs=[
          # Step j reads block j; the last step re-uses the last block.
          pl.BlockSpec((bm, n), lambda j: (jnp.minimum(j, num_m - 1), 0)),
          pl.BlockSpec((k, n), lambda j: (0, 0)),
          pl.BlockSpec((n,), lambda j: (0,)),
      ],
      # Step j writes block j - 1. An output block is copied out only when its
      # index changes, so step 0 writes nothing.
      out_specs=pl.BlockSpec((bm, k), lambda j: (jnp.maximum(j - 1, 0), 0)),
      core_axis_name=_CORE_AXIS_NAME,
      dimension_semantics=(pltpu.ARBITRARY,),
  )(g_hbm, w_hbm, sw_hbm, o_hbm)


def fused_fp8_dlhs_pallas(
    dout: jax.Array,
    rhs_q: jax.Array,
    rhs_scale: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    out_dtype: jnp.dtype | None = None,
    pipelined: bool = True,
) -> jax.Array:
  """Fused FP8 activation gradient: `quantize_rows(dout * rhs_scale) @ rhs_q^T`.

  Args:
    dout: Cotangent of shape `(m, n)`, `bfloat16` or `float32`.
    rhs_q: Weights of shape `(k, n)` already quantized to `float8_e4m3fn`.
    rhs_scale: Per-column weight scales of shape `(n,)`, `float32`.
    bm: Block size along `m`. Must divide `m`.
    bk: Block size along `k`. Must divide `k`.
    bn: Block size along `n`, the contracting dimension. Must divide `n`. With
      `bn == n` each cotangent row is quantized against a single scale and no
      accumulator is needed; with `bn < n` each n block gets its own scale.
    out_dtype: Output dtype. Defaults to `dout.dtype`.
    pipelined: Use the pipelined kernel when it applies (`bk == k`, `bn == n`,
      a TPU with MXU accumulators, and a tiling that fits). Its result is
      bit-identical to the blocked kernel's.

  Returns:
    `dlhs` of shape `(m, k)`.
  """
  m, n = dout.shape
  k, n_w = rhs_q.shape
  if n_w != n:
    raise ValueError(f"Contracting dims differ: dout has n={n}, rhs has {n_w}.")
  if rhs_scale.shape != (n,):
    raise ValueError(
        f"rhs_scale must have shape ({n},), got {rhs_scale.shape}."
    )
  if rhs_q.dtype != base.FP8_DTYPE:
    raise ValueError(f"rhs_q must be {base.FP8_DTYPE}, got {rhs_q.dtype}.")
  if m % bm:
    raise ValueError(f"m ({m}) must be divisible by bm ({bm}).")
  if k % bk:
    raise ValueError(f"k ({k}) must be divisible by bk ({bk}).")
  if n % bn:
    raise ValueError(f"n ({n}) must be divisible by bn ({bn}).")

  out_dtype = out_dtype or dout.dtype
  geometry = _pipelined_geometry(m, k, n, bm, bk, bn) if pipelined else None
  if geometry is not None:
    lanes = geometry["lanes"]
    scratch_types = [
        pltpu.VMEM((bm, n), base.FP8_DTYPE),
        pltpu.VMEM((bm, n), base.FP8_DTYPE),
        pltpu.VMEM((bm, lanes), jnp.float32),
        pltpu.VMEM((bm, lanes), jnp.float32),
        pltpu.VMEM((bm, lanes), dout.dtype),
        pltpu.VMEM((bm, lanes), dout.dtype),
    ]
    # Double-buffered cotangent and output blocks, the resident weight, and
    # the two quantized-cotangent slots; the rest is small.
    vmem_bytes = (
        2 * bm * n * dout.dtype.itemsize
        + k * n * rhs_q.dtype.itemsize
        + 2 * bm * k * jnp.dtype(out_dtype).itemsize
        + 2 * bm * n * jnp.dtype(base.FP8_DTYPE).itemsize
    )
    if vmem_bytes <= 0.9 * fwd_kernel.VMEM_LIMIT_BYTES:
      kernel = functools.partial(_pipelined_dlhs_kernel, bm=bm, **geometry)
      return pl.kernel(
          kernel,
          out_type=jax.ShapeDtypeStruct((m, k), out_dtype),
          mesh=pltpu.create_tensorcore_mesh(axis_name=_CORE_AXIS_NAME),
          scratch_types=scratch_types,
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=fwd_kernel.VMEM_LIMIT_BYTES
          ),
      )(dout, rhs_q, rhs_scale)

  grid_k = pl.cdiv(k, bk)
  grid_n = pl.cdiv(n, bn)

  # Scratch order matches the kernel signature: accum, gq_cache, sg_cache.
  # Buffers the chosen tiling leaves dead are allocated at a minimal shape
  # rather than dropped, so the positional layout stays fixed. At the tiling
  # the heuristics prefer (`bk == k`, `bn == n`) both caches are dead, and
  # holding them at full size (8.4 MB at `bm = 512` on (32768, 3072, 4096))
  # is what used to keep `bm = 1024` from compiling.
  dead = (8, 128)  # One (sublane, lane) tile, the smallest useful allocation.
  scratch_types = [
      pltpu.VMEM((bm, bk) if grid_n > 1 else dead, jnp.float32),
      pltpu.VMEM(
          (grid_n, bm, bn) if grid_k > 1 else (1,) + dead, base.FP8_DTYPE
      ),
      pltpu.VMEM((grid_n, bm, 1) if grid_k > 1 else (1,) + dead, jnp.float32),
  ]

  kernel = functools.partial(_dlhs_kernel, bm=bm, bk=bk, bn=bn)
  return pl.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((m, k), out_dtype),
      mesh=pltpu.create_tensorcore_mesh(axis_name=_CORE_AXIS_NAME),
      scratch_types=scratch_types,
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=fwd_kernel.VMEM_LIMIT_BYTES
      ),
  )(dout, rhs_q, rhs_scale)
