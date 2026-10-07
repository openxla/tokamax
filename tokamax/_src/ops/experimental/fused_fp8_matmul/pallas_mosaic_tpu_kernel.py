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
"""Pallas/Mosaic TPU kernel for the fused FP8 matmul (forward pass).

Computes `out = lhs @ rhs` where `lhs` is a `bfloat16`/`float32` activation and
`rhs` is a weight already quantized to `float8_e4m3fn` with one `float32` scale
per output column. The activation is quantized to FP8 inside the kernel, one
scale per row, so the contraction runs on the FP8 MXU path and the activation
is never materialized in FP8 in HBM (unless the caller asks for it as a
residual for the backward pass).

Only the forward kernel lives here. The backward kernels are added separately.
"""

import functools
from typing import Literal, overload

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_CORE_AXIS_NAME = "core"

FP8_DTYPE = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8_DTYPE).max)  # 448.0

# Scoped VMEM this kernel asks Mosaic for, set per kernel through
# `pltpu.CompilerParams`. The `--xla_tpu_scoped_vmem_limit_kib` flag governs
# XLA's own fusions and does not reach a Mosaic custom call, which otherwise
# gets a 32 MiB default. 64 MiB is the physical VMEM per core on TPU7x.
VMEM_LIMIT_BYTES = 64 * 1024 * 1024

# Budget used when *choosing* tiles, deliberately below the hard limit: the
# estimator below counts the buffers that dominate, not alignment padding or
# small scratch.
VMEM_BUDGET_BYTES = 56 * 1024 * 1024

# Lane width the per-row activation scale is padded to before being stored as
# a residual output.
#
# The scale is logically one float per row, but a Pallas `(bm, 1)` f32 output
# block is pathological: f32 tiles VMEM as (8, 128), so a single-lane block
# touches four bytes per 512-byte VMEM row and the store turns into a scatter.
# Emitting the residual that way cost ~235 us on (m, k, n) = (32768, 3072, 4096)
# against a ~54 us bandwidth estimate. Writing a full lane row instead costs
# one extra `(m, 128)` f32 HBM write (~9 us at that shape) and a slice on the
# way out.
SX_LANES = 128


def pick_tile(dim: int, candidates: tuple[int, ...]) -> int:
  """Returns the first candidate that evenly divides `dim`, else a fallback.

  Args:
    dim: The array dimension to tile.
    candidates: Candidate tile sizes, listed largest-preferred.

  Returns:
    The first candidate that divides `dim`, else the largest power of two
    divisor of `dim` (floor 128, the lane width), so odd shapes keep working
    rather than raising, at some cost in speed.
  """
  for c in candidates:
    if c <= dim and dim % c == 0:
      return c
  c = 128
  while c * 2 <= dim and dim % (c * 2) == 0:
    c *= 2
  return min(c, dim)


def forward_vmem_bytes(bm: int, bk: int, bn: int, k: int, n: int) -> int:
  """Estimates the forward kernel's scoped VMEM footprint in bytes.

  Counts the buffers that dominate, and only the ones the chosen tiling keeps
  live:

  * The weight block is indexed `(c, b)`. At `bk == k` and `bn == n` that index
    map is constant across every grid step, so the pipeline fetches it once and
    holds a single buffer; otherwise it is double buffered.
  * The float32 `(bm, bn)` intermediate: in the single-k body there is no
    accumulator scratch, but `jnp.matmul` still materializes a float32 result
    of that size. In the multi-k body both exist.
  * The activation cache holds one FP8 tile per k step. When the caller wants
    the residual the output block doubles as the cache, and that path requires
    `bk == k`, so `grid_k * bm * bk` is the right figure either way.

  Args:
    bm: Block size along `m`.
    bk: Block size along `k`.
    bn: Block size along `n`.
    k: The contracting dimension.
    n: The output column dimension.

  Returns:
    The estimated footprint in bytes.
  """
  grid_k = -(-k // bk)
  grid_n = -(-n // bn)
  w_buffers = 1 if (grid_k == 1 and grid_n == 1) else 2
  x_block = bm * bk * 2 * 2  # bf16 in, double buffered
  w_block = bk * bn * 1 * w_buffers  # fp8 weights
  o_block = bm * bn * 2 * 2  # bf16 out, double buffered
  xq_cache = grid_k * bm * bk * 1  # fp8 activation cache
  sx_cache = grid_k * bm * 4  # f32 per-row scales
  f32_tmp = bm * bn * 4 * (2 if grid_k > 1 else 1)  # dot result, and accum
  return x_block + w_block + o_block + xq_cache + sx_cache + f32_tmp


def default_block_k(k: int) -> int:
  """Returns the default k block: the full `k` when that is cheap.

  A single k block removes the accumulator scratch and the multi-k state
  machine entirely. Above 4096 the activation and weight blocks get too large
  for that to fit alongside the output block, so fall back to a fixed tile.

  Args:
    k: The contracting dimension.
  """
  if k <= 4096 and k % 256 == 0:
    return k
  return pick_tile(k, (1024, 512, 256))


def fit_forward_tiles(m: int, k: int, n: int, bk: int) -> tuple[int, int]:
  """Picks `(bm, bn)` in measured-speed order, skipping ones that do not fit.

  `bn == n` is what matters most. The activation block is indexed `(a, c)` with
  the n axis innermost, so the pipeline re-fetches each activation tile once
  per n block: at `bn = n / 4` that reads the activation four times. The
  in-kernel cache stops each tile being re-*quantized*, but it cannot stop it
  being re-*fetched*. Collapsing the n grid to a single step removes the
  re-fetch entirely (worth ~110 us, or ~15%, at (32768, 3072, 4096) on TPU7x).

  Within `bn == n`, a larger `bm` means fewer grid steps and is a modest win
  where VMEM allows it. The remaining entries are fallbacks for shapes where
  `n` is too large for a single block.

  Args:
    m: The row dimension.
    k: The contracting dimension.
    n: The output column dimension.
    bk: The k block size already chosen (see `default_block_k`).

  Returns:
    `(bm, bn)`: the first preferred tiling that divides the shape and fits the
    VMEM budget, else a conservative fallback.
  """
  preference = (
      (1024, n),
      (512, n),
      (256, n),
      (1024, 1024),
      (1024, 512),
      (512, 1024),
      (512, 512),
      (512, 256),
      (256, 512),
      (256, 256),
  )
  for bm, bn in preference:
    if bm > m or m % bm or bn > n or n % bn:
      continue
    if forward_vmem_bytes(bm, bk, bn, k, n) <= VMEM_BUDGET_BYTES:
      return bm, bn
  return pick_tile(m, (256, 128)), pick_tile(n, (256, 128))


def quantize_tile_fp8(
    x: jax.Array, eps: float = 1e-7
) -> tuple[jax.Array, jax.Array]:
  """Quantizes a tile to FP8 with one absmax scale per row (last axis).

  The scaling deliberately stays in `x`'s own dtype. Writing the obvious
  `x / scale` with a float32 scale promotes the whole tile to float32, which at
  the tile sizes this kernel wants is an 8 MB temporary against 64 MiB of VMEM
  and is what made large-tile configurations fail to compile. Only the tiny
  per-row scale is float32. There is no clamp: scaling by exactly
  `FP8_MAX / absmax` puts every element inside `[-FP8_MAX, FP8_MAX]` by
  construction.

  Args:
    x: The tile to quantize.
    eps: Lower bound on the per-row absmax, so an all-zero row gets a finite
      scale.

  Returns:
    `(xq, scale)` with `xq` in `FP8_DTYPE` and `scale` a float32 array of
    shape `x.shape[:-1] + (1,)` such that `x ~= xq * scale`.
  """
  max_abs = jnp.maximum(jnp.max(jnp.abs(x), axis=-1, keepdims=True), eps)
  scale = (max_abs / FP8_MAX).astype(jnp.float32)
  inv = (FP8_MAX / max_abs).astype(x.dtype)
  xq = (x * inv).astype(FP8_DTYPE)
  return xq, scale


def _fused_fp8_matmul_kernel(
    x_hbm: jax.Ref,
    w_hbm: jax.Ref,
    sw_hbm: jax.Ref,
    o_hbm: jax.Ref,
    *residual_and_scratch: jax.Ref,
    bm: int,
    bk: int,
    bn: int,
    return_xq: bool,
):
  """Pipelined forward kernel body. See `fused_fp8_matmul_pallas`.

  When `return_xq` is set the kernel also writes out the quantized activations
  and their per-row scales. They are computed here regardless, so emitting them
  costs one extra `(m, k)` FP8 write and saves the backward pass an entire
  re-quantization of the activation (a `(m, k)` bf16 read plus an FP8 write).

  Args:
    x_hbm: `(m, k)` activations.
    w_hbm: `(k, n)` FP8 weights.
    sw_hbm: `(n,)` float32 weight scales.
    o_hbm: `(m, n)` output.
    *residual_and_scratch: With `return_xq`, the `(m, k)` FP8 activation output
      and the `(m, SX_LANES)` float32 scale output, followed by the scratch
      buffers; otherwise just the scratch buffers. Scratch is, in order, the
      float32 accumulator, the FP8 activation cache and the scale cache.
    bm: Block size along `m`.
    bk: Block size along `k`.
    bn: Block size along `n`.
    return_xq: Whether the residual outputs are present.
  """
  if return_xq:
    xq_hbm, sxq_hbm, accum_vmem, xq_cache, sx_cache = residual_and_scratch
  else:
    xq_hbm = sxq_hbm = None
    accum_vmem, xq_cache, sx_cache = residual_and_scratch

  m, k = x_hbm.shape
  _, n = w_hbm.shape
  grid = (pl.cdiv(m, bm), pl.cdiv(n, bn), pl.cdiv(k, bk))

  x_spec = pl.BlockSpec((bm, bk), lambda a, b, c: (a, c))
  w_spec = pl.BlockSpec((bk, bn), lambda a, b, c: (c, b))
  # A 1-D VMEM block for the column scales: a (1, bn) block would hit the same
  # narrow-block trap described at `SX_LANES`.
  sw_spec = pl.BlockSpec((bn,), lambda a, b, c: (b,))
  o_spec = pl.BlockSpec((bm, bn), lambda a, b, c: (a, b))
  # The residual blocks are indexed (a, c) and (a, 0): they do not depend on
  # the n-block index, so the pipeline holds one buffer across the b sweep and
  # flushes it when (a, c) changes.
  xq_out_spec = pl.BlockSpec((bm, bk), lambda a, b, c: (a, c))
  sxq_out_spec = pl.BlockSpec((bm, SX_LANES), lambda a, b, c: (a, 0))

  def _load_xq(
      x_vmem: jax.Ref,
      xq_out_vmem: jax.Ref | None,
      sxq_out_vmem: jax.Ref | None,
  ) -> tuple[jax.Array, jax.Array]:
    """Quantizes each `(bm, bk)` tile exactly once and reuses it across n.

    `x_spec` does not depend on the n-block index `b`, so a naive body would
    re-quantize the identical tile `cdiv(n, bn)` times. Quantization is VPU
    work sitting in an MXU loop, so that redundancy dominates runtime. The
    cache is keyed by the k-block index `c`: with `c` innermost, the `b == 0`
    sweep visits every `c` once and fills the cache before any `b > 0` step
    reads it.

    When the caller wants the residual, the output block *is* the cache.
    `return_xq` requires `bk == k`, so there is exactly one k block and the
    `(bm, bk)` output buffer has the same shape and lifetime as the cache slot
    it would otherwise duplicate; keeping both stored the tile to VMEM twice
    per grid step and cost ~235 us at (32768, 3072, 4096).

    Args:
      x_vmem: The `(bm, bk)` activation block.
      xq_out_vmem: The `(bm, bk)` FP8 residual output block, if requested.
      sxq_out_vmem: The `(bm, SX_LANES)` scale residual output block, if
        requested.

    Returns:
      `(xq, sx)`: the FP8 tile and its `(bm, 1)` float32 row scales.
    """
    nind = pl.program_id(1)
    cind = pl.program_id(2)

    if xq_out_vmem is not None and sxq_out_vmem is not None:
      xq_out, sxq_out = xq_out_vmem, sxq_out_vmem

      @pl.when(nind == 0)
      def _fill_out():
        xq_t, sx_t = quantize_tile_fp8(x_vmem[...])
        xq_out[...] = xq_t
        sxq_out[...] = jnp.broadcast_to(
            sx_t.astype(sxq_out.dtype), sxq_out.shape
        )
        # The scale is still kept narrow in scratch: reading it back out of
        # the lane-padded output block would need a slice inside the kernel.
        sx_cache[cind] = sx_t

      return xq_out[...], sx_cache[cind]

    @pl.when(nind == 0)
    def _fill():
      xq_t, sx_t = quantize_tile_fp8(x_vmem[...])
      xq_cache[cind] = xq_t
      sx_cache[cind] = sx_t

    return xq_cache[cind], sx_cache[cind]

  def body_single_k(x_vmem, w_vmem, sw_vmem, o_vmem, *res_vmem):
    """Direct pass when `bk == k`: no accumulator needed."""
    xq_vmem, sxq_vmem = res_vmem if res_vmem else (None, None)
    xq, sx = _load_xq(x_vmem, xq_vmem, sxq_vmem)
    xw = jnp.matmul(xq, w_vmem[...], preferred_element_type=jnp.float32)
    o_vmem[...] = (xw * (sx * sw_vmem[...])).astype(o_vmem.dtype)

  def body_multi_k(x_vmem, w_vmem, sw_vmem, o_vmem, *res_vmem):
    """Pipelined reduction over k blocks when `bk < k`."""
    xq_vmem, sxq_vmem = res_vmem if res_vmem else (None, None)
    kind = pl.program_id(2)

    @pl.when(kind == 0)
    def _init():
      accum_vmem[...] = jnp.zeros_like(accum_vmem)

    xq, sx = _load_xq(x_vmem, xq_vmem, sxq_vmem)
    xw = jnp.matmul(xq, w_vmem[...], preferred_element_type=jnp.float32)
    # Each k block is quantized against its own absmax, so the partial product
    # must be rescaled before it is accumulated.
    accum_vmem[...] += xw * (sx * sw_vmem[...])

    @pl.when(kind == pl.num_programs(2) - 1)
    def _write():
      o_vmem[...] = accum_vmem[...].astype(o_vmem.dtype)

  single_k = grid[2] == 1
  body = body_single_k if single_k else body_multi_k
  dimension_semantics = (
      pltpu.PARALLEL,
      pltpu.PARALLEL,
      pltpu.PARALLEL if single_k else pltpu.ARBITRARY,
  )

  if return_xq:
    out_specs = [o_spec, xq_out_spec, sxq_out_spec]
    out_refs = (o_hbm, xq_hbm, sxq_hbm)
  else:
    out_specs = o_spec
    out_refs = (o_hbm,)

  pltpu.emit_pipeline(
      body,
      grid=grid,
      in_specs=[x_spec, w_spec, sw_spec],
      out_specs=out_specs,
      core_axis_name=_CORE_AXIS_NAME,
      dimension_semantics=dimension_semantics,
  )(x_hbm, w_hbm, sw_hbm, *out_refs)


@overload
def fused_fp8_matmul_pallas(
    x: jax.Array,
    w: jax.Array,
    sw: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    return_xq: Literal[False] = ...,
) -> jax.Array:
  ...


@overload
def fused_fp8_matmul_pallas(
    x: jax.Array,
    w: jax.Array,
    sw: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    return_xq: Literal[True],
) -> tuple[jax.Array, jax.Array, jax.Array]:
  ...


def fused_fp8_matmul_pallas(
    x: jax.Array,
    w: jax.Array,
    sw: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    return_xq: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
  """Fused FP8 matmul: `(quantize_rows(x) @ w) * (sx * sw)`.

  Args:
    x: Activations of shape `(m, k)`, `bfloat16` or `float32`.
    w: Weights of shape `(k, n)` already quantized to `float8_e4m3fn`.
    sw: Per-column weight scales of shape `(n,)`, `float32`.
    bm: Block size along `m`. Must divide `m`.
    bk: Block size along `k`. Must divide `k`.
    bn: Block size along `n`. Must divide `n`.
    return_xq: Also return the FP8 activations and their per-row scales, for use
      as residuals by the backward pass. Requires `bk == k`: with more than one
      k block the kernel quantizes each block against its own absmax, so a
      single `(m, 1)` scale would not describe the result.

  Returns:
    `out` of shape `(m, n)` in `x.dtype`, or `(out, xq, sx)` when `return_xq`
    with `xq` of shape `(m, k)` in `float8_e4m3fn` and `sx` of shape `(m, 1)`
    in `float32`.
  """
  m, k = x.shape
  k_w, n = w.shape
  if k_w != k:
    raise ValueError(f"Contracting dims differ: x has k={k}, w has k={k_w}.")
  if sw.shape != (n,):
    raise ValueError(f"sw must have shape ({n},), got {sw.shape}.")
  if w.dtype != FP8_DTYPE:
    raise ValueError(f"w must be {FP8_DTYPE}, got {w.dtype}.")
  if m % bm:
    raise ValueError(f"m ({m}) must be divisible by bm ({bm}).")
  if k % bk:
    raise ValueError(f"k ({k}) must be divisible by bk ({bk}).")
  if n % bn:
    raise ValueError(f"n ({n}) must be divisible by bn ({bn}).")
  if return_xq and bk != k:
    raise ValueError(f"return_xq needs a single k block (bk == k), got {bk=}.")

  grid_k = pl.cdiv(k, bk)

  if return_xq:
    out_type = (
        jax.ShapeDtypeStruct((m, n), x.dtype),
        jax.ShapeDtypeStruct((m, k), FP8_DTYPE),
        jax.ShapeDtypeStruct((m, SX_LANES), jnp.float32),
    )
  else:
    out_type = jax.ShapeDtypeStruct((m, n), x.dtype)

  # Scratch order matches the kernel signature: accum, xq_cache, sx_cache.
  # Buffers the chosen tiling leaves dead are allocated at a minimal shape
  # rather than dropped, so the positional layout stays fixed. At the `bn == n`
  # tiling the heuristics prefer, the accumulator is the largest buffer in the
  # kernel, so holding it dead would put the larger `bm` out of VMEM reach.
  #   accum    is read only by `body_multi_k`, i.e. when grid_k > 1.
  #   xq_cache is read only when the caller does not want the residual; with
  #            `return_xq` the output block is the cache (see `_load_xq`).
  #   sx_cache stays full size: `_load_xq` writes it on both paths.
  dead = (8, 128)  # One (sublane, lane) tile, the smallest useful allocation.
  scratch_types = [
      pltpu.VMEM((bm, bn) if grid_k > 1 else dead, jnp.float32),
      pltpu.VMEM((1,) + dead if return_xq else (grid_k, bm, bk), FP8_DTYPE),
      pltpu.VMEM((grid_k, bm, 1), jnp.float32),
  ]

  kernel = functools.partial(
      _fused_fp8_matmul_kernel, bm=bm, bk=bk, bn=bn, return_xq=return_xq
  )
  outputs = pl.kernel(
      kernel,
      out_type=out_type,
      mesh=pltpu.create_tensorcore_mesh(axis_name=_CORE_AXIS_NAME),
      scratch_types=scratch_types,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT_BYTES),
  )(x, w, sw)

  if return_xq:
    out, xq, sx_wide = outputs
    # See `SX_LANES`: narrow the lane-padded scale back to `(m, 1)`.
    return out, xq, sx_wide[:, :1]
  return outputs
