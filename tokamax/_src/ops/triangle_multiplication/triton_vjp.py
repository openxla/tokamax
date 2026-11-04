# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Triton Triangle Multiplication VJP implementation."""

from collections.abc import Sequence
import dataclasses
from typing import Annotated, Any, ClassVar, Final, Literal, override

import immutabledict
import jax
from jax import lax
from jax.extend import backend
import jax.numpy as jnp
import jax_triton as jt
from jaxtyping import Array, Bool, Float  # pylint: disable=g-importing-member,g-multiple-import
import pydantic
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src import precision as precision_lib
from tokamax._src import pydantic as pydantic_lib
from tokamax._src import triton_utils
from tokamax._src.ops import op
from tokamax._src.ops.triangle_multiplication import base
import triton
import triton.language as tl

type Key = immutabledict.immutabledict[str, Any]

_SUPPORTED_DTYPES: Final[frozenset[jnp.dtype]] = frozenset(
    {jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)}
)
_SUPPORTED_CHANNEL_DIMS: Final[frozenset[int]] = frozenset({16, 32, 64, 128})
# The persistent TMA kernels are only used when every channel dim they tile is
# at least this large (64 16-bit elements = 128 bytes, the widest TMA swizzle).
# Narrower channel dims use the pointer-based kernels.
_MIN_TMA_CHANNEL_DIM: Final[int] = 64
# The kernels address memory with int32 offsets. The largest offset is
# `N * N * max(C, H, D) - 1`, which must not overflow.
_MAX_INT32: Final[int] = 2**31 - 1

# `tl.arange` requires power-of-two tile sizes, and TMA descriptors (used by the
# persistent kernels) support at most 256 elements per block dimension.
_BlockM = Annotated[pydantic_lib.PowerOfTwo, pydantic.Field(ge=32, le=256)]
_NumWarps = Annotated[pydantic_lib.PowerOfTwo, pydantic.Field(le=16)]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Configuration for Triton Triangle Multiplication VJP kernels.

  Attributes:
    block_m: Tile size along the flattened token-pair dimension M = N * N. Must
      be a power of two in [32, 256].
    num_warps: Number of warps per CTA. Must be a power of two, at most 16.
    num_stages: Number of software pipeline stages.
    epilogue_block_m: Optional tile size override for the epilogue backward
      kernel.
    epilogue_num_warps: Optional num_warps override for the epilogue backward
      kernel.
    epilogue_num_stages: Optional num_stages override for the epilogue backward
      kernel.
    prologue_persistent: Whether to use the persistent TMA prologue backward
      kernel. If `None`, it is used on SM90 and SM100 when M is a multiple of
      `block_m` and min(C, H) >= 64.
    epilogue_persistent: Whether to use the persistent TMA epilogue backward
      kernel. If `None`, it is used on SM90 and SM100 when M is a multiple of
      `block_m` and min(C, H, D) >= 64.
    ctas_per_sm: Number of persistent CTAs per SM.
  """

  block_m: _BlockM = 64
  num_warps: _NumWarps = 4
  num_stages: pydantic.PositiveInt = 2
  epilogue_block_m: _BlockM | None = None
  epilogue_num_warps: _NumWarps | None = None
  epilogue_num_stages: pydantic.PositiveInt | None = None
  prologue_persistent: bool | None = None
  epilogue_persistent: bool | None = None
  ctas_per_sm: pydantic.PositiveInt = 2


# ------------------------------------------------------------------------------
# Per-tile device helpers.
# ------------------------------------------------------------------------------


@triton.jit
def _hide_from_cse(x):
  """Prevents MLIR CSE from keeping `x.to(tl.float32)` live across `tl.dot`s."""
  return tl.inline_asm_elementwise(
      "mov.b16 $0, $1;",
      "=h,h",
      [x],
      dtype=x.dtype,
      is_pure=True,
      pack=1,
  )


@triton.jit
def _layer_norm_2d_hat_rstd(
    x,
    eps: tl.constexpr,
    dim_size: tl.constexpr,
):
  """Returns normalized `(x - mean) * rstd` and `rstd` along axis 1 in FP32."""
  x_f32 = x.to(tl.float32)
  mean = tl.sum(x_f32, axis=1) * (1.0 / dim_size)
  centered = x_f32 - mean[:, None]
  var = tl.sum(centered * centered, axis=1) * (1.0 / dim_size)
  if eps <= 0.0:
    rstd = tl.where(var > 0.0, tl.rsqrt(var), 0.0)
  else:
    rstd = tl.rsqrt(var + eps)
  return centered * rstd[:, None], rstd


@triton.jit
def _layer_norm_2d(
    x,
    scale,
    offset,
    eps: tl.constexpr,
    dim_size: tl.constexpr,
):
  """Computes LayerNorm along axis 1 of a 2D tile in FP32."""
  normed, _ = _layer_norm_2d_hat_rstd(x, eps, dim_size)
  return normed * scale.to(tl.float32)[None, :] + offset.to(tl.float32)[None, :]


@triton.jit
def _layer_norm_2d_bwd(
    dy,
    x_hat_16,
    rstd,
    scale,
    dim_size: tl.constexpr,
):
  """Computes dx, dscale, doffset for LayerNorm along axis 1 in FP32."""
  x_hat = _hide_from_cse(x_hat_16).to(tl.float32)
  safe_x_hat = tl.where(dy != 0.0, x_hat, 0.0)
  dscale = tl.sum(dy * safe_x_hat, axis=0)
  doffset = tl.sum(dy, axis=0)
  dx_hat = dy * scale.to(tl.float32)[None, :]
  c1 = tl.sum(dx_hat, axis=1) * (1.0 / dim_size)
  c2 = tl.sum(dx_hat * safe_x_hat, axis=1) * (1.0 / dim_size)
  raw_dx = dx_hat - c1[:, None] - safe_x_hat * c2[:, None]
  dx = tl.where(raw_dx != 0.0, raw_dx * rstd[:, None], 0.0)
  return dx, dscale, doffset


@triton.jit
def _load_param_vector(ptr, size: tl.constexpr):
  """Loads a (size,) parameter vector that is reused by every CTA."""
  return tl.load(ptr + tl.arange(0, size), eviction_policy="evict_last")


@triton.jit
def _load_param_matrix(
    ptr, rows: tl.constexpr, cols: tl.constexpr, row_stride: tl.constexpr
):
  """Loads a (rows, cols) weight matrix that is reused by every CTA."""
  offs = tl.arange(0, rows)[:, None] * row_stride + tl.arange(0, cols)[None, :]
  return tl.load(ptr + offs, eviction_policy="evict_last")


@triton.jit
def _store_transposed(
    out_ptr,
    value,
    offs_m,
    m,
    h_dim: tl.constexpr,
    is_even_m,
    eviction_policy,
):
  """Stores a (BLOCK_M, H) tile transposed into an (H, M) array."""
  ptrs = out_ptr + tl.arange(0, h_dim)[:, None] * m + offs_m[None, :]
  value_t = tl.trans(value.to(out_ptr.dtype.element_ty))
  if is_even_m:
    tl.store(ptrs, value_t, eviction_policy=eviction_policy)
  else:
    tl.store(
        ptrs,
        value_t,
        mask=(offs_m < m)[None, :],
        eviction_policy=eviction_policy,
    )


@triton.jit
def _epilogue_bwd_stage1(
    x,
    ln_in_scale,
    ln_in_offset,
    w_gate_out,
    eps: tl.constexpr,
    c_dim: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Computes `left_act` and `gate = sigmoid(left_act @ w_gate_out)`."""
  left_act = _layer_norm_2d(x, ln_in_scale, ln_in_offset, eps, c_dim).to(
      dot_dtype
  )
  g = tl.dot(left_act, w_gate_out, input_precision=input_precision)
  return left_act, tl.sigmoid(g)


@triton.jit
def _epilogue_bwd_stage2(
    t3,
    ln_out_scale,
    ln_out_offset,
    w_proj_out,
    eps: tl.constexpr,
    h_dim: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Computes `t_hat_16`, `rstd_out`, `act`, and `p = act @ w_proj_out`."""
  t_hat, rstd_out = _layer_norm_2d_hat_rstd(tl.trans(t3), eps, h_dim)
  t_hat_16 = t_hat.to(dot_dtype)
  act = (
      t_hat * ln_out_scale.to(tl.float32)[None, :]
      + ln_out_offset.to(tl.float32)[None, :]
  ).to(dot_dtype)
  p = tl.dot(act, w_proj_out, input_precision=input_precision)
  return t_hat_16, rstd_out, act, p


@triton.jit
def _epilogue_bwd_stage3(
    dout,
    gate,
    p,
    act,
    t_hat_16,
    rstd_out,
    left_act,
    offs_m,
    ln_out_scale,
    w_gate_out_t,
    w_proj_out_t,
    dw_proj_acc,
    dw_gate_acc,
    dln_scale_acc,
    dln_offset_acc,
    dt3_ptr,
    dleft_act_epi_ptr,
    m,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    is_even_m,
    eviction_policy,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Computes backward dots, LayerNorm_out bwd, and stores `dt3`, `dleft_act`."""
  dout_f32 = dout.to(tl.float32)
  dout_nz = dout_f32 != 0.0
  dp = tl.where(dout_nz, dout_f32 * gate, 0.0).to(dot_dtype)
  dg = tl.where(dout_nz, dout_f32 * p * gate * (1.0 - gate), 0.0).to(dot_dtype)

  # Complete the proj_out + LayerNorm_out backward path first so `act`, `dp`,
  # `t_hat_16`, and `rstd_out` are dead before the gate_out backward dots.
  dw_proj_acc = tl.dot(
      tl.trans(act), dp, acc=dw_proj_acc, input_precision=input_precision
  )
  dact = tl.dot(dp, w_proj_out_t, input_precision=input_precision)
  dt3_t, dln_scale_tile, dln_offset_tile = _layer_norm_2d_bwd(
      dact, t_hat_16, rstd_out, ln_out_scale, h_dim
  )
  _store_transposed(
      dt3_ptr, dt3_t, offs_m, m, h_dim, is_even_m, eviction_policy
  )

  safe_left_act = tl.where(tl.abs(left_act) < float("inf"), left_act, 0.0)
  dw_gate_acc = tl.dot(
      tl.trans(safe_left_act),
      dg,
      acc=dw_gate_acc,
      input_precision=input_precision,
  )
  dleft_act_epi = tl.dot(
      dg, w_gate_out_t, input_precision=input_precision
  ).to(dleft_act_epi_ptr.dtype.element_ty)
  dleft_ptrs = (
      dleft_act_epi_ptr + offs_m[:, None] * c_dim + tl.arange(0, c_dim)[None, :]
  )
  if is_even_m:
    tl.store(dleft_ptrs, dleft_act_epi, eviction_policy="evict_last")
  else:
    tl.store(
        dleft_ptrs,
        dleft_act_epi,
        mask=(offs_m < m)[:, None],
        eviction_policy="evict_last",
    )
  return (
      dw_proj_acc,
      dw_gate_acc,
      dln_scale_acc + dln_scale_tile,
      dln_offset_acc + dln_offset_tile,
  )


@triton.jit
def _epilogue_bwd_tile(
    x,
    t3,
    dout,
    offs_m,
    ln_in_scale,
    ln_in_offset,
    ln_out_scale,
    ln_out_offset,
    w_gate_out,
    w_proj_out,
    w_gate_out_t,
    w_proj_out_t,
    dw_proj_acc,
    dw_gate_acc,
    dln_scale_acc,
    dln_offset_acc,
    dt3_ptr,
    dleft_act_epi_ptr,
    m,
    eps: tl.constexpr,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    is_even_m,
    eviction_policy,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Computes one tile of the epilogue backward pass."""
  left_act, gate = _epilogue_bwd_stage1(
      x, ln_in_scale, ln_in_offset, w_gate_out, eps, c_dim, dot_dtype,
      input_precision,
  )
  t_hat_16, rstd_out, act, p = _epilogue_bwd_stage2(
      t3, ln_out_scale, ln_out_offset, w_proj_out, eps, h_dim, dot_dtype,
      input_precision,
  )
  return _epilogue_bwd_stage3(
      dout,
      gate,
      p,
      act,
      t_hat_16,
      rstd_out,
      left_act,
      offs_m,
      ln_out_scale,
      w_gate_out_t,
      w_proj_out_t,
      dw_proj_acc,
      dw_gate_acc,
      dln_scale_acc,
      dln_offset_acc,
      dt3_ptr,
      dleft_act_epi_ptr,
      m,
      c_dim,
      h_dim,
      is_even_m,
      eviction_policy,
      dot_dtype,
      input_precision,
  )


@triton.jit
def _store_epilogue_bwd_partials(
    dw_out_part_ptr,
    dln_out_part_ptr,
    pid,
    dw_proj_acc,
    dw_gate_acc,
    dln_scale_acc,
    dln_offset_acc,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    d_dim: tl.constexpr,
):
  """Stores CTA-local partial weight gradients for the epilogue backward."""
  row_base = pid * (h_dim + c_dim) * d_dim
  offs_hd = (
      tl.arange(0, h_dim)[:, None] * d_dim + tl.arange(0, d_dim)[None, :]
  )
  tl.store(dw_out_part_ptr + row_base + offs_hd, dw_proj_acc)
  offs_cd = (
      tl.arange(0, c_dim)[:, None] * d_dim + tl.arange(0, d_dim)[None, :]
  )
  tl.store(
      dw_out_part_ptr + row_base + h_dim * d_dim + offs_cd, dw_gate_acc
  )
  offs_h = tl.arange(0, h_dim)
  tl.store(dln_out_part_ptr + (pid * 2) * h_dim + offs_h, dln_scale_acc)
  tl.store(dln_out_part_ptr + (pid * 2 + 1) * h_dim + offs_h, dln_offset_acc)


@triton.jit
def _prologue_bwd_stage1(
    x,
    mask_val,
    ln_scale,
    ln_offset,
    w_gate,
    eps: tl.constexpr,
    c_dim: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Computes `x_hat_16`, `rstd_in`, `act`, and `s = sigmoid(act @ w_gate)`."""
  mask_b = mask_val.to(tl.int1)[:, None]
  x_hat, rstd_in = _layer_norm_2d_hat_rstd(x, eps, c_dim)
  x_hat_16 = x_hat.to(dot_dtype)
  act = tl.where(
      mask_b,
      x_hat * ln_scale.to(tl.float32)[None, :]
      + ln_offset.to(tl.float32)[None, :],
      0.0,
  ).to(dot_dtype)
  g = tl.dot(act, w_gate, input_precision=input_precision)
  s = tl.sigmoid(g)
  return x_hat_16, rstd_in, mask_b, act, s


@triton.jit
def _prologue_bwd_stage2(
    dbranch,
    mask_b,
    act,
    s,
    w_proj,
    dw_proj_acc,
    dw_gate_acc,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Computes `dp`, `dg`, and accumulates `dw_proj_acc` and `dw_gate_acc`."""
  dbranch_t = tl.where(mask_b, tl.trans(dbranch).to(tl.float32), 0.0)
  dp_f32 = dbranch_t * s
  dp = dp_f32.to(dot_dtype)
  q = dp_f32 * (1.0 - s)
  act_t = tl.trans(act)
  dw_proj_acc = tl.dot(
      act_t, dp, acc=dw_proj_acc, input_precision=input_precision
  )
  p = tl.dot(act, w_proj, input_precision=input_precision)
  dg = (q * p).to(dot_dtype)
  dw_gate_acc = tl.dot(
      act_t, dg, acc=dw_gate_acc, input_precision=input_precision
  )
  return dp, dg, dw_proj_acc, dw_gate_acc


@triton.jit
def _prologue_bwd_stage3(
    dact_in,
    dp,
    dg,
    x_hat_16,
    rstd_in,
    offs_m,
    ln_scale,
    w_proj_t,
    w_gate_t,
    dln_scale_acc,
    dln_offset_acc,
    out_ptr,
    m,
    c_dim: tl.constexpr,
    apply_ln_bwd: tl.constexpr,
    is_even_m,
    eviction_policy,
    input_precision: tl.constexpr,
):
  """Computes `dact`, optional LayerNorm_in bwd, and stores output."""
  dact = tl.dot(
      dp,
      w_proj_t,
      acc=dact_in.to(tl.float32),
      input_precision=input_precision,
  )
  dact = tl.dot(
      dg, w_gate_t, acc=dact, input_precision=input_precision
  )
  if apply_ln_bwd:
    out_val, dln_scale_tile, dln_offset_tile = _layer_norm_2d_bwd(
        dact, x_hat_16, rstd_in, ln_scale, c_dim
    )
    dln_scale_acc += dln_scale_tile
    dln_offset_acc += dln_offset_tile
  else:
    out_val = dact

  out_ptrs = out_ptr + offs_m[:, None] * c_dim + tl.arange(0, c_dim)[None, :]
  out_cast = out_val.to(out_ptr.dtype.element_ty)
  if is_even_m:
    if apply_ln_bwd:
      tl.store(out_ptrs, out_cast, eviction_policy=eviction_policy)
    else:
      tl.store(out_ptrs, out_cast, eviction_policy="evict_last")
  else:
    tl.store(
        out_ptrs,
        out_cast,
        mask=(offs_m < m)[:, None],
        eviction_policy=eviction_policy,
    )
  return dln_scale_acc, dln_offset_acc


@triton.jit
def _store_prologue_bwd_partials(
    dw_branch_part_ptr,
    dln_in_part_ptr,
    pid,
    dw_proj_acc,
    dw_gate_acc,
    dln_scale_acc,
    dln_offset_acc,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    apply_ln_bwd: tl.constexpr,
):
  """Stores CTA-local partial weight gradients for a single GLU branch."""
  stride_ch: tl.constexpr = c_dim * h_dim
  base_ptr = dw_branch_part_ptr + pid * 2 * stride_ch
  offs = (
      tl.arange(0, c_dim)[:, None] * h_dim + tl.arange(0, h_dim)[None, :]
  )
  tl.store(base_ptr + offs, dw_proj_acc)
  tl.store(base_ptr + stride_ch + offs, dw_gate_acc)
  if apply_ln_bwd:
    offs_c = tl.arange(0, c_dim)
    tl.store(dln_in_part_ptr + (pid * 2) * c_dim + offs_c, dln_scale_acc)
    tl.store(dln_in_part_ptr + (pid * 2 + 1) * c_dim + offs_c, dln_offset_acc)


# ------------------------------------------------------------------------------
# Backward kernels.
# ------------------------------------------------------------------------------


@triton.jit
def _epilogue_bwd_kernel(
    dout_ptr,
    t3_ptr,
    x_ptr,
    w_proj_out_ptr,
    w_gate_out_ptr,
    ln_in_scale_ptr,
    ln_in_offset_ptr,
    ln_out_scale_ptr,
    ln_out_offset_ptr,
    dt3_ptr,
    dleft_act_epi_ptr,
    dw_out_part_ptr,
    dln_out_part_ptr,
    m: tl.constexpr,
    eps: tl.constexpr,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    d_dim: tl.constexpr,
    block_m: tl.constexpr,
    is_even_m: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Persistent pointer-based epilogue backward kernel."""
  ln_in_scale = _load_param_vector(ln_in_scale_ptr, c_dim)
  ln_in_offset = _load_param_vector(ln_in_offset_ptr, c_dim)
  ln_out_scale = _load_param_vector(ln_out_scale_ptr, h_dim)
  ln_out_offset = _load_param_vector(ln_out_offset_ptr, h_dim)
  w_gate_out = _load_param_matrix(w_gate_out_ptr, c_dim, d_dim, d_dim).to(
      dot_dtype
  )
  w_proj_out = _load_param_matrix(w_proj_out_ptr, h_dim, d_dim, d_dim).to(
      dot_dtype
  )
  w_gate_out_t = tl.trans(w_gate_out)
  w_proj_out_t = tl.trans(w_proj_out)

  dw_proj_acc = tl.zeros((h_dim, d_dim), dtype=tl.float32)
  dw_gate_acc = tl.zeros((c_dim, d_dim), dtype=tl.float32)
  dln_scale_acc = tl.zeros((h_dim,), dtype=tl.float32)
  dln_offset_acc = tl.zeros((h_dim,), dtype=tl.float32)

  pid = tl.program_id(axis=0)
  num_tiles: tl.constexpr = tl.cdiv(m, block_m)
  offs_c = tl.arange(0, c_dim)[None, :]
  offs_h = tl.arange(0, h_dim)[:, None]
  offs_d = tl.arange(0, d_dim)[None, :]

  for tile_id in tl.range(pid, num_tiles, tl.num_programs(axis=0)):
    offs_m = tile_id * block_m + tl.arange(0, block_m)
    if is_even_m:
      offs_m = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    mask_m = offs_m < m

    x_ptrs = x_ptr + offs_m[:, None] * c_dim + offs_c
    if is_even_m:
      x = tl.load(x_ptrs, eviction_policy="evict_first")
    else:
      x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
    left_act, gate = _epilogue_bwd_stage1(
        x,
        ln_in_scale,
        ln_in_offset,
        w_gate_out,
        eps,
        c_dim,
        dot_dtype,
        input_precision,
    )

    t3_ptrs = t3_ptr + offs_h * m + offs_m[None, :]
    if is_even_m:
      t3 = tl.load(t3_ptrs, eviction_policy="evict_first")
    else:
      t3 = tl.load(t3_ptrs, mask=mask_m[None, :], other=0.0)
    t_hat_16, rstd_out, act, p = _epilogue_bwd_stage2(
        t3,
        ln_out_scale,
        ln_out_offset,
        w_proj_out,
        eps,
        h_dim,
        dot_dtype,
        input_precision,
    )

    dout_ptrs = dout_ptr + offs_m[:, None] * d_dim + offs_d
    if is_even_m:
      dout = tl.load(dout_ptrs, eviction_policy="evict_first")
    else:
      dout = tl.load(dout_ptrs, mask=mask_m[:, None], other=0.0)

    dw_proj_acc, dw_gate_acc, dln_scale_acc, dln_offset_acc = (
        _epilogue_bwd_stage3(
            dout,
            gate,
            p,
            act,
            t_hat_16,
            rstd_out,
            left_act,
            offs_m,
            ln_out_scale,
            w_gate_out_t,
            w_proj_out_t,
            dw_proj_acc,
            dw_gate_acc,
            dln_scale_acc,
            dln_offset_acc,
            dt3_ptr,
            dleft_act_epi_ptr,
            m,
            c_dim,
            h_dim,
            is_even_m,
            "evict_first",
            dot_dtype,
            input_precision,
        )
    )

  _store_epilogue_bwd_partials(
      dw_out_part_ptr,
      dln_out_part_ptr,
      pid,
      dw_proj_acc,
      dw_gate_acc,
      dln_scale_acc,
      dln_offset_acc,
      c_dim,
      h_dim,
      d_dim,
  )


@triton.jit
def _epilogue_bwd_tma_kernel(
    dout_desc,
    t3_desc,
    x_desc,
    w_proj_out_ptr,
    w_gate_out_ptr,
    ln_in_scale_ptr,
    ln_in_offset_ptr,
    ln_out_scale_ptr,
    ln_out_offset_ptr,
    m,
    dt3_ptr,
    dleft_act_epi_ptr,
    dw_out_part_ptr,
    dln_out_part_ptr,
    eps: tl.constexpr,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    d_dim: tl.constexpr,
    block_m: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Persistent TMA-pipelined epilogue backward kernel."""
  ln_in_scale = _load_param_vector(ln_in_scale_ptr, c_dim)
  ln_in_offset = _load_param_vector(ln_in_offset_ptr, c_dim)
  ln_out_scale = _load_param_vector(ln_out_scale_ptr, h_dim)
  ln_out_offset = _load_param_vector(ln_out_offset_ptr, h_dim)
  w_gate_out = _load_param_matrix(w_gate_out_ptr, c_dim, d_dim, d_dim).to(
      dot_dtype
  )
  w_proj_out = _load_param_matrix(w_proj_out_ptr, h_dim, d_dim, d_dim).to(
      dot_dtype
  )
  w_gate_out_t = tl.trans(w_gate_out)
  w_proj_out_t = tl.trans(w_proj_out)

  dw_proj_acc = tl.zeros((h_dim, d_dim), dtype=tl.float32)
  dw_gate_acc = tl.zeros((c_dim, d_dim), dtype=tl.float32)
  dln_scale_acc = tl.zeros((h_dim,), dtype=tl.float32)
  dln_offset_acc = tl.zeros((h_dim,), dtype=tl.float32)

  pid = tl.program_id(0)
  num_tiles = m // block_m
  for tile_id in tl.range(pid, num_tiles, tl.num_programs(0)):
    row_offset = tile_id * block_m
    offs_m = row_offset + tl.arange(0, block_m)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    x = x_desc.load([row_offset, 0])
    t3 = t3_desc.load([0, row_offset])
    dout = dout_desc.load([row_offset, 0])
    dw_proj_acc, dw_gate_acc, dln_scale_acc, dln_offset_acc = (
        _epilogue_bwd_tile(
            x,
            t3,
            dout,
            offs_m,
            ln_in_scale,
            ln_in_offset,
            ln_out_scale,
            ln_out_offset,
            w_gate_out,
            w_proj_out,
            w_gate_out_t,
            w_proj_out_t,
            dw_proj_acc,
            dw_gate_acc,
            dln_scale_acc,
            dln_offset_acc,
            dt3_ptr,
            dleft_act_epi_ptr,
            m,
            eps,
            c_dim,
            h_dim,
            True,  # is_even_m
            "evict_first",  # eviction_policy
            dot_dtype,
            input_precision,
        )
    )

  _store_epilogue_bwd_partials(
      dw_out_part_ptr,
      dln_out_part_ptr,
      pid,
      dw_proj_acc,
      dw_gate_acc,
      dln_scale_acc,
      dln_offset_acc,
      c_dim,
      h_dim,
      d_dim,
  )


@triton.jit
def _prologue_bwd_kernel(
    dbranch_ptr,
    dact_in_ptr,
    x_ptr,
    mask_ptr,
    w_proj_ptr,
    w_gate_ptr,
    ln_scale_ptr,
    ln_offset_ptr,
    out_ptr,
    dw_branch_part_ptr,
    dln_in_part_ptr,
    m: tl.constexpr,
    eps: tl.constexpr,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    branch_idx: tl.constexpr,
    apply_ln_bwd: tl.constexpr,
    block_m: tl.constexpr,
    is_even_m: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Persistent pointer-based single-branch prologue backward kernel."""
  ln_scale = _load_param_vector(ln_scale_ptr, c_dim)
  ln_offset = _load_param_vector(ln_offset_ptr, c_dim)
  w_offset: tl.constexpr = branch_idx * h_dim
  w_proj = _load_param_matrix(
      w_proj_ptr + w_offset, c_dim, h_dim, 2 * h_dim
  ).to(dot_dtype)
  w_gate = _load_param_matrix(
      w_gate_ptr + w_offset, c_dim, h_dim, 2 * h_dim
  ).to(dot_dtype)
  w_proj_t = tl.trans(w_proj)
  w_gate_t = tl.trans(w_gate)

  dw_proj_acc = tl.zeros((c_dim, h_dim), dtype=tl.float32)
  dw_gate_acc = tl.zeros((c_dim, h_dim), dtype=tl.float32)
  dln_scale_acc = tl.zeros((c_dim,), dtype=tl.float32)
  dln_offset_acc = tl.zeros((c_dim,), dtype=tl.float32)

  pid = tl.program_id(axis=0)
  num_tiles: tl.constexpr = tl.cdiv(m, block_m)
  offs_c = tl.arange(0, c_dim)[None, :]
  offs_h = tl.arange(0, h_dim)[:, None]

  for tile_id in tl.range(pid, num_tiles, tl.num_programs(axis=0)):
    offs_m = tile_id * block_m + tl.arange(0, block_m)
    if is_even_m:
      offs_m = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    mask_m = offs_m < m

    x_ptrs = x_ptr + offs_m[:, None] * c_dim + offs_c
    if is_even_m:
      if apply_ln_bwd:
        x = tl.load(x_ptrs, eviction_policy="evict_first")
        mask_val = tl.load(mask_ptr + offs_m, eviction_policy="evict_first")
      else:
        x = tl.load(x_ptrs, eviction_policy="evict_last")
        mask_val = tl.load(mask_ptr + offs_m, eviction_policy="evict_last")
    else:
      x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
      mask_val = tl.load(mask_ptr + offs_m, mask=mask_m, other=0)

    x_hat_16, rstd_in, mask_b, act, s = _prologue_bwd_stage1(
        x,
        mask_val,
        ln_scale,
        ln_offset,
        w_gate,
        eps,
        c_dim,
        dot_dtype,
        input_precision,
    )

    dbranch_ptrs = dbranch_ptr + offs_h * m + offs_m[None, :]
    if is_even_m:
      dbranch = tl.load(dbranch_ptrs, eviction_policy="evict_first")
    else:
      dbranch = tl.load(dbranch_ptrs, mask=mask_m[None, :], other=0.0)

    dp, dg, dw_proj_acc, dw_gate_acc = _prologue_bwd_stage2(
        dbranch,
        mask_b,
        act,
        s,
        w_proj,
        dw_proj_acc,
        dw_gate_acc,
        dot_dtype,
        input_precision,
    )

    dact_in_ptrs = dact_in_ptr + offs_m[:, None] * c_dim + offs_c
    if is_even_m:
      dact_in = tl.load(dact_in_ptrs, eviction_policy="evict_first")
    else:
      dact_in = tl.load(dact_in_ptrs, mask=mask_m[:, None], other=0.0)

    dln_scale_acc, dln_offset_acc = _prologue_bwd_stage3(
        dact_in,
        dp,
        dg,
        x_hat_16,
        rstd_in,
        offs_m,
        ln_scale,
        w_proj_t,
        w_gate_t,
        dln_scale_acc,
        dln_offset_acc,
        out_ptr,
        m,
        c_dim,
        apply_ln_bwd,
        is_even_m,
        "evict_first",
        input_precision,
    )

  _store_prologue_bwd_partials(
      dw_branch_part_ptr,
      dln_in_part_ptr,
      pid,
      dw_proj_acc,
      dw_gate_acc,
      dln_scale_acc,
      dln_offset_acc,
      c_dim,
      h_dim,
      apply_ln_bwd,
  )


@triton.jit
def _prologue_bwd_tma_kernel(
    dbranch_desc,
    dact_in_desc,
    x_desc,
    mask_ptr,
    w_proj_ptr,
    w_gate_ptr,
    ln_scale_ptr,
    ln_offset_ptr,
    out_ptr,
    dw_branch_part_ptr,
    dln_in_part_ptr,
    m: tl.constexpr,
    eps: tl.constexpr,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    branch_idx: tl.constexpr,
    apply_ln_bwd: tl.constexpr,
    block_m: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Persistent TMA-pipelined single-branch prologue backward kernel."""
  ln_scale = _load_param_vector(ln_scale_ptr, c_dim)
  ln_offset = _load_param_vector(ln_offset_ptr, c_dim)
  w_offset: tl.constexpr = branch_idx * h_dim
  w_proj = _load_param_matrix(
      w_proj_ptr + w_offset, c_dim, h_dim, 2 * h_dim
  ).to(dot_dtype)
  w_gate = _load_param_matrix(
      w_gate_ptr + w_offset, c_dim, h_dim, 2 * h_dim
  ).to(dot_dtype)
  w_proj_t = tl.trans(w_proj)
  w_gate_t = tl.trans(w_gate)

  dw_proj_acc = tl.zeros((c_dim, h_dim), dtype=tl.float32)
  dw_gate_acc = tl.zeros((c_dim, h_dim), dtype=tl.float32)
  dln_scale_acc = tl.zeros((c_dim,), dtype=tl.float32)
  dln_offset_acc = tl.zeros((c_dim,), dtype=tl.float32)

  pid = tl.program_id(0)
  num_tiles: tl.constexpr = m // block_m
  for tile_id in tl.range(pid, num_tiles, tl.num_programs(0)):
    row_offset = tile_id * block_m
    offs_m = row_offset + tl.arange(0, block_m)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    x = x_desc.load([row_offset, 0])
    mask_val = tl.load(mask_ptr + offs_m, eviction_policy="evict_first")
    x_hat_16, rstd_in, mask_b, act, s = _prologue_bwd_stage1(
        x,
        mask_val,
        ln_scale,
        ln_offset,
        w_gate,
        eps,
        c_dim,
        dot_dtype,
        input_precision,
    )

    dbranch = dbranch_desc.load([0, row_offset])
    dp, dg, dw_proj_acc, dw_gate_acc = _prologue_bwd_stage2(
        dbranch,
        mask_b,
        act,
        s,
        w_proj,
        dw_proj_acc,
        dw_gate_acc,
        dot_dtype,
        input_precision,
    )

    dact_in = dact_in_desc.load([row_offset, 0])
    dln_scale_acc, dln_offset_acc = _prologue_bwd_stage3(
        dact_in,
        dp,
        dg,
        x_hat_16,
        rstd_in,
        offs_m,
        ln_scale,
        w_proj_t,
        w_gate_t,
        dln_scale_acc,
        dln_offset_acc,
        out_ptr,
        m,
        c_dim,
        apply_ln_bwd,
        True,  # is_even_m
        "evict_first",  # eviction_policy
        input_precision,
    )

  _store_prologue_bwd_partials(
      dw_branch_part_ptr,
      dln_in_part_ptr,
      pid,
      dw_proj_acc,
      dw_gate_acc,
      dln_scale_acc,
      dln_offset_acc,
      c_dim,
      h_dim,
      apply_ln_bwd,
  )


# ------------------------------------------------------------------------------
# Host-side backward launchers.
# ------------------------------------------------------------------------------


def _epilogue_bwd(
    dout_flat: jax.Array,
    t3_flat: jax.Array,
    x_flat: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    *,
    block_m: int,
    num_warps: int,
    num_stages: int,
    persistent: bool,
    max_grid_size: int,
    **common_kwargs,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
  """Backward pass through the epilogue."""
  m, c = x_flat.shape
  h, d = projection_out_weights.shape
  dtype = x_flat.dtype
  num_tiles = triton.cdiv(m, block_m)
  grid_m = min(num_tiles, max_grid_size)
  is_even_m = m % block_m == 0
  out_type = (
      jax.ShapeDtypeStruct((h, m), dtype),
      jax.ShapeDtypeStruct((m, c), dtype),
      jax.ShapeDtypeStruct((grid_m, h + c, d), jnp.float32),
      jax.ShapeDtypeStruct((grid_m, 2, h), jnp.float32),
  )
  kwargs: dict[str, Any] = dict(
      out_type=out_type,
      name="triton_triangle_mult_epilogue_bwd",
      num_warps=num_warps,
      num_stages=num_stages,
      h_dim=h,
      d_dim=d,
      block_m=block_m,
      grid=(grid_m,),
      **common_kwargs,
  )
  weights = (
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
  )

  if persistent and is_even_m and min(c, h, d) >= _MIN_TMA_CHANNEL_DIM:
    dout_desc = jt.TensorDescriptor.from_array(
        dout_flat, block_shape=[block_m, d]
    )
    t3_desc = jt.TensorDescriptor.from_array(t3_flat, block_shape=[h, block_m])
    x_desc = jt.TensorDescriptor.from_array(x_flat, block_shape=[block_m, c])
    dt3_flat, dleft_act_epi, dw_out_part, dln_out_part = jt.triton_call(
        dout_desc,  # pyrefly: ignore[bad-argument-type]
        t3_desc,  # pyrefly: ignore[bad-argument-type]
        x_desc,  # pyrefly: ignore[bad-argument-type]
        *weights,
        m,
        kernel=_epilogue_bwd_tma_kernel,
        **kwargs,
    )
  else:
    dt3_flat, dleft_act_epi, dw_out_part, dln_out_part = jt.triton_call(
        dout_flat,
        t3_flat,
        x_flat,
        *weights,
        kernel=_epilogue_bwd_kernel,
        m=m,
        is_even_m=is_even_m,
        **kwargs,
    )

  dw_out = jnp.sum(dw_out_part, axis=0).astype(dtype)
  dw_proj_out = dw_out[:h]
  dw_gate_out = dw_out[h:]
  dln_out = jnp.sum(dln_out_part, axis=0).astype(dtype)
  return (
      dt3_flat,
      dleft_act_epi,
      dw_proj_out,
      dw_gate_out,
      dln_out[0],
      dln_out[1],
  )


def _contract_bwd(
    dt3: jax.Array,
    left: jax.Array,
    right: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    precision: base.CanonicalPrecision,
) -> tuple[jax.Array, jax.Array]:
  """Backward pass through the batched triangle contraction."""
  if triangle_type == "outgoing":
    # Forward: t3[c, i, j] = sum_k left[c, i, k] * right[c, j, k]
    # dleft[c, i, k] = sum_j dt3[c, i, j] * right[c, j, k]
    # dright[c, j, k] = sum_i dt3[c, i, j] * left[c, i, k]
    dleft = lax.dot_general(
        dt3, right, (((2,), (1,)), ((0,), (0,))), precision=precision
    )
    dright = lax.dot_general(
        dt3, left, (((1,), (1,)), ((0,), (0,))), precision=precision
    )
    return dleft, dright
  # Forward: t3[c, i, j] = sum_k right[c, k, i] * left[c, k, j]
  # dleft[c, k, j] = sum_i right[c, k, i] * dt3[c, i, j]
  # dright[c, k, i] = sum_j left[c, k, j] * dt3[c, i, j]
  dleft = lax.dot_general(
      right, dt3, (((2,), (1,)), ((0,), (0,))), precision=precision
  )
  dright = lax.dot_general(
      left, dt3, (((2,), (2,)), ((0,), (0,))), precision=precision
  )
  return dleft, dright


def _prologue_bwd_branch(
    dbranch_flat: jax.Array,
    dact_in: jax.Array,
    x_flat: jax.Array,
    mask_flat: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    *,
    branch_idx: int,
    apply_ln_bwd: bool,
    block_m: int,
    num_warps: int,
    num_stages: int,
    persistent: bool,
    max_grid_size: int,
    **common_kwargs,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Runs one branch of the split prologue backward pass."""
  m, c = x_flat.shape
  h = projection_in_weights.shape[2]
  dtype = x_flat.dtype
  num_tiles = triton.cdiv(m, block_m)
  grid_m = min(num_tiles, max_grid_size)
  is_even_m = m % block_m == 0
  out_type = (
      jax.ShapeDtypeStruct((m, c), dtype),
      jax.ShapeDtypeStruct((grid_m, 2, c, h), jnp.float32),
      jax.ShapeDtypeStruct((grid_m if apply_ln_bwd else 1, 2, c), jnp.float32),
  )
  kwargs: dict[str, Any] = dict(
      out_type=out_type,
      name=f"triton_triangle_mult_prologue_bwd_b{branch_idx}",
      num_warps=num_warps,
      num_stages=num_stages,
      m=m,
      h_dim=h,
      branch_idx=branch_idx,
      apply_ln_bwd=apply_ln_bwd,
      block_m=block_m,
      grid=(grid_m,),
      **common_kwargs,
  )
  weights = (
      projection_in_weights,
      gate_in_weights,
      layernorm_in_scale,
      layernorm_in_offset,
  )

  if persistent and is_even_m and min(c, h) >= _MIN_TMA_CHANNEL_DIM:
    dbranch_desc = jt.TensorDescriptor.from_array(
        dbranch_flat, block_shape=[h, block_m]
    )
    dact_in_desc = jt.TensorDescriptor.from_array(
        dact_in, block_shape=[block_m, c]
    )
    x_desc = jt.TensorDescriptor.from_array(x_flat, block_shape=[block_m, c])
    return jt.triton_call(
        dbranch_desc,  # pyrefly: ignore[bad-argument-type]
        dact_in_desc,  # pyrefly: ignore[bad-argument-type]
        x_desc,  # pyrefly: ignore[bad-argument-type]
        mask_flat,
        *weights,
        kernel=_prologue_bwd_tma_kernel,
        **kwargs,
    )
  return jt.triton_call(
      dbranch_flat,
      dact_in,
      x_flat,
      mask_flat,
      *weights,
      kernel=_prologue_bwd_kernel,
      is_even_m=is_even_m,
      **kwargs,
  )


def _prologue_bwd(
    dleft_flat: jax.Array,
    dright_flat: jax.Array,
    dleft_act_epi: jax.Array,
    x_flat: jax.Array,
    mask_flat: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    *,
    block_m: int,
    num_warps: int,
    num_stages: int,
    persistent: bool,
    max_grid_size: int,
    **common_kwargs,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Backward pass through the prologue."""
  dtype = x_flat.dtype
  dact_mid, dw_r_part, _ = _prologue_bwd_branch(
      dright_flat,
      dleft_act_epi,
      x_flat,
      mask_flat,
      projection_in_weights,
      gate_in_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      branch_idx=1,
      apply_ln_bwd=False,
      block_m=block_m,
      num_warps=num_warps,
      num_stages=num_stages,
      persistent=persistent,
      max_grid_size=max_grid_size,
      **common_kwargs,
  )
  dx_flat, dw_l_part, dln_in_part = _prologue_bwd_branch(
      dleft_flat,
      dact_mid,
      x_flat,
      mask_flat,
      projection_in_weights,
      gate_in_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      branch_idx=0,
      apply_ln_bwd=True,
      block_m=block_m,
      num_warps=num_warps,
      num_stages=num_stages,
      persistent=persistent,
      max_grid_size=max_grid_size,
      **common_kwargs,
  )
  dw_in = jnp.sum(
      jnp.stack([dw_l_part, dw_r_part], axis=3), axis=0
  ).astype(dtype)
  dln_in = jnp.sum(dln_in_part, axis=0).astype(dtype)
  return dx_flat, dw_in[0], dw_in[1], dln_in[0], dln_in[1]


@dataclasses.dataclass(frozen=True, slots=True)
class TritonTriangleMultiplicationVjp(
    base.TriangleMultiplicationVjp[Config, Key]
):
  """Triton triangle multiplicative update VJP."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      residuals: base.Residuals,
      out: Float[Array, "N N D"],
      dout: Float[Array, "N N D"],
      x: Float[Array, "N N C"],
      mask: Bool[Array, "N N"],
      projection_in_weights: Float[Array, "C 2 H"],
      gate_in_weights: Float[Array, "C 2 H"],
      projection_out_weights: Float[Array, "H D"],
      gate_out_weights: Float[Array, "C D"],
      layernorm_in_scale: Float[Array, "C"],
      layernorm_in_offset: Float[Array, "C"],
      layernorm_out_scale: Float[Array, "H"],
      layernorm_out_offset: Float[Array, "H"],
      triangle_type: Literal["incoming", "outgoing"],
      *,
      precision: base.CanonicalPrecision,
      epsilon: float,
      return_residuals: bool,
      implementation: (
          base.Implementation | Sequence[base.Implementation] | None
      ) = None,
      config: Config,
  ) -> tuple[base.Grads, None]:
    """Computes gradients with fused Triton epilogue/prologue backward kernels."""
    del out, implementation  # Unused.

    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")

    if not gpu_utils.has_triton_support():
      raise NotImplementedError("Triton is not supported on this platform.")

    if not precision_lib.is_default(
        x.dtype, projection_in_weights.dtype, precision
    ):
      raise NotImplementedError(f"{precision=} is not supported.")

    if jnp.dtype(x.dtype) not in _SUPPORTED_DTYPES:
      raise NotImplementedError(
          "Triton triangle_multiplication only supports"
          f" {_SUPPORTED_DTYPES}, got {x.dtype}."
      )

    for w in (
        *residuals,
        dout,
        projection_in_weights,
        gate_in_weights,
        projection_out_weights,
        gate_out_weights,
        layernorm_in_scale,
        layernorm_in_offset,
        layernorm_out_scale,
        layernorm_out_offset,
    ):
      if w.dtype != x.dtype:
        raise NotImplementedError(
            "All residual, weight, norm, and dout arrays must match"
            f" x.dtype={x.dtype}, got {w.dtype}."
        )

    n, _, c = x.shape
    h, d = projection_out_weights.shape
    if not {c, h, d} <= _SUPPORTED_CHANNEL_DIMS:
      raise NotImplementedError(
          "Triton TriangleMultiplicationVjp requires C, H, D in"
          f" {sorted(_SUPPORTED_CHANNEL_DIMS)}, got C={c}, H={h}, D={d}."
      )
    if n == 0:
      raise NotImplementedError("Empty inputs are not supported.")
    m = n * n
    epi_bm = config.epilogue_block_m or config.block_m
    max_m_pad = max(
        triton.cdiv(m, config.block_m) * config.block_m,
        triton.cdiv(m, epi_bm) * epi_bm,
    )
    if max_m_pad * max(c, h, d) - 1 > _MAX_INT32:
      raise NotImplementedError(
          f"N={n} with max(C, H, D)={max(c, h, d)} would overflow the int32"
          " offsets used by the Triton kernels."
      )

    dot_dtype, input_precision = triton_utils.get_dot_dtype_and_precision(
        precision, dout.dtype, dout.dtype
    )
    common_kwargs: dict[str, Any] = dict(
        eps=epsilon,
        c_dim=c,
        dot_dtype=dot_dtype,
        input_precision=input_precision,
    )

    x_flat = lax.collapse(x, 0, 2)
    mask_flat = lax.collapse(mask, 0, 2)
    dout_flat = lax.collapse(dout, 0, 2)
    left, right, t3 = residuals

    pro_block_m = config.block_m
    pro_num_warps = config.num_warps
    pro_num_stages = config.num_stages
    epi_block_m = config.epilogue_block_m or config.block_m
    epi_num_warps = config.epilogue_num_warps or config.num_warps
    epi_num_stages = config.epilogue_num_stages or config.num_stages
    if max(c, h, d) > 64:
      pro_block_m = min(pro_block_m, 32)
      pro_num_stages = 1
      epi_block_m = min(epi_block_m, 32)
      epi_num_stages = 1
    max_grid_size = backend.get_default_device().core_count * config.ctas_per_sm
    has_tma = gpu_utils.is_sm90() or gpu_utils.is_sm100()
    epi_persistent = (
        has_tma and config.epilogue_persistent is not False  # pylint: disable=g-bool-id-comparison
    )
    pro_persistent = (
        has_tma and config.prologue_persistent is not False  # pylint: disable=g-bool-id-comparison
    )

    (
        dt3_flat,
        dleft_act_epi,
        dw_proj_out,
        dw_gate_out,
        dln_out_scale,
        dln_out_offset,
    ) = _epilogue_bwd(
        dout_flat,
        t3.reshape(h, m),
        x_flat,
        projection_out_weights,
        gate_out_weights,
        layernorm_in_scale,
        layernorm_in_offset,
        layernorm_out_scale,
        layernorm_out_offset,
        block_m=epi_block_m,
        num_warps=epi_num_warps,
        num_stages=epi_num_stages,
        persistent=epi_persistent,
        max_grid_size=max_grid_size,
        **common_kwargs,
    )

    dleft, dright = _contract_bwd(
        dt3_flat.reshape(h, n, n),
        left,
        right,
        triangle_type,
        precision,
    )

    dx_flat, dw_proj_in, dw_gate_in, dln_in_scale, dln_in_offset = (
        _prologue_bwd(
            dleft.reshape(h, m),
            dright.reshape(h, m),
            dleft_act_epi,
            x_flat,
            mask_flat,
            projection_in_weights,
            gate_in_weights,
            layernorm_in_scale,
            layernorm_in_offset,
            block_m=pro_block_m,
            num_warps=pro_num_warps,
            num_stages=pro_num_stages,
            persistent=pro_persistent,
            max_grid_size=max_grid_size,
            **common_kwargs,
        )
    )

    return (
        base.Grads(
            x=dx_flat.reshape(n, n, c),
            projection_in_weights=dw_proj_in,
            gate_in_weights=dw_gate_in,
            projection_out_weights=dw_proj_out,
            gate_out_weights=dw_gate_out,
            layernorm_in_scale=dln_in_scale,
            layernorm_in_offset=dln_in_offset,
            layernorm_out_scale=dln_out_scale,
            layernorm_out_offset=dln_out_offset,
        ),
        None,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba  # Unused.
    if gpu_utils.is_sm100():
      return Config(
          block_m=64,
          num_warps=8,
          num_stages=3,
          epilogue_block_m=64,
          epilogue_num_warps=4,
          epilogue_num_stages=2,
          prologue_persistent=True,
          epilogue_persistent=True,
          ctas_per_sm=2,
      )
    if gpu_utils.is_sm90():
      return Config(
          block_m=64,
          num_warps=4,
          num_stages=3,
          prologue_persistent=True,
          epilogue_persistent=True,
          ctas_per_sm=2,
      )
    return Config(
        block_m=64,
        num_warps=4,
        num_stages=2,
        prologue_persistent=False,
        epilogue_persistent=False,
        ctas_per_sm=2,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    if not (gpu_utils.is_sm90() or gpu_utils.is_sm100()):
      return {  # pylint: disable=g-complex-comprehension
          Config(
              block_m=block_m,
              num_warps=num_warps,
              num_stages=2,
              prologue_persistent=False,
              epilogue_persistent=False,
          )
          for block_m in (32, 64)
          for num_warps in (4, 8)
      }
    configs = {
        self._get_heuristics_config(ba),
        Config(
            block_m=64,
            num_warps=4,
            num_stages=2,
            prologue_persistent=False,
            epilogue_persistent=False,
        ),
    }
    for block_m in (32, 64):
      for num_warps in (4, 8):
        for num_stages in (2, 3):
          configs.add(
              Config(
                  block_m=block_m,
                  num_warps=num_warps,
                  num_stages=num_stages,
                  prologue_persistent=True,
                  epilogue_persistent=True,
              )
          )
    return configs
