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
"""Triton implementation of the triangle multiplication op."""

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

Residuals = base.Residuals
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
  """Configuration for Triton Triangle Multiplication kernels.

  Attributes:
    block_m: Tile size along the flattened token-pair dimension M = N * N. Must
      be a power of two in [32, 256].
    num_warps: Number of warps per CTA. Must be a power of two, at most 16.
    num_stages: Number of software pipeline stages.
    epilogue_block_m: Optional tile size override for the epilogue kernel. Same
      constraints as `block_m`.
    epilogue_num_warps: Optional num_warps override for the epilogue kernel.
    epilogue_num_stages: Optional num_stages override for the epilogue kernel.
    prologue_persistent: Whether to use the persistent TMA prologue kernel. If
      `None`, it is used when supported. Ignored on GPUs other than SM90 (it
      exceeds the SM100 tensor memory limit for H=128), or if M is not a
      multiple of `block_m`, or if min(C, H) < 64.
    epilogue_persistent: Whether to use the persistent TMA epilogue kernel. If
      `None`, it is used on SM90 and SM100. Ignored on GPUs without TMA
      (pre-SM90), or if M is not a multiple of the epilogue block size, or if
      min(C, H, D) < 64.
    ctas_per_sm: Number of persistent CTAs per SM when persistent is enabled.
  """

  block_m: _BlockM = 128
  num_warps: _NumWarps = 4
  num_stages: pydantic.PositiveInt = 3
  epilogue_block_m: _BlockM | None = None
  epilogue_num_warps: _NumWarps | None = None
  epilogue_num_stages: pydantic.PositiveInt | None = None
  prologue_persistent: bool | None = None
  epilogue_persistent: bool | None = None
  ctas_per_sm: pydantic.PositiveInt = 2


# ------------------------------------------------------------------------------
# Shared per-tile device helpers.
# ------------------------------------------------------------------------------


@triton.jit
def _layer_norm_2d(
    x,
    scale,
    offset,
    eps: tl.constexpr,
    dim_size: tl.constexpr,
):
  """Computes LayerNorm along axis 1 of a 2D tile in FP32."""
  x_f32 = x.to(tl.float32)
  mean = tl.sum(x_f32, axis=1) * (1.0 / dim_size)
  centered = x_f32 - mean[:, None]
  var = tl.sum(centered * centered, axis=1) * (1.0 / dim_size)
  rstd = tl.rsqrt(var + eps)
  normed = centered * rstd[:, None]
  return normed * scale.to(tl.float32)[None, :] + offset.to(tl.float32)[None, :]


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
def _load_glu_in_weights(w_proj_ptr, w_gate_ptr, c_dim, h_dim: tl.constexpr):
  """Loads the (C, H) left/right slices of the (C, 2, H) GLU weights."""
  w_proj_l = _load_param_matrix(w_proj_ptr, c_dim, h_dim, 2 * h_dim)
  w_gate_l = _load_param_matrix(w_gate_ptr, c_dim, h_dim, 2 * h_dim)
  w_proj_r = _load_param_matrix(w_proj_ptr + h_dim, c_dim, h_dim, 2 * h_dim)
  w_gate_r = _load_param_matrix(w_gate_ptr + h_dim, c_dim, h_dim, 2 * h_dim)
  return w_proj_l, w_gate_l, w_proj_r, w_gate_r


@triton.jit
def _glu(act, w_proj, w_gate, input_precision: tl.constexpr):
  """Computes `(act @ w_proj) * sigmoid(act @ w_gate)` in FP32."""
  p = tl.dot(act, w_proj, input_precision=input_precision)
  g = tl.dot(act, w_gate, input_precision=input_precision)
  return p * tl.sigmoid(g)


@triton.jit
def _store_transposed(
    out_ptr,
    value,
    offs_m,
    m,
    h_dim: tl.constexpr,
    is_even_m: tl.constexpr,
    eviction_policy: tl.constexpr,
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
def _prologue_tile(
    x,
    mask_val,
    offs_m,
    ln_scale,
    ln_offset,
    w_proj_l,
    w_gate_l,
    w_proj_r,
    w_gate_r,
    left_out_ptr,
    right_out_ptr,
    m,
    eps: tl.constexpr,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    is_even_m,
    eviction_policy,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """LayerNorm_in + Mask + GLU for one tile, stored transposed to (H, M).

  `is_even_m` (bool) and `eviction_policy` (str) are compile-time constants
  (Triton passes Python scalars to nested `triton.jit` functions as constexprs).
  """
  act = _layer_norm_2d(x, ln_scale, ln_offset, eps, c_dim)
  act = tl.where(mask_val.to(tl.int1)[:, None], act, 0.0).to(dot_dtype)
  # The left projection is stored before the right one is computed, so only one
  # (BLOCK_M, H) FP32 accumulator pair is live at a time.
  left = _glu(act, w_proj_l, w_gate_l, input_precision)
  _store_transposed(
      left_out_ptr, left, offs_m, m, h_dim, is_even_m, eviction_policy
  )
  right = _glu(act, w_proj_r, w_gate_r, input_precision)
  _store_transposed(
      right_out_ptr, right, offs_m, m, h_dim, is_even_m, eviction_policy
  )


@triton.jit
def _epilogue_tile(
    x,
    t3,
    ln_in_scale,
    ln_in_offset,
    ln_out_scale,
    ln_out_offset,
    w_gate_out,
    w_proj_out,
    eps: tl.constexpr,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """LayerNorm_in(x) gate * Proj(LayerNorm_out(t3^T)) for one tile, in FP32."""
  left_act = _layer_norm_2d(x, ln_in_scale, ln_in_offset, eps, c_dim)
  g = tl.dot(
      left_act.to(dot_dtype), w_gate_out, input_precision=input_precision
  )
  gate = tl.sigmoid(g)
  act = _layer_norm_2d(tl.trans(t3), ln_out_scale, ln_out_offset, eps, h_dim)
  p = tl.dot(act.to(dot_dtype), w_proj_out, input_precision=input_precision)
  return p * gate


# ------------------------------------------------------------------------------
# Kernels. Each kernel only handles addressing; the math is in the helpers.
# ------------------------------------------------------------------------------


@triton.jit
def _prologue_kernel(
    x_ptr,
    mask_ptr,
    w_proj_ptr,
    w_gate_ptr,
    ln_scale_ptr,
    ln_offset_ptr,
    left_out_ptr,
    right_out_ptr,
    m: tl.constexpr,
    eps: tl.constexpr,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    block_m: tl.constexpr,
    is_even_m: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Fused LayerNorm_in + GLU + Mask + Transpose to (H, M)."""
  pid = tl.program_id(axis=0)
  offs_m = pid * block_m + tl.arange(0, block_m)
  if is_even_m:
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
  mask_m = offs_m < m

  x_ptrs = x_ptr + offs_m[:, None] * c_dim + tl.arange(0, c_dim)[None, :]
  if is_even_m:
    x = tl.load(x_ptrs, eviction_policy="evict_first")
    mask_val = tl.load(mask_ptr + offs_m, eviction_policy="evict_first")
  else:
    x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
    mask_val = tl.load(mask_ptr + offs_m, mask=mask_m, other=0)

  ln_scale = _load_param_vector(ln_scale_ptr, c_dim)
  ln_offset = _load_param_vector(ln_offset_ptr, c_dim)
  w_proj_l, w_gate_l, w_proj_r, w_gate_r = _load_glu_in_weights(
      w_proj_ptr, w_gate_ptr, c_dim, h_dim
  )
  _prologue_tile(
      x,
      mask_val,
      offs_m,
      ln_scale,
      ln_offset,
      w_proj_l,
      w_gate_l,
      w_proj_r,
      w_gate_r,
      left_out_ptr,
      right_out_ptr,
      m,
      eps,
      c_dim,
      h_dim,
      is_even_m,
      "",  # eviction_policy
      dot_dtype,
      input_precision,
  )


@triton.jit
def _prologue_tma_kernel(
    x_desc,
    mask_ptr,
    w_proj_ptr,
    w_gate_ptr,
    ln_scale_ptr,
    ln_offset_ptr,
    left_out_ptr,
    right_out_ptr,
    m: tl.constexpr,
    eps: tl.constexpr,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    block_m: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Persistent TMA-pipelined LayerNorm_in + GLU + Mask + Transpose to (H, M).

  Requires M to be a multiple of `block_m`.
  """
  ln_scale = _load_param_vector(ln_scale_ptr, c_dim)
  ln_offset = _load_param_vector(ln_offset_ptr, c_dim)
  w_proj_l, w_gate_l, w_proj_r, w_gate_r = _load_glu_in_weights(
      w_proj_ptr, w_gate_ptr, c_dim, h_dim
  )

  num_tiles: tl.constexpr = m // block_m
  for tile_id in tl.range(tl.program_id(0), num_tiles, tl.num_programs(0)):
    offs_m = tile_id * block_m + tl.arange(0, block_m)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    x = x_desc.load([tile_id * block_m, 0])
    mask_val = tl.load(mask_ptr + offs_m, eviction_policy="evict_first")
    _prologue_tile(
        x,
        mask_val,
        offs_m,
        ln_scale,
        ln_offset,
        w_proj_l,
        w_gate_l,
        w_proj_r,
        w_gate_r,
        left_out_ptr,
        right_out_ptr,
        m,
        eps,
        c_dim,
        h_dim,
        True,  # is_even_m
        "evict_first",  # eviction_policy
        dot_dtype,
        input_precision,
    )


@triton.jit
def _epilogue_kernel(
    t3_ptr,
    x_ptr,
    w_proj_out_ptr,
    w_gate_out_ptr,
    ln_in_scale_ptr,
    ln_in_offset_ptr,
    ln_out_scale_ptr,
    ln_out_offset_ptr,
    out_ptr,
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
  """Fused LayerNorm_in(x) + Gate + Transpose(t3) + LayerNorm_out + Proj."""
  pid = tl.program_id(axis=0)
  offs_m = pid * block_m + tl.arange(0, block_m)
  if is_even_m:
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
  mask_m = offs_m < m
  offs_d = tl.arange(0, d_dim)

  # Issue streaming loads of both x (BLOCK_M, C) and t3 (H, BLOCK_M) upfront
  # so t3's global memory latency is hidden behind LayerNorm_in(x) and the
  # gate projection dot.
  x_ptrs = x_ptr + offs_m[:, None] * c_dim + tl.arange(0, c_dim)[None, :]
  t3_ptrs = t3_ptr + tl.arange(0, h_dim)[:, None] * m + offs_m[None, :]
  if is_even_m:
    x = tl.load(x_ptrs, eviction_policy="evict_first")
    t3 = tl.load(t3_ptrs, eviction_policy="evict_first")
  else:
    x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
    t3 = tl.load(t3_ptrs, mask=mask_m[None, :], other=0.0)

  out = _epilogue_tile(
      x,
      t3,
      _load_param_vector(ln_in_scale_ptr, c_dim),
      _load_param_vector(ln_in_offset_ptr, c_dim),
      _load_param_vector(ln_out_scale_ptr, h_dim),
      _load_param_vector(ln_out_offset_ptr, h_dim),
      _load_param_matrix(w_gate_out_ptr, c_dim, d_dim, d_dim),
      _load_param_matrix(w_proj_out_ptr, h_dim, d_dim, d_dim),
      eps,
      c_dim,
      h_dim,
      dot_dtype,
      input_precision,
  ).to(out_ptr.dtype.element_ty)

  out_ptrs = out_ptr + offs_m[:, None] * d_dim + offs_d[None, :]
  if is_even_m:
    tl.store(out_ptrs, out)
  else:
    tl.store(out_ptrs, out, mask=mask_m[:, None])


@triton.jit
def _epilogue_tma_kernel(
    t3_desc,
    x_desc,
    w_proj_out_ptr,
    w_gate_out_ptr,
    ln_in_scale_ptr,
    ln_in_offset_ptr,
    ln_out_scale_ptr,
    ln_out_offset_ptr,
    m,
    out_ptr,
    eps: tl.constexpr,
    c_dim: tl.constexpr,
    h_dim: tl.constexpr,
    d_dim: tl.constexpr,
    block_m: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
):
  """Persistent TMA-pipelined LayerNorm_in(x) + Gate + Transpose(t3) + Proj.

  Requires M to be a multiple of `block_m`.
  """
  offs_d = tl.arange(0, d_dim)
  ln_in_scale = _load_param_vector(ln_in_scale_ptr, c_dim)
  ln_in_offset = _load_param_vector(ln_in_offset_ptr, c_dim)
  ln_out_scale = _load_param_vector(ln_out_scale_ptr, h_dim)
  ln_out_offset = _load_param_vector(ln_out_offset_ptr, h_dim)
  w_gate_out = _load_param_matrix(w_gate_out_ptr, c_dim, d_dim, d_dim)
  w_proj_out = _load_param_matrix(w_proj_out_ptr, h_dim, d_dim, d_dim)

  num_tiles = m // block_m
  for tile_id in tl.range(tl.program_id(0), num_tiles, tl.num_programs(0)):
    offs_m = tile_id * block_m + tl.arange(0, block_m)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    x = x_desc.load([tile_id * block_m, 0])
    t3 = t3_desc.load([0, tile_id * block_m])
    out = _epilogue_tile(
        x,
        t3,
        ln_in_scale,
        ln_in_offset,
        ln_out_scale,
        ln_out_offset,
        w_gate_out,
        w_proj_out,
        eps,
        c_dim,
        h_dim,
        dot_dtype,
        input_precision,
    ).to(out_ptr.dtype.element_ty)
    out_ptrs = out_ptr + offs_m[:, None] * d_dim + offs_d[None, :]
    tl.store(out_ptrs, out, eviction_policy="evict_first")


# ------------------------------------------------------------------------------
# Host-side launchers.
# ------------------------------------------------------------------------------


def _prologue(
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
) -> tuple[jax.Array, jax.Array]:
  """Returns the masked left/right GLU projections, each of shape (H, M)."""
  m, c = x_flat.shape
  h = projection_in_weights.shape[2]
  out_shape = jax.ShapeDtypeStruct((h, m), x_flat.dtype)
  num_tiles = triton.cdiv(m, block_m)
  is_even_m = m % block_m == 0
  kwargs: dict[str, Any] = dict(
      out_type=(out_shape, out_shape),
      name="triton_triangle_mult_prologue",
      num_warps=num_warps,
      num_stages=num_stages,
      m=m,
      h_dim=h,
      block_m=block_m,
      **common_kwargs,
  )
  weights = (
      projection_in_weights,
      gate_in_weights,
      layernorm_in_scale,
      layernorm_in_offset,
  )

  if persistent and is_even_m and min(c, h) >= _MIN_TMA_CHANNEL_DIM:
    x_desc = jt.TensorDescriptor.from_array(x_flat, block_shape=[block_m, c])
    return jt.triton_call(
        x_desc,  # pyrefly: ignore[bad-argument-type]
        mask_flat,
        *weights,
        kernel=_prologue_tma_kernel,
        grid=(min(num_tiles, max_grid_size),),
        **kwargs,
    )
  return jt.triton_call(
      x_flat,
      mask_flat,
      *weights,
      kernel=_prologue_kernel,
      grid=(num_tiles,),
      is_even_m=is_even_m,
      **kwargs,
  )


def _contract(
    left: jax.Array,
    right: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    precision: base.CanonicalPrecision,
) -> jax.Array:
  """Batched (over H) triangle contraction of two (H, N, N) planes."""
  if triangle_type == "outgoing":
    # t3[c, i, j] = sum_k left[c, i, k] * right[c, j, k]
    dims = (((2,), (2,)), ((0,), (0,)))
    return lax.dot_general(left, right, dims, precision=precision)
  # t3[c, i, j] = sum_k left[c, k, j] * right[c, k, i]
  dims = (((1,), (1,)), ((0,), (0,)))
  return lax.dot_general(right, left, dims, precision=precision)


def _epilogue(
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
) -> jax.Array:
  """Returns the gated output projection of shape (M, D)."""
  m, c = x_flat.shape
  h, d = projection_out_weights.shape
  num_tiles = triton.cdiv(m, block_m)
  is_even_m = m % block_m == 0
  kwargs: dict[str, Any] = dict(
      out_type=jax.ShapeDtypeStruct((m, d), x_flat.dtype),
      name="triton_triangle_mult_epilogue",
      num_warps=num_warps,
      num_stages=num_stages,
      h_dim=h,
      d_dim=d,
      block_m=block_m,
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
    t3_desc = jt.TensorDescriptor.from_array(t3_flat, block_shape=[h, block_m])
    x_desc = jt.TensorDescriptor.from_array(x_flat, block_shape=[block_m, c])
    return jt.triton_call(
        t3_desc,  # pyrefly: ignore[bad-argument-type]
        x_desc,  # pyrefly: ignore[bad-argument-type]
        *weights,
        m,
        kernel=_epilogue_tma_kernel,
        grid=(min(num_tiles, max_grid_size),),
        **kwargs,
    )
  return jt.triton_call(
      t3_flat,
      x_flat,
      *weights,
      kernel=_epilogue_kernel,
      grid=(num_tiles,),
      m=m,
      is_even_m=is_even_m,
      **kwargs,
  )


def _xla_recompute_vjp(
    residuals: Residuals,
    out: jax.Array,
    dout: jax.Array,
    *args,
    precision: base.CanonicalPrecision,
    epsilon: float,
    implementation: base.Implementation | Sequence[base.Implementation] | None,
) -> tuple[jax.Array | None, ...]:
  """Computes gradients by differentiating an XLA recomputation of the forward.

  `jt.triton_call` has no autodiff rule, so the backward pass rematerializes
  the forward pass with the XLA reference implementation and differentiates it.

  Args:
    residuals: Unused (the Triton forward returns no residuals).
    out: Unused.
    dout: The output cotangent, shape `[N, N, D]`.
    *args: The positional op arguments, `(x, mask, ..., triangle_type)`.
    precision: The canonical dot precision.
    epsilon: The LayerNorm epsilon.
    implementation: Unused; the recomputation always uses XLA.

  Returns:
    A gradient for every positional argument. The op's VJP machinery requires
    an entry for each positional argument, so `mask` gets zeros and
    `triangle_type` gets `None`.
  """
  del residuals, out, implementation  # Unused.
  x, mask, *weights, triangle_type = args
  xla_op = base.TriangleMultiplication()

  def fwd(x, *weights):
    return xla_op(
        x,
        mask,
        *weights,
        triangle_type,
        precision=precision,
        epsilon=epsilon,
        implementation="xla",
    )

  _, fwd_vjp = jax.vjp(fwd, x, *weights)
  dx, *dweights = fwd_vjp(dout)
  return (dx, jnp.zeros_like(mask), *dweights, None)


@dataclasses.dataclass(frozen=True, slots=True)
class TritonTriangleMultiplication(base.TriangleMultiplication[Config, Key]):
  """Triton triangle multiplicative update.

  The forward pass is computed by two fused Triton kernels around an XLA
  batched matmul. By default, gradients are computed by recomputing the forward
  pass in XLA (see `_xla_recompute_vjp`).
  """

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", _xla_recompute_vjp)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
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
  ) -> tuple[Float[Array, "N N D"], Residuals]:
    # There are no residuals; the VJP recomputes the forward pass.
    del return_residuals, implementation  # Unused.

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
            f"All weight and norm arrays must match x.dtype={x.dtype}, got"
            f" {w.dtype}."
        )

    # Shapes are consistent (enforced by the `jaxtyping` annotations); only the
    # kernel-specific constraints need to be checked here.
    n, _, c = x.shape
    h, d = projection_out_weights.shape
    if not {c, h, d} <= _SUPPORTED_CHANNEL_DIMS:
      raise NotImplementedError(
          "Triton TriangleMultiplication requires C, H, D in"
          f" {sorted(_SUPPORTED_CHANNEL_DIMS)}, got C={c}, H={h}, D={d}."
      )
    if n == 0:
      raise NotImplementedError("Empty inputs are not supported.")
    m = n * n
    if m * max(c, h, d) - 1 > _MAX_INT32:
      raise NotImplementedError(
          f"N={n} with max(C, H, D)={max(c, h, d)} would overflow the int32"
          " offsets used by the Triton kernels."
      )

    dot_dtype, input_precision = triton_utils.get_dot_dtype_and_precision(
        precision, x.dtype, x.dtype
    )
    common_kwargs: dict[str, Any] = dict(
        eps=epsilon,
        c_dim=c,
        dot_dtype=dot_dtype,
        input_precision=input_precision,
    )

    # On SM100, the persistent prologue needs more than the 512 available tensor
    # memory columns for H=128 or BLOCK_M=256 and fails to compile, so it is
    # only used on SM90.
    pro_persistent = (
        gpu_utils.is_sm90()
        and config.prologue_persistent is not False  # pylint: disable=g-bool-id-comparison
    )
    epi_persistent = (
        gpu_utils.is_sm90() or gpu_utils.is_sm100()
    ) and config.epilogue_persistent is not False  # pylint: disable=g-bool-id-comparison
    max_grid_size = backend.get_default_device().core_count * config.ctas_per_sm

    x_flat = lax.collapse(x, 0, 2)
    left_flat, right_flat = _prologue(
        x_flat,
        lax.collapse(mask, 0, 2),
        projection_in_weights,
        gate_in_weights,
        layernorm_in_scale,
        layernorm_in_offset,
        block_m=config.block_m,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        persistent=pro_persistent,
        max_grid_size=max_grid_size,
        **common_kwargs,
    )

    t3 = _contract(
        left_flat.reshape(h, n, n),
        right_flat.reshape(h, n, n),
        triangle_type,
        precision,
    )

    out_flat = _epilogue(
        t3.reshape(h, m),
        x_flat,
        projection_out_weights,
        gate_out_weights,
        layernorm_in_scale,
        layernorm_in_offset,
        layernorm_out_scale,
        layernorm_out_offset,
        block_m=config.epilogue_block_m or config.block_m,
        num_warps=config.epilogue_num_warps or config.num_warps,
        num_stages=config.epilogue_num_stages or config.num_stages,
        persistent=epi_persistent,
        max_grid_size=max_grid_size,
        **common_kwargs,
    )
    return out_flat.reshape(n, n, d), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    if gpu_utils.is_sm100():
      return Config(
          block_m=128,
          num_warps=8,
          num_stages=2,
          epilogue_block_m=64,
          epilogue_num_warps=4,
          epilogue_num_stages=4,
          prologue_persistent=False,
          epilogue_persistent=True,
          ctas_per_sm=2,
      )
    if gpu_utils.is_sm90():
      return Config(
          block_m=128,
          num_warps=4,
          num_stages=3,
          prologue_persistent=True,
          epilogue_persistent=True,
          ctas_per_sm=2,
      )
    x = ba.args[0]
    n = x.shape[0]
    block_m = 128 if n >= 1024 else 64
    return Config(
        block_m=block_m,
        num_warps=4,
        num_stages=2,
        prologue_persistent=False,
        epilogue_persistent=False,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    if not (gpu_utils.is_sm90() or gpu_utils.is_sm100()):
      # Without TMA, only the pointer kernels are used. They have no loops, so
      # `num_stages` and the persistence options have no effect.
      return {  # pylint: disable=g-complex-comprehension
          Config(
              block_m=block_m,
              num_warps=num_warps,
              num_stages=2,
              prologue_persistent=False,
              epilogue_persistent=False,
          )
          for block_m in (64, 128)
          for num_warps in (4, 8)
      }
    configs = {
        self._get_heuristics_config(ba),
        Config(
            block_m=64,
            num_warps=4,
            num_stages=3,
            prologue_persistent=True,
            epilogue_persistent=True,
        ),
        Config(
            block_m=128,
            num_warps=4,
            num_stages=2,
            prologue_persistent=False,
            epilogue_persistent=False,
        ),
    }
    for block_m in (64, 128):
      for num_warps in (4, 8):
        for num_stages in (2, 3):
          configs.add(
              Config(
                  block_m=block_m,
                  num_warps=num_warps,
                  num_stages=num_stages,
              )
          )
    return configs
